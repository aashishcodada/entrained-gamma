import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

__all__ = ["get_time_BSTD", "plot_sample", "plot_spectrogram", "separate_disconts"]

def get_time_BSTD(sample:pd.Series) -> np.ndarray:
    """
    Return the reconstructed sample timestamps in milliseconds for one BrainSenseTimeDomain sample.

    Reconstructs the time axis by walking backward through the packet metadata
    (packet sizes and tick timestamps), inferring continuity between packets and
    extrapolating each packet's sample times from the packet start time.

    :param sample: contains fields SampleRateInHz, GlobalPacketSizes, and TicksInMses
    :return: time vector in milliseconds
    """
    segments = []
    next_ticks_ms, next_packet_size = np.inf, 0
    sample_period_ms = 1000/sample.SampleRateInHz

    # working backwards through the packets...
    for packet_size, ticks_ms in zip(sample.GlobalPacketSizes[::-1], sample.TicksInMses[::-1]):
        # if the time gap > timespan of the next packet...
        if next_ticks_ms - ticks_ms > (next_packet_size+1)*sample_period_ms:
            t_last = ticks_ms # ... then the segments are discontinuous, so set t_last for this packet to ticks_ms
        else:
            t_last = segments[-1][-1] - sample_period_ms # ...else, they're continuous, so set t_last to just before t_first in the next packet

        packet_time = t_last - np.r_[:packet_size]*sample_period_ms # then, extrapolate backwards from t_last
        segments.append(packet_time)
        next_ticks_ms, next_packet_size = ticks_ms, packet_size

    return np.concatenate(segments)[::-1] # reverse time to be in chronological order

def separate_disconts(sample: pd.Series) -> pd.DataFrame:
    """
    Split a BrainSenseTimeDomain sample into contiguous segments at packet discontinuities.

    This function identifies timestamp jumps larger than a single sample period,
    then partitions the sample into separate segments for TimeDomainData and packet
    metadata. It also shifts the first packet timestamp for each later segment so that
    the reconstructed times remain aligned with the original packet timing.

    Returns a copy of the input series with each discontinuous section stored as a
    separate array/list entry in the appropriate fields.

    :param sample: contains fields TimeDomainData, time_ms, GlobalSequences, GlobalPacketSizes, TicksInMses, and FirstPacketDateTime
    :return: a DataFrame with one row per contiguous segment, with the same columns as the input series
    """

    sample = sample.copy()
    discont_indices = np.nonzero(np.diff(sample.time_ms)>1000/sample.SampleRateInHz)[0]+1
    if len(discont_indices)==0:
        return pd.DataFrame([sample])
    discont_ticks = sample.time_ms[discont_indices-1]

    cum = np.array(sample.GlobalPacketSizes).cumsum()
    pidx = np.searchsorted(cum, discont_indices)
    assert np.all(cum[pidx]==discont_indices), "discont_indices must be a subset of the cumulative packet sizes"
    tidx = np.searchsorted(sample.stim_time_ms, discont_ticks)

    # adjust the first packet time to the first packet of each segment
    delta_ms = np.r_[0, np.array(sample.TicksInMses)[pidx]-sample.TicksInMses[0]]
    sample.FirstPacketDateTime += pd.to_timedelta(delta_ms, unit='ms')

    td_cols = ['TimeDomainData', 'time_ms']
    pd_cols = ['GlobalSequences', 'GlobalPacketSizes', 'TicksInMses']
    ticks_cols = ['stim_time_ms', 'stim_mA', 'LFP']

    for i in td_cols:
        sample[i] = np.split(sample[i], discont_indices)
    for i in pd_cols:
        sample[i] = np.split(sample[i], pidx+1)
    for i in ticks_cols:
        sample[i] = np.split(sample[i], tidx+1)

    split_cols = td_cols + pd_cols + ticks_cols + ['FirstPacketDateTime']
    return pd.DataFrame([sample]).explode(column=split_cols, ignore_index=True)

def plot_spectrogram(channel, fs=250, nperseg=125, noverlap=None, **kwargs):
    f, t, Sxx = spectrogram(channel, fs=fs, nperseg=nperseg, noverlap=noverlap)
    plt.pcolormesh(t, f, 10*np.log10(Sxx), **kwargs)
    plt.colorbar(ax=plt.gca(), fraction=0.046, pad=0.02, label='psd (dB rel. to 1 µV²/Hz)')
    plt.ylabel('frequency (Hz)')
    plt.xlabel('time (s)')
    return f, t, Sxx

def plot_sample(sample: pd.Series, **kwargs):
    """
    Plot a BrainSenseTimeDomain sample with its time-domain signal, spectrogram, and stimulation current.
    """
    time = (sample.time_ms - sample.time_ms[0]) / 1000  # convert to seconds, relative to the first sample
    time_stim = (sample.stim_time_ms - sample.time_ms[0]) / 1000 # same for stim times
    xlim = (0, time[-1])

    plt.figure(layout="constrained")
    plt.subplot(211)
    plt.title(
        f"{sample.patient}_{sample.visit} on {sample.SessionDate.date()}\n"
        f"channel {sample.Channel}, recording time {sample.FirstPacketDateTime.strftime('%H:%M:%S')} (row {sample.name})\n"
        f"pulse width {sample.PulseWidthInMicroSecond} µs, stim. rate {sample.RateInHertz} Hz"
    )
    plt.plot(time, sample.TimeDomainData, lw=1, c="0.8")
    plt.ylabel("voltage (µV)")
    plt.xlabel("time (s)")
    plt.xlim(xlim)

    plt.sca(plt.gca().twinx())
    plt.plot(time_stim, sample.LFP, lw=1.5, c='b', label=f'LFP bandpower at {sample.FrequencyInHertz} Hz')
    plt.ylabel("LFP bandpower (µV²)")
    plt.grid(False)
    plt.legend()

    plt.subplot(212)
    plot_spectrogram(
        sample.TimeDomainData,
        fs=sample.SampleRateInHz,
        nperseg=125,
        noverlap=None,
        vmin=-60,
        vmax=20,
        shading="gouraud",
        cmap='viridis'
    )
    plt.axhline(sample.RateInHertz/2, ls="--", c="w", lw=0.5, label="1:2 eFTG")
    plt.legend(labelcolor="w")
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    plt.xlim(xlim)

    plt.sca(plt.gca().twinx())
    plt.plot(time_stim, sample.stim_mA, lw=1.5, c="r")
    plt.ylabel("stimulation current (mA)")
    plt.grid(False)

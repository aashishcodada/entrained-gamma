import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from entrained_gamma import plot_spectrogram

__all__ = ["bands", "plot_sample"]

def bands(sample, band_ranges, smooth_winlen):
    ap_log = sample.fit_offset - sample.fit_exponent * np.log10(sample.spec_freq_hz[1:])
    ap_lin = 10**ap_log

    def _get_bp(mask):
        ap_bp = ap_lin[mask[1:]].sum() # this is a scalar
        sp_bp = sample.spectrogram[mask].sum(axis=0) # this is a vector
        bp = 10*np.log10(sp_bp/ap_bp) # dB relative to aperiodic baseline
        bp = uniform_filter1d(bp, size=smooth_winlen, mode='nearest')
        return bp

    band_bps = []
    for lo, hi in band_ranges:
        mask = (sample.spec_freq_hz >= lo) & (sample.spec_freq_hz <= hi)
        band_bps.append(_get_bp(mask))

    return band_bps, ap_log

def plot_sample(sample:pd.Series, vmin, vmax):
    """
    Plot a BrainSenseTimeDomain sample with its time-domain signal, spectrogram, and stimulation current.
    """
    # harmonize times
    time_convert = lambda t: (t-sample.time_ms[0])/1000 # convert to seconds
    time = time_convert(sample.time_ms)
    time_stim = time_convert(sample.stim_time_ms)
    time_spec = time_convert(sample.spec_time_ms)
    xlim = (0, time[-1])

    # calculate bandpowers
    eftg_band = (sample.RateInHertz/2-2, sample.RateInHertz/2+2)
    sftg_band = (95, 105)
    beta_band = (13, 30)
    (eftg_bp, sftg_bp, beta_bp), ap_log = bands(
        sample, band_ranges=[eftg_band, sftg_band, beta_band], smooth_winlen=20
    )

    # f_a = | f_s - f * round(f / f_s) |
    aliased_stim_freq = np.abs(
        sample.SampleRateInHz - sample.RateInHertz * np.floor(sample.RateInHertz / sample.SampleRateInHz + 0.5)
    )

    # PLOTTING
    fig = plt.figure(layout="constrained")
    gs = fig.add_gridspec(4, 2)
    ax = [fig.add_subplot(i) for i in [gs[0, 0], gs[1, 0], gs[:2, 1], gs[2:, :]]]

    fig.suptitle(
        f"{sample.patient} on {sample.FirstPacketDateTime.date()} "
        f"{sample.FirstPacketDateTime.strftime('%H:%M:%S')} (rec {sample.rec})\n"
        f"$diagnosis$ {sample.diagnosis}, $implant\\ location$ {sample.implant_location}\n"
        f"$channel$ {sample.Channel}; {sample.PulseWidthInMicroSecond} µs, {sample.RateInHertz} Hz",
        fontsize=16,
    )
    # PLOT 1a: VOLTAGE TRACE
    plt.sca(ax[0])
    plt.plot(time, sample.TimeDomainData, lw=1, c="0.8", zorder=-10)
    plt.gca().set(xlim=xlim, title='time domain', ylabel='voltage (uV)')

    # PLOT 1b: BAND POWERS RELATIVE TO APERIODIC BASELINE
    plt.sca(ax[1])
    plt.plot(time_spec, eftg_bp, label="eFTG")
    plt.plot(time_spec, sftg_bp, label="sFTG")
    plt.plot(time_spec, beta_bp, label="beta")
    plt.gca().set(xlabel='time (s)', ylabel="dB rel. to baseline", xlim=xlim, title='band powers')
    plt.legend()

    # PLOT 2: SPECTRUM w/ APERIODIC FIT
    plt.sca(ax[2])
    plt.axvspan(*eftg_band, color='C0', alpha=0.1, label="1:2 eFTG")
    plt.axvspan(*sftg_band, color='C1', alpha=0.1, label="sFTG")
    plt.axvspan(*beta_band, color='C2', alpha=0.1, label="beta")

    plt.plot(sample.spec_freq_hz, 10*np.log10(sample.spectrum), c='k')
    plt.plot(sample.spec_freq_hz[1:], 10*ap_log, ls="--", c="0.5", lw=0.5, label="aperiodic baseline")

    plt.axvline(aliased_stim_freq, ls="--", c="m", lw=0.5, label="Aliased stim freq")
    plt.gca().set(xlabel='frequency (Hz)', ylabel='psd (dB rel. to 1 µV²/Hz)', title='power spectral density')
    plt.legend()

    # PLOT 3: SPECTROGRAM
    plt.sca(ax[3])
    plot_spectrogram(time_spec, sample.spec_freq_hz, sample.spectrogram, vmin=vmin, vmax=vmax, shading="nearest", cmap='viridis')
    plt.axhline(sample.RateInHertz/2, ls="--", c="w", lw=0.5, label="1:2 eFTG")
    plt.axhline(aliased_stim_freq, ls="--", c="m", lw=0.5, label="Aliased stim freq")
    plt.legend(labelcolor="w", loc='upper right')
    plt.gca().set(xlabel="time (s)", ylabel="frequency (Hz)", xlim=xlim, title='spectrogram')

    plt.sca(plt.gca().twinx())
    plt.plot(time_stim, sample.stim_mA, lw=1.5, c="r")
    plt.ylabel("stimulation current (mA)")
    plt.ylim(0, None)
    plt.grid(False)

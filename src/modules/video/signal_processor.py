"""
signal_processor — physiological signal processing utilities.

Functions
---------
bandpass_filter     — 4th-order zero-phase Butterworth bandpass
detect_peaks        — heartbeat peak finder
estimate_hr         — heart-rate estimation in BPM
compute_snr         — signal-to-noise ratio in dB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from config import (
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    BANDPASS_ORDER,
    PEAK_MIN_DISTANCE_SEC,
    PEAK_PROMINENCE,
    VIDEO_FPS,
)


@dataclass
class SignalStats:
    hr_bpm: float           # estimated heart rate
    snr_db: float           # signal-to-noise ratio
    n_peaks: int            # number of detected heartbeat peaks
    filtered_signal: np.ndarray  # bandpass-filtered BVP


def bandpass_filter(
    signal: np.ndarray,
    fps: float = VIDEO_FPS,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : (T,) float array — raw BVP signal
    fps    : sampling rate in Hz
    low    : lower cut-off frequency in Hz
    high   : upper cut-off frequency in Hz
    order  : filter order

    Returns
    -------
    (T,) filtered signal
    """
    nyq = fps / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def detect_peaks(
    signal: np.ndarray,
    fps: float = VIDEO_FPS,
    min_distance_sec: float = PEAK_MIN_DISTANCE_SEC,
    prominence: float = PEAK_PROMINENCE,
) -> np.ndarray:
    """
    Detect heartbeat peaks in a filtered BVP signal.

    Returns array of peak sample indices.
    """
    min_dist_samples = max(1, int(min_distance_sec * fps))
    peaks, _ = find_peaks(signal, distance=min_dist_samples, prominence=prominence)
    return peaks


def estimate_hr(
    signal: np.ndarray,
    fps: float = VIDEO_FPS,
) -> float:
    """
    Estimate heart rate via FFT dominant frequency.

    Returns BPM (float). Returns 0.0 if signal is too short.
    """
    n = len(signal)
    if n < 2:
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(signal))

    # Restrict to physiological band
    mask = (freqs >= BANDPASS_LOW_HZ) & (freqs <= BANDPASS_HIGH_HZ)
    if not mask.any():
        return 0.0

    dominant_freq = freqs[mask][np.argmax(fft_mag[mask])]
    return float(dominant_freq * 60.0)


def compute_snr(
    signal: np.ndarray,
    fps: float = VIDEO_FPS,
) -> float:
    """
    Compute SNR (dB) as the ratio of in-band power to out-of-band power.

    A real rPPG signal should have SNR ≥ 3 dB.
    """
    n = len(signal)
    if n < 2:
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2

    in_band = (freqs >= BANDPASS_LOW_HZ) & (freqs <= BANDPASS_HIGH_HZ)
    out_band = ~in_band

    signal_power = power[in_band].sum() + 1e-12
    noise_power = power[out_band].sum() + 1e-12

    return float(10.0 * np.log10(signal_power / noise_power))


def process_signal(raw_bvp: np.ndarray, fps: float = VIDEO_FPS) -> SignalStats:
    """
    Full processing pipeline: bandpass → peak detect → HR + SNR estimation.
    """
    # sosfiltfilt needs at least padlen+1 samples; fall back for very short signals
    if len(raw_bvp) < 30:
        return SignalStats(hr_bpm=0.0, snr_db=0.0, n_peaks=0, filtered_signal=raw_bvp.copy())
    filtered = bandpass_filter(raw_bvp, fps)
    hr = estimate_hr(filtered, fps)
    snr = compute_snr(filtered, fps)
    peaks = detect_peaks(filtered, fps)

    return SignalStats(
        hr_bpm=hr,
        snr_db=snr,
        n_peaks=len(peaks),
        filtered_signal=filtered,
    )

"""
Unit tests for the video deepfake detection module.

Tests run without real video files — they use synthetic numpy signals.
"""

import numpy as np
import pytest

from modules.video.signal_processor import (
    bandpass_filter,
    compute_snr,
    estimate_hr,
    process_signal,
)
from modules.video.sync_analyzer import SyncAnalyzer


# ---------------------------------------------------------------------------
# signal_processor tests
# ---------------------------------------------------------------------------

def _make_bvp(fps: float = 30.0, duration: float = 10.0, hr_bpm: float = 70.0) -> np.ndarray:
    """Generate a clean synthetic BVP signal at *hr_bpm*."""
    t = np.linspace(0, duration, int(fps * duration))
    freq = hr_bpm / 60.0
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return signal


def test_bandpass_filter_passes_rppg_band():
    signal = _make_bvp(hr_bpm=70.0)
    filtered = bandpass_filter(signal, fps=30.0)
    # Filtered signal should preserve energy near 70 BPM
    assert filtered.shape == signal.shape
    assert np.abs(filtered).max() > 0.1


def test_bandpass_filter_attenuates_out_of_band():
    fps = 30.0
    n = int(fps * 10)
    t = np.linspace(0, 10, n)
    # 5 Hz — well outside 0.7–3.5 Hz band
    out_of_band = np.sin(2 * np.pi * 5.0 * t).astype(np.float32)
    filtered = bandpass_filter(out_of_band, fps=fps)
    assert np.abs(filtered).max() < 0.1  # heavily attenuated


def test_estimate_hr_clean_signal():
    signal = _make_bvp(hr_bpm=72.0)
    filtered = bandpass_filter(signal, fps=30.0)
    hr = estimate_hr(filtered, fps=30.0)
    assert abs(hr - 72.0) < 5.0, f"Expected ~72 BPM, got {hr:.1f}"


def test_compute_snr_clean_signal():
    signal = _make_bvp(hr_bpm=70.0)
    filtered = bandpass_filter(signal, fps=30.0)
    snr = compute_snr(filtered, fps=30.0)
    assert snr > 0.0, "SNR should be positive for a clean in-band signal"


def test_process_signal_returns_all_fields():
    signal = _make_bvp(hr_bpm=65.0)
    stats = process_signal(signal, fps=30.0)
    assert stats.hr_bpm > 0
    assert stats.n_peaks >= 0
    assert len(stats.filtered_signal) == len(signal)


def test_process_signal_short_input():
    """Very short signal should not crash."""
    signal = np.zeros(5, dtype=np.float32)
    stats = process_signal(signal, fps=30.0)
    assert stats.hr_bpm == 0.0


# ---------------------------------------------------------------------------
# SyncAnalyzer tests
# ---------------------------------------------------------------------------

class TestSyncAnalyzer:
    def setup_method(self):
        self.analyzer = SyncAnalyzer()

    def test_high_sync_real(self):
        """Correlated signals should yield is_real=True."""
        sig = _make_bvp(hr_bpm=70.0)
        noise = np.random.default_rng(0).normal(0, 0.05, len(sig)).astype(np.float32)
        result = self.analyzer.analyze(sig, sig + noise, sig + noise * 0.5)
        # Pearson r should be very high → real
        assert result.forehead_cheek_r > 0.8
        assert result.is_real

    def test_random_signals_fake(self):
        """Uncorrelated random signals should yield is_fake=True."""
        rng = np.random.default_rng(42)
        sig_a = rng.standard_normal(300).astype(np.float32)
        sig_b = rng.standard_normal(300).astype(np.float32)
        sig_c = rng.standard_normal(300).astype(np.float32)
        result = self.analyzer.analyze(sig_a, sig_b, sig_c)
        assert result.is_fake

    def test_constant_signal_handled(self):
        """Constant (degenerate) signals should not raise."""
        sig = np.ones(100, dtype=np.float32)
        result = self.analyzer.analyze(sig, sig, sig)
        assert result.mean_sync == 0.0

    def test_result_contains_explanation(self):
        sig = _make_bvp(hr_bpm=70.0)
        result = self.analyzer.analyze(sig, sig, sig)
        assert isinstance(result.explanation, str) and len(result.explanation) > 0

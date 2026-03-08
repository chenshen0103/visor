"""
SyncAnalyzer — measures cross-region physiological synchrony.

Real faces exhibit high Pearson correlation between forehead and cheek rPPG
signals (r > 0.80) because the same cardiac cycle drives all facial blood flow.
Deepfakes typically show incoherent or uncorrelated signals (r < 0.50).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import pearsonr

from config import SYNC_REAL_THRESHOLD, SYNC_FAKE_THRESHOLD


@dataclass
class SyncResult:
    forehead_cheek_r: float     # Pearson r between forehead and avg cheek
    left_right_r: float         # Pearson r between left and right cheek
    mean_sync: float            # Average of the two coefficients
    is_real: bool               # True if mean_sync ≥ SYNC_REAL_THRESHOLD
    is_fake: bool               # True if mean_sync < SYNC_FAKE_THRESHOLD
    explanation: str


class SyncAnalyzer:
    """
    Compute Pearson cross-correlation between facial ROI rPPG signals.
    """

    def analyze(
        self,
        forehead_sig: np.ndarray,
        left_cheek_sig: np.ndarray,
        right_cheek_sig: np.ndarray,
    ) -> SyncResult:
        """
        Parameters
        ----------
        forehead_sig, left_cheek_sig, right_cheek_sig : 1-D float arrays
            Bandpass-filtered BVP signals (same length expected; truncated if not).

        Returns
        -------
        SyncResult
        """
        # Align lengths
        min_len = min(len(forehead_sig), len(left_cheek_sig), len(right_cheek_sig))
        fh = forehead_sig[:min_len]
        lc = left_cheek_sig[:min_len]
        rc = right_cheek_sig[:min_len]

        avg_cheek = (lc + rc) / 2.0

        fh_cheek_r = self._safe_pearson(fh, avg_cheek)
        lr_r = self._safe_pearson(lc, rc)
        mean_sync = (fh_cheek_r + lr_r) / 2.0

        is_real = mean_sync >= SYNC_REAL_THRESHOLD
        is_fake = mean_sync < SYNC_FAKE_THRESHOLD

        if is_real:
            label = "real"
            explanation = (
                f"High cross-region physiological synchrony "
                f"(r={mean_sync:.3f} ≥ {SYNC_REAL_THRESHOLD}) — "
                "consistent with genuine cardiac-driven blood flow."
            )
        elif is_fake:
            label = "fake"
            explanation = (
                f"Low cross-region synchrony "
                f"(r={mean_sync:.3f} < {SYNC_FAKE_THRESHOLD}) — "
                "likely absence of real physiological signal, indicating deepfake."
            )
        else:
            label = "uncertain"
            explanation = (
                f"Borderline synchrony (r={mean_sync:.3f}); "
                "result is inconclusive — check other signals."
            )

        return SyncResult(
            forehead_cheek_r=fh_cheek_r,
            left_right_r=lr_r,
            mean_sync=mean_sync,
            is_real=is_real,
            is_fake=is_fake,
            explanation=explanation,
        )

    @staticmethod
    def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
        """Return Pearson r, handling degenerate constant arrays."""
        if a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        r, _ = pearsonr(a, b)
        return float(np.clip(r, -1.0, 1.0))

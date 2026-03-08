"""
VideoDetector — orchestrates the full rPPG deepfake-detection pipeline.

Pipeline
--------
1. ROIExtractor      → per-frame RGB series (forehead, left_cheek, right_cheek)
                       + actual video FPS
2. pos_bvp()         → classical POS rPPG BVP signal per region
                       (replaces Transformer; no training required)
3. SignalProcessor   → HR, SNR  (uses actual FPS)
4. SyncAnalyzer      → cross-region Pearson sync
5. Score fusion      → VideoVerdict

Why POS instead of the Transformer?
-------------------------------------
The PhysFormerLite Transformer was trained on a fixed FPS / ROI pipeline
that differed from inference.  POS (de Haan & Jeanne, 2013) is a classical
signal-processing algorithm that requires no model weights and is robust to
face-detection noise.  For *real* faces it recovers a clear BVP signal with
high cross-region synchrony; for AI-generated video the signal is incoherent.
The Transformer is still loaded and its output used as a tie-breaker when
the POS-based verdict is uncertain (real_prob in [0.40, 0.60]).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    SNR_MIN_REAL,
    SYNC_FAKE_THRESHOLD,
    SYNC_REAL_THRESHOLD,
    VIDEO_MIN_FRAMES,
    VIDEO_SYNC_WEIGHT,
    VIDEO_SNR_WEIGHT,
)
from modules.video.roi_extractor import extract_rgb_series
from modules.video.rppg_transformer import RPPGTransformer
from modules.video.signal_processor import pos_bvp, process_signal
from modules.video.sync_analyzer import SyncAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class VideoVerdict:
    is_real: Optional[bool]     # None when inconclusive
    confidence: float           # 0-1
    hr_bpm: float
    pearson_sync: float
    snr_db: float
    status: str                 # "real" | "fake" | "uncertain"
    explanation: str
    processing_time_ms: float


class VideoDetector:
    """
    Singleton-friendly detector; load once at app startup.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._transformer = RPPGTransformer(device=device)
        self._sync_analyzer = SyncAnalyzer()
        logger.info("VideoDetector initialised (device=%s)", device)

    def analyze(self, video_path: str | Path) -> VideoVerdict:
        t0 = time.perf_counter()
        video_path = str(video_path)

        # 1. Extract RGB time-series + actual video FPS
        result = extract_rgb_series(video_path)
        if result is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return VideoVerdict(
                is_real=None,
                confidence=0.0,
                hr_bpm=0.0,
                pearson_sync=0.0,
                snr_db=0.0,
                status="uncertain",
                explanation="Could not detect a face or video too short for rPPG analysis.",
                processing_time_ms=elapsed,
            )

        series, actual_fps = result

        if len(series["forehead"]) < VIDEO_MIN_FRAMES:
            elapsed = (time.perf_counter() - t0) * 1000
            return VideoVerdict(
                is_real=None,
                confidence=0.0,
                hr_bpm=0.0,
                pearson_sync=0.0,
                snr_db=0.0,
                status="uncertain",
                explanation=(
                    f"Video too short: only {len(series['forehead'])} frames "
                    f"(minimum {VIDEO_MIN_FRAMES} required for rPPG analysis)."
                ),
                processing_time_ms=elapsed,
            )

        # 2. Compute BVP signals using POS classical rPPG (no model weights needed)
        bvp_forehead = pos_bvp(series["forehead"])
        bvp_left     = pos_bvp(series["left_cheek"])
        bvp_right    = pos_bvp(series["right_cheek"])

        # 3. Signal processing with ACTUAL video FPS (fixes HR/SNR bin alignment)
        stats = process_signal(bvp_forehead, fps=actual_fps)

        # 4. Cross-region synchrony (POS signals are highly correlated on real faces)
        sync = self._sync_analyzer.analyze(bvp_forehead, bvp_left, bvp_right)

        # 5. Optional Transformer tie-breaker for borderline cases
        transformer_sync: Optional[float] = None
        try:
            tb_forehead = self._transformer.predict(series["forehead"])
            tb_left     = self._transformer.predict(series["left_cheek"])
            tb_right    = self._transformer.predict(series["right_cheek"])
            tb_sync_result = self._sync_analyzer.analyze(tb_forehead, tb_left, tb_right)
            transformer_sync = tb_sync_result.mean_sync
        except Exception:
            pass  # Transformer failure is non-fatal

        # 6. Score fusion
        verdict = self._fuse_scores(
            stats.snr_db,
            sync.mean_sync,
            stats.hr_bpm,
            transformer_sync=transformer_sync,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        verdict.processing_time_ms = elapsed
        return verdict

    # ------------------------------------------------------------------
    def _fuse_scores(
        self,
        snr_db: float,
        sync_r: float,
        hr_bpm: float,
        transformer_sync: Optional[float] = None,
    ) -> VideoVerdict:
        """
        Combine POS SNR and synchrony into a single authenticity score.

        SNR score  : sigmoid-normalised around SNR_MIN_REAL, 1.0 = very real
        Sync score : linear map of Pearson r [-1, 1] -> [0, 1]
                     Avoids the discontinuity in the old piecewise formula;
                     correctly handles POS signals that may be slightly
                     negative on real faces due to motion or lighting.

        For borderline verdicts (0.45 - 0.65), the Transformer sync is used
        as a mild tiebreaker (weight 0.15).
        """
        # Normalise SNR to [0, 1] via sigmoid centred at SNR_MIN_REAL
        snr_score = float(1.0 / (1.0 + np.exp(-(snr_db - SNR_MIN_REAL))))

        # Linear sync score: Pearson r in [-1, 1] mapped to [0, 1]
        sync_score = float(np.clip((sync_r + 1.0) / 2.0, 0.0, 1.0))

        # Primary fusion
        real_prob = VIDEO_SYNC_WEIGHT * sync_score + VIDEO_SNR_WEIGHT * snr_score
        real_prob = float(np.clip(real_prob, 0.0, 1.0))

        # Transformer tie-breaker in borderline zone
        if transformer_sync is not None and 0.45 <= real_prob <= 0.65:
            tb_sync_score = float(np.clip((transformer_sync + 1.0) / 2.0, 0.0, 1.0))
            real_prob = 0.85 * real_prob + 0.15 * tb_sync_score
            real_prob = float(np.clip(real_prob, 0.0, 1.0))

        if real_prob >= 0.65:
            status = "real"
            is_real = True
            confidence = real_prob
            explanation = (
                f"Strong physiological signal: "
                f"cross-region sync r={sync_r:.3f}, SNR={snr_db:.1f} dB, "
                f"HR={hr_bpm:.1f} BPM — consistent with a real human face."
            )
        elif real_prob <= 0.35:
            status = "fake"
            is_real = False
            confidence = 1.0 - real_prob
            explanation = (
                f"Weak/incoherent physiological signal: "
                f"cross-region sync r={sync_r:.3f} (threshold {SYNC_FAKE_THRESHOLD}), "
                f"SNR={snr_db:.1f} dB — likely AI-generated video."
            )
        else:
            status = "uncertain"
            is_real = None
            confidence = 0.5
            explanation = (
                f"Borderline evidence: sync r={sync_r:.3f}, "
                f"SNR={snr_db:.1f} dB, HR={hr_bpm:.1f} BPM. "
                "Manual review recommended."
            )

        return VideoVerdict(
            is_real=is_real,
            confidence=confidence,
            hr_bpm=hr_bpm,
            pearson_sync=sync_r,
            snr_db=snr_db,
            status=status,
            explanation=explanation,
            processing_time_ms=0.0,
        )

"""
VideoDetector — orchestrates the full rPPG deepfake-detection pipeline.

Pipeline
--------
1. ROIExtractor     → per-frame RGB series (forehead, left_cheek, right_cheek)
2. RPPGTransformer  → raw BVP signal per region
3. SignalProcessor  → HR, SNR
4. SyncAnalyzer     → cross-region Pearson sync
5. Score fusion     → VideoVerdict
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
    VIDEO_FPS,
)
from modules.video.roi_extractor import extract_rgb_series
from modules.video.rppg_transformer import RPPGTransformer
from modules.video.signal_processor import process_signal
from modules.video.sync_analyzer import SyncAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class VideoVerdict:
    is_real: Optional[bool]     # None when inconclusive
    confidence: float           # 0–1
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

        # 1. Extract RGB time-series from all ROIs
        series = extract_rgb_series(video_path)
        if series is None or len(series["forehead"]) < VIDEO_MIN_FRAMES:
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

        # 2. Run PhysFormerLite on each region
        bvp_forehead = self._transformer.predict(series["forehead"])
        bvp_left = self._transformer.predict(series["left_cheek"])
        bvp_right = self._transformer.predict(series["right_cheek"])

        # 3. Signal processing — use forehead as primary signal
        stats = process_signal(bvp_forehead, fps=VIDEO_FPS)

        # 4. Cross-region synchrony
        sync = self._sync_analyzer.analyze(bvp_forehead, bvp_left, bvp_right)

        # 5. Score fusion
        verdict = self._fuse_scores(stats.snr_db, sync.mean_sync, stats.hr_bpm)

        elapsed = (time.perf_counter() - t0) * 1000
        verdict.processing_time_ms = elapsed
        return verdict

    # ------------------------------------------------------------------
    def _fuse_scores(
        self,
        snr_db: float,
        sync_r: float,
        hr_bpm: float,
    ) -> VideoVerdict:
        """
        Combine SNR and synchrony into a single authenticity score.

        SNR score  : sigmoid-normalised, 1.0 = very real
        Sync score : linear map [0, 1]
        """
        # Normalise SNR to [0, 1] using a soft threshold around SNR_MIN_REAL
        snr_score = float(1.0 / (1.0 + np.exp(-(snr_db - SNR_MIN_REAL))))

        # Sync score: piecewise mapping using configured thresholds.
        # Pearson < SYNC_FAKE_THRESHOLD → [0, 0.5)  (fake region)
        # Pearson > SYNC_REAL_THRESHOLD → (0.5, 1.0] (real region)
        # Between thresholds           → 0.5         (borderline)
        if sync_r <= SYNC_FAKE_THRESHOLD:
            sync_score = float(sync_r / (SYNC_FAKE_THRESHOLD * 2.0))   # 0 → 0.5
        elif sync_r >= SYNC_REAL_THRESHOLD:
            span = max(1.0 - SYNC_REAL_THRESHOLD, 1e-6)
            sync_score = float(0.5 + (sync_r - SYNC_REAL_THRESHOLD) / (2.0 * span))
        else:
            sync_score = 0.5

        # Weighted fusion (real-probability)
        real_prob = VIDEO_SYNC_WEIGHT * sync_score + VIDEO_SNR_WEIGHT * snr_score
        real_prob = float(np.clip(real_prob, 0.0, 1.0))

        if real_prob >= 0.65:
            status = "real"
            is_real = True
            confidence = real_prob
            explanation = (
                f"Physiological signals appear genuine: "
                f"HR={hr_bpm:.1f} BPM, SNR={snr_db:.1f} dB, "
                f"cross-region sync r={sync_r:.3f}."
            )
        elif real_prob <= 0.35:
            status = "fake"
            is_real = False
            confidence = 1.0 - real_prob
            explanation = (
                f"Physiological signals inconsistent with real face: "
                f"HR={hr_bpm:.1f} BPM, SNR={snr_db:.1f} dB, "
                f"cross-region sync r={sync_r:.3f} (below threshold {SYNC_FAKE_THRESHOLD})."
            )
        else:
            status = "uncertain"
            is_real = None
            confidence = 0.5
            explanation = (
                f"Borderline rPPG evidence: HR={hr_bpm:.1f} BPM, "
                f"SNR={snr_db:.1f} dB, sync r={sync_r:.3f}."
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

"""
VideoDetector — orchestrates the full rPPG deepfake-detection pipeline.

Pipeline
--------
1. ROIExtractor      → per-frame RGB series for forehead, cheeks, AND neck
                       + actual video FPS
2. pos_bvp()         → POS classical rPPG BVP per region (no training needed)
3. SignalProcessor   → HR, SNR  (uses actual video FPS)
4. SyncAnalyzer      → within-face sync  (forehead ↔ cheeks)
5. Face-swap check   → cross-boundary sync  (face ↔ neck)
6. Score fusion      → VideoVerdict

Detection logic
---------------
- Fully AI-generated video:
    No real person → POS gives noise → within-face sync ≈ 0, low SNR → fake
- Face-swap (换臉):
    Real body/neck but synthetic face → within-face sync may look OK,
    but face-vs-neck sync breaks down (different signal sources)
    → face_neck_r will be significantly lower than within-face sync
- Real video:
    Cardiac cycle drives all skin regions equally → both syncs are high

The Transformer is retained as a mild tiebreaker for borderline cases.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr

from config import (
    SNR_MIN_REAL,
    SYNC_FAKE_THRESHOLD,
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
    pearson_sync: float         # within-face sync (forehead ↔ cheeks)
    face_neck_sync: float       # cross-boundary sync (face ↔ neck)
    snr_db: float
    status: str                 # "real" | "fake" | "face_swap" | "uncertain"
    explanation: str
    processing_time_ms: float


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(np.clip(r, -1.0, 1.0))


def _neck_is_skin(neck_series: np.ndarray) -> bool:
    """
    Heuristic check: does the neck ROI actually contain skin?

    The series is stored in BGR order (OpenCV convention):
      index 0 = Blue, index 1 = Green, index 2 = Red

    Skin criteria (empirically calibrated on UBFC-rPPG):
    - Brightness: mean Red > 70  (rules out dark/out-of-frame ROIs)
    - Colour ratio: normalised Red > 0.35  (R dominant over B in skin)
    - Colour ratio: normalised Blue < 0.38 (no blue-cast clothing/background)

    Returns False if the neck ROI is clothing, background, or out-of-frame.
    """
    mean_bgr = neck_series.mean(axis=0)   # [B, G, R]
    b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
    total = b + g + r + 1e-8
    rn = r / total
    bn = b / total
    return r > 70.0 and rn > 0.35 and bn < 0.38


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
                is_real=None, confidence=0.0, hr_bpm=0.0,
                pearson_sync=0.0, face_neck_sync=0.0, snr_db=0.0,
                status="uncertain",
                explanation="Could not detect a face or video too short for rPPG analysis.",
                processing_time_ms=elapsed,
            )

        series, actual_fps = result

        if len(series["forehead"]) < VIDEO_MIN_FRAMES:
            elapsed = (time.perf_counter() - t0) * 1000
            return VideoVerdict(
                is_real=None, confidence=0.0, hr_bpm=0.0,
                pearson_sync=0.0, face_neck_sync=0.0, snr_db=0.0,
                status="uncertain",
                explanation=(
                    f"Video too short: only {len(series['forehead'])} frames "
                    f"(minimum {VIDEO_MIN_FRAMES} required for rPPG analysis)."
                ),
                processing_time_ms=elapsed,
            )

        # 2. POS BVP for all regions
        bvp_forehead = pos_bvp(series["forehead"])
        bvp_left     = pos_bvp(series["left_cheek"])
        bvp_right    = pos_bvp(series["right_cheek"])
        bvp_neck     = pos_bvp(series["neck"])

        # 3. Signal stats (HR, SNR) from forehead signal
        stats = process_signal(bvp_forehead, fps=actual_fps)

        # 4. Within-face synchrony  (forehead ↔ cheeks)
        sync = self._sync_analyzer.analyze(bvp_forehead, bvp_left, bvp_right)
        within_face_r = sync.mean_sync

        # 5. Cross-boundary synchrony  (avg face ↔ neck)
        # Only meaningful when the neck ROI contains real skin — otherwise
        # the ROI lands on clothing, dark background, or an out-of-frame area,
        # producing spurious sync values and false face-swap detections.
        neck_valid   = _neck_is_skin(series["neck"])
        bvp_face_avg = (bvp_forehead + bvp_left + bvp_right) / 3.0
        face_neck_r  = _safe_pearson(bvp_face_avg, bvp_neck) if neck_valid else None

        # 6. Optional Transformer tie-breaker
        transformer_sync: Optional[float] = None
        try:
            tb_fh = self._transformer.predict(series["forehead"])
            tb_lc = self._transformer.predict(series["left_cheek"])
            tb_rc = self._transformer.predict(series["right_cheek"])
            transformer_sync = self._sync_analyzer.analyze(tb_fh, tb_lc, tb_rc).mean_sync
        except Exception:
            pass

        # 7. Score fusion
        verdict = self._fuse_scores(
            snr_db=stats.snr_db,
            within_face_r=within_face_r,
            face_neck_r=face_neck_r,        # None when neck is not skin
            hr_bpm=stats.hr_bpm,
            transformer_sync=transformer_sync,
        )

        verdict.processing_time_ms = (time.perf_counter() - t0) * 1000
        return verdict

    # ------------------------------------------------------------------
    def _fuse_scores(
        self,
        snr_db: float,
        within_face_r: float,
        face_neck_r: Optional[float],   # None when neck ROI is not skin
        hr_bpm: float,
        transformer_sync: Optional[float] = None,
    ) -> VideoVerdict:
        """
        Three-signal fusion for real / face-swap / fully-fake classification.

        Signals
        -------
        SNR          : physiological signal quality (real faces → high)
        within_face_r: forehead ↔ cheeks sync  (high for real AND well-blended fakes)
        face_neck_r  : face ↔ neck sync (None when neck ROI is invalid)
                       real      → moderate-to-high  (same cardiac cycle)
                       face-swap → significantly lower than within-face sync
                       fully-AI  → irrelevant (face also scores low)

        Face-swap heuristic
        -------------------
        Only triggered when neck ROI contains real skin AND:
          - within-face sync ≥ 0.50  (face coherent on its own)
          - sync gap ≥ 0.55          (face-neck discrepancy is large)
          - SNR ≥ SNR_MIN_REAL       (some physiological signal present)
        These stricter thresholds reduce false positives from cases where the
        neck ROI accidentally lands on a shirt collar or camera angle crops it.
        """
        # --- SNR score (sigmoid) ---
        snr_score = float(1.0 / (1.0 + np.exp(-(snr_db - SNR_MIN_REAL))))

        # --- Within-face sync score (linear) ---
        face_sync_score = float(np.clip((within_face_r + 1.0) / 2.0, 0.0, 1.0))

        # --- Primary real probability (uses within-face + SNR) ---
        real_prob = VIDEO_SYNC_WEIGHT * face_sync_score + VIDEO_SNR_WEIGHT * snr_score
        real_prob = float(np.clip(real_prob, 0.0, 1.0))

        # --- Transformer tie-breaker in borderline zone ---
        if transformer_sync is not None and 0.45 <= real_prob <= 0.65:
            tb_score = float(np.clip((transformer_sync + 1.0) / 2.0, 0.0, 1.0))
            real_prob = 0.85 * real_prob + 0.15 * tb_score
            real_prob = float(np.clip(real_prob, 0.0, 1.0))

        # --- Face-swap detection ---
        # Only active when neck ROI is confirmed to contain skin.
        # Requires a large discrepancy between within-face and face-neck sync.
        face_swap_flag = False
        sync_gap = 0.0
        if face_neck_r is not None:
            sync_gap = within_face_r - face_neck_r
            face_swap_flag = (
                within_face_r >= 0.50      # face is coherent (strict)
                and sync_gap >= 0.55       # large face-neck discrepancy (strict)
                and snr_db >= SNR_MIN_REAL # physiological signal exists
            )

        # --- Final verdict ---
        if face_swap_flag and real_prob >= 0.50:
            status    = "face_swap"
            is_real   = False
            confidence = float(np.clip(0.5 + sync_gap / 2.0, 0.5, 0.95))
            explanation = (
                f"Possible face-swap detected: within-face sync r={within_face_r:.3f} "
                f"is high but face-neck sync r={face_neck_r:.3f} is much lower "
                f"(gap={sync_gap:.3f}). The face region may be synthesised while "
                f"the neck/body remains real."
            )
        elif real_prob >= 0.65:
            status    = "real"
            is_real   = True
            confidence = real_prob
            neck_str = f", face-neck sync r={face_neck_r:.3f}" if face_neck_r is not None else ""
            explanation = (
                f"Strong physiological signal: within-face sync r={within_face_r:.3f}"
                f"{neck_str}, SNR={snr_db:.1f} dB, "
                f"HR={hr_bpm:.1f} BPM — consistent with a real human face."
            )
        elif real_prob <= 0.35:
            status    = "fake"
            is_real   = False
            confidence = 1.0 - real_prob
            explanation = (
                f"Weak/incoherent physiological signal: "
                f"within-face sync r={within_face_r:.3f}, "
                f"SNR={snr_db:.1f} dB — likely fully AI-generated video."
            )
        else:
            status    = "uncertain"
            is_real   = None
            confidence = 0.5
            neck_str = f", face-neck sync r={face_neck_r:.3f}" if face_neck_r is not None else ""
            explanation = (
                f"Borderline evidence: within-face sync r={within_face_r:.3f}"
                f"{neck_str}, SNR={snr_db:.1f} dB, HR={hr_bpm:.1f} BPM. "
                "Manual review recommended."
            )

        face_neck_out = face_neck_r if face_neck_r is not None else 0.0
        return VideoVerdict(
            is_real=is_real,
            confidence=confidence,
            hr_bpm=hr_bpm,
            pearson_sync=within_face_r,
            face_neck_sync=face_neck_out,
            snr_db=snr_db,
            status=status,
            explanation=explanation,
            processing_time_ms=0.0,
        )

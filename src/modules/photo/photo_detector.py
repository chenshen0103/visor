"""
PhotoDetector — orchestrates lens geometry + PRNU analysis to detect AI-generated images.

Score fusion
------------
- geometry_score  (weight 0.40): based on vanishing-point consistency and distortion fit
- prnu_score      (weight 0.60): PRNU presence + absence of periodic artifacts
→ weighted real_probability → PhotoVerdict
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from config import (
    PHOTO_GEOMETRY_WEIGHT,
    PHOTO_PRNU_WEIGHT,
)
from modules.photo.lens_geometry import LensGeometryAnalyzer
from modules.photo.prnu_analyzer import PRNUAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PhotoVerdict:
    is_real: Optional[bool]     # None = uncertain
    confidence: float
    status: str                 # "real" | "fake" | "uncertain"
    geometry_score: float       # 0–1 (1 = real-like geometry)
    prnu_score: float           # 0–1 (1 = real camera PRNU)
    has_periodic_artifacts: bool
    explanation: str
    processing_time_ms: float


class PhotoDetector:
    """
    Load once at startup; thread-safe (stateless analyzers).
    """

    def __init__(self) -> None:
        self._geo = LensGeometryAnalyzer()
        self._prnu = PRNUAnalyzer()
        logger.info("PhotoDetector initialised")

    def analyze(self, image_path: Union[str, Path]) -> PhotoVerdict:
        t0 = time.perf_counter()
        image_path = str(image_path)

        img = cv2.imread(image_path)
        if img is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return PhotoVerdict(
                is_real=None,
                confidence=0.0,
                status="uncertain",
                geometry_score=0.5,
                prnu_score=0.5,
                has_periodic_artifacts=False,
                explanation="Could not load image.",
                processing_time_ms=elapsed,
            )

        verdict = self.analyze_array(img)
        verdict.processing_time_ms = (time.perf_counter() - t0) * 1000
        return verdict

    def analyze_array(self, img: np.ndarray) -> PhotoVerdict:
        """Analyse an in-memory BGR uint8 image array."""
        t0 = time.perf_counter()

        # 1. Lens geometry branch
        vp_result = self._geo.analyze_lines(img)
        distortion_result = self._geo.estimate_radial_distortion(img)

        # Geometry score: consistent VP + has distortion → real camera
        vp_score = vp_result.consistency_score
        dist_score = 1.0 if distortion_result.has_distortion else 0.3
        geometry_score = float(0.6 * vp_score + 0.4 * dist_score)

        # 2. PRNU branch
        noise = self._prnu.extract_noise_residual(img)
        prnu_result = self._prnu.compute_prnu_energy(noise)
        upsample_result = self._prnu.detect_upsampling_artifacts(noise)

        # PRNU score: good PRNU + no periodic artifacts → real camera
        artifact_penalty = 0.5 if upsample_result.has_periodic_artifacts else 0.0
        prnu_score = float(np.clip(prnu_result.prnu_score - artifact_penalty, 0.0, 1.0))

        # 3. Fuse
        real_prob = (
            PHOTO_GEOMETRY_WEIGHT * geometry_score
            + PHOTO_PRNU_WEIGHT * prnu_score
        )
        real_prob = float(np.clip(real_prob, 0.0, 1.0))

        status, is_real, confidence = self._classify(real_prob)
        explanation = self._build_explanation(
            status, real_prob, geometry_score, prnu_score,
            vp_result.consistency_score, distortion_result.has_distortion,
            upsample_result.has_periodic_artifacts,
        )

        return PhotoVerdict(
            is_real=is_real,
            confidence=confidence,
            status=status,
            geometry_score=geometry_score,
            prnu_score=prnu_score,
            has_periodic_artifacts=upsample_result.has_periodic_artifacts,
            explanation=explanation,
            processing_time_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _classify(real_prob: float):
        if real_prob >= 0.65:
            return "real", True, real_prob
        elif real_prob <= 0.35:
            return "fake", False, 1.0 - real_prob
        else:
            return "uncertain", None, 0.5

    @staticmethod
    def _build_explanation(
        status: str,
        real_prob: float,
        geo: float,
        prnu: float,
        vp_consistency: float,
        has_distortion: bool,
        has_artifacts: bool,
    ) -> str:
        parts = [f"Image classified as '{status}' (real_prob={real_prob:.2f})."]
        parts.append(
            f"Lens geometry score: {geo:.2f} "
            f"(VP consistency={vp_consistency:.2f}, "
            f"lens distortion={'present' if has_distortion else 'absent'})."
        )
        parts.append(
            f"PRNU score: {prnu:.2f} "
            f"({'no ' if not has_artifacts else ''}periodic upsampling artifacts detected)."
        )
        return " ".join(parts)

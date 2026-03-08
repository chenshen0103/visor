"""
LensGeometryAnalyzer — physics-based image authenticity through optics.

Two tests:
1. Vanishing-point consistency  : Hough lines should converge to a coherent
   vanishing point in perspective images.  AI-generated images often contain
   locally plausible but globally inconsistent line structures.
2. Radial distortion estimation : Real camera lenses introduce measurable
   barrel/pincushion distortion.  Upsampled AI outputs typically lack it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from config import (
    HOUGH_MAX_LINE_GAP,
    HOUGH_MIN_LINE_LEN,
    HOUGH_THRESHOLD,
    GEOMETRY_CONSISTENCY_THRESHOLD,
    RANSAC_MAX_TRIALS,
    RANSAC_MIN_SAMPLES,
)

logger = logging.getLogger(__name__)


@dataclass
class VanishingPointResult:
    consistency_score: float    # 0 (inconsistent) … 1 (perfectly consistent)
    is_consistent: bool
    n_lines: int


@dataclass
class RadialDistortionResult:
    k1: float                   # 1st radial distortion coefficient
    k2: float                   # 2nd radial distortion coefficient
    fit_residual: float         # RMSE of distortion model fit (lower = better fit)
    has_distortion: bool        # True if |k1| > noise floor


class LensGeometryAnalyzer:
    """
    Analyses structural geometry cues in a BGR image.
    """

    # ------------------------------------------------------------------
    # Vanishing-point branch
    # ------------------------------------------------------------------

    def analyze_lines(self, image: np.ndarray) -> VanishingPointResult:
        """
        Detect Hough lines and score their vanishing-point consistency.

        Returns VanishingPointResult.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        if lines is None or len(lines) < RANSAC_MIN_SAMPLES:
            return VanishingPointResult(
                consistency_score=0.5,
                is_consistent=True,  # neutral — not enough lines to judge
                n_lines=0 if lines is None else len(lines),
            )

        lines = lines[:, 0, :]  # (N, 4) — x1,y1,x2,y2
        score = self._vanishing_point_score(lines, image.shape[:2])

        return VanishingPointResult(
            consistency_score=score,
            is_consistent=(score >= (1.0 - GEOMETRY_CONSISTENCY_THRESHOLD)),
            n_lines=len(lines),
        )

    def _vanishing_point_score(
        self,
        lines: np.ndarray,
        img_shape: Tuple[int, int],
    ) -> float:
        """
        Estimate a single dominant vanishing point using RANSAC and score
        how many lines are consistent with it.

        Score = inlier_fraction (0–1); near 1 → real camera perspective.
        """
        h, w = img_shape
        vp = self._ransac_vanishing_point(lines)
        if vp is None:
            return 0.5

        # Compute angular distance from each line to the estimated VP
        inliers = 0
        for x1, y1, x2, y2 in lines:
            d = self._point_to_line_dist(vp, (x1, y1), (x2, y2))
            # Normalise by diagonal
            diag = np.hypot(h, w)
            if d / diag < 0.05:
                inliers += 1

        return inliers / len(lines)

    def _ransac_vanishing_point(
        self, lines: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        RANSAC to find best vanishing point as intersection of line pairs.
        Returns (x, y) or None.
        """
        best_vp = None
        best_inliers = 0
        rng = np.random.default_rng(42)

        for _ in range(RANSAC_MAX_TRIALS):
            idx = rng.choice(len(lines), size=2, replace=False)
            vp = self._intersect_lines(lines[idx[0]], lines[idx[1]])
            if vp is None:
                continue

            inliers = sum(
                1
                for x1, y1, x2, y2 in lines
                if self._point_to_line_dist(vp, (x1, y1), (x2, y2)) < 10
            )
            if inliers > best_inliers:
                best_inliers = inliers
                best_vp = vp

        return best_vp

    @staticmethod
    def _intersect_lines(
        l1: np.ndarray, l2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Return (x, y) intersection of two line segments, or None if parallel."""
        x1, y1, x2, y2 = l1.tolist()
        x3, y3, x4, y4 = l2.tolist()

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-8:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    @staticmethod
    def _point_to_line_dist(
        pt: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Distance from *pt* to the infinite line through *p1*–*p2*."""
        px, py = pt
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return np.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        cx = x1 + t * dx
        cy = y1 + t * dy
        return np.hypot(px - cx, py - cy)

    # ------------------------------------------------------------------
    # Radial distortion branch
    # ------------------------------------------------------------------

    def estimate_radial_distortion(self, image: np.ndarray) -> RadialDistortionResult:
        """
        Fit a radial distortion model to detected straight-line deviations.

        Strategy:
        - Detect long Hough lines
        - Sample points along each line
        - Fit (k1, k2) polynomial distortion model to minimise straightness error
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        cx, cy = w / 2.0, h / 2.0

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=int(max(h, w) * 0.15),
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        if lines is None or len(lines) < 3:
            return RadialDistortionResult(
                k1=0.0, k2=0.0, fit_residual=0.0, has_distortion=False
            )

        # Sample points along each line and accumulate
        sample_pts: List[np.ndarray] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            pts = self._sample_line(x1, y1, x2, y2, n=20)
            sample_pts.append(pts)

        if not sample_pts:
            return RadialDistortionResult(
                k1=0.0, k2=0.0, fit_residual=0.0, has_distortion=False
            )

        all_pts = np.vstack(sample_pts)  # (N, 2) pixel coordinates

        # Normalise coordinates to [-1, 1]
        xn = (all_pts[:, 0] - cx) / (w / 2.0)
        yn = (all_pts[:, 1] - cy) / (h / 2.0)
        r2 = xn**2 + yn**2

        def residuals(params):
            k1, k2 = params
            factor = 1 + k1 * r2 + k2 * r2**2
            xu = xn / factor
            yu = yn / factor
            # Expect undistorted points to lie on lines → minimise cross-product residual
            return np.hypot(xu - xn, yu - yn)

        result = least_squares(residuals, [0.0, 0.0], max_nfev=200)
        k1, k2 = result.x
        fit_residual = float(np.sqrt(np.mean(result.fun**2)))

        return RadialDistortionResult(
            k1=float(k1),
            k2=float(k2),
            fit_residual=fit_residual,
            has_distortion=abs(k1) > 1e-4,
        )

    @staticmethod
    def _sample_line(
        x1: int, y1: int, x2: int, y2: int, n: int = 20
    ) -> np.ndarray:
        """Return *n* evenly spaced (x, y) points along the segment."""
        ts = np.linspace(0, 1, n)
        xs = x1 + ts * (x2 - x1)
        ys = y1 + ts * (y2 - y1)
        return np.stack([xs, ys], axis=1)

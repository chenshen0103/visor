"""
ROIExtractor — uses OpenCV Haar cascade to locate face, then derives
forehead and cheek ROIs from the bounding box for rPPG signal extraction.

Replaces the previous MediaPipe FaceMesh implementation which is no longer
available in mediapipe ≥ 0.10 (solutions API removed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from config import (
    ROI_FOREHEAD_H,
    ROI_FOREHEAD_W,
    ROI_CHEEK_H,
    ROI_CHEEK_W,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Haar cascade once at import time
# ---------------------------------------------------------------------------
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
if _face_cascade.empty():
    logger.error("Failed to load Haar cascade from %s", _CASCADE_PATH)


@dataclass
class ROIPatches:
    forehead: np.ndarray       # (H, W, 3) BGR
    left_cheek: np.ndarray     # (H, W, 3) BGR
    right_cheek: np.ndarray    # (H, W, 3) BGR


def _crop_patch(
    frame: np.ndarray,
    cx: int,
    cy: int,
    half_h: int,
    half_w: int,
) -> np.ndarray:
    """Return a (2*half_h, 2*half_w, 3) BGR patch centred at (cx, cy), clamped to frame edges."""
    h, w = frame.shape[:2]
    y1 = max(0, cy - half_h)
    y2 = min(h, cy + half_h)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)
    patch = frame[y1:y2, x1:x2]
    target = (2 * half_w, 2 * half_h)
    if patch.size == 0:
        return np.zeros((2 * half_h, 2 * half_w, 3), dtype=np.uint8)
    return cv2.resize(patch, target, interpolation=cv2.INTER_AREA)


class ROIExtractor:
    """
    Face ROI extractor using OpenCV Haar cascade.

    Derives three regions from the detected face bounding box:
    - forehead : top 20% of the face box (centred horizontally)
    - left_cheek : left-side, mid-height region
    - right_cheek: right-side, mid-height region

    Falls back to fixed centre-frame regions when no face is detected.
    """

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = False) -> None:
        self._half_fh = ROI_FOREHEAD_H // 2
        self._half_fw = ROI_FOREHEAD_W // 2
        self._half_ch = ROI_CHEEK_H // 2
        self._half_cw = ROI_CHEEK_W // 2

    def extract(self, frame: np.ndarray) -> Optional[ROIPatches]:
        """
        Detect face in *frame* (BGR uint8) and return three ROI patches.
        Returns None if frame is invalid; uses fixed fallback ROI if no face found.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # Fallback: use centre of frame as face region
            fx, fy, fw, fh = w // 4, h // 8, w // 2, int(h * 0.7)
        else:
            # Use largest detected face
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])

        # Forehead: top 20% of face box, centred
        fh_cy = fy + int(fh * 0.10)
        fh_cx = fx + fw // 2

        # Left cheek: left quarter of face at mid height
        lc_cx = fx + fw // 4
        lc_cy = fy + int(fh * 0.55)

        # Right cheek: right quarter at mid height
        rc_cx = fx + int(fw * 0.75)
        rc_cy = fy + int(fh * 0.55)

        forehead   = _crop_patch(frame, fh_cx, fh_cy, self._half_fh, self._half_fw)
        left_cheek = _crop_patch(frame, lc_cx, lc_cy, self._half_ch, self._half_cw)
        right_cheek= _crop_patch(frame, rc_cx, rc_cy, self._half_ch, self._half_cw)

        return ROIPatches(
            forehead=forehead,
            left_cheek=left_cheek,
            right_cheek=right_cheek,
        )

    def close(self) -> None:
        pass  # No resources to release

    def __enter__(self) -> "ROIExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def extract_rgb_series(
    video_path: str,
    max_frames: int = 300,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Decode *video_path*, run ROI extraction on every frame, and return
    mean-RGB time-series per region.

    Returns
    -------
    dict with keys 'forehead', 'left_cheek', 'right_cheek',
    each a float32 array of shape (T, 3), or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return None

    series: Dict[str, list] = {"forehead": [], "left_cheek": [], "right_cheek": []}
    frames_read = 0

    with ROIExtractor() as extractor:
        while frames_read < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            patches = extractor.extract(frame)
            if patches is not None:
                for key, patch in [
                    ("forehead", patches.forehead),
                    ("left_cheek", patches.left_cheek),
                    ("right_cheek", patches.right_cheek),
                ]:
                    mean_rgb = patch.mean(axis=(0, 1)).astype(np.float32)  # (3,)
                    series[key].append(mean_rgb)
            frames_read += 1

    cap.release()

    if not series["forehead"]:
        logger.warning("No usable frames in video: %s", video_path)
        return None

    return {k: np.stack(v, axis=0) for k, v in series.items()}

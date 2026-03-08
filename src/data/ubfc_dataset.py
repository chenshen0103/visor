"""
UBFCDataset — PyTorch Dataset wrapper for UBFC-rPPG.

UBFC-rPPG directory layout expected:
    <root>/
        subject1/
            vid.avi
            ground_truth.txt    (BVP signal, one value per line)
        subject2/
            ...

Reference: Bobbia et al., "Unsupervised skin tissue segmentation for remote
photoplethysmography", 2019.

Download: https://sites.google.com/view/ybenezeth/ubfcrppg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_MIN_FRAMES
from modules.video.roi_extractor import extract_rgb_series

logger = logging.getLogger(__name__)


class UBFCDataset(Dataset):
    """
    Returns (rgb_series, bvp_label) pairs.

    Parameters
    ----------
    root : str | Path
        Path to UBFC-rPPG dataset root directory.
    max_frames : int
        Truncate each video to this many frames.
    augment : bool
        If True, apply simple temporal jitter.
    """

    def __init__(
        self,
        root: str | Path,
        max_frames: int = VIDEO_MAX_FRAMES,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.max_frames = max_frames
        self.augment = augment
        self.samples: List[Tuple[Path, Path]] = self._discover()
        logger.info("UBFCDataset: %d subjects found in %s", len(self.samples), root)

    def _discover(self) -> List[Tuple[Path, Path]]:
        """Return list of (video_path, gt_path) tuples."""
        samples = []
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue
            video = subject_dir / "vid.avi"
            gt = subject_dir / "ground_truth.txt"
            if video.exists() and gt.exists():
                samples.append((video, gt))
            else:
                logger.warning("Skipping %s (missing vid.avi or ground_truth.txt)", subject_dir)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, gt_path = self.samples[idx]

        # Extract forehead RGB series
        # extract_rgb_series returns (series_dict, actual_fps) or None
        result = extract_rgb_series(str(video_path), max_frames=self.max_frames)
        if result is None:
            # Return zeros as fallback (should not happen on clean dataset)
            rgb = np.zeros((self.max_frames, 3), dtype=np.float32)
            bvp = np.zeros(self.max_frames, dtype=np.float32)
        else:
            series, _fps = result
            rgb = series["forehead"][: self.max_frames]  # (T, 3)
            bvp = self._load_bvp(gt_path)[: self.max_frames]

        # Pad / truncate to exactly max_frames
        T = rgb.shape[0]
        if T < self.max_frames:
            pad = self.max_frames - T
            rgb = np.pad(rgb, ((0, pad), (0, 0)), mode="edge")
            bvp = np.pad(bvp, (0, pad), mode="edge")
        else:
            rgb = rgb[: self.max_frames]
            bvp = bvp[: self.max_frames]

        # Normalise RGB
        mu = rgb.mean(axis=0, keepdims=True)
        sigma = rgb.std(axis=0, keepdims=True) + 1e-8
        rgb = (rgb - mu) / sigma

        # Normalise BVP
        bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-8)

        if self.augment:
            rgb, bvp = self._temporal_jitter(rgb, bvp)

        return torch.from_numpy(rgb).float(), torch.from_numpy(bvp).float()

    @staticmethod
    def _load_bvp(gt_path: Path) -> np.ndarray:
        """Parse UBFC ground-truth file (whitespace-separated floats)."""
        text = gt_path.read_text(encoding="utf-8").strip()
        values = [float(v) for v in text.split() if v]
        return np.array(values, dtype=np.float32)

    @staticmethod
    def _temporal_jitter(
        rgb: np.ndarray, bvp: np.ndarray, max_shift: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random temporal crop for data augmentation."""
        shift = np.random.randint(0, max_shift + 1)
        if shift > 0:
            rgb = rgb[shift:]
            bvp = bvp[shift:]
        return rgb, bvp

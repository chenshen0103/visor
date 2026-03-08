"""
PRNUAnalyzer — Photo-Response Non-Uniformity analysis for camera fingerprinting.

Real photographs contain a consistent, camera-specific noise pattern (PRNU) that
survives mild JPEG compression.  AI-generated images lack a true PRNU and instead
show upsampling/generation artifacts that manifest as periodic patterns in the FFT.

Pipeline per image
------------------
1. Wavelet denoise   → denoised image F(I)
2. Noise residual    → W = I – F(I)
3. 2-D FFT analysis → spatial frequency energy distribution
4. Detect periodic spikes (upsampling artifacts)
5. Estimate structured noise energy (PRNU presence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage.restoration import denoise_wavelet
from skimage.util import img_as_float32

from config import (
    PERIODIC_ARTIFACT_THRESHOLD,
    PRNU_ENERGY_REAL_THRESHOLD,
    WAVELET,
    WAVELET_LEVEL,
)

logger = logging.getLogger(__name__)


@dataclass
class PRNUResult:
    prnu_score: float       # 0 (no PRNU) … 1 (strong camera PRNU)
    is_real_camera: bool
    noise_energy: float     # overall noise residual energy


@dataclass
class UpsamplingResult:
    has_periodic_artifacts: bool
    artifact_ratio: float   # fraction of FFT energy concentrated in peaks
    peak_frequencies: list  # dominant peak frequencies


class PRNUAnalyzer:
    """
    Stateless analyser; all methods accept a BGR uint8 numpy image.
    """

    # ------------------------------------------------------------------
    # Noise residual extraction
    # ------------------------------------------------------------------

    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the noise residual W = I − F(I) using wavelet denoising.

        Parameters
        ----------
        image : (H, W, 3) BGR uint8

        Returns
        -------
        (H, W, 3) float32 noise residual in [−1, 1]
        """
        img_f = img_as_float32(image[..., ::-1])  # BGR → RGB, float32 [0,1]

        denoised = denoise_wavelet(
            img_f,
            channel_axis=-1,
            wavelet=WAVELET,
            wavelet_levels=WAVELET_LEVEL,
            method="BayesShrink",
            mode="soft",
        ).astype(np.float32)

        residual = img_f - denoised
        return residual  # (H, W, 3) float32

    # ------------------------------------------------------------------
    # PRNU energy
    # ------------------------------------------------------------------

    def compute_prnu_energy(self, noise: np.ndarray) -> PRNUResult:
        """
        Score the structured PRNU energy in the noise residual.

        Real cameras produce spatially non-uniform PRNU.  We measure this
        as the variance of the residual relative to its mean magnitude.
        AI images show near-uniform noise with lower spatial structure.

        Parameters
        ----------
        noise : (H, W, 3) float32 noise residual

        Returns
        -------
        PRNUResult
        """
        # Use luminance channel
        lum = noise.mean(axis=2)  # (H, W)

        noise_energy = float(np.sqrt(np.mean(lum ** 2)))

        # Spatial variance of noise energy (block-wise)
        block_energies = self._block_energies(lum, block_size=32)
        if block_energies.size < 2:
            prnu_score = 0.5
        else:
            cv = block_energies.std() / (block_energies.mean() + 1e-8)
            # High CV → spatially non-uniform → real PRNU
            prnu_score = float(np.clip(cv / 0.5, 0.0, 1.0))

        is_real = prnu_score >= (1.0 - PRNU_ENERGY_REAL_THRESHOLD)

        return PRNUResult(
            prnu_score=prnu_score,
            is_real_camera=is_real,
            noise_energy=noise_energy,
        )

    @staticmethod
    def _block_energies(img: np.ndarray, block_size: int = 32) -> np.ndarray:
        """Return array of RMS energy in non-overlapping blocks."""
        h, w = img.shape
        energies = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img[y : y + block_size, x : x + block_size]
                energies.append(np.sqrt(np.mean(block ** 2)))
        return np.array(energies, dtype=np.float32)

    # ------------------------------------------------------------------
    # Upsampling / generation artifact detection via 2-D FFT
    # ------------------------------------------------------------------

    def detect_upsampling_artifacts(self, noise: np.ndarray) -> UpsamplingResult:
        """
        Look for periodic peaks in the 2-D power spectrum.

        AI image generators and up-samplers often leave grid-like artifacts
        at regular spatial frequencies (e.g., every N pixels due to stride).

        Parameters
        ----------
        noise : (H, W, 3) float32

        Returns
        -------
        UpsamplingResult
        """
        lum = noise.mean(axis=2)  # (H, W)
        fft2 = np.fft.fft2(lum)
        power = np.abs(np.fft.fftshift(fft2)) ** 2

        h, w = power.shape
        # Zero out DC component
        power[h // 2, w // 2] = 0.0

        total_power = power.sum() + 1e-12

        # Detect peaks above mean + 3*std
        threshold = power.mean() + 3.0 * power.std()
        peak_mask = power > threshold
        peak_power = power[peak_mask].sum()

        artifact_ratio = float(peak_power / total_power)
        has_artifacts = artifact_ratio > PERIODIC_ARTIFACT_THRESHOLD

        # Collect peak frequency positions (normalised to [-0.5, 0.5])
        ys, xs = np.where(peak_mask)
        freq_y = (ys - h // 2) / h
        freq_x = (xs - w // 2) / w
        peaks = list(zip(freq_x.tolist(), freq_y.tolist()))[:10]

        return UpsamplingResult(
            has_periodic_artifacts=has_artifacts,
            artifact_ratio=artifact_ratio,
            peak_frequencies=peaks,
        )

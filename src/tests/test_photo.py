"""
Unit tests for the photo deepfake detection module.

Uses synthetic numpy images so no real image files are required.
"""

import numpy as np
import pytest

from modules.photo.lens_geometry import LensGeometryAnalyzer
from modules.photo.prnu_analyzer import PRNUAnalyzer
from modules.photo.photo_detector import PhotoDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _checkerboard(h: int = 256, w: int = 256) -> np.ndarray:
    """BGR checkerboard — contains strong edges for Hough line detection."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    block = 32
    for y in range(0, h, block):
        for x in range(0, w, block):
            if (y // block + x // block) % 2 == 0:
                img[y : y + block, x : x + block] = 255
    return img


def _gradient_image(h: int = 256, w: int = 256) -> np.ndarray:
    """Simple gradient image (real-camera-like smooth content)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.arange(0, 256, 256 / w, dtype=np.uint8), (h, 1))
    img[:, :, 1] = np.tile(
        np.arange(0, 256, 256 / h, dtype=np.uint8).reshape(-1, 1), (1, w)
    )
    img[:, :, 2] = 128
    return img


def _noise_image(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """Pure Gaussian noise image (resembles AI-generated flat texture)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# LensGeometryAnalyzer
# ---------------------------------------------------------------------------

class TestLensGeometryAnalyzer:
    def setup_method(self):
        self.analyzer = LensGeometryAnalyzer()

    def test_analyze_lines_returns_result(self):
        img = _checkerboard()
        result = self.analyzer.analyze_lines(img)
        assert 0.0 <= result.consistency_score <= 1.0
        assert isinstance(result.is_consistent, bool)
        assert result.n_lines >= 0

    def test_estimate_distortion_returns_result(self):
        img = _checkerboard()
        result = self.analyzer.estimate_radial_distortion(img)
        assert isinstance(result.k1, float)
        assert isinstance(result.fit_residual, float)

    def test_blank_image_does_not_crash(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = self.analyzer.analyze_lines(img)
        assert result.n_lines == 0

    def test_single_pixel_image(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = self.analyzer.analyze_lines(img)
        assert result is not None


# ---------------------------------------------------------------------------
# PRNUAnalyzer
# ---------------------------------------------------------------------------

class TestPRNUAnalyzer:
    def setup_method(self):
        self.analyzer = PRNUAnalyzer()

    def test_extract_noise_residual_shape(self):
        img = _gradient_image()
        noise = self.analyzer.extract_noise_residual(img)
        assert noise.shape == img.shape
        assert noise.dtype == np.float32

    def test_prnu_energy_result(self):
        img = _gradient_image()
        noise = self.analyzer.extract_noise_residual(img)
        result = self.analyzer.compute_prnu_energy(noise)
        assert 0.0 <= result.prnu_score <= 1.0
        assert isinstance(result.is_real_camera, bool)

    def test_upsampling_artifacts_detection(self):
        img = _noise_image()
        noise = self.analyzer.extract_noise_residual(img)
        result = self.analyzer.detect_upsampling_artifacts(noise)
        assert isinstance(result.has_periodic_artifacts, bool)
        assert 0.0 <= result.artifact_ratio <= 1.0

    def test_small_image(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        noise = self.analyzer.extract_noise_residual(img)
        assert noise is not None


# ---------------------------------------------------------------------------
# PhotoDetector (integration)
# ---------------------------------------------------------------------------

class TestPhotoDetector:
    def setup_method(self):
        self.detector = PhotoDetector()

    def test_analyze_array_gradient(self):
        img = _gradient_image()
        verdict = self.detector.analyze_array(img)
        assert verdict.status in ("real", "fake", "uncertain")
        assert 0.0 <= verdict.confidence <= 1.0
        assert 0.0 <= verdict.geometry_score <= 1.0
        assert 0.0 <= verdict.prnu_score <= 1.0
        assert isinstance(verdict.has_periodic_artifacts, bool)

    def test_analyze_array_checkerboard(self):
        img = _checkerboard()
        verdict = self.detector.analyze_array(img)
        assert verdict.status in ("real", "fake", "uncertain")

    def test_analyze_array_noise(self):
        img = _noise_image()
        verdict = self.detector.analyze_array(img)
        assert verdict.status in ("real", "fake", "uncertain")

    def test_explanation_is_string(self):
        img = _gradient_image()
        verdict = self.detector.analyze_array(img)
        assert isinstance(verdict.explanation, str) and len(verdict.explanation) > 0

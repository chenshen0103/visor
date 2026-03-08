"""
Integration tests for the FastAPI endpoints.

Uses httpx.AsyncClient with the ASGI transport so no server is needed.
Model loading is mocked where possible to speed up CI.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# App fixture — override lifespan to inject lightweight mocks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_with_mocks():
    """
    Return a FastAPI app instance where detectors are replaced with stubs
    that return fixed verdicts, avoiding heavy model loading in CI.
    """
    from modules.video.video_detector import VideoVerdict
    from modules.photo.photo_detector import PhotoVerdict
    from modules.text.text_detector import TextVerdict

    class _StubVideoDetector:
        def analyze(self, path):
            return VideoVerdict(
                is_real=True,
                confidence=0.92,
                hr_bpm=72.0,
                pearson_sync=0.88,
                snr_db=5.2,
                status="real",
                explanation="Stub verdict.",
                processing_time_ms=10.0,
            )

    class _StubPhotoDetector:
        def analyze_array(self, img):
            return PhotoVerdict(
                is_real=True,
                confidence=0.80,
                status="real",
                geometry_score=0.75,
                prnu_score=0.85,
                has_periodic_artifacts=False,
                explanation="Stub verdict.",
                processing_time_ms=5.0,
            )
        def analyze(self, path):
            return self.analyze_array(None)

    class _StubTextDetector:
        def load(self): pass
        def analyze(self, text):
            return TextVerdict(
                is_scam=True,
                confidence=0.87,
                status="scam",
                closest_archetype="government_impersonation",
                closest_archetype_zh="假冒公務機關詐騙",
                intent_similarity=0.81,
                rag_scam_ratio=0.8,
                rag_evidence=[],
                explanation="Stub: government impersonation detected.",
                processing_time_ms=3.0,
            )

    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from api.router import api_router
    from config import API_V1_PREFIX

    @asynccontextmanager
    async def _mock_lifespan(application):
        application.state.video_detector = _StubVideoDetector()
        application.state.photo_detector = _StubPhotoDetector()
        application.state.text_detector = _StubTextDetector()
        application.state.models_loaded = {
            "video_detector": "stub",
            "photo_detector": "stub",
            "text_detector": "stub",
        }
        yield

    test_app = FastAPI(lifespan=_mock_lifespan)
    # Pre-set state directly so ASGITransport tests work without lifespan trigger
    test_app.state.video_detector = _StubVideoDetector()
    test_app.state.photo_detector = _StubPhotoDetector()
    test_app.state.text_detector = _StubTextDetector()
    test_app.state.models_loaded = {
        "video_detector": "stub",
        "photo_detector": "stub",
        "text_detector": "stub",
    }
    test_app.include_router(api_router, prefix=API_V1_PREFIX)
    return test_app


@pytest_asyncio.fixture
async def client(app_with_mocks):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks), base_url="http://test"
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


# ---------------------------------------------------------------------------
# Text endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_scam(client):
    r = await client.post(
        "/api/v1/analyze/text",
        json={"text": "您好，刑事局通知您帳戶涉嫌洗錢，請配合轉帳至監管帳戶。"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "scam"
    assert data["confidence"] > 0.5
    assert "closest_archetype" in data
    assert "explanation" in data


@pytest.mark.asyncio
async def test_text_empty_body_fails(client):
    r = await client.post("/api/v1/analyze/text", json={})
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_text_too_long_fails(client):
    r = await client.post(
        "/api/v1/analyze/text",
        json={"text": "A" * 10_001},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Photo endpoint
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(h: int = 64, w: int = 64) -> bytes:
    """Create a minimal JPEG image as bytes."""
    from PIL import Image
    img = Image.fromarray(
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8), "RGB"
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_photo_real(client):
    jpeg = _make_jpeg_bytes()
    r = await client.post(
        "/api/v1/analyze/photo",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("real", "fake", "uncertain")
    assert "geometry_score" in data
    assert "prnu_score" in data


@pytest.mark.asyncio
async def test_photo_wrong_extension(client):
    r = await client.post(
        "/api/v1/analyze/photo",
        files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 415


# ---------------------------------------------------------------------------
# Video endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_wrong_extension(client):
    r = await client.post(
        "/api/v1/analyze/video",
        files={"file": ("test.txt", b"not a video", "text/plain")},
    )
    assert r.status_code == 415


# ---------------------------------------------------------------------------
# Unified endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unified_text_only(client):
    r = await client.post(
        "/api/v1/analyze/unified",
        data={"text": "您好，刑事局通知您帳戶涉嫌洗錢，請配合轉帳至監管帳戶。"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "overall_status" in data
    assert data["text"] is not None
    assert data["video"] is None
    assert data["photo"] is None


@pytest.mark.asyncio
async def test_unified_photo_and_text(client):
    jpeg = _make_jpeg_bytes()
    r = await client.post(
        "/api/v1/analyze/unified",
        data={"text": "詐騙文字測試"},
        files={"photo": ("test.jpg", jpeg, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["photo"] is not None
    assert data["text"] is not None
    assert "overall_status" in data


@pytest.mark.asyncio
async def test_unified_empty_request(client):
    """Request with no modalities should still return 200 with uncertain status."""
    r = await client.post("/api/v1/analyze/unified")
    assert r.status_code == 200
    data = r.json()
    assert data["overall_status"] == "uncertain"

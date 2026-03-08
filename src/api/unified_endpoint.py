"""
POST /api/v1/analyze/unified

Accepts optional video, image, and text fields in a single multipart request
and returns a combined verdict.
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from PIL import Image

from api.schemas import (
    PhotoAnalysisResponse,
    RAGEvidenceItem,
    TextAnalysisResponse,
    UnifiedAnalysisResponse,
    VideoAnalysisResponse,
)
from config import ALLOWED_IMAGE_EXTS, ALLOWED_VIDEO_EXTS
from modules.photo.photo_detector import PhotoDetector
from modules.text.text_detector import TextDetector
from modules.video.video_detector import VideoDetector

router = APIRouter()


def get_video_detector(request: Request) -> VideoDetector:
    return request.app.state.video_detector


def get_photo_detector(request: Request) -> PhotoDetector:
    return request.app.state.photo_detector


def get_text_detector(request: Request) -> TextDetector:
    return request.app.state.text_detector


@router.post(
    "/unified",
    response_model=UnifiedAnalysisResponse,
    summary="Multi-modal deepfake + scam analysis in a single request",
)
async def analyze_unified(
    video: Optional[UploadFile] = File(None, description="Optional video file"),
    photo: Optional[UploadFile] = File(None, description="Optional image file"),
    text: Optional[str] = Form(None, description="Optional scam text to analyse"),
    video_detector: VideoDetector = Depends(get_video_detector),
    photo_detector: PhotoDetector = Depends(get_photo_detector),
    text_detector: TextDetector = Depends(get_text_detector),
) -> UnifiedAnalysisResponse:
    t0 = time.perf_counter()

    video_resp: Optional[VideoAnalysisResponse] = None
    photo_resp: Optional[PhotoAnalysisResponse] = None
    text_resp: Optional[TextAnalysisResponse] = None

    # --- Video ---
    if video is not None:
        suffix = Path(video.filename or "").suffix.lower()
        if suffix in ALLOWED_VIDEO_EXTS:
            content = await video.read()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            v = video_detector.analyze(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)
            video_resp = VideoAnalysisResponse(
                status=v.status,
                confidence=v.confidence,
                explanation=v.explanation,
                processing_time_ms=v.processing_time_ms,
                hr_bpm=v.hr_bpm,
                pearson_sync=v.pearson_sync,
                snr_db=v.snr_db,
            )

    # --- Photo ---
    if photo is not None:
        suffix = Path(photo.filename or "").suffix.lower()
        if suffix in ALLOWED_IMAGE_EXTS:
            content = await photo.read()
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            p = photo_detector.analyze_array(img_bgr)
            photo_resp = PhotoAnalysisResponse(
                status=p.status,
                confidence=p.confidence,
                explanation=p.explanation,
                processing_time_ms=p.processing_time_ms,
                geometry_score=p.geometry_score,
                prnu_score=p.prnu_score,
                has_periodic_artifacts=p.has_periodic_artifacts,
            )

    # --- Text ---
    if text:
        t = text_detector.analyze(text)
        text_resp = TextAnalysisResponse(
            status=t.status,
            confidence=t.confidence,
            explanation=t.explanation,
            processing_time_ms=t.processing_time_ms,
            closest_archetype=t.closest_archetype,
            closest_archetype_zh=t.closest_archetype_zh,
            intent_similarity=t.intent_similarity,
            rag_scam_ratio=t.rag_scam_ratio,
            rag_evidence=[
                RAGEvidenceItem(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    label=c.label,
                    archetype=c.archetype,
                )
                for c in t.rag_evidence
            ],
        )

    overall_status, overall_confidence, summary = _aggregate(
        video_resp, photo_resp, text_resp
    )

    return UnifiedAnalysisResponse(
        video=video_resp,
        photo=photo_resp,
        text=text_resp,
        overall_status=overall_status,
        overall_confidence=overall_confidence,
        summary=summary,
        processing_time_ms=(time.perf_counter() - t0) * 1000,
    )


def _aggregate(
    video: Optional[VideoAnalysisResponse],
    photo: Optional[PhotoAnalysisResponse],
    text: Optional[TextAnalysisResponse],
) -> tuple[str, float, str]:
    """Combine individual modality verdicts into a unified score."""
    results = []
    labels = []

    if video is not None:
        results.append(video.confidence if video.status == "fake" else 1.0 - video.confidence)
        labels.append(video.status)
    if photo is not None:
        results.append(photo.confidence if photo.status == "fake" else 1.0 - photo.confidence)
        labels.append(photo.status)
    if text is not None:
        results.append(text.confidence if text.status == "scam" else 1.0 - text.confidence)
        labels.append(text.status)

    if not results:
        return "uncertain", 0.0, "No modalities provided."

    fraud_count = sum(1 for l in labels if l in ("fake", "scam"))
    safe_count = sum(1 for l in labels if l in ("real", "safe"))
    uncertain_count = len(labels) - fraud_count - safe_count

    avg_fraud_prob = sum(results) / len(results)

    if fraud_count > safe_count:
        status = "fake/scam"
        confidence = avg_fraud_prob
    elif safe_count > fraud_count:
        status = "real/safe"
        confidence = 1.0 - avg_fraud_prob
    else:
        status = "uncertain"
        confidence = 0.5

    parts = []
    if video:
        parts.append(f"Video: {video.status} ({video.confidence:.0%})")
    if photo:
        parts.append(f"Photo: {photo.status} ({photo.confidence:.0%})")
    if text:
        parts.append(f"Text: {text.status} ({text.confidence:.0%})")

    summary = f"Overall: {status} | " + " | ".join(parts)
    return status, float(round(confidence, 4)), summary

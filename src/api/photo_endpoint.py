"""
POST /api/v1/analyze/photo
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from PIL import Image
import io
import cv2

from api.schemas import PhotoAnalysisResponse
from config import ALLOWED_IMAGE_EXTS, MAX_UPLOAD_SIZE_MB
from modules.photo.photo_detector import PhotoDetector

router = APIRouter()


def get_photo_detector(request: Request) -> PhotoDetector:
    return request.app.state.photo_detector


@router.post(
    "/photo",
    response_model=PhotoAnalysisResponse,
    summary="Detect AI-generated / manipulated image via lens geometry and PRNU",
)
async def analyze_photo(
    file: UploadFile = File(..., description="Image file (jpg, png, bmp, webp)"),
    detector: PhotoDetector = Depends(get_photo_detector),
) -> PhotoAnalysisResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format '{suffix}'. Allowed: {ALLOWED_IMAGE_EXTS}",
        )

    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Decode via PIL → numpy BGR (avoids filesystem write for images)
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img_rgb = np.array(pil_img, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    verdict = detector.analyze_array(img_bgr)

    return PhotoAnalysisResponse(
        status=verdict.status,
        confidence=verdict.confidence,
        explanation=verdict.explanation,
        processing_time_ms=verdict.processing_time_ms,
        geometry_score=verdict.geometry_score,
        prnu_score=verdict.prnu_score,
        has_periodic_artifacts=verdict.has_periodic_artifacts,
    )

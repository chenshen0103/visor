"""
POST /api/v1/analyze/video
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File

from api.schemas import VideoAnalysisResponse
from config import ALLOWED_VIDEO_EXTS, MAX_UPLOAD_SIZE_MB
from modules.video.video_detector import VideoDetector

router = APIRouter()


def get_video_detector(request: Request) -> VideoDetector:
    """Dependency — resolved from app state at startup."""
    return request.app.state.video_detector


@router.post(
    "/video",
    response_model=VideoAnalysisResponse,
    summary="Detect deepfake video via rPPG physiological signals",
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, mkv, webm)"),
    detector: VideoDetector = Depends(get_video_detector),
) -> VideoAnalysisResponse:
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format '{suffix}'. Allowed: {ALLOWED_VIDEO_EXTS}",
        )

    # Read and size-check
    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # Write to temp file (OpenCV requires a filesystem path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    verdict = detector.analyze(tmp_path)

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)

    return VideoAnalysisResponse(
        status=verdict.status,
        confidence=verdict.confidence,
        explanation=verdict.explanation,
        processing_time_ms=verdict.processing_time_ms,
        hr_bpm=verdict.hr_bpm,
        pearson_sync=verdict.pearson_sync,
        snr_db=verdict.snr_db,
    )

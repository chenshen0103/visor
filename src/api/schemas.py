"""
Pydantic request / response schemas for all API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class BaseAnalysisResponse(BaseModel):
    status: str = Field(..., description="Classification result label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0–1")
    explanation: str = Field(..., description="Human-readable reasoning")
    processing_time_ms: float = Field(..., description="Wall-clock processing time in ms")


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

class VideoAnalysisResponse(BaseAnalysisResponse):
    """
    status: "real" | "fake" | "face_swap" | "uncertain"
    """
    hr_bpm: float = Field(..., description="Estimated heart rate in BPM")
    pearson_sync: float = Field(
        ..., description="Within-face Pearson sync (forehead ↔ cheeks)"
    )
    face_neck_sync: float = Field(
        0.0, description="Cross-boundary Pearson sync (face ↔ neck); low on face-swaps"
    )
    snr_db: float = Field(..., description="rPPG signal-to-noise ratio in dB")


# ---------------------------------------------------------------------------
# Photo
# ---------------------------------------------------------------------------

class PhotoAnalysisResponse(BaseAnalysisResponse):
    """
    status: "real" | "fake" | "uncertain"
    """
    geometry_score: float = Field(..., description="Lens geometry authenticity score 0–1")
    prnu_score: float = Field(..., description="PRNU camera fingerprint score 0–1")
    has_periodic_artifacts: bool = Field(
        ..., description="Whether AI upsampling artifacts were detected"
    )


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Text to analyse")


class RAGEvidenceItem(BaseModel):
    chunk_id: str
    text: str
    source: str
    label: str
    archetype: str


class TextAnalysisResponse(BaseAnalysisResponse):
    """
    status: "scam" | "safe" | "suspicious"
    """
    closest_archetype: str = Field(..., description="Internal archetype key")
    closest_archetype_zh: str = Field(..., description="Archetype name in Chinese")
    intent_similarity: float = Field(
        ..., description="Cosine similarity to closest scam archetype"
    )
    rag_scam_ratio: float = Field(
        ..., description="Fraction of top-k RAG chunks matching scam patterns"
    )
    rag_evidence: List[RAGEvidenceItem] = Field(
        default_factory=list, description="Retrieved anti-fraud knowledge base chunks"
    )


# ---------------------------------------------------------------------------
# Unified
# ---------------------------------------------------------------------------

class UnifiedAnalysisResponse(BaseModel):
    video: Optional[VideoAnalysisResponse] = None
    photo: Optional[PhotoAnalysisResponse] = None
    text: Optional[TextAnalysisResponse] = None
    overall_status: str = Field(
        ...,
        description=(
            "Combined verdict: 'real'/'safe', 'fake'/'scam', "
            "'uncertain', or 'partial' when only some modalities were provided"
        ),
    )
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., description="Brief multi-modal summary")
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: Dict[str, Any] = Field(default_factory=dict)

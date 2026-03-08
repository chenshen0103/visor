"""
POST /api/v1/analyze/text
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.schemas import RAGEvidenceItem, TextAnalysisResponse, TextRequest
from modules.text.text_detector import TextDetector

router = APIRouter()


def get_text_detector(request: Request) -> TextDetector:
    return request.app.state.text_detector


@router.post(
    "/text",
    response_model=TextAnalysisResponse,
    summary="Detect scam / fraud text via intent embeddings and RAG fact-checking",
)
async def analyze_text(
    body: TextRequest,
    detector: TextDetector = Depends(get_text_detector),
) -> TextAnalysisResponse:
    verdict = detector.analyze(body.text)

    evidence_items = [
        RAGEvidenceItem(
            chunk_id=c.chunk_id,
            text=c.text,
            source=c.source,
            label=c.label,
            archetype=c.archetype,
        )
        for c in verdict.rag_evidence
    ]

    return TextAnalysisResponse(
        status=verdict.status,
        confidence=verdict.confidence,
        explanation=verdict.explanation,
        processing_time_ms=verdict.processing_time_ms,
        closest_archetype=verdict.closest_archetype,
        closest_archetype_zh=verdict.closest_archetype_zh,
        intent_similarity=verdict.intent_similarity,
        rag_scam_ratio=verdict.rag_scam_ratio,
        rag_evidence=evidence_items,
    )

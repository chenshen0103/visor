"""
TextDetector — orchestrates intent embedding + RAG retrieval for scam detection.

Score fusion
------------
- intent_score (weight 0.40): cosine similarity to nearest scam archetype
- rag_score    (weight 0.60): fraction of top-k RAG chunks matching known scams
→ weighted scam_probability → TextVerdict
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import (
    SCAM_SIMILARITY_HIGH,
    SCAM_SIMILARITY_MID,
    TEXT_INTENT_WEIGHT,
    TEXT_RAG_WEIGHT,
)
from modules.text.intent_embedder import IntentEmbedder, IntentResult
from modules.text.rag_retriever import Chunk, RAGResult, RAGRetriever

logger = logging.getLogger(__name__)


@dataclass
class TextVerdict:
    is_scam: Optional[bool]     # None = suspicious/uncertain
    confidence: float           # 0–1
    status: str                 # "scam" | "safe" | "suspicious"
    closest_archetype: str
    closest_archetype_zh: str
    intent_similarity: float
    rag_scam_ratio: float
    rag_evidence: List[Chunk]
    explanation: str
    processing_time_ms: float


class TextDetector:
    """
    Load once at startup.  Both sub-components are lazy but shared.
    """

    def __init__(self) -> None:
        self._embedder = IntentEmbedder()
        self._retriever = RAGRetriever()
        self._rag_available = False

    def load(self) -> None:
        """Load models and FAISS index.  Called at app lifespan startup."""
        self._embedder.load()
        self._retriever.set_embedder(self._embedder)
        self._rag_available = self._retriever.load()

    def analyze(self, text: str) -> TextVerdict:
        t0 = time.perf_counter()

        # 1. Intent branch
        intent: IntentResult = self._embedder.compute_scam_distances(text)

        # 2. RAG branch
        rag: RAGResult = self._retriever.fact_check(text)

        # 3. Score fusion
        # Intent score: normalise similarity from [0, 1] range
        intent_score = float(np.clip(intent.max_similarity, 0.0, 1.0))

        # RAG score: use scam_chunk_ratio; default 0.5 if RAG unavailable
        rag_score = rag.scam_chunk_ratio if self._rag_available else 0.5

        scam_prob = (
            TEXT_INTENT_WEIGHT * intent_score
            + TEXT_RAG_WEIGHT * rag_score
        )
        scam_prob = float(np.clip(scam_prob, 0.0, 1.0))

        status, is_scam, confidence = self._classify(scam_prob)
        explanation = self._build_explanation(
            status, scam_prob, intent, rag, self._rag_available
        )

        return TextVerdict(
            is_scam=is_scam,
            confidence=confidence,
            status=status,
            closest_archetype=intent.closest_archetype,
            closest_archetype_zh=intent.closest_name_zh,
            intent_similarity=intent.max_similarity,
            rag_scam_ratio=rag.scam_chunk_ratio,
            rag_evidence=rag.evidence,
            explanation=explanation,
            processing_time_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _classify(scam_prob: float):
        if scam_prob >= 0.60:
            return "scam", True, scam_prob
        elif scam_prob <= 0.30:
            return "safe", False, 1.0 - scam_prob
        else:
            return "suspicious", None, scam_prob

    @staticmethod
    def _build_explanation(
        status: str,
        scam_prob: float,
        intent: IntentResult,
        rag: RAGResult,
        rag_available: bool,
    ) -> str:
        parts = [
            f"Text classified as '{status}' (scam_prob={scam_prob:.2f}).",
            f"Closest scam archetype: '{intent.closest_name_zh}' "
            f"({intent.closest_name_en}) with similarity={intent.max_similarity:.3f}.",
        ]
        if rag_available:
            parts.append(
                f"RAG retrieval: {len(rag.evidence)} chunks retrieved, "
                f"{rag.scam_chunk_ratio:.0%} match known scam patterns."
            )
            if rag.contradicts_official:
                parts.append("Official anti-fraud warning matched in knowledge base.")
        else:
            parts.append("RAG knowledge base unavailable; intent-only scoring applied.")
        return " ".join(parts)

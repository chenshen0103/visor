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
from modules.text.red_flags import RedFlag, RedFlagAnalyzer
from modules.text.explainer import ScamExplainer

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
    red_flags: List[RedFlag]
    explanation: str
    llm_explanation: Optional[str] = None
    processing_time_ms: float = 0.0


class TextDetector:
    """
    Load once at startup.  Both sub-components are lazy but shared.
    """

    def __init__(self) -> None:
        self._embedder = IntentEmbedder()
        self._retriever = RAGRetriever()
        self._red_flag_analyzer = RedFlagAnalyzer()
        self._explainer = ScamExplainer()
        self._rag_available = False

    def load(self) -> None:
        """Load models and FAISS index.  Called at app lifespan startup."""
        self._embedder.load()
        self._retriever.set_embedder(self._embedder)
        self._rag_available = self._retriever.load()
        # Optional: Explainer can be heavy, could be loaded lazily but 
        # we follow the lifespan pattern.
        # self._explainer.load() 

    def analyze(self, text: str, history: List[str] = None) -> TextVerdict:
        t0 = time.perf_counter()

        # 1. Intent branch
        intent: IntentResult = self._embedder.compute_scam_distances(text)

        # 2. RAG branch
        rag: RAGResult = self._retriever.fact_check(text)

        # 3. Context-aware branch (Multi-turn)
        # If history exists, we create a 'context_text' to check if the 
        # conversation as a whole resembles a scam pattern.
        # We limit to last 3 messages to avoid noise.
        context_intent = None
        context_rag = None
        if history:
            recent_history = history[-3:]
            context_text = "\n".join(recent_history + [text])
            context_intent = self._embedder.compute_scam_distances(context_text)
            context_rag = self._retriever.fact_check(context_text)

        # 4. Red Flag (Heuristic) branch
        red_flags = self._red_flag_analyzer.analyze(text)
        
        # If RAG strongly matches an official warning (exact match), 
        # we skip the red flag boost to avoid false positives on education/warning text.
        red_flag_boost = 0.0
        is_official_warning_match = rag.confidence >= 0.95 and not rag.matches_known_scam
        
        if not is_official_warning_match:
            red_flag_boost = sum(rf.severity for rf in red_flags)

        # 5. Score fusion
        # Intent score: normalise similarity from [0, 1] range
        intent_score = float(np.clip(intent.max_similarity, 0.0, 1.0))

        # RAG score: use scam_chunk_ratio; default 0.5 if RAG unavailable
        rag_score = rag.scam_chunk_ratio if self._rag_available else 0.5

        # Base score for current message
        scam_prob = (
            TEXT_INTENT_WEIGHT * intent_score
            + TEXT_RAG_WEIGHT * rag_score
        )

        # If context is available, compute a context_scam_prob
        if context_intent and context_rag:
            ctx_intent_score = float(np.clip(context_intent.max_similarity, 0.0, 1.0))
            ctx_rag_score = context_rag.scam_chunk_ratio if self._rag_available else 0.5
            ctx_scam_prob = (
                TEXT_INTENT_WEIGHT * ctx_intent_score
                + TEXT_RAG_WEIGHT * ctx_rag_score
            )
            # Use max(current, context) to catch scams that are only obvious in context
            scam_prob = max(scam_prob, ctx_scam_prob)
            
            # If context matched a different archetype more strongly, use it
            if context_intent.max_similarity > intent.max_similarity:
                intent = context_intent

        # Apply heuristic boost (capping at 1.0)
        # This allows clear "red flags" (like phishing links) to push a message 
        # from "suspicious" to "scam" even if the semantic embedding isn't 100% certain.
        scam_prob = min(1.0, scam_prob + red_flag_boost)
        scam_prob = float(np.clip(scam_prob, 0.0, 1.0))

        status, is_scam, confidence = self._classify(scam_prob)
        explanation = self._build_explanation(
            status, scam_prob, intent, rag, self._rag_available, bool(history), red_flags
        )

        # 6. LLM Explainer (Optional, only for suspicious/scam)
        llm_explanation = None
        if status in ("scam", "suspicious") and not is_official_warning_match:
            try:
                # Lazy load explainer on first scam detection to save VRAM for rPPG if not needed
                self._explainer.load()
                llm_explanation = self._explainer.generate_explanation(
                    text=text,
                    status=status,
                    archetype_zh=intent.closest_name_zh,
                    evidence=rag.evidence,
                    red_flags=red_flags,
                    history=history
                )
            except Exception as e:
                logger.error("LLM explanation generation failed: %s", e)

        return TextVerdict(
            is_scam=is_scam,
            confidence=confidence,
            status=status,
            closest_archetype=intent.closest_archetype,
            closest_archetype_zh=intent.closest_name_zh,
            intent_similarity=intent.max_similarity,
            rag_scam_ratio=rag.scam_chunk_ratio,
            rag_evidence=rag.evidence,
            red_flags=red_flags,
            explanation=explanation,
            llm_explanation=llm_explanation,
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
        has_history: bool,
        red_flags: List[RedFlag],
    ) -> str:
        parts = [
            f"Text classified as '{status}' (scam_prob={scam_prob:.2f}).",
            f"Closest scam archetype: '{intent.closest_name_zh}' "
            f"({intent.closest_name_en}) with similarity={intent.max_similarity:.3f}.",
        ]
        if has_history:
            parts[0] = f"Text (with conversation context) classified as '{status}' (scam_prob={scam_prob:.2f})."

        if red_flags:
            flags_desc = ", ".join(f"[{rf.description}]" for rf in red_flags)
            parts.append(f"Heuristic red flags detected: {flags_desc}.")

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

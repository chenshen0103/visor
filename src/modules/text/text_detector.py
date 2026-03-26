"""
TextDetector — orchestrates intent embedding + RAG retrieval for scam detection.

Score fusion
------------
- intent_score (weight 0.40): cosine similarity to nearest scam archetype
- rag_score    (weight 0.60): fraction of top-k RAG chunks matching known scams
- red_flag_boost            : heuristic boost from URL / urgency / sensitive-action patterns
→ weighted scam_probability → TextVerdict
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

logger = logging.getLogger(__name__)


@dataclass
class TextVerdict:
    is_scam: Optional[bool]         # None = suspicious/uncertain
    confidence: float               # 0–1
    status: str                     # "scam" | "safe" | "suspicious"
    closest_archetype: str
    closest_archetype_zh: str
    intent_similarity: float
    all_similarities: Dict[str, float]   # archetype_key → cosine similarity
    rag_scam_ratio: float
    rag_evidence: List[Chunk]
    red_flags: List[RedFlag]
    explanation: str
    processing_time_ms: float = 0.0


class TextDetector:
    """
    Load once at startup.  Sub-components are lazy but shared.
    """

    def __init__(self) -> None:
        self._embedder = IntentEmbedder()
        self._retriever = RAGRetriever()
        self._red_flag_analyzer = RedFlagAnalyzer()
        self._rag_available = False

    def load(self) -> None:
        """Load models and FAISS index.  Called at app lifespan startup."""
        self._embedder.load()
        self._retriever.set_embedder(self._embedder)
        self._rag_available = self._retriever.load()

    def analyze(self, text: str, history: Optional[List[str]] = None) -> TextVerdict:
        t0 = time.perf_counter()

        # 1. Intent branch
        intent: IntentResult = self._embedder.compute_scam_distances(text)

        # 2. RAG branch
        rag: RAGResult = self._retriever.fact_check(text)

        # 3. Multi-turn context branch
        # If history provided, also score the conversation as a whole.
        context_intent = None
        context_rag = None
        if history:
            recent = history[-3:]
            context_text = "\n".join(recent + [text])
            context_intent = self._embedder.compute_scam_distances(context_text)
            context_rag = self._retriever.fact_check(context_text)

        # 4. Red-flag heuristic branch
        red_flags = self._red_flag_analyzer.analyze(text)

        # Skip red-flag boost when the query IS an official warning (avoid FP)
        is_official_match = rag.confidence >= 0.95 and not rag.matches_known_scam
        raw_red_flag_boost = 0.0 if is_official_match else sum(rf.severity for rf in red_flags)

        # 5. Score fusion
        intent_score = float(np.clip(intent.max_similarity, 0.0, 1.0))
        # RAG fallback = 0.0: no evidence should not contribute to scam probability
        raw_rag_score = rag.scam_chunk_ratio if self._rag_available else 0.0

        # Attenuate RAG score when intent similarity is weak.
        # Legitimate content (lawyers, educators) discussing fraud topics will
        # semantically match the RAG corpus without actually being scam text.
        # Intent must anchor the RAG signal — same logic as red_flag_boost.
        if intent_score >= SCAM_SIMILARITY_MID:
            rag_score = raw_rag_score
        else:
            rag_score = raw_rag_score * (intent_score / SCAM_SIMILARITY_MID)

        # Attenuate red-flag boost when intent similarity is weak.
        # Keywords like "轉帳" or "立即" appear legitimately in legal/educational content;
        # only amplify them when the text already resembles a known scam archetype.
        if intent_score >= SCAM_SIMILARITY_MID:
            red_flag_boost = raw_red_flag_boost
        else:
            # Scale boost proportionally: near-zero intent → near-zero boost
            red_flag_boost = raw_red_flag_boost * (intent_score / SCAM_SIMILARITY_MID)

        scam_prob = (
            TEXT_INTENT_WEIGHT * intent_score
            + TEXT_RAG_WEIGHT * rag_score
        )

        # Incorporate context score when available
        if context_intent and context_rag:
            ctx_intent_score = float(np.clip(context_intent.max_similarity, 0.0, 1.0))
            ctx_rag_score = context_rag.scam_chunk_ratio if self._rag_available else 0.0
            ctx_prob = TEXT_INTENT_WEIGHT * ctx_intent_score + TEXT_RAG_WEIGHT * ctx_rag_score
            scam_prob = max(scam_prob, ctx_prob)
            # If context matched a stronger archetype, surface it
            if context_intent.max_similarity > intent.max_similarity:
                intent = context_intent

        # Apply heuristic boost (capped at 1.0)
        scam_prob = float(np.clip(scam_prob + red_flag_boost, 0.0, 1.0))

        status, is_scam, confidence = self._classify(scam_prob)
        explanation = self._build_explanation(
            status, scam_prob, intent, rag, self._rag_available,
            has_history=bool(history), red_flags=red_flags,
        )

        return TextVerdict(
            is_scam=is_scam,
            confidence=confidence,
            status=status,
            closest_archetype=intent.closest_archetype,
            closest_archetype_zh=intent.closest_name_zh,
            intent_similarity=intent.max_similarity,
            all_similarities=intent.all_similarities,
            rag_scam_ratio=rag.scam_chunk_ratio,
            rag_evidence=rag.evidence,
            red_flags=red_flags,
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
        has_history: bool = False,
        red_flags: Optional[List[RedFlag]] = None,
    ) -> str:
        ctx_note = "（含對話歷史）" if has_history else ""
        parts = [
            f"文字{ctx_note}判定為 '{status}'（詐騙概率={scam_prob:.2f}）。",
            f"最近詐騙意圖：'{intent.closest_name_zh}'（{intent.closest_name_en}），"
            f"相似度={intent.max_similarity:.3f}。",
        ]
        if red_flags:
            flags_desc = "、".join(f"[{rf.description}]" for rf in red_flags)
            parts.append(f"啟發式紅旗：{flags_desc}。")
        if rag_available:
            parts.append(
                f"RAG 知識庫：取回 {len(rag.evidence)} 筆，"
                f"{rag.scam_chunk_ratio:.0%} 符合已知詐騙模式。"
            )
            if rag.contradicts_official:
                parts.append("命中官方反詐騙警示。")
        else:
            parts.append("RAG 知識庫不可用，僅以意圖相似度評分。")
        return " ".join(parts)

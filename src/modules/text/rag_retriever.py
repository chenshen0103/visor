"""
RAGRetriever — FAISS-backed retrieval over the 165-chunk anti-fraud knowledge base.

Build the index once with:
    python src/training/build_rag_index.py

Then at runtime:
    retriever = RAGRetriever()
    retriever.load()
    result = retriever.fact_check("您帳戶涉嫌洗錢，請配合轉帳")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import FAISS_INDEX_PATH, FAISS_META_PATH, RAG_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    label: str          # "scam_example" | "official_warning" | "safe_example"
    archetype: str      # optional scam archetype key


@dataclass
class RAGResult:
    matches_known_scam: bool
    contradicts_official: bool
    evidence: List[Chunk]
    scam_chunk_ratio: float     # fraction of top-k chunks labelled scam
    confidence: float           # 0–1


class RAGRetriever:
    """
    FAISS nearest-neighbour retrieval over pre-embedded anti-fraud documents.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH,
    ) -> None:
        self._index_path = Path(index_path)
        self._meta_path = Path(meta_path)
        self._index = None
        self._chunks: List[Chunk] = []
        self._embedder = None   # set via set_embedder()

    def set_embedder(self, embedder) -> None:
        """Inject the shared IntentEmbedder (avoids loading a second model)."""
        self._embedder = embedder

    def load(self) -> bool:
        """
        Load FAISS index and chunk metadata.

        Returns True on success, False if index files are missing (graceful
        degradation: RAG branch returns neutral scores).
        """
        if not self._index_path.exists() or not self._meta_path.exists():
            logger.warning(
                "FAISS index not found at %s. "
                "RAG branch will return neutral scores. "
                "Run training/build_rag_index.py to build it.",
                self._index_path,
            )
            return False

        try:
            import faiss
            self._index = faiss.read_index(str(self._index_path))
            _fields = {"chunk_id", "text", "source", "label", "archetype"}
            with open(self._meta_path, encoding="utf-8") as f:
                self._chunks = [
                    Chunk(**{k: v for k, v in json.loads(line).items() if k in _fields})
                    for line in f if line.strip()
                ]
            logger.info(
                "RAGRetriever loaded: %d chunks, %d vectors",
                len(self._chunks),
                self._index.ntotal,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Chunk]:
        """
        Return the *top_k* most similar chunks to *query*.

        Falls back to empty list if index is unavailable.
        """
        if self._index is None or self._embedder is None:
            return []

        vec = self._embedder.embed(query).reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        _, indices = self._index.search(vec, k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks):
                results.append(self._chunks[idx])
        return results

    def fact_check(self, message: str) -> RAGResult:
        """
        Check *message* against the knowledge base.

        Returns
        -------
        RAGResult with:
        - matches_known_scam  : True if ≥ 60% of top-k chunks are scam examples
        - contradicts_official: True if any retrieved chunk is an official warning
          and contradicts typical safe patterns
        - evidence            : retrieved chunks
        - confidence          : scam_chunk_ratio
        """
        if self._index is None:
            return RAGResult(
                matches_known_scam=False,
                contradicts_official=False,
                evidence=[],
                scam_chunk_ratio=0.0,
                confidence=0.0,
            )

        chunks = self.retrieve(message)

        if not chunks:
            return RAGResult(
                matches_known_scam=False,
                contradicts_official=False,
                evidence=[],
                scam_chunk_ratio=0.0,
                confidence=0.0,
            )

        # Only direct scam examples count as positive evidence.
        # official_warning chunks describe scam *patterns* for education — they
        # raise the contradicts_official flag but do NOT inflate scam_ratio,
        # which prevents safe text from being falsely classified as scam.
        # safe_example chunks actively lower the effective ratio.
        scam_count = sum(1 for c in chunks if c.label == "scam_example")
        safe_count = sum(1 for c in chunks if c.label == "safe_example")
        scam_ratio = max(0.0, (scam_count - safe_count)) / len(chunks)

        has_official_warning = any(c.label == "official_warning" for c in chunks)

        return RAGResult(
            matches_known_scam=scam_ratio >= 0.6,
            contradicts_official=has_official_warning,
            evidence=chunks,
            scam_chunk_ratio=scam_ratio,
            confidence=scam_ratio,
        )

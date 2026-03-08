"""
IntentEmbedder — embeds text queries and computes cosine similarity to scam archetypes.

Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, multilingual)
Archetype centroids are computed once at initialisation by averaging the
exemplar embeddings for each scam type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_DIM, SCAM_SIMILARITY_HIGH, SCAM_SIMILARITY_MID, SENTENCE_MODEL_NAME
from modules.text.scam_patterns import SCAM_ARCHETYPES, ScamArchetype

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    closest_archetype: str      # archetype key
    closest_name_zh: str
    closest_name_en: str
    max_similarity: float       # cosine similarity to closest archetype
    all_similarities: Dict[str, float]  # key → similarity
    is_high_risk: bool          # similarity ≥ SCAM_SIMILARITY_HIGH
    is_suspicious: bool         # similarity ≥ SCAM_SIMILARITY_MID


class IntentEmbedder:
    """
    Lazy-initialises SentenceTransformer on first use or explicit load.
    Thread-safe after initialisation.
    """

    def __init__(self, model_name: str = SENTENCE_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._centroids: Dict[str, np.ndarray] = {}  # key → (384,)

    def load(self) -> None:
        """Explicitly load model and build archetype centroids."""
        if self._model is not None:
            return
        logger.info("Loading SentenceTransformer: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._build_centroids()
        logger.info("IntentEmbedder ready, %d archetypes loaded.", len(self._centroids))

    def _build_centroids(self) -> None:
        """Compute mean embedding for each scam archetype's exemplars."""
        for archetype in SCAM_ARCHETYPES:
            embeddings = self._model.encode(
                archetype.exemplars,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )  # (n_exemplars, 384)
            centroid = embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            self._centroids[archetype.key] = centroid.astype(np.float32)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def embed(self, text: str) -> np.ndarray:
        """Return L2-normalised 384-dim embedding for *text*."""
        self._ensure_loaded()
        vec = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vec.astype(np.float32)

    def compute_scam_distances(self, text: str) -> IntentResult:
        """
        Embed *text* and compute cosine similarity to all archetypes.

        Parameters
        ----------
        text : str
            The message to analyse.

        Returns
        -------
        IntentResult
        """
        self._ensure_loaded()
        query_vec = self.embed(text)  # (384,) normalised

        similarities: Dict[str, float] = {}
        for key, centroid in self._centroids.items():
            sim = float(np.dot(query_vec, centroid))  # cosine (both normalised)
            similarities[key] = sim

        closest_key = max(similarities, key=similarities.get)
        max_sim = similarities[closest_key]

        from modules.text.scam_patterns import ARCHETYPE_BY_KEY
        archetype = ARCHETYPE_BY_KEY[closest_key]

        return IntentResult(
            closest_archetype=closest_key,
            closest_name_zh=archetype.name_zh,
            closest_name_en=archetype.name_en,
            max_similarity=max_sim,
            all_similarities=similarities,
            is_high_risk=max_sim >= SCAM_SIMILARITY_HIGH,
            is_suspicious=max_sim >= SCAM_SIMILARITY_MID,
        )

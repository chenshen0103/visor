"""
One-time script: reads 165_chunks.jsonl → builds FAISS index + meta file.

Usage
-----
python src/training/build_rag_index.py \
    --chunks src/data/scam_corpus/165_chunks.jsonl \
    --output-index src/data/scam_corpus/faiss.index \
    --output-meta  src/data/scam_corpus/faiss_meta.jsonl

JSONL format (one JSON object per line):
{
    "chunk_id": "c001",
    "text": "...",
    "source": "內政部警政署",
    "label": "official_warning",   // "scam_example" | "official_warning" | "safe_example"
    "archetype": "government_impersonation"
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    CHUNKS_JSONL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    SENTENCE_MODEL_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(
    chunks_path: Path,
    index_path: Path,
    meta_path: Path,
    model_name: str = SENTENCE_MODEL_NAME,
    batch_size: int = 64,
) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    # --- Load chunks ---
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        logger.error("Chunks file not found: %s", chunks_path)
        sys.exit(1)

    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    if not chunks:
        logger.error("No chunks found in %s", chunks_path)
        sys.exit(1)

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # --- Embed ---
    logger.info("Loading SentenceTransformer: %s", model_name)
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d texts …", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # (N, 384)

    logger.info("Embeddings shape: %s", embeddings.shape)

    # --- Build FAISS index (Inner Product = cosine for unit vectors) ---
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors", index.ntotal)

    # --- Save ---
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s", index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Metadata saved to %s", meta_path)

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for anti-fraud RAG")
    parser.add_argument(
        "--chunks", default=str(CHUNKS_JSONL), help="Input JSONL chunks file"
    )
    parser.add_argument(
        "--output-index", default=str(FAISS_INDEX_PATH), help="Output FAISS index path"
    )
    parser.add_argument(
        "--output-meta", default=str(FAISS_META_PATH), help="Output meta JSONL path"
    )
    parser.add_argument("--model", default=SENTENCE_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_index(
        chunks_path=Path(args.chunks),
        index_path=Path(args.output_index),
        meta_path=Path(args.output_meta),
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

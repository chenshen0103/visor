"""
Multi-Modal Anti-Fraud Defense Framework
FastAPI application entrypoint.

Usage
-----
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is on sys.path when running as `uvicorn src.main:app`
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_V1_PREFIX, LOG_LEVEL
from api.router import api_router

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load all models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load detector models on startup; release on shutdown."""
    logger.info("=== Starting up Anti-Fraud Framework ===")

    # Video detector
    from modules.video.video_detector import VideoDetector
    application.state.video_detector = VideoDetector(device="cpu")

    # Photo detector
    from modules.photo.photo_detector import PhotoDetector
    application.state.photo_detector = PhotoDetector()

    # Text detector (loads sentence-transformer + FAISS)
    from modules.text.text_detector import TextDetector
    text_detector = TextDetector()
    text_detector.load()
    application.state.text_detector = text_detector

    application.state.models_loaded = {
        "video_detector": "PhysFormerLite (rPPG)",
        "photo_detector": "LensGeometry + PRNU",
        "text_detector": "MiniLM-L12 + FAISS RAG",
    }

    logger.info("=== All models loaded — server ready ===")
    yield
    logger.info("=== Shutting down Anti-Fraud Framework ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Modal Anti-Fraud Defense Framework",
    description=(
        "Detects deepfake videos/photos and scam text using physics-based and "
        "physiological signals rather than CNN appearance classifiers.\n\n"
        "Competition project — Lab605, Tatung University."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS (open for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=API_V1_PREFIX)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Multi-Modal Anti-Fraud Defense Framework",
        "docs": "/docs",
        "health": f"{API_V1_PREFIX}/health",
    }

"""
Central API router — registers all sub-routers under /analyze.
"""

from fastapi import APIRouter

from api import video_endpoint, photo_endpoint, text_endpoint, unified_endpoint
from api.schemas import HealthResponse

api_router = APIRouter()

# Modality sub-routers
api_router.include_router(
    video_endpoint.router,
    prefix="/analyze",
    tags=["Video Analysis"],
)
api_router.include_router(
    photo_endpoint.router,
    prefix="/analyze",
    tags=["Photo Analysis"],
)
api_router.include_router(
    text_endpoint.router,
    prefix="/analyze",
    tags=["Text Analysis"],
)
api_router.include_router(
    unified_endpoint.router,
    prefix="/analyze",
    tags=["Unified Analysis"],
)


@api_router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    from main import app
    return HealthResponse(
        status="ok",
        models_loaded=getattr(app.state, "models_loaded", {}),
    )

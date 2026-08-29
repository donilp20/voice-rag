"""
Voice RAG — FastAPI application entrypoint.

Boots the app, validates config via Pydantic Settings (fail-fast on
missing/invalid env vars), and mounts routes.
"""

import logging

from fastapi import FastAPI

from app.config import get_settings
from app.routes.websocket import router as ws_router

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("voice_rag")

app = FastAPI(
    title="Indic Voice-to-Voice RAG",
    description="Low-cost, low-latency cross-lingual voice RAG platform.",
    version="0.1.0",
)

app.include_router(ws_router)


@app.get("/health")
async def health() -> dict:
    """Basic liveness probe for the deployment platform (Fly.io/Railway)."""
    return {
        "status": "ok",
        "env": settings.app_env,
        "collection": settings.qdrant_collection_name,
    }


@app.on_event("startup")
async def on_startup() -> None:
    logger.info(
        "Voice RAG starting | env=%s | qdrant_collection=%s | abstain_threshold=%s",
        settings.app_env,
        settings.qdrant_collection_name,
        settings.hybrid_abstain_threshold,
    )
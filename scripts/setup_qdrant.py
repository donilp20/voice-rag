"""
Step 3 setup script: creates the Qdrant collection with the hybrid
dense+sparse schema (idempotent — safe to re-run).

Usage:
    python -m scripts.setup_qdrant
    python -m scripts.setup_qdrant --recreate   # drops and rebuilds (dev only)
"""

import argparse
import logging

from app.config import get_settings
from app.core.qdrant_client import create_collection, get_collection_info, get_qdrant_client

logging.basicConfig(level="INFO")
logger = logging.getLogger("voice_rag.setup_qdrant")

settings = get_settings()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Voice RAG Qdrant collection.")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection (destructive).")
    args = parser.parse_args()

    client = get_qdrant_client()
    create_collection(client, recreate=args.recreate)

    info = get_collection_info(client)
    logger.info("Collection status: %s | points: %s", info.status, info.points_count)


if __name__ == "__main__":
    main()
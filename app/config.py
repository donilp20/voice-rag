"""
Centralized, typed configuration for the Voice RAG system.
All values are loaded from environment variables (.env in development,
real env vars in production). Pydantic Settings validates types and
required fields at application boot — fail-fast on misconfiguration.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Sarvam AI (STT) ──────────────────────────────
    sarvam_api_key: str
    sarvam_stt_model: str = "saaras:v3"
    sarvam_ws_url: str = "wss://api.sarvam.ai/speech-to-text/ws"

    # ── Groq (Generation + LLM-as-Judge) ─────────────
    groq_api_key: str
    groq_generation_model: str = "llama-3.1-8b-instant"
    groq_judge_model: str = "llama-3.1-8b-instant"

    # ── Qdrant Cloud ──────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "voice_rag_msmarco_xi"
    qdrant_use_grpc: bool = True

    # ── Embedding Model ───────────────────────────────
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_onnx_path: str = "./data/models/bge-m3-int8.onnx"
    embedding_dim: int = 1024

    # ── Guardrail Thresholds ──────────────────────────
    hybrid_abstain_threshold: float = 0.55
    safety_regex_enabled: bool = True

    # ── Data Ingestion (Step 2) ───────────────────────
    # Raw MS MARCO-XI parquet files, one per language.
    # Replace with real paths once your download completes.
    raw_data_paths: dict[str, str] = {
        "hi": "./data/raw/msmarco_xi_hi.parquet",
        "gu": "./data/raw/msmarco_xi_gu.parquet",
    }
    processed_data_dir: str = "./data/processed"
    target_child_chunk_count: int = 100_000  # V1 curated subset per PRD
    parent_chunk_max_words: int = 150
    child_chunk_sentence_span: int = 2  # 1-2 sentences per child chunk
    embedding_batch_size: int = 32

    # ── Latency Budgets (ms) ──────────────────────────
    sla_retrieval_core_ms: int = 200
    sla_ttft_ms: int = 500


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor — import this, not Settings() directly."""
    return Settings()
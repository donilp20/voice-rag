"""
MS MARCO-XI Ingestion (Step 2, architecture doc Layer 2).

Loads raw ai4bharat/MSMARCO-XI parquet files (one per Indic language, per
V1 scope: hi, gu), extracts the English_passages field from each row
(per the "Zero-Translation Search" design — we index English passages
only, never Translated_passages), dedupes shared passages across
queries, and drives them through chunking.py to produce typed
ParentChunk / ChildChunk sets ready for embedding (Step 2 continued)
and Qdrant upsert (Step 3).

Expected raw schema per row (ai4bharat/MSMARCO-XI):
    query_id            : int | str
    target_lang         : str   (e.g. "hin_Deva", "guj_Gujr")
    passages.is_selected      : list[int]  (0/1 flags)
    passages.English_passages : list[str]
"""

import logging
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.core.chunking import chunk_passage
from app.models.ingestion_schemas import ChildChunk, IngestionStats, ParentChunk

logger = logging.getLogger("voice_rag.ingestion")
settings = get_settings()

# ai4bharat/MSMARCO-XI uses full language-script codes (e.g. "hin_Deva").
# We normalize to the short codes used everywhere else in this system
# (config, Qdrant payloads, FSM schemas) — "hi", "gu".
_LANG_CODE_MAP = {
    "hin_Deva": "hi",
    "guj_Gujr": "gu",
}


def _normalize_lang(raw_lang: str, fallback: str) -> str:
    """Map dataset's full lang-script code to our short code; fall back to the config key if unknown."""
    return _LANG_CODE_MAP.get(raw_lang, fallback)


def load_raw_language_file(lang: str) -> pd.DataFrame:
    """
    Load one language's raw MS MARCO-XI parquet file.

    Path comes from settings.raw_data_paths[lang] — currently placeholder
    paths (./data/raw/msmarco_xi_<lang>.parquet). Swap in real paths once
    your download completes; nothing else in this module needs to change.
    """
    if lang not in settings.raw_data_paths:
        raise ValueError(
            f"No raw_data_paths entry configured for language '{lang}'. "
            f"Configured languages: {list(settings.raw_data_paths.keys())}"
        )

    path = Path(settings.raw_data_paths[lang])
    if not path.exists():
        raise FileNotFoundError(
            f"Raw MS MARCO-XI file not found for '{lang}' at {path}. "
            "Update raw_data_paths in .env / config.py once your download is in place."
        )

    logger.info("Loading raw MS MARCO-XI parquet for lang=%s from %s", lang, path)
    return pd.read_parquet(path)


def extract_parent_chunks(df: pd.DataFrame, lang: str) -> dict[str, ParentChunk]:
    """
    Walk every row's English_passages list, dedupe by parent_id (identical
    passage text -> one ParentChunk regardless of how many queries share it).

    Returns a dict keyed by parent_id so downstream code can look up
    parents by id without a second pass over the raw data.
    """
    parents: dict[str, ParentChunk] = {}

    for row in df.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        raw_lang = getattr(row, "target_lang", lang)
        source_lang = _normalize_lang(raw_lang, fallback=lang)

        passages = getattr(row, "passages")
        english_passages: list[str] = passages.get("English_passages", [])
        is_selected_flags: list[int] = passages.get("is_selected", [])

        for i, passage_text in enumerate(english_passages):
            if not passage_text or not passage_text.strip():
                continue

            is_selected = bool(is_selected_flags[i]) if i < len(is_selected_flags) else False

            parent, _ = chunk_passage(
                passage_text=passage_text,
                query_id=query_id,
                source_lang=source_lang,
                is_selected=is_selected,
            )

            # Dedup: keep the first-seen version, but prefer is_selected=True
            # if the same passage shows up as selected under a different query.
            existing = parents.get(parent.parent_id)
            if existing is None:
                parents[parent.parent_id] = parent
            elif is_selected and not existing.is_selected:
                parents[parent.parent_id] = parent

    return parents


def build_child_chunks_for_parents(parents: dict[str, ParentChunk]) -> list[ChildChunk]:
    """Re-chunk every deduped parent into its child sentence-span chunks."""
    from app.core.chunking import build_child_chunks

    children: list[ChildChunk] = []
    for parent in parents.values():
        children.extend(build_child_chunks(parent))
    return children


def ingest_languages(languages: list[str] | None = None) -> tuple[list[ParentChunk], list[ChildChunk], IngestionStats]:
    """
    Full Step 2 ingestion entrypoint: load raw files for the given
    languages (defaults to everything in settings.raw_data_paths, i.e.
    hi + gu per V1 scope), dedupe passages, and chunk them.

    Caps output at settings.target_child_chunk_count (100k per PRD V1
    phasing) by truncating the child chunk list — parents are kept
    intact so no passage is half-chunked.
    """
    languages = languages or list(settings.raw_data_paths.keys())

    all_parents: dict[str, ParentChunk] = {}
    raw_examples_seen = 0

    for lang in languages:
        df = load_raw_language_file(lang)
        raw_examples_seen += len(df)
        lang_parents = extract_parent_chunks(df, lang)
        logger.info("lang=%s: %d raw examples -> %d deduped parent passages", lang, len(df), len(lang_parents))
        all_parents.update(lang_parents)  # parent_id is content-hash based, safe to merge across langs

    all_children = build_child_chunks_for_parents(all_parents)

    if len(all_children) > settings.target_child_chunk_count:
        logger.info(
            "Truncating child chunks from %d to target_child_chunk_count=%d",
            len(all_children),
            settings.target_child_chunk_count,
        )
        all_children = all_children[: settings.target_child_chunk_count]

    stats = IngestionStats(
        languages_processed=languages,
        raw_examples_seen=raw_examples_seen,
        parent_chunks_created=len(all_parents),
        child_chunks_created=len(all_children),
        child_chunks_embedded=0,  # filled in after Step 2's embedding stage
    )

    return list(all_parents.values()), all_children, stats
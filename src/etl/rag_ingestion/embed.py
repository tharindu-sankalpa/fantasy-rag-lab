"""
Stage 2 — Embedding generation and Parquet cache creation.

Reads raw chunks from MongoDB, generates embeddings using Gemini Embedding 2 Preview,
and persists them incrementally to a staging directory of numbered Parquet batch files.

Why batch files instead of a single appended Parquet?
  Parquet files are immutable — you cannot safely append to one. A crash mid-write
  corrupts the file. Writing each batch to its own file is atomic: either the file
  exists and is complete, or it does not exist. A checkpoint JSON tracks which
  chunk_ids have been embedded so restarts skip already-completed work.

On completion, all batch files are merged into one final Parquet.

Staging layout:
    cache/embeddings/{series}/
        batches/
            checkpoint.json          <- set of embedded chunk_ids
            batch_000000.parquet     <- one file per batch
            batch_000001.parquet
            ...
        {series}_embeddings.parquet  <- final merged output

Usage:
    # Generate embeddings (resumable — safe to Ctrl+C and restart)
    uv run python -m src.etl.rag_ingestion.embed \\
        --universe "Wheel of Time" --output-dir cache/embeddings

    # Only merge existing batch files without new API calls
    uv run python -m src.etl.rag_ingestion.embed \\
        --universe "Wheel of Time" --output-dir cache/embeddings --merge-only
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

from src.services.llm import UnifiedLLMService
from src.services.mongodb_service import MongoDBService

logger = structlog.get_logger(__name__)

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIM = 3072
DEFAULT_BATCH_SIZE = 20
MAX_RETRIES = 5
BASE_BACKOFF = 2.0  # seconds, doubles on each retry
CALL_DELAY = 0.05   # seconds between individual API calls (~20/sec, well under the 50/sec Tier 1 limit)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(staging_dir: Path) -> Path:
    return staging_dir / "checkpoint.json"


def _load_checkpoint(staging_dir: Path) -> set[str]:
    """Return the set of chunk_ids that have already been successfully embedded."""
    path = _checkpoint_path(staging_dir)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("embedded_ids", []))


def _save_checkpoint(staging_dir: Path, embedded_ids: set[str]) -> None:
    _checkpoint_path(staging_dir).write_text(
        json.dumps({"embedded_ids": sorted(embedded_ids)}, indent=2)
    )


# ---------------------------------------------------------------------------
# Embedding with retry / exponential backoff
# ---------------------------------------------------------------------------

async def _embed_one(
    text: str,
    chunk_id: str,
    llm_service: UnifiedLLMService,
    model: str,
    embedding_provider: str,
    log,
) -> list[float]:
    """Embed a single text with exponential backoff retry.

    Distinguishes rate-limit errors (longer backoff) from transient errors (shorter).
    Raises on the final attempt so callers can decide whether to skip or abort.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await llm_service.generate_embeddings(
                texts=text, model=model, provider=embedding_provider
            )
            embedding = result["embeddings"]
            # generate_embeddings returns a flat list when given a single string
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            return embedding
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(
                kw in err_str for kw in ("429", "resource exhausted", "quota exceeded")
            )
            is_transient = any(
                kw in err_str for kw in ("503", "unavailable", "timeout", "deadline")
            )

            if attempt == MAX_RETRIES:
                log.error(
                    "embed_failed_all_retries",
                    chunk_id=chunk_id,
                    attempts=attempt,
                    error=str(e),
                )
                raise

            backoff = BASE_BACKOFF * (2 ** (attempt - 1))
            if is_rate_limit:
                backoff = max(backoff, 60.0)
                log.warning(
                    "rate_limit_backing_off",
                    chunk_id=chunk_id,
                    attempt=attempt,
                    backoff_seconds=backoff,
                )
            else:
                if is_transient:
                    log.warning(
                        "transient_error_retrying",
                        chunk_id=chunk_id,
                        attempt=attempt,
                        backoff_seconds=backoff,
                        error=str(e),
                    )
                else:
                    log.warning(
                        "unknown_error_retrying",
                        chunk_id=chunk_id,
                        attempt=attempt,
                        backoff_seconds=backoff,
                        error=str(e),
                    )
            await asyncio.sleep(backoff)

    raise RuntimeError(f"Unreachable: embed loop exited without returning for {chunk_id}")


# ---------------------------------------------------------------------------
# Parquet batch write + merge
# ---------------------------------------------------------------------------

def _write_batch_parquet(
    staging_dir: Path, records: list[dict], batch_num: int
) -> Path:
    """Write a list of embedding records to a single numbered Parquet file."""
    path = staging_dir / f"batch_{batch_num:06d}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    return path


def merge_parquet_files(staging_dir: Path, output_path: Path, log) -> None:
    """Merge all batch_*.parquet files in staging_dir into one final Parquet."""
    batch_files = sorted(staging_dir.glob("batch_*.parquet"))
    if not batch_files:
        log.warning("no_batch_files_found", staging_dir=str(staging_dir))
        return

    log.info("merging_batch_files", count=len(batch_files), output=str(output_path))
    merged = pd.concat(
        [pd.read_parquet(f) for f in batch_files], ignore_index=True
    )
    merged.to_parquet(output_path, index=False)
    log.info("merge_complete", total_rows=len(merged), output=str(output_path))


# ---------------------------------------------------------------------------
# Main Stage 2 function
# ---------------------------------------------------------------------------

async def embed_to_parquet(
    universe: str,
    output_dir: str,
    model: str = EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    merge_only: bool = False,
    embedding_provider: str = "google",
) -> None:
    """Read chunks from MongoDB, generate embeddings, save to Parquet.

    Supports full resume: restarting the process continues from where it left off.

    Args:
        universe: Fantasy universe name — must match what was used in Stage 1.
        output_dir: Root directory for embedding cache output.
        model: Embedding model ID. Use "gemini-embedding-2-preview" for the
               google provider and "gemini/gemini-embedding-2-preview" for litellm.
        batch_size: Number of chunks per batch (one Parquet file per batch).
        merge_only: If True, skip embedding and only merge existing batch files.
        embedding_provider: Backend to use — "google" (default) or "litellm".
    """
    series = universe.lower().replace(" ", "_")
    root = Path(output_dir) / series
    staging_dir = root / "batches"
    staging_dir.mkdir(parents=True, exist_ok=True)
    final_path = root / f"{series}_embeddings.parquet"

    log = logger.bind(universe=universe, series=series, model=model, provider=embedding_provider)

    if merge_only:
        merge_parquet_files(staging_dir, final_path, log)
        return

    # --- Fetch chunks from MongoDB ---
    mongodb = MongoDBService()
    await mongodb.connect()
    try:
        chunks = await mongodb.get_rag_chunks_by_series(series)
        log.info("chunks_fetched", total=len(chunks))
    finally:
        await mongodb.disconnect()

    if not chunks:
        log.warning("no_chunks_found_in_mongodb", series=series)
        return

    # --- Resume: determine what still needs embedding ---
    embedded_ids = _load_checkpoint(staging_dir)
    pending = [c for c in chunks if c["chunk_id"] not in embedded_ids]
    log.info(
        "resume_status",
        total=len(chunks),
        already_done=len(embedded_ids),
        remaining=len(pending),
    )

    if not pending:
        log.info("all_chunks_embedded_merging")
        merge_parquet_files(staging_dir, final_path, log)
        return

    # --- Generate embeddings in batches ---
    llm_service = UnifiedLLMService()
    existing_batch_count = len(sorted(staging_dir.glob("batch_*.parquet")))
    total_batches = (len(pending) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(pending), batch_size):
        batch = pending[batch_idx: batch_idx + batch_size]
        batch_num = existing_batch_count + (batch_idx // batch_size)
        batch_log = log.bind(
            batch=batch_num,
            total_batches=total_batches,
            chunks_in_batch=len(batch),
        )
        batch_log.info("batch_started")

        records: list[dict] = []

        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            text = chunk["text_content"]
            meta = chunk.get("metadata", {})

            try:
                embedding = await _embed_one(text, chunk_id, llm_service, model, embedding_provider, log)
                await asyncio.sleep(CALL_DELAY)

                records.append(
                    {
                        "chunk_id": chunk_id,
                        "series": chunk.get("series", series),
                        "text_content": text,
                        "character_count": chunk.get("character_count", len(text)),
                        "book_name": meta.get("book_name", ""),
                        "chapter_number": meta.get("chapter_number", ""),
                        "chapter_title": meta.get("chapter_title", ""),
                        "universe": meta.get("universe", universe),
                        "embedding": embedding,
                        "embedding_model": model,
                        "embedding_dim": len(embedding),
                        "embedded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                embedded_ids.add(chunk_id)

            except Exception as e:
                # Log and skip — chunk will be retried on the next run
                batch_log.error(
                    "chunk_skipped_after_all_retries",
                    chunk_id=chunk_id,
                    error=str(e),
                )

        if records:
            parquet_path = _write_batch_parquet(staging_dir, records, batch_num)
            _save_checkpoint(staging_dir, embedded_ids)
            batch_log.info(
                "batch_complete",
                file=parquet_path.name,
                written=len(records),
                total_embedded=len(embedded_ids),
                skipped=len(batch) - len(records),
            )
        else:
            batch_log.warning("batch_produced_no_records_all_chunks_failed")

    # --- Merge into final file ---
    log.info("embedding_generation_complete", total_embedded=len(embedded_ids))
    merge_parquet_files(staging_dir, final_path, log)
    log.info("stage2_complete", output=str(final_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for Stage 2: embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate Gemini embeddings for RAG chunks and save to Parquet.",
        epilog=f"""
Examples:
  # Google direct (default)
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Wheel of Time" --output-dir cache/embeddings

  # LiteLLM proxy — note the gemini/ prefix in the model name
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Wheel of Time" --output-dir cache/embeddings \\
      --embedding-provider litellm \\
      --model "gemini/gemini-embedding-2-preview"

  # A Song of Ice and Fire
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Song of Ice and Fire" --output-dir cache/embeddings

  # Harry Potter
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Harry Potter" --output-dir cache/embeddings

  # Dune
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Dune" --output-dir cache/embeddings

  # Resume interrupted run — just rerun the same command
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Wheel of Time" --output-dir cache/embeddings

  # Merge existing batch files without new API calls
  uv run python -m src.etl.rag_ingestion.embed \\
      --universe "Wheel of Time" --output-dir cache/embeddings --merge-only

Default model : {EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)
Tier 1 limits : 3,000 RPM / 1,000,000 TPM / Unlimited RPD
LiteLLM config: set LITELLM_BASE_URL and LITELLM_API_KEY in .env
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--universe", required=True, help="Fantasy universe name")
    parser.add_argument(
        "--output-dir",
        default="cache/embeddings",
        help="Root directory for Parquet output (default: cache/embeddings)",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=f"Gemini embedding model ID (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per batch — one Parquet file per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip embedding generation; only merge existing batch files into final Parquet",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["google", "litellm"],
        default="google",
        help=(
            "Embedding backend to use (default: google). "
            "'google' calls the Gemini API directly via the google-genai SDK. "
            "'litellm' routes through your local LiteLLM proxy (LITELLM_BASE_URL in .env). "
            "When using 'litellm', set --model to the LiteLLM format: "
            "gemini/gemini-embedding-2-preview"
        ),
    )

    args = parser.parse_args()
    asyncio.run(
        embed_to_parquet(
            universe=args.universe,
            output_dir=args.output_dir,
            model=args.model,
            batch_size=args.batch_size,
            merge_only=args.merge_only,
            embedding_provider=args.embedding_provider,
        )
    )


if __name__ == "__main__":
    main()

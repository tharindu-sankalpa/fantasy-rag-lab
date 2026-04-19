"""
Stage 3 — Insert embeddings from Parquet cache into Milvus.

Reads the consolidated embedding Parquet file produced by Stage 2 and inserts
all records into the Milvus collection in batches. All metadata fields from
the Parquet (book_name, chapter_number, chapter_title, universe, series) are
preserved during insertion.

This step is idempotent in the sense that re-running it re-inserts all records.
Milvus uses auto-assigned INT64 primary keys, so duplicate runs will produce
duplicate entries — clear the collection first if you need a clean reload.

Usage:
    uv run python -m src.etl.rag_ingestion.load_milvus \\
        --parquet cache/embeddings/wheel_of_time/wheel_of_time_embeddings.parquet
"""

import argparse

import pandas as pd
import structlog

from src.services.milvus_service import MilvusService

logger = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 200


def parquet_to_milvus(parquet_path: str, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """Load all embeddings from a Parquet file into Milvus.

    Args:
        parquet_path: Path to the consolidated embedding Parquet file.
        batch_size: Number of rows per Milvus insert call.
    """
    log = logger.bind(parquet=parquet_path, batch_size=batch_size)

    df = pd.read_parquet(parquet_path)
    log.info(
        "parquet_loaded",
        rows=len(df),
        columns=list(df.columns),
        embedding_dim=len(df["embedding"].iloc[0]) if len(df) > 0 else 0,
    )

    if "embedding" not in df.columns:
        raise ValueError(f"Parquet file is missing the 'embedding' column: {parquet_path}")

    milvus = MilvusService()
    milvus.create_collection_if_not_exists()

    total = len(df)
    total_inserted = 0

    for start in range(0, total, batch_size):
        batch_df = df.iloc[start: start + batch_size]

        documents = [
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "embedding": list(row["embedding"]),
                "text": str(row.get("text_content", "")),
                "universe": str(row.get("universe", "")),
                "book_title": str(row.get("book_name", "")),
                "chapter_number": str(row.get("chapter_number", "")),
                "chapter_title": str(row.get("chapter_title", "")),
                "series": str(row.get("series", "")),
            }
            for _, row in batch_df.iterrows()
        ]

        milvus.insert_documents(documents)
        total_inserted += len(documents)
        log.info(
            "batch_inserted",
            batch_start=start,
            batch_size=len(documents),
            total_inserted=total_inserted,
            remaining=total - total_inserted,
        )

    log.info("stage3_complete", total_inserted=total_inserted)


def main() -> None:
    """CLI entry point for Stage 3: Parquet → Milvus."""
    parser = argparse.ArgumentParser(
        description="Insert embeddings from a Parquet cache file into Milvus.",
        epilog="""
Examples:
  uv run python -m src.etl.rag_ingestion.load_milvus \\
      --parquet cache/embeddings/wheel_of_time/wheel_of_time_embeddings.parquet

  uv run python -m src.etl.rag_ingestion.load_milvus \\
      --parquet cache/embeddings/song_of_ice_and_fire/song_of_ice_and_fire_embeddings.parquet

  uv run python -m src.etl.rag_ingestion.load_milvus \\
      --parquet cache/embeddings/harry_potter/harry_potter_embeddings.parquet

  uv run python -m src.etl.rag_ingestion.load_milvus \\
      --parquet cache/embeddings/dune/dune_embeddings.parquet --batch-size 500
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to the consolidated embedding Parquet file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per Milvus insert call (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()
    parquet_to_milvus(parquet_path=args.parquet, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

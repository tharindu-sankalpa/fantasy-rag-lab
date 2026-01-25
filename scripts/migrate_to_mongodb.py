#!/usr/bin/env python3
# Dependencies:
# pip install motor pymongo structlog pandas pyarrow

"""
Migration script: Migrate existing file-based data to MongoDB Atlas.

This script migrates:
- data/processed_books_*/      -> processed_chunks collection
- data/extracted_graph_*/      -> extraction_results collection
- data/schemas/                -> schemas collection
- cache/embeddings/*.parquet   -> embedding_cache collection

Usage:
    uv run python scripts/migrate_to_mongodb.py --all
    uv run python scripts/migrate_to_mongodb.py --chunks --schemas
    uv run python scripts/migrate_to_mongodb.py --dry-run
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog


# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Base paths
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")


async def migrate_processed_chunks(mongodb, dry_run: bool = False) -> int:
    """Migrate processed book chunks to MongoDB.

    Scans for directories matching data/processed_books_* and migrates
    all .txt files with their .meta.json metadata.

    Args:
        mongodb: MongoDBService instance.
        dry_run: If True, only report what would be migrated.

    Returns:
        Number of documents migrated.
    """
    log = logger.bind(task="migrate_chunks")
    log.info("scanning_for_processed_books")

    # Find all processed_books directories
    processed_dirs = list(DATA_DIR.glob("processed_books*"))

    if not processed_dirs:
        log.warning("no_processed_books_directories_found")
        return 0

    total_migrated = 0

    for processed_dir in processed_dirs:
        log.info("processing_directory", directory=str(processed_dir))

        # Find all .txt files (chunks)
        txt_files = sorted(processed_dir.glob("*.txt"))

        for txt_file in txt_files:
            meta_file = txt_file.with_suffix(".meta.json")

            # Read chunk text
            text_content = txt_file.read_text(encoding="utf-8")

            # Read metadata if exists
            if meta_file.exists():
                with open(meta_file) as f:
                    metadata = json.load(f)
            else:
                # Create minimal metadata
                metadata = {
                    "chunk_id": txt_file.stem,
                    "token_count": len(text_content.split()),  # Rough estimate
                }

            # Build document
            document = {
                "chunk_id": metadata.get("chunk_id", txt_file.stem),
                "series": metadata.get("series", _infer_series_from_path(txt_file)),
                "text_content": text_content,
                "token_count": metadata.get("token_count", 0),
                "character_count": metadata.get("character_count", len(text_content)),
                "context_window_used": metadata.get("context_window_used"),
                "safety_margin": metadata.get("safety_margin"),
                "included_books": metadata.get("included_books", []),
                "source_dir": str(processed_dir.name),
                "migrated_at": datetime.now(timezone.utc).isoformat(),
            }

            if dry_run:
                log.info(
                    "would_migrate_chunk",
                    chunk_id=document["chunk_id"],
                    series=document["series"],
                    tokens=document["token_count"],
                )
            else:
                await mongodb.upsert_chunk(document)

            total_migrated += 1

    log.info("chunks_migration_complete", total=total_migrated, dry_run=dry_run)
    return total_migrated


async def migrate_extraction_results(mongodb, dry_run: bool = False) -> int:
    """Migrate extracted graph results to MongoDB.

    Scans for directories matching data/extracted_graph_* and migrates
    all *_extracted.json files.

    Args:
        mongodb: MongoDBService instance.
        dry_run: If True, only report what would be migrated.

    Returns:
        Number of documents migrated.
    """
    log = logger.bind(task="migrate_extractions")
    log.info("scanning_for_extracted_graphs")

    # Find all extracted_graph directories
    extracted_dirs = list(DATA_DIR.glob("extracted_graph*"))

    if not extracted_dirs:
        log.warning("no_extracted_graph_directories_found")
        return 0

    total_migrated = 0

    for extracted_dir in extracted_dirs:
        log.info("processing_directory", directory=str(extracted_dir))

        # Find all *_extracted.json files
        json_files = sorted(extracted_dir.glob("*_extracted.json"))

        for json_file in json_files:
            with open(json_file) as f:
                extraction_data = json.load(f)

            # Build document
            chunk_id = json_file.stem.replace("_extracted", "")
            document = {
                "chunk_id": chunk_id,
                "series": _infer_series_from_filename(chunk_id),
                "entities": extraction_data.get("entities", []),
                "relationships": extraction_data.get("relationships", []),
                "schema_proposals": extraction_data.get("schema_proposals", []),
                "extraction_metadata": {
                    "source_file": str(json_file.name),
                    "source_dir": str(extracted_dir.name),
                },
                "migrated_at": datetime.now(timezone.utc).isoformat(),
            }

            if dry_run:
                log.info(
                    "would_migrate_extraction",
                    chunk_id=document["chunk_id"],
                    entities=len(document["entities"]),
                    relationships=len(document["relationships"]),
                )
            else:
                await mongodb.upsert_extraction(document)

            total_migrated += 1

    log.info("extractions_migration_complete", total=total_migrated, dry_run=dry_run)
    return total_migrated


async def migrate_schemas(mongodb, dry_run: bool = False) -> int:
    """Migrate ontology schemas to MongoDB.

    Scans data/schemas/ for *_schema.json files.

    Args:
        mongodb: MongoDBService instance.
        dry_run: If True, only report what would be migrated.

    Returns:
        Number of documents migrated.
    """
    log = logger.bind(task="migrate_schemas")
    schemas_dir = DATA_DIR / "schemas"

    if not schemas_dir.exists():
        log.warning("schemas_directory_not_found")
        return 0

    log.info("scanning_for_schemas", directory=str(schemas_dir))

    # Find all schema files
    schema_files = list(schemas_dir.glob("*_schema.json"))

    if not schema_files:
        log.warning("no_schema_files_found")
        return 0

    total_migrated = 0

    for schema_file in schema_files:
        with open(schema_file) as f:
            schema_data = json.load(f)

        # Infer series name from filename
        series = schema_file.stem.replace("_schema", "").replace("_", " ").title()

        # Handle special cases
        if "ice_and_fire" in schema_file.stem.lower():
            series = "A Song of Ice and Fire"
        elif "wheel_of_time" in schema_file.stem.lower():
            series = "The Wheel of Time"
        elif "harry_potter" in schema_file.stem.lower():
            series = "Harry Potter"

        document = {
            "series": series,
            "entity_types": schema_data.get("entity_types", []),
            "relationship_types": schema_data.get("relationship_types", []),
            "canonical_renaming_rules": schema_data.get("canonical_renaming_rules", {}),
            "source_file": str(schema_file.name),
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        }

        if dry_run:
            log.info(
                "would_migrate_schema",
                series=document["series"],
                entity_types=len(document["entity_types"]),
                relationship_types=len(document["relationship_types"]),
            )
        else:
            await mongodb.upsert_schema(document)

        total_migrated += 1

    log.info("schemas_migration_complete", total=total_migrated, dry_run=dry_run)
    return total_migrated


async def migrate_embedding_cache(mongodb, dry_run: bool = False) -> int:
    """Migrate Parquet embedding caches to MongoDB.

    Scans cache/embeddings/ for *.parquet files.

    Args:
        mongodb: MongoDBService instance.
        dry_run: If True, only report what would be migrated.

    Returns:
        Number of documents migrated.
    """
    log = logger.bind(task="migrate_embeddings")
    embeddings_dir = CACHE_DIR / "embeddings"

    if not embeddings_dir.exists():
        log.warning("embeddings_cache_directory_not_found")
        return 0

    log.info("scanning_for_embedding_caches", directory=str(embeddings_dir))

    # Find all parquet cache files
    parquet_files = list(embeddings_dir.glob("*.parquet"))

    if not parquet_files:
        log.warning("no_parquet_cache_files_found")
        return 0

    total_migrated = 0

    for parquet_file in parquet_files:
        log.info("processing_cache_file", file=str(parquet_file.name))

        # Infer model name from filename (e.g., wheel_of_time_voyage_3_large_cache.parquet)
        parts = parquet_file.stem.replace("_cache", "").split("_")
        # Heuristic: Last parts after universe name are model name
        model_name = "unknown"
        if "voyage" in parquet_file.stem:
            model_name = "voyage-3-large"
        elif "text_embedding" in parquet_file.stem:
            model_name = "text-embedding-004"

        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            log.error("parquet_read_failed", file=str(parquet_file), error=str(e))
            continue

        if df.empty:
            log.info("empty_cache_file_skipped", file=str(parquet_file.name))
            continue

        # Prepare items for bulk insert
        items = []
        for _, row in df.iterrows():
            items.append(
                {
                    "text_hash": row["text_hash"],
                    "embedding": row["embedding"],
                    "metadata": {
                        "source_file": str(parquet_file.name),
                        "text_length": len(row.get("text", "")),
                    },
                }
            )

        if dry_run:
            log.info(
                "would_migrate_embeddings",
                file=str(parquet_file.name),
                count=len(items),
                model=model_name,
            )
        else:
            count = await mongodb.bulk_cache_embeddings(items, model_name)
            log.info(
                "embeddings_migrated",
                file=str(parquet_file.name),
                count=count,
                model=model_name,
            )

        total_migrated += len(items)

    log.info("embeddings_migration_complete", total=total_migrated, dry_run=dry_run)
    return total_migrated


def _infer_series_from_path(file_path: Path) -> str:
    """Infer series name from file path.

    Args:
        file_path: Path to a chunk or extraction file.

    Returns:
        Inferred series name.
    """
    stem = file_path.stem.lower()
    if "wheel_of_time" in stem or "wot" in stem:
        return "wheel_of_time"
    elif "harry_potter" in stem or "hp" in stem:
        return "harry_potter"
    elif "asoiaf" in stem or "song_of_ice" in stem or "ice_and_fire" in stem:
        return "song_of_ice_and_fire"
    return "unknown"


def _infer_series_from_filename(filename: str) -> str:
    """Infer series name from filename.

    Args:
        filename: Filename without extension.

    Returns:
        Inferred series name.
    """
    filename = filename.lower()
    if "wheel_of_time" in filename or "wot" in filename:
        return "wheel_of_time"
    elif "harry_potter" in filename or "hp" in filename:
        return "harry_potter"
    elif "asoiaf" in filename or "song_of_ice" in filename or "ice_and_fire" in filename:
        return "song_of_ice_and_fire"
    return "unknown"


async def main() -> None:
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate file-based data to MongoDB Atlas.",
        epilog="""
Examples:
  # Migrate everything
  uv run python scripts/migrate_to_mongodb.py --all

  # Dry run to see what would be migrated
  uv run python scripts/migrate_to_mongodb.py --all --dry-run

  # Migrate only chunks and schemas
  uv run python scripts/migrate_to_mongodb.py --chunks --schemas

  # Migrate only extractions
  uv run python scripts/migrate_to_mongodb.py --extractions
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all data types",
    )
    parser.add_argument(
        "--chunks",
        action="store_true",
        help="Migrate processed chunks",
    )
    parser.add_argument(
        "--extractions",
        action="store_true",
        help="Migrate extraction results",
    )
    parser.add_argument(
        "--schemas",
        action="store_true",
        help="Migrate ontology schemas",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Migrate embedding cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--create-indexes",
        action="store_true",
        help="Create MongoDB indexes after migration",
    )

    args = parser.parse_args()

    # Check if any migration type is selected
    if not any([args.all, args.chunks, args.extractions, args.schemas, args.embeddings]):
        parser.error("Must specify at least one migration type (--all, --chunks, etc.)")

    log = logger.bind(task="migration")
    log.info("migration_started", dry_run=args.dry_run)

    # Initialize MongoDB service
    if not args.dry_run:
        from src.services.mongodb_service import MongoDBService

        mongodb = MongoDBService()
        await mongodb.connect()
    else:
        mongodb = None

    results = {}

    try:
        # Migrate chunks
        if args.all or args.chunks:
            results["chunks"] = await migrate_processed_chunks(mongodb, args.dry_run)

        # Migrate extractions
        if args.all or args.extractions:
            results["extractions"] = await migrate_extraction_results(
                mongodb, args.dry_run
            )

        # Migrate schemas
        if args.all or args.schemas:
            results["schemas"] = await migrate_schemas(mongodb, args.dry_run)

        # Migrate embeddings
        if args.all or args.embeddings:
            results["embeddings"] = await migrate_embedding_cache(mongodb, args.dry_run)

        # Create indexes
        if args.create_indexes and not args.dry_run:
            log.info("creating_indexes")
            await mongodb.create_indexes()

        # Print summary
        log.info("migration_complete", results=results, dry_run=args.dry_run)

        if not args.dry_run:
            # Get final stats
            stats = await mongodb.get_collection_stats()
            log.info("final_collection_stats", stats=stats)

    finally:
        if mongodb and not args.dry_run:
            await mongodb.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

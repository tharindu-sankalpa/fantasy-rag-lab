# Dependencies:
# pip install pandas structlog motor pymongo

"""
RAG ingestion pipeline for Fantasy books.

This module processes EPUB files, generates embeddings using Google's
text-embedding models, and stores them in Milvus for semantic search.

Supports two caching backends:
- File-based (Parquet): Default, no external dependencies
- MongoDB: Optional, for persistent centralized caching
"""

import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from src.etl.rag_ingestion.analyzer import ChunkAnalyzer
from src.etl.rag_ingestion.processor import FantasyBookProcessor
from src.services.llm import UnifiedLLMService
from src.services.milvus_service import MilvusService


logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """Manages caching of generated embeddings to avoid expensive API calls.

    This file-based cache stores embeddings in Parquet format for efficient
    retrieval and persistence.

    Attributes:
        cache_dir: Directory to store cache files.
        file_path: Full path to the specific cache file.
        cache: The loaded cache data as a DataFrame.
    """

    def __init__(self, cache_dir: str, universe: str, model_name: str):
        """Initialize the EmbeddingCache.

        Args:
            cache_dir: Directory path for storing cache files.
            universe: The fantasy universe name (used in filename).
            model_name: The embedding model name (used in filename).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Unique cache file per universe and model
        safe_universe = universe.lower().replace(" ", "_")
        safe_model = model_name.replace("/", "_").replace("-", "_")
        self.file_path = self.cache_dir / f"{safe_universe}_{safe_model}_cache.parquet"

        self.log = logger.bind(
            component="embedding_cache", file_path=str(self.file_path)
        )
        self.cache = self._load_cache()
        self.log.info("embedding_cache_initialized", cached_entries=len(self.cache))

    def _load_cache(self) -> pd.DataFrame:
        """Load the cache from the parquet file.

        Returns:
            DataFrame containing text_hash, text, and embedding columns.
        """
        if self.file_path.exists():
            try:
                df = pd.read_parquet(self.file_path)
                self.log.info("cache_loaded", rows=len(df))
                return df
            except Exception as e:
                self.log.error("cache_load_failed", error=str(e))
                return pd.DataFrame(columns=["text_hash", "text", "embedding"])
        return pd.DataFrame(columns=["text_hash", "text", "embedding"])

    def _hash_text(self, text: str) -> str:
        """Hash text using MD5.

        Args:
            text: The input text to hash.

        Returns:
            The MD5 hash of the text.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embeddings(self, texts: list[str]) -> dict[str, list[float]]:
        """Retrieve cached embeddings for a list of texts.

        Args:
            texts: List of texts to look up.

        Returns:
            Dictionary mapping text to embedding for found items.
        """
        if self.cache.empty:
            return {}

        hashes = [self._hash_text(t) for t in texts]
        found_df = self.cache[self.cache["text_hash"].isin(hashes)]

        return dict(zip(found_df["text"], found_df["embedding"]))

    def update(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Update the cache with new text-embedding pairs and persist to disk.

        Args:
            texts: List of text strings.
            embeddings: List of corresponding embedding vectors.
        """
        new_data = []
        for t, emb in zip(texts, embeddings):
            new_data.append(
                {
                    "text_hash": self._hash_text(t),
                    "text": t,
                    "embedding": emb,
                }
            )

        if not new_data:
            return

        new_df = pd.DataFrame(new_data)
        self.cache = pd.concat(
            [self.cache, new_df], ignore_index=True
        ).drop_duplicates(subset=["text_hash"], keep="last")

        try:
            self.cache.to_parquet(self.file_path)
            self.log.info(
                "cache_updated_on_disk",
                new_entries=len(new_data),
                total_size=len(self.cache),
            )
        except Exception as e:
            self.log.error("cache_write_failed", error=str(e))


class MongoDBEmbeddingCache:
    """MongoDB-backed embedding cache for centralized persistent storage.

    This cache stores embeddings in MongoDB Atlas, allowing shared access
    across multiple machines and persistent storage.

    Requires MongoDB service to be available.
    """

    def __init__(self, mongodb_service, model_name: str):
        """Initialize the MongoDB embedding cache.

        Args:
            mongodb_service: The MongoDBService instance.
            model_name: The embedding model name.
        """
        self.mongodb = mongodb_service
        self.model_name = model_name
        self.log = logger.bind(component="mongodb_embedding_cache", model=model_name)

    def _hash_text(self, text: str) -> str:
        """Hash text using MD5."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def get_embeddings(self, texts: list[str]) -> dict[str, list[float]]:
        """Retrieve cached embeddings from MongoDB.

        Args:
            texts: List of texts to look up.

        Returns:
            Dictionary mapping text to embedding for found items.
        """
        result = {}
        for text in texts:
            text_hash = self._hash_text(text)
            embedding = await self.mongodb.get_cached_embedding(
                text_hash, self.model_name
            )
            if embedding:
                result[text] = embedding
        return result

    async def update(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Update the MongoDB cache with new embeddings.

        Args:
            texts: List of text strings.
            embeddings: List of corresponding embedding vectors.
        """
        items = []
        for text, embedding in zip(texts, embeddings):
            items.append(
                {
                    "text_hash": self._hash_text(text),
                    "embedding": embedding,
                    "metadata": {"text_length": len(text)},
                }
            )

        if items:
            count = await self.mongodb.bulk_cache_embeddings(items, self.model_name)
            self.log.info("mongodb_cache_updated", new_entries=count)


async def ingest(
    directory: str,
    universe: str,
    embedding_model: str = "text-embedding-004",
    batch_size: int = 50,
    dry_run: bool = False,
    output: Optional[str] = None,
    use_mongodb_cache: bool = False,
) -> None:
    """Main ingestion function to process, embed, and store fantasy books.

    Args:
        directory: Directory containing EPUB files.
        universe: The fantasy universe name.
        embedding_model: The Google embedding model to use.
        batch_size: Number of documents to process in a single batch.
        dry_run: If True, only processes chunks and prints analysis.
        output: File path to save dry-run analysis output.
        use_mongodb_cache: If True, use MongoDB for embedding cache.
    """
    log = logger.bind(
        task="ingestion",
        directory=directory,
        universe=universe,
        model=embedding_model,
        dry_run=dry_run,
    )
    log.info("ingestion_started")

    # Initialize services
    llm_service = None
    milvus_service = None
    cache = None

    if not dry_run:
        try:
            milvus_service = MilvusService()
            milvus_service.create_collection_if_not_exists()
            llm_service = UnifiedLLMService()

            # Initialize cache
            if use_mongodb_cache:
                from src.services.mongodb_service import get_mongodb_service

                mongodb = await get_mongodb_service()
                cache = MongoDBEmbeddingCache(mongodb, embedding_model)
                log.info("using_mongodb_cache")
            else:
                cache = EmbeddingCache(
                    cache_dir="./cache/embeddings",
                    universe=universe,
                    model_name=embedding_model,
                )
                log.info("using_file_cache")

        except Exception as e:
            log.exception("service_initialization_failed")
            return
    else:
        log.info("dry_run_mode_enabled_skipping_services")

    # Process books
    processor = FantasyBookProcessor(universe=universe)
    docs = processor.process_series(directory)

    if not docs:
        log.warning("no_documents_found", directory=directory)
        return

    log.info("documents_processed", count=len(docs))

    if dry_run:
        analyzer = ChunkAnalyzer(docs)
        analyzer.log_summary()
        if output:
            analyzer.save_data(output)
        return

    # Embed and insert in batches
    total_docs = len(docs)
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        texts = [doc.page_content for doc in batch]

        # Check cache
        if use_mongodb_cache:
            cached_map = await cache.get_embeddings(texts)
        else:
            cached_map = cache.get_embeddings(texts)

        texts_to_embed = [t for t in texts if t not in cached_map]

        log.info(
            "processing_batch",
            batch_index=i,
            batch_size=len(batch),
            cache_hits=len(cached_map),
            texts_to_embed=len(texts_to_embed),
        )

        try:
            # Generate missing embeddings using Google
            if texts_to_embed:
                result = await llm_service.generate_embeddings(
                    texts=texts_to_embed,
                    model=embedding_model,
                )
                new_embeddings = result["embeddings"]

                # Update cache
                if use_mongodb_cache:
                    await cache.update(texts_to_embed, new_embeddings)
                else:
                    cache.update(texts_to_embed, new_embeddings)

                # Update local map
                cached_map.update(zip(texts_to_embed, new_embeddings))

            # Reconstruct correct order list of embeddings
            final_embeddings = [cached_map[t] for t in texts]

            # Prepare data for Milvus
            insert_data = []
            for doc, emb in zip(batch, final_embeddings):
                data = {
                    "text": doc.page_content,
                    "embedding": emb,
                    "universe": doc.metadata.get("universe"),
                    "book_title": doc.metadata.get("book_name"),
                    "chapter": f"{doc.metadata.get('chapter_number')} - {doc.metadata.get('chapter_title')}",
                }
                insert_data.append(data)

            # Insert into Milvus
            milvus_service.insert_documents(insert_data)

        except Exception as e:
            log.exception("batch_processing_failed", batch_index=i)
            continue

    log.info("ingestion_complete", universe=universe)


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest fantasy books into Milvus with Google embeddings.",
        epilog="""
Examples:
  # Basic ingestion
  uv run python -m src.etl.rag_ingestion.ingest \\
      --dir data/wheel_of_time --universe "Wheel of Time"

  # Dry run to analyze chunks
  uv run python -m src.etl.rag_ingestion.ingest \\
      --dir data/harry_potter --universe "Harry Potter" --dry-run

  # Use MongoDB for embedding cache
  uv run python -m src.etl.rag_ingestion.ingest \\
      --dir data/wheel_of_time --universe "Wheel of Time" --mongodb-cache
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dir", required=True, help="Directory containing EPUB files"
    )
    parser.add_argument(
        "--universe", required=True, help="Name of the fantasy universe"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-004",
        help="Google embedding model to use (default: text-embedding-004)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis without embedding or storing",
    )
    parser.add_argument(
        "--output", help="Output file for dry-run analysis (JSON)"
    )
    parser.add_argument(
        "--mongodb-cache",
        action="store_true",
        help="Use MongoDB for embedding cache instead of file-based cache",
    )

    args = parser.parse_args()

    asyncio.run(
        ingest(
            directory=args.dir,
            universe=args.universe,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            output=args.output,
            use_mongodb_cache=args.mongodb_cache,
        )
    )


if __name__ == "__main__":
    main()

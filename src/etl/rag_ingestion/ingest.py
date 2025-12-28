# Dependencies:
# pip install pandas structlog

import argparse
import asyncio
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import json

import pandas as pd
from src.utils.logger import logger
from src.ingestion.processor import FantasyBookProcessor
from src.ingestion.analyzer import ChunkAnalyzer
from src.services.milvus_service import MilvusService
from src.services.llm_service import LLMService

class EmbeddingCache:
    """
    Manages caching of generated embeddings to differ expensive API calls.

    Attributes:
        cache_dir (Path): Directory to store cache files.
        file_path (Path): Full path to the specific cache file.
        log (structlog.stdlib.BoundLogger): Logger with context bound.
        cache (pd.DataFrame): The loaded cache data.
    """

    def __init__(self, cache_dir: str, universe: str, model_name: str):
        """
        Initializes the EmbeddingCache.

        Args:
            cache_dir: Directory path for storing cache files.
            universe: The fantasy universe name (used in filename).
            model_name: The embedding model name (used in filename).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Unique cache file per universe and model
        self.file_path = self.cache_dir / f"{universe.lower().replace(' ', '_')}_{model_name}_cache.parquet"
        self.log = logger.bind(component="embedding_cache", file_path=str(self.file_path))
        self.cache = self._load_cache()
        self.log.info("embedding_cache_initialized", cached_entries=len(self.cache))

    def _load_cache(self) -> pd.DataFrame:
        """
        Loads the cache from the parquet file.

        Returns:
            pd.DataFrame: DataFrame containing text_hash, text, and embedding columns.
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
        """
        Hashes text using MD5.

        Args:
            text: The input text to hash.

        Returns:
            str: The MD5 hash of the text.
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_embeddings(self, texts: List[str]) -> Dict[str, List[float]]:
        """
        Retrieves cached embeddings for a list of texts.

        Args:
            texts: List of texts to look up.

        Returns:
            Dict[str, List[float]]: Dictionary mapping text to embedding for found items.
        """
        if self.cache.empty:
            return {}
            
        hashes = [self._hash_text(t) for t in texts]
        # Filter where hash exists
        found_df = self.cache[self.cache['text_hash'].isin(hashes)]
        
        # Create map
        return dict(zip(found_df['text'], found_df['embedding']))

    def update(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Updates the cache with new text-embedding pairs and persists to disk.

        Args:
            texts: List of text strings.
            embeddings: List of corresponding embedding vectors.
        """
        new_data = []
        for t, emb in zip(texts, embeddings):
            new_data.append({
                "text_hash": self._hash_text(t),
                "text": t,
                "embedding": emb
            })
        
        if not new_data:
            return

        new_df = pd.DataFrame(new_data)
        self.cache = pd.concat([self.cache, new_df], ignore_index=True).drop_duplicates(subset=['text_hash'], keep='last')
        
        # Atomic write (ish)
        try:
            self.cache.to_parquet(self.file_path)
            self.log.info("cache_updated_on_disk", new_entries=len(new_data), total_size=len(self.cache))
        except Exception as e:
            self.log.error("cache_write_failed", error=str(e))


async def ingest(directory: str, universe: str, embedding_model: str = "voyage", batch_size: int = 50, dry_run: bool = False, output: str = None) -> None:
    """
    Main ingestion function to process, embed, and store fantasy books.

    Args:
        directory: Directory containing EPUB files.
        universe: The fantasy universe name.
        embedding_model: The embedding model to use (default: "voyage").
        batch_size: Number of documents to process in a single batch.
        dry_run: If True, only processes chunks and prints analysis without embedding.
        output: File path to save dry-run analysis output.
    """
    log = logger.bind(task="ingestion", directory=directory, universe=universe, model=embedding_model, dry_run=dry_run)
    log.info("ingestion_started")
    
    # 1. Init Services
    # 1. Init Services
    if not dry_run:
        try:
            milvus_service = MilvusService()
            milvus_service.create_collection_if_not_exists()
            llm_service = LLMService()
            # Initialize Cache
            cache = EmbeddingCache(cache_dir="./cache/embeddings", universe=universe, model_name=embedding_model)
        except Exception as e:
            log.exception("service_initialization_failed")
            return
    else:
        log.info("dry_run_mode_enabled_skipping_services")

    # 2. Process Books
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
    
    # 3. Embed and Insert in Batches
    total_docs = len(docs)
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        
        # Prepare text list for embedding
        texts = [doc.page_content for doc in batch]
        
        # Check Cache
        cached_map = cache.get_embeddings(texts)
        
        texts_to_embed = [t for t in texts if t not in cached_map]
        
        log.info("processing_batch", 
                   batch_index=i, 
                   batch_size=len(batch), 
                   cache_hits=len(cached_map), 
                   texts_to_embed=len(texts_to_embed))
        
        try:
            # Generate Missing Embeddings
            if texts_to_embed:
                new_embeddings = llm_service.generate_embeddings_batch(texts_to_embed, model_name=embedding_model)
                # Update Cache
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
                    "chapter": f"{doc.metadata.get('chapter_number')} - {doc.metadata.get('chapter_title')}"
                }
                insert_data.append(data)
            
            # Insert
            milvus_service.insert_documents(insert_data)
            
        except Exception as e:
            log.exception("batch_processing_failed", batch_index=i)
            continue
            
    log.info("ingestion_complete", universe=universe)

if __name__ == "__main__":
    # Standard CLI invocation
    parser = argparse.ArgumentParser(description="Ingest fantasy books into Milvus")
    parser.add_argument("--dir", required=True, help="Directory containing EPUB files")
    parser.add_argument("--universe", required=True, help="Name of the fantasy universe")
    parser.add_argument("--embedding_model", default="voyage", help="Embedding model to use: voyage, openai, google")
    parser.add_argument("--dry-run", action="store_true", help="Run analysis without embedding")
    parser.add_argument("--output", help="Output file for dry-run analysis (json)")
    
    args = parser.parse_args()
    
    # Run async
    asyncio.run(ingest(args.dir, args.universe, args.embedding_model, dry_run=args.dry_run, output=args.output))

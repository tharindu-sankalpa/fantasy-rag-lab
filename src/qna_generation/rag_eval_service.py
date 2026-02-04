"""RAG Evaluation Q&A Generation Service.

This service generates Q&A pairs specifically for evaluating RAG systems.
It reads from the rag_chunks collection (small ~1000 token chunks) and
generates questions that explicitly track which chunks contain the answers.

Key features:
- Batches ~100 chunks at a time (~100k tokens total)
- Orders chunks by chunk_id for sequential processing
- Tracks source_chunk_ids for each Q&A pair
- Stores results in wot_qna with category="rag_evaluation"
"""

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings
from src.services.llm.google_provider import GoogleProvider
from src.services.mongodb_service import MongoDBService

from .rag_eval_prompts import RAG_EVAL_SYSTEM_INSTRUCTION, get_rag_eval_prompt
from .rag_eval_schemas import RAGEvalGenerationResult, RAGEvalProgress, RAGEvalQAPair
from .service import RateLimiter

logger = structlog.get_logger(__name__)


class RAGEvalQAService:
    """Service for generating RAG evaluation Q&A pairs.

    Processes RAG chunks in batches and generates Q&A pairs with explicit
    source chunk tracking for RAG system evaluation.
    """

    def __init__(
        self,
        mongodb_service: Optional[MongoDBService] = None,
        google_provider: Optional[GoogleProvider] = None,
        model: str = "gemini-3-flash-preview",
        max_output_tokens: int = 65536,
        temperature: float = 0.3,
        batch_size: int = 100,
        requests_per_minute: int = 1000,
        requests_per_day: int = 10000,
        tokens_per_minute: int = 1_000_000,
    ):
        """Initialize the RAG evaluation Q&A service.

        Args:
            mongodb_service: MongoDB service instance (created if not provided)
            google_provider: Google LLM provider (created if not provided)
            model: Gemini model to use
            max_output_tokens: Max output tokens per request
            temperature: Generation temperature
            batch_size: Number of chunks per batch (default 100)
            requests_per_minute: RPM limit
            requests_per_day: RPD limit
            tokens_per_minute: TPM limit
        """
        self.mongodb = mongodb_service
        self.provider = google_provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_day=requests_per_day,
            tokens_per_minute=tokens_per_minute,
        )
        self.log = logger.bind(component="RAGEvalQAService", model=model)

    async def initialize(self) -> None:
        """Initialize service connections."""
        self.log.info("initializing_rag_eval_service")

        if not self.mongodb:
            self.mongodb = MongoDBService()
            await self.mongodb.connect()

        if not self.provider:
            api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY."
                )
            self.provider = GoogleProvider(api_key=api_key)

        await self._create_indexes()
        self.log.info("rag_eval_service_initialized")

    async def _create_indexes(self) -> None:
        """Create indexes for RAG evaluation Q&A."""
        await self.mongodb.db.wot_qna.create_index("qa_id", unique=True)
        await self.mongodb.db.wot_qna.create_index("category")
        await self.mongodb.db.wot_qna.create_index("metadata.source_chunk_ids")
        await self.mongodb.db.wot_qna.create_index(
            [("category", 1), ("metadata.series", 1)]
        )
        self.log.info("rag_eval_indexes_created")

    async def _generate_qa_for_batch(
        self,
        chunks: list[dict[str, Any]],
        batch_index: int,
    ) -> list[RAGEvalQAPair]:
        """Generate Q&A pairs for a batch of chunks.

        Args:
            chunks: List of RAG chunk documents
            batch_index: Index of this batch for logging

        Returns:
            List of generated Q&A pairs with source tracking
        """
        # Estimate tokens (1 token ≈ 4 chars)
        total_chars = sum(len(c.get("text_content", "")) for c in chunks)
        estimated_tokens = total_chars // 4

        log = self.log.bind(
            batch_index=batch_index,
            chunk_count=len(chunks),
            estimated_tokens=estimated_tokens,
        )
        log.info("generating_qa_for_batch")

        # Acquire rate limit
        await self.rate_limiter.acquire(estimated_tokens)

        # Build prompt with chunk IDs
        prompt = get_rag_eval_prompt(chunks)

        # Generate with retry
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=2, min=4, max=60),
                retry=retry_if_exception_type((Exception,)),
                reraise=True,
            ):
                with attempt:
                    log.info(
                        "generation_attempt",
                        attempt_number=attempt.retry_state.attempt_number,
                    )

                    result = await self.provider.generate_structured(
                        prompt=prompt,
                        model=self.model,
                        schema=RAGEvalGenerationResult,
                        max_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                        system_instruction=RAG_EVAL_SYSTEM_INSTRUCTION,
                    )

                    parsed: RAGEvalGenerationResult = result["parsed"]
                    usage = result["usage"]

                    log.info(
                        "batch_qa_generated",
                        num_pairs=len(parsed.qa_pairs),
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )

                    return parsed.qa_pairs

        except Exception as e:
            log.error("batch_generation_failed", error=str(e))
            raise

    async def _validate_chunk_ids(
        self,
        qa_pairs: list[RAGEvalQAPair],
        valid_chunk_ids: set[str],
    ) -> list[RAGEvalQAPair]:
        """Validate that source_chunk_ids reference valid chunks.

        Args:
            qa_pairs: Generated Q&A pairs
            valid_chunk_ids: Set of chunk IDs that were in the batch

        Returns:
            List of Q&A pairs with valid chunk references only
        """
        validated = []
        for qa in qa_pairs:
            # Filter to only valid chunk IDs
            valid_sources = [
                cid for cid in qa.source_chunk_ids if cid in valid_chunk_ids
            ]
            if valid_sources:
                qa.source_chunk_ids = valid_sources
                validated.append(qa)
            else:
                self.log.warning(
                    "qa_pair_skipped_invalid_chunks",
                    question=qa.question[:50],
                    invalid_ids=qa.source_chunk_ids,
                )
        return validated

    async def _store_qa_pairs(
        self,
        qa_pairs: list[RAGEvalQAPair],
        chunks: list[dict[str, Any]],
        batch_index: int,
        series: str,
    ) -> int:
        """Store generated Q&A pairs in MongoDB.

        Args:
            qa_pairs: List of Q&A pairs to store
            chunks: Source chunks for metadata extraction
            batch_index: Batch index for ID generation
            series: Series identifier

        Returns:
            Number of documents stored
        """
        # Build chunk_id to metadata mapping
        chunk_metadata = {
            c["chunk_id"]: {
                "book_name": c.get("metadata", {}).get("book_name"),
                "chapter_title": c.get("metadata", {}).get("chapter_title"),
            }
            for c in chunks
        }

        documents = []
        now = datetime.now(timezone.utc)

        for idx, qa in enumerate(qa_pairs):
            qa_id = f"{series}_rag_eval_{batch_index:04d}_{idx:04d}"

            # Collect included books from source chunks
            included_books = list(
                set(
                    chunk_metadata.get(cid, {}).get("book_name")
                    for cid in qa.source_chunk_ids
                    if chunk_metadata.get(cid, {}).get("book_name")
                )
            )

            included_chapters = list(
                set(
                    chunk_metadata.get(cid, {}).get("chapter_title")
                    for cid in qa.source_chunk_ids
                    if chunk_metadata.get(cid, {}).get("chapter_title")
                )
            )

            doc = {
                "qa_id": qa_id,
                "question": qa.question,
                "answer": qa.answer,
                "category": "rag_evaluation",
                "complexity": "rag_eval",  # Not applicable for RAG eval
                "evidence_quote": qa.evidence_quote,
                "metadata": {
                    "source_chunk_ids": qa.source_chunk_ids,
                    "included_books": included_books,
                    "included_chapters": included_chapters,
                    "series": series,
                    "batch_index": batch_index,
                    "generation_model": self.model,
                },
                "created_at": now,
                "updated_at": now,
            }
            documents.append(doc)

        # Bulk upsert
        from pymongo import UpdateOne

        operations = [
            UpdateOne({"qa_id": doc["qa_id"]}, {"$set": doc}, upsert=True)
            for doc in documents
        ]

        if operations:
            result = await self.mongodb.db.wot_qna.bulk_write(operations)
            stored_count = result.upserted_count + result.modified_count
            self.log.info(
                "rag_eval_qa_stored",
                batch_index=batch_index,
                stored_count=stored_count,
            )
            return stored_count

        return 0

    async def process_batch(
        self,
        chunks: list[dict[str, Any]],
        batch_index: int,
        series: str,
    ) -> int:
        """Process a single batch of chunks.

        Args:
            chunks: List of RAG chunk documents
            batch_index: Index of this batch
            series: Series identifier

        Returns:
            Number of Q&A pairs generated and stored
        """
        try:
            # Get valid chunk IDs for validation
            valid_chunk_ids = {c["chunk_id"] for c in chunks}

            # Generate Q&A pairs
            qa_pairs = await self._generate_qa_for_batch(chunks, batch_index)

            # Validate chunk ID references
            qa_pairs = await self._validate_chunk_ids(qa_pairs, valid_chunk_ids)

            if qa_pairs:
                return await self._store_qa_pairs(qa_pairs, chunks, batch_index, series)
            return 0

        except RetryError as e:
            self.log.error(
                "batch_failed_after_retries",
                batch_index=batch_index,
                error=str(e.last_attempt.exception()),
            )
            raise
        except Exception as e:
            self.log.error(
                "batch_processing_failed",
                batch_index=batch_index,
                error=str(e),
            )
            raise

    async def process_series(
        self,
        series: str = "wheel_of_time",
        skip_processed: bool = True,
        start_batch: int = 0,
        max_batches: Optional[int] = None,
    ) -> RAGEvalProgress:
        """Process all RAG chunks for a series in batches.

        Args:
            series: Series identifier
            skip_processed: Skip batches that already have Q&A pairs
            start_batch: Batch index to start from (for resuming)
            max_batches: Maximum number of batches to process (for testing)

        Returns:
            RAGEvalProgress with processing results
        """
        self.log.info(
            "starting_rag_eval_processing",
            series=series,
            batch_size=self.batch_size,
            start_batch=start_batch,
        )

        # Load all RAG chunks for the series, sorted by chunk_id
        all_chunks = await self.mongodb.get_rag_chunks_by_series(series)

        if not all_chunks:
            self.log.warning("no_rag_chunks_found", series=series)
            return RAGEvalProgress(
                series=series,
                total_chunks=0,
                total_batches=0,
            )

        # Calculate batches
        total_batches = math.ceil(len(all_chunks) / self.batch_size)
        if max_batches:
            total_batches = min(total_batches, start_batch + max_batches)

        self.log.info(
            "rag_chunks_loaded",
            total_chunks=len(all_chunks),
            total_batches=total_batches,
            batch_size=self.batch_size,
        )

        # Get already processed batch indices if skipping
        processed_batches = set()
        if skip_processed:
            cursor = self.mongodb.db.wot_qna.distinct(
                "metadata.batch_index",
                {
                    "metadata.series": series,
                    "category": "rag_evaluation",
                },
            )
            processed_batches = set(await cursor)
            self.log.info(
                "skipping_processed_batches",
                already_processed=len(processed_batches),
            )

        # Initialize progress
        progress = RAGEvalProgress(
            series=series,
            total_chunks=len(all_chunks),
            total_batches=total_batches,
        )

        # Process batches
        for batch_idx in range(start_batch, total_batches):
            # Skip if already processed
            if skip_processed and batch_idx in processed_batches:
                self.log.info("skipping_processed_batch", batch_index=batch_idx)
                progress.processed_batches += 1
                continue

            # Get chunks for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(all_chunks))
            batch_chunks = all_chunks[start_idx:end_idx]

            try:
                qa_count = await self.process_batch(batch_chunks, batch_idx, series)
                progress.processed_batches += 1
                progress.total_qa_pairs += qa_count
                progress.last_processed_batch = batch_idx
                progress.updated_at = datetime.now(timezone.utc)

                self.log.info(
                    "batch_completed",
                    batch_index=batch_idx,
                    qa_count=qa_count,
                    progress=f"{progress.processed_batches}/{total_batches}",
                )

                # Delay between batches
                await asyncio.sleep(2)

            except Exception as e:
                self.log.error(
                    "batch_failed",
                    batch_index=batch_idx,
                    error=str(e),
                )
                progress.failed_batches.append(batch_idx)

                # Abort after 3 failures
                if len(progress.failed_batches) >= 3:
                    self.log.error(
                        "too_many_failures_aborting",
                        failed_count=len(progress.failed_batches),
                    )
                    break

        self.log.info(
            "rag_eval_processing_complete",
            series=series,
            total_qa_pairs=progress.total_qa_pairs,
            failed_batches=len(progress.failed_batches),
        )

        return progress

    async def get_rag_eval_stats(self, series: str = "wheel_of_time") -> dict[str, Any]:
        """Get statistics about RAG evaluation Q&A pairs.

        Args:
            series: Series identifier

        Returns:
            Dictionary with statistics
        """
        pipeline = [
            {
                "$match": {
                    "metadata.series": series,
                    "category": "rag_evaluation",
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_count": {"$sum": 1},
                    "unique_batches": {"$addToSet": "$metadata.batch_index"},
                    "all_source_chunks": {"$push": "$metadata.source_chunk_ids"},
                }
            },
        ]

        cursor = self.mongodb.db.wot_qna.aggregate(pipeline)
        results = await cursor.to_list(length=1)

        if not results:
            return {
                "series": series,
                "total_qa_pairs": 0,
                "unique_batches": 0,
                "unique_source_chunks": 0,
            }

        result = results[0]

        # Flatten source chunks to count unique
        all_sources = []
        for sources in result.get("all_source_chunks", []):
            if sources:
                all_sources.extend(sources)

        return {
            "series": series,
            "total_qa_pairs": result["total_count"],
            "unique_batches": len(result.get("unique_batches", [])),
            "unique_source_chunks": len(set(all_sources)),
            "rate_limiter_status": self.rate_limiter.get_status(),
        }

    async def close(self) -> None:
        """Close service connections."""
        if self.mongodb:
            await self.mongodb.disconnect()
        self.log.info("rag_eval_service_closed")

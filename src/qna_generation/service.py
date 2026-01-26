"""QA Generation Service with rate limiting and retry logic.

This module provides the core QA generation functionality with:
- Gemini Pro integration for large context processing
- Category-based question generation (5 pillars)
- Rate limiting to respect API quotas (25 RPM, 250 RPD, 1M tokens/min)
- Exponential backoff retry with tenacity
- Progress tracking and resumable processing
- MongoDB persistence for generated QA pairs

Architecture:
    QAGenerationService
        ├── GoogleProvider (LLM calls)
        ├── MongoDBService (persistence)
        ├── RateLimiter (quota management)
        └── RetryHandler (fault tolerance)
"""

import asyncio
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

from .prompts import (
    CATEGORY_INFO,
    QuestionCategory,
    SYSTEM_INSTRUCTION,
    get_category_prompt,
)
from .schemas import (
    GenerationProgress,
    QAGenerationResult,
    QAPair,
)

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter for Gemini API quotas.

    Implements a sliding window rate limiter that respects:
    - Requests per minute: 25
    - Requests per day: 250
    - Input tokens per minute: 1,000,000

    The limiter uses a conservative approach, tracking requests in sliding
    windows and blocking when approaching limits.
    """

    def __init__(
        self,
        requests_per_minute: int = 25,
        requests_per_day: int = 250,
        tokens_per_minute: int = 1_000_000,
    ):
        """Initialize rate limiter with quota limits.

        Args:
            requests_per_minute: Max requests per minute (default: 25)
            requests_per_day: Max requests per day (default: 250)
            tokens_per_minute: Max input tokens per minute (default: 1M)
        """
        self.rpm_limit = requests_per_minute
        self.rpd_limit = requests_per_day
        self.tpm_limit = tokens_per_minute

        # Sliding window tracking
        self._minute_requests: list[float] = []
        self._day_requests: list[float] = []
        self._minute_tokens: list[tuple[float, int]] = []

        self._lock = asyncio.Lock()
        self.log = logger.bind(component="RateLimiter")

    async def acquire(self, estimated_tokens: int) -> None:
        """Acquire permission to make a request.

        Blocks until the request can be made within quota limits.
        Uses exponential backoff when rate limited.

        Args:
            estimated_tokens: Estimated input tokens for the request
        """
        async with self._lock:
            now = datetime.now(timezone.utc).timestamp()

            # Clean old entries (sliding window)
            minute_ago = now - 60
            day_ago = now - 86400

            self._minute_requests = [t for t in self._minute_requests if t > minute_ago]
            self._day_requests = [t for t in self._day_requests if t > day_ago]
            self._minute_tokens = [
                (t, tokens)
                for t, tokens in self._minute_tokens
                if t > minute_ago
            ]

            # Calculate current usage
            current_rpm = len(self._minute_requests)
            current_rpd = len(self._day_requests)
            current_tpm = sum(tokens for _, tokens in self._minute_tokens)

            # Check and wait for RPM limit
            if current_rpm >= self.rpm_limit:
                wait_time = 60 - (now - self._minute_requests[0])
                self.log.info(
                    "rpm_limit_reached",
                    current=current_rpm,
                    limit=self.rpm_limit,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(max(wait_time, 1))

            # Check RPD limit (hard stop if exceeded)
            if current_rpd >= self.rpd_limit:
                self.log.warning(
                    "rpd_limit_reached",
                    current=current_rpd,
                    limit=self.rpd_limit,
                )
                raise RuntimeError(
                    f"Daily request limit ({self.rpd_limit}) exceeded. "
                    "Wait until tomorrow or use a different API key."
                )

            # Check TPM limit
            if current_tpm + estimated_tokens > self.tpm_limit:
                wait_time = 60 - (now - self._minute_tokens[0][0])
                self.log.info(
                    "tpm_limit_reached",
                    current=current_tpm,
                    estimated=estimated_tokens,
                    limit=self.tpm_limit,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(max(wait_time, 1))

            # Record this request
            self._minute_requests.append(now)
            self._day_requests.append(now)
            self._minute_tokens.append((now, estimated_tokens))

            self.log.debug(
                "rate_limit_acquired",
                rpm=len(self._minute_requests),
                rpd=len(self._day_requests),
                tpm=sum(t for _, t in self._minute_tokens),
            )

    def get_status(self) -> dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Dictionary with current usage and limits
        """
        now = datetime.now(timezone.utc).timestamp()
        minute_ago = now - 60
        day_ago = now - 86400

        current_rpm = len([t for t in self._minute_requests if t > minute_ago])
        current_rpd = len([t for t in self._day_requests if t > day_ago])
        current_tpm = sum(
            tokens for t, tokens in self._minute_tokens if t > minute_ago
        )

        return {
            "requests_per_minute": {"current": current_rpm, "limit": self.rpm_limit},
            "requests_per_day": {"current": current_rpd, "limit": self.rpd_limit},
            "tokens_per_minute": {"current": current_tpm, "limit": self.tpm_limit},
        }


class QAGenerationService:
    """Service for generating QA pairs from graph chunks.

    This service orchestrates the QA generation pipeline:
    1. Loads graph chunks from MongoDB
    2. Sends chunks to Gemini with category-specific prompts
    3. Parses and validates the output
    4. Stores results in the wot_qna collection
    5. Tracks progress for resumable processing

    The service respects Gemini API quotas and implements robust error handling.
    """

    def __init__(
        self,
        mongodb_service: Optional[MongoDBService] = None,
        google_provider: Optional[GoogleProvider] = None,
        model: str = "gemini-3-pro-preview",
        max_output_tokens: int = 8192,
        temperature: float = 0.3,
    ):
        """Initialize the QA generation service.

        Args:
            mongodb_service: MongoDB service instance (created if not provided)
            google_provider: Google LLM provider (created if not provided)
            model: Gemini model to use (default: gemini-2.5-pro for large context)
            max_output_tokens: Max tokens for generated output
            temperature: Generation temperature (0.3 for balanced creativity)
        """
        self.mongodb = mongodb_service
        self.provider = google_provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.rate_limiter = RateLimiter()
        self.log = logger.bind(component="QAGenerationService", model=model)

    async def initialize(self) -> None:
        """Initialize service connections.

        Creates MongoDB and Google provider instances if not provided,
        establishes connections, and creates necessary indexes.
        """
        self.log.info("initializing_service")

        # Initialize MongoDB
        if not self.mongodb:
            self.mongodb = MongoDBService()
            await self.mongodb.connect()

        # Initialize Google provider
        if not self.provider:
            api_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY."
                )
            self.provider = GoogleProvider(api_key=api_key)

        # Create indexes for wot_qna collection
        await self._create_indexes()

        self.log.info("service_initialized")

    async def _create_indexes(self) -> None:
        """Create indexes for the wot_qna collection."""
        await self.mongodb.db.wot_qna.create_index("qa_id", unique=True)
        await self.mongodb.db.wot_qna.create_index("metadata.source_chunk_id")
        await self.mongodb.db.wot_qna.create_index("category")
        await self.mongodb.db.wot_qna.create_index("complexity")
        await self.mongodb.db.wot_qna.create_index(
            [("metadata.source_chunk_id", 1), ("category", 1)]
        )
        self.log.info("wot_qna_indexes_created")

    async def _generate_qa_for_chunk(
        self,
        chunk: dict[str, Any],
        category: QuestionCategory,
    ) -> list[QAPair]:
        """Generate QA pairs for a single chunk with a specific category.

        Args:
            chunk: Graph chunk document from MongoDB
            category: The question category to focus on

        Returns:
            List of generated QA pairs

        Raises:
            ValueError: If generation or parsing fails
        """
        chunk_id = chunk["chunk_id"]
        text_content = chunk["text_content"]
        token_count = chunk.get("token_count", len(text_content.split()))

        log = self.log.bind(
            chunk_id=chunk_id,
            token_count=token_count,
            category=category.value,
        )
        log.info("generating_qa_for_chunk")

        # Acquire rate limit
        await self.rate_limiter.acquire(token_count)

        # Build the category-specific prompt
        prompt = get_category_prompt(category, text_content)

        # Generate with retry logic
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
                    schema=QAGenerationResult,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    system_instruction=SYSTEM_INSTRUCTION,
                )

                parsed: QAGenerationResult = result["parsed"]
                usage = result["usage"]

                log.info(
                    "qa_generated",
                    num_pairs=len(parsed.qa_pairs),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )

                return parsed.qa_pairs

        # This line should not be reached due to reraise=True
        return []

    async def _store_qa_pairs(
        self,
        qa_pairs: list[QAPair],
        chunk: dict[str, Any],
        category: QuestionCategory,
    ) -> int:
        """Store generated QA pairs in MongoDB.

        Args:
            qa_pairs: List of QA pairs to store
            chunk: Source chunk for metadata
            category: The question category

        Returns:
            Number of documents stored
        """
        chunk_id = chunk["chunk_id"]
        included_books = chunk.get("included_books", [])
        series = chunk.get("series", "wheel_of_time")

        documents = []
        now = datetime.now(timezone.utc)

        for idx, qa in enumerate(qa_pairs):
            # Include category in qa_id for uniqueness across passes
            qa_id = f"{chunk_id}_{category.value}_{idx:04d}"

            doc = {
                "qa_id": qa_id,
                "question": qa.question,
                "answer": qa.answer,
                "category": category.value,
                "complexity": qa.complexity.value,
                "evidence_quote": qa.evidence_quote,
                "metadata": {
                    "source_chunk_id": chunk_id,
                    "included_books": included_books,
                    "series": series,
                    "generation_model": self.model,
                    "category_name": CATEGORY_INFO[category].name,
                },
                "created_at": now,
                "updated_at": now,
            }
            documents.append(doc)

        # Bulk upsert
        from pymongo import UpdateOne

        operations = [
            UpdateOne(
                {"qa_id": doc["qa_id"]},
                {"$set": doc},
                upsert=True,
            )
            for doc in documents
        ]

        if operations:
            result = await self.mongodb.db.wot_qna.bulk_write(operations)
            stored_count = result.upserted_count + result.modified_count
            self.log.info(
                "qa_pairs_stored",
                chunk_id=chunk_id,
                category=category.value,
                stored_count=stored_count,
            )
            return stored_count

        return 0

    async def process_chunk(
        self,
        chunk: dict[str, Any],
        category: QuestionCategory,
    ) -> int:
        """Process a single chunk for a specific category.

        Args:
            chunk: Graph chunk document
            category: The question category to generate

        Returns:
            Number of QA pairs generated and stored
        """
        try:
            qa_pairs = await self._generate_qa_for_chunk(chunk, category)
            if qa_pairs:
                return await self._store_qa_pairs(qa_pairs, chunk, category)
            return 0
        except RetryError as e:
            self.log.error(
                "chunk_processing_failed_after_retries",
                chunk_id=chunk["chunk_id"],
                category=category.value,
                error=str(e.last_attempt.exception()),
            )
            raise
        except Exception as e:
            self.log.error(
                "chunk_processing_failed",
                chunk_id=chunk["chunk_id"],
                category=category.value,
                error=str(e),
            )
            raise

    async def process_series(
        self,
        series: str = "wheel_of_time",
        category: QuestionCategory = QuestionCategory.CHARACTERS,
        skip_processed: bool = True,
        chunk_ids: Optional[list[str]] = None,
    ) -> GenerationProgress:
        """Process all chunks for a series with a specific category.

        Args:
            series: Series identifier (default: wheel_of_time)
            category: Question category to generate
            skip_processed: Skip chunks that already have QA pairs for this category
            chunk_ids: Optional list of specific chunk IDs to process

        Returns:
            GenerationProgress with processing results
        """
        self.log.info(
            "starting_series_processing",
            series=series,
            category=category.value,
        )

        # Load all graph chunks for the series
        if chunk_ids:
            chunks = []
            for cid in chunk_ids:
                chunk = await self.mongodb.get_graph_chunk(cid)
                if chunk:
                    chunks.append(chunk)
        else:
            chunks = await self.mongodb.get_graph_chunks_by_series(series)

        if not chunks:
            self.log.warning("no_chunks_found", series=series)
            return GenerationProgress(
                series=series,
                category=category.value,
                total_chunks=0,
                processed_chunks=0,
            )

        # Get already processed chunk IDs for this category if skipping
        processed_chunk_ids = set()
        if skip_processed:
            cursor = self.mongodb.db.wot_qna.distinct(
                "metadata.source_chunk_id",
                {
                    "metadata.series": series,
                    "category": category.value,
                },
            )
            processed_chunk_ids = set(await cursor)
            self.log.info(
                "skipping_processed_chunks",
                category=category.value,
                already_processed=len(processed_chunk_ids),
            )

        # Initialize progress tracking
        progress = GenerationProgress(
            series=series,
            category=category.value,
            total_chunks=len(chunks),
        )

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            # Skip if already processed for this category
            if skip_processed and chunk_id in processed_chunk_ids:
                self.log.info(
                    "skipping_already_processed",
                    chunk_id=chunk_id,
                    category=category.value,
                )
                progress.processed_chunks += 1
                continue

            try:
                qa_count = await self.process_chunk(chunk, category)
                progress.processed_chunks += 1
                progress.total_qa_pairs += qa_count
                progress.last_processed_chunk = chunk_id
                progress.updated_at = datetime.now(timezone.utc)

                self.log.info(
                    "chunk_completed",
                    chunk_id=chunk_id,
                    category=category.value,
                    qa_count=qa_count,
                    progress=f"{progress.processed_chunks}/{progress.total_chunks}",
                )

                # Small delay between chunks for stability
                await asyncio.sleep(1)

            except Exception as e:
                self.log.error(
                    "chunk_failed",
                    chunk_id=chunk_id,
                    category=category.value,
                    error=str(e),
                )
                progress.failed_chunks.append(chunk_id)

                # Check if we should abort (too many failures)
                if len(progress.failed_chunks) >= 3:
                    self.log.error(
                        "too_many_failures_aborting",
                        failed_count=len(progress.failed_chunks),
                    )
                    break

        self.log.info(
            "series_processing_complete",
            series=series,
            category=category.value,
            total_qa_pairs=progress.total_qa_pairs,
            failed_chunks=len(progress.failed_chunks),
        )

        return progress

    async def get_qa_stats(self, series: str = "wheel_of_time") -> dict[str, Any]:
        """Get statistics about generated QA pairs.

        Args:
            series: Series identifier

        Returns:
            Dictionary with QA statistics
        """
        # Overall stats
        pipeline = [
            {"$match": {"metadata.series": series}},
            {
                "$group": {
                    "_id": None,
                    "total_count": {"$sum": 1},
                    "unique_chunks": {"$addToSet": "$metadata.source_chunk_id"},
                }
            },
        ]

        cursor = self.mongodb.db.wot_qna.aggregate(pipeline)
        results = await cursor.to_list(length=1)

        if not results:
            return {
                "series": series,
                "total_qa_pairs": 0,
                "unique_source_chunks": 0,
                "by_category": {},
                "by_complexity": {},
            }

        result = results[0]

        # Count by category
        category_pipeline = [
            {"$match": {"metadata.series": series}},
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        ]
        cursor = self.mongodb.db.wot_qna.aggregate(category_pipeline)
        category_results = await cursor.to_list(length=None)
        category_counts = {r["_id"]: r["count"] for r in category_results}

        # Count by complexity
        complexity_pipeline = [
            {"$match": {"metadata.series": series}},
            {"$group": {"_id": "$complexity", "count": {"$sum": 1}}},
        ]
        cursor = self.mongodb.db.wot_qna.aggregate(complexity_pipeline)
        complexity_results = await cursor.to_list(length=None)
        complexity_counts = {r["_id"]: r["count"] for r in complexity_results}

        return {
            "series": series,
            "total_qa_pairs": result["total_count"],
            "unique_source_chunks": len(result["unique_chunks"]),
            "by_category": category_counts,
            "by_complexity": complexity_counts,
            "rate_limiter_status": self.rate_limiter.get_status(),
        }

    async def close(self) -> None:
        """Close service connections."""
        if self.mongodb:
            await self.mongodb.disconnect()
        self.log.info("service_closed")

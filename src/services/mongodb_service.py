# Dependencies:
# pip install motor pymongo structlog

"""
MongoDB Atlas service for Fantasy RAG Lab.

This module provides async MongoDB operations using the Motor driver for storing
and retrieving processed chunks, extraction results, schemas, and embedding caches.

Collections:
- processed_chunks: Processed book text chunks from EPUB files
- extraction_results: Entity/relationship extractions from LLM
- schemas: Ontology schemas per series
- embedding_cache: Cached embeddings for text chunks
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from src.core.config import settings


logger = structlog.get_logger(__name__)


class MongoDBService:
    """Async MongoDB service for Fantasy RAG Lab data storage.

    This service provides CRUD operations for all data collections used in the
    ETL and RAG pipelines. It uses Motor for async operations and supports
    MongoDB Atlas deployments.

    Collections:
    - processed_chunks: Text chunks from processed books
    - extraction_results: Entity/relationship extractions
    - schemas: Ontology schemas per series
    - embedding_cache: Cached text embeddings

    Attributes:
        client: The Motor async MongoDB client.
        db: The database instance.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
    ):
        """Initialize the MongoDB service.

        Args:
            uri: MongoDB connection URI. Defaults to settings.MONGODB_URI.
            database: Database name. Defaults to settings.MONGODB_DATABASE.

        Raises:
            ValueError: If URI is not provided and not set in settings.
        """
        self.uri = uri or settings.MONGODB_URI or os.getenv("MONGODB_URI")
        self.database_name = (
            database or settings.MONGODB_DATABASE or os.getenv("MONGODB_DATABASE", "fantasy_rag")
        )

        if not self.uri:
            raise ValueError(
                "MongoDB URI not provided. Set MONGODB_URI environment variable."
            )

        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.log = logger.bind(component="MongoDBService", database=self.database_name)

    async def connect(self) -> None:
        """Establish connection to MongoDB.

        This creates the client and database references. The actual connection
        is established lazily on first operation.
        """
        self.log.info("connecting_to_mongodb")
        self.client = AsyncIOMotorClient(self.uri)
        self.db = self.client[self.database_name]

        # Verify connection by pinging the server
        try:
            await self.client.admin.command("ping")
            self.log.info("mongodb_connected")
        except PyMongoError as e:
            self.log.error("mongodb_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.log.info("mongodb_disconnected")

    # =========================================================================
    # PROCESSED CHUNKS COLLECTION
    # =========================================================================

    async def upsert_chunk(self, chunk: dict[str, Any]) -> str:
        """Insert or update a processed chunk.

        Args:
            chunk: Chunk document with required fields:
                - chunk_id: Unique identifier
                - series: Series name
                - text_content: The chunk text
                - token_count: Number of tokens
                - included_books: List of book names

        Returns:
            The chunk_id of the upserted document.

        Raises:
            PyMongoError: If the operation fails.
        """
        log = self.log.bind(chunk_id=chunk.get("chunk_id"))

        # Add timestamps
        chunk["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in chunk:
            chunk["created_at"] = chunk["updated_at"]

        try:
            result = await self.db.processed_chunks.update_one(
                {"chunk_id": chunk["chunk_id"]},
                {"$set": chunk},
                upsert=True,
            )

            if result.upserted_id:
                log.info("chunk_inserted")
            else:
                log.info("chunk_updated")

            return chunk["chunk_id"]

        except PyMongoError as e:
            log.error("chunk_upsert_failed", error=str(e))
            raise

    async def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a processed chunk by ID.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            The chunk document or None if not found.
        """
        return await self.db.processed_chunks.find_one({"chunk_id": chunk_id})

    async def get_chunks_by_series(self, series: str) -> list[dict[str, Any]]:
        """Retrieve all chunks for a series.

        Args:
            series: The series identifier.

        Returns:
            List of chunk documents sorted by chunk_id.
        """
        cursor = self.db.processed_chunks.find({"series": series}).sort("chunk_id", 1)
        return await cursor.to_list(length=None)

    async def bulk_upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Bulk upsert multiple chunks.

        Args:
            chunks: List of chunk documents.

        Returns:
            Number of documents upserted/modified.
        """
        if not chunks:
            return 0

        now = datetime.now(timezone.utc)
        operations = []

        for chunk in chunks:
            chunk["updated_at"] = now
            if "created_at" not in chunk:
                chunk["created_at"] = now

            operations.append(
                UpdateOne(
                    {"chunk_id": chunk["chunk_id"]},
                    {"$set": chunk},
                    upsert=True,
                )
            )

        result = await self.db.processed_chunks.bulk_write(operations)
        self.log.info(
            "bulk_upsert_complete",
            inserted=result.upserted_count,
            modified=result.modified_count,
        )
        return result.upserted_count + result.modified_count

    # =========================================================================
    # EXTRACTION RESULTS COLLECTION
    # =========================================================================

    async def upsert_extraction(self, extraction: dict[str, Any]) -> str:
        """Insert or update an extraction result.

        Args:
            extraction: Extraction document with required fields:
                - chunk_id: Source chunk identifier
                - series: Series name
                - entities: List of extracted entities
                - relationships: List of extracted relationships

        Returns:
            The chunk_id of the upserted document.
        """
        log = self.log.bind(chunk_id=extraction.get("chunk_id"))

        extraction["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in extraction:
            extraction["created_at"] = extraction["updated_at"]

        try:
            result = await self.db.extraction_results.update_one(
                {"chunk_id": extraction["chunk_id"]},
                {"$set": extraction},
                upsert=True,
            )

            if result.upserted_id:
                log.info("extraction_inserted")
            else:
                log.info("extraction_updated")

            return extraction["chunk_id"]

        except PyMongoError as e:
            log.error("extraction_upsert_failed", error=str(e))
            raise

    async def get_extraction(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve an extraction result by chunk ID.

        Args:
            chunk_id: The source chunk identifier.

        Returns:
            The extraction document or None if not found.
        """
        return await self.db.extraction_results.find_one({"chunk_id": chunk_id})

    async def get_extractions_by_series(self, series: str) -> list[dict[str, Any]]:
        """Retrieve all extractions for a series.

        Args:
            series: The series identifier.

        Returns:
            List of extraction documents.
        """
        cursor = self.db.extraction_results.find({"series": series})
        return await cursor.to_list(length=None)

    # =========================================================================
    # SCHEMAS COLLECTION
    # =========================================================================

    async def upsert_schema(self, schema: dict[str, Any]) -> str:
        """Insert or update an ontology schema.

        Args:
            schema: Schema document with required fields:
                - series: Series name
                - entity_types: List of entity type definitions
                - relationship_types: List of relationship type definitions

        Returns:
            The series name of the upserted document.
        """
        log = self.log.bind(series=schema.get("series"))

        schema["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in schema:
            schema["created_at"] = schema["updated_at"]

        try:
            result = await self.db.schemas.update_one(
                {"series": schema["series"]},
                {"$set": schema},
                upsert=True,
            )

            if result.upserted_id:
                log.info("schema_inserted")
            else:
                log.info("schema_updated")

            return schema["series"]

        except PyMongoError as e:
            log.error("schema_upsert_failed", error=str(e))
            raise

    async def get_schema(self, series: str) -> Optional[dict[str, Any]]:
        """Retrieve a schema by series name.

        Args:
            series: The series identifier.

        Returns:
            The schema document or None if not found.
        """
        return await self.db.schemas.find_one({"series": series})

    async def list_schemas(self) -> list[dict[str, Any]]:
        """List all available schemas.

        Returns:
            List of schema documents.
        """
        cursor = self.db.schemas.find({})
        return await cursor.to_list(length=None)

    # =========================================================================
    # EMBEDDING CACHE COLLECTION
    # =========================================================================

    async def cache_embedding(
        self,
        text_hash: str,
        embedding: list[float],
        model: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Cache an embedding for a text hash.

        Args:
            text_hash: Hash of the source text.
            embedding: The embedding vector.
            model: Model used to generate the embedding.
            metadata: Optional additional metadata.

        Returns:
            The text_hash of the cached embedding.
        """
        doc = {
            "text_hash": text_hash,
            "embedding": embedding,
            "model": model,
            "metadata": metadata or {},
            "updated_at": datetime.now(timezone.utc),
        }

        await self.db.embedding_cache.update_one(
            {"text_hash": text_hash, "model": model},
            {"$set": doc},
            upsert=True,
        )

        return text_hash

    async def get_cached_embedding(
        self, text_hash: str, model: str
    ) -> Optional[list[float]]:
        """Retrieve a cached embedding.

        Args:
            text_hash: Hash of the source text.
            model: Model used to generate the embedding.

        Returns:
            The embedding vector or None if not cached.
        """
        doc = await self.db.embedding_cache.find_one(
            {"text_hash": text_hash, "model": model}
        )
        return doc["embedding"] if doc else None

    async def bulk_cache_embeddings(
        self,
        items: list[dict[str, Any]],
        model: str,
    ) -> int:
        """Bulk cache multiple embeddings.

        Args:
            items: List of dicts with 'text_hash' and 'embedding' keys.
            model: Model used to generate the embeddings.

        Returns:
            Number of documents upserted.
        """
        if not items:
            return 0

        now = datetime.now(timezone.utc)
        operations = []

        for item in items:
            doc = {
                "text_hash": item["text_hash"],
                "embedding": item["embedding"],
                "model": model,
                "metadata": item.get("metadata", {}),
                "updated_at": now,
            }
            operations.append(
                UpdateOne(
                    {"text_hash": item["text_hash"], "model": model},
                    {"$set": doc},
                    upsert=True,
                )
            )

        result = await self.db.embedding_cache.bulk_write(operations)
        return result.upserted_count + result.modified_count

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def get_collection_stats(self) -> dict[str, int]:
        """Get document counts for all collections.

        Returns:
            Dictionary with collection names and document counts.
        """
        stats = {}
        for collection_name in [
            "processed_chunks",
            "extraction_results",
            "schemas",
            "embedding_cache",
        ]:
            count = await self.db[collection_name].count_documents({})
            stats[collection_name] = count

        self.log.info("collection_stats", stats=stats)
        return stats

    async def create_indexes(self) -> None:
        """Create indexes for efficient querying.

        This should be called once during application setup.
        """
        self.log.info("creating_indexes")

        # processed_chunks indexes
        await self.db.processed_chunks.create_index("chunk_id", unique=True)
        await self.db.processed_chunks.create_index("series")

        # extraction_results indexes
        await self.db.extraction_results.create_index("chunk_id", unique=True)
        await self.db.extraction_results.create_index("series")

        # schemas indexes
        await self.db.schemas.create_index("series", unique=True)

        # embedding_cache indexes
        await self.db.embedding_cache.create_index(
            [("text_hash", 1), ("model", 1)], unique=True
        )

        self.log.info("indexes_created")


# Singleton instance for convenience
_mongodb_service: Optional[MongoDBService] = None


async def get_mongodb_service() -> MongoDBService:
    """Get or create the MongoDB service singleton.

    Returns:
        The MongoDB service instance.
    """
    global _mongodb_service
    if _mongodb_service is None:
        _mongodb_service = MongoDBService()
        await _mongodb_service.connect()
    return _mongodb_service

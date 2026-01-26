# Dependencies:
# pip install motor pymongo structlog

"""
MongoDB Atlas service for Fantasy RAG Lab.

This module provides async MongoDB operations using the Motor driver for storing
and retrieving chunks, extraction results, schemas, and embedding caches.

Collections:
- graph_chunks: Large chunks for LLM entity/relationship extraction (100k-900k tokens)
- rag_chunks: Small chunks for vector search and RAG (1000 chars)
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
    - graph_chunks: Large chunks for entity extraction (from epubs_to_chunks.py)
    - rag_chunks: Small chunks for semantic search (from processor.py/ingest.py)
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
            database
            or settings.MONGODB_DATABASE
            or os.getenv("MONGODB_DATABASE", "fantasy_rag")
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
    # GRAPH CHUNKS COLLECTION (Large chunks for entity extraction)
    # =========================================================================

    async def upsert_graph_chunk(self, chunk: dict[str, Any]) -> str:
        """Insert or update a graph extraction chunk.

        These are large chunks (100k-900k tokens) used for LLM entity extraction.

        Args:
            chunk: Chunk document with required fields:
                - chunk_id: Unique identifier (e.g., "wheel_of_time_section_01")
                - series: Series name
                - text_content: The chunk text
                - token_count: Number of tokens
                - context_window_used: Context window size used
                - included_books: List of book names

        Returns:
            The chunk_id of the upserted document.
        """
        log = self.log.bind(chunk_id=chunk.get("chunk_id"), collection="graph_chunks")

        chunk["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in chunk:
            chunk["created_at"] = chunk["updated_at"]

        try:
            result = await self.db.graph_chunks.update_one(
                {"chunk_id": chunk["chunk_id"]},
                {"$set": chunk},
                upsert=True,
            )

            if result.upserted_id:
                log.info("graph_chunk_inserted")
            else:
                log.info("graph_chunk_updated")

            return chunk["chunk_id"]

        except PyMongoError as e:
            log.error("graph_chunk_upsert_failed", error=str(e))
            raise

    async def get_graph_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a graph chunk by ID.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            The chunk document or None if not found.
        """
        return await self.db.graph_chunks.find_one({"chunk_id": chunk_id})

    async def get_graph_chunks_by_series(self, series: str) -> list[dict[str, Any]]:
        """Retrieve all graph chunks for a series.

        Args:
            series: The series identifier.

        Returns:
            List of chunk documents sorted by chunk_id.
        """
        cursor = self.db.graph_chunks.find({"series": series}).sort("chunk_id", 1)
        return await cursor.to_list(length=None)

    async def bulk_upsert_graph_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Bulk upsert multiple graph chunks.

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

        result = await self.db.graph_chunks.bulk_write(operations)
        self.log.info(
            "graph_chunks_bulk_upsert_complete",
            inserted=result.upserted_count,
            modified=result.modified_count,
        )
        return result.upserted_count + result.modified_count

    async def delete_graph_chunks_by_series(self, series: str) -> int:
        """Delete all graph chunks for a series.

        Args:
            series: The series identifier.

        Returns:
            Number of documents deleted.
        """
        result = await self.db.graph_chunks.delete_many({"series": series})
        self.log.info("graph_chunks_deleted", series=series, count=result.deleted_count)
        return result.deleted_count

    # =========================================================================
    # RAG CHUNKS COLLECTION (Small chunks for vector search)
    # =========================================================================

    async def upsert_rag_chunk(self, chunk: dict[str, Any]) -> str:
        """Insert or update a RAG chunk.

        These are small chunks (~1000 chars) used for vector search and RAG.

        Args:
            chunk: Chunk document with required fields:
                - chunk_id: Unique identifier
                - series: Series name (universe)
                - text_content: The chunk text
                - book_name: Source book
                - chapter_number: Chapter number
                - chapter_title: Chapter title

        Returns:
            The chunk_id of the upserted document.
        """
        log = self.log.bind(chunk_id=chunk.get("chunk_id"), collection="rag_chunks")

        chunk["updated_at"] = datetime.now(timezone.utc)
        if "created_at" not in chunk:
            chunk["created_at"] = chunk["updated_at"]

        try:
            result = await self.db.rag_chunks.update_one(
                {"chunk_id": chunk["chunk_id"]},
                {"$set": chunk},
                upsert=True,
            )

            if result.upserted_id:
                log.info("rag_chunk_inserted")
            else:
                log.info("rag_chunk_updated")

            return chunk["chunk_id"]

        except PyMongoError as e:
            log.error("rag_chunk_upsert_failed", error=str(e))
            raise

    async def get_rag_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a RAG chunk by ID.

        Args:
            chunk_id: The unique chunk identifier.

        Returns:
            The chunk document or None if not found.
        """
        return await self.db.rag_chunks.find_one({"chunk_id": chunk_id})

    async def get_rag_chunks_by_series(self, series: str) -> list[dict[str, Any]]:
        """Retrieve all RAG chunks for a series.

        Args:
            series: The series identifier.

        Returns:
            List of chunk documents sorted by chunk_id.
        """
        cursor = self.db.rag_chunks.find({"series": series}).sort("chunk_id", 1)
        return await cursor.to_list(length=None)

    async def get_rag_chunks_by_book(
        self, series: str, book_name: str
    ) -> list[dict[str, Any]]:
        """Retrieve all RAG chunks for a specific book.

        Args:
            series: The series identifier.
            book_name: The book name.

        Returns:
            List of chunk documents.
        """
        cursor = self.db.rag_chunks.find(
            {"series": series, "book_name": book_name}
        ).sort("chunk_id", 1)
        return await cursor.to_list(length=None)

    async def bulk_upsert_rag_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Bulk upsert multiple RAG chunks.

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

        result = await self.db.rag_chunks.bulk_write(operations)
        self.log.info(
            "rag_chunks_bulk_upsert_complete",
            inserted=result.upserted_count,
            modified=result.modified_count,
        )
        return result.upserted_count + result.modified_count

    async def delete_rag_chunks_by_series(self, series: str) -> int:
        """Delete all RAG chunks for a series.

        Args:
            series: The series identifier.

        Returns:
            Number of documents deleted.
        """
        result = await self.db.rag_chunks.delete_many({"series": series})
        self.log.info("rag_chunks_deleted", series=series, count=result.deleted_count)
        return result.deleted_count

    # =========================================================================
    # EXTRACTION RESULTS COLLECTION
    # =========================================================================

    async def upsert_extraction(self, extraction: dict[str, Any]) -> str:
        """Insert or update an extraction result.

        Args:
            extraction: Extraction document with required fields:
                - chunk_id: Source chunk identifier (from graph_chunks)
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
            "graph_chunks",
            "rag_chunks",
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

        # graph_chunks indexes
        await self.db.graph_chunks.create_index("chunk_id", unique=True)
        await self.db.graph_chunks.create_index("series")

        # rag_chunks indexes
        await self.db.rag_chunks.create_index("chunk_id", unique=True)
        await self.db.rag_chunks.create_index("series")
        await self.db.rag_chunks.create_index([("series", 1), ("book_name", 1)])

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

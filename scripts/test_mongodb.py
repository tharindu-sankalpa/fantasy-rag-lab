#!/usr/bin/env python3
"""Quick test script for MongoDB service."""

import asyncio
import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger()


async def test_mongodb_connection():
    """Test MongoDB connection and basic operations."""
    from src.services.mongodb_service import MongoDBService

    logger.info("testing_mongodb_connection")

    # Initialize and connect
    mongodb = MongoDBService()
    await mongodb.connect()

    # Create indexes
    logger.info("creating_indexes")
    await mongodb.create_indexes()

    # Test upsert a sample chunk
    logger.info("testing_chunk_upsert")
    test_chunk = {
        "chunk_id": "test_chunk_001",
        "series": "test_series",
        "text_content": "This is a test chunk for MongoDB verification.",
        "token_count": 10,
        "included_books": ["test_book.epub"],
    }
    await mongodb.upsert_chunk(test_chunk)

    # Retrieve it back
    retrieved = await mongodb.get_chunk("test_chunk_001")
    logger.info("chunk_retrieved", chunk_id=retrieved["chunk_id"])

    # Get collection stats
    stats = await mongodb.get_collection_stats()
    logger.info("collection_stats", stats=stats)

    # Clean up test data
    await mongodb.db.processed_chunks.delete_one({"chunk_id": "test_chunk_001"})
    logger.info("test_data_cleaned_up")

    await mongodb.disconnect()
    return True


async def main():
    """Run MongoDB tests."""
    logger.info("starting_mongodb_tests")

    try:
        await test_mongodb_connection()
        logger.info("mongodb_tests_passed")
    except Exception as e:
        logger.exception("test_failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())

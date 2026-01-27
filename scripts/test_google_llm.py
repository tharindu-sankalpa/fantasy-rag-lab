#!/usr/bin/env python3
"""Quick test script for Google LLM service."""

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


async def test_text_generation():
    """Test basic text generation."""
    from src.services.llm import UnifiedLLMService

    logger.info("testing_text_generation")
    service = UnifiedLLMService()

    response = await service.generate_text(
        prompt="What is the capital of France? Answer in one word.",
        model="gemini-2.0-flash",
        max_tokens=50,
    )

    logger.info(
        "text_generation_result",
        content=response.content[:100],
        tokens=response.usage.total_tokens,
    )
    return True


async def test_embeddings():
    """Test embedding generation."""
    from src.services.llm import UnifiedLLMService

    logger.info("testing_embeddings")
    service = UnifiedLLMService()

    result = await service.generate_embeddings(
        texts=["Hello world", "Machine learning is fascinating"],
        model="gemini-embedding-001",
    )

    embeddings = result["embeddings"]
    logger.info(
        "embeddings_result",
        num_embeddings=len(embeddings),
        embedding_dim=len(embeddings[0]) if embeddings else 0,
    )
    return True


async def main():
    """Run all tests."""
    logger.info("starting_google_llm_tests")

    try:
        await test_text_generation()
        await test_embeddings()
        logger.info("all_tests_passed")
    except Exception as e:
        logger.exception("test_failed", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())

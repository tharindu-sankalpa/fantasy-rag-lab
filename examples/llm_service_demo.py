# Dependencies:
# pip install anthropic openai google-genai voyageai structlog pydantic tenacity

"""
Demonstration of the unified LLM service.

This example shows how to use the refactored LLM service to:
1. Generate text with different providers
2. Use fallback chains for resilience
3. Generate structured outputs
4. Create embeddings
5. Track usage and costs

Before running:
1. Set environment variables for the providers you want to use:
   - ANTHROPIC_API_KEY
   - OPENAI_API_KEY
   - GOOGLE_API_KEY
   - OPENROUTER_API_KEY
   - VOYAGE_API_KEY

2. You only need keys for providers you actually want to use.
   The service will work with whatever providers are available.
"""

import asyncio
from pydantic import BaseModel

import structlog

# Configure structlog for this demo
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

# Import the unified service
from src.services.llm import UnifiedLLMService


# ============================================================================
# EXAMPLE 1: Simple text generation
# ============================================================================


async def example_text_generation():
    """
    Demonstrate basic text generation with different providers.
    """
    log = logger.bind(example="text_generation")
    log.info("starting_example")

    # Initialize service (loads all available providers)
    service = UnifiedLLMService()

    # Example 1a: Anthropic/Claude
    log.info("testing_anthropic")
    try:
        response = await service.generate_text(
            prompt="Explain what a Large Language Model is in one sentence.",
            provider="anthropic",
            model="claude-opus-4-5-20251101",
            max_tokens=150,
        )
        log.info(
            "anthropic_response",
            content=response.content[:100] + "...",
            tokens=response.usage.total_tokens,
        )
    except Exception as e:
        log.error("anthropic_example_failed", error=str(e))

    # Example 1b: OpenAI/GPT
    log.info("testing_openai")
    try:
        response = await service.generate_text(
            prompt="What is the speed of light?",
            provider="openai",
            model="gpt-5.2-pro",
            max_tokens=100,
        )
        log.info(
            "openai_response",
            content=response.content[:100],
            tokens=response.usage.total_tokens,
        )
    except Exception as e:
        log.error("openai_example_failed", error=str(e))

    # Example 1c: Google Gemini
    log.info("testing_google")
    try:
        response = await service.generate_text(
            prompt="Write a haiku about Python programming.",
            provider="google",
            model="gemini-3-pro-preview",
            max_tokens=100,
        )
        log.info(
            "google_response",
            content=response.content,
            tokens=response.usage.total_tokens,
        )
    except Exception as e:
        log.error("google_example_failed", error=str(e))


# ============================================================================
# EXAMPLE 2: Fallback chains for resilience
# ============================================================================


async def example_fallback_chains():
    """
    Demonstrate automatic fallback to alternative providers.
    """
    log = logger.bind(example="fallback_chains")
    log.info("starting_example")

    service = UnifiedLLMService()

    # Try primary provider with fallbacks
    log.info("testing_fallback_chain")
    try:
        response = await service.generate_text(
            prompt="What is machine learning?",
            provider="google",  # Try Google first
            model="gemini-3-pro-preview",
            # If Google fails, try these alternatives in order
            fallback_chain=[
                "openrouter/google/gemini-3-pro-preview",
                "openai/gpt-5.2-pro",
                "anthropic/claude-opus-4-5-20251101",
            ],
            max_tokens=200,
        )

        log.info(
            "fallback_success",
            provider_used=response.usage.provider,
            model_used=response.usage.model,
            content=response.content[:100],
        )
    except Exception as e:
        log.error("fallback_example_failed", error=str(e))


# ============================================================================
# EXAMPLE 3: Structured outputs
# ============================================================================


async def example_structured_outputs():
    """
    Demonstrate schema-constrained generation with Pydantic models.
    """
    log = logger.bind(example="structured_outputs")
    log.info("starting_example")

    # Define output schema
    class PersonInfo(BaseModel):
        """Schema for extracted person information."""

        name: str
        age: int
        occupation: str
        location: str

    service = UnifiedLLMService()

    # Example 3a: OpenAI (best structured output support)
    log.info("testing_openai_structured")
    try:
        result = await service.generate_structured(
            prompt="Extract information: Alice Johnson is a 32-year-old data scientist working in San Francisco.",
            schema=PersonInfo,
            provider="openai",
            model="gpt-5.2-pro",
        )

        log.info(
            "structured_output_success",
            parsed_data=result["parsed"].model_dump(),
            tokens=result["usage"].total_tokens,
        )
    except Exception as e:
        log.error("openai_structured_failed", error=str(e))

    # Example 3b: Anthropic (via tool calling)
    log.info("testing_anthropic_structured")
    try:
        result = await service.generate_structured(
            prompt="Extract information: Bob Smith, 45, architect, lives in New York.",
            schema=PersonInfo,
            provider="anthropic",
            model="claude-opus-4-5-20251101",
        )

        log.info(
            "structured_output_success",
            parsed_data=result["parsed"].model_dump(),
            tokens=result["usage"].total_tokens,
        )
    except Exception as e:
        log.error("anthropic_structured_failed", error=str(e))


# ============================================================================
# EXAMPLE 4: Embeddings for semantic search
# ============================================================================


async def example_embeddings():
    """
    Demonstrate embedding generation with different providers.
    """
    log = logger.bind(example="embeddings")
    log.info("starting_example")

    service = UnifiedLLMService()

    # Example 4a: Voyage AI (best for retrieval)
    log.info("testing_voyage_embeddings")
    try:
        # Embed documents for indexing
        doc_result = await service.generate_embeddings(
            texts=["Python is a programming language.", "Machine learning uses data."],
            provider="voyage",
            model="voyage-3-large",
            input_type="document",  # Optimize for document embeddings
        )

        log.info(
            "voyage_document_embeddings_generated",
            num_embeddings=len(doc_result["embeddings"]),
            embedding_dim=len(doc_result["embeddings"][0]),
            tokens=doc_result["usage"].total_tokens,
        )

        # Embed query for search
        query_result = await service.generate_embeddings(
            texts="programming language",
            provider="voyage",
            model="voyage-3-large",
            input_type="query",  # Optimize for query embeddings
        )

        log.info(
            "voyage_query_embedding_generated",
            embedding_dim=len(query_result["embeddings"]),
            tokens=query_result["usage"].total_tokens,
        )
    except Exception as e:
        log.error("voyage_embeddings_failed", error=str(e))

    # Example 4b: OpenAI embeddings
    log.info("testing_openai_embeddings")
    try:
        result = await service.generate_embeddings(
            texts=["hello world", "goodbye world"],
            provider="openai",
            model="text-embedding-3-large",
        )

        log.info(
            "openai_embeddings_generated",
            num_embeddings=len(result["embeddings"]),
            embedding_dim=len(result["embeddings"][0]),
            tokens=result["usage"].total_tokens,
        )
    except Exception as e:
        log.error("openai_embeddings_failed", error=str(e))


# ============================================================================
# EXAMPLE 5: OpenRouter with cost tracking
# ============================================================================


async def example_openrouter_cost_tracking():
    """
    Demonstrate OpenRouter's direct cost reporting.
    """
    log = logger.bind(example="openrouter_cost_tracking")
    log.info("starting_example")

    service = UnifiedLLMService()

    # OpenRouter provides direct USD cost in responses
    log.info("testing_openrouter_with_cost")
    try:
        response = await service.generate_text(
            prompt="Write a short poem about artificial intelligence.",
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
            max_tokens=150,
        )

        log.info(
            "openrouter_response",
            content=response.content,
            tokens=response.usage.total_tokens,
            # OpenRouter's key feature: direct cost reporting
            input_cost_usd=response.usage.input_cost_usd,
            output_cost_usd=response.usage.output_cost_usd,
            total_cost_usd=response.usage.total_cost_usd,
        )
    except Exception as e:
        log.error("openrouter_example_failed", error=str(e))


# ============================================================================
# EXAMPLE 6: Usage tracking and cost monitoring
# ============================================================================


async def example_usage_tracking():
    """
    Demonstrate usage tracking across all requests.
    """
    log = logger.bind(example="usage_tracking")
    log.info("starting_example")

    service = UnifiedLLMService()

    # Make several requests
    log.info("making_multiple_requests")

    try:
        await service.generate_text(
            prompt="Test 1", provider="openai", model="gpt-5.2-pro", max_tokens=50
        )
    except:
        pass

    try:
        await service.generate_text(
            prompt="Test 2", provider="anthropic", model="claude-opus-4-5-20251101", max_tokens=50
        )
    except:
        pass

    try:
        await service.generate_embeddings(
            texts=["test"], provider="voyage", model="voyage-3-large"
        )
    except:
        pass

    # Get usage summary
    summary = service.get_usage_summary()

    log.info(
        "usage_summary",
        total_requests=summary["total_requests"],
        total_tokens=summary["total_tokens"],
        total_cost_usd=summary["total_cost_usd"],
    )

    # Provider breakdown
    log.info("usage_by_provider")
    for provider, stats in summary["by_provider"].items():
        log.info(
            "provider_stats",
            provider=provider,
            requests=stats["requests"],
            tokens=stats["tokens"],
            cost_usd=stats["cost_usd"],
        )

    # Model breakdown
    log.info("usage_by_model")
    for model, stats in summary["by_model"].items():
        log.info(
            "model_stats",
            model=model,
            requests=stats["requests"],
            tokens=stats["tokens"],
            cost_usd=stats["cost_usd"],
        )


# ============================================================================
# MAIN RUNNER
# ============================================================================


async def main():
    """
    Run all examples.
    """
    log = logger.bind(component="main")
    log.info("starting_llm_service_demo")

    # Run examples one by one
    await example_text_generation()
    await example_fallback_chains()
    await example_structured_outputs()
    await example_embeddings()
    await example_openrouter_cost_tracking()
    await example_usage_tracking()

    log.info("demo_completed")


if __name__ == "__main__":
    asyncio.run(main())

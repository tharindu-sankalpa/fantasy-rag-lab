"""
Unified LLM service package.

This package provides a production-grade, modular LLM service with support for
multiple providers (Anthropic, OpenAI, Google, OpenRouter, Voyage).

Architecture:
- Thin abstractions tailored to actual use cases (not over-engineered)
- Official SDKs used underneath for reliability
- Provider-specific features accessible through kwargs
- Comprehensive usage tracking and cost monitoring
- Automatic fallback chains for resilience
- Batch API support for cost optimization (50% discount)

Main Entry Point:
    UnifiedLLMService - The only class you need to import for all LLM operations

Example:
    from src.services.llm import UnifiedLLMService

    service = UnifiedLLMService()

    # Text generation
    response = await service.generate_text(
        prompt="Hello",
        provider="anthropic",
        model="claude-sonnet-4-20250514"
    )

    # Structured output
    result = await service.generate_structured(
        prompt="Extract info",
        schema=MySchema,
        provider="openai",
        model="gpt-4o"
    )

    # Embeddings
    embeddings = await service.generate_embeddings(
        texts=["doc1", "doc2"],
        provider="voyage",
        model="voyage-3-large"
    )

    # Batch processing (50% cost reduction)
    from src.services.llm import AnthropicBatchProvider
    batch_provider = AnthropicBatchProvider(api_key="...")
    job = await batch_provider.create_batch(requests, output_dir, model)
    await batch_provider.poll_until_complete(job.id)
    results = await batch_provider.process_results_to_files(job.id, schema)

Module Structure:
- base: Foundation (enums, data classes, abstract interfaces)
- anthropic_provider: Anthropic/Claude real-time implementation
- anthropic_batch_provider: Anthropic/Claude batch API (50% discount)
- batch_models: Data models for batch operations
- openai_provider: OpenAI/GPT implementation
- google_provider: Google/Gemini implementation
- openrouter_provider: OpenRouter multi-provider gateway
- voyage_provider: Voyage AI embeddings
- service: Unified orchestrator (main entry point)
"""

# Main service (primary export)
from .service import UnifiedLLMService

# Base classes and types (useful for type hints and advanced use)
from .base import (
    BaseLLMProvider,
    GenerationResponse,
    UsageMetrics,
    ProviderType,
    EndpointType,
)

# Individual providers (for advanced use cases where direct access is needed)
from .anthropic_provider import AnthropicProvider
from .anthropic_batch_provider import AnthropicBatchProvider, build_extraction_batch_request
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .openrouter_provider import OpenRouterProvider
from .voyage_provider import VoyageProvider

# Batch API models (for batch processing workflows)
from .batch_models import (
    BatchJob,
    BatchRequestItem,
    BatchRequestParams,
    BatchResultItem,
    BatchProcessingStatus,
    BatchResultType,
)

# Define public API
__all__ = [
    # Main service (most users only need this)
    "UnifiedLLMService",
    # Base classes and types
    "BaseLLMProvider",
    "GenerationResponse",
    "UsageMetrics",
    "ProviderType",
    "EndpointType",
    # Individual providers (for advanced use)
    "AnthropicProvider",
    "AnthropicBatchProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "VoyageProvider",
    # Batch API models and utilities
    "BatchJob",
    "BatchRequestItem",
    "BatchRequestParams",
    "BatchResultItem",
    "BatchProcessingStatus",
    "BatchResultType",
    "build_extraction_batch_request",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Fantasy RAG Lab"
__description__ = "Production-grade unified LLM service with multi-provider support"

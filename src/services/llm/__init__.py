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

Module Structure:
- base: Foundation (enums, data classes, abstract interfaces)
- anthropic_provider: Anthropic/Claude implementation
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
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .openrouter_provider import OpenRouterProvider
from .voyage_provider import VoyageProvider

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
    "OpenAIProvider",
    "GoogleProvider",
    "OpenRouterProvider",
    "VoyageProvider",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Fantasy RAG Lab"
__description__ = "Production-grade unified LLM service with multi-provider support"

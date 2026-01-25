"""
Unified LLM service package (Google-only).

This package provides a production-grade LLM service using Google's Gemini models
via both the Developer API and Vertex AI.

Architecture:
- Thin abstractions tailored to actual use cases (not over-engineered)
- Official google-genai SDK used underneath for reliability
- Comprehensive usage tracking
- Automatic fallback chains for resilience

Main Entry Point:
    UnifiedLLMService - The only class you need to import for all LLM operations

Example:
    from src.services.llm import UnifiedLLMService

    service = UnifiedLLMService()

    # Text generation
    response = await service.generate_text(
        prompt="Hello",
        model="gemini-3-pro-preview"
    )

    # Structured output
    result = await service.generate_structured(
        prompt="Extract info",
        schema=MySchema,
        model="gemini-3-pro-preview"
    )

    # Embeddings
    embeddings = await service.generate_embeddings(
        texts=["doc1", "doc2"],
        model="text-embedding-004"
    )

Module Structure:
- base: Foundation (enums, data classes, abstract interfaces)
- google_provider: Google/Gemini implementation (Developer API + Vertex AI)
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

# Google provider (for advanced use cases where direct access is needed)
from .google_provider import GoogleProvider

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
    # Google provider (for advanced use)
    "GoogleProvider",
]

# Package metadata
__version__ = "2.0.0"
__author__ = "Fantasy RAG Lab"
__description__ = "Production-grade unified LLM service with Google Gemini support"

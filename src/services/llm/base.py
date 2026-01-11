# Dependencies:
# pip install structlog pydantic tenacity

"""
Base classes, enums, and data structures for LLM service.

This module defines the foundational abstractions used across all LLM providers:
- Provider type enumeration for consistent provider identification
- Endpoint type classification for different API operations
- Usage metrics data class for unified token/cost tracking
- Generation response wrapper for text completions
- Abstract base provider interface that all concrete providers must implement

Architecture Philosophy:
- Thin abstractions tailored to actual use cases (not over-engineered)
- Unified interface while preserving provider-specific capabilities
- Comprehensive usage tracking with cost monitoring where available
- Type safety through strict typing and Pydantic validation
"""

import time
from typing import Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import structlog

# Initialize logger for this module
logger = structlog.get_logger()


# ============================================================================
# ENUMS
# ============================================================================


class ProviderType(str, Enum):
    """
    Enumeration of supported LLM providers.

    Each value corresponds to a concrete provider implementation.
    Provider names are lowercase and match common naming conventions.
    """

    ANTHROPIC = "anthropic"  # Claude models via Anthropic API
    OPENAI = "openai"  # GPT models via OpenAI API
    GOOGLE = "google"  # Gemini models via Google Gen AI SDK
    VOYAGE = "voyage"  # Voyage AI embeddings (embeddings only)
    OPENROUTER = "openrouter"  # Multi-provider gateway with unified access


class EndpointType(str, Enum):
    """
    Classification of API endpoint types.

    Different endpoints have different input/output formats and capabilities:
    - TEXT_GENERATION: Standard chat/completion endpoints
    - STRUCTURED_OUTPUT: Schema-constrained generation (JSON mode, function calling)
    - EMBEDDING: Text vectorization for semantic search/retrieval
    """

    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"
    EMBEDDING = "embedding"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class UsageMetrics:
    """
    Unified usage metrics across all LLM providers.

    This class normalizes usage data from different providers into a common format.
    Different providers report metrics differently:
    - OpenAI/Anthropic: prompt_tokens, completion_tokens
    - Google: input_token_count, output_token_count
    - OpenRouter: Provides direct USD cost (unique feature)

    For providers that don't report certain metrics (e.g., cost), those fields remain None.

    Attributes:
        input_tokens: Number of tokens in the prompt/input
        output_tokens: Number of tokens in the generated response
        total_tokens: Sum of input and output tokens
        cached_tokens: Number of tokens served from cache (prompt caching feature)
        reasoning_tokens: Extended thinking tokens (for reasoning models like o1)
        input_cost_usd: Cost of input tokens in USD (OpenRouter provides this)
        output_cost_usd: Cost of output tokens in USD
        total_cost_usd: Total cost in USD for this request
        provider: Provider name (e.g., 'anthropic', 'openai')
        model: Model identifier used for this request
        timestamp: Unix timestamp when metrics were recorded
        api_key_last4: Last 4 chars of API key (for tracking which key was used)
    """

    # Token counts (standard across all providers)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Advanced token metrics (provider-specific)
    cached_tokens: int = 0  # Prompt caching (Anthropic, OpenAI)
    reasoning_tokens: int = 0  # Extended thinking (OpenAI o1 models)

    # Cost information (OpenRouter provides this directly, others need calculation)
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None

    # Metadata for tracking and debugging
    provider: Optional[str] = None
    model: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    api_key_last4: Optional[str] = None


@dataclass
class GenerationResponse:
    """
    Response wrapper for text generation endpoints.

    This standardized response format makes it easy to work with outputs
    from different providers without provider-specific parsing logic.

    Attributes:
        content: The generated text content (main output)
        usage: Token usage and cost metrics for this generation
        raw_response: Original response object from the provider (for debugging/advanced use)
        finish_reason: Why generation stopped (e.g., 'stop', 'length', 'content_filter')
    """

    content: str  # The actual generated text
    usage: UsageMetrics  # Normalized usage metrics
    raw_response: Any = None  # Original provider response (useful for debugging)
    finish_reason: Optional[str] = None  # Reason generation stopped


# ============================================================================
# ABSTRACT BASE PROVIDER
# ============================================================================


class BaseLLMProvider(ABC):
    """
    Abstract base class defining the interface all LLM providers must implement.

    This is a THIN abstraction - it only defines methods we actually need for this
    application, not attempting to be universally compatible with every possible
    LLM feature. Provider-specific features are still accessible via kwargs.

    Design Philosophy:
    - Simple, focused interface with only the methods we use
    - Type hints everywhere for safety and IDE support
    - Consistent async patterns across all providers
    - Logging bound at provider level for context
    - Error handling delegated to concrete implementations

    Concrete providers (Anthropic, OpenAI, Google, etc.) inherit from this
    and implement these methods using their official SDKs.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize the provider with API credentials and optional configuration.

        Args:
            api_key: API key for authentication with the provider
            **kwargs: Provider-specific configuration options (e.g., base_url, timeout)
        """
        self.api_key = api_key
        # Bind logger with provider class name for context
        self.log = logger.bind(provider=self.__class__.__name__)

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text from a prompt (standard chat completion).

        This is the primary method for text generation. All providers must implement
        this to provide consistent text generation capabilities.

        Args:
            prompt: Input text prompt to generate from
            model: Model identifier (provider-specific, e.g., 'claude-sonnet-4', 'gpt-4o')
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            **kwargs: Provider-specific parameters (system prompts, stop sequences, etc.)

        Returns:
            GenerationResponse containing the generated text and usage metrics

        Raises:
            ValueError: If required parameters are missing or invalid
            httpx.HTTPStatusError: If API request fails
            Exception: For other provider-specific errors
        """
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        model: str,
        schema: Any,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured output conforming to a Pydantic schema.

        This method ensures the LLM output matches a specific structure, useful for
        extracting entities, parsing documents, or enforcing output formats.

        Implementation varies by provider:
        - OpenAI: Native structured outputs (response_format)
        - Anthropic: Tool calling with schema enforcement
        - Google: JSON mode with schema validation

        Args:
            prompt: Input text prompt describing what to extract/generate
            model: Model identifier
            schema: Pydantic BaseModel defining the expected output structure
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with keys:
            - 'parsed': Validated Pydantic model instance
            - 'usage': UsageMetrics for this request

        Raises:
            ValueError: If schema is not a Pydantic BaseModel or parsing fails
            httpx.HTTPStatusError: If API request fails
        """
        pass

    def _mask_api_key(self) -> str:
        """
        Mask API key for safe logging (show only last 4 characters).

        This prevents accidental exposure of API keys in logs while still
        allowing identification of which key was used (useful when rotating keys).

        Returns:
            Masked API key string (e.g., "...xyz1" for key ending in xyz1)
        """
        if len(self.api_key) > 4:
            return f"...{self.api_key[-4:]}"
        return "***"  # For very short keys, mask completely

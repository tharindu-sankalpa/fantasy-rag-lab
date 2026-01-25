# Dependencies:
# pip install structlog pydantic google-genai

"""
Unified LLM service orchestrator (Google-only).

This module provides the main entry point for all LLM operations using Google's
Gemini models via both the Developer API and Vertex AI.

Key Responsibilities:
1. Provider Management: Initialize Google provider instances
2. Unified Interface: Single API for text generation, structured outputs, embeddings
3. Fallback Chains: Automatic retry with alternative models on failure
4. Usage Tracking: Aggregate and report token usage across all requests

Architecture Philosophy:
- This is the ONLY class application code should interact with directly
- All provider-specific complexity is hidden behind this unified interface
- Simple, predictable API that works the same regardless of backend (Developer API vs Vertex AI)

Example Usage:
    from src.services.llm import UnifiedLLMService

    service = UnifiedLLMService()

    # Text generation
    response = await service.generate_text(
        prompt="Explain quantum computing",
        provider="google",
        model="gemini-3-pro-preview"
    )

    # Structured output
    result = await service.generate_structured(
        prompt="Extract person info",
        schema=PersonSchema,
        provider="google",
        model="gemini-3-pro-preview"
    )

    # Embeddings
    embeddings = await service.generate_embeddings(
        texts=["doc 1", "doc 2"],
        model="text-embedding-004"
    )

    # Get cost summary
    summary = service.get_usage_summary()
"""

import os
from typing import Optional, Any, Union

import structlog
from dotenv import load_dotenv

from .base import (
    BaseLLMProvider,
    GenerationResponse,
    UsageMetrics,
    ProviderType,
)
from .google_provider import GoogleProvider

# Load environment variables from .env file
load_dotenv()

# Initialize module-level logger
logger = structlog.get_logger()


class UnifiedLLMService:
    """
    Unified service orchestrating Google LLM providers.

    This is the main entry point for the application. It provides:
    - A consistent interface for Google Gemini models
    - Support for both Developer API (API key) and Vertex AI (GCP project)
    - Centralized usage tracking
    - Simple provider management based on environment variables

    The service automatically initializes providers for which credentials are available
    in the environment. If credentials are not available, the provider is skipped.

    Usage Tracking:
    All requests are automatically logged to usage_history, allowing you to:
    - Monitor token consumption
    - Identify expensive operations for optimization
    - Generate usage reports

    Fallback Chains:
    Fallback chains allow specifying alternative models to try if the primary fails.
    Format: ["model1", "model2", ...]
    Example: ["gemini-3-pro-preview", "gemini-2.0-flash"]
    """

    def __init__(self):
        """
        Initialize the unified LLM service.

        Automatically loads Google providers for which credentials are available
        in the environment. Providers without credentials are skipped (not an error).

        Environment Variables:
        - GOOGLE_API_KEY or GEMINI_API_KEY: For Google Gemini (Developer API)
        - GOOGLE_CLOUD_PROJECT: For Google Vertex AI (if GOOGLE_GENAI_USE_VERTEXAI=true)
        - GOOGLE_CLOUD_LOCATION: GCP region (default: us-central1)
        """
        self.log = logger.bind(component="UnifiedLLMService")
        self.log.info("initializing_unified_llm_service")

        # Provider instances (maps provider name -> provider instance)
        self.providers: dict[str, BaseLLMProvider] = {}

        # Usage tracking (all requests across all providers)
        self.usage_history: list[UsageMetrics] = []

        # Load providers based on available credentials
        self._load_providers()

        # Log summary of loaded providers
        provider_names = list(self.providers.keys())
        self.log.info("service_initialized", available_providers=provider_names)

    def _load_providers(self) -> None:
        """
        Load and initialize Google providers based on environment variables.

        This method checks for API keys/credentials and initializes providers accordingly.
        If credentials are not available, the provider is silently skipped (not an error).

        This approach allows flexible deployment - you only need credentials for the
        backend you actually want to use (Developer API or Vertex AI).
        """
        self.log.info("loading_providers")

        # Google Gemini (Developer API)
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            try:
                self.providers["google"] = GoogleProvider(api_key=google_key)
                self.log.info("provider_loaded", provider="google")
            except Exception as e:
                self.log.error("provider_load_failed", provider="google", error=str(e))

        # Google Vertex AI (GCP-based)
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
        if use_vertex:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if project_id:
                try:
                    self.providers["google_vertex"] = GoogleProvider(
                        use_vertex=True, project_id=project_id, location=location
                    )
                    self.log.info(
                        "provider_loaded",
                        provider="google_vertex",
                        project=project_id,
                        location=location,
                    )
                except Exception as e:
                    self.log.error(
                        "provider_load_failed", provider="google_vertex", error=str(e)
                    )

        # Warn if no providers loaded
        if not self.providers:
            self.log.warning(
                "no_providers_loaded",
                message="No Google credentials found. Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT.",
            )

    def _get_provider(self, provider: str = "google") -> GoogleProvider:
        """
        Get the Google provider instance.

        Args:
            provider: Provider name ('google' or 'google_vertex')

        Returns:
            GoogleProvider instance

        Raises:
            ValueError: If provider not available
        """
        provider_instance = self.providers.get(provider)
        if not provider_instance:
            # Try to fall back to any available provider
            if self.providers:
                fallback = next(iter(self.providers.keys()))
                self.log.warning(
                    "provider_fallback",
                    requested=provider,
                    using=fallback,
                )
                provider_instance = self.providers[fallback]
            else:
                raise ValueError(
                    f"No Google providers available. Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT."
                )
        return provider_instance

    async def generate_text(
        self,
        prompt: str,
        provider: str = "google",
        model: str = "gemini-3-pro-preview",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        fallback_chain: Optional[list[str]] = None,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using Google Gemini models.

        This is the primary method for text generation. It tries the specified model
        first, then falls back to alternatives if provided.

        Args:
            prompt: Input text prompt
            provider: Provider name ('google' or 'google_vertex')
            model: Gemini model name (default: gemini-3-pro-preview)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, higher = more creative)
            fallback_chain: List of fallback models to try if primary fails
                Example: ["gemini-2.0-flash", "gemini-1.5-pro"]
            **kwargs: Provider-specific parameters (system prompts, stop sequences, etc.)

        Returns:
            GenerationResponse with content, usage metrics, and metadata

        Raises:
            ValueError: If no providers are available or all attempts fail
            Exception: Last exception from final failed attempt

        Example:
            # With fallback chain
            response = await service.generate_text(
                prompt="What is Python?",
                model="gemini-3-pro-preview",
                fallback_chain=["gemini-2.0-flash", "gemini-1.5-pro"],
                max_tokens=200
            )

            # Simple call without fallback
            response = await service.generate_text(
                prompt="Hello",
                model="gemini-2.0-flash"
            )
        """
        log = self.log.bind(provider=provider, model=model, endpoint="generate_text")

        # Build attempt queue: [primary, *fallbacks]
        attempt_queue = [model]

        if fallback_chain:
            attempt_queue.extend(fallback_chain)

        last_exception = None

        # Get provider instance
        provider_instance = self._get_provider(provider)

        # Try each model in the queue
        for attempt_num, attempt_model in enumerate(attempt_queue, 1):
            log_attempt = log.bind(
                attempt_num=attempt_num,
                total_attempts=len(attempt_queue),
                attempt_model=attempt_model,
            )

            try:
                log_attempt.info("attempting_text_generation")

                # Call provider
                response = await provider_instance.generate_text(
                    prompt=prompt,
                    model=attempt_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )

                # Track usage
                self.usage_history.append(response.usage)

                log_attempt.info(
                    "text_generation_success",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

                return response

            except Exception as e:
                log_attempt.warning(
                    "text_generation_attempt_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                last_exception = e
                # Continue to next model in fallback chain

        # All attempts failed
        log.error(
            "all_text_generation_attempts_failed",
            attempts=len(attempt_queue),
            last_error=str(last_exception) if last_exception else None,
        )
        raise last_exception or ValueError(
            "No models available for text generation"
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        provider: str = "google",
        model: str = "gemini-3-pro-preview",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        fallback_chain: Optional[list[str]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured output with optional fallback chain.

        This method ensures LLM output matches a specific Pydantic schema, useful for:
        - Data extraction (entities, facts, structured information)
        - Form filling and structured responses
        - API-compatible outputs
        - Type-safe parsing of LLM responses

        Args:
            prompt: Input text prompt describing what to extract/generate
            schema: Pydantic BaseModel defining the output structure
            provider: Provider name ('google' or 'google_vertex')
            model: Gemini model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fallback_chain: List of fallback models
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with:
            - 'parsed': Validated Pydantic model instance
            - 'usage': UsageMetrics for this request

        Raises:
            ValueError: If schema is invalid or no providers available
            Exception: Last exception from final failed attempt

        Example:
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int
                occupation: str

            result = await service.generate_structured(
                prompt="Extract: Jane Doe, 28, software engineer",
                schema=Person,
                model="gemini-3-pro-preview"
            )
            # result['parsed'] is a validated Person instance
            # result['parsed'].name == "Jane Doe"
        """
        log = self.log.bind(
            provider=provider, model=model, endpoint="generate_structured"
        )

        # Build attempt queue
        attempt_queue = [model]

        if fallback_chain:
            attempt_queue.extend(fallback_chain)

        last_exception = None

        # Get provider instance
        provider_instance = self._get_provider(provider)

        for attempt_num, attempt_model in enumerate(attempt_queue, 1):
            log_attempt = log.bind(
                attempt_num=attempt_num,
                total_attempts=len(attempt_queue),
                attempt_model=attempt_model,
            )

            try:
                log_attempt.info(
                    "attempting_structured_generation", schema_name=schema.__name__
                )

                # Call provider
                result = await provider_instance.generate_structured(
                    prompt=prompt,
                    model=attempt_model,
                    schema=schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )

                # Track usage
                self.usage_history.append(result["usage"])

                log_attempt.info(
                    "structured_generation_success",
                    input_tokens=result["usage"].input_tokens,
                    output_tokens=result["usage"].output_tokens,
                    schema_name=schema.__name__,
                )

                return result

            except Exception as e:
                log_attempt.warning(
                    "structured_generation_attempt_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                last_exception = e
                continue

        # All attempts failed
        log.error("all_structured_generation_attempts_failed")
        raise last_exception or ValueError(
            "No models available for structured generation"
        )

    async def generate_embeddings(
        self,
        texts: Union[str, list[str]],
        model: str = "text-embedding-004",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate embeddings using Google's text-embedding models.

        Embeddings convert text to dense vectors useful for:
        - Semantic search and retrieval
        - Clustering and classification
        - Similarity computation
        - RAG (Retrieval-Augmented Generation) systems

        Args:
            texts: Single text or list of texts to embed
            model: Embedding model name (default: text-embedding-004)
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - 'embeddings': List of embedding vectors (or single vector if input was string)
            - 'usage': UsageMetrics with token counts

        Raises:
            ValueError: If provider is not available

        Example:
            # Embed documents for indexing
            doc_embeddings = await service.generate_embeddings(
                texts=["Document 1", "Document 2", "Document 3"],
                model="text-embedding-004"
            )

            # Embed search query
            query_embedding = await service.generate_embeddings(
                texts="python programming"
            )
        """
        log = self.log.bind(model=model, endpoint="generate_embeddings")

        try:
            # Get Google provider (prefer Vertex AI if available)
            google_instance = self.providers.get("google_vertex") or self.providers.get("google")
            if not google_instance:
                raise ValueError(
                    "Google provider not initialized (check GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT)"
                )

            # Normalize to list
            if isinstance(texts, str):
                texts = [texts]
                was_single = True
            else:
                was_single = False

            log.info(
                "generating_google_embeddings", num_texts=len(texts), model=model
            )

            # Use GoogleProvider's generate_embeddings method
            result = await google_instance.generate_embeddings(
                texts=texts, model=model, **kwargs
            )

            # Track usage
            self.usage_history.append(result["usage"])

            log.info(
                "embeddings_generated",
                provider="google",
                model=model,
                num_embeddings=len(result["embeddings"]) if isinstance(result["embeddings"], list) else 1,
                total_tokens=result["usage"].total_tokens,
            )

            # Return single embedding if input was single string
            if was_single and isinstance(result["embeddings"], list) and len(result["embeddings"]) == 1:
                result["embeddings"] = result["embeddings"][0]

            return result

        except Exception as e:
            log.exception("embedding_generation_failed", error=str(e))
            raise

    def get_usage_summary(self) -> dict[str, Any]:
        """
        Get summary of usage across all requests.

        This provides insights into token consumption across all
        requests made through this service instance.

        Returns:
            Dictionary containing:
            - total_requests: Number of API calls made
            - total_tokens: Total tokens across all requests
            - by_model: Usage breakdown by model

        Example:
            summary = service.get_usage_summary()
            print(f"Total requests: {summary['total_requests']}")
            print(f"Total tokens: {summary['total_tokens']}")

            # Model breakdown
            for model, stats in summary['by_model'].items():
                print(f"{model}: {stats['tokens']} tokens")
        """
        log = self.log.bind(endpoint="get_usage_summary")

        total_requests = len(self.usage_history)
        total_tokens = sum(u.total_tokens for u in self.usage_history)

        # Group by model
        by_model: dict[str, dict[str, Any]] = {}
        for usage in self.usage_history:
            model_key = usage.model or "unknown"
            if model_key not in by_model:
                by_model[model_key] = {"requests": 0, "tokens": 0}
            by_model[model_key]["requests"] += 1
            by_model[model_key]["tokens"] += usage.total_tokens

        summary = {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "by_model": by_model,
        }

        log.info("usage_summary_generated", summary=summary)

        return summary

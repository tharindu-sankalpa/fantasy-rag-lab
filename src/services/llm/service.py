# Dependencies:
# pip install structlog pydantic

"""
Unified LLM service orchestrator.

This module provides the main entry point for all LLM operations across multiple
providers. It orchestrates provider initialization, request routing, fallback chains,
and usage tracking.

Key Responsibilities:
1. Provider Management: Initialize and manage multiple provider instances
2. Unified Interface: Single API for text generation, structured outputs, embeddings
3. Fallback Chains: Automatic retry with alternative providers on failure
4. Usage Tracking: Aggregate and report token usage and costs across all providers
5. Configuration: Load providers based on environment variables

Architecture Philosophy:
- This is the ONLY class application code should interact with directly
- All provider-specific complexity is hidden behind this unified interface
- Fallback chains allow resilience against provider outages or rate limits
- Usage history enables cost monitoring and optimization
- Simple, predictable API that works the same regardless of provider

Example Usage:
    from src.services.llm import UnifiedLLMService

    service = UnifiedLLMService()

    # Text generation with automatic fallback
    response = await service.generate_text(
        prompt="Explain quantum computing",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        fallback_chain=["openai/gpt-4o", "google/gemini-2.0-flash-exp"]
    )

    # Structured output
    result = await service.generate_structured(
        prompt="Extract person info",
        schema=PersonSchema,
        provider="openai",
        model="gpt-4o"
    )

    # Embeddings
    embeddings = await service.generate_embeddings(
        texts=["doc 1", "doc 2"],
        provider="voyage",
        model="voyage-3-large"
    )

    # Get cost summary
    summary = service.get_usage_summary()
"""

import os
from typing import Optional, Any, Union, Literal

import structlog
from dotenv import load_dotenv

from .base import (
    BaseLLMProvider,
    GenerationResponse,
    UsageMetrics,
    ProviderType,
)
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .openrouter_provider import OpenRouterProvider
from .voyage_provider import VoyageProvider

# Load environment variables from .env file
load_dotenv()

# Initialize module-level logger
logger = structlog.get_logger()


class UnifiedLLMService:
    """
    Unified service orchestrating all LLM providers.

    This is the main entry point for the application. It provides:
    - A consistent interface regardless of which provider is being used
    - Automatic fallback chains for resilience
    - Centralized usage tracking and cost monitoring
    - Simple provider management based on environment variables

    The service automatically initializes all providers for which API keys are available
    in the environment. If a provider is not available (no API key), it's silently skipped.

    Usage Tracking:
    All requests are automatically logged to usage_history, allowing you to:
    - Monitor token consumption across providers
    - Track costs (especially useful with OpenRouter's direct cost reporting)
    - Identify expensive operations for optimization
    - Generate usage reports

    Fallback Chains:
    Fallback chains allow specifying alternative providers to try if the primary fails.
    Format: ["provider/model", "provider/model", ...]
    Example: ["openai/gpt-4o", "openrouter/anthropic/claude-sonnet-4"]

    This provides resilience against:
    - Provider outages or API failures
    - Rate limiting or quota exhaustion
    - Regional availability issues
    """

    def __init__(self):
        """
        Initialize the unified LLM service.

        Automatically loads all providers for which API keys are available
        in the environment. Providers without API keys are skipped (not an error).

        Environment Variables:
        - ANTHROPIC_API_KEY: For Anthropic/Claude
        - OPENAI_API_KEY: For OpenAI/GPT
        - GOOGLE_API_KEY or GEMINI_API_KEY: For Google Gemini (Developer API)
        - GOOGLE_CLOUD_PROJECT: For Google Vertex AI (if GOOGLE_GENAI_USE_VERTEXAI=true)
        - OPENROUTER_API_KEY: For OpenRouter
        - VOYAGE_API_KEY: For Voyage embeddings
        """
        self.log = logger.bind(component="UnifiedLLMService")
        self.log.info("initializing_unified_llm_service")

        # Provider instances (maps provider name -> provider instance)
        self.providers: dict[str, BaseLLMProvider] = {}

        # Voyage provider (separate since it doesn't follow BaseLLMProvider interface)
        self.voyage_provider: Optional[VoyageProvider] = None

        # Usage tracking (all requests across all providers)
        self.usage_history: list[UsageMetrics] = []

        # Load providers based on available API keys
        self._load_providers()

        # Log summary of loaded providers
        provider_names = list(self.providers.keys())
        if self.voyage_provider:
            provider_names.append("voyage")
        self.log.info("service_initialized", available_providers=provider_names)

    def _load_providers(self):
        """
        Load and initialize all available providers based on environment variables.

        This method checks for API keys and initializes providers accordingly.
        If a provider's API key is not available, it's silently skipped (not an error).

        This approach allows flexible deployment - you only need API keys for the
        providers you actually want to use.
        """
        self.log.info("loading_providers")

        # Anthropic (Claude)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(api_key=anthropic_key)
                self.log.info("provider_loaded", provider="anthropic")
            except Exception as e:
                self.log.error(
                    "provider_load_failed", provider="anthropic", error=str(e)
                )

        # OpenAI (GPT)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(api_key=openai_key)
                self.log.info("provider_loaded", provider="openai")
            except Exception as e:
                self.log.error("provider_load_failed", provider="openai", error=str(e))

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

        # OpenRouter (multi-provider gateway)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                self.providers["openrouter"] = OpenRouterProvider(
                    api_key=openrouter_key
                )
                self.log.info("provider_loaded", provider="openrouter")
            except Exception as e:
                self.log.error(
                    "provider_load_failed", provider="openrouter", error=str(e)
                )

        # Voyage AI (embeddings only)
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if voyage_key:
            try:
                self.voyage_provider = VoyageProvider(api_key=voyage_key)
                self.log.info("provider_loaded", provider="voyage")
            except Exception as e:
                self.log.error("provider_load_failed", provider="voyage", error=str(e))

        # Warn if no providers loaded
        if not self.providers and not self.voyage_provider:
            self.log.warning(
                "no_providers_loaded",
                message="No API keys found in environment. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.",
            )

    async def generate_text(
        self,
        prompt: str,
        provider: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        fallback_chain: Optional[list[str]] = None,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using specified provider with optional fallback.

        This is the primary method for text generation. It tries the specified provider
        first, then falls back to alternatives if provided.

        Args:
            prompt: Input text prompt
            provider: Provider name ('anthropic', 'openai', 'google', 'google_vertex', 'openrouter')
            model: Model identifier (provider-specific)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, higher = more creative)
            fallback_chain: List of "provider/model" strings to try if primary fails
                Example: ["openai/gpt-4o", "openrouter/anthropic/claude-sonnet-4"]
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
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                fallback_chain=["openai/gpt-4o", "google/gemini-2.0-flash-exp"],
                max_tokens=200
            )

            # Simple call without fallback
            response = await service.generate_text(
                prompt="Hello",
                provider="openai",
                model="gpt-4o"
            )
        """
        log = self.log.bind(provider=provider, model=model, endpoint="generate_text")

        # Build attempt queue: [primary, *fallbacks]
        attempt_queue = [(provider, model)]

        if fallback_chain:
            for fallback in fallback_chain:
                # Parse "provider/model" format
                if "/" in fallback:
                    fb_provider, fb_model = fallback.split("/", 1)
                    attempt_queue.append((fb_provider, fb_model))
                else:
                    log.warning(
                        "invalid_fallback_format",
                        fallback=fallback,
                        expected_format="provider/model",
                    )

        last_exception = None

        # Try each provider in the queue
        for attempt_num, (attempt_provider, attempt_model) in enumerate(
            attempt_queue, 1
        ):
            log_attempt = log.bind(
                attempt_num=attempt_num,
                total_attempts=len(attempt_queue),
                attempt_provider=attempt_provider,
                attempt_model=attempt_model,
            )

            try:
                # Get provider instance
                provider_instance = self.providers.get(attempt_provider)
                if not provider_instance:
                    log_attempt.warning(
                        "provider_not_available",
                        message=f"Provider '{attempt_provider}' not initialized (check API key)",
                    )
                    continue

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
                    cost_usd=response.usage.total_cost_usd,
                )

                return response

            except Exception as e:
                log_attempt.warning(
                    "text_generation_attempt_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                last_exception = e
                # Continue to next provider in fallback chain

        # All attempts failed
        log.error(
            "all_text_generation_attempts_failed",
            attempts=len(attempt_queue),
            last_error=str(last_exception) if last_exception else None,
        )
        raise last_exception or ValueError(
            "No providers available for text generation"
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        provider: str,
        model: str,
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
            provider: Provider name
            model: Model identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fallback_chain: List of fallback "provider/model" strings
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
                provider="openai",
                model="gpt-4o"
            )
            # result['parsed'] is a validated Person instance
            # result['parsed'].name == "Jane Doe"
        """
        log = self.log.bind(
            provider=provider, model=model, endpoint="generate_structured"
        )

        # Build attempt queue
        attempt_queue = [(provider, model)]

        if fallback_chain:
            for fallback in fallback_chain:
                if "/" in fallback:
                    fb_provider, fb_model = fallback.split("/", 1)
                    attempt_queue.append((fb_provider, fb_model))

        last_exception = None

        for attempt_num, (attempt_provider, attempt_model) in enumerate(
            attempt_queue, 1
        ):
            log_attempt = log.bind(
                attempt_num=attempt_num,
                total_attempts=len(attempt_queue),
                attempt_provider=attempt_provider,
                attempt_model=attempt_model,
            )

            try:
                # Get provider instance
                provider_instance = self.providers.get(attempt_provider)
                if not provider_instance:
                    log_attempt.warning("provider_not_available")
                    continue

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
            "No providers available for structured generation"
        )

    async def generate_embeddings(
        self,
        texts: Union[str, list[str]],
        provider: Literal["voyage", "openai", "google"] = "voyage",
        model: Optional[str] = None,
        input_type: Optional[Literal["query", "document"]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate embeddings using specified provider.

        Embeddings convert text to dense vectors useful for:
        - Semantic search and retrieval
        - Clustering and classification
        - Similarity computation
        - RAG (Retrieval-Augmented Generation) systems

        Supported Providers:
        - Voyage AI: Specialized embeddings, best retrieval quality (recommended)
        - OpenAI: text-embedding-3-large, text-embedding-3-small
        - Google: text-embedding-004 (via Vertex AI)

        Args:
            texts: Single text or list of texts to embed
            provider: 'voyage', 'openai', or 'google'
            model: Model name (uses provider default if not specified)
                - Voyage: voyage-3-large (default)
                - OpenAI: text-embedding-3-large (default)
                - Google: text-embedding-004 (default)
            input_type: 'query' or 'document' for retrieval optimization (Voyage only)
                - Use 'query' for user search queries
                - Use 'document' for corpus documents
            **kwargs: Provider-specific parameters

        Returns:
            Dictionary with:
            - 'embeddings': List of embedding vectors (or single vector if input was string)
            - 'usage': UsageMetrics with token counts

        Raises:
            ValueError: If provider is not available or parameters invalid

        Example:
            # Embed documents for indexing
            doc_embeddings = await service.generate_embeddings(
                texts=["Document 1", "Document 2", "Document 3"],
                provider="voyage",
                model="voyage-3-large",
                input_type="document"
            )

            # Embed search query
            query_embedding = await service.generate_embeddings(
                texts="python programming",
                provider="voyage",
                input_type="query"
            )
        """
        log = self.log.bind(provider=provider, endpoint="generate_embeddings")

        try:
            if provider == "voyage":
                # Use Voyage provider
                if not self.voyage_provider:
                    raise ValueError(
                        "Voyage provider not initialized (check VOYAGE_API_KEY)"
                    )

                model = model or "voyage-3-large"
                result = await self.voyage_provider.generate_embeddings(
                    texts=texts, model=model, input_type=input_type, **kwargs
                )

                # Track usage
                self.usage_history.append(result["usage"])

                log.info(
                    "embeddings_generated",
                    provider="voyage",
                    model=model,
                    num_texts=len(texts) if isinstance(texts, list) else 1,
                    total_tokens=result["usage"].total_tokens,
                )

                return result

            elif provider == "openai":
                # Use OpenAI embeddings
                openai_instance = self.providers.get("openai")
                if not openai_instance or not isinstance(
                    openai_instance, OpenAIProvider
                ):
                    raise ValueError(
                        "OpenAI provider not initialized (check OPENAI_API_KEY)"
                    )

                model = model or "text-embedding-3-large"

                # Normalize to list
                if isinstance(texts, str):
                    texts = [texts]
                    was_single = True
                else:
                    was_single = False

                log.info(
                    "generating_openai_embeddings", num_texts=len(texts), model=model
                )

                # Call OpenAI embeddings API
                response = await openai_instance.client.embeddings.create(
                    input=texts, model=model
                )

                embeddings = [item.embedding for item in response.data]

                usage = UsageMetrics(
                    total_tokens=response.usage.total_tokens,
                    input_tokens=response.usage.total_tokens,
                    output_tokens=0,
                    provider=ProviderType.OPENAI.value,
                    model=model,
                    api_key_last4=openai_instance._mask_api_key(),
                )

                self.usage_history.append(usage)

                log.info(
                    "embeddings_generated",
                    provider="openai",
                    model=model,
                    num_embeddings=len(embeddings),
                    embedding_dim=len(embeddings[0]) if embeddings else 0,
                    total_tokens=usage.total_tokens,
                )

                return {
                    "embeddings": embeddings[0] if was_single else embeddings,
                    "usage": usage,
                }

            elif provider == "google":
                # Use Google embeddings (Vertex AI or Developer API)
                google_instance = self.providers.get(
                    "google_vertex"
                ) or self.providers.get("google")
                if not google_instance or not isinstance(
                    google_instance, GoogleProvider
                ):
                    raise ValueError(
                        "Google provider not initialized (check GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT)"
                    )

                model = model or "text-embedding-004"

                # Normalize to list
                if isinstance(texts, str):
                    texts = [texts]
                    was_single = True
                else:
                    was_single = False

                log.info(
                    "generating_google_embeddings", num_texts=len(texts), model=model
                )

                # Use Google's embed_content API
                embeddings_list = []
                total_tokens = 0

                for text in texts:
                    response = await google_instance.client.aio.models.embed_content(
                        model=model, content=text
                    )
                    embeddings_list.append(response.embedding)
                    # Approximate token count (Google doesn't always provide this)
                    total_tokens += len(text.split())

                usage = UsageMetrics(
                    total_tokens=total_tokens,
                    input_tokens=total_tokens,
                    output_tokens=0,
                    provider=ProviderType.GOOGLE.value,
                    model=model,
                    api_key_last4=google_instance._mask_api_key(),
                )

                self.usage_history.append(usage)

                log.info(
                    "embeddings_generated",
                    provider="google",
                    model=model,
                    num_embeddings=len(embeddings_list),
                    total_tokens=total_tokens,
                )

                return {
                    "embeddings": embeddings_list[0] if was_single else embeddings_list,
                    "usage": usage,
                }

            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        except Exception as e:
            log.exception("embedding_generation_failed", error=str(e), provider=provider)
            raise

    def get_usage_summary(self) -> dict[str, Any]:
        """
        Get summary of usage across all providers.

        This provides insights into token consumption and costs across all
        requests made through this service instance.

        Returns:
            Dictionary containing:
            - total_requests: Number of API calls made
            - total_tokens: Total tokens across all requests
            - total_cost_usd: Total cost in USD (where available, e.g., OpenRouter)
            - by_provider: Usage breakdown by provider
            - by_model: Usage breakdown by model

        Example:
            summary = service.get_usage_summary()
            print(f"Total requests: {summary['total_requests']}")
            print(f"Total tokens: {summary['total_tokens']}")
            print(f"Total cost: ${summary['total_cost_usd']:.4f}")

            # Provider breakdown
            for provider, stats in summary['by_provider'].items():
                print(f"{provider}: {stats['tokens']} tokens, ${stats['cost_usd']:.4f}")
        """
        log = self.log.bind(endpoint="get_usage_summary")

        total_requests = len(self.usage_history)
        total_tokens = sum(u.total_tokens for u in self.usage_history)
        total_cost = sum(
            u.total_cost_usd for u in self.usage_history if u.total_cost_usd
        )

        # Group by provider
        by_provider: dict[str, dict[str, Any]] = {}
        for usage in self.usage_history:
            if usage.provider not in by_provider:
                by_provider[usage.provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                }
            by_provider[usage.provider]["requests"] += 1
            by_provider[usage.provider]["tokens"] += usage.total_tokens
            if usage.total_cost_usd:
                by_provider[usage.provider]["cost_usd"] += usage.total_cost_usd

        # Group by model
        by_model: dict[str, dict[str, Any]] = {}
        for usage in self.usage_history:
            model_key = f"{usage.provider}/{usage.model}"
            if model_key not in by_model:
                by_model[model_key] = {"requests": 0, "tokens": 0, "cost_usd": 0.0}
            by_model[model_key]["requests"] += 1
            by_model[model_key]["tokens"] += usage.total_tokens
            if usage.total_cost_usd:
                by_model[model_key]["cost_usd"] += usage.total_cost_usd

        summary = {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "by_provider": by_provider,
            "by_model": by_model,
        }

        log.info("usage_summary_generated", summary=summary)

        return summary

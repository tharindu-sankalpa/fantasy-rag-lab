# Dependencies:
# pip install openai structlog pydantic

"""
OpenRouter provider implementation.

OpenRouter is a unified gateway providing access to 400+ models from 60+ providers
through a single API. It's particularly valuable for:
- Testing multiple models without managing multiple API keys
- Automatic failover and load balancing
- Direct cost reporting in USD (not just tokens)
- Access to models not directly available (gated models, regional availability)

Key Features:
- Single API key for all providers
- Direct cost data in API responses (unique advantage over direct providers)
- Generation IDs for detailed tracking in OpenRouter dashboard
- Automatic model routing and failover
- Competitive pricing with volume discounts

OpenRouter's official SDK is still in beta, so we use the OpenAI SDK with a custom
base URL (https://openrouter.ai/api/v1) as recommended in their documentation.
This approach is stable and production-ready.

Available Model Examples:
- anthropic/claude-sonnet-4
- openai/gpt-4o
- google/gemini-2.0-flash-exp
- meta-llama/llama-3.1-405b
- mistralai/mistral-large
- deepseek/deepseek-v3
- And 400+ more...

Note: Model names are prefixed with provider (e.g., 'anthropic/claude-sonnet-4')
"""

from typing import Any, Optional
from pydantic import BaseModel

import structlog
from openai import AsyncOpenAI

from .base import BaseLLMProvider, GenerationResponse, UsageMetrics, ProviderType

# Initialize module-level logger
logger = structlog.get_logger()


class OpenRouterProvider(BaseLLMProvider):
    """
    Provider for OpenRouter - unified gateway to multiple LLM providers.

    OpenRouter provides a unified API compatible with OpenAI's format, making it
    easy to switch between different providers and models without code changes.

    Key Advantages:
    1. Direct cost reporting: OpenRouter returns actual USD cost in usage data
       (most providers only return token counts, requiring manual price calculation)
    2. Single API key: Access all models without managing multiple provider keys
    3. Automatic failover: If a model is unavailable, can auto-route to alternatives
    4. Generation tracking: Each request gets a unique ID for detailed analytics

    Architecture:
    Uses the OpenAI SDK with a custom base URL. This is the officially recommended
    approach and provides full compatibility with OpenAI's interface.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize OpenRouter provider with API key.

        Args:
            api_key: OpenRouter API key (get from openrouter.ai/keys)
            **kwargs: Additional configuration:
                - app_name: Application name for attribution (visible in dashboard)
                - site_url: Site URL for attribution
                - timeout: Request timeout in seconds
        """
        # Call parent constructor for logging setup
        super().__init__(api_key, **kwargs)

        # Extract app metadata for attribution (shows in OpenRouter dashboard)
        self.app_name = kwargs.get("app_name", "fantasy-rag-lab")
        self.site_url = kwargs.get("site_url", "https://fantasy-rag-lab.com")

        # Initialize OpenAI client with OpenRouter base URL
        # This is the officially recommended approach (OpenRouter SDK is beta)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            **{k: v for k, v in kwargs.items() if k not in ["app_name", "site_url"]},
        )

        self.log.info(
            "openrouter_provider_initialized",
            api_key=self._mask_api_key(),
            app_name=self.app_name,
        )

    def _build_headers(self) -> dict[str, str]:
        """
        Build OpenRouter-specific headers for attribution and tracking.

        These headers are recommended by OpenRouter for:
        - Showing your app in their dashboard
        - Tracking usage by application
        - Proper attribution in their analytics

        Returns:
            Dictionary of HTTP headers to include in requests
        """
        return {
            "HTTP-Referer": self.site_url,  # Your site URL
            "X-Title": self.app_name,  # Your app name
        }

    async def generate_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text via OpenRouter.

        OpenRouter uses OpenAI-compatible format but provides additional features
        like direct cost reporting and generation IDs for tracking.

        Args:
            prompt: User input/question
            model: Full model path (e.g., 'anthropic/claude-sonnet-4', 'google/gemini-2.0-flash-exp')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            **kwargs: Additional parameters:
                - system: System message for instructions
                - transforms: List of transforms (e.g., ['middle-out'] for combining providers)
                - route: Routing strategy ('fallback' for automatic failover)
                - models: List of models for fallback routing

        Returns:
            GenerationResponse with generated text, usage metrics, and cost data

        Raises:
            openai.APIError: If API request fails (wrapped OpenRouter errors)
            ValueError: If model name is invalid or parameters are wrong

        Example:
            response = await provider.generate_text(
                prompt="What is Python?",
                model="anthropic/claude-sonnet-4",
                max_tokens=200,
                system="You are a helpful assistant"
            )
            # response.usage.total_cost_usd contains actual cost!
        """
        log = self.log.bind(model=model, endpoint="generate_text")

        try:
            # Build messages array (OpenAI-compatible format)
            messages = [{"role": "user", "content": prompt}]

            # Extract and prepend system message if provided
            system_msg = kwargs.pop("system", None)
            if system_msg:
                messages.insert(0, {"role": "system", "content": system_msg})

            # Enable detailed usage reporting to get cost data
            # This is an OpenRouter-specific feature
            extra_body = kwargs.pop("extra_body", {})
            extra_body["usage"] = {"include": True}  # Request cost information

            log.info(
                "generating_text_via_openrouter",
                prompt_length=len(prompt),
                has_system=system_msg is not None,
            )

            # Call OpenRouter API (via OpenAI SDK)
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=self._build_headers(),  # Attribution headers
                extra_body=extra_body,  # Request cost data
                **kwargs,
            )

            # Extract generated text
            content = response.choices[0].message.content or ""

            # Extract usage data (includes token counts AND cost)
            usage_data = response.usage

            # Extract cached tokens safely (prompt_tokens_details is a Pydantic object, not dict)
            cached_tokens = 0
            if hasattr(usage_data, "prompt_tokens_details"):
                details = usage_data.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    cached_tokens = details.cached_tokens

            # Build usage metrics with cost data
            # This is unique to OpenRouter - direct USD cost reporting
            usage = UsageMetrics(
                input_tokens=getattr(usage_data, "prompt_tokens", 0),
                output_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
                # Cached tokens (if model supports prompt caching)
                cached_tokens=cached_tokens,
                # OpenRouter provides direct cost in USD - this is a key differentiator
                input_cost_usd=getattr(usage_data, "prompt_cost", None),
                output_cost_usd=getattr(usage_data, "completion_cost", None),
                total_cost_usd=getattr(usage_data, "total_cost", None),
                provider=ProviderType.OPENROUTER.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            # Extract generation ID for tracking in OpenRouter dashboard
            generation_id = response.id

            log.info(
                "text_generated_via_openrouter",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_cost_usd=usage.total_cost_usd,
                generation_id=generation_id,
                finish_reason=response.choices[0].finish_reason,
            )

            return GenerationResponse(
                content=content,
                usage=usage,
                raw_response=response,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            log.exception("openrouter_text_generation_failed", error=str(e), model=model)
            raise

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
        Generate structured output via OpenRouter.

        OpenRouter supports structured outputs for compatible models (OpenAI models,
        some Anthropic models). The availability depends on the underlying provider.

        Important: Not all models on OpenRouter support structured outputs.
        Check model capabilities at openrouter.ai/models before using this feature.

        Args:
            prompt: Instruction describing what to extract/generate
            model: Full model path (must support structured outputs)
            schema: Pydantic BaseModel defining output structure
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - 'parsed': Validated Pydantic model instance
            - 'usage': UsageMetrics including cost data

        Raises:
            ValueError: If schema is not a Pydantic BaseModel or model doesn't support structured outputs
            openai.APIError: If API request fails

        Example:
            class Email(BaseModel):
                subject: str
                body: str
                recipients: list[str]

            result = await provider.generate_structured(
                prompt="Write an email about the meeting",
                model="openai/gpt-4o",  # Use OpenAI model via OpenRouter
                schema=Email
            )
            # result['usage'].total_cost_usd shows exact cost
        """
        log = self.log.bind(model=model, endpoint="generate_structured")

        try:
            # Validate schema is a Pydantic model
            if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                raise ValueError("Schema must be a Pydantic BaseModel class")

            # Build messages
            messages = [{"role": "user", "content": prompt}]

            # Extract and prepend system message if provided
            system_msg = kwargs.pop("system", None)
            if system_msg:
                messages.insert(0, {"role": "system", "content": system_msg})

            # Enable cost reporting
            extra_body = kwargs.pop("extra_body", {})
            extra_body["usage"] = {"include": True}

            log.info(
                "generating_structured_output_via_openrouter",
                prompt_length=len(prompt),
                schema_name=schema.__name__,
            )

            # Use OpenAI's beta structured outputs API (via OpenRouter)
            # Note: This only works for models that support structured outputs
            completion = await self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,  # Pydantic model for structured output
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=self._build_headers(),
                extra_body=extra_body,
                **kwargs,
            )

            # Extract parsed output
            parsed_output = completion.choices[0].message.parsed

            if not parsed_output:
                log.error("parsing_failed", refusal=completion.choices[0].message.refusal)
                raise ValueError("Failed to parse structured output from response")

            # Extract usage data
            usage_data = completion.usage

            # Extract cached tokens safely
            cached_tokens = 0
            if hasattr(usage_data, "prompt_tokens_details"):
                details = usage_data.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    cached_tokens = details.cached_tokens

            # Extract usage with cost data
            usage = UsageMetrics(
                input_tokens=getattr(usage_data, "prompt_tokens", 0),
                output_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
                cached_tokens=cached_tokens,
                # OpenRouter's key feature: direct cost reporting
                input_cost_usd=getattr(usage_data, "prompt_cost", None),
                output_cost_usd=getattr(usage_data, "completion_cost", None),
                total_cost_usd=getattr(usage_data, "total_cost", None),
                provider=ProviderType.OPENROUTER.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            log.info(
                "structured_output_generated_via_openrouter",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_cost_usd=usage.total_cost_usd,
                generation_id=completion.id,
                schema_name=schema.__name__,
            )

            return {"parsed": parsed_output, "usage": usage}

        except Exception as e:
            log.exception(
                "openrouter_structured_generation_failed", error=str(e), model=model
            )
            raise

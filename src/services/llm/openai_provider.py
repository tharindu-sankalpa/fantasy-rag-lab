# Dependencies:
# pip install openai structlog pydantic

"""
OpenAI GPT provider implementation.

This module provides access to OpenAI's GPT models using the official openai
Python SDK. It uses the modern Responses API (/v1/responses) introduced in 2025
as the primary endpoint, with automatic fallback to Chat Completions for older models.

Supported Models (as of 2025):
- gpt-5.2-pro: Latest flagship model (responses API)
- gpt-4o: Optimized GPT-4 variant, fast and cost-effective
- gpt-4-turbo: Previous generation, still highly capable
- o1, o3: Reasoning models with extended thinking
- o1-mini: Smaller reasoning model
- gpt-3.5-turbo: Fast, low-cost for simple tasks

Key Features:
- Responses API: Stateful conversations with preserved reasoning
- Structured outputs: JSON schema validation (both APIs)
- Function calling and tool use
- Vision capabilities (GPT-4V, GPT-4o)
- Embeddings (text-embedding-3-large, text-embedding-3-small)
- Prompt caching: 40-80% better cache utilization with Responses API

Responses API vs Chat Completions:
- Responses: Stateful, structured output list, better for reasoning models
- Chat Completions: Stateless, simple turn-based, better for basic use cases
"""

from typing import Any, Optional, Dict, List
from pydantic import BaseModel
import httpx

import structlog
from openai import AsyncOpenAI

from .base import BaseLLMProvider, GenerationResponse, UsageMetrics, ProviderType

# Initialize module-level logger
logger = structlog.get_logger()

# Models that should use Responses API (GPT-5, o-series, reasoning models)
RESPONSES_API_MODELS = {
    "gpt-5",
    "gpt-5.2",
    "gpt-5.2-pro",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
}


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI's GPT models.

    Uses the official openai Python SDK which provides:
    - Async/await support for high-performance applications
    - Automatic retries with exponential backoff
    - Streaming support for real-time responses
    - Type hints and excellent IDE support
    - Native structured outputs (beta feature, very reliable)

    Architecture:
    This is a thin wrapper around AsyncOpenAI client, translating our unified
    interface into OpenAI-specific API calls while preserving provider-specific
    features through kwargs.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize OpenAI provider with API key.

        Args:
            api_key: OpenAI API key (get from platform.openai.com)
            **kwargs: Additional configuration:
                - organization: Organization ID (for multi-org accounts)
                - base_url: Custom API endpoint (for proxies or Azure)
                - timeout: Request timeout in seconds
                - max_retries: Number of retry attempts
        """
        # Call parent constructor for logging setup
        super().__init__(api_key, **kwargs)

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(api_key=api_key, **kwargs)

        # Log initialization with masked key
        self.log.info("openai_provider_initialized", api_key=self._mask_api_key())

    def _should_use_responses_api(self, model: str, kwargs: Dict) -> bool:
        """
        Determine if we should use Responses API vs Chat Completions.

        Responses API is used for:
        - GPT-5 series models
        - o-series reasoning models (o1, o3)
        - When explicitly requested via use_responses_api=True

        Args:
            model: Model identifier
            kwargs: Additional parameters (may contain use_responses_api flag)

        Returns:
            True if Responses API should be used, False for Chat Completions
        """
        # Check if explicitly requested
        if kwargs.get("use_responses_api") is True:
            return True

        # Check if explicitly disabled
        if kwargs.get("use_responses_api") is False:
            return False

        # Auto-detect based on model name
        # Check if model name starts with any RESPONSES_API_MODELS prefix
        model_lower = model.lower()
        for api_model in RESPONSES_API_MODELS:
            if model_lower.startswith(api_model):
                return True

        return False

    async def generate_text(
        self,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using GPT models.

        Automatically routes to the appropriate endpoint:
        - Responses API (/v1/responses) for GPT-5, o-series, and reasoning models
        - Chat Completions (/v1/chat/completions) for older models

        Args:
            prompt: User input/question
            model: OpenAI model name (default: gpt-4o)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            **kwargs: Additional parameters:
                - system: System message for instructions
                - use_responses_api: Force use of Responses API (True/False/None=auto)
                - previous_response_id: For stateful conversations (Responses API only)
                - presence_penalty: Penalty for token presence (-2.0 to 2.0)
                - frequency_penalty: Penalty for token frequency (-2.0 to 2.0)
                - top_p: Nucleus sampling parameter
                - stop: Stop sequences (string or list)

        Returns:
            GenerationResponse with generated text and usage metrics

        Raises:
            openai.APIError: If API request fails
            openai.RateLimitError: If rate limit exceeded
            ValueError: If parameters are invalid

        Example:
            # Auto-detects API based on model
            response = await provider.generate_text(
                prompt="What is Python?",
                model="gpt-5.2-pro",
                max_tokens=200,
                system="You are a helpful programming tutor"
            )
        """
        log = self.log.bind(model=model, endpoint="generate_text")

        try:
            # Determine which API to use
            use_responses = self._should_use_responses_api(model, kwargs)

            # Remove our custom flag before passing to API
            kwargs.pop("use_responses_api", None)

            if use_responses:
                log.info("using_responses_api", model=model)
                return await self._generate_with_responses_api(
                    prompt, model, max_tokens, temperature, log, **kwargs
                )
            else:
                log.info("using_chat_completions_api", model=model)
                return await self._generate_with_chat_completions(
                    prompt, model, max_tokens, temperature, log, **kwargs
                )

        except Exception as e:
            log.exception("text_generation_failed", error=str(e), model=model)
            raise

    async def _generate_with_responses_api(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        log: Any,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using Responses API (/v1/responses).

        This is the modern OpenAI endpoint optimized for reasoning models,
        stateful conversations, and multimodal interactions.
        """
        # Extract system instructions (called "instructions" in Responses API)
        instructions = kwargs.pop("system", None) or kwargs.pop("instructions", None)

        # Build request payload
        payload = {
            "model": model,
            "input": prompt,  # Simple string input
        }

        # Add optional parameters
        # Note: Some models (like gpt-5.2-pro) don't support temperature
        # Don't include temperature at all for Responses API to avoid errors
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        if instructions:
            payload["instructions"] = instructions

        # Pass through additional Responses API parameters
        for key in ["previous_response_id", "store", "reasoning", "metadata"]:
            if key in kwargs:
                payload[key] = kwargs.pop(key)

        log.info(
            "calling_responses_api",
            prompt_length=len(prompt),
            has_instructions=instructions is not None,
        )

        # Use httpx to call Responses API directly (not yet in official SDK)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            # Log error details if request fails
            if response.status_code != 200:
                error_body = response.text
                log.error(
                    "responses_api_error",
                    status_code=response.status_code,
                    error_body=error_body,
                    payload=payload,
                )

            response.raise_for_status()
            data = response.json()

        # Parse Responses API output
        # Output is a list of items (messages, reasoning, function_calls, etc.)
        output_items = data.get("output", [])

        # Extract text content from message items
        content_parts = []
        for item in output_items:
            if item.get("type") == "message":
                # Extract text from output_text content blocks
                for content_block in item.get("content", []):
                    if content_block.get("type") == "output_text":
                        content_parts.append(content_block.get("text", ""))

        content = "\n".join(content_parts) if content_parts else ""

        # Extract usage metrics
        usage_data = data.get("usage", {})
        input_tokens_details = usage_data.get("input_tokens_details", {})

        usage = UsageMetrics(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cached_tokens=input_tokens_details.get("cached_tokens", 0),
            reasoning_tokens=usage_data.get("output_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
            provider=ProviderType.OPENAI.value,
            model=data.get("model", model),
            api_key_last4=self._mask_api_key(),
        )

        log.info(
            "responses_api_success",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            status=data.get("status"),
        )

        return GenerationResponse(
            content=content,
            usage=usage,
            raw_response=data,
            finish_reason=data.get("status"),  # 'completed', 'failed', etc.
        )

    async def _generate_with_chat_completions(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        log: Any,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using Chat Completions API (/v1/chat/completions).

        This is the traditional OpenAI endpoint, still supported for
        non-reasoning models and simpler use cases.
        """
        # Build messages array
        messages = [{"role": "user", "content": prompt}]

        # Extract system message if provided
        system_msg = kwargs.pop("system", None)
        if system_msg:
            messages.insert(0, {"role": "system", "content": system_msg})

        log.info(
            "calling_chat_completions_api",
            prompt_length=len(prompt),
            has_system=system_msg is not None,
        )

        # Call OpenAI Chat Completions API
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Extract content
        content = response.choices[0].message.content or ""

        # Build usage metrics
        usage_data = response.usage
        cached_tokens = 0
        if hasattr(usage_data, "prompt_tokens_details"):
            cached_tokens = getattr(
                usage_data.prompt_tokens_details, "cached_tokens", 0
            )

        usage = UsageMetrics(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
            cached_tokens=cached_tokens,
            provider=ProviderType.OPENAI.value,
            model=model,
            api_key_last4=self._mask_api_key(),
        )

        log.info(
            "chat_completions_api_success",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            finish_reason=response.choices[0].finish_reason,
        )

        return GenerationResponse(
            content=content,
            usage=usage,
            raw_response=response,
            finish_reason=response.choices[0].finish_reason,
        )

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
        Generate structured output using OpenAI's structured outputs.

        Automatically routes to the appropriate API:
        - Responses API for GPT-5, o-series models (uses text.format with json_schema)
        - Chat Completions for other models (uses beta.chat.completions.parse)

        Args:
            prompt: Instruction describing what to extract/generate
            model: OpenAI model name
            schema: Pydantic BaseModel defining output structure
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - 'parsed': Validated Pydantic model instance
            - 'usage': UsageMetrics for this request

        Raises:
            ValueError: If schema is not a Pydantic BaseModel
            openai.APIError: If API request fails

        Example:
            class Product(BaseModel):
                name: str
                price: float

            result = await provider.generate_structured(
                prompt="Extract: iPhone 15 costs $999",
                model="gpt-5.2-pro",
                schema=Product
            )
        """
        log = self.log.bind(model=model, endpoint="generate_structured")

        try:
            # Validate schema is a Pydantic model
            if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                raise ValueError("Schema must be a Pydantic BaseModel class")

            # Determine which API to use
            use_responses = self._should_use_responses_api(model, kwargs)
            kwargs.pop("use_responses_api", None)

            if use_responses:
                log.info("using_responses_api_for_structured", model=model)
                return await self._generate_structured_with_responses_api(
                    prompt, model, schema, max_tokens, temperature, log, **kwargs
                )
            else:
                log.info("using_chat_completions_for_structured", model=model)
                return await self._generate_structured_with_chat_completions(
                    prompt, model, schema, max_tokens, temperature, log, **kwargs
                )

        except Exception as e:
            log.exception("structured_generation_failed", error=str(e), model=model)
            raise

    async def _generate_structured_with_responses_api(
        self,
        prompt: str,
        model: str,
        schema: Any,
        max_tokens: int,
        temperature: float,
        log: Any,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured output using Responses API with JSON schema.
        """
        # Extract instructions
        instructions = kwargs.pop("system", None) or kwargs.pop("instructions", None)

        # Get JSON schema from Pydantic model
        json_schema = schema.model_json_schema()

        # OpenAI requires additionalProperties: false for strict schema validation
        # Add this recursively to all objects in the schema
        def make_strict_schema(schema_obj):
            """Recursively add additionalProperties: false to all objects."""
            if isinstance(schema_obj, dict):
                if schema_obj.get("type") == "object":
                    schema_obj["additionalProperties"] = False
                # Recurse into properties
                if "properties" in schema_obj:
                    for prop_schema in schema_obj["properties"].values():
                        make_strict_schema(prop_schema)
                # Recurse into items (for arrays)
                if "items" in schema_obj:
                    make_strict_schema(schema_obj["items"])
                # Recurse into definitions/$defs
                for key in ["definitions", "$defs"]:
                    if key in schema_obj:
                        for def_schema in schema_obj[key].values():
                            make_strict_schema(def_schema)
            return schema_obj

        strict_schema = make_strict_schema(json_schema)

        # Build payload with proper structure for Responses API
        # Structure: text.format.{type, name, schema, strict} - all at format level
        payload = {
            "model": model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "schema": strict_schema,
                    "strict": True,
                }
            },
        }

        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        if instructions:
            payload["instructions"] = instructions

        log.info(
            "calling_responses_api_structured",
            prompt_length=len(prompt),
            schema_name=schema.__name__,
        )

        # Call Responses API
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if response.status_code != 200:
                log.error(
                    "responses_api_error",
                    status_code=response.status_code,
                    error_body=response.text,
                )

            response.raise_for_status()
            data = response.json()

        # Extract JSON content from output
        import json

        output_items = data.get("output", [])
        json_text = None

        for item in output_items:
            if item.get("type") == "message":
                for content_block in item.get("content", []):
                    if content_block.get("type") == "output_text":
                        json_text = content_block.get("text", "")
                        break
                if json_text:
                    break

        if not json_text:
            raise ValueError("No JSON output found in response")

        # Parse and validate JSON against schema
        parsed_data = json.loads(json_text)
        parsed_output = schema.model_validate(parsed_data)

        # Extract usage
        usage_data = data.get("usage", {})
        usage = UsageMetrics(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            provider=ProviderType.OPENAI.value,
            model=data.get("model", model),
            api_key_last4=self._mask_api_key(),
        )

        log.info(
            "responses_api_structured_success",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            schema_name=schema.__name__,
        )

        return {"parsed": parsed_output, "usage": usage}

    async def _generate_structured_with_chat_completions(
        self,
        prompt: str,
        model: str,
        schema: Any,
        max_tokens: int,
        temperature: float,
        log: Any,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured output using Chat Completions with beta.parse API.
        """
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        system_msg = kwargs.pop("system", None)
        if system_msg:
            messages.insert(0, {"role": "system", "content": system_msg})

        log.info(
            "calling_chat_completions_structured",
            prompt_length=len(prompt),
            schema_name=schema.__name__,
        )

        # Use OpenAI's beta structured outputs API
        completion = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        parsed_output = completion.choices[0].message.parsed

        if not parsed_output:
            log.error("parsing_failed", refusal=completion.choices[0].message.refusal)
            raise ValueError("Failed to parse structured output")

        # Build usage metrics
        usage_data = completion.usage
        usage = UsageMetrics(
            input_tokens=usage_data.prompt_tokens,
            output_tokens=usage_data.completion_tokens,
            total_tokens=usage_data.total_tokens,
            provider=ProviderType.OPENAI.value,
            model=model,
            api_key_last4=self._mask_api_key(),
        )

        log.info(
            "chat_completions_structured_success",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            schema_name=schema.__name__,
        )

        return {"parsed": parsed_output, "usage": usage}

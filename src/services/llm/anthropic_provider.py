# Dependencies:
# pip install anthropic structlog pydantic

"""
Anthropic Claude provider implementation.

This module provides access to Anthropic's Claude models using the official
anthropic Python SDK. The SDK is production-ready and provides:
- Reliable streaming and error handling
- Automatic retry logic with exponential backoff
- Type safety and excellent developer experience
- Support for all Claude features (prompt caching, tool use, vision)

Supported Models (as of 2025):
- claude-opus-4-5-20251101: Most capable, best for complex tasks
- claude-sonnet-4-20250514: Balanced performance/cost
- claude-sonnet-4: Latest Sonnet (alias)
- claude-haiku-4: Fast, cost-effective for simple tasks

Key Features:
- Text generation with system prompts
- Structured output via tool calling (Anthropic's approach to schema enforcement)
- Prompt caching for repeated context (can reduce costs by 90%)
- Vision capabilities (images, PDFs)
- Tool use for function calling
"""

from typing import Any, Optional
from pydantic import BaseModel

import structlog
from anthropic import AsyncAnthropic

from .base import BaseLLMProvider, GenerationResponse, UsageMetrics, ProviderType

# Initialize module-level logger
logger = structlog.get_logger()


class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic's Claude models.

    Uses the official anthropic Python SDK which provides:
    - Async/await support for non-blocking operations
    - Automatic retry with exponential backoff
    - Streaming support for real-time responses
    - Type hints for all API parameters

    Architecture:
    This class is a thin wrapper around the AsyncAnthropic client. It translates
    our unified interface into Anthropic-specific API calls while preserving
    access to provider-specific features through kwargs.
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize Anthropic provider with API key.

        Args:
            api_key: Anthropic API key (get from console.anthropic.com)
            **kwargs: Additional configuration (timeout, max_retries, etc.)
        """
        # Call parent constructor to set up logging
        super().__init__(api_key, **kwargs)

        # Initialize async Anthropic client with provided API key
        self.client = AsyncAnthropic(api_key=api_key, **kwargs)

        # Log initialization with masked key for security
        self.log.info("anthropic_provider_initialized", api_key=self._mask_api_key())

    async def generate_text(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using Claude models.

        Anthropic's API uses a messages format where each message has a role
        (user, assistant) and content. The system prompt is passed separately.

        Args:
            prompt: User input/question
            model: Claude model name (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens to generate (required by Anthropic)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            **kwargs: Additional parameters:
                - system: System prompt for instruction/context
                - stop_sequences: List of sequences that stop generation
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter

        Returns:
            GenerationResponse with generated text and usage metrics

        Raises:
            anthropic.APIError: If API request fails
            ValueError: If parameters are invalid

        Example:
            response = await provider.generate_text(
                prompt="Explain quantum computing",
                model="claude-sonnet-4",
                max_tokens=500,
                system="You are a physics professor"
            )
        """
        # Bind model to logger for request-specific context
        log = self.log.bind(model=model, endpoint="generate_text")

        try:
            # Build messages array (Anthropic expects list of message dicts)
            messages = [{"role": "user", "content": prompt}]

            # Extract system prompt if provided (Anthropic uses separate system param)
            system = kwargs.pop("system", None)

            # Log generation attempt with prompt length for debugging
            log.info("generating_text", prompt_length=len(prompt), has_system=system is not None)

            # Build API call parameters
            api_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }

            # Only include system parameter if provided (avoid passing None)
            if system:
                api_params["system"] = system

            # Merge any additional kwargs
            api_params.update(kwargs)

            # Call Anthropic API
            # Note: We use await since this is an async operation
            response = await self.client.messages.create(**api_params)

            # Extract text content from response
            # Anthropic returns content as a list of content blocks
            content = response.content[0].text if response.content else ""

            # Build normalized usage metrics
            # Anthropic provides detailed token counts including cache hits
            usage = UsageMetrics(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                # Prompt caching: Tokens served from cache (90% cost reduction)
                cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
                provider=ProviderType.ANTHROPIC.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            # Log successful generation with usage details
            log.info(
                "text_generated",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                finish_reason=response.stop_reason,
            )

            # Return standardized response
            return GenerationResponse(
                content=content,
                usage=usage,
                raw_response=response,  # Include raw response for advanced use cases
                finish_reason=response.stop_reason,  # 'end_turn', 'max_tokens', etc.
            )

        except Exception as e:
            # Log exception with full context (structlog captures stack trace)
            log.exception("text_generation_failed", error=str(e), model=model)
            raise  # Re-raise to allow caller to handle

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
        Generate structured output using Claude via tool calling.

        Anthropic doesn't have native "JSON mode" like OpenAI. Instead, we use
        tool calling to enforce schema compliance. This approach:
        1. Defines a tool with the desired schema
        2. Forces Claude to use that tool (tool_choice)
        3. Extracts the structured output from the tool call

        This method is reliable because Claude is forced to call the tool,
        ensuring the output matches the schema.

        Args:
            prompt: Instruction describing what to extract/generate
            model: Claude model name
            schema: Pydantic BaseModel defining output structure
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - 'parsed': Validated Pydantic model instance
            - 'usage': UsageMetrics for this request

        Raises:
            ValueError: If schema is not a Pydantic BaseModel or tool not used
            anthropic.APIError: If API request fails

        Example:
            class Person(BaseModel):
                name: str
                age: int

            result = await provider.generate_structured(
                prompt="Extract: John is 30 years old",
                model="claude-sonnet-4",
                schema=Person
            )
            # result['parsed'] is a validated Person instance
        """
        log = self.log.bind(model=model, endpoint="generate_structured")

        try:
            # Validate that schema is a Pydantic model
            if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                raise ValueError("Schema must be a Pydantic BaseModel class")

            # Convert Pydantic schema to Anthropic tool format
            # Anthropic tools use JSON schema for input validation
            tool_schema = {
                "name": "extract_information",  # Tool name (Claude will call this)
                "description": "Extract structured information from the text",
                "input_schema": schema.model_json_schema(),  # Pydantic -> JSON schema
            }

            log.info(
                "generating_structured_output",
                prompt_length=len(prompt),
                schema_name=schema.__name__,
                max_tokens=max_tokens,
            )

            # Determine if streaming is required
            # Anthropic SDK requires streaming for operations that may take > 10 minutes
            # This is typically triggered by large max_tokens values (>= 8000)
            use_streaming = max_tokens >= 8000

            if use_streaming:
                log.info(
                    "using_streaming_mode",
                    reason="large_max_tokens",
                    max_tokens=max_tokens
                )

                # Call Anthropic API with streaming enabled
                async with self.client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": "extract_information"},
                    **kwargs,
                ) as stream:
                    # Accumulate the streamed response
                    response = await stream.get_final_message()
            else:
                # Use non-streaming mode for small requests
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": "extract_information"},
                    **kwargs,
                )

            # Extract tool use result from response
            # Response contains content blocks; we need the tool_use block
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_information":
                    tool_use_block = block
                    break

            if not tool_use_block:
                # This shouldn't happen with tool_choice, but handle gracefully
                log.error("no_tool_use_found", response_content=response.content)
                raise ValueError("Claude did not use the requested tool")

            # Validate tool output against Pydantic schema
            # This ensures type safety and validation rules are enforced
            parsed_output = schema.model_validate(tool_use_block.input)

            # Build usage metrics
            usage = UsageMetrics(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
                provider=ProviderType.ANTHROPIC.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            log.info(
                "structured_output_generated",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                schema_name=schema.__name__,
            )

            # Return parsed output and usage metrics
            return {"parsed": parsed_output, "usage": usage}

        except Exception as e:
            log.exception("structured_generation_failed", error=str(e), model=model)
            raise

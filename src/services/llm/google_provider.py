# Dependencies:
# pip install google-genai structlog pydantic

"""
Google Gemini provider implementation.

This module provides access to Google's Gemini models using the official
google-genai SDK (the modern unified SDK replacing the older vertexai package).

The google-genai SDK supports both:
1. Gemini Developer API (via API key) - Free tier available
2. Vertex AI (via GCP project) - Production workloads with SLAs

Supported Models (as of 2025):
- gemini-2.0-flash-exp: Latest experimental, fastest, most cost-effective
- gemini-3-pro-preview: Most capable (if you have access)
- gemini-2.0-flash: Production-ready fast model
- gemini-1.5-pro: Previous generation, still capable
- gemini-1.5-flash: Fast and economical

Key Features:
- Text generation with system instructions
- Native JSON mode with schema validation
- Vision capabilities (images, videos, PDFs)
- Audio understanding
- Long context windows (up to 2M tokens)
- Grounding with Google Search
- Prompt caching (context caching for repeated use)

Note: The old `vertexai.generative_models` was deprecated June 2025.
This implementation uses the new unified SDK: `google.genai`
"""

from typing import Any, Optional
import json

import structlog
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from .base import BaseLLMProvider, GenerationResponse, UsageMetrics, ProviderType

# Initialize module-level logger
logger = structlog.get_logger()


class GoogleProvider(BaseLLMProvider):
    """
    Provider for Google's Gemini models via google-genai SDK.

    This provider supports two authentication modes:
    1. API Key mode (Gemini Developer API): Simple, free tier available
    2. Vertex AI mode: GCP project-based, for production workloads

    The provider automatically handles:
    - Async operations for non-blocking I/O
    - Token counting and usage tracking
    - Schema validation for structured outputs
    - Error handling with detailed logging

    Architecture:
    Thin wrapper around google.genai.Client, providing a unified interface
    while preserving Google-specific features (grounding, safety settings, etc.)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_vertex: bool = False,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        **kwargs,
    ):
        """
        Initialize Google Gemini provider.

        Args:
            api_key: Gemini API key for Developer API mode (get from aistudio.google.com)
            use_vertex: If True, use Vertex AI instead of Developer API
            project_id: GCP project ID (required if use_vertex=True)
            location: GCP location/region (default: us-central1)
            **kwargs: Additional configuration

        Raises:
            ValueError: If use_vertex=True but project_id not provided, or if
                       use_vertex=False but api_key not provided
        """
        # Call parent constructor (handles logging setup)
        super().__init__(api_key or "", **kwargs)

        self.use_vertex = use_vertex

        # Initialize client based on mode
        if use_vertex:
            # Vertex AI mode (production GCP workloads)
            if not project_id:
                raise ValueError("project_id is required when use_vertex=True")

            # Create Vertex AI client
            self.client = genai.Client(
                vertexai=True, project=project_id, location=location
            )

            self.log.info(
                "google_vertex_provider_initialized",
                project=project_id,
                location=location,
            )
        else:
            # Developer API mode (API key based)
            if not api_key:
                raise ValueError("api_key is required when use_vertex=False")

            # Create Developer API client
            self.client = genai.Client(api_key=api_key)

            self.log.info(
                "google_gemini_provider_initialized", api_key=self._mask_api_key()
            )

    async def generate_text(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-exp",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs,
    ) -> GenerationResponse:
        """
        Generate text using Gemini models.

        Google's API is simpler than others - it takes a content string directly
        rather than a messages array. System instructions can be passed via config.

        Args:
            prompt: Input text prompt
            model: Gemini model name (default: gemini-2.0-flash-exp)
            max_tokens: Maximum output tokens (called max_output_tokens in Gemini)
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
            **kwargs: Additional parameters:
                - system_instruction: System prompt (like system message)
                - safety_settings: Content safety filters
                - stop_sequences: List of stop sequences
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter

        Returns:
            GenerationResponse with generated text and usage metrics

        Raises:
            google.api_core.exceptions.GoogleAPIError: If API request fails
            ValueError: If parameters are invalid

        Example:
            response = await provider.generate_text(
                prompt="Explain machine learning",
                model="gemini-2.0-flash-exp",
                max_tokens=500,
                system_instruction="You are a helpful AI assistant"
            )
        """
        log = self.log.bind(model=model, endpoint="generate_text")

        try:
            # Extract system instruction if provided
            system_instruction = kwargs.pop("system_instruction", None)
            # Also support 'system' for consistency with other providers
            if not system_instruction:
                system_instruction = kwargs.pop("system", None)

            log.info(
                "generating_text",
                prompt_length=len(prompt),
                has_system=system_instruction is not None,
            )

            # Build generation configuration
            # Google uses max_output_tokens instead of max_tokens
            config = GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=system_instruction,
                **kwargs,
            )

            # Call Google Gen AI API
            # Uses async client for non-blocking operation
            response = await self.client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )

            # Extract text from response
            # Gemini returns text in a specific format
            content = response.text if hasattr(response, "text") else ""

            # Extract usage metadata
            # Google reports tokens as prompt_token_count, candidates_token_count, etc.
            usage_metadata = (
                response.usage_metadata if hasattr(response, "usage_metadata") else None
            )

            if usage_metadata:
                input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(usage_metadata, "total_token_count", 0)
                cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0)
            else:
                # Fallback if usage metadata not available
                input_tokens = output_tokens = total_tokens = cached_tokens = 0

            usage = UsageMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                provider=ProviderType.GOOGLE.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            # Extract finish reason
            # Google uses finish_reason in candidates
            finish_reason = None
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

            log.info(
                "text_generated",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                finish_reason=finish_reason,
            )

            return GenerationResponse(
                content=content,
                usage=usage,
                raw_response=response,
                finish_reason=finish_reason,
            )

        except Exception as e:
            log.exception("text_generation_failed", error=str(e), model=model)
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
        Generate structured output using Gemini's native JSON mode.

        Google's Gen AI SDK supports response schemas for structured outputs.
        This is similar to OpenAI's structured outputs but uses JSON schema directly.

        How it works:
        1. Convert Pydantic model to JSON schema
        2. Set response_mime_type to "application/json"
        3. Pass response_schema to constrain output
        4. Parse and validate the JSON response

        Args:
            prompt: Instruction describing what to extract/generate
            model: Gemini model name
            schema: Pydantic BaseModel defining output structure
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - 'parsed': Validated Pydantic model instance
            - 'usage': UsageMetrics for this request

        Raises:
            ValueError: If schema is not a Pydantic BaseModel or parsing fails
            google.api_core.exceptions.GoogleAPIError: If API request fails

        Example:
            class Recipe(BaseModel):
                name: str
                ingredients: list[str]
                steps: list[str]

            result = await provider.generate_structured(
                prompt="Create a recipe for chocolate cake",
                model="gemini-2.0-flash-exp",
                schema=Recipe
            )
            # result['parsed'] is a validated Recipe instance
        """
        log = self.log.bind(model=model, endpoint="generate_structured")

        try:
            # Validate schema is a Pydantic model
            if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                raise ValueError("Schema must be a Pydantic BaseModel class")

            log.info(
                "generating_structured_output",
                prompt_length=len(prompt),
                schema_name=schema.__name__,
            )

            # Convert Pydantic schema to JSON schema
            json_schema = schema.model_json_schema()

            # Clean schema for Gemini compatibility
            # Gemini API doesn't support certain JSON schema properties that Pydantic includes
            # We need to recursively remove these unsupported properties
            def clean_schema_for_gemini(schema_dict):
                """
                Recursively clean schema for Gemini API compatibility.

                Gemini has several limitations:
                1. No additionalProperties support
                2. OBJECT types must have at least one property defined
                3. Free-form dicts (Dict[str, str]) are not supported directly

                For free-form dicts, we convert them to STRING type and add a description
                that the LLM should output JSON.
                """
                if not isinstance(schema_dict, dict):
                    return schema_dict

                # Remove unsupported properties FIRST
                unsupported_properties = [
                    "additionalProperties",
                    "title",  # Gemini doesn't need titles
                ]

                for prop in unsupported_properties:
                    schema_dict.pop(prop, None)

                # Now check if this is a free-form dict (object with no/empty properties)
                # This happens with Dict[str, str] in Pydantic models
                # After removing additionalProperties, these become empty objects
                is_empty_object = (
                    schema_dict.get("type") == "object" and
                    (
                        "properties" not in schema_dict or  # No properties key at all
                        len(schema_dict.get("properties", {})) == 0  # Or empty properties
                    )
                )

                if is_empty_object:
                    # Convert to STRING type with instruction to use JSON format
                    log.info(
                        "converting_free_form_dict_to_string",
                        original_schema=schema_dict
                    )
                    return {
                        "type": "string",
                        "description": (schema_dict.get("description", "Key-value pairs") +
                                      " (Provide as JSON string with key-value pairs)")
                    }

                # Recursively clean nested schemas
                if "properties" in schema_dict:
                    for key, value in schema_dict["properties"].items():
                        schema_dict["properties"][key] = clean_schema_for_gemini(value)

                if "items" in schema_dict:
                    schema_dict["items"] = clean_schema_for_gemini(schema_dict["items"])

                if "anyOf" in schema_dict:
                    schema_dict["anyOf"] = [clean_schema_for_gemini(s) for s in schema_dict["anyOf"]]

                if "allOf" in schema_dict:
                    schema_dict["allOf"] = [clean_schema_for_gemini(s) for s in schema_dict["allOf"]]

                if "$defs" in schema_dict:
                    for key, value in schema_dict["$defs"].items():
                        schema_dict["$defs"][key] = clean_schema_for_gemini(value)

                return schema_dict

            # Apply cleaning to the schema
            json_schema = clean_schema_for_gemini(json_schema)

            log.info(
                "schema_cleaned_for_gemini",
                schema_name=schema.__name__,
                removed_properties=["additionalProperties", "title"]
            )

            # Extract system instruction if provided
            system_instruction = kwargs.pop("system_instruction", None)
            if not system_instruction:
                system_instruction = kwargs.pop("system", None)

            # Build configuration with JSON mode and schema
            config = GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=system_instruction,
                # Enable JSON mode with schema validation
                response_mime_type="application/json",
                response_schema=json_schema,
                **kwargs,
            )

            # Call Google Gen AI API
            response = await self.client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )

            # Extract JSON text from response
            content_text = response.text

            # Parse JSON and validate against schema
            try:
                parsed_json = json.loads(content_text)

                # Post-process: Convert JSON string fields back to dicts
                # This is needed because we converted Dict[str, str] fields to strings for Gemini compatibility
                def parse_json_string_fields(obj):
                    """Recursively find and parse JSON string fields that should be dicts."""
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            # Check if this looks like a JSON string
                            if isinstance(value, str) and value.strip().startswith('{'):
                                try:
                                    obj[key] = json.loads(value)
                                except json.JSONDecodeError:
                                    # Not valid JSON, keep as string
                                    pass
                            elif isinstance(value, (dict, list)):
                                parse_json_string_fields(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            parse_json_string_fields(item)
                    return obj

                parsed_json = parse_json_string_fields(parsed_json)

                # Now validate with Pydantic
                parsed_output = schema.model_validate(parsed_json)
            except json.JSONDecodeError as e:
                log.error("json_parsing_failed", content=content_text, error=str(e))
                raise ValueError(f"Failed to parse JSON response: {e}")
            except Exception as e:
                log.error("schema_validation_failed", content=content_text, error=str(e))
                raise ValueError(f"Failed to validate against schema: {e}")

            # Extract usage metadata
            usage_metadata = (
                response.usage_metadata if hasattr(response, "usage_metadata") else None
            )

            if usage_metadata:
                input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(usage_metadata, "total_token_count", 0)
                cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0)
            else:
                input_tokens = output_tokens = total_tokens = cached_tokens = 0

            usage = UsageMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                provider=ProviderType.GOOGLE.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            log.info(
                "structured_output_generated",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                schema_name=schema.__name__,
            )

            return {"parsed": parsed_output, "usage": usage}

        except Exception as e:
            log.exception("structured_generation_failed", error=str(e), model=model)
            raise

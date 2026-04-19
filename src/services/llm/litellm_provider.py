"""
LiteLLM proxy provider for embedding generation.

Routes embedding calls through a LiteLLM proxy server (OpenAI-compatible REST API)
instead of calling Google directly. All other LLM operations (text generation,
structured output) are not implemented here — use GoogleProvider for those.

Why use LiteLLM instead of the direct Google SDK?
- Centralised key management through the proxy
- Built-in proxy-level rate limiting and caching
- Model aliasing (one proxy can serve multiple backends)
- Same Gemini model, same embeddings — just a different call path

The proxy must expose the OpenAI-compatible /v1/embeddings endpoint. The
request/response format follows the OpenAI Embeddings API spec, which LiteLLM
implements for all supported providers.

Configuration (set in .env):
    LITELLM_BASE_URL  — proxy base URL (default: http://localhost:4000)
    LITELLM_API_KEY   — Bearer token expected by the proxy

Model name convention with LiteLLM:
    Use the provider-prefixed format: "gemini/gemini-embedding-2-preview"
    (The gemini/ prefix tells LiteLLM which upstream provider to use.)
"""

from typing import Any

import httpx
import structlog

from .base import BaseLLMProvider, GenerationResponse, UsageMetrics, ProviderType

logger = structlog.get_logger()

# Generous timeout for proxy cold-starts or slow upstream responses
_HTTP_TIMEOUT = 60.0


class LiteLLMProvider(BaseLLMProvider):
    """Embedding provider that calls a LiteLLM proxy via HTTP REST.

    Only generate_embeddings() is implemented. Text generation and structured
    output raise NotImplementedError — use GoogleProvider for those operations.

    Attributes:
        base_url: LiteLLM proxy base URL (e.g. http://localhost:4000).
    """

    def __init__(self, base_url: str, api_key: str):
        """Initialise the LiteLLM provider.

        Args:
            base_url: Proxy base URL. Trailing slashes are stripped.
            api_key: Bearer token sent in the Authorization header.
        """
        super().__init__(api_key=api_key)
        self.base_url = base_url.rstrip("/")
        self.log.info(
            "litellm_provider_initialized",
            base_url=self.base_url,
            api_key_last4=self._mask_api_key(),
        )

    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "gemini/gemini-embedding-2-preview",
        **kwargs,
    ) -> dict[str, Any]:
        """Generate embeddings by calling the LiteLLM proxy.

        Sends one request per text (matching the GoogleProvider behaviour so that
        retry and rate-limit logic in embed.py works identically for both providers).

        Args:
            texts: List of texts to embed.
            model: LiteLLM model name — must include the provider prefix,
                   e.g. "gemini/gemini-embedding-2-preview".
            **kwargs: Ignored (accepted for interface compatibility).

        Returns:
            Dictionary with:
            - "embeddings": list of embedding vectors (list[list[float]])
            - "usage": UsageMetrics with token counts from the proxy response
        """
        log = self.log.bind(model=model, num_texts=len(texts))
        log.info("generating_embeddings_via_litellm")

        embeddings_list: list[list[float]] = []
        total_tokens = 0

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/v1/embeddings",
                    headers=headers,
                    json={"model": model, "input": text},
                )
                # Raises httpx.HTTPStatusError for 4xx/5xx — message includes the
                # status code string so _embed_one() rate-limit detection works.
                response.raise_for_status()

                data = response.json()
                embedding = data["data"][0]["embedding"]
                embeddings_list.append(embedding)

                usage = data.get("usage", {})
                total_tokens += usage.get("total_tokens", len(text.split()))

        usage_metrics = UsageMetrics(
            input_tokens=total_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
            provider=ProviderType.LITELLM.value,
            model=model,
            api_key_last4=self._mask_api_key(),
        )

        log.info(
            "embeddings_generated",
            num_embeddings=len(embeddings_list),
            embedding_dim=len(embeddings_list[0]) if embeddings_list else 0,
            total_tokens=total_tokens,
        )

        return {"embeddings": embeddings_list, "usage": usage_metrics}

    async def generate_text(self, prompt: str, model: str, **kwargs) -> GenerationResponse:
        raise NotImplementedError(
            "LiteLLMProvider only supports embeddings. Use GoogleProvider for text generation."
        )

    async def generate_structured(self, prompt: str, model: str, schema: Any, **kwargs) -> dict:
        raise NotImplementedError(
            "LiteLLMProvider only supports embeddings. Use GoogleProvider for structured output."
        )

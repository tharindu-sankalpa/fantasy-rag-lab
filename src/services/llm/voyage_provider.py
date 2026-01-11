# Dependencies:
# pip install voyageai structlog

"""
Voyage AI embeddings provider implementation.

Voyage AI specializes in high-quality embeddings for retrieval and semantic search.
They offer state-of-the-art embedding models that often outperform alternatives
on retrieval benchmarks.

Supported Models (as of 2025):
- voyage-3-large: Flagship model, best retrieval quality (1024 dims)
- voyage-3: Balanced quality/speed (1024 dims)
- voyage-3-lite: Fast and cost-effective (512 dims)
- voyage-3.5: Newer generation with multimodal support
- voyage-code-3: Specialized for code embeddings
- voyage-finance-2: Domain-specialized for financial text
- voyage-law-2: Domain-specialized for legal text

Key Features:
- Retrieval-optimized embeddings (outperform on MTEB benchmarks)
- Input type specification (query vs document) for better retrieval
- Configurable output dimensions (reduce dims for speed/storage)
- Multimodal embeddings (text, images, videos)
- Batch processing with automatic rate limiting
- Competitive pricing

Note: Voyage is embeddings-only (no text generation). For text generation,
use Anthropic, OpenAI, Google, or OpenRouter providers.
"""

from typing import Union, Optional, Literal, Any
import asyncio

import structlog
import voyageai

from .base import UsageMetrics, ProviderType

# Initialize module-level logger
logger = structlog.get_logger()


class VoyageProvider:
    """
    Provider for Voyage AI embeddings.

    Voyage AI specializes in high-quality embeddings for retrieval and RAG applications.
    This provider uses the official voyageai Python SDK.

    Unlike other providers, this does NOT inherit from BaseLLMProvider because it's
    embeddings-only and doesn't implement text generation methods. It has its own
    specialized interface focused on embedding generation.

    Architecture:
    - Thin wrapper around voyageai.Client
    - Async support via asyncio executor (SDK is synchronous)
    - Automatic batch processing
    - Token usage tracking
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Initialize Voyage AI embeddings provider.

        Args:
            api_key: Voyage API key (get from voyageai.com)
            **kwargs: Additional configuration (timeout, max_retries, etc.)
        """
        self.api_key = api_key
        # Initialize Voyage client (synchronous client)
        self.client = voyageai.Client(api_key=api_key, **kwargs)
        # Bind logger with provider name
        self.log = logger.bind(provider="VoyageProvider")
        self.log.info("voyage_provider_initialized", api_key=self._mask_api_key())

    def _mask_api_key(self) -> str:
        """
        Mask API key for safe logging (show only last 4 characters).

        Returns:
            Masked API key string (e.g., "...xyz1")
        """
        if len(self.api_key) > 4:
            return f"...{self.api_key[-4:]}"
        return "***"

    async def generate_embeddings(
        self,
        texts: Union[str, list[str]],
        model: str = "voyage-3-large",
        input_type: Optional[Literal["query", "document"]] = None,
        output_dimension: Optional[int] = None,
        truncation: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate embeddings for text(s).

        Voyage AI provides high-quality embeddings optimized for retrieval tasks.
        The input_type parameter allows optimizing embeddings for queries vs documents,
        which can improve retrieval performance.

        Args:
            texts: Single string or list of strings to embed
            model: Voyage model name (default: voyage-3-large)
            input_type: Optimize for 'query' or 'document' (improves retrieval when specified)
                - 'query': Use for search queries (what users type)
                - 'document': Use for corpus documents (what you're searching through)
                - None: General-purpose embeddings
            output_dimension: Output embedding dimension (None for default, or reduce for efficiency)
                - voyage-3-large default: 1024
                - Can be reduced to 512, 256, etc. for speed/storage savings
            truncation: If True, truncate inputs longer than max length; if False, error
            **kwargs: Additional parameters:
                - encoding_format: 'float' (default) or 'base64'

        Returns:
            Dictionary with:
            - 'embeddings': List of embedding vectors (list of lists of floats)
            - 'usage': UsageMetrics with token counts

        Raises:
            voyageai.error.VoyageError: If API request fails
            ValueError: If parameters are invalid

        Example:
            # For indexing documents
            result = await provider.generate_embeddings(
                texts=["Document 1 text", "Document 2 text"],
                model="voyage-3-large",
                input_type="document"
            )

            # For search queries
            query_result = await provider.generate_embeddings(
                texts="search query",
                model="voyage-3-large",
                input_type="query"
            )

            # With reduced dimensions for efficiency
            efficient_result = await provider.generate_embeddings(
                texts=["text 1", "text 2"],
                model="voyage-3-large",
                output_dimension=512  # Half size, faster
            )
        """
        log = self.log.bind(model=model, endpoint="generate_embeddings")

        try:
            # Normalize input to list for consistent processing
            if isinstance(texts, str):
                texts = [texts]
                was_single = True
            else:
                was_single = False

            log.info(
                "generating_embeddings",
                num_texts=len(texts),
                input_type=input_type,
                output_dimension=output_dimension,
                model=model,
            )

            # Call Voyage API
            # Note: Voyage SDK is synchronous, so we wrap in executor for async
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: self.client.embed(
                    texts=texts,
                    model=model,
                    input_type=input_type,
                    output_dimension=output_dimension,
                    truncation=truncation,
                    **kwargs,
                ),
            )

            # Extract embeddings and token count
            embeddings = result.embeddings
            total_tokens = result.total_tokens

            # Build usage metrics
            usage = UsageMetrics(
                total_tokens=total_tokens,
                # Embeddings don't have input/output distinction, just total
                input_tokens=total_tokens,  # For consistency
                output_tokens=0,
                provider=ProviderType.VOYAGE.value,
                model=model,
                api_key_last4=self._mask_api_key(),
            )

            # Calculate embedding dimensions (useful for debugging/validation)
            embedding_dim = len(embeddings[0]) if embeddings else 0

            log.info(
                "embeddings_generated",
                num_embeddings=len(embeddings),
                embedding_dim=embedding_dim,
                total_tokens=total_tokens,
                input_type=input_type,
            )

            # Return embeddings and usage
            return {
                "embeddings": embeddings[0] if was_single else embeddings,
                "usage": usage,
            }

        except Exception as e:
            log.exception("embedding_generation_failed", error=str(e), model=model)
            raise

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        model: str = "voyage-3-large",
        input_type: Optional[Literal["query", "document"]] = None,
        output_dimension: Optional[int] = None,
        batch_size: int = 128,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate embeddings for large batches with automatic batching.

        For very large corpora, this method automatically splits the input into
        smaller batches to avoid rate limits and optimize throughput.

        Args:
            texts: List of strings to embed (can be large)
            model: Voyage model name
            input_type: 'query' or 'document' for retrieval optimization
            output_dimension: Output embedding dimension
            batch_size: Maximum texts per API call (default: 128, Voyage's recommended size)
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
            - 'embeddings': List of all embedding vectors
            - 'usage': Aggregated UsageMetrics across all batches

        Example:
            # Embed 10,000 documents efficiently
            large_corpus = ["doc " + str(i) for i in range(10000)]
            result = await provider.generate_embeddings_batch(
                texts=large_corpus,
                model="voyage-3-large",
                input_type="document",
                batch_size=100
            )
            # Result contains all 10k embeddings
        """
        log = self.log.bind(model=model, endpoint="generate_embeddings_batch")

        log.info(
            "starting_batch_embeddings",
            total_texts=len(texts),
            batch_size=batch_size,
        )

        all_embeddings = []
        total_tokens_used = 0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            log.info(
                "processing_batch",
                batch_num=batch_num,
                total_batches=total_batches,
                batch_size=len(batch),
            )

            # Generate embeddings for this batch
            result = await self.generate_embeddings(
                texts=batch,
                model=model,
                input_type=input_type,
                output_dimension=output_dimension,
                **kwargs,
            )

            # Accumulate results
            all_embeddings.extend(result["embeddings"])
            total_tokens_used += result["usage"].total_tokens

        # Build aggregated usage metrics
        usage = UsageMetrics(
            total_tokens=total_tokens_used,
            input_tokens=total_tokens_used,
            output_tokens=0,
            provider=ProviderType.VOYAGE.value,
            model=model,
            api_key_last4=self._mask_api_key(),
        )

        log.info(
            "batch_embeddings_complete",
            total_embeddings=len(all_embeddings),
            total_tokens=total_tokens_used,
        )

        return {"embeddings": all_embeddings, "usage": usage}

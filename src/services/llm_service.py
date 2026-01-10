# Dependencies:
# pip install structlog langchain-openai langchain-google-genai langchain-voyageai

from typing import List, Optional, Any, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from src.core.config import settings
from src.utils.logger import logger
import os
import logging
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from tenacity.retry import retry_if_exception_message
import time

# Define a custom retry strategy for Resource Exhausted errors
def is_resource_exhausted(exception):
    return "429" in str(exception) or "RESOURCE_EXHAUSTED" in str(exception)


class LLMService:
    """
    Service for interacting with various LLM and Embedding providers.

    Attributes:
        log (structlog.stdlib.BoundLogger): Logger instance.
        openai_key (str): OpenAI API key.
        google_key (str): Google API key.
        voyage_key (str): Voyage AI API key.
        embedding_models (Dict[str, Any]): Dictionary of initialized embedding models.
        chat_model (Optional[ChatOpenAI]): Initialized OpenAI chat model.
        google_chat_model (Optional[ChatGoogleGenerativeAI]): Initialized Google chat model.
    """

    def __init__(self):
        """Initializes the LLMService with models based on available API keys."""
        self.log = logger.bind(component="llm_service")
        
        # Ensure env vars are loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        self.openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.google_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        self.voyage_key = settings.VOYAGE_API_KEY or os.getenv("VOYAGE_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        self.embedding_models: Dict[str, Any] = {}
        
        # 1. Voyage AI Embeddings (Preferred for this project: voyage-3-large)
        if self.voyage_key:
            try:
                self.embedding_models["voyage"] = VoyageAIEmbeddings(
                    voyage_api_key=self.voyage_key,
                    model="voyage-3-large" 
                )
                self.log.info("voyage_embeddings_initialized", model="voyage-3-large")
            except Exception as e:
                self.log.error("voyage_embeddings_init_failed", error=str(e))

        # 2. OpenAI Embeddings (text-embedding-3-large)
        if self.openai_key:
            try:
                self.embedding_models["openai"] = OpenAIEmbeddings(
                    api_key=self.openai_key, 
                    model="text-embedding-3-large"
                )
                self.log.info("openai_embeddings_initialized", model="text-embedding-3-large")
            except Exception as e:
                self.log.error("openai_embeddings_init_failed", error=str(e))

        # 3. Google Gemini Embeddings (text-embedding-004)
        if self.google_key:
            try:
                self.embedding_models["google"] = GoogleGenerativeAIEmbeddings(
                    google_api_key=self.google_key,
                    model="models/text-embedding-004"
                )
                self.log.info("google_embeddings_initialized", model="text-embedding-004")
            except Exception as e:
                self.log.error("google_embeddings_init_failed", error=str(e))

        if not self.embedding_models:
             self.log.warning("no_embedding_models_initialized")

        # Chat Model (Generic - defaults to GPT-4o)
        # TODO: Allow switching chat models too if needed
        # 4. Google Chat Model (Gemini)
        if self.google_key:
            try:
                self.google_chat_model = ChatGoogleGenerativeAI(
                    google_api_key=self.google_key,
                    model="gemini-3-pro-preview", 
                    temperature=0
                )
                self.log.info("google_chat_model_initialized", model="gemini-3-pro-preview")
            except Exception as e:
                self.log.error("google_chat_model_init_failed", error=str(e))
                self.google_chat_model = None
        else:
            self.google_chat_model = None

        # Chat Model (Generic - defaults to GPT-4o)
        # TODO: Allow switching chat models too if needed
        if self.openai_key:
            self.chat_model = ChatOpenAI(api_key=self.openai_key, model="gpt-4o")
        else:
            self.chat_model = None

        # 5. OpenRouter Chat Model
        if self.openrouter_key:
            try:
                self.openrouter_model = ChatOpenAI(
                    api_key=self.openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="google/gemini-3-pro-preview", # Default to a safe OpenRouter model
                    default_headers={"HTTP-Referer": "https://fantasy-rag-lab.com", "X-Title": "Fantasy RAG Lab"}
                )
                self.log.info("openrouter_chat_model_initialized", model="google/gemini-3-pro-preview")
            except Exception as e:
                self.log.error("openrouter_init_failed", error=str(e))
                self.openrouter_model = None
        else:
            self.openrouter_model = None

        # Quota Tracking State
        self._quota_window_start = time.time()
        self._quota_window_tokens = 0
        self._quota_window_requests = 0
        self._daily_tokens = 0
        self._daily_requests = 0
        self._quota_window_duration = 300  # 5 minutes in seconds
            
            
    def get_chat_model(self, provider: str, model_name: Optional[str] = None) -> Any:
        """Retrieves a chat model instance for the specified provider."""
        
        # Google
        if provider == "google":
            if not self.google_chat_model:
                raise ValueError("Google provider not initialized")
            # Return default if no specific model requested, or create new instance
            if model_name:
                return ChatGoogleGenerativeAI(
                    google_api_key=self.google_key,
                    model=model_name,
                    temperature=0
                )
            return self.google_chat_model

        # OpenAI
        elif provider == "openai":
            if not self.chat_model:
                 raise ValueError("OpenAI provider not initialized")
            if model_name:
                return ChatOpenAI(api_key=self.openai_key, model=model_name)
            return self.chat_model

        # OpenRouter
        elif provider == "openrouter":
            if not self.openrouter_key:
                raise ValueError("OpenRouter provider not initialized")
            
            target_model = model_name or "google/gemini-3-pro-preview"
            return ChatOpenAI(
                api_key=self.openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                model=target_model,
                default_headers={"HTTP-Referer": "https://fantasy-rag-lab.com", "X-Title": "Fantasy RAG Lab"}
            )
            
        # Anthropic (Claud)
        elif provider == "anthropic":
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_key:
                 raise ValueError("Anthropic provider not initialized (missing ANTHROPIC_API_KEY)")
            
            from langchain_anthropic import ChatAnthropic
            target_model = model_name or "claude-3-opus-20240229"
            return ChatAnthropic(
                api_key=anthropic_key,
                model_name=target_model,
                temperature=0
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _update_quota_metrics(self, usage_metadata: Dict[str, Any]):
        """Updates internal quota counters based on usage metadata."""
        current_time = time.time()
        
        # Reset window if 5 minutes have passed
        if current_time - self._quota_window_start >= self._quota_window_duration:
            self._quota_window_start = current_time
            self._quota_window_tokens = 0
            self._quota_window_requests = 0
            
        total_tokens = usage_metadata.get("total_tokens", 0)
        
        # Update counters
        self._quota_window_tokens += total_tokens
        self._quota_window_requests += 1
        self._daily_tokens += total_tokens
        self._daily_requests += 1

    def _get_quota_stats(self) -> Dict[str, Any]:
        """Returns current quota statistics."""
        return {
            "window_remaining_sec": int(self._quota_window_duration - (time.time() - self._quota_window_start)),
            "window_tokens": self._quota_window_tokens,
            "window_requests": self._quota_window_requests,
            "daily_tokens": self._daily_tokens,
            "daily_requests": self._daily_requests
        }

    async def generate_structured_response(
        self, 
        prompt: str, 
        schema: Any, 
        context: str = "", 
        provider: str = "google",
        model_name: str = None,
        fallback_providers: List[str] = None
    ) -> Any:
        """Generates a structured response with automatic failover."""
        log = self.log.bind(task="generate_structured_response")
        
        # Default fallback chain if not specified
        if fallback_providers is None:
            fallback_providers = []
            if provider == "google":
                # If Google fails, try OpenRouter (Gemini), then OpenAI
                if self.openrouter_key: fallback_providers.append("openrouter")
                if self.openai_key: fallback_providers.append("openai")
        
        # Create a queue of providers to try: [primary, *fallbacks]
        provider_queue = [provider] + fallback_providers
        
        last_exception = None
        
        for current_provider in provider_queue:
            log_p = log.bind(current_provider=current_provider)
            
            try:
                # Resolve model name for the current provider
                # If current_provider matches the requested provider, use the requested model_name
                # Otherwise, rely on defaults or mappings (simple pass-through for now)
                # Ideally, we'd map 'gemini-3-pro' -> 'google/gemini-3-pro' for OpenRouter, etc.
                
                target_model_name = model_name if current_provider == provider else None
                
                model = self.get_chat_model(current_provider, model_name=target_model_name)
                
                # Special handling for OpenRouter models if we want to mimic the requested Google model
                if current_provider == "openrouter" and provider == "google" and not target_model_name:
                    # Swap to an equivalent OpenRouter model if no specific fallback model was computed
                    model = self.get_chat_model("openrouter", model_name="google/gemini-3-pro-preview")

                log_p.info("attempting_generation", context_len=len(context))
                
                
                # Use include_raw=True to get metadata
                
                # Truncate context if strictly necessary (Naive truncation)
                # Max context for many models is 128k ~ 200k tokens. 
                # 250k char ~ 60k tokens. Wait, the error said 255k TOKENS. That's huge (~1M chars).
                # The user's prompt includes "TEXT CONTENT:\n{text_content}". 
                # If text_content is massive, we must split or truncate.
                # Here we implement a safety truncation to ~120,000 tokens (approx 480,000 chars) to stay safe for large context models.
                # For smaller models, this might still fail, but it prevents the massive overflow seen in logs.
                
                max_chars = 480000 
                if len(context) > max_chars:
                    log_p.warning("context_truncated", original_len=len(context), new_len=max_chars)
                    context = context[:max_chars] + "...[TRUNCATED]"

                structured_llm = model.with_structured_output(schema, include_raw=True)
                full_prompt = f"Context:\n{context}\n\nTask: {prompt}"

                # Retry logic for *this specific provider*
                async for attempt in tenacity.AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=2, min=4, max=20),
                    retry=retry_if_exception_type(Exception), # Broad retry for now, can refine
                    before_sleep=before_sleep_log(logger, logging.WARNING)
                ):
                    with attempt:
                        # result is a dict with 'parsed', 'raw', 'parsing_error'
                        result = await structured_llm.ainvoke(full_prompt)
                        
                        parsed_response = result.get("parsed")
                        raw_response = result.get("raw")

                        # Extract metadata
                        usage_metadata = {}
                        model_version = "unknown"
                        
                        if raw_response:
                            usage_metadata = raw_response.usage_metadata or {}
                            # Try to extract model version from response metadata if available
                            if hasattr(raw_response, 'response_metadata'):
                                model_version = raw_response.response_metadata.get('model_version', 'unknown')

                        # Fallback/Confirmation of model name from instance
                        if model_version == "unknown":
                            if hasattr(model, 'model_name'): # OpenAI / OpenRouter
                                model_version = model.model_name
                            elif hasattr(model, 'model'): # Google
                                model_version = model.model

                        # Extract and mask API Key
                        masked_key = "unknown"
                        api_key_val = None
                        
                        # Try to find the key attribute
                        if hasattr(model, 'api_key'):
                            api_key_val = model.api_key
                        elif hasattr(model, 'google_api_key'):
                            api_key_val = model.google_api_key
                        elif hasattr(model, 'openai_api_key'):
                            api_key_val = model.openai_api_key
                        elif hasattr(model, 'anthropic_api_key'):
                            api_key_val = model.anthropic_api_key
                            
                        # Extract string from SecretStr if needed
                        if api_key_val:
                            s_val = api_key_val.get_secret_value() if hasattr(api_key_val, 'get_secret_value') else str(api_key_val)
                            if len(s_val) > 4:
                                masked_key = f"...{s_val[-4:]}"
                            else:
                                masked_key = "***"

                        self._update_quota_metrics(usage_metadata)
                        quota_stats = self._get_quota_stats()

                        log_p.info("generation_success", 
                                   usage=usage_metadata,
                                   quota=quota_stats,
                                   model=model_version,
                                   api_key_used=masked_key)
                                   
                        return parsed_response

            except Exception as e:
                log_p.warning("provider_attempt_failed", error=str(e))
                last_exception = e
                # Continue to next provider in queue
        
        # If we exhaust all providers
        log.error("all_providers_failed")
        raise last_exception

    async def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate a text response from the chat model.

        Args:
            prompt: The user query or prompt.
            context: Additional context to include (e.g., RAG context).

        Returns:
            str: The generated response text.

        Raises:
            ValueError: If the chat model is not initialized.
        """
        log = self.log.bind(task="generate_response")
        if not self.chat_model:
            log.error("chat_model_missing")
            raise ValueError("Chat model not initialized (missing API key)")
        
        log.info("generating_response_start", prompt_length=len(prompt), context_length=len(context))
        
        # Basic generation (can be expanded for specific RAG prompts)
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant specialized in fantasy literature."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
        
        try:
            response = await self.chat_model.ainvoke(messages)
            log.info("response_generated", response_length=len(response.content))
            return str(response.content)
        except Exception as e:
            log.exception("response_generation_failed")
            raise e
        
    def get_embedding_model(self, model_name: str = "voyage") -> Any:
        """
        Retrieve a specific embedding model instance.

        Args:
            model_name: The name of the model to retrieve (default: "voyage").

        Returns:
            Any: The embedding model instance (LangChain compatible).

        Raises:
            ValueError: If no models are available.
        """
        model = self.embedding_models.get(model_name)
        if not model:
            # Fallback logic if preferred model isn't available
            if self.embedding_models:
                fallback_name = list(self.embedding_models.keys())[0]
                self.log.warning("requested_model_not_found_fallback", requested=model_name, fallback=fallback_name)
                return self.embedding_models[fallback_name]
            else:
                self.log.error("no_embedding_models_available", requested=model_name)
                raise ValueError(f"No embedding models available. Cannot satisfy request for '{model_name}'.")
        return model

    def generate_embedding(self, text: str, model_name: str = "voyage") -> List[float]:
        """
        Generate embedding for a single string.

        Args:
            text: The input text.
            model_name: The embedding model to use.

        Returns:
            List[float]: The generated embedding vector.
        """
        model = self.get_embedding_model(model_name)
        
        # Log start
        self.log.info("generating_embedding_single", model=model_name, text_snippet=text[:50] + "...")
        
        vector = model.embed_query(text)
        
        # Log end with stats
        self.log.info("embedding_generated", model=model_name, vector_size=len(vector))
        return vector

    def generate_embeddings_batch(self, texts: List[str], model_name: str = "voyage") -> List[List[float]]:
        """
        Generate embeddings for a batch of strings.

        Args:
            texts: List of input texts.
            model_name: The embedding model to use.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        model = self.get_embedding_model(model_name)
        
        self.log.info("generating_embeddings_batch", model=model_name, batch_size=len(texts))
        
        vectors = model.embed_documents(texts)
        
        if vectors:
             self.log.info("batch_embeddings_generated", model=model_name, count=len(vectors), dim=len(vectors[0]))
        else:
             self.log.warning("batch_embedding_result_empty", model=model_name)
             
        return vectors

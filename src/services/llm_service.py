# Dependencies:
# pip install structlog langchain-openai langchain-google-genai langchain-voyageai

import structlog
from typing import List, Optional, Any, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from src.core.config import settings
from src.utils.logger import logger

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
        self.openai_key = settings.OPENAI_API_KEY
        self.google_key = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY
        self.voyage_key = settings.VOYAGE_API_KEY
        
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
            
    async def generate_structured_response(self, prompt: str, schema: Any, context: str = "", provider: str = "google") -> Any:
        """Generates a response from the LLM that strictly adheres to a Pydantic schema.

        Args:
            prompt (str): The specific task or question for the LLM.
            schema (Any): The Pydantic model class to enforce the output structure.
            context (str, optional): Background information or source text. Defaults to "".
            provider (str, optional): The LLM provider to use ('google' or 'openai'). Defaults to "google".

        Returns:
            Any: An instance of the provided Pydantic schema populated with the LLM's response.

        Raises:
            ValueError: If the requested provider is not initialized.
            Exception: If generation fails.
        """
        log = self.log.bind(task="generate_structured_response", provider=provider)
        
        model = None
        if provider == "google" and self.google_chat_model:
            model = self.google_chat_model
        elif provider == "openai" and self.chat_model:
            model = self.chat_model
            
        if not model:
            log.error("requested_model_provider_not_available", provider=provider)
            raise ValueError(f"Provider {provider} not initialized")

        log.info("generating_structured", prompt_len=len(prompt), context_len=len(context))
        
        structured_llm = model.with_structured_output(schema)
        
        full_prompt = f"Context:\n{context}\n\nTask: {prompt}"
        
        try:
            response = await structured_llm.ainvoke(full_prompt)
            log.info("structured_response_generated")
            return response
        except Exception as e:
            log.exception("structured_generation_failed")
            raise e

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

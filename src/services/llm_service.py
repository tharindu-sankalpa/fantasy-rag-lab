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
        chat_model (Optional[ChatOpenAI]): Initialized chat model.
    """

    def __init__(self):
        """
        Initialize the LLMService and set up embedding/chat models based on available API keys.
        """
        self.log = logger.bind(component="llm_service")
        self.openai_key = settings.OPENAI_API_KEY
        self.google_key = settings.GOOGLE_API_KEY
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
        if self.openai_key:
            self.chat_model = ChatOpenAI(api_key=self.openai_key, model="gpt-4o")
        else:
            self.chat_model = None

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

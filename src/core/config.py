# Dependencies:
# pip install pydantic-settings

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This class defines the configuration schema for the application, validating
    environment variables using Pydantic. It supports loading from a `.env` file.

    Attributes:
        PROJECT_NAME (str): Name of the application. Defaults to "Fantasy RAG Lab".
        VERSION (str): Current application version. Defaults to "0.1.0".
        API_PREFIX (str): Prefix for all API endpoints. Defaults to "/api/v1".
        
        OPENAI_API_KEY (str): API key for OpenAI services.
        GOOGLE_API_KEY (str): Legacy Google Cloud API Key.
        GEMINI_API_KEY (str): API key for Google Gemini models (AI Studio).
        VOYAGE_API_KEY (str): API key for Voyage AI embeddings.
        
        MILVUS_URI (str): Connection URI for Milvus/Zilliz Cloud vector database.
        MILVUS_TOKEN (str): Authentication token for Milvus/Zilliz.
        
        NEO4J_URI (str): Connection URI for Neo4j Graph Database (Bolt protocol).
        NEO4J_USERNAME (str): Username for Neo4j authentication.
        NEO4J_PASSWORD (str): Password for Neo4j authentication.
        
        CHUNK_SIZE (int): default size of text chunks for RAG. Defaults to 1000.
        CHUNK_OVERLAP (int): Default overlap between chunks. Defaults to 200.
    """
    PROJECT_NAME: str = "Fantasy RAG Lab"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    
    # OpenAI / LLM
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    
    # Milvus / Zilliz
    MILVUS_URI: str = ""
    MILVUS_TOKEN: str = ""
    
    # Neo4j
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""
    
    # Text Processing (Ingestion)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Entity Extraction (ETL Sizing)
    # Target: ~800k tokens to maximize context window (1M limit) with safety buffer.
    # 800,000 tokens * 4 chars/token = 3,200,000 chars
    ETL_TARGET_TOKEN_SIZE: int = 800000 
    ETL_OVERLAP_TOKEN_SIZE: int = 4000 # ~16k chars overlap for continuity

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields in .env that aren't defined here
        extra="ignore"
    )

settings = Settings()

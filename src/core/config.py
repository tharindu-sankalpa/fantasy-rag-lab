# Dependencies:
# pip install pydantic-settings

"""
Application configuration settings.

This module defines the configuration schema for the application, loaded from
environment variables using Pydantic. It supports loading from a `.env` file.

Configuration Categories:
- Application: Project name, version, API prefix
- Google Cloud: Gemini API keys, Vertex AI settings
- MongoDB Atlas: Connection URI and database name
- Vector/Graph DBs: Milvus and Neo4j settings
- Text Processing: Chunking and ETL parameters
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        PROJECT_NAME: Name of the application. Defaults to "Fantasy RAG Lab".
        VERSION: Current application version. Defaults to "0.1.0".
        API_PREFIX: Prefix for all API endpoints. Defaults to "/api/v1".

        GOOGLE_API_KEY: Google Cloud API Key (Developer API).
        GEMINI_API_KEY: API key for Google Gemini models (AI Studio).
        GOOGLE_GENAI_USE_VERTEXAI: Whether to use Vertex AI instead of Developer API.
        GOOGLE_CLOUD_PROJECT: GCP project ID for Vertex AI.
        GOOGLE_CLOUD_LOCATION: GCP region for Vertex AI.

        MONGODB_URI: MongoDB Atlas connection URI.
        MONGODB_DATABASE: MongoDB database name.

        MILVUS_URI: Connection URI for Milvus/Zilliz Cloud vector database.
        MILVUS_TOKEN: Authentication token for Milvus/Zilliz.

        NEO4J_URI: Connection URI for Neo4j Graph Database (Bolt protocol).
        NEO4J_USERNAME: Username for Neo4j authentication.
        NEO4J_PASSWORD: Password for Neo4j authentication.

        CHUNK_SIZE: Default size of text chunks for RAG. Defaults to 1000.
        CHUNK_OVERLAP: Default overlap between chunks. Defaults to 200.

        ETL_TARGET_TOKEN_SIZE: Target token size for ETL processing.
        ETL_OVERLAP_TOKEN_SIZE: Overlap tokens for ETL continuity.
    """

    PROJECT_NAME: str = "Fantasy RAG Lab"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    # Google Cloud (ONLY LLM PROVIDER)
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GOOGLE_GENAI_USE_VERTEXAI: bool = False
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "us-central1"

    # MongoDB Atlas
    MONGODB_URI: str = ""
    MONGODB_DATABASE: str = "fantasy_rag"

    # Milvus / Zilliz
    MILVUS_URI: str = ""
    MILVUS_TOKEN: str = ""

    # Neo4j
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""

    # Text Processing (RAG Ingestion)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Entity Extraction (ETL Sizing)
    # Target: ~800k tokens to maximize context window (1M limit) with safety buffer.
    # 800,000 tokens * 4 chars/token = 3,200,000 chars
    ETL_TARGET_TOKEN_SIZE: int = 800000
    ETL_OVERLAP_TOKEN_SIZE: int = 4000  # ~16k chars overlap for continuity

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields in .env that aren't defined here
        extra="ignore",
    )


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Fantasy RAG Lab"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    
    # OpenAI / LLM
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    VOYAGE_API_KEY: str = ""
    
    # Milvus / Zilliz
    MILVUS_URI: str = ""
    MILVUS_TOKEN: str = ""
    
    # Neo4j
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""
    
    # Text Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields in .env that aren't defined here
        extra="ignore"
    )

settings = Settings()

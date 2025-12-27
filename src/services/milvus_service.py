# Dependencies:
# pip install pymilvus structlog

import structlog
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from src.core.config import settings
from src.utils.logger import logger

class MilvusService:
    """
    Service for interacting with Milvus/Zilliz Cloud vector database.
    
    Attributes:
        uri (str): Milvus URI.
        token (str): Authentication token.
        collection_name (str): Name of the collection.
        log (structlog.stdlib.BoundLogger): Logger instance.
    """

    def __init__(self):
        """
        Initialize the Milvus Service and establish connection.
        
        Raises:
            Exception: If connection to Milvus fails.
        """
        self.log = logger.bind(component="milvus_service")
        self.uri = settings.MILVUS_URI
        self.token = settings.MILVUS_TOKEN
        self.collection_name = "fantasy_rag_collection"
        
        try:
            connections.connect(
                alias="default", 
                uri=self.uri, 
                token=self.token
            )
            self.log.info("milvus_connected", uri=self.uri)
        except Exception as e:
            self.log.error("milvus_connect_failed", error=str(e))
            raise e

    def create_collection_if_not_exists(self) -> Collection:
        """
        Creates the collection with the specific schema if it doesn't exist.
        
        Returns:
            Collection: The loaded Milvus Collection object.
        """
        log = self.log.bind(collection=self.collection_name)
        
        if utility.has_collection(self.collection_name):
            log.info("collection_exists")
            return Collection(self.collection_name)

        log.info("creating_new_collection")
        
        # Define Schema
        fields = [
            # ID field
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # Embedding field (1536 dim for OpenAI)
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            # Text content
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            # Metadata fields for filtering
            FieldSchema(name="universe", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=200)
        ]
        
        schema = CollectionSchema(fields=fields, description="Fantasy RAG Collection")
        
        collection = Collection(
            name=self.collection_name, 
            schema=schema, 
            consistency_level="Strong"
        )
        
        # Create Index for faster search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX", # Zilliz Cloud optimized
            "params": {}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        log.info("collection_created_and_indexed")
        
        return collection

    def insert_documents(self, documents: List[Dict[str, Any]]) -> Any:
        """
        Insert processed documents into Milvus.

        Args:
            documents: List of dictionaries matching the schema.
                       Expected keys: embedding, text, universe, book_title, chapter.

        Returns:
            MutationResult: Result of the insertion operation.

        Raises:
            Exception: If insertion fails.
        """
        collection = Collection(self.collection_name)
        
        # Flatten documents to match schema
        data_rows = []
        for doc in documents:
            row = {
                "embedding": doc.get("embedding"),
                "text": doc.get("text")[:65535], # Truncate if too long safety check
                "universe": doc.get("universe", "Unknown"),
                "book_title": doc.get("book_title", "Unknown"),
                "chapter": doc.get("chapter", "Unknown")
            }
            data_rows.append(row)
            
        try:
            res = collection.insert(data_rows)
            # collection.flush() # Heavy operation, use sparsely in loop
            self.log.info("documents_inserted", count=len(data_rows), ids_count=len(res.primary_keys))
            return res
        except Exception as e:
            self.log.error("insert_failed", error=str(e))
            raise e

    def search(self, query_embedding: List[float], top_k: int = 20, universe: Optional[str] = None) -> Any:
        """
        Search for relevant documents using vector similarity.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            universe: Optional filter by universe.

        Returns:
            SearchResult: Milvus search results.
        """
        collection = Collection(self.collection_name)
        collection.load() # Ensure collection is loaded
        
        search_params = {
            "metric_type": "COSINE", 
            "params": {}
        }
        
        expr = None
        if universe:
            expr = f"universe == '{universe}'"
            
        results = collection.search(
            data=[query_embedding], 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k, 
            expr=expr,
            output_fields=["text", "universe", "book_title", "chapter"]
        )
        
        return results

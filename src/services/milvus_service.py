import structlog
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from src.core.config import settings

logger = structlog.get_logger()

class MilvusService:
    def __init__(self):
        """
        Initialize the Milvus Service.
        Establishes a connection to the Milvus/Zilliz server.
        """
        self.uri = settings.MILVUS_URI
        self.token = settings.MILVUS_TOKEN
        self.collection_name = "fantasy_rag_collection"
        
        # Connect to Milvus
        try:
            connections.connect(
                alias="default", 
                uri=self.uri, 
                token=self.token
            )
            logger.info("Connected to Milvus", uri=self.uri)
        except Exception as e:
            logger.error("Failed to connect to Milvus", error=str(e))
            raise e

    def create_collection_if_not_exists(self):
        """
        Creates the collection with the specific schema if it doesn't exist.
        """
        if utility.has_collection(self.collection_name):
            logger.info("Collection already exists", collection=self.collection_name)
            return Collection(self.collection_name)

        logger.info("Creating new collection", collection=self.collection_name)
        
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
        logger.info("Collection created and indexed", collection=self.collection_name)
        
        return collection

    def insert_documents(self, documents: list):
        """
        Insert processed documents into Milvus.
        Expects a list of dictionaries with keys matching the schema.
        """
        collection = Collection(self.collection_name)
        
        # Prepare data for insertion (column-based format for some Milvus versions, but PyMilvus supports row-based too now)
        # But safest is usually column-based or list of dicts if using high level client.
        # Here using ORM-style Collection object, which accepts list of rows?
        # Actually Collection.insert expects list of lists (columns) or list of dicts.
        # Let's verify data format.
        
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
            logger.info("Inserted documents", count=len(data_rows), ids=res.primary_keys)
            return res
        except Exception as e:
            logger.error("Failed to insert documents", error=str(e))
            raise e

    def search(self, query_embedding: list, top_k: int = 20, universe: str = None):
        """
        Search for relevant documents.
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

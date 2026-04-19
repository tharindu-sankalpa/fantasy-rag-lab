# Dependencies:
# pip install pymilvus structlog

"""Milvus / Zilliz Cloud vector database service.

Supports:
- Dense vector search (HNSW, COSINE) on Gemini 3072-dim embeddings
- Full-text search (BM25 sparse vectors) on the text field
- Hybrid search combining both via RRF reranking
"""

import structlog
from typing import Any, Optional

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    RRFRanker,
    WeightedRanker,
    connections,
    utility,
)

from src.core.config import settings
from src.utils.logger import logger


class MilvusService:
    """Service for interacting with Milvus/Zilliz Cloud vector database.

    Attributes:
        uri: Milvus connection URI.
        token: Authentication token.
        collection_name: Target collection name.
    """

    COLLECTION_NAME = "fantasy_rag_collection"

    def __init__(self):
        self.log = logger.bind(component="milvus_service")
        self.uri = settings.MILVUS_URI
        self.token = settings.MILVUS_TOKEN
        self.collection_name = self.COLLECTION_NAME

        try:
            connections.connect(alias="default", uri=self.uri, token=self.token)
            self.log.info("milvus_connected", uri=self.uri)
        except Exception as e:
            self.log.error("milvus_connect_failed", error=str(e))
            raise

    def create_collection_if_not_exists(self) -> Collection:
        """Create the collection with HNSW + BM25 indexes if it doesn't exist.

        Schema:
          - embedding: FLOAT_VECTOR(3072)  → HNSW index (dense search)
          - text:      VARCHAR             → BM25 Function source (full-text search)
          - text_sparse: SPARSE_FLOAT_VECTOR → BM25 output (auto-populated)
          - metadata fields: chunk_id, series, universe, book_title, chapter_*

        Returns:
            Loaded Collection object ready for insert and search.
        """
        log = self.log.bind(collection=self.collection_name)

        if utility.has_collection(self.collection_name):
            log.info("collection_exists")
            return Collection(self.collection_name)

        log.info("creating_new_collection")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # Dense vector — Gemini Embedding 2 Preview (3072 dims)
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
            # Source text — analyzer enabled so BM25 Function can tokenize it
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535,
                        enable_analyzer=True),
            # BM25 sparse vector — auto-populated by the Function below, never set manually
            FieldSchema(name="text_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # Metadata
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="series", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="universe", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="book_title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chapter_number", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chapter_title", dtype=DataType.VARCHAR, max_length=1000),
        ]

        # BM25 Function: text → text_sparse (runs server-side at insert and query time)
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["text_sparse"],
        )

        schema = CollectionSchema(
            fields=fields,
            functions=[bm25_function],
            description="Fantasy RAG Collection — dense + BM25 hybrid search",
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level="Strong",
        )

        # HNSW — max quality: M=64 (graph connections), efConstruction=512 (build depth)
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 64, "efConstruction": 512},
            },
        )
        log.info("hnsw_index_created", field="embedding", M=64, efConstruction=512)

        # Sparse inverted index for BM25 full-text search
        collection.create_index(
            field_name="text_sparse",
            index_params={
                "metric_type": "BM25",
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"bm25_k1": 1.5, "bm25_b": 0.75},
            },
        )
        log.info("bm25_index_created", field="text_sparse")

        # Zilliz Serverless auto-loads on first query — explicit load() not supported
        return collection

    def insert_documents(self, documents: list[dict[str, Any]]) -> Any:
        """Insert a batch of documents. text_sparse is NOT included — BM25 fills it.

        Args:
            documents: List of dicts with keys: embedding, text, chunk_id, series,
                       universe, book_title, chapter_number, chapter_title.

        Returns:
            MutationResult from Milvus.
        """
        collection = Collection(self.collection_name)

        data_rows = [
            {
                "embedding": doc["embedding"],
                "text": str(doc.get("text", ""))[:65535],
                "chunk_id": str(doc.get("chunk_id", "")),
                "series": str(doc.get("series", "")),
                "universe": str(doc.get("universe", "")),
                "book_title": str(doc.get("book_title", "")),
                "chapter_number": str(doc.get("chapter_number", "")),
                "chapter_title": str(doc.get("chapter_title", "")),
            }
            for doc in documents
        ]

        try:
            res = collection.insert(data_rows)
            self.log.info("documents_inserted", count=len(data_rows),
                          ids_count=len(res.primary_keys))
            return res
        except Exception as e:
            self.log.error("insert_failed", error=str(e))
            raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        universe: Optional[str] = None,
    ) -> Any:
        """Dense vector search only (ANN on HNSW index).

        Args:
            query_embedding: 3072-dim query vector.
            top_k: Number of results.
            universe: Optional metadata filter.

        Returns:
            Milvus SearchResult.
        """
        collection = Collection(self.collection_name)
        expr = f"universe == '{universe}'" if universe else None

        # ef >= top_k; 512 = maximum recall at query time
        return collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 512}},
            limit=top_k,
            expr=expr,
            output_fields=["text", "chunk_id", "universe", "book_title",
                           "chapter_number", "chapter_title", "series"],
        )

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 20,
        universe: Optional[str] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> Any:
        """Hybrid search: dense HNSW + BM25 sparse, reranked with RRF.

        Args:
            query_embedding: 3072-dim dense query vector.
            query_text: Raw query string for BM25 (tokenized server-side).
            top_k: Final number of results after reranking.
            universe: Optional metadata filter applied to both legs.
            dense_weight: RRF weight for the dense leg (default 0.7).
            sparse_weight: RRF weight for the BM25 leg (default 0.3).

        Returns:
            Milvus search results after RRF reranking.
        """
        collection = Collection(self.collection_name)
        collection.load()

        expr = f"universe == '{universe}'" if universe else None
        output_fields = ["text", "chunk_id", "universe", "book_title",
                         "chapter_number", "chapter_title", "series"]

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 512}},
            limit=top_k,
            expr=expr,
        )

        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="text_sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
            expr=expr,
        )

        return collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=WeightedRanker(dense_weight, sparse_weight),
            limit=top_k,
            output_fields=output_fields,
        )

# Fantasy RAG Lab

## Project Concept: Building and Evaluating Advanced RAG Systems Using Fantasy Literature

This project involves building and evaluating multiple Retrieval-Augmented Generation (RAG) architectures—specifically **Graph RAG**, **Agentic RAG**, and **Hybrid RAG**—using well-known fantasy book universes as the knowledge domain.

The goal is to master advanced techniques like query optimization, probability-based ranking, and re-ranking key strategies, moving beyond naïve RAG implementations.

### Core Idea and Motivation

By using familiar fantasy worlds (e.g., _The Wheel of Time_, _A Song of Ice and Fire_, _Harry Potter_), we can manually evaluate RAG outputs using personal domain knowledge. This makes validating response accuracy and relevance significantly more reliable than using unfamiliar datasets.

---

## High-Level Architecture

The system is split into two decoupled subsystems:

1.  **The Ingestion Engine (Offline/Batch)**: Processes raw books into Vectors (Milvus) and Knowledge Graphs (Neo4j).
2.  **The Retrieval API (Online/Real-time)**: A FastAPI service with distinct endpoints for each RAG flavor.

### Phase 1: The Ingestion Pipeline ("The Grimoire")

A standalone Python module that runs on-demand to populate databases.

- **Data Processing**:
  - **Text Extraction**: Use Unstructured or PyMuPDF for EPUBs/PDFs.
  - **Chunking**: Semantic Chunking or RecursiveCharacterTextSplitter with overlap to preserve context.
  - **Metadata Extraction**: Critical for filtering (Universe, Series Order, Book Title, Chapter, Page Number).
- **Vector Database (Milvus/Zilliz Cloud)**:
  - Stores embeddings (e.g., OpenAI/Cohere 1536 dim).
  - Rich metadata for hybrid search and filtering.
- **Knowledge Graph (Neo4j)**:
  - Uses LangChain’s `LLMGraphTransformer` to extract entities (Structure: `(Subject)-[RELATION]->(Object)`).
  - Focuses on fantasy-relevant entities (Characters, Spells, Locations).

### Phase 2: The API Service ("The Portal")

A **FastAPI** service designed for modularity and asynchronous operations.

**Proposed Structure**:

```
/src (or /fantasy_rag)
├── /api
│   ├── /endpoints
│   │   ├── rag_naive.py
│   │   ├── rag_advanced.py  (Query Exp + Re-ranking)
│   │   ├── rag_hybrid.py    (Vector + Keyword + RRF)
│   │   ├── rag_graph.py     (Neo4j)
│   │   └── rag_agentic.py   (LangGraph)
├── /core
│   ├── config.py            (Env vars, DB connections)
│   └── logging.py           (Structlog setup)
├── /services
│   ├── milvus_service.py
│   ├── neo4j_service.py
│   └── llm_service.py
└── main.py
```

### Phase 3: RAG Flavors & Strategies

1.  **Hybrid RAG**:
    - **Logic**: Parallel search (Vector + BM25/Keyword) -> Reciprocal Rank Fusion (RRF) -> Generation.
    - **Why**: Combines semantic understanding with exact keyword matching (critical for proper nouns in fantasy).
2.  **Graph RAG**:
    - **Logic**: Entity Extraction -> Graph Traversal (Neo4j) -> Context Injection -> Vector Fallback.
    - **Why**: Captures structured relationships (e.g., "Lanfear is Mierin Eronaile") that vector search might miss.
3.  **Agentic RAG**:
    - **Logic**: Uses **LangGraph** for planning.
      - _Router_: Analyzes question (Metadata filter).
      - _Retrieval_: Fetches docs.
      - _Grader_: checks if text answers the question (Self-Correction loop).
    - **Why**: Handles complex multi-step reasoning.

### Evaluaton & Deployment

- **Evaluation**:
  - **Golden Dataset**: 50+ manually verified questions.
  - **Metrics**: Faithfulness, Answer Relevancy (using Ragas/DeepEval).
  - **Manual Review**: Tracing agent decision paths.
- **Deployment**:
  - API Dockerized and deployed via Kubernetes (AKS).
  - Databases managed via Cloud (Zilliz, Neo4j Aura).

---

## Tech Stack

- **Language**: Python
- **Package Management**: `uv`, `pyenv`
- **Frameworks**: FastAPI, LangChain, LangGraph
- **Databases**:
  - Vector: Milvus (Zilliz Cloud)
  - Graph: Neo4j (Aura)
- **Infrastructure**: Kubernetes (AKS), Docker
- **Logging**: Structlog

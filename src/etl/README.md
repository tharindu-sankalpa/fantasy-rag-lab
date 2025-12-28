# ETL Pipeline Architecture

This module handles the extraction, transformation, and loading of data from raw sources (EPUBs) into our downstream systems.

A key architectural pattern in this project is the use of **two distinct extraction strategies** for the same source data, optimized for different consumption patterns.

## The Dual-Pipeline Strategy

We deliberately process EPUB files in two different ways depending on whether the destination is the Knowledge Graph or the Vector Database.

| Feature    | Graph Extraction (`src/etl/graph_extraction`)      | RAG Ingestion (`src/etl/rag_ingestion`)              |
| :--------- | :------------------------------------------------- | :--------------------------------------------------- |
| **Logic**  | Manual (`ebooklib` + `BeautifulSoup`)              | Automated (`Unstructured` + `LangChain`)             |
| **Output** | **Continuous Text** (1 Book ≈ 1 huge String)       | **Chunks** (1 Book ≈ 5,000 small Documents)          |
| **Goal**   | Maintain perfect narrative flow for LLM analysis.  | optimize for semantic search and metadata retrieval. |
| **Method** | Iterates specific HTML tags; custom joining logic. | Standard parser following the EPUB "spine".          |

### 1. Graph Extraction (`extract_epubs.py`)

**Purpose:** Feed massive contexts into Gemini Pro (2M context window) to extract complex entities and relationships.

- **Why Manual?**
  - **Narrative Continuity:** We need the text to read exactly like a book. Automated tools often insert generic newlines or headers that interrupt sentences. We manually control paragraph joining (`\n\n`) to ensure the LLM never loses the thread of a sentence across a page break.
  - **Defensive Parsing:** This script iterates through **all** internal HTML files in the EPUB package, ignoring the official "Spine" (Table of Contents). This ensures that even in malformed EPUBs, we extract every scrap of text so no entity is missed.
  - **Noise Reduction:** We employ strict heuristics (e.g., removing blocks < 50 chars) to strip page numbers and headers which confuse entity extraction models.

### 2. RAG Ingestion (`processor.py`)

**Purpose:** Create discrete, searchable chunks for Vector Search.

- **Why Unstructured?**
  - **Metadata Focus:** RAG relies on knowing _where_ a piece of information came from (Chapter 5, Prologue, etc.). `Unstructured` standardizes the document format, allowing us to focus our code on Regex logic to extract Chapter Metadata.
  - **Standardization:** The output needs to be compatible with `LangChain`'s `RecursiveCharacterTextSplitter`. Using the standard `UnstructuredEPubLoader` provides us with `Document` objects that fit natively into the rest of the LangChain ecosystem.

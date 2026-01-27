# Unified EPUB ETL & MongoDB Ingestion Guide

## High-Level Overview

This repository uses a **model-agnostic** ETL pipeline designed to ingest fantasy books into two distinct data structures: **Vector Search (RAG)** and **Knowledge Graphs**.

The pipeline has been refactored to rely exclusively on **Google Cloud** (Gemini Models & Embeddings). All vendor abstractions (OpenAI, Anthropic, etc.) have been removed.

### The Dual-Pipeline Strategy

We process the same source data (EPUBs) in two different ways to optimize for different downstream tasks:

1.  **RAG Extraction**: Creates thousands of small, overlapping chunks optimized for semantic search. Preserves granular metadata like Chapter Titles and Numbers.
2.  **Graph Extraction**: Creates massive, continuous text chunks (up to 2M tokens) optimized for Gemini 3 Pro preview's long context window to extract deep entity relationships without breaking narrative flow.

## Extraction Modes

### 1. RAG Extraction

- **Goal**: Create discrete, searchable chunks for Milvus/Vector Search.
- **Method**: Uses `UnstructuredEPubLoader` to standardize text, followed by `RecursiveCharacterTextSplitter` from LangChain.
- **Chunking**:
  - **Chunk Size**: 4000 characters (~1000 tokens)
  - **Overlap**: 800 characters (~200 tokens)
  - **Conversion**: 1 token ≈ 4 characters
- **Metadata**:
  - `universe`: The fantasy universe (e.g., "wheel_of_time").
  - `book_name`: The source filename.
  - `chapter_number`: Extracted via Regex patterns (e.g., "1", "PROLOGUE", "POV").
  - `chapter_title`: Extracted title (e.g., "The Boy Who Lived").

### 2. Graph Extraction

- **Goal**: Create massive context blocks for Knowledge Graph construction.
- **Method**: Custom `ebooklib` + `BeautifulSoup` parsing.
- **Chunking**: Accumulates text across chapters and even books until the strict **Context Window Limit** is reached.
- **Structure**:
  - Double newlines (`\n\n`) separate paragraphs.
  - No artificial headers or page numbers are inserted to maintain narrative flow.
  - Output is stored in `graph_chunks` collection.

## Context Window System

The Graph Extraction pipeline is driven entirely by the `context_window_size` parameter. This ensures we maximize the utility of long-context models like Gemini 3 Pro preview.

- **Logic**: `Max Chunk Size = Context Window * (1 - Safety Margin)`
- **Valid Values**:
  - `100000` (100k)
  - `200000` (200k)
  - `1000000` (1M) - Recommended for Gemini 3 Pro preview
  - `2000000` (2M)
- **Impact**:
  - Larger windows = Fewer chunks, better entity consistency, fewer "boundary" errors.
  - Smaller windows = More chunks, faster parallel processing, but more fragmented graph.

## Running the Pipeline

All commands use `uv` for dependency management.

### 1. RAG Ingestion (Vector Search)

The RAG pipeline supports three modes: **Analysis**, **MongoDB Storage**, and **Full Embedding**.

#### A. Analysis Only (Dry Run)

Processes EPUBs and outputs chunk statistics without any database writes. Use this to verify extraction quality.

```bash
# Basic analysis (logs to console)
uv run python -m src.etl.rag_ingestion.ingest \
  --dir data/wheel_of_time \
  --universe "Wheel of Time" \
  --dry-run

# Analysis with JSON output
uv run python -m src.etl.rag_ingestion.ingest \
  --dir data/wheel_of_time \
  --universe "Wheel of Time" \
  --dry-run \
  --output analysis_wot.json
```

#### B. MongoDB Only (Save RAG Chunks)

Saves chunks to MongoDB `rag_chunks` collection without generating embeddings. Useful for inspecting data before committing to expensive embedding calls.

```bash
uv run python -m src.etl.rag_ingestion.ingest \
  --dir data/wheel_of_time \
  --universe "Wheel of Time" \
  --mongodb \
  --dry-run
```

#### C. Full Pipeline (MongoDB + Milvus Embeddings)

Processes EPUBs, saves chunks to MongoDB, generates embeddings via Google, and stores vectors in Milvus.

```bash
# Standard embedding run
uv run python -m src.etl.rag_ingestion.ingest \
  --dir data/wheel_of_time \
  --universe "Wheel of Time" \
  --mongodb \
  --batch-size 50

# With MongoDB-based embedding cache (for distributed runs)
uv run python -m src.etl.rag_ingestion.ingest \
  --dir data/wheel_of_time \
  --universe "Wheel of Time" \
  --mongodb \
  --mongodb-cache \
  --batch-size 50
```

#### RAG Ingestion CLI Reference

| Flag | Description |
|------|-------------|
| `--dir` | Directory containing EPUB files (required) |
| `--universe` | Fantasy universe name (required) |
| `--dry-run` | Analysis only, no DB writes |
| `--output` | Save analysis to file (JSON/CSV/Parquet) |
| `--mongodb` | Save chunks to MongoDB `rag_chunks` |
| `--mongodb-cache` | Use MongoDB for embedding cache |
| `--embedding-model` | Google model (default: `gemini-embedding-001`) |
| `--batch-size` | Docs per batch (default: 50) |

### 2. Graph Extraction (Context Window Chunking)

Creates massive text blocks for the Graph Builder using token-based chunking.

#### A. Local Files Only

Saves chunks as `.txt` and `.meta.json` files for inspection.

```bash
uv run python -m src.etl.graph_extraction.epubs_to_chunks \
  --series wheel_of_time \
  --context-window 1000000
```

#### B. MongoDB Only (Recommended)

Saves directly to MongoDB `graph_chunks` without local files.

```bash
uv run python -m src.etl.graph_extraction.epubs_to_chunks \
  --series wheel_of_time \
  --context-window 1000000 \
  --mongodb --no-files
```

#### C. Both Local + MongoDB

Saves to both local files and MongoDB (useful for debugging).

```bash
uv run python -m src.etl.graph_extraction.epubs_to_chunks \
  --series harry_potter \
  --context-window 200000 \
  --safety-margin 0.15 \
  --mongodb
```

#### Graph Extraction CLI Reference

| Flag | Description |
|------|-------------|
| `--series` | Series to process: `wheel_of_time`, `harry_potter`, `song_of_ice_and_fire`, `asoiaf` |
| `--context-window` | Max tokens per chunk (e.g., `1000000` for 1M) |
| `--safety-margin` | Buffer percentage (default: `0.10` = 10%) |
| `--output-dir` | Custom output directory |
| `--mongodb` | Save to MongoDB `graph_chunks` |
| `--no-files` | Skip local file output (requires `--mongodb`) |

## Storage Modes

### MongoDB Only (Recommended)

This is the default production mode. Data is written directly to MongoDB Atlas.

- **Flag**: `--mongodb` (and `--no-files` for Graph Extraction to skip local JSONs).
- **Why**: Centralized source of truth, allows re-running graph/RAG steps without re-parsing EPUBs.

### Local + MongoDB

Useful for debugging. Writes JSON/TXT files to `data/processed_books_{series}`.

- **Default**: If `--no-files` is NOT passed to `epubs_to_chunks`.
- **Use Case**: Inspecting exactly where chunks split.

## MongoDB Setup

The pipeline requires a MongoDB Atlas cluster.

1.  **Connection**: Set your URI in `.env`:
    ```bash
    MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
    ```
2.  **Verify Connection**:
    ```bash
    mongosh "$(grep MONGODB_URI .env | cut -d '=' -f2-)"
    ```

## Collections & Schemas

### `rag_chunks`

Stores text segments for Vector Search visibility.

```json
{
  "_id": ObjectId("..."),
  "chunk_id": "wheel_of_time_rag_00001",
  "series": "wheel_of_time",
  "text_content": "The Wheel of Time turns, and Ages come and pass...",
  "character_count": 120,
  "metadata": {
    "book_name": "The Eye of the World",
    "chapter_number": "1",
    "chapter_title": "An Empty Road",
    "universe": "Wheel of Time"
  },
  "created_at": "2024-01-01T12:00:00+00:00"
}
```

### `graph_chunks`

Stores massive text blocks for Graph extraction.

```json
{
  "_id": ObjectId("..."),
  "chunk_id": "wheel_of_time_section_01",
  "series": "wheel_of_time",
  "text_content": "...[100,000 words of text]...",
  "token_count": 950000,
  "character_count": 3500000,
  "context_window_used": 1000000,
  "safety_margin": 0.1,
  "included_books": ["The Eye of the World.epub", "The Great Hunt.epub"],
  "created_at": "2024-01-01T12:00:00+00:00"
}
```

## Common Queries

```javascript
// check dbs
show dbs

// Select database
use fantasy_rag

// check collections
show collections

// Sample RAG Data
db.rag_chunks.find().limit(5);

// Sample RAG Data with filter by series
db.rag_chunks.find({ series: "harry_potter" }).limit(5);

// Sample Graph Data
db.graph_chunks.find().limit(1);

// Sample Graph Data without text_content
db.graph_chunks.find().limit(5).projection({ text_content: 0 });

// Sample Graph Data without text_content (filter base on series)
db.graph_chunks.find({ series: "harry_potter" }).limit(5).projection({ text_content: 0 });

// Count RAG Chunks by Universe
db.rag_chunks.aggregate([{ $group: { _id: "$series", count: { $sum: 1 } } }]);

// Count Graph Chunks by Universe
db.graph_chunks.aggregate([{ $group: { _id: "$series", count: { $sum: 1 } } }]);

// Get Graph Chunks for specific series
db.graph_chunks.find(
  { series: "harry_potter" },
  { chunk_id: 1, token_count: 1 },
);

// Find Unique Chapters in a Universe
db.rag_chunks.distinct("metadata.chapter_title", { series: "wheel_of_time" });

// Find unique combainations of book_name, chapter_number, chapter_title and universe
db.rag_chunks.aggregate([
  {
    $group: {
      _id: {
        book_name: "$metadata.book_name",
        chapter_number: "$metadata.chapter_number",
        chapter_title: "$metadata.chapter_title",
        universe: "$metadata.universe",
      },
    },
  },
  {
    $project: {
      _id: 0,
      book_name: "$_id.book_name",
      chapter_number: "$_id.chapter_number",
      chapter_title: "$_id.chapter_title",
      universe: "$_id.universe",
    },
  },
  {
    $sort: {
      universe: 1,
      book_name: 1,
      chapter_number: 1,
    },
  },
]);

// Graph data group by included_books for specific series

db.graph_chunks.aggregate([
  {
    $group: {
      _id: "$included_books",
      count: { $sum: 1 },
    },
  },
]);
```

## Deleting a Universe (Re-ingestion)

To cleanly re-ingest a universe, delete its data from both collections.

```javascript
// Delete all RAG chunks for a universe
db.rag_chunks.deleteMany({ series: "harry_potter" });

// Delete all Graph chunks for a universe
db.graph_chunks.deleteMany({ series: "harry_potter" });
```

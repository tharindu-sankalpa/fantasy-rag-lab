# ETL Pipeline Architecture

This module handles the extraction, transformation, and loading of data from raw sources (EPUBs) into downstream systems.

A key architectural pattern in this project is the use of **two distinct extraction strategies** for the same source data, optimized for different consumption patterns.

## The Dual-Pipeline Strategy

We deliberately process EPUB files in two different ways depending on whether the destination is the Knowledge Graph or the Vector Database.

| Feature    | Graph Extraction (`src/etl/graph_extraction`)      | RAG Ingestion (`src/etl/rag_ingestion`)              |
| :--------- | :------------------------------------------------- | :--------------------------------------------------- |
| **Logic**  | Manual (`ebooklib` + `BeautifulSoup`)              | Automated (`Unstructured` + `LangChain`)             |
| **Output** | **Continuous Text** (1 Book ≈ 1 huge String)       | **Chunks** (1 Book ≈ 5,000 small Documents)          |
| **Goal**   | Maintain perfect narrative flow for LLM analysis.  | Optimise for semantic search and metadata retrieval. |
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
  - **Metadata Focus:** RAG relies on knowing _where_ a piece of information came from (Chapter 5, Prologue, etc.). `Unstructured` standardises the document format, allowing us to focus our code on Regex logic to extract Chapter Metadata.
  - **Standardisation:** The output needs to be compatible with `LangChain`'s `RecursiveCharacterTextSplitter`. Using the standard `UnstructuredEPubLoader` provides us with `Document` objects that fit natively into the rest of the LangChain ecosystem.

---

## RAG Ingestion — 3-Stage Pipeline

The RAG ingestion pipeline is split into three independent, composable stages. Run them separately so that expensive operations (embedding API calls) are not repeated unnecessarily.

```
Stage 1: EPUBs → MongoDB
Stage 2: MongoDB → Gemini Embedding 2 Preview → Parquet cache (resumable)
Stage 3: Parquet cache → Milvus
```

### Chunking parameters

| Parameter      | Default | Notes                                    |
| :------------- | :------ | :--------------------------------------- |
| Chunk size     | 4 000 chars | ≈ 1 000 tokens at 4 chars/token      |
| Chunk overlap  | 800 chars   | ≈ 200 tokens                         |
| Splitter       | `RecursiveCharacterTextSplitter` | Universe-specific separators |

### Metadata preserved end-to-end

Every chunk carries these fields through all three stages:

| Field            | Example value                                    |
| :--------------- | :----------------------------------------------- |
| `chunk_id`       | `wheel_of_time_rag_00042`                        |
| `series`         | `wheel_of_time`                                  |
| `universe`       | `Wheel of Time`                                  |
| `book_name`      | `01_the_eye_of_the_world`                        |
| `chapter_number` | `8`                                              |
| `chapter_title`  | `Whirlpools of Color`                            |
| `text_content`   | raw chunk text                                   |
| `embedding`      | 3 072-dim float vector (Gemini Embedding 2)      |
| `embedding_model`| `gemini-embedding-2-preview`                     |
| `embedding_dim`  | `3072`                                           |
| `embedded_at`    | ISO-8601 UTC timestamp                           |

---

## Stage 1 — Chunk Extraction and MongoDB Storage

**Script:** `src/etl/rag_ingestion/ingest.py`

Parses EPUBs, creates chunks, optionally saves raw chunks to MongoDB. Also supports a legacy all-in-one path (Stage 1 + embed + Milvus) for quick experiments.

### Commands

```bash
# --- Wheel of Time (flat directory) ---
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/wheel_of_time \
    --universe "Wheel of Time" \
    --mongodb \
    --dry-run      # analyse chunks only, no storage

uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/wheel_of_time \
    --universe "Wheel of Time" \
    --mongodb      # save chunks to MongoDB

# --- A Song of Ice and Fire (flat directory) ---
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/song_of_ice_and_fire \
    --universe "Song of Ice and Fire" \
    --mongodb

# --- Harry Potter (flat directory) ---
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/harry_potter \
    --universe "Harry Potter" \
    --mongodb

# --- Dune (nested sub-series directories — requires --recursive) ---
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/dune \
    --universe "Dune" \
    --mongodb \
    --recursive
```

### Parameters

| Flag              | Default                        | Description                                                |
| :---------------- | :----------------------------- | :--------------------------------------------------------- |
| `--dir`           | required                       | Directory containing EPUB files                            |
| `--universe`      | required                       | Fantasy universe name                                      |
| `--mongodb`       | off                            | Save raw chunks to MongoDB `rag_chunks` collection         |
| `--recursive`     | off                            | Walk one level of subdirectories (required for Dune)       |
| `--dry-run`       | off                            | Analyse chunks only; skip storage and embedding            |
| `--output`        | —                              | Save dry-run analysis to a JSON file                       |
| `--embedding-model` | `gemini-embedding-2-preview` | Gemini model (only used in the legacy all-in-one path)     |
| `--batch-size`    | `50`                           | Documents per embedding batch (legacy all-in-one path)     |
| `--mongodb-cache` | off                            | Use MongoDB for embedding cache (legacy all-in-one path)   |

> **Dune book_name format:** With `--recursive`, the `book_name` metadata field is set to `{subdir}_{filename}.epub`, e.g. `01_butlerian_jihad_01_the_butlerian_jihad.epub`.

---

## Stage 2 — Embedding Generation and Parquet Cache

**Script:** `src/etl/rag_ingestion/embed.py`

Reads raw chunks from MongoDB, generates Gemini Embedding 2 Preview vectors, and writes them incrementally to a staging directory of numbered Parquet batch files. Supports full resume on interruption.

### Embedding model

| Property       | Value                          |
| :------------- | :----------------------------- |
| Model ID       | `gemini-embedding-2-preview`   |
| Dimensions     | **3 072** (full, no reduction) |
| Max input      | 8 192 tokens                   |
| Tier 1 limits  | 3 000 RPM / 1 000 000 TPM      |

### How resumability works

1. Each batch of N chunks is embedded and written atomically to `batches/batch_NNNNNN.parquet`.
2. After each batch, `batches/checkpoint.json` is updated with the set of `chunk_id`s that have been successfully embedded.
3. On restart, the script reads the checkpoint and skips already-embedded chunks, processing only what remains.
4. At the end, all batch files are merged into `{series}_embeddings.parquet`.

A chunk that fails all retries is **skipped** (not added to the checkpoint) so it will be retried on the next run rather than silently dropped from the final output.

### Retry and rate-limit handling

| Condition       | Backoff strategy                                  |
| :-------------- | :------------------------------------------------ |
| Rate limit (429)| Exponential backoff starting at 60 s, up to 5 retries |
| Transient error | Exponential backoff starting at 2 s (doubles each attempt) |
| All retries exhausted | Chunk is skipped; logged as error; retried on next run |

### Commands

```bash
# --- Wheel of Time ---
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Wheel of Time" \
    --output-dir cache/embeddings

# --- A Song of Ice and Fire ---
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Song of Ice and Fire" \
    --output-dir cache/embeddings

# --- Harry Potter ---
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Harry Potter" \
    --output-dir cache/embeddings

# --- Dune ---
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Dune" \
    --output-dir cache/embeddings

# Resume an interrupted run — just rerun the same command
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Wheel of Time" \
    --output-dir cache/embeddings

# Merge existing batch files without new API calls
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Wheel of Time" \
    --output-dir cache/embeddings \
    --merge-only
```

### Parameters

| Flag           | Default                      | Description                                         |
| :------------- | :--------------------------- | :-------------------------------------------------- |
| `--universe`   | required                     | Fantasy universe name (must match Stage 1)          |
| `--output-dir` | `cache/embeddings`           | Root directory for Parquet output                   |
| `--model`      | `gemini-embedding-2-preview` | Gemini embedding model ID                           |
| `--batch-size` | `20`                         | Chunks per batch (one Parquet file per batch)       |
| `--merge-only` | off                          | Skip embedding; only merge existing batch files     |

### Output layout

```
cache/embeddings/
└── wheel_of_time/
    ├── batches/
    │   ├── checkpoint.json          ← embedded chunk_ids (resume state)
    │   ├── batch_000000.parquet     ← 20 chunks each
    │   ├── batch_000001.parquet
    │   └── ...
    └── wheel_of_time_embeddings.parquet   ← final merged file (input for Stage 3)
```

---

## Stage 3 — Milvus Insertion from Parquet

**Script:** `src/etl/rag_ingestion/load_milvus.py`

Reads the final Parquet file and inserts all embeddings into the Milvus collection. All metadata fields are preserved.

### Milvus collection schema

| Field           | Type          | Notes                           |
| :-------------- | :------------ | :------------------------------ |
| `id`            | INT64 (PK)    | Auto-assigned                   |
| `embedding`     | FLOAT_VECTOR  | dim=3072, metric=COSINE         |
| `text`          | VARCHAR 65535 | Raw chunk text                  |
| `chunk_id`      | VARCHAR 200   | MongoDB chunk_id for traceability |
| `series`        | VARCHAR 100   | e.g. `wheel_of_time`            |
| `universe`      | VARCHAR 100   | e.g. `Wheel of Time`            |
| `book_title`    | VARCHAR 300   | Source book name                |
| `chapter_number`| VARCHAR 50    | Chapter number or PROLOGUE etc. |
| `chapter_title` | VARCHAR 200   | Chapter title                   |

### Commands

```bash
# --- Wheel of Time ---
uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/wheel_of_time/wheel_of_time_embeddings.parquet

# --- A Song of Ice and Fire ---
uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/song_of_ice_and_fire/song_of_ice_and_fire_embeddings.parquet

# --- Harry Potter ---
uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/harry_potter/harry_potter_embeddings.parquet

# --- Dune ---
uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/dune/dune_embeddings.parquet

# Larger batch for faster insertion
uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/dune/dune_embeddings.parquet \
    --batch-size 500
```

### Parameters

| Flag           | Default | Description                                     |
| :------------- | :------ | :---------------------------------------------- |
| `--parquet`    | required | Path to consolidated embedding Parquet file    |
| `--batch-size` | `200`   | Rows per Milvus insert call                     |

> **Re-running Stage 3:** Milvus uses auto-assigned INT64 primary keys so re-running Stage 3 will insert duplicate records. Clear the collection in Milvus before a full reload.

---

## Full Pipeline — Universe-by-Universe Reference

### Wheel of Time

```bash
# Stage 1
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/wheel_of_time --universe "Wheel of Time" --mongodb

# Stage 2
uv run python -m src.etl.rag_ingestion.embed \
    --universe "Wheel of Time" --output-dir cache/embeddings

# Stage 3
uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/wheel_of_time/wheel_of_time_embeddings.parquet
```

### A Song of Ice and Fire

```bash
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/song_of_ice_and_fire --universe "Song of Ice and Fire" --mongodb

uv run python -m src.etl.rag_ingestion.embed \
    --universe "Song of Ice and Fire" --output-dir cache/embeddings

uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/song_of_ice_and_fire/song_of_ice_and_fire_embeddings.parquet
```

### Harry Potter

```bash
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/harry_potter --universe "Harry Potter" --mongodb

uv run python -m src.etl.rag_ingestion.embed \
    --universe "Harry Potter" --output-dir cache/embeddings

uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/harry_potter/harry_potter_embeddings.parquet
```

### Dune

```bash
# Stage 1: --recursive required for nested sub-series layout
uv run python -m src.etl.rag_ingestion.ingest \
    --dir data/dune --universe "Dune" --mongodb --recursive

uv run python -m src.etl.rag_ingestion.embed \
    --universe "Dune" --output-dir cache/embeddings

uv run python -m src.etl.rag_ingestion.load_milvus \
    --parquet cache/embeddings/dune/dune_embeddings.parquet
```

# Ingestion Pipeline Guide

This guide details how to use the ingestion pipeline to process fantasy books, generate embeddings using various models (Voyage AI, OpenAI, Google), and store them in Milvus.

## Overview

The pipeline (`src/ingestion`) consists of two main components:

1.  **`processor.py`**: Handles parsing EPUBs, extracting metadata (chapters, titles), and chunking text specific to different fantasy universes.
2.  **`ingest.py`**: Orchestrates the flow—loading docs, generating embeddings via `LLMService`, and inserting into Milvus.

## Supported Models

The system supports multiple embedding backends. Configure your `.env` with the necessary API keys.

| Model Name | Provider      | Env Var          | Description                                                     |
| :--------- | :------------ | :--------------- | :-------------------------------------------------------------- |
| `voyage`   | Voyage AI     | `VOYAGE_API_KEY` | **Default**. Uses `voyage-3-large`. Best for retrieval quality. |
| `openai`   | OpenAI        | `OPENAI_API_KEY` | Uses `text-embedding-3-large`. Strong general purpose.          |
| `google`   | Google Gemini | `GOOGLE_API_KEY` | Uses `text-embedding-004`. Cost effective.                      |

## Supported Universes

The processor has specialized logic for:

- **"Wheel of Time"**: Handles complex prologue/epilogue/chapter patterns.
- **"Harry Potter"**: Handles standard "Chapter X" formats.
- **"Song of Ice and Fire"**: Handles POV-based chapter naming (e.g., "EDDARD", "ARYA").

## Usage

Run the ingestion script from the root directory.

### Basic Usage (Defaults to Voyage AI)

```bash
python -m src.ingestion.ingest \
  --dir "./data/wheel_of_time" \
  --universe "Wheel of Time"
```

### Specifying an Embedding Model

Use the `--embedding_model` argument to switch providers.

**Using Google Gemini:**

```bash
python -m src.ingestion.ingest \
  --dir "./data/harry_potter" \
  --universe "Harry Potter" \
  --embedding_model google
```

**Using OpenAI:**

```bash
python -m src.ingestion.ingest \
  --dir "./data/song_of_ice_and_fire" \
  --universe "Song of Ice and Fire" \
  --embedding_model openai
```

## Directory Structure

Ensure your data directory is organized by universe:

```
data/
├── wheel_of_time/
│   ├── book1.epub
│   └── ...
├── harry_potter/
│   ├── book1.epub
│   └── ...
└── song_of_ice_and_fire/
    ├── book1.epub
    └── ...
```

## Troubleshooting

- **API Key Errors**: Ensure the relevant API key (`VOYAGE_API_KEY`, etc.) is set in your `.env` file.
- **Chapter Detection**: If chapters aren't being detected correctly for a new book, check `src/ingestion/processor.py` and implement a new pattern in `_extract_chapter_info`.

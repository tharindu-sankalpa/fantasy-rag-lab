# Q&A Generation Module

This module is responsible for generating Question & Answer pairs from text chunks for RAG (Retrieval-Augmented Generation) evaluation. It uses Google's Gemini models to process batches of text and generate structured Q&A pairs across different thematic categories.

## Overview

The generation script (`src.qna_generation.generate`) processes text chunks and generates questions organized into 5 distinct categories (pillars) to ensure comprehensive coverage of the source material:

1. **characters** - Who people really are (names, origins, roles)
2. **events** - What happened (battles, deaths, plot points)
3. **magic** - How the system works (weaves, techniques, abilities)
4. **artifacts** - Special objects/locations (ter'angreal, places)
5. **lore** - Deep history & metaphysics (prophecies, cosmology)

All generated Q&A pairs are automatically saved to the MongoDB database (`fantasy_rag` -> `wot_rag_qna` collection).

## Basic Usage

The script is executed using `uv run python -m src.qna_generation.generate`.

### Generate for a Specific Category

To generate Q&A pairs for a specific category (e.g., `events`):

```bash
uv run python -m src.qna_generation.generate --category events
```

### Generate Using a Specific Model with Custom Batching

You can customize the model, batch size, and overlap. For example, using `gemini-3-pro-preview` with larger batches:

```bash
uv run python -m src.qna_generation.generate \
    --category events \
    --batch-size 40 \
    --batch-overlap 2 \
    --model "models/gemini-3-pro-preview"
```

### View Generation Statistics

To see how many Q&A pairs have been generated so far without running a new generation:

```bash
uv run python -m src.qna_generation.generate --stats-only
```

### List Available Categories

```bash
uv run python -m src.qna_generation.generate --list-categories
```

## Command Line Options

Here is the full list of available arguments:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--category` | string | `characters` | The question category to generate (`characters`, `events`, `magic`, `artifacts`, `lore`). |
| `--series` | string | `wheel_of_time` | The series identifier to process. |
| `--chunk-ids` | list | None | Specific chunk IDs to process (default is all chunks). |
| `--model` | string | `gemini-3-flash-preview` | The Gemini model to use for generation. See "Supported Models" below. |
| `--batch-size` | int | `10` | Number of chunks per generation window. |
| `--batch-overlap` | int | `2` | Number of overlapping chunks between batches to preserve context. |
| `--max-output-tokens`| int | `65536` | Maximum output tokens per request. |
| `--temperature` | float | `0.3` | Generation temperature (lower means more deterministic). |
| `--rpm` | int | `1000` | Requests per minute rate limit. |
| `--rpd` | int | `10000` | Requests per day rate limit. |
| `--tpm` | int | `1000000` | Input tokens per minute rate limit. |
| `--no-skip-processed`| flag | False | If set, reprocesses chunks that already have QA pairs for this category. |
| `--stats-only` | flag | False | Only show statistics, don't generate new QA pairs. |
| `--list-categories` | flag | False | List all available question categories and exit. |

## Supported Models

While the script can theoretically accept any Gemini model string, it is recommended to use one of the following Gemini 3 or 3.1 models for optimal Q&A generation quality:

- `models/gemini-3-flash-preview` *(Default - Fast and cost-effective)*
- `models/gemini-3-pro-preview` *(Slower but higher reasoning quality)*
- `models/gemini-3.1-flash-lite-preview` *(Lightweight, very fast)*
- `models/gemini-3.1-pro-preview` *(Latest Pro model, highest reasoning quality)*

*Note: You may occasionally encounter `503 UNAVAILABLE` errors if a specific model is experiencing high demand. If this happens, try switching to a different model or try again later.*

## Rate Limits & Quotas

The script includes built-in rate limiting to respect API quotas. By default, it is configured for the Gemini Paid Tier 1 limits:
- 1,000 Requests per minute
- 10,000 Requests per day
- 1,000,000 Input tokens per minute

If you are hitting rate limits, you can adjust these using the `--rpm`, `--rpd`, and `--tpm` flags.

## Resuming & Idempotency

By default, the script checks the MongoDB database and **skips chunks that have already been processed** for the specified category. This means you can safely cancel the script (using `Ctrl+C`) and restart it later; it will resume exactly where it left off. 

If you want to force regeneration for already-processed chunks, use the `--no-skip-processed` flag.

## Generation Types & Collection Schemas

There are two distinct types of Q&A generation in this module, which serve different purposes and output different schemas to the `wot_rag_qna` collection.

### 1. Knowledge Base Generation (`generate.py`)
- **Goal:** Build a comprehensive database of lore, facts, and trivia.
- **Categories:** `characters`, `events`, `magic`, `artifacts`, `lore`.
- **How it works:** Processes large batches of RAG chunks and extracts thematic facts.
- **Schema Highlight:** The metadata tracks the *batch* of chunks but does not strictly tie the answer to a single chunk.
  ```json
  {
    "qa_id": "wheel_of_time_rag_00000_to_wheel_of_time_rag_00039_characters_0000",
    "category": "characters",
    "metadata": {
      "source_batch_id": "wheel_of_time_rag_00000_to_wheel_of_time_rag_00039",
      "all_batch_chunk_ids": ["wheel_of_time_rag_00000", "...", "wheel_of_time_rag_00039"]
    }
  }
  ```

### 2. RAG Evaluation Generation (`generate_rag_eval.py`)
- **Goal:** Create a strict "Ground Truth" test suite to evaluate the RAG retrieval system.
- **Category:** Always `"rag_evaluation"`.
- **How it works:** Instructs the LLM to create questions that can *only* be answered by specific chunks, and forces it to track those exact chunk IDs.
- **Schema Highlight:** The metadata explicitly tracks the exact `source_chunk_ids` required to answer the question. This allows you to test if your vector database successfully retrieves these exact chunks.
  ```json
  {
    "qa_id": "wheel_of_time_rag_eval_0000_0003",
    "category": "rag_evaluation",
    "metadata": {
      "source_chunk_ids": [
        "wheel_of_time_rag_00051",
        "wheel_of_time_rag_00059",
        "wheel_of_time_rag_00060"
      ],
      "batch_index": 0
    }
  }
  ```
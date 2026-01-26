# QA Dataset Generation for RAG Evaluation

This document describes the QA generation pipeline that creates Question & Answer pairs from The Wheel of Time book series for evaluating Retrieval-Augmented Generation (RAG) systems.

## Overview

The pipeline extracts graph chunks (~200k tokens each) from MongoDB and uses Google Gemini to generate context-grounded QA pairs. Questions are organized into **5 categories** to ensure comprehensive coverage and diversity.

## The 5 Question Categories

| Category       | Description                                               | Example Questions                         |
| -------------- | --------------------------------------------------------- | ----------------------------------------- |
| **characters** | Who people really are (names, origins, roles, identities) | "What was Lanfear's original name?"       |
| **events**     | What happened (battles, deaths, plot points)              | "How did Egwene al'Vere die?"             |
| **magic**      | How the system works (weaves, techniques, abilities)      | "How does Rand know ancient weaves?"      |
| **artifacts**  | Special objects/locations (ter'angreal, places)           | "What are the glass columns in Rhuidean?" |
| **lore**       | Deep history & metaphysics (prophecies, cosmology)        | "What are bubbles of evil?"               |

## Quick Start

```bash
# Generate questions for a single category
uv run python -m src.qna_generation.generate --category characters

# Generate all 5 categories (run each separately)
uv run python -m src.qna_generation.generate --category characters
uv run python -m src.qna_generation.generate --category events
uv run python -m src.qna_generation.generate --category magic
uv run python -m src.qna_generation.generate --category artifacts
uv run python -m src.qna_generation.generate --category lore

# View current statistics
uv run python -m src.qna_generation.generate --stats-only

# List available categories
uv run python -m src.qna_generation.generate --list-categories
```

## How It Works

### Processing Flow

1. **Load chunks**: Fetches all `wheel_of_time` graph chunks from MongoDB (45 documents)
2. **Filter processed**: Skips chunks already processed for the selected category
3. **Generate QA**: Sends each chunk to Gemini with a category-specific prompt
4. **Parse & validate**: Extracts structured QA pairs from the response
5. **Store**: Saves results to the `wot_qna` MongoDB collection
6. **Repeat**: Processes next chunk until all are complete

### One Request Per Chunk Per Category

- Each chunk is processed **exactly once** per category
- Running `--category characters` sends 47 requests (one per chunk)
- Running again skips all 47 (already processed)
- Use `--no-skip-processed` to regenerate

### Expected Output

| Metric                               | Value            |
| ------------------------------------ | ---------------- |
| Chunks                               | 47               |
| Categories                           | 5                |
| Questions per chunk per category     | ~15-25           |
| **Total questions (all categories)** | **~3,500-5,800** |

## Rate Limits

The pipeline respects Gemini API quotas:

| Quota           | Limit     | Handling             |
| --------------- | --------- | -------------------- |
| Requests/minute | 25        | Waits automatically  |
| Requests/day    | 250       | Hard stop with error |
| Tokens/minute   | 1,000,000 | Waits automatically  |

With 47 chunks per category, you can complete **5 categories in ~10-15 minutes** (within daily quota).

## MongoDB Schema

QA pairs are stored in the `wot_qna` collection:

```javascript
{
  "qa_id": "wheel_of_time_section_01_characters_0005",
  "question": "Who led the effort to create the Bore?",
  "answer": "Mierin Eronaile, later known as Lanfear, led the research team...",
  "category": "characters",
  "complexity": "moderate",
  "evidence_quote": "Direct quote from the text...",
  "metadata": {
    "source_chunk_id": "wheel_of_time_section_01",
    "included_books": ["01_The_Eye_of_the_World.epub", ...],
    "series": "wheel_of_time",
    "generation_model": "gemini-3-pro-preview",
    "category_name": "Characters & Identities"
  },
  "created_at": ISODate("..."),
  "updated_at": ISODate("...")
}
```

## CLI Reference

```
usage: generate.py [--category {characters,events,magic,artifacts,lore}]
                   [--series SERIES]
                   [--chunk-ids CHUNK_IDS [CHUNK_IDS ...]]
                   [--model MODEL]
                   [--no-skip-processed]
                   [--stats-only]
                   [--list-categories]

Options:
  --category         Question category to generate (default: characters)
  --series           Series identifier (default: wheel_of_time)
  --chunk-ids        Process only specific chunks
  --model            Gemini model (default: gemini-3-pro-preview)
  --no-skip-processed  Regenerate existing QA pairs
  --stats-only       Show statistics without generating
  --list-categories  List all categories and exit
```

## FAQ

### Q: Does it process only Wheel of Time chunks?

**Yes.** The `--series wheel_of_time` filter ensures only those 47 chunks are processed. Other series (Harry Potter, etc.) are not touched.

### Q: What happens if I run the same command twice?

**Nothing.** By default, already-processed chunks are skipped. Use `--no-skip-processed` to regenerate.

### Q: Does re-running generate the same questions?

**Similar, but not identical.** The LLM will produce ~70-80% overlap with slight variations. To get truly different questions, use different categories.

### Q: How do I get more questions?

Run all 5 categories. Each category produces unique questions focused on different aspects:

```bash
# This gives you 5x the questions with minimal overlap
uv run python -m src.qna_generation.generate --category characters
uv run python -m src.qna_generation.generate --category events
uv run python -m src.qna_generation.generate --category magic
uv run python -m src.qna_generation.generate --category artifacts
uv run python -m src.qna_generation.generate --category lore
```

### Q: Can I process just 1-2 specific chunks?

**Yes.** Use `--chunk-ids`:

```bash
uv run python -m src.qna_generation.generate \
    --category magic \
    --chunk-ids wheel_of_time_section_01 wheel_of_time_section_02
```

### Q: What if the API fails mid-processing?

The pipeline:

- Retries 3 times with exponential backoff (4s → 8s → 16s)
- Tracks failed chunks and reports them at the end
- Aborts after 3 consecutive failures
- Re-running resumes from where it left off (skips completed chunks)

## File Structure

```
src/qna_generation/
├── __init__.py      # Package exports
├── schemas.py       # Pydantic models (QAPair, QADocument, etc.)
├── prompts.py       # Category definitions and prompt templates
├── service.py       # QAGenerationService with rate limiting
└── generate.py      # CLI entry point

scripts/
└── test_qna_generation.py  # Test script
```

## Testing

```bash
# Run basic tests (MongoDB + rate limiter)
uv run python scripts/test_qna_generation.py

# Test actual generation with one chunk
uv run python scripts/test_qna_generation.py --generate --category characters

# View current stats
uv run python scripts/test_qna_generation.py --stats
```

## Context Grounding

All generated QA pairs are **strictly grounded** in the source text:

- Every answer must be fully supported by the chunk
- An `evidence_quote` field contains the exact supporting text
- No external knowledge or hallucination is allowed
- This ensures fair RAG evaluation (the answer exists in the corpus)

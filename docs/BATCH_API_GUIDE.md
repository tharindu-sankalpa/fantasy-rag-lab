# Anthropic Claude Batch API Guide

## Overview

The Anthropic Message Batches API enables asynchronous processing of large volumes of entity extraction requests with a **50% cost reduction** compared to real-time API calls. This guide explains how to use the batch API for knowledge graph extraction in the Fantasy RAG Lab project.

## Key Benefits

| Feature | Real-time API | Batch API |
|---------|--------------|-----------|
| Cost | Full price | **50% discount** |
| Processing | Immediate | Up to 24 hours |
| Max requests | 1 at a time | 10,000 per batch |
| Use case | Interactive | Bulk processing |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entity Extraction Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │  Real-time Mode  │         │   Batch Mode     │              │
│  │                  │         │                  │              │
│  │ extract_entities │         │ extract_entities │              │
│  │      .py         │         │    _batch.py     │              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           │                            │                         │
│           ▼                            ▼                         │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │ UnifiedLLMService│         │AnthropicBatch    │              │
│  │                  │         │    Provider      │              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           │                            │                         │
│           ▼                            ▼                         │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │ AnthropicProvider│         │ Batch Job        │              │
│  │ (real-time)      │         │ Management       │              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           │                            │                         │
│           ▼                            ▼                         │
│  ┌─────────────────────────────────────────────────┐            │
│  │              Anthropic Claude API               │            │
│  │  /v1/messages (real-time) │ /v1/messages/batches│            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Same Output Format          │
              │   (ExtractionResult JSON)     │
              │                               │
              │   - entities[]                │
              │   - relationships[]           │
              │   - schema_proposals[]        │
              └───────────────────────────────┘
```

## Batch API Workflow

### 1. Job Creation

Submit a batch of extraction requests to the Anthropic API:

```python
import json
from pathlib import Path

from src.services.llm import AnthropicBatchProvider, build_extraction_batch_request
from src.knowledge_graph.schemas import ExtractionResult, Ontology

# ============================================================================
# Extraction prompt template (from src/knowledge_graph/extract_entities_batch.py)
# ============================================================================
EXTRACTION_PROMPT_TEMPLATE = """
You are an expert knowledge graph extractor for the "{series_name}" series.
Your goal is to extract a comprehensive list of entities and relationships from the provided text chunk.

STRICT ADHERENCE & ADAPTIVE EVOLOUTION:
1. You must primarily use the provided Ontology Schema to categorize entities and relationships.
2. IF you encounter a significant entity or relationship that strongly clearly does NOT fit existing definitions:
   - DO NOT force it into an incorrect category.
   - DO NOT ignore it if it is important.
   - PROPOSE a schema update in the `schema_proposals` field (e.g., "new_entity_type", "new_relationship_type").
3. Use the Canonical Renaming Rules to normalize entity names.

INPUT CONTEXT:
This text is from: {book_names}
Section ID: {section_id}

TASK:
1. Identify all significant entities (Characters, Locations, Organizations, Artifacts, Events, etc.).
2. Extract detailed attributes for each entity.
3. Identify all relationships between these entities.
4. Extract evidence (quotes) for relationships.
5. Assign a CONFIDENCE score (0.0-1.0) to each extraction.
6. Identify gaps in the schema and propose updates in `schema_proposals`.

OUTPUT FORMAT:
Return a JSON object strictly matching the `ExtractionResult` Pydantic model.
"""

# ============================================================================
# Initialize provider
# ============================================================================
provider = AnthropicBatchProvider(
    api_key="your-api-key",  # Or use os.getenv("ANTHROPIC_API_KEY")
    jobs_dir="data/batch_jobs"  # Persists job metadata for resumability
)

# ============================================================================
# Load schema and build extraction prompt for one section
# ============================================================================
# Load ontology schema for the series
with open("data/schemas/harry_potter_schema.json", "r") as f:
    schema_data = json.load(f)
ontology = Ontology(**schema_data)
schema_context = f"ONTOLOGY:\n{ontology.model_dump_json(indent=2)}"

# Load one section file
section_file = Path("data/processed_books_claude_200k/harry_potter_section_01.txt")
text_content = section_file.read_text()

# Load metadata
with open("data/processed_books_claude_200k/harry_potter_section_01.meta.json") as f:
    meta = json.load(f)
book_names = ", ".join(meta.get("books", []))
section_id = str(meta.get("section", "1"))

# Build full prompt
extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
    series_name="Harry Potter",
    book_names=book_names,
    section_id=section_id,
)
full_prompt = f"{extraction_prompt}\n\n{schema_context}\n\nTEXT CONTENT:\n{text_content}"

# Create batch request using ExtractionResult schema
requests = [
    build_extraction_batch_request(
        custom_id="harry_potter_section_01",
        prompt=full_prompt,
        model="claude-opus-4-5-20251101",
        schema=ExtractionResult,  # Pydantic model from src/knowledge_graph/schemas.py
        max_tokens=64000,
        temperature=0.0,
    )
]

# ============================================================================
# Submit batch
# ============================================================================
job = await provider.create_batch(
    requests=requests,
    output_dir="data/extracted_graph_batch",
    model="claude-opus-4-5-20251101",
    series_name="Harry Potter"
)

print(f"Batch ID: {job.id}")  # e.g., "msgbatch_01234567890"
```

The `ExtractionResult` schema (from `src/knowledge_graph/schemas.py`) defines:

```python
class ExtractionResult(BaseModel):
    """Validates the output of the extraction phase."""
    entities: List[EntityInstance]        # Extracted characters, locations, items, etc.
    relationships: List[RelationshipInstance]  # Connections between entities
    schema_proposals: List[SchemaUpdateProposal] = []  # Suggested schema updates
```

### 2. Job ID Tracking

The batch ID is automatically persisted to `data/batch_jobs/{batch_id}.json`:

```json
{
  "id": "msgbatch_01234567890",
  "processing_status": "in_progress",
  "request_counts": {
    "processing": 7,
    "succeeded": 0,
    "errored": 0
  },
  "created_at": "2026-01-12T10:00:00Z",
  "expires_at": "2026-01-13T10:00:00Z",
  "local_request_mapping": {
    "harry_potter_section_01": "data/extracted_graph_batch/harry_potter_section_01_extracted.json",
    "harry_potter_section_02": "data/extracted_graph_batch/harry_potter_section_02_extracted.json"
  },
  "output_dir": "data/extracted_graph_batch",
  "model": "claude-opus-4-5-20251101",
  "series_name": "Harry Potter"
}
```

### 3. Polling Mechanisms

Monitor batch progress with automatic polling:

```python
# Poll until complete (with configurable interval)
job = await provider.poll_until_complete(
    batch_id="msgbatch_01234567890",
    poll_interval_seconds=60,  # Check every minute
    max_wait_hours=24.0        # Timeout after 24 hours
)

# Or check status manually
status = await provider.get_batch_status("msgbatch_01234567890")
print(f"Status: {status.processing_status}")
print(f"Succeeded: {status.request_counts.succeeded}")
print(f"Processing: {status.request_counts.processing}")
```

### 4. Result Retrieval

Once the batch completes, retrieve and process results:

```python
# Retrieve results and save to files
result = await provider.process_results_to_files(
    batch_id="msgbatch_01234567890",
    schema=ExtractionResult
)

print(f"Processed: {result['processed']}")
print(f"Failed: {result['failed']}")
print(f"Total tokens: {result['usage'].total_tokens}")
print(f"Files written: {result['files_written']}")
```

## Command-Line Usage

### Full Workflow (Recommended)

Submit, poll, and retrieve in one command:

```bash
# Extract Harry Potter series with Claude Opus 4.5 (batch mode)
python src/knowledge_graph/extract_entities_batch.py \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus_batch \
    --series "Harry Potter"

# Extract all series
python src/knowledge_graph/extract_entities_batch.py \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus_batch
```

### Submit Only (For Long-Running Jobs)

Submit batch and get ID for later retrieval:

```bash
python src/knowledge_graph/extract_entities_batch.py \
    --mode submit \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus_batch

# Output: {"batches": [{"series": "Harry Potter", "batch_id": "msgbatch_xxx"}]}
```

### Check Status

Monitor a running batch:

```bash
python src/knowledge_graph/extract_entities_batch.py \
    --mode status \
    --batch-id msgbatch_xxx

# Output: {"batch_id": "msgbatch_xxx", "status": "in_progress", "succeeded": 3, "processing": 4}
```

### Retrieve Results

Download results from a completed batch:

```bash
python src/knowledge_graph/extract_entities_batch.py \
    --mode retrieve \
    --batch-id msgbatch_xxx \
    --output-dir data/extracted_graph_claude_opus_batch
```

## Comparison: Real-time vs Batch

### When to Use Real-time API

- Interactive applications requiring immediate responses
- Small number of extractions (< 10 sections)
- Testing and development
- Time-sensitive workflows

```bash
# Real-time extraction
python src/knowledge_graph/extract_entities.py \
    --provider anthropic \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus
```

### When to Use Batch API

- Large-scale extraction (many sections)
- Cost optimization is priority
- Processing time flexibility (up to 24 hours)
- Bulk comparison experiments

```bash
# Batch extraction (50% cost savings)
python src/knowledge_graph/extract_entities_batch.py \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus_batch
```

## Cost Estimation

Token counts from `data/processed_books_claude_200k/*.meta.json`:

| Series | Sections | Input Tokens | Est. Output* |
|--------|----------|--------------|--------------|
| Harry Potter | 7 | 1,562,150 | ~234K |
| ASOIAF | 8 | 3,041,607 | ~456K |
| Wheel of Time | 16 | 6,717,033 | ~1.01M |
| **Total** | **31** | **11,320,790** | **~1.7M** |

*Output estimated at ~15% of input for structured extraction*

### Claude Opus 4.5 Pricing

| Series | Real-time Cost | Batch Cost (50% off) | Savings |
|--------|----------------|----------------------|---------|
| Harry Potter | ~$13.65 | **~$6.83** | $6.82 |
| ASOIAF | ~$26.60 | **~$13.30** | $13.30 |
| Wheel of Time | ~$58.85 | **~$29.43** | $29.42 |
| **Total** | **~$99.10** | **~$49.56** | **$49.54** |

*Based on [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): $5/MTok input, $25/MTok output (real-time) | $2.50/MTok input, $12.50/MTok output (batch)*

### Claude Sonnet 4.5 Pricing (Budget Option)

| Series | Real-time Cost | Batch Cost (50% off) | Savings |
|--------|----------------|----------------------|---------|
| Harry Potter | ~$8.19 | **~$4.10** | $4.09 |
| ASOIAF | ~$15.96 | **~$7.98** | $7.98 |
| Wheel of Time | ~$35.31 | **~$17.66** | $17.65 |
| **Total** | **~$59.46** | **~$29.74** | **$29.72** |

*Based on [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): $3/MTok input, $15/MTok output (real-time) | $1.50/MTok input, $7.50/MTok output (batch)*

### Model Comparison Summary

| Model | Real-time Total | Batch Total | Best For |
|-------|-----------------|-------------|----------|
| Claude Opus 4.5 | ~$99 | **~$50** | Highest quality extraction |
| Claude Sonnet 4.5 | ~$59 | **~$30** | Cost-effective extraction |

## Error Handling

### Batch-Level Errors

```python
try:
    job = await provider.create_batch(requests, output_dir, model)
except anthropic.APIError as e:
    logger.error("batch_creation_failed", error=str(e))
```

### Request-Level Errors

Individual requests can fail while others succeed:

```python
results = await provider.retrieve_results(batch_id)

for result in results:
    if result.result_type == BatchResultType.SUCCEEDED:
        # Process successful result
        pass
    elif result.result_type == BatchResultType.ERRORED:
        logger.error(
            "request_failed",
            custom_id=result.custom_id,
            error=result.error.message
        )
```

### Resumable Workflows

Job metadata is persisted, enabling resume after interruption:

```python
# If script crashes, restart and retrieve existing batch
job = await provider._load_job("msgbatch_xxx")
if job and job.processing_status == BatchProcessingStatus.ENDED:
    results = await provider.process_results_to_files(job.id, schema)
```

## Supported Models

| Model | Context Window | Batch Support |
|-------|---------------|---------------|
| claude-opus-4-5-20251101 | 200K | ✅ |
| claude-sonnet-4-5-20250929 | 200K | ✅ |

## File Structure

```
data/
├── batch_jobs/                          # Batch job metadata
│   ├── msgbatch_xxx.json               # Job tracking file
│   └── msgbatch_yyy.json
├── processed_books_claude_200k/         # Input sections
│   ├── harry_potter_section_01.txt
│   ├── harry_potter_section_01.meta.json
│   └── ...
└── extracted_graph_claude_opus_batch/   # Batch output
    ├── harry_potter_section_01_extracted.json
    ├── harry_potter_section_02_extracted.json
    └── ...
```

## Troubleshooting

### Batch Not Completing

- Check status: `--mode status --batch-id xxx`
- Batches can take up to 24 hours
- Check for errors in request_counts

### Results URL Expired

- Results are available for 29 days after batch creation
- Re-submit batch if expired

### Partial Failures

- Check `request_counts.errored` for failed requests
- Review error messages in results
- Re-submit failed sections individually

## Integration with Existing Pipeline

The batch extraction produces **identical output format** to real-time extraction:

```json
{
  "entities": [
    {
      "id": "harry_potter",
      "type": "Character",
      "name": "Harry Potter",
      "attributes": {...},
      "mentions": [{"file_name": "harry_potter_section_01.txt", "chunk_id": "1"}],
      "confidence": {"score": 0.95}
    }
  ],
  "relationships": [...],
  "schema_proposals": [...]
}
```

This ensures downstream consumers (knowledge graph builders, analyzers) work without modification.

## References

- [Anthropic Message Batches API Documentation](https://docs.anthropic.com/en/api/creating-message-batches)
- [Anthropic Batch API Announcement](https://www.anthropic.com/news/message-batches-api)
- [Project Multi-Model Extraction Guide](./MULTI_MODEL_EXTRACTION_GUIDE.md)

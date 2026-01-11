# Multi-Model Knowledge Graph Extraction Guide

This guide explains how to use the updated ETL pipeline to extract knowledge graphs from fantasy books using different LLM models (Gemini, Claude, GPT) with their respective context window sizes.

## Overview

The pipeline now supports **model-aware adaptive chunking**, allowing you to:
1. Process books with different chunk sizes based on each model's context window
2. Compare extraction quality across different LLMs
3. Preserve existing Gemini extractions while testing alternatives
4. Maintain separate outputs for fair comparison

## Supported Models & Context Windows

| Provider   | Model                         | Context Window | Recommended Chunking |
|------------|-------------------------------|----------------|----------------------|
| **Google** | gemini-3-pro-preview          | 1,000,000 tokens | 800,000 tokens/chunk |
| **Anthropic** | claude-opus-4-5-20251101   | ~200,000 tokens  | 160,000 tokens/chunk |
| **Anthropic** | claude-sonnet-4-5-20250929 | ~200,000 tokens  | 160,000 tokens/chunk |
| **OpenAI** | gpt-5.2-pro                   | ~400,000 tokens  | 320,000 tokens/chunk |

**Note:** Chunking uses 80% of the context window to reserve space for:
- System prompts and extraction instructions
- Ontology schema JSON
- Structured output response buffer

---

## Workflow

### Step 1: Process EPUBs with Model-Specific Chunking

The `extract_epubs.py` script now accepts `--context-window` and `--output-dir` parameters to create model-appropriate text chunks.

#### For Gemini 3 Pro Preview (1M context) - **Already Done**

Your existing `data/processed_books/` contains Gemini-optimized chunks. These should be preserved!

**Recommended:** Rename to clearly indicate context window:
```bash
# Rename existing directory (preserves your $20 investment in Gemini extractions)
mv data/processed_books data/processed_books_gemini_1m
mv data/extracted_graph-gemini data/extracted_graph_gemini
```

#### For Claude Opus 4.5 / Claude Sonnet 4.5 (200k context)

```bash
# Process with 200k context window
python src/etl/graph_extraction/extract_epubs.py \
    --context-window 200000 \
    --output-dir data/processed_books_claude_200k

# Expected output:
# - data/processed_books_claude_200k/harry_potter_section_01.txt
# - data/processed_books_claude_200k/harry_potter_section_01.meta.json
# - ... (more sections, since books are split into smaller chunks)
```

**What changes:**
- Books that were paired together for Gemini (e.g., HP 1+2) will be split into more sections
- Large books (e.g., Order of the Phoenix) may become individual sections
- More total sections = more API calls, but manageable chunks for Claude

#### For GPT-5.2 (400k context)

```bash
# Process with 400k context window
python src/etl/graph_extraction/extract_epubs.py \
    --context-window 400000 \
    --output-dir data/processed_books_gpt_400k
```

**What to expect:**
- Middle ground between Gemini and Claude
- Fewer sections than Claude processing
- Still allows pairing some smaller books

#### Dry Run for Testing

Test chunking strategy without processing all books:

```bash
# Dry run: processes only first book of first series
python src/etl/graph_extraction/extract_epubs.py \
    --context-window 200000 \
    --output-dir data/processed_books_test \
    --dry-run

# Check output token counts in metadata files
cat data/processed_books_test/*.meta.json | jq '.estimated_tokens'
```

---

### Step 2: Extract Entities with Target LLM

Use `extract_entities.py` to run knowledge graph extraction on the processed chunks.

#### Extract with Gemini 3 Pro Preview (Baseline)

```bash
# Extract from your existing Gemini-optimized chunks
python src/knowledge_graph/extract_entities.py \
    --provider google \
    --model gemini-3-pro-preview \
    --input-dir data/processed_books_gemini_1m \
    --output-dir data/extracted_graph_gemini
```

**Output:**
- `data/extracted_graph_gemini/harry_potter_section_01_extracted.json`
- Contains: entities, relationships, confidence scores, schema proposals

#### Extract with Claude Opus 4.5

```bash
# Extract using Claude Opus 4.5 (highest quality Claude model)
python src/knowledge_graph/extract_entities.py \
    --provider anthropic \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus

# Process only Harry Potter for quick test
python src/knowledge_graph/extract_entities.py \
    --provider anthropic \
    --model claude-opus-4-5-20251101 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_opus \
    --series "Harry Potter"
```

**Expected behavior:**
- More API calls than Gemini (smaller chunks = more sections)
- May extract different entities due to smaller context
- Potentially higher precision but less cross-book relationship detection

#### Extract with Claude Sonnet 4.5

```bash
# Extract using Claude Sonnet 4.5 (faster, cheaper than Opus)
python src/knowledge_graph/extract_entities.py \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --input-dir data/processed_books_claude_200k \
    --output-dir data/extracted_graph_claude_sonnet
```

**Use case:**
- Cost-effective alternative to Opus
- Similar context window, faster processing
- Good for comparing quality vs cost trade-offs

#### Extract with GPT-5.2

```bash
# Extract using GPT-5.2 (400k context)
python src/knowledge_graph/extract_entities.py \
    --provider openai \
    --model gpt-5.2-pro \
    --input-dir data/processed_books_gpt_400k \
    --output-dir data/extracted_graph_gpt
```

**Expected behavior:**
- Middle ground between Gemini and Claude
- Fewer sections than Claude (larger chunks)
- Strong structured output support

---

## Directory Structure

After following this workflow, your data directory should look like:

```
data/
├── raw/                                    # Original EPUB files
│   ├── harry_potter/
│   ├── song_of_ice_and_fire/
│   └── wheel_of_time/
│
├── schemas/                                # Series-specific ontologies
│   ├── harry_potter_schema.json
│   ├── a_song_of_ice_and_fire_schema.json
│   └── the_wheel_of_time_schema.json
│
├── processed_books_gemini_1m/              # Gemini-optimized chunks (existing)
│   ├── harry_potter_section_01.txt         (~800k tokens)
│   ├── harry_potter_section_01.meta.json
│   └── ...
│
├── processed_books_claude_200k/            # Claude-optimized chunks (new)
│   ├── harry_potter_section_01.txt         (~160k tokens)
│   ├── harry_potter_section_02.txt         (~160k tokens)
│   └── ...
│
├── processed_books_gpt_400k/               # GPT-optimized chunks (new)
│   ├── harry_potter_section_01.txt         (~320k tokens)
│   └── ...
│
├── extracted_graph_gemini/                 # Gemini extractions (existing)
│   ├── harry_potter_section_01_extracted.json
│   └── ...
│
├── extracted_graph_claude_opus/            # Claude Opus extractions (new)
│   ├── harry_potter_section_01_extracted.json
│   └── ...
│
├── extracted_graph_claude_sonnet/          # Claude Sonnet extractions (new)
│   └── ...
│
└── extracted_graph_gpt/                    # GPT extractions (new)
    └── ...
```

---

## Cost Estimation

### Token Usage Example (Harry Potter Series)

| Model | Context Window | Sections | Est. Input Tokens | Est. Cost |
|-------|----------------|----------|-------------------|-----------|
| Gemini 3 Pro | 1M | 6 sections | ~4.8M tokens | ~$24* |
| Claude Opus 4.5 | 200k | 25 sections | ~4.0M tokens | ~$60** |
| Claude Sonnet 4.5 | 200k | 25 sections | ~4.0M tokens | ~$12** |
| GPT-5.2 | 400k | 12 sections | ~3.8M tokens | ~$38*** |

_* Based on your existing Gemini extractions_
_** Approximate based on Anthropic pricing (input: $15/M, output: $75/M for Opus)_
_*** Approximate based on OpenAI pricing_

**Important:** These are rough estimates. Always:
1. Start with `--series "Harry Potter"` to test one series first
2. Monitor token usage in logs (`tokens_used` field)
3. Use `--dry-run` to verify chunking before full extraction

---

## Comparing Extraction Quality

### Metrics to Track

After extraction, compare models using these metrics:

#### 1. Entity Extraction Quality

```bash
# Count entities extracted per model
echo "Gemini entities:"
cat data/extracted_graph_gemini/*.json | jq '.entities | length' | awk '{s+=$1} END {print s}'

echo "Claude Opus entities:"
cat data/extracted_graph_claude_opus/*.json | jq '.entities | length' | awk '{s+=$1} END {print s}'

echo "GPT-5.2 entities:"
cat data/extracted_graph_gpt/*.json | jq '.entities | length' | awk '{s+=$1} END {print s}'
```

#### 2. Relationship Extraction Quality

```bash
# Count relationships per model
cat data/extracted_graph_gemini/*.json | jq '.relationships | length' | awk '{s+=$1} END {print s}'
```

#### 3. Confidence Scores

```bash
# Average confidence scores
cat data/extracted_graph_gemini/*.json | jq '.entities[].confidence.score' | awk '{s+=$1; c++} END {print s/c}'
```

#### 4. Attribute Completeness

```bash
# Check how many entities have full attributes
cat data/extracted_graph_gemini/harry_potter_section_01_extracted.json | \
    jq '.entities[] | select(.attributes | length > 3) | .name'
```

---

## Troubleshooting

### Issue: "Model not allowed for provider"

**Cause:** Attempting to use a model not in the allowed list.

**Solution:** Use only these models:
- Google: `gemini-3-pro-preview`
- Anthropic: `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`
- OpenAI: `gpt-5.2-pro`

### Issue: "Input directory not found"

**Cause:** Mismatch between processing and extraction input directories.

**Solution:** Ensure you've run `extract_epubs.py` first:
```bash
# 1. Process books first
python src/etl/graph_extraction/extract_epubs.py --context-window 200000 --output-dir data/processed_books_claude_200k

# 2. Then extract entities
python src/knowledge_graph/extract_entities.py --input-dir data/processed_books_claude_200k
```

### Issue: Rate Limit Errors

**Cause:** API rate limits from providers.

**Solution:**
- Process one series at a time: `--series "Harry Potter"`
- Use checkpoint system (script auto-resumes)
- Add retries with backoff (built into UnifiedLLMService)

### Issue: Out of Memory

**Cause:** Processing very large books with small context windows creates many sections.

**Solution:**
- Increase `--context-window` if model supports it
- Process series individually
- Monitor system resources

---

## Advanced: Fallback Chains

The UnifiedLLMService supports automatic fallback to alternative providers if primary fails.

**Example:** Try Gemini first, fall back to GPT if unavailable:

```python
# In extract_entities.py, modify the generate_structured call:
response = await self.llm_service.generate_structured(
    prompt=full_prompt,
    schema=ExtractionResult,
    provider="google",
    model="gemini-3-pro-preview",
    fallback_chain=[
        "openrouter/google/gemini-3-pro-preview",
        "openai/gpt-5.2-pro"
    ]
)
```

---

## Best Practices

1. **Always start with dry runs** to verify chunking strategy
2. **Process one series first** to estimate costs and validate output
3. **Preserve existing extractions** - don't overwrite your Gemini results!
4. **Use descriptive directory names** to distinguish chunking strategies
5. **Monitor token usage** via logs to track costs
6. **Compare like-to-like** - use consistent ontology schemas across models
7. **Track extraction dates** in metadata for reproducibility

---

## Next Steps

After extracting with multiple models:

1. **Aggregate Results:** Merge extracted graphs into a unified knowledge base
2. **Evaluate Quality:** Compare entity accuracy, relationship completeness, confidence scores
3. **Analyze Trade-offs:** Balance cost vs quality vs context window size
4. **Optimize Prompts:** Iterate on extraction prompts based on model-specific strengths
5. **Publish Findings:** Document which model performs best for fantasy fiction knowledge graphs

---

## Questions or Issues?

- Check logs in terminal output (structured logging with `structlog`)
- Review checkpoint files in output directories
- Verify API keys are set: `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- Ensure all dependencies are installed: `pip install -r requirements.txt`

---

**Last Updated:** 2026-01-11
**Pipeline Version:** 2.0 (Multi-Model Support)

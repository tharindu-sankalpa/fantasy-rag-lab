#!/bin/bash

# Multi-Model Knowledge Graph Extraction Examples
# This script provides copy-paste commands for common workflows
# DO NOT RUN THIS SCRIPT DIRECTLY - copy commands individually as needed

set -e  # Exit on error

# =============================================================================
# STEP 0: Preserve Existing Gemini Data (One-Time Setup)
# =============================================================================

echo "=== Preserve Existing Gemini Extractions ==="
echo "Run these commands once to rename your existing directories:"
echo ""
echo "# Rename processed books to indicate Gemini/1M context"
echo "mv data/processed_books data/processed_books_gemini_1m"
echo ""
echo "# Rename extracted graphs (if needed)"
echo "mv data/extracted_graph-gemini data/extracted_graph_gemini"
echo ""

# =============================================================================
# STEP 1: Process EPUBs for Different Context Windows
# =============================================================================

echo "=== Process EPUBs for Claude (200k context) ==="
echo "python src/etl/graph_extraction/extract_epubs.py \\"
echo "    --context-window 200000 \\"
echo "    --output-dir data/processed_books_claude_200k"
echo ""

echo "=== Process EPUBs for GPT-5.2 (400k context) ==="
echo "python src/etl/graph_extraction/extract_epubs.py \\"
echo "    --context-window 400000 \\"
echo "    --output-dir data/processed_books_gpt_400k"
echo ""

echo "=== Dry Run: Test Chunking Strategy ==="
echo "python src/etl/graph_extraction/extract_epubs.py \\"
echo "    --context-window 200000 \\"
echo "    --output-dir data/processed_books_test \\"
echo "    --dry-run"
echo ""

echo "=== Process Only Harry Potter Series ==="
echo "python src/etl/graph_extraction/extract_epubs.py \\"
echo "    --context-window 200000 \\"
echo "    --output-dir data/processed_books_claude_200k \\"
echo "    --series harry_potter"
echo ""

# =============================================================================
# STEP 2: Extract Entities with Different Models
# =============================================================================

echo "=== Extract with Gemini 3 Pro Preview (Baseline) ==="
echo "python src/knowledge_graph/extract_entities.py \\"
echo "    --provider google \\"
echo "    --model gemini-3-pro-preview \\"
echo "    --input-dir data/processed_books_gemini_1m \\"
echo "    --output-dir data/extracted_graph_gemini"
echo ""

echo "=== Extract with Claude Opus 4.5 ==="
echo "python src/knowledge_graph/extract_entities.py \\"
echo "    --provider anthropic \\"
echo "    --model claude-opus-4-5-20251101 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_opus"
echo ""

echo "=== Extract with Claude Sonnet 4.5 ==="
echo "python src/knowledge_graph/extract_entities.py \\"
echo "    --provider anthropic \\"
echo "    --model claude-sonnet-4-5-20250929 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_sonnet"
echo ""

echo "=== Extract with GPT-5.2 ==="
echo "python src/knowledge_graph/extract_entities.py \\"
echo "    --provider openai \\"
echo "    --model gpt-5.2-pro \\"
echo "    --input-dir data/processed_books_gpt_400k \\"
echo "    --output-dir data/extracted_graph_gpt"
echo ""

echo "=== Extract Only Harry Potter with Claude Opus ==="
echo "python src/knowledge_graph/extract_entities.py \\"
echo "    --provider anthropic \\"
echo "    --model claude-opus-4-5-20251101 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_opus \\"
echo "    --series 'Harry Potter'"
echo ""

# =============================================================================
# STEP 3: Compare Extraction Quality
# =============================================================================

echo "=== Count Total Entities Extracted ==="
echo "# Gemini"
echo "cat data/extracted_graph_gemini/*.json | jq '.entities | length' | awk '{s+=\$1} END {print s}'"
echo ""
echo "# Claude Opus"
echo "cat data/extracted_graph_claude_opus/*.json | jq '.entities | length' | awk '{s+=\$1} END {print s}'"
echo ""
echo "# GPT-5.2"
echo "cat data/extracted_graph_gpt/*.json | jq '.entities | length' | awk '{s+=\$1} END {print s}'"
echo ""

echo "=== Count Total Relationships Extracted ==="
echo "# Gemini"
echo "cat data/extracted_graph_gemini/*.json | jq '.relationships | length' | awk '{s+=\$1} END {print s}'"
echo ""
echo "# Claude Opus"
echo "cat data/extracted_graph_claude_opus/*.json | jq '.relationships | length' | awk '{s+=\$1} END {print s}'"
echo ""

echo "=== Calculate Average Confidence Scores ==="
echo "cat data/extracted_graph_gemini/*.json | jq '.entities[].confidence.score' | awk '{s+=\$1; c++} END {print s/c}'"
echo ""

echo "=== List Entity Types Extracted ==="
echo "cat data/extracted_graph_gemini/*.json | jq -r '.entities[].type' | sort | uniq -c | sort -rn"
echo ""

echo "=== Check Extraction Token Usage ==="
echo "# Review logs for token usage:"
echo "grep 'extraction_complete' logs/extraction.log | jq '.tokens_used'"
echo ""

# =============================================================================
# STEP 4: Verify Chunking Results
# =============================================================================

echo "=== Verify Token Counts per Section ==="
echo "# Check metadata for Claude chunks"
echo "cat data/processed_books_claude_200k/*.meta.json | jq '{file: .books, tokens: .estimated_tokens}'"
echo ""
echo "# Compare to Gemini chunks"
echo "cat data/processed_books_gemini_1m/*.meta.json | jq '{file: .books, tokens: .estimated_tokens}'"
echo ""

echo "=== Count Total Sections per Model ==="
echo "ls -1 data/processed_books_gemini_1m/*.txt | wc -l"
echo "ls -1 data/processed_books_claude_200k/*.txt | wc -l"
echo "ls -1 data/processed_books_gpt_400k/*.txt | wc -l"
echo ""

# =============================================================================
# STEP 5: Useful Inspection Commands
# =============================================================================

echo "=== Inspect Specific Entity ==="
echo "# Find all mentions of 'Harry Potter' across extractions"
echo "cat data/extracted_graph_gemini/*.json | jq '.entities[] | select(.name == \"Harry Potter\")'"
echo ""

echo "=== Compare Entity Attributes ==="
echo "# Compare Harry Potter attributes between models"
echo "cat data/extracted_graph_gemini/harry_potter_section_01_extracted.json | \\"
echo "    jq '.entities[] | select(.name == \"Harry Potter\") | .attributes'"
echo ""
echo "cat data/extracted_graph_claude_opus/harry_potter_section_01_extracted.json | \\"
echo "    jq '.entities[] | select(.name == \"Harry Potter\") | .attributes'"
echo ""

echo "=== Find Schema Proposals ==="
echo "# Check if models proposed new entity types"
echo "cat data/extracted_graph_gemini/*.json | jq '.schema_proposals'"
echo ""

# =============================================================================
# STEP 6: Checkpoint Management
# =============================================================================

echo "=== Resume Interrupted Extraction ==="
echo "# The extractor automatically skips completed sections"
echo "# Just re-run the same command:"
echo "python src/knowledge_graph/extract_entities.py \\"
echo "    --provider anthropic \\"
echo "    --model claude-opus-4-5-20251101 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_opus"
echo ""

echo "=== Clear Checkpoints (Force Re-extraction) ==="
echo "# Delete existing extractions to force re-run"
echo "rm data/extracted_graph_claude_opus/*_extracted.json"
echo ""

# =============================================================================
# STEP 7: Batch API Extraction (50% Cost Reduction)
# =============================================================================

echo "=== Batch API: Full Workflow (Submit, Poll, Retrieve) ==="
echo "# Extract Harry Potter with Claude Opus 4.5 using batch API"
echo "python src/knowledge_graph/extract_entities_batch.py \\"
echo "    --model claude-opus-4-5-20251101 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_opus_batch \\"
echo "    --series 'Harry Potter'"
echo ""

echo "=== Batch API: Submit Only (Get Batch ID) ==="
echo "python src/knowledge_graph/extract_entities_batch.py \\"
echo "    --mode submit \\"
echo "    --model claude-opus-4-5-20251101 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_opus_batch"
echo ""

echo "=== Batch API: Check Status ==="
echo "python src/knowledge_graph/extract_entities_batch.py \\"
echo "    --mode status \\"
echo "    --batch-id msgbatch_xxx"
echo ""

echo "=== Batch API: Retrieve Results ==="
echo "python src/knowledge_graph/extract_entities_batch.py \\"
echo "    --mode retrieve \\"
echo "    --batch-id msgbatch_xxx \\"
echo "    --output-dir data/extracted_graph_claude_opus_batch"
echo ""

echo "=== Batch API: Extract All Series ==="
echo "python src/knowledge_graph/extract_entities_batch.py \\"
echo "    --model claude-opus-4-5-20251101 \\"
echo "    --input-dir data/processed_books_claude_200k \\"
echo "    --output-dir data/extracted_graph_claude_opus_batch"
echo ""

# =============================================================================
# Quick Reference: Model Context Windows
# =============================================================================

echo "=== Model Context Window Reference ==="
echo "| Model                     | Context Window | Recommended Chunking |"
echo "|---------------------------|----------------|----------------------|"
echo "| gemini-3-pro-preview      | 1,000,000      | 800,000 tokens       |"
echo "| claude-opus-4-5           | ~200,000       | 160,000 tokens       |"
echo "| claude-sonnet-4-5         | ~200,000       | 160,000 tokens       |"
echo "| gpt-5.2-pro               | ~400,000       | 320,000 tokens       |"
echo ""

echo "=== Cost Comparison: Real-time vs Batch API ==="
echo "| API Mode   | Cost      | Processing Time | Best For              |"
echo "|------------|-----------|-----------------|----------------------|"
echo "| Real-time  | Full      | Immediate       | Interactive, testing |"
echo "| Batch      | 50% off   | Up to 24 hours  | Bulk extraction      |"
echo ""

echo "=== End of Examples ==="
echo "For detailed documentation, see:"
echo "  - docs/MULTI_MODEL_EXTRACTION_GUIDE.md"
echo "  - docs/BATCH_API_GUIDE.md"

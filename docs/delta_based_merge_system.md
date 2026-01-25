# Delta-Based Merge System for Knowledge Graph Construction

## Overview

This document explains the conceptual design of a delta-based merge system for incremental knowledge graph construction when processing large documents with LLMs that have limited output token capacity.

---

## The Core Problem

When processing a large book series in chunks, you face a critical dilemma:

### Full Graph Approach (Current Implementation)

- **Process chunk 1** → Extract complete graph with 100 entities
- **Process chunk 2** → Extract complete graph with 250 entities (includes the 100 from before + 150 new)
- **Process chunk 3** → Extract complete graph with 450 entities
- **...**
- **Process chunk 10** → Try to extract complete graph with 2,000 entities... **FAILS** because the JSON output exceeds 64K tokens

### The Bottleneck

The model can **read** your 1M token context, but can only **write** 64K tokens (or less). As the accumulated graph grows, you hit the output ceiling even though the model could process more input.

**Known Output Limits:**
- **Gemini 3 Pro (API)**: 64K tokens
- **Gemini 3 Pro (Vertex AI)**: 32K tokens
- **Claude Sonnet 4.5 / Opus 4.5**: 64K tokens
- **GPT-4o Standard**: 16K-32K tokens
- **GPT-4o Long Output**: 64K tokens

---

## The Delta Concept

Instead of asking "give me the entire graph," you ask "what's new or different?"

Think of it like **version control (Git)**:
- Git doesn't store the entire codebase every time you commit
- It stores **diffs** (deltas) - what changed between commits
- You can reconstruct any version by applying deltas sequentially

Similarly, your knowledge graph construction becomes:

### Delta Approach

- **Process chunk 1** → Extract **complete** graph with 100 entities (first time only)
- **Process chunk 2** → Extract only **changes**: "Add 150 new entities, update Harry Potter's description with 2 new sentences, add 200 new relationships"
- **Process chunk 3** → Extract only **changes**: "Add 200 new entities, update 15 existing ones, add 300 relationships"
- **...**
- **Process chunk 10** → Extract only **changes**: "Add 50 entities, update 30 existing, add 100 relationships"

### Key Insight

**The output size now scales with *chunk complexity*, not *total graph size*.**

---

## The Four Delta Operations

### 1. ADD NEW ENTITY

When you encounter a brand new character, location, or object not in the previous graph.

**Example:**
```json
Chunk 5 introduces "Sirius Black" for the first time.

Delta operation: ADD
{
  "id": "sirius_black",
  "name": "Sirius Black",
  "type": "Character",
  "description": "Harry's godfather, escaped from Azkaban prison..."
}
```

### 2. UPDATE EXISTING ENTITY

When new context enriches an entity you've already seen.

**Example:**
```
Chunk 1: "Harry Potter is an 11-year-old boy living with the Dursleys."
Chunk 3: "Harry discovers he can speak Parseltongue."

Delta operation for chunk 3: UPDATE
{
  "id": "harry_potter",
  "description_append": "Discovered he can speak Parseltongue, suggesting a connection to Slytherin.",
  "attributes_to_update": {
    "parseltongue_speaker": true
  }
}
```

**Critical Rule:** Only append *genuinely new* information. Don't repeat what's already captured.

### 3. ADD NEW RELATIONSHIP

When you discover a new connection between entities.

**Example:**
```json
Chunk 7: "Dumbledore reveals that Snape and Lily Potter were childhood friends."

Delta operation: ADD RELATIONSHIP
{
  "source_id": "severus_snape",
  "target_id": "lily_potter",
  "type": "friend_of",
  "evidence": "They knew each other before Hogwarts, having grown up in the same town."
}
```

### 4. UPDATE EXISTING RELATIONSHIP (Less Common)

When new context changes how you understand a relationship.

**Example:**
```
Chunk 2: "Snape dislikes Harry."
Chunk 20: "Snape's dislike is complicated by his love for Harry's mother."

Delta operation: UPDATE RELATIONSHIP
{
  "source_id": "severus_snape",
  "target_id": "harry_potter",
  "type": "dislikes",
  "description_append": "However, this antagonism is deeply complicated by Snape's unrequited love for Harry's mother, creating internal conflict."
}
```

---

## The Incremental Processing Workflow

### Step-by-Step Process

#### Chunk 1 (Bootstrap)

1. **Send**: Schema + Text Chunk 1
2. **Receive**: **Full extraction** (100 entities, 150 relationships)
3. **Store locally**: `current_graph.json` with 100 entities

#### Chunk 2 (First Delta)

1. **Prepare context:**
   - Schema
   - **Compressed summary of current graph** (list of entity IDs + brief descriptions)
   - Text Chunk 2

2. **Prompt the model:**
   ```
   Here are the entities you've already extracted:
   [harry_potter, ron_weasley, hermione_granger, ...]

   Here are the relationships:
   [harry_potter -> friend_of -> ron_weasley, ...]

   Now process this new chunk. Return ONLY:
   - New entities not in the above list
   - Updates to existing entities (only NEW information)
   - New relationships
   - Updates to existing relationships (only if context changes)
   ```

3. **Receive delta:**
   - NEW: 50 new entities
   - UPDATE: 5 existing entities with additional context
   - NEW: 80 new relationships

4. **Merge locally:**
   - Load `current_graph.json`
   - Add the 50 new entities
   - For the 5 updates: append descriptions, merge attributes
   - Add the 80 new relationships
   - Save updated `current_graph.json` (now 150 entities)

#### Chunk 3, 4, 5... (Repeat)

Same process. Each time:
- Input includes the current state
- Output is just the delta
- Merge happens locally

---

## The Local Merge Logic

The merge process is deterministic and rule-based:

### Entity Merge

#### ADD Operation:
- Simply append the new entity to your entities list
- No conflict possible (new ID)

#### UPDATE Operation:
```
1. Find entity by ID in current graph
2. Aliases: Union of old + new (no duplicates)
3. Attributes: Shallow merge (new values overwrite old for same keys)
4. Description:
   - If description_append exists:
     current_description += "\n" + description_append
   - Else: keep current description
5. Mentions: Append new mentions to list
6. Confidence: Keep highest score
```

### Relationship Merge

#### ADD Operation:
- Append to relationships list
- Relationships are identified by (source_id, target_id, type) tuple

#### UPDATE Operation:
```
1. Find relationship by (source_id, target_id, type)
2. If description_append: current_description += "\n" + description_append
3. If evidence_append: current_evidence += "\n" + evidence_append
4. Properties: Shallow merge
5. Source references: Append new ones
```

---

## Handling the Output Token Limit

Let's calculate why this works:

### Full Graph Approach (Fails)

**Chunk 10 processing:**
- **Input**: Schema (5K tokens) + Previous graph (80K tokens) + New text (180K tokens) = 265K input ✓
- **Output**: Complete updated graph (120K tokens) = **EXCEEDS 64K LIMIT ✗**

### Delta Approach (Works)

**Chunk 10 processing:**
- **Input**: Schema (5K) + **Graph summary** (20K) + New text (180K) = 205K input ✓
- **Output**: **Just the delta** - 30 new entities, 15 updates, 60 relationships (8K tokens) = **WELL UNDER 64K LIMIT ✓**

### Key Differences

1. **Input side:** Send a compressed summary instead of full graph details
2. **Output side:** Receive only changes, not the complete graph

---

## Graph Summary vs. Full Graph

When sending context to the model about previous extractions:

### Full Graph (Don't Send)

```json
{
  "id": "harry_potter",
  "name": "Harry Potter",
  "aliases": ["The Boy Who Lived", "Potter", "The Chosen One"],
  "description": "Harry James Potter is an eleven-year-old wizard who survived the killing curse from Lord Voldemort as a baby, leaving him with a distinctive lightning bolt scar on his forehead. He was raised by his aunt and uncle, the Dursleys, who treated him poorly and hid his magical heritage from him. Upon turning eleven, he received his Hogwarts letter and discovered he was famous in the wizarding world...",
  "attributes": {
    "blood_status": "Half-blood",
    "house": "Gryffindor",
    "wand": "11 inch holly with phoenix feather core",
    ...
  },
  ...
}
```

**Token cost**: 500-1000 tokens per entity × 500 entities = **250K-500K tokens**

### Compressed Summary (Send This)

```
Entities already extracted:
- harry_potter (Character): The Boy Who Lived, protagonist
- hermione_granger (Character): Brilliant witch, Harry's friend
- ron_weasley (Character): Harry's best friend, Gryffindor
- voldemort (Character): Dark Lord, Harry's nemesis
... (500 more entries)

Relationships already extracted:
- harry_potter -> friend_of -> ron_weasley
- harry_potter -> enemy_of -> voldemort
- hermione_granger -> friend_of -> harry_potter
... (1000 more entries)
```

**Token cost**: 10-20 tokens per entity × 500 entities = **5K-10K tokens**

---

## Consistency & Quality Concerns

### How do you prevent duplicates?

#### Entity Deduplication

- The model sees the list of all entity IDs already extracted
- Prompt explicitly:
  > "If you see 'Harry,' 'Harry Potter,' or 'The Boy Who Lived,' they all map to entity ID 'harry_potter' which already exists. Do NOT create a new entity - add an UPDATE if there's new information."

#### Relationship Deduplication

- The model sees existing relationships
- Instruction:
  > "If Harry and Ron's friendship is already recorded, don't add it again unless there's genuinely new context."

### What if the model makes mistakes?

**Scenario:** Model creates a duplicate entity in chunk 5 that should have been an update.

**Solution 1 (Manual):** Post-processing deduplication script

**Solution 2 (Model-driven):** Include a "DELETE" operation type where the model can self-correct:

```json
Chunk 6 delta includes:
{
  "deletions": [{
    "entity_id": "h_potter_duplicate",
    "reason": "Duplicate of harry_potter, created in error"
  }]
}
```

### What if updates overwrite important information?

#### Protection Strategy

- Use **append-only** semantics for descriptions (never replace, always add)
- For attributes, newer values overwrite (assumption: newer is more accurate)
- Keep a changelog/history if you need auditability

---

## Edge Cases & Advanced Considerations

### 1. Chunk Boundaries

**Problem:** A relationship is mentioned across two chunks.

```
Chunk 5 ends: "Harry began to suspect that Snape..."
Chunk 6 starts: "...was actually protecting him all along."
```

**Solution:** The model processes chunk 6 with context of chunk 5's entities, so it can create the relationship when it has full context. The slight overlap or model's understanding bridges the gap.

### 2. Contradictions

**Problem:**
- Chunk 3: "Snape hates Harry"
- Chunk 20: "Snape loved Harry's mother and protected Harry"

**Solution:** Don't delete the first relationship. Add the second one too. Your graph now has:
- `snape -> antagonizes -> harry`
- `snape -> protects -> harry`

This captures the narrative complexity. Optionally, add UPDATE to the first with context:
> "This antagonism masks a deeper protective instinct."

### 3. Retcons & Corrections

If the model or source material reveals earlier information was wrong:

**Use DELETE operation:**

```json
{
  "deletions": [{
    "entity_id": "fake_character",
    "reason": "Later revealed to be an illusion/false identity"
  }]
}
```

### 4. Growing Context Size

Even graph summaries grow. After processing 100 chunks:

#### Mitigation Strategies

**A. Hierarchical Summarization:**
- After every 10 chunks, create a "super-summary"
- Send: Recent detailed summary (last 10 chunks) + older super-summary

**B. Relevance Filtering:**
- Use semantic search on entity descriptions
- Only send entities likely relevant to the current chunk
- Example: Chunk is about Hogwarts classes → prioritize teacher/student entities, deprioritize Ministry of Magic entities

**C. Sliding Window:**
- Keep detailed context for recent chunks (e.g., last 5)
- Older chunks: only send entity IDs without descriptions

---

## Why This Solves Your Problem

### Your Original Issue

> "Gemini with 1M context produces sparse graphs. When I use smaller chunks, I get duplicates and fragmentation."

### How Deltas Solve This

1. **Use Small Chunks (200K like Claude):** ✓
   - Better detail capture
   - Model doesn't overly summarize

2. **Avoid Duplicates:** ✓
   - Model sees what already exists
   - Explicitly prompted to update, not recreate

3. **Maintain Global Consistency:** ✓
   - Each chunk builds on the previous state
   - Cross-chunk relationships can be added as entities are discovered

4. **Never Hit Output Limits:** ✓
   - Output size bounded by chunk complexity, not total graph size
   - Even processing chunk 100 only outputs ~10K-20K tokens

5. **Preserve Detail:** ✓
   - Smaller chunks = more granular extraction
   - Incremental updates = nothing gets lost

---

## Comparison with Alternatives

### Alternative 1: Post-Processing Deduplication

Process chunks independently, then deduplicate.

**Problems:**
- Merging 10 different "Harry Potter" entities with conflicting attributes is hard
- Relationships might reference different entity IDs
- Loses narrative flow

### Alternative 2: RAG-Style Retrieval

For each chunk, retrieve relevant previous entities via vector search.

**Problems:**
- Doesn't guarantee consistency
- Might miss important entities not semantically similar
- Complex to implement

### Alternative 3: Full Graph Compression

Try to fit the full graph in output by aggressive compression.

**Problems:**
- Lose detail/richness
- Still hits limits eventually
- Model has to do extra work compressing

### Why Delta Approach Wins

- **Guarantees consistency** (single authoritative state)
- **Scales indefinitely** (output size doesn't grow)
- **Preserves all detail** (nothing compressed away)
- **Natural for LLMs** (similar to conversation memory)

---

## System Architecture

### Components

1. **GraphStateManager**
   - Maintains the current graph state on disk
   - Applies delta operations
   - Generates compressed summaries for model input

2. **Delta Schemas**
   - Pydantic models for ADD/UPDATE/DELETE operations
   - ExtractionDelta replaces ExtractionResult

3. **Incremental Extractor**
   - Orchestrates the chunk-by-chunk processing
   - Builds prompts with graph summaries
   - Calls GraphStateManager to merge results

4. **Merge Engine**
   - Implements deterministic merge rules
   - Handles deduplication
   - Validates consistency

### Data Flow

```
[Text Chunk N] + [Schema] + [Graph Summary N-1]
            ↓
    [LLM Processing]
            ↓
    [ExtractionDelta N]
            ↓
    [Merge Engine]
            ↓
    [Updated Graph State N]
            ↓
[Store to disk] → [Generate Summary N] → [Use for Chunk N+1]
```

---

## Implementation Considerations

### Token Budget Allocation

For a 200K token context window:

- **Schema**: ~5K tokens (fixed)
- **Graph Summary**: 20K-40K tokens (grows slowly)
- **Text Chunk**: 150K-175K tokens (main content)

Total: Fits comfortably within limits while allowing rich context.

### Performance Optimization

1. **Lazy Loading**: Only load full graph when merging, not for summary generation
2. **Incremental Summaries**: Update summary incrementally instead of regenerating
3. **Parallel Processing**: Process multiple series in parallel (independent graphs)
4. **Checkpointing**: Save state after each chunk for fault tolerance

### Quality Assurance

1. **Validation**: Ensure all referenced entity IDs in relationships exist
2. **Duplicate Detection**: Post-process to find near-duplicate entities
3. **Confidence Tracking**: Flag low-confidence extractions for review
4. **Diff Reports**: Generate human-readable change logs for each chunk

---

## Summary

### The delta-based merge system works by:

1. **Maintaining local state:** You keep the "source of truth" graph on disk
2. **Sending compressed context:** Model sees what exists but in brief form
3. **Receiving minimal changes:** Model only outputs new/updated information
4. **Merging deterministically:** You apply the delta with clear rules
5. **Repeating:** Each chunk builds on the last

### This solves the output token limit

The model never tries to output the entire graph—only the changes from one chunk to the next.

### This maintains consistency

The model explicitly knows what already exists and is instructed to extend rather than recreate.

### This scales indefinitely

Output size is bounded by individual chunk complexity, not cumulative graph size.

---

## Next Steps

1. Implement delta schemas (Pydantic models)
2. Build GraphStateManager class
3. Create incremental extraction script
4. Modify prompts for delta-based extraction
5. Test with a single book first
6. Scale to full series
7. Add quality assurance tools
8. Optimize for production use

---

*Document Version: 1.0*
*Last Updated: 2026-01-16*

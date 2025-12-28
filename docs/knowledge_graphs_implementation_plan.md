# Comprehensive Implementation Plan for Building High-Quality Knowledge Graphs for Fantasy Book Series Using Gemini 3 Pro

## Project Overview

You are building separate knowledge graphs for three major fantasy book series:

- **Harry Potter series** by J.K. Rowling
- **A Song of Ice and Fire series** by George R.R. Martin
- **The Wheel of Time series** by Robert Jordan

The goal is to create canonically accurate, relationship-rich knowledge graphs that capture characters, locations, events, organizations, magical systems, and their complex interconnections across entire narrative arcs.

---

## Current Data State & Challenge

### Existing Chunked Data (Not Suitable for Knowledge Graph Construction)

You currently have pre-chunked data:

- **Harry Potter**: 8,494 chunks, averaging 849 characters per chunk
- **A Song of Ice and Fire**: 16,644 chunks, averaging 781 characters per chunk
- **Wheel of Time**: 36,838 chunks, averaging 778 characters per chunk

**Critical Problem**: These chunks were created for RAG (Retrieval-Augmented Generation) retrieval purposes, where small, semantically coherent chunks are ideal for finding relevant passages. However, **knowledge graph construction requires the opposite approach**—the model needs to see large, continuous narrative contexts to:

1. **Maintain entity consistency** (understanding that "Harry," "Harry Potter," "The Boy Who Lived," and "The Chosen One" all refer to the same person)
2. **Track relationship evolution** (how Ron and Hermione's relationship develops from friendship in Book 1 to romance in Book 7)
3. **Understand temporal sequences** (the chronological order of events across multiple books)
4. **Resolve entity disambiguation** (distinguishing Barty Crouch Sr. from Barty Crouch Jr.)
5. **Capture implicit context** (knowing that "The Dragon Reborn" refers to Rand al'Thor based on earlier books)

**Knowledge graphs break when the model can't see the surrounding canon.** Processing small chunks independently results in:

- Entity fragmentation (same character appearing as multiple disconnected entities)
- Incomplete relationship graphs (missing connections that span chapters or books)
- Timeline inconsistencies (events appearing out of order)
- Loss of narrative context (references that make no sense without earlier context)

---

## Solution: Extract Full-Context Text from Original Books

**You need to re-extract text from the original book files (EPUB format) specifically for knowledge graph construction.**

### Why Re-extract?

1. **Preserve narrative continuity**: Extract complete books or large book sections that maintain story flow
2. **Maximize context windows**: Create extracts sized to fit Gemini 3 Pro's large context window (2 million tokens)
3. **Include all canonical text**: Ensure no information is lost during chunking
4. **Maintain chronological order**: Books processed in their publication/reading order
5. **Enable cross-book understanding**: Allow the model to see how stories connect across series installments

---

## Token Count Analysis

Based on your chunk statistics, here are the approximate **total token counts** for each series:

### Harry Potter Series

- Total chunks: 8,494
- Average chunk size: 849 characters
- **Total characters**: 8,494 × 849 ≈ 7,211,406 characters
- **Estimated tokens**: ~1,800,000 tokens (using 4 chars ≈ 1 token)
- **Books**: 9 (7 main series + 2 companion books)

### A Song of Ice and Fire Series

- Total chunks: 16,644
- Average chunk size: 781 characters
- **Total characters**: 16,644 × 781 ≈ 12,998,964 characters
- **Estimated tokens**: ~3,250,000 tokens
- **Books**: 8 (5 main novels + 3 companion books)

### Wheel of Time Series

- Total chunks: 36,838
- Average chunk size: 778 characters
- **Total characters**: 36,838 × 778 ≈ 28,659,964 characters
- **Estimated tokens**: ~7,165,000 tokens
- **Books**: 16 (14 main series + 1 prequel + 1 companion)

**Total across all series**: ~12,215,000 tokens

---

## Rate Limits Analysis

You have access to Gemini 3 Pro through **two API pathways**, both with identical rate limits since OpenRouter routes Gemini requests to Google's backend:

### Google Vertex AI API (GEMINI_API_KEY)

**Paid Tier 1 Limits:**

- **Requests per minute**: 25
- **Requests per day**: 250
- **Input tokens per minute**: 1,000,000 tokens

### OpenRouter API (OPENROUTER_API_KEY)

**When using Gemini 3 Pro through OpenRouter:**

- Same backend (Google's infrastructure)
- Same rate limits: 25 req/min, 250 req/day, 1M tokens/min
- **Advantage**: Can fallback to other models (Claude, GPT-4) if needed
- **Your status**: Paid tier (`is_free_tier: False`), unlimited spending, no hard caps

### Critical Rate Limit Constraint

The **250 requests per day** limit is your primary constraint. This means:

- You need to carefully plan request allocation across the multi-day extraction process
- Each request must maximize context (send large text blocks, not small chunks)
- Aim for **80-120 total requests** across all three series to stay comfortably within limits
- Process can take **5-7 days** at a conservative pace

The **1 million input tokens per minute** limit is generous—you can send large book sections without hitting this throttle in normal operation.

---

## Three-Phase Implementation Strategy

### Phase 1: Full-Context Text Re-Extraction from Books

**Objective**: Extract clean, continuous text from original EPUB files in large context blocks optimized for Gemini 3 Pro's 2M token window.

**Process**:

1. **For each series, extract text book-by-book** maintaining:

   - Chapter structure and boundaries
   - Chronological reading order
   - Book metadata (title, book number in series)
   - Clean formatting (remove EPUB artifacts, images, metadata sections)

2. **Create extraction chunks sized at ~800,000-1,500,000 tokens** to fit comfortably within Gemini 3 Pro's 2M token context window with room for system prompts and output

3. **Implement intelligent book grouping**:

   - **Harry Potter** (~1.8M tokens total): Can be sent as 2 large sections

     - Section 1: Books 1-4 (~900K tokens)
     - Section 2: Books 4-7 (~900K tokens with Book 4 overlap for continuity)

   - **A Song of Ice and Fire** (~3.25M tokens total): Split into 3 sections

     - Section 1: Books 1-3 (~1.2M tokens)
     - Section 2: Books 3-5 (~1.2M tokens with Book 3 overlap)
     - Section 3: Books 5-8 (~1.2M tokens with Book 5 overlap)

   - **Wheel of Time** (~7.15M tokens total): Split into 5 sections
     - Section 1: Books 1-3 (~1.4M tokens)
     - Section 2: Books 3-6 (~1.4M tokens with Book 3 overlap)
     - Section 3: Books 6-9 (~1.4M tokens with Book 6 overlap)
     - Section 4: Books 9-12 (~1.4M tokens with Book 9 overlap)
     - Section 5: Books 12-16 (~1.4M tokens with Book 12 overlap)

4. **Overlap strategy**: Include 1-2 books at boundaries between sections to ensure the model sees character/relationship continuity across section breaks

5. **Output format**: Save each extracted section as a clean text file with clear markers for:
   - Series name
   - Books included in this section
   - Section number
   - Total estimated tokens

**Why this matters**: By re-extracting text specifically for knowledge graph construction rather than using your existing RAG chunks, you preserve the narrative continuity and context that prevents knowledge graph fragmentation.

---

### Phase 2: Schema Generation & Ontology Creation (Per Series)

**Objective**: Have Gemini 3 Pro analyze each complete series to generate a comprehensive ontology defining all entities, relationship types, and canonical naming conventions.

**Process for Each Series**:

1. **Send the largest possible context** (preferably entire series if it fits, otherwise largest book groupings)

2. **Prompt the model to generate**:

   - **Complete entity inventory**:

     - All characters (with aliases, titles, nicknames)
     - All locations (with alternate names, regional variations)
     - All organizations (Houses, Orders, Factions, Governments)
     - All magical/supernatural elements (spells, artifacts, powers)
     - All significant events (battles, treaties, ceremonies)

   - **Canonical entity naming**:

     - Primary name for each entity
     - All known aliases and how they map to primary names
     - Disambiguation rules (e.g., "Jon Snow" vs "Jon Connington")

   - **Relationship type taxonomy**:

     - Family relationships (parent, child, sibling, spouse)
     - Social relationships (friend, enemy, ally, rival, mentor, student)
     - Political relationships (ruler, subject, ally, vassal)
     - Organizational relationships (member of, leader of, founded)
     - Location relationships (lives in, born in, rules, visited)
     - Temporal relationships (occurred before, caused, prevented)

   - **Temporal framework**:
     - Timeline of major events
     - Book boundaries (which events happen in which books)
     - Chronological ordering across series

3. **Request multiple schema passes** if needed:
   - First pass: High-level structure (major entities and relationships)
   - Second pass: Detailed attributes and rare entities
   - Third pass: Validation and gap-filling

**Resource Allocation**:

- **Harry Potter**: 4-6 requests (2 sections × 2-3 schema passes)
- **A Song of Ice and Fire**: 6-9 requests (3 sections × 2-3 passes)
- **Wheel of Time**: 10-15 requests (5 sections × 2-3 passes)
- **Total Phase 2**: ~20-30 requests over 2-3 days

**Critical Success Factor**: The model must see complete narrative arcs to understand that "Hermione," "Hermione Granger," and "the brightest witch of her age" are the same entity, and that her relationship with Ron evolves from acquaintance → friend → romantic interest → spouse across seven books.

---

### Phase 3: Detailed Entity & Relationship Extraction (Per Series)

**Objective**: Extract granular entities, relationships, attributes, and events while maintaining consistency with the schema from Phase 2.

**Process for Each Series**:

1. **Process each large text section** (the same sections from Phase 1)

2. **Include the generated schema from Phase 2** in every request as reference context to ensure extraction consistency

3. **Prompt the model to extract**:

   - All entity mentions with:

     - Canonical entity ID (mapped to schema)
     - Specific mention/alias used in text
     - Book and approximate chapter location
     - Attributes revealed in this mention

   - All relationships with:

     - Source entity (canonical ID)
     - Relationship type (from schema taxonomy)
     - Target entity (canonical ID)
     - Relationship attributes (strength, duration, sentiment)
     - Evidence text (quote supporting this relationship)
     - Book and chapter where relationship is revealed/changed

   - All events with:
     - Event name/description
     - Participating entities
     - Location
     - Temporal information (when in timeline)
     - Causal relationships (what led to this, what resulted)

4. **Handle schema evolution**:

   - Flag any NEW entities not in Phase 2 schema
   - Flag any NEW relationship types
   - Request model to suggest schema updates

5. **Request format**: Structure extraction output as JSON or structured format for easy graph database import

**Resource Allocation**:

- **Harry Potter**: 4-6 requests (2 sections × 2-3 extraction passes)
- **A Song of Ice and Fire**: 6-12 requests (3 sections × 2-4 passes)
- **Wheel of Time**: 10-20 requests (5 sections × 2-4 passes)
- **Total Phase 3**: ~20-38 requests over 2-3 days

**Why large contexts are essential here**: When extracting from "Harry Potter and the Deathly Hallows," the model needs to remember context from earlier books to correctly identify that "the Elder Wand" mentioned here connects to "Dumbledore's wand" mentioned in earlier books, and that both refer to one of the "Deathly Hallows" introduced in this final book.

---

### Phase 4: Entity Resolution & Graph Consolidation (Per Series)

**Objective**: Merge all extracted data, resolve any remaining entity duplicates, reconcile conflicting information, and produce final clean knowledge graph.

**Process for Each Series**:

1. **Aggregate all extraction outputs** from Phase 3 into a preliminary graph structure

2. **Send consolidated entity lists and relationship graphs** to Gemini 3 Pro for final resolution

3. **Prompt the model to**:

   - **Merge duplicate entities**: Identify and consolidate any entities that were split across sections (e.g., if "Hermione" in Section 1 was treated as different from "Hermione Granger" in Section 2)

   - **Resolve relationship conflicts**: When two sections provide contradictory relationship information, determine the correct relationship based on narrative context

   - **Validate timeline consistency**: Ensure events are in proper chronological order and no temporal contradictions exist

   - **Complete missing connections**: Identify implied relationships that may have been missed (e.g., if A is B's parent and B is C's parent, then A is C's grandparent)

   - **Attribute consolidation**: Merge character attributes collected from different sections

   - **Quality scoring**: Assign confidence scores to entities and relationships based on evidence strength

4. **Generate final graph statistics**:
   - Total unique entities by type
   - Total relationships by type
   - Graph density and connectivity metrics
   - Coverage analysis (percentage of books fully processed)

**Resource Allocation**:

- **Harry Potter**: 2-3 requests
- **A Song of Ice and Fire**: 3-5 requests
- **Wheel of Time**: 4-6 requests
- **Total Phase 4**: ~9-14 requests over 1-2 days

**Critical validation step**: This phase catches problems like "Daenerys Targaryen," "Dany," "Khaleesi," and "Mother of Dragons" being treated as four separate entities instead of one character with multiple titles.

---

## API Key Strategy & Rate Limit Management

### Dual API Key Usage Plan

**Primary**: Use **GEMINI_API_KEY** (Google Vertex AI direct)

- More reliable (no intermediary)
- Direct Google infrastructure
- Slightly lower latency

**Backup**: Use **OPENROUTER_API_KEY** when:

- You hit daily request limits on primary key
- You need to distribute requests across two pathways to increase effective throughput
- Google API experiences downtime
- You want to test alternative models (Claude, GPT-4) for comparison

### Rate Limit Management Strategy

**Total estimated requests across all phases**: 80-120 requests

**Daily allocation** (to stay within 250 req/day limit):

**Day 1-2: Phase 2 Schema Generation**

- Process Harry Potter: 6 requests
- Process ASOIAF: 9 requests
- Process WOT: 15 requests
- **Daily load**: 15 requests/day (well under limit)

**Day 3-5: Phase 3 Detailed Extraction**

- Day 3: Harry Potter (6 requests) + ASOIAF start (12 requests) = 18 requests
- Day 4: ASOIAF finish (0 requests) + WOT start (20 requests) = 20 requests
- Day 5: WOT finish (0 requests)
- **Daily load**: 20 requests/day (well under limit)

**Day 6-7: Phase 4 Entity Resolution**

- All three series: 14 requests spread over 2 days
- **Daily load**: 7 requests/day

**Total timeline**: 7 days at conservative pace, could be compressed to 4-5 days if needed

### Token Rate Management

With **1 million tokens/minute** limit and typical requests of ~800K-1.5M tokens:

- You can send one large request every 1-2 minutes
- For a day with 20 requests, spread them over 40 minutes (very manageable)
- **No token throttling concerns** with this workload

---

## Quality Assurance Principles

### Why This Approach Produces High-Quality Knowledge Graphs

1. **Narrative Continuity**: Model sees complete story arcs, understanding character development and relationship evolution

2. **Entity Consistency**: By seeing all mentions of an entity across entire series, model correctly identifies that different names/titles refer to the same entity

3. **Relationship Depth**: Can identify complex, evolving relationships that span multiple books (e.g., Rand al'Thor's relationships with multiple women across 14 books)

4. **Temporal Accuracy**: Understands chronological event sequences and cause-effect relationships

5. **Context Preservation**: Maintains understanding of implicit references, prophecies, foreshadowing, and narrative callbacks

6. **Canonical Accuracy**: Schema-driven extraction ensures all entities use consistent canonical names

7. **Attribute Completeness**: Captures how character attributes, allegiances, and abilities change throughout the series

---

## Expected Outputs

### For Each Series, You Will Have:

1. **Entity Database**:

   - Canonical entity list with all aliases
   - Entity attributes (for characters: age, house, abilities, titles, etc.)
   - Entity mentions mapped to book/chapter locations

2. **Relationship Graph**:

   - All relationships between entities
   - Relationship type taxonomy
   - Temporal information (when relationships formed/changed/ended)
   - Relationship attributes (strength, sentiment, duration)

3. **Event Timeline**:

   - Chronological sequence of major events
   - Event participants and locations
   - Causal chains (which events led to others)

4. **Knowledge Graph Statistics**:

   - Entity counts by type
   - Relationship counts by type
   - Graph connectivity metrics
   - Coverage analysis

5. **Graph Visualization Data**:
   - Node and edge lists formatted for graph databases (Neo4j, etc.)
   - JSON/CSV exports for analysis tools
   - Query-ready format for RAG integration

---

## Critical Success Factors

1. **Re-extract text from books**: Do NOT use existing RAG chunks—extract fresh, continuous text in large sections

2. **Maximize context windows**: Send 800K-1.5M token sections to leverage Gemini 3 Pro's 2M token capacity

3. **Maintain overlap**: Include 1-2 books at section boundaries to preserve continuity

4. **Schema-driven extraction**: Always include Phase 2 schema in Phase 3 extraction requests

5. **Entity resolution**: Dedicate adequate resources to Phase 4 consolidation to catch entity duplicates

6. **Rate limit planning**: Spread work over 5-7 days to comfortably stay within 250 req/day limit

7. **Quality validation**: Manually spot-check extracted entities and relationships against source text

---

## Summary

This implementation plan ensures you build **canonically accurate, relationship-rich knowledge graphs** by:

- Re-extracting book text in large, continuous sections (NOT using existing chunks)
- Leveraging Gemini 3 Pro's massive 2M token context window to see complete narrative arcs
- Using a three-phase approach: Schema → Extraction → Resolution
- Working within your rate limits: 25 req/min, 250 req/day, 1M tokens/min
- Processing ~12.2M tokens across three series in ~80-120 API requests over 5-7 days
- Using both GEMINI_API_KEY and OPENROUTER_API_KEY for redundancy and flexibility

The result will be high-quality knowledge graphs that understand complex relationships, maintain entity consistency, and preserve temporal narrative structures—exactly what you need for sophisticated RAG applications on these rich fantasy series.

"""Prompt templates for RAG Evaluation Q&A generation.

This module contains prompts specifically designed for generating Q&A pairs
that can be used to evaluate RAG (Retrieval-Augmented Generation) systems.

Key requirements:
- Each Q&A must explicitly reference which chunk IDs it was derived from
- Questions should be answerable by retrieving the correct chunks
- Supports multi-chunk questions (spanning 2-3 chunks for harder evaluation)
"""

RAG_EVAL_SYSTEM_INSTRUCTION = """You are an expert at creating Question & Answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.

Your task is to generate question-answer pairs from the provided text chunks. Each chunk has a unique ID that you MUST track and return.

CRITICAL REQUIREMENTS:

1. SOURCE TRACKING (MANDATORY):
   - Every Q&A pair MUST include the exact chunk_id(s) that contain the answer
   - If a question requires information from multiple chunks, list ALL relevant chunk_ids
   - The chunk_ids you return MUST exactly match the IDs provided in the input

2. QUESTION QUALITY:
   - Questions should be specific and answerable from the provided chunks
   - Mix of single-chunk questions (easy) and multi-chunk questions (harder)
   - Avoid yes/no questions - ask "what", "how", "why", "who" questions
   - Questions should test whether a RAG system can retrieve the right chunks

3. ANSWER QUALITY:
   - Answers must be grounded ONLY in the provided chunk text
   - Include enough detail to verify correctness (3-5 sentences)
   - Do not use external knowledge

4. EVIDENCE:
   - Provide a direct quote from the source chunk(s) that supports the answer
   - The quote must actually appear in the provided text

5. COVERAGE:
   - Try to generate questions that cover different chunks
   - Don't focus all questions on just one or two chunks
   - Aim for broad coverage across the provided content

You are working with fantasy book content. Generate diverse, high-quality questions that will effectively test RAG retrieval accuracy."""


def get_rag_eval_prompt(chunks: list[dict]) -> str:
    """Generate prompt for RAG evaluation Q&A generation.

    Args:
        chunks: List of RAG chunk documents with chunk_id and text_content

    Returns:
        Formatted prompt string with chunk IDs clearly labeled
    """
    # Format chunks with clear ID labels
    chunks_text = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk["text_content"]
        # Include metadata for context
        metadata = chunk.get("metadata", {})
        book = metadata.get("book_name", "Unknown")
        chapter = metadata.get("chapter_title", "Unknown")

        chunks_text.append(
            f"=== CHUNK_ID: {chunk_id} ===\n"
            f"[Book: {book} | Chapter: {chapter}]\n"
            f"{text}\n"
            f"=== END {chunk_id} ===\n"
        )

    all_chunks_text = "\n".join(chunks_text)
    chunk_ids_list = [c["chunk_id"] for c in chunks]

    return f"""Analyze the following text chunks and generate Question-Answer pairs for RAG evaluation.

## AVAILABLE CHUNK IDs (use these EXACTLY in your responses):
{', '.join(chunk_ids_list)}

---

## TEXT CHUNKS:

{all_chunks_text}

---

## GENERATION INSTRUCTIONS:

1. Generate as many high-quality Q&A pairs as possible from this content
2. For EACH Q&A pair, you MUST specify which chunk_id(s) contain the answer
3. The chunk IDs in your response must EXACTLY match the IDs listed above
4. Mix of question types:
   - Single-chunk questions: Answer found in one chunk
   - Multi-chunk questions: Answer requires combining info from 2-3 chunks
5. Cover different topics and chunks - don't focus on just one area
6. Questions should be specific enough that only the correct chunk(s) can answer them

IMPORTANT: The source_chunk_ids field is CRITICAL for RAG evaluation.
Double-check that each chunk_id you reference actually contains the relevant information."""

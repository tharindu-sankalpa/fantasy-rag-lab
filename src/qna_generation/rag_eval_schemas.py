"""Pydantic schemas for RAG Evaluation Q&A dataset generation.

This module defines data models for generating Question & Answer pairs
specifically designed for RAG system evaluation. Each Q&A pair is linked
to the exact source chunk IDs it was derived from.

Key differences from regular QA generation:
- Uses RAG chunks (small, ~1000 token chunks) instead of graph chunks
- Tracks source_chunk_ids as an array (questions may span multiple chunks)
- Category is always "rag_evaluation"
- No category-based classification (characters/magic/etc.)
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class RAGEvalQAPair(BaseModel):
    """A single Q&A pair for RAG evaluation with source chunk tracking.

    The source_chunk_ids field is critical for RAG evaluation as it allows
    us to verify that the retrieval system can find the correct chunks.
    """

    question: str = Field(
        ...,
        description="A specific question that can be answered from the provided chunks",
        min_length=10,
    )
    answer: str = Field(
        ...,
        description="Detailed answer grounded in the chunk text (3-5 sentences)",
        min_length=50,
    )
    source_chunk_ids: list[str] = Field(
        ...,
        description="List of chunk IDs that contain the information needed to answer this question",
        min_length=1,
    )
    evidence_quote: str = Field(
        ...,
        description="Direct quote from the chunks that supports the answer",
        min_length=20,
    )


class RAGEvalGenerationResult(BaseModel):
    """Result from a RAG evaluation Q&A generation request.

    Contains multiple Q&A pairs generated from a batch of RAG chunks.
    """

    qa_pairs: list[RAGEvalQAPair] = Field(
        ...,
        description="List of generated QA pairs with source tracking",
        min_length=1,
    )
    generation_notes: Optional[str] = Field(
        default=None,
        description="Notes about generation process or content coverage",
    )


class RAGEvalQADocument(BaseModel):
    """MongoDB document schema for RAG evaluation Q&A pairs.

    Stored in the wot_qna collection with category="rag_evaluation".
    The source_chunk_ids array enables proper RAG evaluation by linking
    each question to its ground truth retrieval targets.
    """

    qa_id: str = Field(
        ...,
        description="Unique identifier: {series}_rag_eval_{batch}_{index:04d}",
    )
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer text")
    category: str = Field(
        default="rag_evaluation",
        description="Always 'rag_evaluation' for this type",
    )
    evidence_quote: str = Field(..., description="Supporting quote from chunks")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata including source_chunk_ids, included_books, etc.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class RAGEvalProgress(BaseModel):
    """Tracks progress of RAG evaluation Q&A generation.

    Used for resumable processing and progress reporting.
    """

    series: str = Field(..., description="Series being processed")
    total_chunks: int = Field(..., description="Total RAG chunks in series")
    processed_batches: int = Field(default=0, description="Batches processed")
    total_batches: int = Field(default=0, description="Total batches to process")
    total_qa_pairs: int = Field(default=0, description="Total QA pairs generated")
    last_processed_batch: int = Field(default=0, description="Last batch index")
    failed_batches: list[int] = Field(
        default_factory=list, description="Batch indices that failed"
    )
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

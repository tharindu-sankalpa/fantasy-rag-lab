"""Pydantic schemas for QA dataset generation.

This module defines the data models for generating Question & Answer pairs
from large context chunks. The schemas are designed to work with Gemini's
native JSON mode for structured output generation.

The schemas enforce:
- Strict context grounding (answers must cite the source)
- Category-based classification (5 pillars)
- Complexity indicators for balanced datasets
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """The 5 pillars of question categories.

    Each category represents a distinct aspect of the story that can be
    queried, ensuring comprehensive coverage for RAG evaluation.
    """

    CHARACTERS = "characters"  # Who people really are
    EVENTS = "events"  # What happened
    MAGIC = "magic"  # How the system works
    ARTIFACTS = "artifacts"  # Special objects/locations
    LORE = "lore"  # Deep history & metaphysics


class QuestionComplexity(str, Enum):
    """Complexity level of the question.

    Used to ensure the dataset contains appropriately challenging questions
    for meaningful RAG evaluation.
    """

    SIMPLE = "simple"  # Single fact, direct answer
    MODERATE = "moderate"  # Requires 2-3 pieces of information
    COMPLEX = "complex"  # Multi-hop reasoning
    EXPERT = "expert"  # Deep synthesis of multiple elements


class QAPair(BaseModel):
    """A single Question-Answer pair.

    The answer must be entirely grounded in the provided chunk text.
    No external knowledge should be used.
    """

    question: str = Field(
        ...,
        description="A specific, well-formed question about the chunk content",
        min_length=10,
    )
    answer: str = Field(
        ...,
        description="Detailed, explanatory answer grounded in the chunk text (3-5 sentences)",
        min_length=50,
    )
    complexity: QuestionComplexity = Field(
        ..., description="Difficulty level of the question"
    )
    evidence_quote: str = Field(
        ...,
        description="Direct quote from the chunk that supports the answer",
        min_length=20,
    )


class QAGenerationResult(BaseModel):
    """Result from a single QA generation request.

    Contains multiple QA pairs generated from a chunk for a specific category.
    """

    qa_pairs: list[QAPair] = Field(
        ..., description="List of generated QA pairs", min_length=1
    )
    generation_notes: Optional[str] = Field(
        default=None,
        description="Any notes about the generation process or limitations",
    )


class QADocument(BaseModel):
    """MongoDB document schema for storing generated QA pairs.

    This is the final document structure stored in the wot_qna collection.
    Each document represents a single QA pair with full provenance tracking.
    """

    qa_id: str = Field(
        ..., description="Unique identifier: {source_chunk_id}_{category}_{index:04d}"
    )
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer text")
    category: str = Field(..., description="Question category (characters/events/magic/artifacts/lore)")
    complexity: str = Field(..., description="Complexity level")
    evidence_quote: str = Field(..., description="Supporting quote from chunk")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata including source_chunk_id, included_books",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class GenerationProgress(BaseModel):
    """Tracks progress of QA generation across chunks.

    Used for resumable processing and progress reporting.
    """

    series: str = Field(..., description="Series being processed")
    category: str = Field(..., description="Category being processed")
    total_chunks: int = Field(..., description="Total number of chunks")
    processed_chunks: int = Field(default=0, description="Chunks processed so far")
    total_qa_pairs: int = Field(default=0, description="Total QA pairs generated")
    failed_chunks: list[str] = Field(
        default_factory=list, description="Chunk IDs that failed"
    )
    last_processed_chunk: Optional[str] = Field(
        default=None, description="Last successfully processed chunk ID"
    )
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

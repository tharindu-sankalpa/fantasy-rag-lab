"""QA Dataset Generation for RAG Evaluation.

This package provides tools for automatically generating high-quality
Question & Answer datasets from large context chunks for RAG evaluation.

The system uses 5 question categories (pillars) for comprehensive coverage:
1. Characters & Identities - Who people really are
2. Major Events & Deaths - What happened
3. Magic & Power Mechanics - How the system works
4. Artifacts & Places - Special objects/locations
5. Lore & World-Building - Deep history & metaphysics

Modules:
- schemas: Pydantic models for QA data structures
- prompts: Category-specific prompt templates for Gemini
- service: Core QA generation service with rate limiting
- generate: CLI entry point

Usage:
    # Generate questions for a category
    uv run python -m src.qna_generation.generate --category characters

    # Show statistics
    uv run python -m src.qna_generation.generate --stats-only
"""

from .prompts import CATEGORY_INFO, QuestionCategory, get_category_prompt
from .schemas import (
    GenerationProgress,
    QADocument,
    QAGenerationResult,
    QAPair,
    QuestionComplexity,
)
from .service import QAGenerationService, RateLimiter

__all__ = [
    # Schemas
    "QAPair",
    "QADocument",
    "QAGenerationResult",
    "QuestionComplexity",
    "GenerationProgress",
    # Prompts
    "QuestionCategory",
    "CATEGORY_INFO",
    "get_category_prompt",
    # Service
    "QAGenerationService",
    "RateLimiter",
]

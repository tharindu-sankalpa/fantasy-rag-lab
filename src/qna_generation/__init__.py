"""QA Dataset Generation for RAG Evaluation.

This package provides tools for automatically generating high-quality
Question & Answer datasets from large context chunks for RAG evaluation.

Two generation modes are available:

1. Category-Based (generate.py):
   Uses 5 question categories for comprehensive coverage:
   - Characters & Identities - Who people really are
   - Major Events & Deaths - What happened
   - Magic & Power Mechanics - How the system works
   - Artifacts & Places - Special objects/locations
   - Lore & World-Building - Deep history & metaphysics

2. RAG Evaluation (generate_rag_eval.py):
   Generates Q&A pairs with explicit source chunk tracking for RAG system evaluation.
   Uses smaller RAG chunks (~1000 tokens) and tracks which chunks contain answers.

Usage:
    # Category-based generation
    uv run python -m src.qna_generation.generate --category characters

    # RAG evaluation generation
    uv run python -m src.qna_generation.generate_rag_eval --series wheel_of_time
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

# RAG Evaluation imports
from .rag_eval_schemas import (
    RAGEvalGenerationResult,
    RAGEvalProgress,
    RAGEvalQADocument,
    RAGEvalQAPair,
)
from .rag_eval_service import RAGEvalQAService

__all__ = [
    # Category-Based Schemas
    "QAPair",
    "QADocument",
    "QAGenerationResult",
    "QuestionComplexity",
    "GenerationProgress",
    # Category-Based Prompts
    "QuestionCategory",
    "CATEGORY_INFO",
    "get_category_prompt",
    # Category-Based Service
    "QAGenerationService",
    "RateLimiter",
    # RAG Evaluation Schemas
    "RAGEvalQAPair",
    "RAGEvalQADocument",
    "RAGEvalGenerationResult",
    "RAGEvalProgress",
    # RAG Evaluation Service
    "RAGEvalQAService",
]

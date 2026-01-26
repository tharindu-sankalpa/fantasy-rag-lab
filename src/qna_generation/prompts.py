"""Prompt templates for QA generation.

This module contains the carefully crafted prompt templates used to instruct
Gemini to generate high-quality, context-grounded QA pairs.

The prompts are organized by 5 question categories:
1. Characters & Identities - Who people really are
2. Major Events & Deaths - What happened
3. Magic & Power Mechanics - How the system works
4. Artifacts & Places - Special objects/locations
5. Lore & World-Building - Deep history & metaphysics

Each category has a dedicated prompt to ensure diverse question generation
across multiple passes.
"""

from enum import Enum
from typing import NamedTuple


class QuestionCategory(str, Enum):
    """The 5 pillars of question categories for RAG evaluation."""

    CHARACTERS = "characters"
    EVENTS = "events"
    MAGIC = "magic"
    ARTIFACTS = "artifacts"
    LORE = "lore"


class CategoryInfo(NamedTuple):
    """Metadata for a question category."""

    name: str
    description: str
    focus_areas: list[str]
    example_questions: list[str]


# Category definitions with examples
CATEGORY_INFO: dict[QuestionCategory, CategoryInfo] = {
    QuestionCategory.CHARACTERS: CategoryInfo(
        name="Characters & Identities",
        description="Questions about who someone is, their real name, origin, role, or identity.",
        focus_areas=[
            "Character true names and aliases",
            "Character origins and backgrounds",
            "Character roles and titles",
            "Character relationships and family ties",
            "Character transformations or identity changes",
            "Secret identities revealed",
        ],
        example_questions=[
            "Who led the effort to create the Bore?",
            "What was Lanfear's original name before she swore allegiance to the Dark One?",
            "Who is the only Forsaken from the modern age (not the Age of Legends)?",
            "Who are the Gray Men, and what are they?",
            "Who are the Aelfinn and the Eelfinn?",
            "Who is Rand's mentor connected to the Age of Legends?",
        ],
    ),
    QuestionCategory.EVENTS: CategoryInfo(
        name="Major Events & Deaths",
        description="Questions about important plot events, battles, defeats, deaths, or turning points.",
        focus_areas=[
            "Character deaths and circumstances",
            "Major battles and their outcomes",
            "Pivotal plot moments",
            "Defeats and victories",
            "Rescue missions and escapes",
            "Political upheavals and power shifts",
        ],
        example_questions=[
            "What was the method and circumstance of Egwene al'Vere's death in the final battle?",
            "Who stilled Siuan Sanche, and under what circumstances did it happen?",
            "Who kills the giant Shadowspawn at Shayol Ghul, and how is it defeated?",
            "What were the circumstances of Lanfear's final fate?",
            "What happened to Moghedien after she was defeated by Nynaeve and Elayne?",
            "When and how did Mat Cauthon rescue Moiraine from the Aelfinn and Eelfinn?",
        ],
    ),
    QuestionCategory.MAGIC: CategoryInfo(
        name="Magic, Weaves & Power Mechanics",
        description="Questions about how the One Power works, special abilities, weaves, and techniques.",
        focus_areas=[
            "Specific weaves and how they work",
            "Lost or ancient techniques",
            "Channeling abilities and limitations",
            "Healing and stilling/gentling",
            "Combat techniques with the Power",
            "Unique magical abilities",
        ],
        example_questions=[
            "Are there moments where Rand uses weaves or techniques unknown to modern channelers?",
            "How does Rand know ancient or lost weaves from the Age of Legends?",
            "What are bubbles of evil and how do they work?",
            "How are Gray Men created, and what is the cost of becoming one?",
            "What method did Nynaeve use to capture and overpower Moghedien?",
            "Was Siuan Sanche ever healed from stilling? If so, who healed her and how?",
        ],
    ),
    QuestionCategory.ARTIFACTS: CategoryInfo(
        name="Artifacts, Places & Mystical Objects",
        description="Questions about special locations, ter'angreal, sa'angreal, or legendary items.",
        focus_areas=[
            "Ter'angreal and their functions",
            "Sa'angreal and angreal",
            "Magical locations and their properties",
            "Portal Stones and Ways",
            "Ancient structures and their purposes",
            "Legendary objects and weapons",
        ],
        example_questions=[
            "What are the glass columns in Rhuidean, and what is their purpose?",
            "What is Avendesora, and where is it located?",
            "What ter'angreal or sa'angreal were used in the cleansing of saidin?",
            "What does a person feel when standing beneath Avendesora?",
            "What do Rand's visions in the glass columns reveal about the Aiel's past?",
            "Why does Avendesora make people feel peace or calm?",
        ],
    ),
    QuestionCategory.LORE: CategoryInfo(
        name="Lore & World-Building",
        description="Deep background questions about history, metaphysics, prophecies, or hidden systems.",
        focus_areas=[
            "Age of Legends history",
            "Prophecies and their interpretations",
            "Cosmology and metaphysics",
            "Cultural customs and traditions",
            "Historical events before the story",
            "Rules governing magical beings",
        ],
        example_questions=[
            "What are the bubbles of evil in The Wheel of Time?",
            "What kind of beings are the Aelfinn and Eelfinn, and what are their powers?",
            "What rules govern the Aelfinn and Eelfinn's world?",
            "How does Rand access lost knowledge and ancient weaves from the Age of Legends?",
            "What role did Moghedien play in the story after her capture?",
            "How are the Aelfinn and Eelfinn connected to Moiraine's fate?",
        ],
    ),
}


SYSTEM_INSTRUCTION = """You are an expert at creating high-quality Question & Answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.

Your task is to generate question-answer pairs that:
1. Are STRICTLY grounded in the provided text - no external knowledge
2. Focus on the specified question category
3. Require genuine understanding and reasoning to answer
4. Would be useful for evaluating how well a RAG system can retrieve and synthesize information

CRITICAL RULES:
- Every answer MUST be fully supported by the provided text
- If information is not in the text, you CANNOT include it in any QA pair
- Include a direct quote from the text that supports each answer
- Questions should require reading comprehension, not just keyword matching
- Generate as many questions as possible within the specified category
- Skip the category if no relevant content exists in the text

You are working with The Wheel of Time fantasy series by Robert Jordan."""


def get_category_prompt(category: QuestionCategory, chunk_text: str) -> str:
    """Generate a category-specific prompt for QA generation.

    Args:
        category: The question category to focus on
        chunk_text: The source text chunk

    Returns:
        Formatted prompt string
    """
    info = CATEGORY_INFO[category]

    focus_areas_str = "\n".join(f"- {area}" for area in info.focus_areas)
    examples_str = "\n".join(f"- {q}" for q in info.example_questions)

    return f"""Analyze the following text excerpt from The Wheel of Time and generate Question-Answer pairs.

## CATEGORY FOCUS: {info.name}

{info.description}

### Focus Areas for This Category:
{focus_areas_str}

### Example Questions (for style reference only - do NOT copy these):
{examples_str}

---

<text>
{chunk_text}
</text>

---

## Instructions

Generate as many high-quality QA pairs as possible that fit the **{info.name}** category.

### Quality Standards:
- Questions must be specific and unambiguous
- Answers must be detailed and explanatory (3-5 sentences minimum)
- Every answer must be directly supported by the text
- Include the exact quote from the text that supports your answer
- If the text doesn't contain content for this category, generate fewer questions

### Complexity Levels to Include:
- **Simple**: Direct fact from the text
- **Moderate**: Requires connecting 2-3 pieces of information
- **Complex**: Multi-hop reasoning across the text
- **Expert**: Deep understanding requiring synthesis of multiple elements

Generate all valid QA pairs you can find for the **{info.name}** category in this text.
If the text has rich content for this category, aim for 15-25 pairs.
If content is limited, generate only what is genuinely supported by the text."""


# Legacy prompt for backward compatibility (general generation)
QA_GENERATION_PROMPT = """Analyze the following text excerpt from The Wheel of Time and generate as many high-quality Question-Answer pairs as possible.

<text>
{chunk_text}
</text>

Generate QA pairs following these guidelines:

## Question Categories to Cover
1. **Characters & Identities**: Who people really are, names, origins, roles
2. **Major Events & Deaths**: What happened, battles, defeats, deaths
3. **Magic & Power Mechanics**: How the system works, weaves, techniques
4. **Artifacts & Places**: Special objects, locations, ter'angreal
5. **Lore & World-Building**: Deep history, metaphysics, prophecies

## Complexity Levels
- **Simple**: Direct fact from the text
- **Moderate**: Requires connecting 2-3 pieces of information
- **Complex**: Multi-hop reasoning across the text
- **Expert**: Deep understanding of multiple interconnected elements

## Quality Standards
- Questions must be specific and unambiguous
- Answers must be detailed and explanatory (not one-liners)
- Every answer must be traceable to the provided text
- Include the supporting quote from the text

Generate as many valid QA pairs as you can find in this text. Aim for 20-30 pairs if the content supports it."""


VALIDATION_PROMPT = """Review this QA pair for accuracy against the source text.

Source Text:
{chunk_text}

Question: {question}
Answer: {answer}
Evidence Quote: {evidence_quote}

Verify:
1. Is the answer completely supported by the source text?
2. Does the evidence quote actually appear in the text?
3. Is there any information in the answer not found in the text?

Respond with a JSON object:
{{
    "is_valid": true/false,
    "issues": ["list of issues if invalid"],
    "confidence": 0.0-1.0
}}"""

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
            "Character true names and aliases - who they were before vs after",
            "Character origins, backgrounds, and how they came to be who they are",
            "Character roles, titles, and positions of power",
            "Character relationships, family ties, and connections to others",
            "Character transformations, identity changes, and pivotal moments",
            "Secret identities revealed and their implications",
        ],
        example_questions=[
            "Who led the effort to create the Bore, and what was their original motivation?",
            "What was Lanfear's original name before she turned to the Shadow, and what circumstances led to her transformation?",
            "Who is the only Forsaken from the modern age rather than the Age of Legends, and how did they come to be raised?",
            "Who are the Gray Men, what are they, and what sacrifice do they make to become what they are?",
            "Who are the Aelfinn and the Eelfinn, what distinguishes them, and what are their respective powers?",
            "Who serves as Rand's connection to the Age of Legends, and through what mechanism does this occur?",
        ],
    ),
    QuestionCategory.EVENTS: CategoryInfo(
        name="Major Events & Deaths",
        description="Questions about important plot events, battles, defeats, deaths, or turning points.",
        focus_areas=[
            "Character deaths - the circumstances, who was responsible, and the aftermath",
            "Major battles - who fought, what tactics were used, and what was at stake",
            "Pivotal plot moments that changed the course of the story",
            "Defeats and victories - how they were achieved and their consequences",
            "Rescue missions and escapes - who was saved, by whom, and how",
            "Political upheavals and power shifts - what triggered them and their effects",
        ],
        example_questions=[
            "What was the method and circumstance of Egwene al'Vere's death, and what did she accomplish in her final moments?",
            "Who stilled Siuan Sanche, under what circumstances, and what were the political motivations?",
            "Who kills the giant Shadowspawn at Shayol Ghul, how is it defeated, and what made this victory possible?",
            "What were the circumstances of Lanfear's final fate, who was responsible, and how did events lead to that moment?",
            "What happened to Moghedien after she was defeated, and how did her fate unfold through the rest of the series?",
            "When and how did Mat Cauthon rescue Moiraine, what price was paid, and what rules governed this rescue?",
        ],
    ),
    QuestionCategory.MAGIC: CategoryInfo(
        name="Magic, Weaves & Power Mechanics",
        description="Questions about how the One Power works, special abilities, weaves, and techniques.",
        focus_areas=[
            "Specific weaves - how they are constructed, what elements they use, and their effects",
            "Lost or ancient techniques - what makes them different from modern knowledge",
            "Channeling abilities and limitations - what can and cannot be done",
            "Healing and stilling/gentling - the mechanics and consequences",
            "Combat techniques with the Power - strategies, defenses, and attacks",
            "Unique magical abilities and Talents - how they manifest and what they enable",
        ],
        example_questions=[
            "In what scenes does Rand use weaves unknown to modern channelers, what are these weaves, and how do witnesses react?",
            "How does Rand access ancient weaves from the Age of Legends, through what mechanism, and what are examples?",
            "What are bubbles of evil, what causes them, how do they manifest, and what incidents demonstrate their effects?",
            "How are Gray Men created, what is the cost, what do they sacrifice, and why does this make them dangerous?",
            "What method did Nynaeve use to overpower Moghedien, how did it work, and what made it possible?",
            "Was Siuan Sanche healed from stilling, who healed her, how was it possible, and what were the consequences?",
        ],
    ),
    QuestionCategory.ARTIFACTS: CategoryInfo(
        name="Artifacts, Places & Mystical Objects",
        description="Questions about special locations, ter'angreal, sa'angreal, or legendary items.",
        focus_areas=[
            "Ter'angreal - their functions, how they are activated, and their history",
            "Sa'angreal and angreal - their power levels and who has used them",
            "Magical locations - their properties, what happens there, and their significance",
            "Portal Stones and the Ways - how they work and their dangers",
            "Ancient structures - their purposes, who built them, and what secrets they hold",
            "Legendary objects and weapons - their origins, powers, and who wields them",
        ],
        example_questions=[
            "What are the glass columns in Rhuidean, what is their purpose, what visions do they show, and what do they reveal?",
            "What is Avendesora, where is it located, why does it make people feel peace, and what is its significance?",
            "What ter'angreal or sa'angreal were used in the cleansing of saidin, what role did each play?",
            "What does a person experience when passing through the glass columns, and what truth is revealed?",
            "What do Rand's visions in the glass columns reveal about the complete history of the Aiel?",
            "What is the function of the Oath Rod, how does it work, what are its effects, and what is its origin?",
        ],
    ),
    QuestionCategory.LORE: CategoryInfo(
        name="Lore & World-Building",
        description="Deep background questions about history, metaphysics, prophecies, or hidden systems.",
        focus_areas=[
            "Age of Legends history - what the world was like and how it fell",
            "Prophecies and their interpretations - what they foretold and how they were fulfilled",
            "Cosmology and metaphysics - the nature of the Pattern, the Wheel, and reality",
            "Cultural customs and traditions - why they exist and what they mean",
            "Historical events before the story - the Breaking, the Trolloc Wars, etc.",
            "Rules governing magical beings - the Finn, Shadowspawn, and others",
        ],
        example_questions=[
            "What are the bubbles of evil, why do they occur, what causes them, and what are examples of how they manifest?",
            "What kind of beings are the Aelfinn and Eelfinn, what are their powers, and what rules govern interactions with them?",
            "What rules govern the Finn's world, what happens to those who break them, and how do Mat's experiences illustrate this?",
            "How does Rand access lost knowledge from the Age of Legends, what is the mechanism, and how does this change him?",
            "What role did Moghedien play after her capture, how did she escape, and what were her actions afterward?",
            "How are the Aelfinn and Eelfinn connected to Moiraine's fate, what happened to her, and what did it take to rescue her?",
        ],
    ),
}


SYSTEM_INSTRUCTION = """You are an expert at creating comprehensive, challenging Question & Answer datasets for evaluating Retrieval-Augmented Generation (RAG) systems.

Your task is to generate question-answer pairs that:
1. Are STRICTLY grounded in the provided text - no external knowledge
2. Focus on the specified question category
3. Require DEEP understanding, multi-hop reasoning, and synthesis of information
4. Have DETAILED, EXPLANATORY answers that thoroughly address the question
5. Would challenge a RAG system to retrieve and synthesize information effectively

CRITICAL QUALITY RULES:

1. NO SIMPLE QUESTIONS:
   - NEVER ask yes/no questions like "Can X do Y?"
   - NEVER ask single-fact questions like "What is X?"
   - ALWAYS require connecting multiple pieces of information

2. MULTI-HOP REASONING REQUIRED:
   - Questions must require finding and synthesizing 2+ facts
   - Ask "How does X connect to Y?" not "What is X?"
   - Ask "What circumstances led to X and what were the consequences?" not "Did X happen?"

3. DETAILED ANSWERS (8-15 sentences):
   - Provide comprehensive explanations, not brief summaries
   - Include context, circumstances, and implications
   - Reference specific scenes, characters, or events from the text
   - Explain the "why" and "how", not just the "what"

4. RICH EVIDENCE QUOTES:
   - Include substantial quotes (2-4 sentences) that directly support the answer
   - Choose quotes that demonstrate the complexity of the topic

5. COMPLEXITY DISTRIBUTION:
   - Generate ZERO "simple" questions
   - At least 40% should be "complex" or "expert" level
   - "Moderate" questions should still require connecting multiple facts

You are working with The Wheel of Time fantasy series by Robert Jordan.

Generate as many high-quality questions as possible WITHOUT compromising quality. Let the content richness determine the quantity - quality is non-negotiable."""


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

    return f"""Analyze the following text excerpt from The Wheel of Time and generate comprehensive Question-Answer pairs.

## CATEGORY FOCUS: {info.name}

{info.description}

### Focus Areas for This Category:
{focus_areas_str}

### Example Questions (for style and complexity reference - do NOT copy these):
{examples_str}

---

## QUALITY REQUIREMENTS

### Questions MUST:
- Require multi-hop reasoning (connecting 2+ facts from the text)
- Be specific about scenes, circumstances, or mechanisms
- Ask "how", "why", "what circumstances", "what connection" - NOT simple "what is"
- Challenge a reader to synthesize information

### Questions MUST NOT:
- Be answerable with yes/no
- Be answerable with a single fact or sentence
- Be simple definitions or identifications
- Be trivial or surface-level

### Answers MUST:
- Be 8-15 sentences providing thorough explanation
- Include specific details from the text (names, places, circumstances)
- Explain context, causes, and consequences
- Reference the relevant scenes or passages

### Evidence Quotes MUST:
- Be 2-4 sentences long
- Directly support the key claims in the answer
- Be exact quotes from the provided text

### Complexity Levels:
- **Moderate** (minimum): Requires connecting 2-3 pieces of information
- **Complex**: Multi-hop reasoning, understanding cause-effect chains
- **Expert**: Deep synthesis across multiple scenes/concepts, understanding implications

DO NOT generate any "simple" complexity questions.

---

<text>
{chunk_text}
</text>

---

## GENERATION INSTRUCTIONS

Generate as many high-quality QA pairs as possible for the **{info.name}** category WITHOUT compromising quality.

- Let the content richness determine how many questions you generate
- Every question MUST require multi-hop reasoning - this is non-negotiable
- Every answer MUST be comprehensive (8-15 sentences) - do not shorten answers to generate more questions
- Quality over quantity - it is better to generate fewer excellent questions than many mediocre ones
- Cover different aspects within the category
- Vary the complexity between moderate, complex, and expert levels

The quality standards are strict. Generate as many questions as you can while meeting ALL quality requirements."""


# Legacy prompt for backward compatibility (general generation)
QA_GENERATION_PROMPT = """Analyze the following text excerpt from The Wheel of Time and generate comprehensive Question-Answer pairs.

<text>
{chunk_text}
</text>

Generate QA pairs following these strict guidelines:

## Question Categories to Cover
1. **Characters & Identities**: Who people really are, names, origins, roles
2. **Major Events & Deaths**: What happened, battles, defeats, deaths
3. **Magic & Power Mechanics**: How the system works, weaves, techniques
4. **Artifacts & Places**: Special objects, locations, ter'angreal
5. **Lore & World-Building**: Deep history, metaphysics, prophecies

## STRICT QUALITY REQUIREMENTS

### Questions MUST:
- Require multi-hop reasoning (connecting 2+ facts)
- Ask about circumstances, connections, and implications
- Be specific about scenes and mechanisms

### Questions MUST NOT:
- Be yes/no questions
- Be single-fact lookups
- Be simple "what is X" definitions

### Answers MUST:
- Be 8-15 sentences with thorough explanation
- Include specific details (names, places, scenes)
- Explain causes, effects, and implications

### Complexity Levels (NO "simple" questions):
- **Moderate**: Connects 2-3 pieces of information
- **Complex**: Multi-hop reasoning across the text
- **Expert**: Deep synthesis of multiple concepts

Generate as many high-quality QA pairs as possible WITHOUT compromising quality. Let the content determine quantity."""


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

import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

# Use the new unified LLM service
from src.services.llm import UnifiedLLMService
from src.knowledge_graph.schemas import (
    ExtractionResult,
    Ontology,
    SourceReference,
    ExtractionConfidence,
    EntityInstance,
    RelationshipInstance
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert knowledge graph extractor for the "{series_name}" series.
Your goal is to extract a comprehensive list of entities and relationships from the provided text chunk.

STRICT ADHERENCE & ADAPTIVE EVOLOUTION:
1. You must primarily use the provided Ontology Schema to categorize entities and relationships.
2. IF you encounter a significant entity or relationship that strongly clearly does NOT fit existing definitions:
   - DO NOT force it into an incorrect category.
   - DO NOT ignore it if it is important.
   - PROPOSE a schema update in the `schema_proposals` field (e.g., "new_entity_type", "new_relationship_type").
3. Use the Canonical Renaming Rules to normalize entity names.

INPUT CONTEXT:
This text is from: {book_names}
Section ID: {section_id}

TASK:
1. Identify all significant entities (Characters, Locations, Organizations, Artifacts, Events, etc.).
2. Extract detailed attributes for each entity.
3. Identify all relationships between these entities.
4. Extract evidence (quotes) for relationships.
5. Assign a CONFIDENCE score (0.0-1.0) to each extraction.
6. Identify gaps in the schema and propose updates in `schema_proposals`.

OUTPUT FORMAT:
Return a JSON object strictly matching the `ExtractionResult` Pydantic model.
"""


SERIES_CONFIG = {
    "Harry Potter": {
        "schema": "harry_potter_schema.json",
        "prefix": "harry_potter"
    },
    "A Song of Ice and Fire": {
        "schema": "a_song_of_ice_and_fire_schema.json",
        "prefix": "asoiaf"
    },
    "asoiaf": {
        "schema": "a_song_of_ice_and_fire_schema.json",
        "prefix": "asoiaf"
    },
    "The Wheel of Time": {
        "schema": "the_wheel_of_time_schema.json",
        "prefix": "wheel_of_time"
    },
    "wheel_of_time": {
        "schema": "the_wheel_of_time_schema.json",
        "prefix": "wheel_of_time"
    }
}

class EntityExtractor:
    """Extracts entities and relationships from text sections using LLMs.

    This extractor uses the unified LLM service to support multiple providers
    (Google/Gemini, Anthropic/Claude, OpenAI/GPT, OpenRouter) with configurable
    models and fallback chains.

    Attributes:
        output_dir (Path): Directory to save extracted JSON files.
        provider (str): The LLM provider to use (google, anthropic, openai, openrouter).
        model_name (Optional[str]): Specific model name to use (e.g., "gemini-3-pro-preview").
        llm_service (UnifiedLLMService): Service for LLM interactions.
    """

    def __init__(
        self,
        output_dir: str = "data/extracted_graph",
        provider: str = "google",
        model_name: Optional[str] = None
    ):
        """Initializes the EntityExtractor.

        Args:
            output_dir: Path to the output directory. Defaults to "data/extracted_graph".
                Use descriptive names like "data/extracted_graph_claude" to distinguish
                different model extractions.
            provider: The LLM provider to use. Defaults to "google" (Gemini).
                Options: "google", "anthropic", "openai", "openrouter"
            model_name: Specific model name to use. Defaults to None (uses provider default).
                Examples:
                - Google: "gemini-3-pro-preview"
                - Anthropic: "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929"
                - OpenAI: "gpt-5.2-pro"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model_name = model_name

        # Initialize the unified LLM service
        self.llm_service = UnifiedLLMService()

    def load_schema(self, series_name: str) -> Ontology:
        """Loads the pre-generated schema for the series.

        Args:
            series_name (str): The name of the series.

        Returns:
            Ontology: The loaded ontology object.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        # Look up schema filename from config or fall back to default convention
        config = SERIES_CONFIG.get(series_name)
        if config:
            schema_filename = config["schema"]
        else:
            schema_filename = f"{series_name.lower().replace(' ', '_')}_schema.json"
            
        schema_path = Path("data/schemas") / schema_filename
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found for {series_name} at {schema_path}")
        
        with open(schema_path, "r") as f:
            data = json.load(f)
        return Ontology(**data)

    def load_text_section(self, file_path: Path) -> str:
        """Reads the content of a text file.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            str: The content of the file.
        """
        with open(file_path, "r") as f:
            return f.read()

    def get_checkpoint_path(self, section_file: Path) -> Path:
        """Generates the path for the checkpoint file.

        Args:
            section_file (Path): The processed section file.

        Returns:
            Path: The path where the extraction result should be saved.
        """
        return self.output_dir / f"{section_file.stem}_extracted.json"
    
    async def extract_section(self, series_name: str, section_file: Path, schema: Ontology):
        """Extracts entities and relationships from a single text section."""
        checkpoint_path = self.get_checkpoint_path(section_file)
        
        if checkpoint_path.exists():
            logger.info("checkpoint_found_skipping", file=section_file.name)
            return

        logger.info("processing_section_start", file=section_file.name)
        
        text_content = self.load_text_section(section_file)
        
        # Load metadata
        meta_path = section_file.with_suffix(".meta.json")
        book_names = "Unknown"
        section_id = "Unknown"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                book_names = ", ".join(meta.get("books", []))
                section_id = str(meta.get("section", "Unknown"))

        # Construct the full prompt with ontology schema and text content
        schema_context = f"ONTOLOGY:\n{schema.model_dump_json(indent=2)}"
        alias, canonical = next(iter(schema.canonical_renaming_rules.items())) if schema.canonical_renaming_rules else ("Alias", "Canonical")

        # Format the extraction prompt with series-specific context
        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            series_name=series_name,
            book_names=book_names,
            section_id=section_id,
            alias_example=alias,
            canonical_example=canonical
        )

        # Combine prompt components (UnifiedLLMService expects a single prompt string)
        full_prompt = f"{extraction_prompt}\n\n{schema_context}\n\nTEXT CONTENT:\n{text_content}"

        try:
            # Use UnifiedLLMService to generate structured output
            # The new API returns a dict with 'parsed', 'usage', and 'raw' keys
            #
            # CRITICAL: Set max_tokens to maximum to avoid truncating extraction results
            # Different models have different output token limits:
            # - Claude Opus 4.5 / Sonnet 4.5: 64,000 tokens (Anthropic API limit)
            # - Gemini 3 Pro Preview: 8,192 tokens (Google API limit)
            # - GPT-5.2: 16,384 tokens (OpenAI API limit - estimated)
            #
            # Set model-specific max_tokens to ensure complete extraction without truncation
            model_max_tokens = {
                "anthropic": 64000,  # Claude Opus 4.5 / Sonnet 4.5
                "google": 64000,      # Gemini 3 Pro Preview / Gemini 2.0
                "openai": 64000,     # GPT-5.2 (verify actual limit)
            }

            max_output_tokens = model_max_tokens.get(self.provider, 8192)

            logger.info(
                "extraction_request_config",
                provider=self.provider,
                model=self.model_name,
                max_output_tokens=max_output_tokens,
                input_prompt_length=len(full_prompt)
            )

            response = await self.llm_service.generate_structured(
                prompt=full_prompt,
                schema=ExtractionResult,
                provider=self.provider,
                model=self.model_name,
                max_tokens=max_output_tokens,  # Provider-specific maximum
                temperature=0.0  # Deterministic output for consistency
            )

            # Extract the parsed result from the response dict
            result: ExtractionResult = response["parsed"]
            usage = response["usage"]

            # Attach source references to all entities and relationships
            source_ref = SourceReference(file_name=section_file.name, chunk_id=section_id)

            for entity in result.entities:
                entity.mentions.append(source_ref)
                # Ensure confidence is set (default to 1.0 if missing)
                if not entity.confidence:
                    entity.confidence = ExtractionConfidence(score=1.0)

            for rel in result.relationships:
                rel.source_ref = source_ref
                # Ensure confidence is set (default to 1.0 if missing)
                if not rel.confidence:
                    rel.confidence = ExtractionConfidence(score=1.0)

            # Persist extraction results to disk
            with open(checkpoint_path, "w") as f:
                f.write(result.model_dump_json(indent=2))

            # Log successful extraction with token usage metrics
            logger.info(
                "extraction_complete",
                file=section_file.name,
                entities=len(result.entities),
                relationships=len(result.relationships),
                tokens_used=usage.total_tokens,
                provider=usage.provider,
                model=usage.model
            )

        except Exception as e:
            # Log detailed error information for debugging
            logger.error(
                "extraction_failed",
                file=section_file.name,
                provider=self.provider,
                model=self.model_name,
                error=str(e)
            )
            raise e

    async def process_series(self, series_name: str, input_dir: Path):
        """Processes all text sections for a given series.

        Args:
            series_name (str): The name of the series to process.
            input_dir (Path): The directory containing processed text sections.
        """
        schema = self.load_schema(series_name)
        
        # Find all section files for this series
        # Naming convention: {series_name}_section_{idx}.txt
        # We need to map friendly name to file prefix
        
        config = SERIES_CONFIG.get(series_name)
        if config:
            file_prefix = config["prefix"]
        else:
            file_prefix = series_name.lower().replace(" ", "_")
        
        files = sorted([f for f in input_dir.glob(f"{file_prefix}_section_*.txt")])
        
        if not files:
            logger.warning("no_files_found_for_series", series=series_name, prefix=file_prefix)
            return

        for section_file in files:
            await self.extract_section(series_name, section_file, schema)

async def main():
    """
    Main entry point for entity extraction.

    Processes text sections from processed_books directories and extracts
    structured knowledge graphs using specified LLM providers and models.
    """

    # Define allowed models for strict enforcement
    # Model IDs verified against actual API responses
    ALLOWED_MODELS = {
        "google": ["gemini-3-pro-preview"],
        "anthropic": ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929"],
        "openai": ["gpt-5.2-pro"]
    }

    parser = argparse.ArgumentParser(
        description="Extract entities and relationships from processed books using LLMs.",
        epilog="""
Examples:
  # Extract using Gemini 3 Pro Preview (default)
  python extract_entities.py --input-dir data/processed_books --output-dir data/extracted_graph_gemini

  # Extract using Claude Opus 4.5
  python extract_entities.py --provider anthropic --model claude-opus-4-5-20251101 \\
      --input-dir data/processed_books_claude_200k --output-dir data/extracted_graph_claude_opus

  # Extract using Claude Sonnet 4.5
  python extract_entities.py --provider anthropic --model claude-sonnet-4-5-20250929 \\
      --input-dir data/processed_books_claude_200k --output-dir data/extracted_graph_claude_sonnet

  # Extract using GPT-5.2
  python extract_entities.py --provider openai --model gpt-5.2-pro \\
      --input-dir data/processed_books_gpt_400k --output-dir data/extracted_graph_gpt

  # Process only Harry Potter series
  python extract_entities.py --series "Harry Potter" --provider anthropic --model claude-opus-4-5-20251101
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--series",
        type=str,
        help='Specific series to process. Options: "Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"'
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed_books",
        help="Directory containing processed text sections. "
             "Use model-specific directories (e.g., data/processed_books_claude_200k) "
             "to match the chunking strategy. Default: data/processed_books"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extracted_graph",
        help="Directory to save extracted knowledge graphs. "
             "Use descriptive names (e.g., data/extracted_graph_claude_opus) to distinguish "
             "extractions from different models. Default: data/extracted_graph"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "anthropic", "openai"],
        help="LLM Provider to use. Options: google, anthropic, openai. Default: google"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to use. If not specified, uses provider default. "
             "Allowed models: "
             "Google: gemini-3-pro-preview | "
             "Anthropic: claude-opus-4-5-20251101, claude-sonnet-4-5-20250929 | "
             "OpenAI: gpt-5.2-pro"
    )

    args = parser.parse_args()

    # Validate model selection against allowed models
    if args.model:
        if args.provider not in ALLOWED_MODELS:
            parser.error(f"Invalid provider: {args.provider}. Choose from: {list(ALLOWED_MODELS.keys())}")

        if args.model not in ALLOWED_MODELS[args.provider]:
            parser.error(
                f"Model '{args.model}' is not allowed for provider '{args.provider}'. "
                f"Allowed models: {ALLOWED_MODELS[args.provider]}"
            )

    # Log configuration for transparency
    logger.info(
        "entity_extractor_initialized",
        provider=args.provider,
        model=args.model or f"{args.provider}_default",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        series_filter=args.series or "all"
    )

    # Initialize extractor with configuration
    extractor = EntityExtractor(
        output_dir=args.output_dir,
        provider=args.provider,
        model_name=args.model
    )

    input_dir = Path(args.input_dir)

    # Validate input directory exists
    if not input_dir.exists():
        logger.error("input_directory_not_found", path=str(input_dir))
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Define series to process
    series_list = ["Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"]
    if args.series:
        series_list = [args.series]

    # Process each series
    for series_name in series_list:
        logger.info(
            "starting_extraction_series",
            series=series_name,
            provider=args.provider,
            model=args.model or "default"
        )
        await extractor.process_series(series_name, input_dir)

    # Log completion with usage summary
    usage_summary = extractor.llm_service.get_usage_summary()
    logger.info(
        "entity_extraction_complete",
        total_tokens=usage_summary["total_tokens"],
        total_cost_usd=usage_summary["total_cost_usd"],
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    asyncio.run(main())

"""
Entity and relationship extraction from text sections using Google Gemini.

This module extracts structured knowledge graphs from processed book text sections
using Google's Gemini models via the unified LLM service.
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional

import structlog

from src.services.llm import UnifiedLLMService
from src.knowledge_graph.schemas import (
    ExtractionResult,
    Ontology,
    SourceReference,
    ExtractionConfidence,
)


logger = structlog.get_logger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert knowledge graph extractor for the "{series_name}" series.
Your goal is to extract a comprehensive list of entities and relationships from the provided text chunk.

STRICT ADHERENCE & ADAPTIVE EVOLUTION:
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
        "prefix": "harry_potter",
    },
    "A Song of Ice and Fire": {
        "schema": "a_song_of_ice_and_fire_schema.json",
        "prefix": "asoiaf",
    },
    "asoiaf": {
        "schema": "a_song_of_ice_and_fire_schema.json",
        "prefix": "asoiaf",
    },
    "The Wheel of Time": {
        "schema": "the_wheel_of_time_schema.json",
        "prefix": "wheel_of_time",
    },
    "wheel_of_time": {
        "schema": "the_wheel_of_time_schema.json",
        "prefix": "wheel_of_time",
    },
}

# Allowed models for entity extraction (Google-only)
ALLOWED_MODELS = [
    "gemini-3-pro-preview",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]


class EntityExtractor:
    """Extracts entities and relationships from text sections using Google Gemini.

    This extractor uses the unified LLM service to process text sections and extract
    structured knowledge graphs with entities, relationships, and attributes.

    Attributes:
        output_dir: Directory to save extracted JSON files.
        model_name: Specific Gemini model name to use.
        llm_service: Service for LLM interactions.
    """

    def __init__(
        self,
        output_dir: str = "data/extracted_graph",
        model_name: str = "gemini-3-pro-preview",
    ):
        """Initializes the EntityExtractor.

        Args:
            output_dir: Path to the output directory. Defaults to "data/extracted_graph".
            model_name: Gemini model name to use. Defaults to "gemini-3-pro-preview".
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Initialize the unified LLM service (Google-only)
        self.llm_service = UnifiedLLMService()

    def load_schema(self, series_name: str) -> Ontology:
        """Loads the pre-generated schema for the series.

        Args:
            series_name: The name of the series.

        Returns:
            The loaded ontology object.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        config = SERIES_CONFIG.get(series_name)
        if config:
            schema_filename = config["schema"]
        else:
            schema_filename = f"{series_name.lower().replace(' ', '_')}_schema.json"

        schema_path = Path("data/schemas") / schema_filename
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema not found for {series_name} at {schema_path}"
            )

        with open(schema_path, "r") as f:
            data = json.load(f)
        return Ontology(**data)

    def load_text_section(self, file_path: Path) -> str:
        """Reads the content of a text file.

        Args:
            file_path: Path to the text file.

        Returns:
            The content of the file.
        """
        with open(file_path, "r") as f:
            return f.read()

    def get_checkpoint_path(self, section_file: Path) -> Path:
        """Generates the path for the checkpoint file.

        Args:
            section_file: The processed section file.

        Returns:
            The path where the extraction result should be saved.
        """
        return self.output_dir / f"{section_file.stem}_extracted.json"

    async def extract_section(
        self, series_name: str, section_file: Path, schema: Ontology
    ) -> None:
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

        # Format the extraction prompt with series-specific context
        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            series_name=series_name,
            book_names=book_names,
            section_id=section_id,
        )

        # Combine prompt components
        full_prompt = (
            f"{extraction_prompt}\n\n{schema_context}\n\nTEXT CONTENT:\n{text_content}"
        )

        try:
            # Use Google's maximum output token limit
            max_output_tokens = 64000

            logger.info(
                "extraction_request_config",
                model=self.model_name,
                max_output_tokens=max_output_tokens,
                input_prompt_length=len(full_prompt),
            )

            response = await self.llm_service.generate_structured(
                prompt=full_prompt,
                schema=ExtractionResult,
                model=self.model_name,
                max_tokens=max_output_tokens,
                temperature=0.0,  # Deterministic output for consistency
            )

            # Extract the parsed result from the response dict
            result: ExtractionResult = response["parsed"]
            usage = response["usage"]

            # Attach source references to all entities and relationships
            source_ref = SourceReference(
                file_name=section_file.name, chunk_id=section_id
            )

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
                model=usage.model,
            )

        except Exception as e:
            logger.error(
                "extraction_failed",
                file=section_file.name,
                model=self.model_name,
                error=str(e),
            )
            raise e

    async def process_series(self, series_name: str, input_dir: Path) -> None:
        """Processes all text sections for a given series.

        Args:
            series_name: The name of the series to process.
            input_dir: The directory containing processed text sections.
        """
        schema = self.load_schema(series_name)

        # Find all section files for this series
        config = SERIES_CONFIG.get(series_name)
        if config:
            file_prefix = config["prefix"]
        else:
            file_prefix = series_name.lower().replace(" ", "_")

        files = sorted([f for f in input_dir.glob(f"{file_prefix}_section_*.txt")])

        if not files:
            logger.warning(
                "no_files_found_for_series", series=series_name, prefix=file_prefix
            )
            return

        for section_file in files:
            await self.extract_section(series_name, section_file, schema)


async def main() -> None:
    """Main entry point for entity extraction.

    Processes text sections from processed_books directories and extracts
    structured knowledge graphs using Google Gemini models.
    """
    parser = argparse.ArgumentParser(
        description="Extract entities and relationships from processed books using Google Gemini.",
        epilog="""
Examples:
  # Extract using Gemini 3 Pro Preview (default)
  uv run python -m src.knowledge_graph.extract_entities \\
      --input-dir data/processed_books --output-dir data/extracted_graph_gemini

  # Extract using Gemini 2.0 Flash (faster, cheaper)
  uv run python -m src.knowledge_graph.extract_entities \\
      --model gemini-2.0-flash --input-dir data/processed_books

  # Process only Harry Potter series
  uv run python -m src.knowledge_graph.extract_entities \\
      --series "Harry Potter" --model gemini-3-pro-preview
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--series",
        type=str,
        help='Specific series to process. Options: "Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"',
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed_books",
        help="Directory containing processed text sections. Default: data/processed_books",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extracted_graph",
        help="Directory to save extracted knowledge graphs. Default: data/extracted_graph",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        choices=ALLOWED_MODELS,
        help=f"Gemini model to use. Options: {', '.join(ALLOWED_MODELS)}. Default: gemini-3-pro-preview",
    )

    args = parser.parse_args()

    # Log configuration
    logger.info(
        "entity_extractor_initialized",
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        series_filter=args.series or "all",
    )

    # Initialize extractor
    extractor = EntityExtractor(
        output_dir=args.output_dir,
        model_name=args.model,
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
            model=args.model,
        )
        await extractor.process_series(series_name, input_dir)

    # Log completion with usage summary
    usage_summary = extractor.llm_service.get_usage_summary()
    logger.info(
        "entity_extraction_complete",
        total_tokens=usage_summary["total_tokens"],
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())

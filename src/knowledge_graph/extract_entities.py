import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from src.services.llm_service import LLMService
from src.knowledge_graph.schemas import (
    ExtractionResult, 
    Ontology, 
    SourceReference, 
    ExtractionConfidence
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """
You are an expert knowledge graph extractor for the "{series_name}" series.
Your goal is to extract a comprehensive list of entities and relationships from the provided text chunk.

STRICT ADHERENCE TO SCHEMA:
You must use the provided Ontology Schema to categorize entities and relationships.
- Only use Entity Types defined in the schema.
- Only use Relationship Types defined in the schema.
- Use the Canonical Renaming Rules to normalize entity names (e.g., if text says "{alias_example}", output "{canonical_example}").

INPUT CONTEXT:
This text is from: {book_names}
Section ID: {section_id}

TASK:
1. Identify all significant entities (Characters, Locations, Organizations, Artifacts, Events, etc.).
2. Extract detailed attributes for each entity.
3. Identify all relationships between these entities.
4. Extract evidence (quotes) for relationships.
5. assign a CONFIDENCE score (0.0-1.0) to each extraction based on how explicitly it is supported by the text.

OUTPUT FORMAT:
Return a JSON object strictly matching the `ExtractionResult` Pydantic model.
"""

class EntityExtractor:
    """Extracts entities and relationships from text sections using LLMs.

    Attributes:
        output_dir (Path): Directory to save extracted JSON files.
        llm_service (LLMService): Service for LLM interactions.
    """

    def __init__(self, output_dir: str = "data/extracted_graph"):
        """Initializes the EntityExtractor.

        Args:
            output_dir (str): Path to the output directory. Defaults to "data/extracted_graph".
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_service = LLMService()

    def load_schema(self, series_name: str) -> Ontology:
        """Loads the pre-generated schema for the series.

        Args:
            series_name (str): The name of the series.

        Returns:
            Ontology: The loaded ontology object.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        schema_path = Path("data/schemas") / f"{series_name.lower().replace(' ', '_')}_schema.json"
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
        """Extracts entities and relationships from a single text section.

        This method:
        1. Checks for existing checkpoints to skip processing.
        2. Loads text and metadata.
        3. Constructs the LLM prompt with schema context.
        4. Calls the LLM to generate structured output.
        5. Post-processes the result to add source references.
        6. Saves the result to disk.

        Args:
            series_name (str): The name of the series.
            section_file (Path): The path to the text section file.
            schema (Ontology): The ontology definition for the series.

        Raises:
            Exception: If extraction fails, it logs the error and re-raises/continues based on policy.
        """
        checkpoint_path = self.get_checkpoint_path(section_file)
        
        if checkpoint_path.exists():
            logger.info("checkpoint_found_skipping", file=section_file.name)
            return

        logger.info("processing_section_start", file=section_file.name)
        
        # Load text and metadata
        text_content = self.load_text_section(section_file)
        
        # Load metadata if exists
        meta_path = section_file.with_suffix(".meta.json")
        book_names = "Unknown"
        section_id = "Unknown"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                book_names = ", ".join(meta.get("books", []))
                section_id = str(meta.get("section", "Unknown"))

        # Pick an alias example from schema for the prompt
        alias, canonical = next(iter(schema.canonical_renaming_rules.items())) if schema.canonical_renaming_rules else ("Alias", "Canonical")

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            series_name=series_name,
            book_names=book_names,
            section_id=section_id,
            alias_example=alias,
            canonical_example=canonical
        )
        
        # We pass the Schema definitions as context to the model 
        schema_context = f"ONTOLOGY:\n{schema.model_dump_json(indent=2)}"

        try:
            # Call LLM with strict schema enforcement
            result: ExtractionResult = await self.llm_service.generate_structured_response(
                prompt=prompt,
                schema=ExtractionResult,
                context=f"{schema_context}\n\nTEXT CONTENT:\n{text_content}",
                provider="google"
            )
            
            # Post-processing: Add Source References and Defaults
            source_ref = SourceReference(file_name=section_file.name, chunk_id=section_id)
            
            for entity in result.entities:
                # Add source ref if not present (though prompt doesn't key it, we add it here)
                entity.mentions.append(source_ref)
                
                # Default confidence if model missed it
                if not entity.confidence:
                    entity.confidence = ExtractionConfidence(score=1.0, needs_review=False)
            
            for rel in result.relationships:
                rel.source_ref = source_ref
                 # Default confidence if model missed it
                if not rel.confidence:
                     rel.confidence = ExtractionConfidence(score=1.0, needs_review=False)

            # Save result (Checkpoint)
            with open(checkpoint_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
                
            logger.info("extraction_complete", file=section_file.name, entities=len(result.entities), relationships=len(result.relationships))
            
        except Exception as e:
            logger.error("extraction_failed", file=section_file.name, error=str(e))
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
        # Harry Potter -> harry_potter
        file_prefix = series_name.lower().replace(" ", "_")
        
        files = sorted([f for f in input_dir.glob(f"{file_prefix}_section_*.txt")])
        
        if not files:
            logger.warning("no_files_found_for_series", series=series_name, prefix=file_prefix)
            return

        for section_file in files:
            await self.extract_section(series_name, section_file, schema)

async def main():
    parser = argparse.ArgumentParser(description="Extract entities and relationships from processed books.")
    parser.add_argument("--series", type=str, help="Specific series to process")
    parser.add_argument("--input-dir", type=str, default="data/processed_books", help="Directory containing text sections")
    parser.add_argument("--output-dir", type=str, default="data/extracted_graph", help="Directory to save extracted graphs")
    args = parser.parse_args()

    extractor = EntityExtractor(output_dir=args.output_dir)
    input_dir = Path(args.input_dir)

    series_list = ["Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"]
    if args.series:
        series_list = [args.series]

    for series_name in series_list:
        logger.info("starting_extraction_series", series=series_name)
        await extractor.process_series(series_name, input_dir)

if __name__ == "__main__":
    asyncio.run(main())

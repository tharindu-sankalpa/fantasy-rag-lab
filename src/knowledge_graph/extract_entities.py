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

    Attributes:
        output_dir (Path): Directory to save extracted JSON files.
        llm_service (LLMService): Service for LLM interactions.
    """


    def __init__(self, output_dir: str = "data/extracted_graph", provider: str = "google", model_name: str = None):
        """Initializes the EntityExtractor.

        Args:
            output_dir (str): Path to the output directory. Defaults to "data/extracted_graph".
            provider (str): The LLM provider to use. Defaults to "google".
            model_name (str): Specific model name to use. Defaults to None (provider default).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.model_name = model_name
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

        schema_context = f"ONTOLOGY:\n{schema.model_dump_json(indent=2)}"
        alias, canonical = next(iter(schema.canonical_renaming_rules.items())) if schema.canonical_renaming_rules else ("Alias", "Canonical")

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            series_name=series_name,
            book_names=book_names,
            section_id=section_id,
            alias_example=alias,
            canonical_example=canonical
        )
        
        try:
            # Use LLMService with failover
            result: ExtractionResult = await self.llm_service.generate_structured_response(
                prompt=prompt,
                schema=ExtractionResult,
                context=f"{schema_context}\n\nTEXT CONTENT:\n{text_content}",
                provider=self.provider,
                model_name=self.model_name
            )
            
            # Attach source ref immediately
            source_ref = SourceReference(file_name=section_file.name, chunk_id=section_id)
            for entity in result.entities:
                entity.mentions.append(source_ref)
                if not entity.confidence: 
                    entity.confidence = ExtractionConfidence(score=1.0)
            
            for rel in result.relationships:
                rel.source_ref = source_ref
                if not rel.confidence: 
                    rel.confidence = ExtractionConfidence(score=1.0)
            
            # Persist Results
            with open(checkpoint_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
                
            logger.info("extraction_complete", 
                        file=section_file.name, 
                        entities=len(result.entities), 
                        relationships=len(result.relationships))
            
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
    parser = argparse.ArgumentParser(description="Extract entities and relationships from processed books.")
    parser.add_argument("--series", type=str, help="Specific series to process")
    parser.add_argument("--input-dir", type=str, default="data/processed_books", help="Directory containing text sections")
    parser.add_argument("--output-dir", type=str, default="data/extracted_graph", help="Directory to save extracted graphs")
    parser.add_argument("--provider", type=str, default="google", help="LLM Provider (google, openai, openrouter, anthropic)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g. gpt-4o, claude-3-opus)")

    args = parser.parse_args()

    extractor = EntityExtractor(output_dir=args.output_dir, provider=args.provider, model_name=args.model)
    input_dir = Path(args.input_dir)

    series_list = ["Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"]
    if args.series:
        series_list = [args.series]

    for series_name in series_list:
        logger.info("starting_extraction_series", series=series_name)
        await extractor.process_series(series_name, input_dir)

if __name__ == "__main__":
    asyncio.run(main())

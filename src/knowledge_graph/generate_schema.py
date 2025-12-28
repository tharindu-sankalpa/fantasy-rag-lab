import asyncio
import json
import argparse
from pathlib import Path
from typing import Dict

from src.services.llm_service import LLMService
from src.knowledge_graph.schemas import Ontology
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_PROMPT_TEMPLATE = """
You are an expert fantasy literature ontologist and knowledge graph architect.
Your task is to design a comprehensive ontology for the book series: "{series_name}".

The ontology must capture the full depth of the narrative, enabling a knowledge graph that can answer complex questions about:
1. Relationships (social, political, magical, familial)
2. Character evolution (aliases, titles, roles)
3. Locations (geography, magical properties)
4. Events (battles, ceremonies, timeline)

Requirements:
- Define 8-15 distinct Entity Types (e.g., Character, Location, Organization, Artifact, Spell, Event, CulturalGroup, Creature).
- Define 10-15 distinct Relationship Types (e.g., parent_of, enemy_of, member_of, located_in, killed_by, wields).
- Provide a list of strictly Canonical Renaming Rules for the most ambiguous entities (e.g., "The Boy Who Lived" -> "Harry Potter").

Your output must strictly adhere to the provided schema.
"""

# Sample text to provide context (optional, but helpful if we want to confirm style)
# For schema generation, general knowledge is usually sufficient for these famous series, 
# but we can feed it a summary if needed. 
# We will rely on the model's internal knowledge for the schema structure as these are famous series.

async def generate_schema_for_series(series_name: str, output_dir: Path):
    """Generates and saves the ontology for a specific series.

    Uses the LLM to design a comprehensive ontology including entity types,
    relationship types, and renaming rules suitable for a knowledge graph.

    Args:
        series_name (str): The name of the book series (e.g., "Harry Potter").
        output_dir (Path): The directory where the generated JSON schema will be saved.

    Raises:
        Exception: If the LLM generation fails or the file cannot be written.
    """
    llm_service = LLMService()
    
    logger.info("generating_schema_start", series=series_name)
    
    try:
        # Prompt the LLM to generate the ontology.
        # We use 'generate_structured_response' which forces the LLM to output valid JSON
        # adhering to the 'Ontology' Pydantic model defined in schemas.py.
        # This ensures type safety and that we get exactly the fields we need (entity_types, relationship_types, etc.).
        ontology: Ontology = await llm_service.generate_structured_response(
            prompt=SCHEMA_PROMPT_TEMPLATE.format(series_name=series_name),
            schema=Ontology, # Pass the Pydantic class to define the expected JSON structure
            provider="google", # Using Gemini for its strong long-context and reasoning capabilities
            context=f"Target Series: {series_name}"
        )
        
        # Save the validated ontology to a JSON file.
        # This file acts as the configuration/source-of-truth for the extraction phase.
        filename = f"{series_name.lower().replace(' ', '_')}_schema.json"
        output_path = output_dir / filename
        
        with open(output_path, "w") as f:
            f.write(ontology.model_dump_json(indent=2))
            
        logger.info("schema_saved", path=str(output_path))
        
    except Exception as e:
        logger.error("schema_generation_failed", series=series_name, error=str(e))
        raise e

async def main():
    """Main execution entry point.
    
    Handles command line arguments to allow generating schemas for all configured series
    or targeting a specific one for testing/updates.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, help="Specific series to process (optional)")
    parser.add_argument("--output-dir", type=str, default="data/schemas", help="Output directory for schemas")
    args = parser.parse_args()
    
    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default list of series to support in the platform
    series_list = [
        "Harry Potter",
        "A Song of Ice and Fire",
        "The Wheel of Time"
    ]
    
    # Logic to handle user overrides (e.g., python generate_schema.py --series "Harry Potter")
    if args.series:
        # Normalize input to match our list convention
        series_list = [s for s in series_list if s.lower() == args.series.lower().replace("_", " ")]
        if not series_list:
             # Fallback: Allow arbitrary new series if not in our default list
             series_list = [args.series]

    for series in series_list:
        await generate_schema_for_series(series, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

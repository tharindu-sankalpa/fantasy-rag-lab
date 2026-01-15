# Dependencies:
# pip install anthropic structlog pydantic python-dotenv

"""
Batch Entity Extraction using Anthropic Claude Message Batches API.

This script provides batch processing capabilities for entity extraction,
offering 50% cost reduction compared to real-time API calls. It is designed
to work alongside the existing extract_entities.py for real-time extraction.

Key Features:
- Submit extraction requests as batches (up to 10,000 per batch)
- Poll for completion with configurable intervals
- Resume interrupted batch jobs
- Process results into the same format as real-time extraction
- Full compatibility with downstream knowledge graph consumers

Workflow Modes:
1. SUBMIT: Create a new batch job from input sections
2. STATUS: Check the status of an existing batch job
3. RETRIEVE: Download and process results from a completed batch
4. FULL: Submit, poll, and retrieve in one operation (default)

Cost Optimization:
- Batch API: 50% discount on input and output tokens
- Processing time: Up to 24 hours (often much faster)
- Best for: Non-time-sensitive bulk extraction workloads

Example Usage:
    # Full workflow (submit, poll, retrieve)
    python extract_entities_batch.py --model claude-opus-4-5-20251101 \\
        --input-dir data/processed_books_claude_200k \\
        --output-dir data/extracted_graph_claude_opus_batch \\
        --series "Harry Potter"

    # Submit only (returns batch_id for later retrieval)
    python extract_entities_batch.py --mode submit --model claude-opus-4-5-20251101 \\
        --input-dir data/processed_books_claude_200k \\
        --output-dir data/extracted_graph_claude_opus_batch

    # Check status of existing batch
    python extract_entities_batch.py --mode status --batch-id msgbatch_xxx

    # Retrieve results from completed batch
    python extract_entities_batch.py --mode retrieve --batch-id msgbatch_xxx
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.services.llm import (
    AnthropicBatchProvider,
    BatchRequestItem,
    BatchRequestParams,
    BatchProcessingStatus,
    build_extraction_batch_request,
)
from src.knowledge_graph.schemas import (
    ExtractionResult,
    Ontology,
    SourceReference,
    ExtractionConfidence,
)
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Extraction prompt template (same as real-time extraction for consistency)
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

# Series configuration (same as real-time extraction)
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

# Allowed models for batch extraction (Anthropic only)
ALLOWED_MODELS = {
    "anthropic": ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929"]
}


class BatchEntityExtractor:
    """
    Batch entity extractor using Anthropic's Message Batches API.
    
    This class orchestrates the batch extraction workflow:
    1. Prepare extraction requests from input sections
    2. Submit batch to Anthropic API
    3. Poll for completion
    4. Process results into extraction files
    
    The output format is identical to real-time extraction, ensuring
    seamless integration with downstream knowledge graph consumers.
    
    Attributes:
        output_dir: Directory for extraction results
        model_name: Claude model identifier
        batch_provider: AnthropicBatchProvider instance
        max_tokens: Maximum output tokens per request
    """

    def __init__(
        self,
        output_dir: str = "data/extracted_graph_batch",
        model_name: str = "claude-opus-4-5-20251101",
        jobs_dir: str = "data/batch_jobs",
        max_tokens: int = 64000,
    ):
        """
        Initialize the batch entity extractor.
        
        Args:
            output_dir: Directory to save extracted JSON files
            model_name: Claude model to use for extraction
            jobs_dir: Directory for batch job metadata persistence
            max_tokens: Maximum output tokens per request (default: 64000)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Initialize batch provider with API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.batch_provider = AnthropicBatchProvider(
            api_key=api_key,
            jobs_dir=jobs_dir
        )
        
        logger.info(
            "batch_extractor_initialized",
            output_dir=str(self.output_dir),
            model=model_name,
            max_tokens=max_tokens
        )

    def load_schema(self, series_name: str) -> Ontology:
        """
        Load the ontology schema for a series.
        
        Args:
            series_name: Name of the book series
            
        Returns:
            Ontology object with entity types and relationship definitions
        """
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

    def get_section_files(self, series_name: str, input_dir: Path) -> List[Path]:
        """
        Get all section files for a series that need processing.
        
        Skips sections that already have extraction results (checkpointing).
        
        Args:
            series_name: Name of the book series
            input_dir: Directory containing processed text sections
            
        Returns:
            List of section file paths to process
        """
        config = SERIES_CONFIG.get(series_name)
        if config:
            file_prefix = config["prefix"]
        else:
            file_prefix = series_name.lower().replace(" ", "_")
        
        # Find all section files
        all_files = sorted([f for f in input_dir.glob(f"{file_prefix}_section_*.txt")])
        
        # Filter out already processed sections (checkpointing)
        files_to_process = []
        for section_file in all_files:
            checkpoint_path = self.output_dir / f"{section_file.stem}_extracted.json"
            if not checkpoint_path.exists():
                files_to_process.append(section_file)
            else:
                logger.info("checkpoint_found_skipping", file=section_file.name)
        
        return files_to_process

    def build_extraction_prompt(
        self,
        series_name: str,
        section_file: Path,
        schema: Ontology
    ) -> str:
        """
        Build the full extraction prompt for a section.
        
        Args:
            series_name: Name of the book series
            section_file: Path to the text section file
            schema: Ontology schema for the series
            
        Returns:
            Complete prompt string for extraction
        """
        # Load text content
        with open(section_file, "r") as f:
            text_content = f.read()
        
        # Load metadata
        meta_path = section_file.with_suffix(".meta.json")
        book_names = "Unknown"
        section_id = "Unknown"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                book_names = ", ".join(meta.get("books", []))
                section_id = str(meta.get("section", "Unknown"))
        
        # Build schema context
        schema_context = f"ONTOLOGY:\n{schema.model_dump_json(indent=2)}"
        alias, canonical = next(
            iter(schema.canonical_renaming_rules.items())
        ) if schema.canonical_renaming_rules else ("Alias", "Canonical")
        
        # Format extraction prompt
        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            series_name=series_name,
            book_names=book_names,
            section_id=section_id,
            alias_example=alias,
            canonical_example=canonical
        )
        
        # Combine all components
        full_prompt = f"{extraction_prompt}\n\n{schema_context}\n\nTEXT CONTENT:\n{text_content}"
        
        return full_prompt

    async def submit_batch(
        self,
        series_name: str,
        input_dir: Path,
    ) -> str:
        """
        Submit a batch job for entity extraction.
        
        Prepares extraction requests for all unprocessed sections and
        submits them as a single batch to the Anthropic API.
        
        Args:
            series_name: Name of the book series to process
            input_dir: Directory containing processed text sections
            
        Returns:
            Batch ID for tracking and result retrieval
        """
        log = logger.bind(
            operation="submit_batch",
            series=series_name,
            model=self.model_name
        )
        
        # Load schema for this series
        schema = self.load_schema(series_name)
        
        # Get files to process (respecting checkpoints)
        files_to_process = self.get_section_files(series_name, input_dir)
        
        if not files_to_process:
            log.info("no_files_to_process", message="All sections already extracted")
            return ""
        
        log.info("preparing_batch_requests", num_files=len(files_to_process))
        
        # Build batch requests
        batch_requests = []
        for section_file in files_to_process:
            # Build prompt for this section
            prompt = self.build_extraction_prompt(series_name, section_file, schema)
            
            # Create batch request item
            # Use section file stem as custom_id for result correlation
            request = build_extraction_batch_request(
                custom_id=section_file.stem,
                prompt=prompt,
                model=self.model_name,
                schema=ExtractionResult,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            batch_requests.append(request)
            
            log.debug(
                "request_prepared",
                custom_id=section_file.stem,
                prompt_length=len(prompt)
            )
        
        # Submit batch to Anthropic API
        log.info("submitting_batch", num_requests=len(batch_requests))
        
        job = await self.batch_provider.create_batch(
            requests=batch_requests,
            output_dir=str(self.output_dir),
            model=self.model_name,
            series_name=series_name,
        )
        
        log.info(
            "batch_submitted",
            batch_id=job.id,
            status=job.processing_status,
            expires_at=job.expires_at.isoformat()
        )
        
        return job.id

    async def check_status(self, batch_id: str) -> dict:
        """
        Check the status of a batch job.
        
        Args:
            batch_id: The batch ID to check
            
        Returns:
            Dictionary with status information
        """
        job = await self.batch_provider.get_batch_status(batch_id)
        
        return {
            "batch_id": job.id,
            "status": job.processing_status,
            "succeeded": job.request_counts.succeeded,
            "errored": job.request_counts.errored,
            "processing": job.request_counts.processing,
            "created_at": job.created_at.isoformat(),
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        }

    async def retrieve_results(self, batch_id: str) -> dict:
        """
        Retrieve and process results from a completed batch.
        
        Downloads results from the Anthropic API, validates against the
        ExtractionResult schema, and saves to output files. Also applies
        post-processing (source references, confidence defaults) to match
        the real-time extraction output format.
        
        Args:
            batch_id: The batch ID to retrieve results for
            
        Returns:
            Dictionary with processing statistics
        """
        log = logger.bind(operation="retrieve_results", batch_id=batch_id)
        
        # Load job metadata for series info
        job = await self.batch_provider._load_job(batch_id)
        if not job:
            raise ValueError(f"Job metadata not found for batch {batch_id}")
        
        # Retrieve and process results
        result = await self.batch_provider.process_results_to_files(
            batch_id=batch_id,
            schema=ExtractionResult,
        )
        
        # Post-process each result file to add source references
        # This matches the real-time extraction behavior
        for output_path in result["files_written"]:
            await self._post_process_result(output_path)
        
        log.info(
            "results_retrieved",
            processed=result["processed"],
            failed=result["failed"],
            total_tokens=result["usage"].total_tokens
        )
        
        return {
            "processed": result["processed"],
            "failed": result["failed"],
            "total_tokens": result["usage"].total_tokens,
            "files_written": result["files_written"],
        }

    async def _post_process_result(self, output_path: str) -> None:
        """
        Post-process an extraction result file.
        
        Adds source references and default confidence scores to match
        the real-time extraction output format.
        
        Args:
            output_path: Path to the extraction result JSON file
        """
        output_file = Path(output_path)
        
        # Load existing result
        with open(output_file, "r") as f:
            data = json.load(f)
        
        # Parse as ExtractionResult
        result = ExtractionResult.model_validate(data)
        
        # Extract section info from filename (e.g., "harry_potter_section_01_extracted.json")
        stem = output_file.stem.replace("_extracted", "")
        section_file_name = f"{stem}.txt"
        
        # Try to get section_id from metadata file
        section_id = "Unknown"
        meta_path = output_file.parent.parent / "processed_books_claude_200k" / f"{stem}.meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                section_id = str(meta.get("section", "Unknown"))
        
        # Create source reference
        source_ref = SourceReference(file_name=section_file_name, chunk_id=section_id)
        
        # Apply to entities
        for entity in result.entities:
            if not entity.mentions:
                entity.mentions = []
            entity.mentions.append(source_ref)
            if not entity.confidence:
                entity.confidence = ExtractionConfidence(score=1.0)
        
        # Apply to relationships
        for rel in result.relationships:
            rel.source_ref = source_ref
            if not rel.confidence:
                rel.confidence = ExtractionConfidence(score=1.0)
        
        # Write updated result
        with open(output_file, "w") as f:
            f.write(result.model_dump_json(indent=2))

    async def run_full_workflow(
        self,
        series_name: str,
        input_dir: Path,
        poll_interval_seconds: int = 60,
        max_wait_hours: float = 24.0,
    ) -> dict:
        """
        Run the complete batch extraction workflow.
        
        Submits a batch, polls until completion, and retrieves results.
        This is the default mode for batch extraction.
        
        Args:
            series_name: Name of the book series to process
            input_dir: Directory containing processed text sections
            poll_interval_seconds: Seconds between status checks
            max_wait_hours: Maximum hours to wait for completion
            
        Returns:
            Dictionary with workflow results
        """
        log = logger.bind(
            operation="full_workflow",
            series=series_name,
            model=self.model_name
        )
        
        # Step 1: Submit batch
        batch_id = await self.submit_batch(series_name, input_dir)
        
        if not batch_id:
            return {"status": "no_work", "message": "All sections already extracted"}
        
        # Step 2: Poll until complete
        log.info("polling_for_completion", batch_id=batch_id)
        
        job = await self.batch_provider.poll_until_complete(
            batch_id=batch_id,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_hours=max_wait_hours,
        )
        
        # Step 3: Retrieve results
        if job.processing_status == BatchProcessingStatus.ENDED:
            results = await self.retrieve_results(batch_id)
            return {
                "status": "completed",
                "batch_id": batch_id,
                **results
            }
        else:
            return {
                "status": job.processing_status,
                "batch_id": batch_id,
                "message": f"Batch ended with status: {job.processing_status}"
            }


async def main():
    """
    Main entry point for batch entity extraction.
    
    Supports multiple workflow modes:
    - full: Submit, poll, and retrieve (default)
    - submit: Create batch job only
    - status: Check batch status
    - retrieve: Download and process results
    """
    parser = argparse.ArgumentParser(
        description="Batch entity extraction using Anthropic Claude Message Batches API (50% cost reduction).",
        epilog="""
Examples:
  # Full workflow (submit, poll, retrieve) - DEFAULT
  python extract_entities_batch.py --model claude-opus-4-5-20251101 \\
      --input-dir data/processed_books_claude_200k \\
      --output-dir data/extracted_graph_claude_opus_batch \\
      --series "Harry Potter"

  # Submit batch only (returns batch_id)
  python extract_entities_batch.py --mode submit --model claude-opus-4-5-20251101 \\
      --input-dir data/processed_books_claude_200k \\
      --output-dir data/extracted_graph_claude_opus_batch

  # Check status of existing batch
  python extract_entities_batch.py --mode status --batch-id msgbatch_xxx

  # Retrieve results from completed batch
  python extract_entities_batch.py --mode retrieve --batch-id msgbatch_xxx \\
      --output-dir data/extracted_graph_claude_opus_batch

  # Process all series with batch API
  python extract_entities_batch.py --model claude-opus-4-5-20251101 \\
      --input-dir data/processed_books_claude_200k \\
      --output-dir data/extracted_graph_claude_opus_batch

Cost Comparison:
  - Real-time API: Full price
  - Batch API: 50% discount (same quality, delayed processing)
  - Processing time: Up to 24 hours (often much faster)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "submit", "status", "retrieve"],
        help="Workflow mode: full (default), submit, status, or retrieve"
    )
    
    parser.add_argument(
        "--batch-id",
        type=str,
        help="Batch ID for status/retrieve modes"
    )
    
    parser.add_argument(
        "--series",
        type=str,
        help='Specific series to process. Options: "Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"'
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed_books_claude_200k",
        help="Directory containing processed text sections (default: data/processed_books_claude_200k)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extracted_graph_batch",
        help="Directory to save extracted knowledge graphs (default: data/extracted_graph_batch)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-5-20251101",
        choices=ALLOWED_MODELS["anthropic"],
        help="Claude model to use (default: claude-opus-4-5-20251101)"
    )
    
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks when polling (default: 60)"
    )
    
    parser.add_argument(
        "--max-wait-hours",
        type=float,
        default=24.0,
        help="Maximum hours to wait for batch completion (default: 24.0)"
    )
    
    parser.add_argument(
        "--jobs-dir",
        type=str,
        default="data/batch_jobs",
        help="Directory for batch job metadata persistence (default: data/batch_jobs)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ["status", "retrieve"] and not args.batch_id:
        parser.error(f"--batch-id is required for mode '{args.mode}'")
    
    # Log configuration
    logger.info(
        "batch_extraction_started",
        mode=args.mode,
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        series=args.series or "all",
        batch_id=args.batch_id
    )
    
    # Initialize extractor
    extractor = BatchEntityExtractor(
        output_dir=args.output_dir,
        model_name=args.model,
        jobs_dir=args.jobs_dir,
    )
    
    # Execute based on mode
    if args.mode == "status":
        # Check status of existing batch
        status = await extractor.check_status(args.batch_id)
        logger.info("batch_status", **status)
        print(json.dumps(status, indent=2))
        
    elif args.mode == "retrieve":
        # Retrieve results from completed batch
        results = await extractor.retrieve_results(args.batch_id)
        logger.info("batch_results", **results)
        print(json.dumps(results, indent=2, default=str))
        
    elif args.mode == "submit":
        # Submit batch only
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Define series to process
        series_list = ["Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"]
        if args.series:
            series_list = [args.series]
        
        # Submit batches for each series
        batch_ids = []
        for series_name in series_list:
            logger.info("submitting_series_batch", series=series_name)
            batch_id = await extractor.submit_batch(series_name, input_dir)
            if batch_id:
                batch_ids.append({"series": series_name, "batch_id": batch_id})
        
        logger.info("batches_submitted", batches=batch_ids)
        print(json.dumps({"batches": batch_ids}, indent=2))
        
    else:  # mode == "full"
        # Full workflow: submit, poll, retrieve
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Define series to process
        series_list = ["Harry Potter", "A Song of Ice and Fire", "The Wheel of Time"]
        if args.series:
            series_list = [args.series]
        
        # Process each series
        all_results = []
        for series_name in series_list:
            logger.info("processing_series", series=series_name)
            result = await extractor.run_full_workflow(
                series_name=series_name,
                input_dir=input_dir,
                poll_interval_seconds=args.poll_interval,
                max_wait_hours=args.max_wait_hours,
            )
            all_results.append({"series": series_name, **result})
        
        # Log final summary
        total_processed = sum(r.get("processed", 0) for r in all_results)
        total_failed = sum(r.get("failed", 0) for r in all_results)
        total_tokens = sum(r.get("total_tokens", 0) for r in all_results)
        
        logger.info(
            "batch_extraction_complete",
            total_processed=total_processed,
            total_failed=total_failed,
            total_tokens=total_tokens,
            output_dir=args.output_dir
        )
        
        print(json.dumps({"results": all_results}, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())

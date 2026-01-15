# Dependencies:
# pip install anthropic structlog pydantic

"""
Anthropic Claude Batch API provider implementation.

This module extends the Anthropic provider with support for the Message Batches API,
which offers 50% cost reduction for asynchronous processing of large request volumes.

Key Features:
- Submit batches of up to 10,000 requests
- Poll for batch completion with configurable intervals
- Retrieve results in the same format as real-time API
- Automatic job persistence for resumable workflows
- Support for structured output via tool calling (same as real-time)

Batch API Workflow:
1. Create batch: Submit requests with custom_ids
2. Track job: Store batch_id for status monitoring
3. Poll status: Check processing_status until "ended"
4. Retrieve results: Download from results_url
5. Parse results: Match custom_ids to original requests

Cost Optimization:
- Batch API: 50% discount on input and output tokens
- Processing time: Up to 24 hours (often much faster)
- Best for: Non-time-sensitive bulk extraction workloads

Reference: https://docs.anthropic.com/en/api/creating-message-batches
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import structlog
from pydantic import BaseModel
from anthropic import AsyncAnthropic

from .base import UsageMetrics, ProviderType
from .batch_models import (
    BatchJob,
    BatchRequestItem,
    BatchRequestParams,
    BatchResultItem,
    BatchProcessingStatus,
    BatchResultType,
    RequestCounts,
)

# Initialize module-level logger
logger = structlog.get_logger()


class AnthropicBatchProvider:
    """
    Provider for Anthropic's Claude Message Batches API.
    
    This class handles all batch-specific operations while maintaining
    compatibility with the existing AnthropicProvider for real-time calls.
    
    The batch API is accessed through the same AsyncAnthropic client but
    uses different endpoints (/v1/messages/batches instead of /v1/messages).
    
    Architecture:
    - Separate from AnthropicProvider to keep concerns isolated
    - Can be composed with AnthropicProvider in UnifiedLLMService
    - Handles persistence of batch job state for resumability
    - Provides polling utilities for monitoring batch progress
    
    Attributes:
        client: AsyncAnthropic client instance
        api_key: Anthropic API key (masked in logs)
        jobs_dir: Directory for persisting batch job metadata
        log: Structured logger with provider context
    """

    def __init__(
        self, 
        api_key: str, 
        jobs_dir: str = "data/batch_jobs",
        **kwargs
    ):
        """
        Initialize the Anthropic Batch provider.
        
        Args:
            api_key: Anthropic API key (from console.anthropic.com)
            jobs_dir: Directory to persist batch job metadata for resumability
            **kwargs: Additional configuration passed to AsyncAnthropic client
        """
        self.api_key = api_key
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize async Anthropic client
        self.client = AsyncAnthropic(api_key=api_key, **kwargs)
        
        # Bind logger with provider context
        self.log = logger.bind(provider="AnthropicBatchProvider")
        self.log.info(
            "batch_provider_initialized", 
            api_key=self._mask_api_key(),
            jobs_dir=str(self.jobs_dir)
        )

    def _mask_api_key(self) -> str:
        """Mask API key for safe logging (show only last 4 characters)."""
        if len(self.api_key) > 4:
            return f"...{self.api_key[-4:]}"
        return "***"

    async def create_batch(
        self,
        requests: List[BatchRequestItem],
        output_dir: str,
        model: str,
        series_name: Optional[str] = None,
    ) -> BatchJob:
        """
        Submit a batch of requests to the Anthropic API.
        
        This method creates a new batch job and persists its metadata locally
        for tracking and resumability.
        
        Args:
            requests: List of BatchRequestItem objects (max 10,000)
            output_dir: Directory where results will be saved
            model: Model identifier for logging/tracking
            series_name: Optional series name for context
            
        Returns:
            BatchJob object with batch_id and initial status
            
        Raises:
            anthropic.APIError: If batch creation fails
            ValueError: If requests list is empty or exceeds 10,000
            
        Example:
            requests = [
                BatchRequestItem(
                    custom_id="section_01",
                    params=BatchRequestParams(
                        model="claude-opus-4-5-20251101",
                        max_tokens=64000,
                        messages=[{"role": "user", "content": "Extract entities..."}]
                    )
                )
            ]
            job = await provider.create_batch(requests, "data/extracted", "claude-opus-4-5")
        """
        log = self.log.bind(
            endpoint="create_batch",
            num_requests=len(requests),
            model=model,
            series=series_name
        )
        
        # Validate request count
        if not requests:
            raise ValueError("Batch must contain at least one request")
        if len(requests) > 10000:
            raise ValueError(f"Batch cannot exceed 10,000 requests (got {len(requests)})")
        
        log.info("creating_batch", num_requests=len(requests))
        
        try:
            # Build request mapping for result correlation
            # Maps custom_id -> local file path where result should be saved
            request_mapping = {}
            for req in requests:
                # Derive output path from custom_id (e.g., "section_01" -> "section_01_extracted.json")
                output_path = str(Path(output_dir) / f"{req.custom_id}_extracted.json")
                request_mapping[req.custom_id] = output_path
            
            # Format requests for Anthropic API
            # The API expects a list of dicts with custom_id and params
            api_requests = []
            for req in requests:
                api_request = {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": req.params.model,
                        "max_tokens": req.params.max_tokens,
                        "temperature": req.params.temperature,
                        "messages": req.params.messages,
                    }
                }
                
                # Add optional fields if present
                if req.params.tools:
                    api_request["params"]["tools"] = req.params.tools
                if req.params.tool_choice:
                    api_request["params"]["tool_choice"] = req.params.tool_choice
                if req.params.system:
                    api_request["params"]["system"] = req.params.system
                    
                api_requests.append(api_request)
            
            # Submit batch to Anthropic API
            # Uses the messages.batches.create endpoint
            response = await self.client.messages.batches.create(
                requests=api_requests
            )
            
            # Parse response into BatchJob model
            batch_job = BatchJob(
                id=response.id,
                type=response.type,
                processing_status=BatchProcessingStatus(response.processing_status),
                request_counts=RequestCounts(
                    processing=response.request_counts.processing,
                    succeeded=response.request_counts.succeeded,
                    errored=response.request_counts.errored,
                    canceled=response.request_counts.canceled,
                    expired=response.request_counts.expired,
                ),
                created_at=response.created_at,
                ended_at=response.ended_at,
                expires_at=response.expires_at,
                results_url=response.results_url,
                # Local tracking fields
                local_request_mapping=request_mapping,
                output_dir=output_dir,
                provider="anthropic",
                model=model,
                series_name=series_name,
            )
            
            # Persist job metadata for resumability
            await self._save_job(batch_job)
            
            log.info(
                "batch_created",
                batch_id=batch_job.id,
                status=batch_job.processing_status,
                expires_at=batch_job.expires_at.isoformat()
            )
            
            return batch_job
            
        except Exception as e:
            log.exception("batch_creation_failed", error=str(e))
            raise

    async def get_batch_status(self, batch_id: str) -> BatchJob:
        """
        Retrieve the current status of a batch job.
        
        This method queries the Anthropic API for the latest batch status
        and updates the local job metadata.
        
        Args:
            batch_id: The batch ID returned from create_batch
            
        Returns:
            Updated BatchJob with current status and request counts
            
        Raises:
            anthropic.APIError: If status retrieval fails
            FileNotFoundError: If local job metadata not found
        """
        log = self.log.bind(endpoint="get_batch_status", batch_id=batch_id)
        
        try:
            # Query Anthropic API for current status
            response = await self.client.messages.batches.retrieve(batch_id)
            
            # Load existing job metadata (for local tracking fields)
            existing_job = await self._load_job(batch_id)
            
            # Update job with latest status from API
            updated_job = BatchJob(
                id=response.id,
                type=response.type,
                processing_status=BatchProcessingStatus(response.processing_status),
                request_counts=RequestCounts(
                    processing=response.request_counts.processing,
                    succeeded=response.request_counts.succeeded,
                    errored=response.request_counts.errored,
                    canceled=response.request_counts.canceled,
                    expired=response.request_counts.expired,
                ),
                created_at=response.created_at,
                ended_at=response.ended_at,
                expires_at=response.expires_at,
                results_url=response.results_url,
                # Preserve local tracking fields
                local_request_mapping=existing_job.local_request_mapping if existing_job else {},
                output_dir=existing_job.output_dir if existing_job else None,
                provider="anthropic",
                model=existing_job.model if existing_job else "",
                series_name=existing_job.series_name if existing_job else None,
            )
            
            # Persist updated status
            await self._save_job(updated_job)
            
            log.info(
                "batch_status_retrieved",
                status=updated_job.processing_status,
                succeeded=updated_job.request_counts.succeeded,
                errored=updated_job.request_counts.errored,
                processing=updated_job.request_counts.processing
            )
            
            return updated_job
            
        except Exception as e:
            log.exception("batch_status_retrieval_failed", error=str(e))
            raise

    async def poll_until_complete(
        self,
        batch_id: str,
        poll_interval_seconds: int = 60,
        max_wait_hours: float = 24.0,
    ) -> BatchJob:
        """
        Poll a batch job until it completes or times out.
        
        This method repeatedly checks the batch status at the specified interval
        until the batch reaches a terminal state (ended, canceled, expired).
        
        Args:
            batch_id: The batch ID to poll
            poll_interval_seconds: Seconds between status checks (default: 60)
            max_wait_hours: Maximum hours to wait before timing out (default: 24)
            
        Returns:
            Final BatchJob with terminal status
            
        Raises:
            TimeoutError: If max_wait_hours exceeded
            anthropic.APIError: If status retrieval fails
        """
        log = self.log.bind(
            endpoint="poll_until_complete",
            batch_id=batch_id,
            poll_interval=poll_interval_seconds,
            max_wait_hours=max_wait_hours
        )
        
        log.info("starting_batch_poll")
        
        max_wait_seconds = max_wait_hours * 3600
        elapsed_seconds = 0
        poll_count = 0
        
        while elapsed_seconds < max_wait_seconds:
            poll_count += 1
            
            # Get current status
            job = await self.get_batch_status(batch_id)
            
            # Check for terminal states
            terminal_states = {
                BatchProcessingStatus.ENDED,
                BatchProcessingStatus.CANCELED,
                BatchProcessingStatus.EXPIRED,
            }
            
            if job.processing_status in terminal_states:
                log.info(
                    "batch_poll_complete",
                    final_status=job.processing_status,
                    poll_count=poll_count,
                    elapsed_minutes=elapsed_seconds / 60
                )
                return job
            
            # Log progress
            log.info(
                "batch_poll_progress",
                status=job.processing_status,
                succeeded=job.request_counts.succeeded,
                processing=job.request_counts.processing,
                poll_count=poll_count,
                elapsed_minutes=elapsed_seconds / 60
            )
            
            # Wait before next poll
            await asyncio.sleep(poll_interval_seconds)
            elapsed_seconds += poll_interval_seconds
        
        # Timeout reached
        log.error(
            "batch_poll_timeout",
            batch_id=batch_id,
            elapsed_hours=elapsed_seconds / 3600
        )
        raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait_hours} hours")

    async def retrieve_results(self, batch_id: str) -> List[BatchResultItem]:
        """
        Download and parse results from a completed batch.
        
        This method uses the Anthropic SDK to stream results from the batch,
        parsing each result into structured BatchResultItem objects.
        
        Args:
            batch_id: The batch ID to retrieve results for
            
        Returns:
            List of BatchResultItem objects with responses or errors
            
        Raises:
            ValueError: If batch is not in "ended" state
            anthropic.APIError: If results retrieval fails
        """
        log = self.log.bind(endpoint="retrieve_results", batch_id=batch_id)
        
        # Get current job status
        job = await self.get_batch_status(batch_id)
        
        if job.processing_status != BatchProcessingStatus.ENDED:
            raise ValueError(
                f"Cannot retrieve results: batch status is '{job.processing_status}', "
                f"expected 'ended'"
            )
        
        log.info("retrieving_batch_results", batch_id=batch_id)
        
        try:
            # Use the Anthropic SDK to stream results
            # The SDK handles authentication and pagination automatically
            results = []
            
            # Get the async iterator by awaiting the results() coroutine
            results_iterator = await self.client.messages.batches.results(batch_id)
            
            async for result in results_iterator:
                # Each result has custom_id and result (with type, message/error)
                result_type = BatchResultType(result.result.type)
                
                # Extract message or error based on result type
                message_data = None
                error_data = None
                
                if result_type == BatchResultType.SUCCEEDED:
                    # Convert the message object to dict format expected by BatchResultItem
                    msg = result.result.message
                    message_data = {
                        "id": msg.id,
                        "type": msg.type,
                        "role": msg.role,
                        "content": [
                            {"type": block.type, **block.model_dump()} 
                            for block in msg.content
                        ],
                        "model": msg.model,
                        "stop_reason": msg.stop_reason,
                        "usage": {
                            "input_tokens": msg.usage.input_tokens,
                            "output_tokens": msg.usage.output_tokens,
                        }
                    }
                elif result_type == BatchResultType.ERRORED:
                    error_data = {
                        "type": result.result.error.type,
                        "message": result.result.error.message,
                    }
                
                result_item = BatchResultItem(
                    custom_id=result.custom_id,
                    result_type=result_type,
                    message=message_data,
                    error=error_data,
                )
                results.append(result_item)
            
            log.info(
                "batch_results_retrieved",
                total_results=len(results),
                succeeded=sum(1 for r in results if r.result_type == BatchResultType.SUCCEEDED),
                errored=sum(1 for r in results if r.result_type == BatchResultType.ERRORED)
            )
            
            return results
                
        except Exception as e:
            log.exception("batch_results_retrieval_failed", error=str(e))
            raise

    async def process_results_to_files(
        self,
        batch_id: str,
        schema: Any,
    ) -> Dict[str, Any]:
        """
        Process batch results and save extracted data to files.
        
        This method retrieves batch results, parses structured output from
        tool calls, validates against the provided schema, and saves results
        to the output files specified in the job's request mapping.
        
        This is the key method that makes batch results compatible with
        the existing extraction pipeline - downstream consumers receive
        the same file format regardless of real-time vs batch processing.
        
        Args:
            batch_id: The batch ID to process
            schema: Pydantic BaseModel class for validating extracted data
            
        Returns:
            Dictionary with:
            - 'processed': Number of successfully processed results
            - 'failed': Number of failed results
            - 'usage': Aggregated UsageMetrics
            - 'files_written': List of output file paths
            
        Raises:
            ValueError: If batch not completed or schema validation fails
        """
        log = self.log.bind(
            endpoint="process_results_to_files",
            batch_id=batch_id,
            schema_name=schema.__name__
        )
        
        # Load job metadata for request mapping
        job = await self._load_job(batch_id)
        if not job:
            raise ValueError(f"Job metadata not found for batch {batch_id}")
        
        # Retrieve results from API
        results = await self.retrieve_results(batch_id)
        
        # Track processing statistics
        processed_count = 0
        failed_count = 0
        files_written = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for result in results:
            result_log = log.bind(custom_id=result.custom_id)
            
            # Get output path from request mapping
            output_path = job.local_request_mapping.get(result.custom_id)
            if not output_path:
                result_log.warning("no_output_path_for_result")
                continue
            
            if result.result_type == BatchResultType.SUCCEEDED:
                try:
                    # Extract tool use result from response
                    # Anthropic structured output uses tool calling
                    tool_use_block = None
                    for block in result.message.content:
                        if block.get("type") == "tool_use" and block.get("name") == "extract_information":
                            tool_use_block = block
                            break
                    
                    if not tool_use_block:
                        result_log.error("no_tool_use_in_response")
                        failed_count += 1
                        continue
                    
                    # Validate against schema
                    parsed_output = schema.model_validate(tool_use_block["input"])
                    
                    # Write to output file
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, "w") as f:
                        f.write(parsed_output.model_dump_json(indent=2))
                    
                    files_written.append(str(output_file))
                    processed_count += 1
                    
                    # Accumulate token usage
                    if result.message.usage:
                        total_input_tokens += result.message.usage.get("input_tokens", 0)
                        total_output_tokens += result.message.usage.get("output_tokens", 0)
                    
                    result_log.info(
                        "result_processed",
                        output_file=str(output_file)
                    )
                    
                except Exception as e:
                    result_log.exception("result_processing_failed", error=str(e))
                    failed_count += 1
                    
            else:
                # Log failed result
                error_msg = result.error.message if result.error else "Unknown error"
                result_log.error(
                    "batch_request_failed",
                    result_type=result.result_type,
                    error=error_msg
                )
                failed_count += 1
        
        # Build aggregated usage metrics
        usage = UsageMetrics(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
            provider=ProviderType.ANTHROPIC.value,
            model=job.model,
            api_key_last4=self._mask_api_key(),
        )
        
        log.info(
            "batch_processing_complete",
            processed=processed_count,
            failed=failed_count,
            total_tokens=usage.total_tokens,
            files_written=len(files_written)
        )
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "usage": usage,
            "files_written": files_written,
        }

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        """
        Request cancellation of a batch job.
        
        Cancellation is not immediate - the batch transitions to "canceling"
        state and eventually to "canceled" once all in-flight requests complete.
        
        Args:
            batch_id: The batch ID to cancel
            
        Returns:
            Updated BatchJob with canceling/canceled status
        """
        log = self.log.bind(endpoint="cancel_batch", batch_id=batch_id)
        
        try:
            response = await self.client.messages.batches.cancel(batch_id)
            
            job = await self.get_batch_status(batch_id)
            
            log.info("batch_cancellation_requested", status=job.processing_status)
            
            return job
            
        except Exception as e:
            log.exception("batch_cancellation_failed", error=str(e))
            raise

    async def list_batches(self, limit: int = 20) -> List[BatchJob]:
        """
        List recent batch jobs.
        
        Args:
            limit: Maximum number of batches to return (default: 20)
            
        Returns:
            List of BatchJob objects ordered by creation date (newest first)
        """
        log = self.log.bind(endpoint="list_batches", limit=limit)
        
        try:
            response = await self.client.messages.batches.list(limit=limit)
            
            jobs = []
            for batch in response.data:
                job = BatchJob(
                    id=batch.id,
                    type=batch.type,
                    processing_status=BatchProcessingStatus(batch.processing_status),
                    request_counts=RequestCounts(
                        processing=batch.request_counts.processing,
                        succeeded=batch.request_counts.succeeded,
                        errored=batch.request_counts.errored,
                        canceled=batch.request_counts.canceled,
                        expired=batch.request_counts.expired,
                    ),
                    created_at=batch.created_at,
                    ended_at=batch.ended_at,
                    expires_at=batch.expires_at,
                    results_url=batch.results_url,
                )
                jobs.append(job)
            
            log.info("batches_listed", count=len(jobs))
            
            return jobs
            
        except Exception as e:
            log.exception("batch_listing_failed", error=str(e))
            raise

    async def _save_job(self, job: BatchJob) -> None:
        """
        Persist batch job metadata to disk.
        
        Job metadata is saved as JSON in the jobs_dir, enabling:
        - Resumable workflows after script restart
        - Result correlation via request mapping
        - Historical tracking of batch jobs
        
        Args:
            job: BatchJob to persist
        """
        job_path = self.jobs_dir / f"{job.id}.json"
        
        with open(job_path, "w") as f:
            f.write(job.model_dump_json(indent=2))
        
        self.log.debug("job_saved", batch_id=job.id, path=str(job_path))

    async def _load_job(self, batch_id: str) -> Optional[BatchJob]:
        """
        Load batch job metadata from disk.
        
        Args:
            batch_id: The batch ID to load
            
        Returns:
            BatchJob if found, None otherwise
        """
        job_path = self.jobs_dir / f"{batch_id}.json"
        
        if not job_path.exists():
            self.log.debug("job_not_found", batch_id=batch_id)
            return None
        
        with open(job_path, "r") as f:
            data = json.load(f)
        
        return BatchJob.model_validate(data)

    def get_job_path(self, batch_id: str) -> Path:
        """
        Get the file path for a batch job's metadata.
        
        Args:
            batch_id: The batch ID
            
        Returns:
            Path to the job metadata file
        """
        return self.jobs_dir / f"{batch_id}.json"


def build_extraction_batch_request(
    custom_id: str,
    prompt: str,
    model: str,
    schema: Any,
    max_tokens: int = 64000,
    temperature: float = 0.0,
) -> BatchRequestItem:
    """
    Build a batch request item for entity extraction.
    
    This helper function creates a properly formatted BatchRequestItem
    for structured extraction, using tool calling to enforce schema compliance.
    
    Args:
        custom_id: Unique identifier for this request (e.g., "section_01")
        prompt: Full extraction prompt including schema and text content
        model: Claude model identifier
        schema: Pydantic BaseModel defining extraction output structure
        max_tokens: Maximum tokens to generate (default: 64000 for Claude)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        
    Returns:
        BatchRequestItem ready for batch submission
        
    Example:
        request = build_extraction_batch_request(
            custom_id="harry_potter_section_01",
            prompt="Extract entities from: ...",
            model="claude-opus-4-5-20251101",
            schema=ExtractionResult
        )
    """
    # Convert Pydantic schema to Anthropic tool format
    tool_schema = {
        "name": "extract_information",
        "description": "Extract structured information from the text",
        "input_schema": schema.model_json_schema(),
    }
    
    return BatchRequestItem(
        custom_id=custom_id,
        params=BatchRequestParams(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": "extract_information"},
        )
    )

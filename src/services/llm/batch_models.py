# Dependencies:
# pip install pydantic structlog

"""
Batch API data models for Anthropic Claude Message Batches.

This module defines the data structures used for batch processing operations:
- BatchRequest: Individual request within a batch
- BatchJob: Tracks batch job state and metadata
- BatchResult: Individual result from a completed batch
- BatchJobStatus: Enumeration of batch processing states

The batch API offers 50% cost reduction compared to real-time API calls,
making it ideal for large-scale entity extraction workloads.

Architecture:
These models are designed to:
1. Mirror the Anthropic API response structure for easy deserialization
2. Provide type safety through Pydantic validation
3. Enable persistence of batch job state for resumable workflows
4. Support seamless integration with existing extraction pipelines

Reference: https://docs.anthropic.com/en/api/creating-message-batches
"""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class BatchProcessingStatus(str, Enum):
    """
    Processing status states for a batch job.
    
    The batch lifecycle follows this progression:
    1. CREATED -> Initial state after submission
    2. IN_PROGRESS -> Batch is being processed
    3. ENDED -> All requests completed (check individual results for success/failure)
    4. CANCELING -> Cancellation requested, in progress
    5. CANCELED -> Batch was canceled before completion
    6. EXPIRED -> Batch expired before completion (rare)
    """
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    CANCELING = "canceling"
    CANCELED = "canceled"
    EXPIRED = "expired"


class BatchResultType(str, Enum):
    """
    Result type for individual requests within a batch.
    
    Each request in a batch can have one of these outcomes:
    - SUCCEEDED: Request completed successfully, response available
    - ERRORED: Request failed with an error
    - CANCELED: Request was canceled before processing
    - EXPIRED: Request expired before processing
    """
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"


class BatchRequestParams(BaseModel):
    """
    Parameters for a single request within a batch.
    
    This mirrors the standard Messages API parameters, allowing the same
    prompts and configurations used in real-time calls to be batched.
    
    Attributes:
        model: Claude model identifier (e.g., "claude-opus-4-5-20251101")
        max_tokens: Maximum tokens to generate in the response
        temperature: Sampling temperature (0.0 = deterministic)
        messages: List of message dicts with role and content
        tools: Optional list of tool definitions for structured output
        tool_choice: Optional tool selection configuration
        system: Optional system prompt
    """
    model: str = Field(..., description="Claude model identifier")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    messages: List[Dict[str, Any]] = Field(..., description="Message list with role/content")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool definitions")
    tool_choice: Optional[Dict[str, Any]] = Field(default=None, description="Tool selection config")
    system: Optional[str] = Field(default=None, description="System prompt")


class BatchRequestItem(BaseModel):
    """
    A single request item to be included in a batch submission.
    
    Each item has a custom_id that is preserved in the results, allowing
    you to match responses back to their original requests.
    
    Attributes:
        custom_id: Unique identifier for this request (preserved in results)
        params: The actual request parameters
    """
    custom_id: str = Field(..., description="Unique identifier for matching results")
    params: BatchRequestParams = Field(..., description="Request parameters")


class RequestCounts(BaseModel):
    """
    Counts of requests in various states within a batch.
    
    These counts are updated as the batch processes and can be used
    to track progress and identify issues.
    
    Attributes:
        processing: Requests currently being processed
        succeeded: Requests that completed successfully
        errored: Requests that failed with errors
        canceled: Requests that were canceled
        expired: Requests that expired before processing
    """
    processing: int = Field(default=0, description="Currently processing")
    succeeded: int = Field(default=0, description="Completed successfully")
    errored: int = Field(default=0, description="Failed with error")
    canceled: int = Field(default=0, description="Canceled before processing")
    expired: int = Field(default=0, description="Expired before processing")


class BatchJob(BaseModel):
    """
    Represents a batch job submitted to the Anthropic API.
    
    This model tracks the full lifecycle of a batch, from creation through
    completion. It can be serialized to JSON for persistence, enabling
    resumable batch workflows.
    
    Attributes:
        id: Unique batch identifier from Anthropic API
        type: Always "message_batch" for this API
        processing_status: Current state of the batch
        request_counts: Breakdown of request states
        created_at: Timestamp when batch was created
        ended_at: Timestamp when batch completed (if ended)
        expires_at: Timestamp when batch will expire
        results_url: URL to download results (available when ended)
        
        # Local tracking fields (not from API)
        local_request_mapping: Maps custom_id to local file paths
        output_dir: Directory where results should be saved
        provider: LLM provider name (always "anthropic" for this)
        model: Model used for this batch
    """
    # API response fields
    id: str = Field(..., description="Batch ID from Anthropic API")
    type: str = Field(default="message_batch", description="Resource type")
    processing_status: BatchProcessingStatus = Field(..., description="Current processing state")
    request_counts: RequestCounts = Field(default_factory=RequestCounts, description="Request state counts")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    ended_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    results_url: Optional[str] = Field(default=None, description="URL to download results")
    
    # Local tracking fields (for persistence and result mapping)
    local_request_mapping: Dict[str, str] = Field(
        default_factory=dict, 
        description="Maps custom_id to local file path for result storage"
    )
    output_dir: Optional[str] = Field(default=None, description="Output directory for results")
    provider: str = Field(default="anthropic", description="LLM provider")
    model: str = Field(default="", description="Model used for batch")
    series_name: Optional[str] = Field(default=None, description="Series being processed")

    class Config:
        """Pydantic configuration for JSON serialization."""
        use_enum_values = True


class BatchResultMessage(BaseModel):
    """
    The message content from a successful batch result.
    
    This mirrors the standard Messages API response structure.
    
    Attributes:
        id: Message ID
        type: Always "message"
        role: Always "assistant" for responses
        content: List of content blocks (text, tool_use, etc.)
        model: Model that generated the response
        stop_reason: Why generation stopped
        usage: Token usage information
    """
    id: str = Field(..., description="Message ID")
    type: str = Field(default="message", description="Resource type")
    role: str = Field(default="assistant", description="Message role")
    content: List[Dict[str, Any]] = Field(..., description="Content blocks")
    model: str = Field(..., description="Model used")
    stop_reason: Optional[str] = Field(default=None, description="Stop reason")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage")


class BatchResultError(BaseModel):
    """
    Error information for a failed batch request.
    
    Attributes:
        type: Error type identifier
        message: Human-readable error message
    """
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")


class BatchResultItem(BaseModel):
    """
    A single result item from a completed batch.
    
    Each result corresponds to a request with matching custom_id.
    The result_type indicates success or failure.
    
    Attributes:
        custom_id: Matches the custom_id from the original request
        result_type: Success/error status
        message: The response message (if succeeded)
        error: Error details (if errored)
    """
    custom_id: str = Field(..., description="Request identifier")
    result_type: BatchResultType = Field(..., description="Result status")
    message: Optional[BatchResultMessage] = Field(default=None, description="Response (if succeeded)")
    error: Optional[BatchResultError] = Field(default=None, description="Error (if failed)")

    class Config:
        """Pydantic configuration for JSON serialization."""
        use_enum_values = True


class BatchSubmissionRequest(BaseModel):
    """
    Request body for submitting a new batch.
    
    This is the top-level structure sent to POST /v1/messages/batches.
    
    Attributes:
        requests: List of individual request items (max 10,000)
    """
    requests: List[BatchRequestItem] = Field(..., description="Batch requests (max 10,000)")


class BatchJobSummary(BaseModel):
    """
    Summary information for a batch job (used in listings).
    
    This is a lighter-weight representation used when listing batches,
    containing just the essential status information.
    
    Attributes:
        id: Batch ID
        processing_status: Current state
        request_counts: Request state breakdown
        created_at: Creation timestamp
    """
    id: str = Field(..., description="Batch ID")
    processing_status: BatchProcessingStatus = Field(..., description="Current state")
    request_counts: RequestCounts = Field(default_factory=RequestCounts)
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        """Pydantic configuration for JSON serialization."""
        use_enum_values = True

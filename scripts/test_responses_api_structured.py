#!/usr/bin/env python3
"""Test OpenAI Responses API structured output."""

import asyncio
import json
import os
import sys
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import after path setup
from dotenv import load_dotenv

load_dotenv()

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

from src.services.llm import UnifiedLLMService


class PersonInfo(BaseModel):
    """Simple schema for testing."""

    name: str
    age: int


async def test_structured_output():
    """Test structured output with Responses API."""
    log = structlog.get_logger()

    log.info("initializing_service")
    service = UnifiedLLMService()

    log.info("testing_gpt52pro_structured_output")

    try:
        result = await service.generate_structured(
            prompt="Extract information: Alice Johnson is 30 years old.",
            schema=PersonInfo,
            provider="openai",
            model="gpt-5.2-pro",
            max_tokens=500,
        )

        log.info(
            "structured_output_success",
            parsed=result["parsed"].model_dump(),
            tokens=result["usage"].total_tokens,
        )

        print("\n✓ SUCCESS!")
        print(f"Parsed: {result['parsed'].model_dump()}")
        print(f"Tokens: {result['usage'].total_tokens}")

    except Exception as e:
        log.exception("structured_output_failed", error=str(e))
        print(f"\n✗ FAILED: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_structured_output())

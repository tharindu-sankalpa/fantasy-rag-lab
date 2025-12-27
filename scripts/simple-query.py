# Dependencies:
# pip install google-genai python-dotenv structlog

import os
import structlog
import logging
import sys
from typing import Optional
from google import genai
from dotenv import load_dotenv

# 1. Configure Structlog (Scenario A)
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

# 2. Initialize the logger
logger = structlog.get_logger()

def main() -> None:
    """
    Main function to execute a simple query against Gemini.
    """
    log = logger.bind(task="simple_query")
    
    # 3. Load environment variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)

    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    if not api_key:
        log.error("api_key_missing", hint="Check your .env file")
        sys.exit(1)

    # 4. Initialize Gemini Client
    model_name = "gemini-3-pro-preview"
    log.info("initializing_client", model=model_name)
    
    try:
        client = genai.Client(api_key=api_key)

        # 5. Make the request
        log.info("sending_request")
        
        response = client.models.generate_content(
            model=model_name,
            contents="What are the highest performing vector embeddings models for retrieval?"
        )

        # 6. Log the success
        if response and response.text:
            log.info("response_received", content=response.text[:200] + "...", full_length=len(response.text))
        else:
             log.warning("response_empty")

    except Exception as e:
        log.exception("query_failed")

if __name__ == "__main__":
    main()
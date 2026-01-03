# Dependencies:
# pip install openai python-dotenv structlog

import os
import structlog
import logging
import sys
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Configure structlog (Scenario A: Standalone Script)
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
logger = structlog.get_logger()

# Get the path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
load_dotenv(os.path.join(project_root, ".env"))

def check_openai_access() -> None:
    """
    Checks if the OPENAI_API_KEY has access to OpenAI models.

    Specifically checks for available models and verifies access.

    Raises:
        Exception: If there is an error during API checks or client configuration.
    """
    log = logger.bind(task="check_openai_access")
    
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        log.error("api_key_missing", hint="Ensure OPENAI_API_KEY is set in .env")
        return

    # Mask key for privacy
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    log.info("api_key_found", key_preview=masked_key)
    
    # Configure the client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        log.error("client_config_failed", error=str(e))
        return

    log.info("checking_available_models")
    
    try:
        # List all models
        models = client.models.list()
        
        # Filter for chat models (gpt-4, gpt-3.5)
        chat_models = [m.id for m in models.data if "gpt" in m.id]
        chat_models.sort()
        
        if not chat_models:
            log.warning("no_gpt_models_found")
            return

        log.info("models_found", count=len(chat_models), sample=chat_models[:5])

        # Test generation with gpt-4o or fallback to first available
        target_model = "gpt-4o" if "gpt-4o" in chat_models else chat_models[0]
        
        log.info("testing_model", model=target_model)
        
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "user", "content": "Hello! Please confirm you are operational."}
                ]
            )
            
            content = response.choices[0].message.content
            if content:
                log.info("generation_successful", model=target_model, response=content.strip())
            else:
                log.warning("generation_empty", model=target_model)
                
        except Exception as e:
                log.exception("generation_failed", model=target_model)

    except Exception as e:
        log.exception("api_check_failed")

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.simplefilter("ignore")
    check_openai_access()

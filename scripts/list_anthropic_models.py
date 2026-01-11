# Dependencies:
# pip install requests python-dotenv structlog

import os
import sys
import requests
import json
from dotenv import load_dotenv

sys.path.append(os.getcwd())

try:
    from src.utils.logger import logger
except ImportError:
    import structlog
    import logging
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

def list_anthropic_models():
    # Load environment variables to get the key
    load_dotenv()
    log = logger.bind(task="list_anthropic_models")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("api_key_missing", hint="Set ANTHROPIC_API_KEY in .env file")
        return

    # Endpoint for listing models
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01", 
        "content-type": "application/json"
    }
    
    try:
        log.info("fetching_models", url=url)
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            log.error("request_failed", status_code=response.status_code, response=response.text)
            return
            
        data = response.json()
        models = data.get("data", [])
        
        log.info("models_fetched", count=len(models))
        
        print(f"\n{'='*80}")
        print(f"ANTHROPIC MODELS ({len(models)})")
        print(f"{'='*80}")
        print(f"{'ID':<40} | {'Name':<30}")
        print(f"{'-'*80}")
        
        # Sort by ID or creation if available
        # The list models response typically contains id, display_name, created_at, etc.
        for m in sorted(models, key=lambda x: x.get('created_at', 0), reverse=True):
            mid = m.get("id", "unknown")
            name = m.get("display_name", mid)
            print(f"{mid:<40} | {name:<30}")
            
        print(f"{'='*80}\n")
        
    except Exception as e:
        log.exception("unexpected_error", error=str(e))

if __name__ == "__main__":
    list_anthropic_models()

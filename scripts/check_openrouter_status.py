# Dependencies:
# pip install requests structlog python-dotenv

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Ensure we can import from src if running from root
sys.path.append(os.getcwd())

try:
    from src.utils.logger import logger
except ImportError:
    # Fallback if running directly from scripts dir and src is not found
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

def check_openrouter_status() -> None:
    """
    Checks the OpenRouter API key status, including rate limits and credits.
    Fetches data from https://openrouter.ai/api/v1/key and logs the result.
    """
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    log = logger.bind(task="check_openrouter_status")
    
    if not api_key:
        log.error("api_key_missing", variable="OPENROUTER_API_KEY", hint="Add OPENROUTER_API_KEY to .env file")
        return

    url = "https://openrouter.ai/api/v1/key"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        log.info("sending_request", url=url)
        response = requests.get(url, headers=headers, timeout=10)
        
        # Handle non-200 responses gracefully to show body
        if response.status_code != 200:
            log.error("request_failed_status", status_code=response.status_code, response=response.text)
            return
            
        data = response.json()
        key_data = data.get("data", {})

        log.info("response_received", data=data)
        
        if not key_data:
            log.warning("unexpected_response_format", raw_data=data)
            return

        # Log main credit info
        log.info("credit_status", 
                 limit=key_data.get("limit"), 
                 limit_remaining=key_data.get("limit_remaining"), 
                 is_free_tier=key_data.get("is_free_tier"),
                 label=key_data.get("label")
        )
        
        # Log usage stats
        log.info("usage_statistics", 
                 usage_total=key_data.get("usage"),
                 usage_daily=key_data.get("usage_daily"),
                 usage_weekly=key_data.get("usage_weekly"),
                 usage_monthly=key_data.get("usage_monthly")
        )
        
    except requests.exceptions.RequestException as e:
        log.exception("request_execution_failed", error=str(e))
    except json.JSONDecodeError as e:
        log.exception("json_decode_failed", error=str(e), response_preview=response.text[:200])
    except Exception as e:
        log.exception("unknown_error", error=str(e))

if __name__ == "__main__":
    check_openrouter_status()

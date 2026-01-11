# Dependencies:
# pip install requests python-dotenv structlog

import os
import sys
import argparse
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

def mask_key(k):
    if not k: return "None"
    if len(k) > 8: return f"{k[:4]}...{k[-4:]}"
    return "***"

def test_model(provider: str, model_name: str):
    load_dotenv()
    log = logger.bind(task="test_model_generation")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("api_key_missing", hint="Check .env for OPENROUTER_API_KEY")
        return

    log.info("configuration", api_key_used=mask_key(api_key))
    
    # Resolve Model ID
    if provider:
        target_model = f"{provider}/{model_name}"
    else:
        target_model = model_name
        
    log.info("preparing_request", target_model=target_model)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://fantasy-rag-lab.com", 
        "X-Title": "Fantasy RAG Lab Test Script",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": target_model,
        "messages": [
            {"role": "user", "content": "Hello! Please identify yourself."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        
        if response.status_code != 200:
            log.error("request_failed", status_code=response.status_code, response=response.text)
            return
            
        data = response.json()
        
        # Log Usage / Model info
        usage = data.get("usage", {})
        model_used = data.get("model", "unknown")
        
        log.info("response_received", model_used=model_used, usage=usage)
        
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            print(f"\n--- MODEL OUTPUT ({target_model}) ---\n")
            print(content)
            print("\n---------------------------------------\n")
        else:
            log.warning("no_choices_found", raw=data)
            
    except Exception as e:
        log.exception("test_failed", error=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation with an OpenRouter model.")
    parser.add_argument("model", help="Model name (e.g. gemini-pro-1.5) or full ID (google/gemini-pro-1.5)")
    parser.add_argument("--provider", "-p", help="Provider prefix (e.g. google). If provided, ID will be constructed as {provider}/{model}", default=None)
    
    args = parser.parse_args()
    test_model(args.provider, args.model)

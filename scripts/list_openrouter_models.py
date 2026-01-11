# Dependencies:
# pip install requests python-dotenv structlog

import requests
import json
import os
import sys
from collections import defaultdict
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

def fetch_and_list_models():
    load_dotenv()
    log = logger.bind(task="list_openrouter_models")
    
    url = "https://openrouter.ai/api/v1/models"
    
    try:
        log.info("fetching_models", url=url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        models = data.get("data", [])
        log.info("models_fetched", count=len(models))
        
        # Group by provider
        by_provider = defaultdict(list)
        for m in models:
            mid = m.get("id", "")
            if "/" in mid:
                provider = mid.split("/")[0]
            else:
                provider = "other"
            by_provider[provider].append(m)
            
        print(f"\n{'='*100}")
        print(f"OPENROUTER MODELS (Total: {len(models)})")
        print(f"{'='*100}")
        
        for provider in sorted(by_provider.keys()):
            p_models = by_provider[provider]
            print(f"\n[ {provider.upper()} ] - {len(p_models)} models")
            print(f"{'-'*100}")
            print(f"{'ID':<50} | {'Context':<10} | {'Prompt ($/1M)':<16} | {'Comp ($/1M)':<16}")
            print(f"{'-'*100}")
            
            for m in sorted(p_models, key=lambda x: x['id']):
                mid = m['id']
                pricing = m.get("pricing", {})
                
                # Pricing allows strings or floats in raw json usually
                try:
                    p_prompt = float(pricing.get("prompt", 0)) * 1_000_000
                    p_comp = float(pricing.get("completion", 0)) * 1_000_000
                    p_str = f"${p_prompt:.4f}"
                    c_str = f"${p_comp:.4f}"
                except:
                    p_str = str(pricing.get("prompt", "?"))
                    c_str = str(pricing.get("completion", "?"))

                ctx = m.get("context_length", 0)
                
                print(f"{mid:<50} | {str(ctx):<10} | {p_str:<16} | {c_str:<16}")
        
        print(f"\n{'='*100}\n")
        
    except Exception as e:
        log.exception("failed_to_list_models")

if __name__ == "__main__":
    fetch_and_list_models()

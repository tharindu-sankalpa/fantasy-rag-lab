# Dependencies:
# pip install google-genai python-dotenv structlog

import os
import structlog
import logging
import sys
from typing import List, Optional
from google import genai
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

def check_gemini_access() -> None:
    """
    Checks if the GEMINI_API_KEY has access to Gemini models.

    Specifically looks for 'Gemini 3' or the latest available versions and tests
    generation capabilities.

    Raises:
        Exception: If there is an error during API checks or client configuration.
    """
    log = logger.bind(task="check_gemini_access")
    
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        log.error("api_key_missing", hint="Ensure GEMINI_API_KEY is set in .env")
        return

    # Mask key for privacy
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    log.info("api_key_found", key_preview=masked_key)
    
    # Configure the client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        log.error("client_config_failed", error=str(e))
        return

    log.info("checking_available_models")
    
    try:
        # List all models
        all_models = list(client.models.list())
        
        # Filter for 'generateContent' capable models and 'gemini' in name
        chat_models = [
            m for m in all_models 
            if m.supported_actions and 'generateContent' in m.supported_actions and 'gemini' in m.name
        ]
        chat_models.sort(key=lambda x: x.name)

        if not chat_models:
            log.warning("no_gemini_models_found")
            return
        
        gemini_3_models = []
        model_names = [m.name for m in chat_models]
        
        log.info("models_found", count=len(chat_models), models=model_names)
        
        for model in chat_models:
             if "gemini-3" in model.name.lower():
                gemini_3_models.append(model)

        ifgemini_3_models_found = bool(gemini_3_models)
        
        if gemini_3_models:
            g3_names = [m.name for m in gemini_3_models]
            log.info("gemini_3_models_found", count=len(gemini_3_models), models=g3_names)
            
            log.info("testing_gemini_3_models")
            for model in gemini_3_models:
                target_name = model.name
                log.info("testing_model", model=target_name)
                
                try:
                    response = client.models.generate_content(
                        model=target_name,
                        contents="Hello! Please confirm you are operational."
                    )
                    
                    if response and response.text:
                        log.info("generation_successful", model=target_name, response=response.text.strip())
                    else:
                        log.warning("generation_empty", model=target_name)
                        
                except Exception as e:
                     log.exception("generation_failed", model=target_name)

        else:
            log.info("gemini_3_not_explicitly_found")
            # Could add fallback test here if desired

    except Exception as e:
        log.exception("api_check_failed")

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.simplefilter("ignore")
    check_gemini_access()

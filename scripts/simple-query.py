import os
import structlog
from google import genai
from dotenv import load_dotenv

# 1. Configure Structlog for human-readable terminal output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        # ConsoleRenderer handles the "pretty" coloring and formatting
        structlog.dev.ConsoleRenderer(colors=True) 
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

# 2. Initialize the logger
logger = structlog.get_logger()

# 3. Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# Ensure path exists or fallback to current directory for safety
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    logger.error("API Key missing", hint="Check your .env file")
    exit(1)

# 4. Initialize Gemini Client
logger.info("Initializing Gemini Client", model="gemini-3-pro-preview")
client = genai.Client(api_key=api_key)

try:
    # 5. Make the request
    logger.info("Sending request to Gemini...")
    
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents="What are the highest performing vector embeddings models for retrieval?"
    )

    # 6. Log the success with the response content
    # We pass the long text as a keyword argument (content=...) so it looks cleaner
    logger.info("Response received", content=response.text)

except Exception as e:
    logger.exception("Failed to generate content", error=str(e))
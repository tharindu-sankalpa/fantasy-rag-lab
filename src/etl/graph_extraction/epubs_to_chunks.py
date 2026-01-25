# Dependencies:
# pip install ebooklib beautifulsoup4 tiktoken structlog

"""EPUB Text Extraction and Chunking Module.

This script extracts text content from EPUB files and organizes it into chunks 
tailored to specific LLM context windows (e.g., Claude, Gemini, GPT). It handles
token counting, safety margins, and metadata generation for each chunk.
"""

import os
import json
import glob
from typing import List, Dict, Any, Tuple
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import tiktoken
from src.utils.logger import get_logger

# Initialize project logger
logger = get_logger(__name__)

# Constants
DATA_DIR = os.path.abspath("data")

# Series directories to process
SERIES_DIRS = [
    "harry_potter",
    "wheel_of_time",
    "song_of_ice_and_fire"
]

# Model Configurations
# Limits are in tokens.
MODEL_CONFIGS = {
    "claude": {
        "context_window": 200_000,
        "output_dir_name": "processed_books_claude_200k",
        "description": "Claude 200k Context Content"
    },
    "gemini": {
        "context_window": 1_000_000,
        "output_dir_name": "processed_books_gemini_1m",
        "description": "Gemini 1M Context Content"
    },
    "gpt": {
        "context_window": 400_000,
        "output_dir_name": "processed_books_gpt_400k",
        "description": "GPT-5.2 400k Context Content"
    }
}

SAFETY_MARGIN = 0.10 # 10% safety buffer

def get_base_encoding() -> tiktoken.Encoding:
    """Retrieves the tokenizer encoding.

    Returns:
        tiktoken.Encoding: The 'cl100k_base' encoding used for GPT-4/3.5 models.
    """
    return tiktoken.get_encoding("cl100k_base")

def extract_text_from_epub(epub_path: str) -> str:
    """Extracts raw text content from a single EPUB file.

    This function reads the EPUB file using `ebooklib`, iterates through its 
    document items, and uses `BeautifulSoup` to strip HTML tags, returning 
    clean text.

    Args:
        epub_path (str): The absolute path to the EPUB file.

    Returns:
        str: The extracted text content, separated by double newlines. 
             Returns an empty string if extraction fails.
    """
    log = logger.bind(task="extract_text", file_path=epub_path)
    
    try:
        # Open the EPUB file
        book = epub.read_epub(epub_path)
        text_content = []
        
        # Iterate over all items in the book
        for item in book.get_items():
            # We only care about document items (chapters, sections)
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Use BS4 to strip HTML tags and get clean text
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                
                # Only append if there is actual text content
                if text:
                    text_content.append(text)
        
        # Join all parts with double newlines to preserve paragraph structure
        full_text = "\n\n".join(text_content)
        
        log.info("text_extracted", char_count=len(full_text))
        return full_text

    except Exception as e:
        log.exception("extraction_failed", error=str(e))
        return ""

def list_epubs() -> List[str]:
    """Finds all EPUB files in the configured series directories.

    Scans the `DATA_DIR` for specific series folders and recursively finds
    all `.epub` files.

    Returns:
        List[str]: A sorted list of absolute paths to EPUB files.
    """
    log = logger.bind(task="list_epubs", data_dir=DATA_DIR)
    epub_files = []
    
    for series in SERIES_DIRS:
        series_path = os.path.join(DATA_DIR, series)
        
        # Check if directory exists
        if not os.path.exists(series_path):
            log.warning("series_dir_not_found", path=series_path)
            continue
            
        # Recursive search for .epub files using glob
        log.info("scanning_series", series=series)
        files = glob.glob(os.path.join(series_path, "**/*.epub"), recursive=True)
        
        # Sort files to ensure deterministic order (e.g., Book 1, Book 2)
        files.sort() 
        epub_files.extend(files)
    
    log.info("epubs_found", count=len(epub_files))
    return epub_files

def save_chunk(
    output_dir: str, 
    chunk_text: str, 
    books: List[str], 
    chunk_index: int, 
    config_name: str
) -> None:
    """Saves a text chunk and its corresponding metadata file.

    Args:
        output_dir (str): The directory where files will be saved.
        chunk_text (str): The text content of the chunk.
        books (List[str]): List of book file paths included in this chunk.
        chunk_index (int): The sequential index of the chunk.
        config_name (str): The name of the model config (e.g., 'claude').
    """
    # Bind context for logging
    log = logger.bind(
        task="save_chunk", 
        output_dir=output_dir, 
        chunk_index=chunk_index,
        config=config_name
    )

    if not chunk_text.strip():
        log.warning("empty_chunk_skipped")
        return

    # Ensure output directory exists (create if not present)
    os.makedirs(output_dir, exist_ok=True)

    # Determine series name from the first book in the chunk for file naming
    # Heuristic: Use the parent folder name of the first book as the series name
    first_book_path = books[0]
    rel_path = os.path.relpath(first_book_path, DATA_DIR)
    series_name = rel_path.split(os.sep)[0].lower().replace(" ", "_")
    
    # Construct filenames
    base_filename = f"{series_name}_section_{chunk_index:02d}"
    txt_filename = f"{base_filename}.txt"
    meta_filename = f"{base_filename}.meta.json"
    
    txt_path = os.path.join(output_dir, txt_filename)
    meta_path = os.path.join(output_dir, meta_filename)
    
    # Calculate stats for metadata
    encoding = get_base_encoding()
    token_count = len(encoding.encode(chunk_text))
    char_count = len(chunk_text)
    
    # Write Text Chunk
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(chunk_text)
        
    # Prepare Metadata dictionary
    metadata = {
        "chunk_id": base_filename,
        "token_count": token_count,
        "character_count": char_count,
        "included_books": [os.path.basename(b) for b in books],
        "model_config": config_name
    }
    
    # Write Metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    log.info(
        "chunk_saved", 
        filename=base_filename, 
        tokens=token_count, 
        chars=char_count, 
        book_count=len(books)
    )

def process_chunks_for_model(
    epub_paths: List[str], 
    config_key: str, 
    config: Dict[str, Any]
) -> None:
    """Processes EPUBs into chunks based on a specific model's token limits.

    Iterates through all EPUBs, accumulates text, and splits it into chunks
    that fit within the 'safe_limit' (context window - safety margin).
    Handles cases where a single book exceeds the limit by splitting it.

    Args:
        epub_paths (List[str]): List of paths to EPUB files.
        config_key (str): Key for the model config (e.g., 'claude').
        config (Dict[str, Any]): Configuration dictionary containing 'context_window' etc.
    """
    limit = config['context_window']
    safe_limit = int(limit * (1 - SAFETY_MARGIN))
    output_dir = os.path.join(DATA_DIR, config['output_dir_name'])
    
    log = logger.bind(
        task="process_model", 
        model=config_key, 
        limit=limit, 
        safe_limit=safe_limit
    )
    
    log.info("processing_started")
    
    encoding = get_base_encoding()
    
    # State variables for chunk accumulation
    current_chunk_text = ""
    current_chunk_tokens = 0
    current_books = []
    chunk_counter = 1
    
    for epub_path in epub_paths:
        book_name = os.path.basename(epub_path)
        log.info("processing_book", book=book_name)
        
        # Extract Text
        text = extract_text_from_epub(epub_path)
        
        if not text:
            log.warning("skipping_empty_book", book=book_name)
            continue
            
        text_tokens = len(encoding.encode(text))
        
        # Check if the book itself is larger than the safe limit
        if text_tokens > safe_limit:
            log.info(
                "large_book_detected", 
                book=book_name, 
                tokens=text_tokens, 
                limit=safe_limit
            )
            
            # Case 1: The single book is too huge.
            # First, FLUSH any existing accumulated chunk to clear the buffer.
            if current_chunk_text:
                save_chunk(output_dir, current_chunk_text, current_books, chunk_counter, config_key)
                chunk_counter += 1
                current_chunk_text = ""
                current_chunk_tokens = 0
                current_books = []
            
            # Now split the massive book into sub-chunks.
            # We split by double newlines (paragraphs) to find safe boundaries.
            paragraphs = text.split("\n\n")
            temp_chunk = ""
            temp_count = 0
            
            for para in paragraphs:
                # Add 1 token approx for the newline/whitespace we stripped during split
                para_tokens = len(encoding.encode(para)) + 1 
                
                # Check if adding this paragraph exceeds the limit
                if temp_count + para_tokens > safe_limit:
                    # Save the current temporary chunk
                    save_chunk(output_dir, temp_chunk, [epub_path], chunk_counter, config_key)
                    chunk_counter += 1
                    
                    # Start new temporary chunk with current paragraph
                    temp_chunk = para + "\n\n"
                    temp_count = para_tokens
                else:
                    # Accumulate
                    temp_chunk += para + "\n\n"
                    temp_count += para_tokens
            
            # Flush remainder of the large book
            if temp_chunk:
                save_chunk(output_dir, temp_chunk, [epub_path], chunk_counter, config_key)
                chunk_counter += 1
            
            # Continue to next book after handling the large one
            continue 
            
        # Case 2: Book fits. Check if it fits in CURRENT accumulated chunk
        if current_chunk_tokens + text_tokens > safe_limit:
            # Full, flush current chunk
            save_chunk(output_dir, current_chunk_text, current_books, chunk_counter, config_key)
            chunk_counter += 1
            
            # Reset with new book as the start of the next chunk
            current_chunk_text = text
            current_chunk_tokens = text_tokens
            current_books = [epub_path]
        else:
            # Add to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n" + text 
            else:
                current_chunk_text = text
            
            current_chunk_tokens += text_tokens
            current_books.append(epub_path)
            
    # Flush final accumulated chunk if any remains
    if current_chunk_text:
        save_chunk(output_dir, current_chunk_text, current_books, chunk_counter, config_key)

    log.info("processing_complete")

def main() -> None:
    """Main execution function."""
    log = logger.bind(task="main_execution")
    log.info("script_started")
    
    # 1. Identify all EPUBs
    epub_paths = list_epubs()
    if not epub_paths:
        log.error("no_epubs_found", hint="Check data directory structure")
        return
        
    # 2. Process for each model configuration
    for key, config in MODEL_CONFIGS.items():
        process_chunks_for_model(epub_paths, key, config)
        
    log.info("script_finished")

if __name__ == "__main__":
    main()

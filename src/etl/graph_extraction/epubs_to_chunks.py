# Dependencies:
# pip install ebooklib beautifulsoup4 tiktoken structlog

"""EPUB Text Extraction and Chunking Module.

This script extracts text content from EPUB files and organizes it into chunks
tailored to specific context window sizes. It's model-agnostic - you specify
the context window and safety margin via CLI arguments.

Example usage:
    # Process wheel_of_time with 1M context window (Gemini)
    uv run python -m src.etl.graph_extraction.epubs_to_chunks \\
        --series wheel_of_time --context-window 1000000 --output-dir data/processed_books

    # Process harry_potter with 200k context window
    uv run python -m src.etl.graph_extraction.epubs_to_chunks \\
        --series harry_potter --context-window 200000 --safety-margin 0.15
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ebooklib
import tiktoken
from bs4 import BeautifulSoup
from ebooklib import epub

import structlog


# Configure structlog
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

# Constants
DATA_DIR = Path(os.path.abspath("data"))

# Series directory mapping
SERIES_DIR_MAP = {
    "harry_potter": "harry_potter",
    "wheel_of_time": "wheel_of_time",
    "song_of_ice_and_fire": "song_of_ice_and_fire",
    "asoiaf": "song_of_ice_and_fire",
}


def get_base_encoding() -> tiktoken.Encoding:
    """Retrieves the tokenizer encoding.

    Returns:
        The 'cl100k_base' encoding used for token counting.
    """
    return tiktoken.get_encoding("cl100k_base")


def extract_text_from_epub(epub_path: Path) -> str:
    """Extracts raw text content from a single EPUB file.

    This function reads the EPUB file using `ebooklib`, iterates through its
    document items, and uses `BeautifulSoup` to strip HTML tags, returning
    clean text.

    Args:
        epub_path: The path to the EPUB file.

    Returns:
        The extracted text content, separated by double newlines.
        Returns an empty string if extraction fails.
    """
    log = logger.bind(task="extract_text", file_path=str(epub_path))

    try:
        book = epub.read_epub(str(epub_path))
        text_content = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                if text:
                    text_content.append(text)

        full_text = "\n\n".join(text_content)
        log.info("text_extracted", char_count=len(full_text))
        return full_text

    except Exception as e:
        log.exception("extraction_failed", error=str(e))
        return ""


def list_epubs_for_series(series: str) -> list[Path]:
    """Finds all EPUB files for a specific series.

    Args:
        series: Series identifier (e.g., 'wheel_of_time', 'harry_potter').

    Returns:
        A sorted list of paths to EPUB files.

    Raises:
        ValueError: If series is not recognized.
        FileNotFoundError: If series directory doesn't exist.
    """
    log = logger.bind(task="list_epubs", series=series)

    # Map series name to directory
    dir_name = SERIES_DIR_MAP.get(series.lower())
    if not dir_name:
        raise ValueError(
            f"Unknown series: {series}. Valid options: {list(SERIES_DIR_MAP.keys())}"
        )

    series_path = DATA_DIR / dir_name

    if not series_path.exists():
        raise FileNotFoundError(f"Series directory not found: {series_path}")

    # Recursive search for .epub files
    files = sorted(series_path.glob("**/*.epub"))

    log.info("epubs_found", count=len(files), series=series)
    return files


def save_chunk(
    output_dir: Path,
    chunk_text: str,
    books: list[Path],
    chunk_index: int,
    series: str,
    context_window: int,
    safety_margin: float,
) -> dict[str, Any]:
    """Saves a text chunk and its corresponding metadata file.

    Args:
        output_dir: The directory where files will be saved.
        chunk_text: The text content of the chunk.
        books: List of book file paths included in this chunk.
        chunk_index: The sequential index of the chunk.
        series: Series identifier.
        context_window: Context window size used for chunking.
        safety_margin: Safety margin percentage used.

    Returns:
        Document structure that can be used for MongoDB insertion.
    """
    log = logger.bind(
        task="save_chunk",
        output_dir=str(output_dir),
        chunk_index=chunk_index,
        series=series,
    )

    if not chunk_text.strip():
        log.warning("empty_chunk_skipped")
        return {}

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct filenames
    base_filename = f"{series}_section_{chunk_index:02d}"
    txt_filename = f"{base_filename}.txt"
    meta_filename = f"{base_filename}.meta.json"

    txt_path = output_dir / txt_filename
    meta_path = output_dir / meta_filename

    # Calculate stats for metadata
    encoding = get_base_encoding()
    token_count = len(encoding.encode(chunk_text))
    char_count = len(chunk_text)

    # Write text chunk
    txt_path.write_text(chunk_text, encoding="utf-8")

    # Prepare document structure (MongoDB-ready)
    document = {
        "chunk_id": base_filename,
        "series": series,
        "text_content": chunk_text,
        "token_count": token_count,
        "character_count": char_count,
        "context_window_used": context_window,
        "safety_margin": safety_margin,
        "included_books": [b.name for b in books],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write metadata JSON (for file-based workflow)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)

    log.info(
        "chunk_saved",
        filename=base_filename,
        tokens=token_count,
        chars=char_count,
        book_count=len(books),
    )

    return document


def process_chunks(
    epub_paths: list[Path],
    series: str,
    context_window: int,
    safety_margin: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Processes EPUBs into chunks based on context window and safety margin.

    Iterates through all EPUBs, accumulates text, and splits it into chunks
    that fit within the safe limit (context window * (1 - safety margin)).
    Handles cases where a single book exceeds the limit by splitting it.

    Args:
        epub_paths: List of paths to EPUB files.
        series: Series identifier.
        context_window: Maximum context window size in tokens.
        safety_margin: Percentage of context window to reserve as buffer.
        output_dir: Directory for output files.

    Returns:
        List of document dictionaries suitable for MongoDB insertion.
    """
    safe_limit = int(context_window * (1 - safety_margin))

    log = logger.bind(
        task="process_chunks",
        series=series,
        context_window=context_window,
        safe_limit=safe_limit,
    )

    log.info("processing_started")

    encoding = get_base_encoding()
    documents = []

    # State variables for chunk accumulation
    current_chunk_text = ""
    current_chunk_tokens = 0
    current_books: list[Path] = []
    chunk_counter = 1

    for epub_path in epub_paths:
        book_name = epub_path.name
        log.info("processing_book", book=book_name)

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
                limit=safe_limit,
            )

            # Flush any existing accumulated chunk
            if current_chunk_text:
                doc = save_chunk(
                    output_dir,
                    current_chunk_text,
                    current_books,
                    chunk_counter,
                    series,
                    context_window,
                    safety_margin,
                )
                if doc:
                    documents.append(doc)
                chunk_counter += 1
                current_chunk_text = ""
                current_chunk_tokens = 0
                current_books = []

            # Split the large book into sub-chunks by paragraphs
            paragraphs = text.split("\n\n")
            temp_chunk = ""
            temp_count = 0

            for para in paragraphs:
                para_tokens = len(encoding.encode(para)) + 1

                if temp_count + para_tokens > safe_limit:
                    doc = save_chunk(
                        output_dir,
                        temp_chunk,
                        [epub_path],
                        chunk_counter,
                        series,
                        context_window,
                        safety_margin,
                    )
                    if doc:
                        documents.append(doc)
                    chunk_counter += 1
                    temp_chunk = para + "\n\n"
                    temp_count = para_tokens
                else:
                    temp_chunk += para + "\n\n"
                    temp_count += para_tokens

            # Flush remainder of the large book
            if temp_chunk:
                doc = save_chunk(
                    output_dir,
                    temp_chunk,
                    [epub_path],
                    chunk_counter,
                    series,
                    context_window,
                    safety_margin,
                )
                if doc:
                    documents.append(doc)
                chunk_counter += 1

            continue

        # Book fits - check if it fits in current accumulated chunk
        if current_chunk_tokens + text_tokens > safe_limit:
            # Flush current chunk
            doc = save_chunk(
                output_dir,
                current_chunk_text,
                current_books,
                chunk_counter,
                series,
                context_window,
                safety_margin,
            )
            if doc:
                documents.append(doc)
            chunk_counter += 1

            # Reset with new book
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

    # Flush final accumulated chunk
    if current_chunk_text:
        doc = save_chunk(
            output_dir,
            current_chunk_text,
            current_books,
            chunk_counter,
            series,
            context_window,
            safety_margin,
        )
        if doc:
            documents.append(doc)

    log.info("processing_complete", total_chunks=len(documents))
    return documents


def main() -> None:
    """Main execution function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract and chunk EPUB files for LLM processing.",
        epilog="""
Examples:
  # Process wheel_of_time with 1M context window (Gemini)
  uv run python -m src.etl.graph_extraction.epubs_to_chunks \\
      --series wheel_of_time --context-window 1000000

  # Process harry_potter with 200k context window and 15% safety margin
  uv run python -m src.etl.graph_extraction.epubs_to_chunks \\
      --series harry_potter --context-window 200000 --safety-margin 0.15

  # Custom output directory
  uv run python -m src.etl.graph_extraction.epubs_to_chunks \\
      --series asoiaf --context-window 500000 --output-dir data/custom_chunks
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--series",
        type=str,
        required=True,
        choices=list(SERIES_DIR_MAP.keys()),
        help="Series to process (e.g., wheel_of_time, harry_potter, song_of_ice_and_fire)",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        required=True,
        help="Context window size in tokens (e.g., 1000000 for 1M, 200000 for 200k)",
    )

    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.10,
        help="Safety margin as decimal (default: 0.10 = 10%%)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for chunks. Default: data/processed_books_{series}",
    )

    args = parser.parse_args()

    log = logger.bind(task="main_execution")
    log.info(
        "script_started",
        series=args.series,
        context_window=args.context_window,
        safety_margin=args.safety_margin,
    )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / f"processed_books_{args.series}"

    # Find EPUBs for the series
    try:
        epub_paths = list_epubs_for_series(args.series)
    except (ValueError, FileNotFoundError) as e:
        log.error("epub_discovery_failed", error=str(e))
        return

    if not epub_paths:
        log.error("no_epubs_found", series=args.series)
        return

    # Process chunks
    documents = process_chunks(
        epub_paths=epub_paths,
        series=args.series,
        context_window=args.context_window,
        safety_margin=args.safety_margin,
        output_dir=output_dir,
    )

    log.info(
        "script_finished",
        total_chunks=len(documents),
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()

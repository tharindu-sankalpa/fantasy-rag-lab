"""EPUB Text Extraction and Processing Pipeline.

This module provides an ETL (Extract, Transform, Load) pipeline for processing
EPUB files. It extracts raw text from EPUB documents, cleans the content by removing
HTML tags and scripts, and groups the text into large blocks suitable for knowledge
graph construction.

Dependencies:
    - ebooklib: For reading EPUB files.
    - beautifulsoup4: For parsing HTML and extracting text.
    - structlog: For structured logging.
    - tqdm: For progress bars (imported but used implicitly via logger or extensions).
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import structlog
from tqdm import tqdm

# Configure logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress ebooklib warnings to keep console output clean
warnings.filterwarnings('ignore', category=UserWarning, module='ebooklib')
warnings.filterwarnings('ignore', category=FutureWarning, module='ebooklib')


class EpubProcessor:
    """Extracts text from EPUBs, cleans it, and groups it into large text blocks.

    This class handles the core logic of reading an EPUB file, iterating through
    its chapters, stripping HTML tags, and saving the resulting text with
    associated metadata.

    Attributes:
        data_dir (Path): The root directory containing raw EPUB files.
        output_dir (Path): The directory where processed text files will be saved.
        target_context_window (int): Maximum context window size in tokens for the target LLM.
        max_content_tokens (int): Maximum tokens for content (reserves 20% for prompt/schema).
    """

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "data/processed_books",
        target_context_window: int = 1_000_000
    ):
        """Initializes the EpubProcessor.

        Args:
            data_dir: Path to the directory containing input EPUB files.
                Defaults to "data".
            output_dir: Path to the directory where output text files will be stored.
                Defaults to "data/processed_books".
            target_context_window: Maximum context window size in tokens for the target LLM.
                Defaults to 1,000,000 (Gemini 1.5 Pro). Use 200,000 for Claude models,
                400,000 for GPT-5.2, etc.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store context window configuration
        self.target_context_window = target_context_window
        # Reserve ~20% for prompt, schema, and response buffer
        self.max_content_tokens = int(target_context_window * 0.80)

    def extract_text_from_epub(self, epub_path: Path) -> str:
        """Reads an EPUB file and extracts clean text from all chapters.

        This method iterates through the document items of an EPUB, strips
        HTML tags, scripts, and styles, and performs basic whitespace cleaning.
        It adds visual separators between the book header and the content.

        Args:
            epub_path: The file path to the target EPUB file.

        Returns:
            A string containing the concatenated, cleaned text of the entire book,
            preceded by a visual header.

        Raises:
            Exception: If the EPUB file cannot be read or processed, the error
                is logged and re-raised.
        """
        logger.info("extracting_epub", file=epub_path.name)
        try:
            # Load the EPUB file into memory. The 'epub' library handles the unzipping
            # and parsing of the .epub container (which is essentially a ZIP of HTML/XML files).
            book = epub.read_epub(str(epub_path))
            full_text = []

            # EPUBs are structured as a collection of resources (images, styles, texts).
            # We specifically request ITEM_DOCUMENT items, which correspond to the actual
            # XHTML/HTML files containing the book's chapters and narrative content.
            # Note: We iterate through ALL documents found in the package rather than strictly
            # following the 'spine' (linear reading order) defined in the OPF file.
            # While the spine defines the intended order, some malformed EPUBs might
            # miss content if we only rely on the spine. Processing all documents ensures coverage.
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                
                # Extract the raw bytes content from the item (chapter/document)
                content = item.get_content()
                
                # Use BeautifulSoup to parse the HTML content. 
                # This allows us to traverse the DOM and extract text while ignoring tags.
                soup = BeautifulSoup(content, 'html.parser')

                # Clean the DOM: Remove non-narrative elements.
                # - script/style: Executable code or CSS, irrelevant for text analysis.
                # - head/title/meta: Page metadata, duplicates of book info, or browser instructions.
                for element in soup(["script", "style", "head", "title", "meta"]):
                    element.extract()

                # Extract the raw text from the remaining HTML.
                # separator='\n\n' preserves paragraph breaks (e.g. <p> tags) as double newlines,
                # which helps the chunker later distinguish between paragraphs.
                text = soup.get_text(separator='\n\n')

                # Text Normalization & Cleaning:
                # 1. splitlines(): Break into individual lines to handle OS-specific line endings.
                # 2. line.strip(): Remove leading/trailing whitespace from each line.
                # 3. split("  "): Heuristic to break apart words/sentences accidentally glued together
                #    by excessive internal spacing (common in some OCR'd or converted ebooks).
                # 4. phrase.strip(): Clean the resulting chunks.
                # 5. if chunk: Filter out empty strings, effectively removing blank lines.
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                
                # Reassemble the text into a single clean block for this partial document.
                text = '\n'.join(chunk for chunk in chunks if chunk)

                # Heuristic Filtering:
                # If the resulting text is very short (> 50 chars), it's likely a page number,
                # a chapter title repeated, an image caption, or a blank page artifact.
                # We skip these to reduce noise in the knowledge graph.
                if len(text) > 50:
                    full_text.append(text)

            # Assemble the final artifact:
            # 1. Add a massive separator line to distinguish this book from others in a combined corpus.
            # 2. Add the book metadata (filename) clearly.
            # 3. Join all chapter texts with double newlines to maintain paragraph separation.
            return "\n\n" + ("=" * 50) + f"\nBOOK START: {epub_path.name}\n" + ("=" * 50) + "\n\n" + "\n\n".join(full_text)

        except Exception as e:
            # Log the full error context before re-raising to make debugging bulk ingestions easier.
            logger.error("failed_to_extract_epub", file=epub_path.name, error=str(e))
            raise e

    def estimate_tokens(self, text: str) -> int:
        """Estimates the number of tokens in a text string.

        Uses a simple heuristic where 1 token is approximately 4 characters.

        Args:
            text: The input text string.

        Returns:
            An integer representing the estimated token count.
        """
        return len(text) // 4

    def save_section(self, series_name: str, section_idx: int, text: str, book_names: List[str]):
        """Saves a processed section of text to a file with metadata.

        Creates two files:
        1. A .txt file containing the actual content.
        2. A .meta.json file containing metadata (source books, token count, etc.).

        Args:
            series_name: The name of the book series (e.g., "harry_potter").
            section_idx: The sequential index of this section (e.g., 1).
            text: The combined text content to save.
            book_names: A list of filenames included in this section.
        """
        token_count = self.estimate_tokens(text)
        filename = f"{series_name}_section_{section_idx:02d}.txt"
        output_path = self.output_dir / filename

        metadata = {
            "series": series_name,
            "section": section_idx,
            "books": book_names,
            "estimated_tokens": token_count,
            "char_count": len(text)
        }

        logger.info("saving_section", filename=filename, tokens=token_count, books=book_names)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Save metadata side-by-side
        with open(output_path.with_suffix('.meta.json'), "w") as f:
            json.dump(metadata, f, indent=2)

    def process_series(self, series_name: str, groups: List[List[str]], directory: Path, dry_run: bool = False):
        """Processes a complete series defined by groups of filenames.

        Iterates through the defined groups, extracts text from the books in each group,
        combines them, and saves the result.

        Args:
            series_name: The identifier for the series.
            groups: A list of lists, where each inner list contains the filenames
                belonging to a single output section.
            directory: The directory path where the source EPUBs are located.
            dry_run: If True, processes only the first book of the first group
                for testing purposes.
        """
        logger.info("processing_series", series=series_name, dry_run=dry_run)

        for idx, group_files in enumerate(groups, 1):
            if dry_run and idx > 1:
                break  # Only process first group in dry run

            section_text = []

            # In dry run, only process the first book of the first group
            files_to_process = group_files[:1] if dry_run else group_files

            for filename in files_to_process:
                file_path = directory / filename
                if not file_path.exists():
                    logger.warning("file_not_found", path=str(file_path))
                    continue

                text = self.extract_text_from_epub(file_path)
                section_text.append(text)

            combined_text = "\n".join(section_text)
            self.save_section(series_name, idx, combined_text, files_to_process)

            if dry_run:
                logger.info("dry_run_complete_for_series", series=series_name)
                break


def get_series_definitions(
    data_root: Path,
    max_tokens: int = 1_000_000
) -> Dict[str, Tuple[Path, List[List[str]]]]:
    """Defines the file groups and directory structures for specific book series.

    This configuration function maps series names to their directory paths and
    logical groupings of books. Groupings are created using a sliding window
    strategy with 1-book overlap, respecting the specified token budget.

    Args:
        data_root: The root path where series subdirectories are located.
        max_tokens: Maximum token budget per section (default: 1,000,000 for Gemini).
            Use 200,000 for Claude models, 400,000 for GPT-5.2, etc.
            This determines how many books can be grouped together.

    Returns:
        A dictionary where:
        - Keys are series names (str).
        - Values are tuples containing:
            1. The Path to the series directory.
            2. A List of Lists of strings (the filename groups).

    Note:
        The grouping strategy adapts to the specified max_tokens:
        - Larger context windows (1M): Can pair multiple books together
        - Smaller context windows (200K): May require individual book processing
    """
    

    # Pre-calculated token estimates to avoid expensive re-computation
    # Keys are partial filenames to ensure robust matching
    BOOK_TOKEN_ESTIMATES = {
        'Harry Potter 1': 109449, 'Harry Potter 2': 122812, 'Harry Potter 3': 155791, 
        'Harry Potter 4': 274449, 'Harry Potter 5': 371076, 'Harry Potter 6': 244530, 'Harry Potter 7': 284043,
        'Fire _ Blood': 361274, 'Rise of the Dragon': 120038, 'Knight of the Seven': 139967,
        'Game Of Thrones': 401146, 'Clash of Kings': 440695, 'Storm of Swords': 573252,
        'Feast For Crows': 427198, 'Dance With Dragons': 578037,
        'New Spring': 170245, 'Eye of the World': 425298, 'Great Hunt': 363711, 'Dragon Reborn': 340677,
        'Shadow Rising': 539117, 'Fires of Heaven': 484025, 'Lord of Chaos': 555478, 'Crown of Swords': 409717,
        'Path of Daggers': 318535, 'Winter_s Heart': 336712, 'Crossroads of Twilight': 378022,
        'Knife of Dreams': 449528, 'Gathering Storm': 427662, 'Towers of Midnight': 468399,
        'Memory of Light': 497967, 'Wheel of Time Companion': 551940
    }

    def get_token_estimate(filename: str) -> int:
        for key, tokens in BOOK_TOKEN_ESTIMATES.items():
            if key in filename:
                return tokens
        return 0 # Should not happen with correct keys

    def create_safe_sliding_windows(
        books: List[str],
        max_tokens: int = 1_000_000
    ) -> List[List[str]]:
        """
        Creates sliding window groups of books based on token budget.

        Groups consecutive books together while respecting the maximum token limit.
        Uses a sliding window with 1-book overlap strategy to maintain narrative continuity.

        Args:
            books: List of book filenames to group.
            max_tokens: Maximum token budget per group (default: 1,000,000 for Gemini).
                Use 200,000 for Claude models, 400,000 for GPT-5.2, etc.

        Returns:
            List of lists, where each inner list contains book filenames that fit
            within the token budget.

        Strategy:
            - Pairs consecutive books (book1+book2, book2+book3, etc.)
            - If a pair fits under max_tokens, include it
            - If a pair exceeds the limit, break it into singletons
            - Ensures every book appears in at least one section
        """
        if not books:
            return []
        if len(books) == 1:
            return [books]

        groups = []
        pairs = []

        # Generate candidate pairs (sliding window with 1-book overlap)
        for i in range(len(books) - 1):
            pairs.append((books[i], books[i+1]))

        # Evaluate each pair against the token budget
        for i, (b1, b2) in enumerate(pairs):
            s1 = get_token_estimate(b1)
            s2 = get_token_estimate(b2)
            total_tokens = s1 + s2

            # Check if the pair fits within the configured max_tokens limit
            if total_tokens < max_tokens:
                groups.append([b1, b2])
            else:
                # Pair exceeds limit.
                # Strategy: Break the link and handle books as singletons.

                # If this is the very first pair and it failed, we must ensure b1 is covered.
                if i == 0:
                    groups.append([b1])

                # Now decide for b2.
                # If the NEXT pair is safe, b2 will be covered there (as the first element of next pair).
                # If the next pair is NOT safe (or doesn't exist), we must cover b2 here as a singleton.
                next_is_safe = False
                if i + 1 < len(pairs):
                    nb1, nb2 = pairs[i+1]
                    ns1 = get_token_estimate(nb1)
                    ns2 = get_token_estimate(nb2)
                    # Check next pair against max_tokens limit
                    if ns1 + ns2 < max_tokens:
                        next_is_safe = True

                # If next pair is unsafe or doesn't exist, add b2 as a singleton
                if not next_is_safe:
                    groups.append([b2])

        return groups

    # Harry Potter
    hp_dir = data_root / "harry_potter"
    hp_books = [
        "Harry Potter 1 - Harry Potter and the Sorcerer_s Stone - J. K. Rowling _ Mary Grandpre.epub",
        "Harry Potter 2 - Harry Potter and the Chamber of Secrets - J. K. Rowling _ Mary Grandpre.epub",
        "Harry Potter 3 - Harry Potter and the Prisoner of Azkaban - J. K. Rowling _ Mary Grandpre.epub",
        "Harry Potter 4 - Harry Potter and the Goblet of Fire - J. K. Rowling _ Mary Grandpre.epub",
        "Harry Potter 5 - Harry Potter and the Order of the Phoenix - J. K. Rowling _ Mary Grandpre.epub",
        "Harry Potter 6 - Harry Potter and the Half-Blood Prince - J. K. Rowling _ Mary Grandpre.epub",
        "Harry Potter 7 - Harry Potter and the Deathly Hallows - J. K. Rowling _ Mary Grandpre.epub",
    ]
    # Apply the token budget to generate appropriate groupings for each series
    hp_groups = create_safe_sliding_windows(hp_books, max_tokens)

    # ASOIAF (A Song of Ice and Fire) - Timeline Order
    asoiaf_dir = data_root / "song_of_ice_and_fire"
    asoiaf_books = [
        "Fire _ Blood by George R.r. Martin.epub",
        "The Rise of the Dragon - An Illustrated History of the Targaryen Dynasty, Volume One (US Edition).epub",
        "A Knight of the Seven Kingdoms.epub",
        "A Game Of Thrones - George RR Martin.epub",
        "A Clash of Kings - George RR Martin.epub",
        "A Storm of Swords - George RR Martin.epub",
        "A Feast For Crows - George RR Martin.epub",
        "A Dance With Dragons - George RR Martin.epub",
    ]
    asoiaf_groups = create_safe_sliding_windows(asoiaf_books, max_tokens)

    # Wheel of Time
    wot_dir = data_root / "wheel_of_time"
    wot_books = [
        "00. New Spring.epub",
        "01. The Eye of the World.epub",
        "02. The Great Hunt.epub",
        "03. The Dragon Reborn.epub",
        "04. The Shadow Rising.epub",
        "05. The Fires of Heaven.epub",
        "06. Lord of Chaos.epub",
        "07. A Crown of Swords.epub",
        "08. The Path of Daggers.epub",
        "09. Winter_s Heart.epub",
        "10. Crossroads of Twilight.epub",
        "11. Knife of Dreams.epub",
        "12. The Gathering Storm.epub",
        "13. Towers of Midnight.epub",
        "14. A Memory of Light.epub",
        "The Wheel of Time Companion _ The People, Places and History of the Bestselling Series.epub"
    ]
    wot_groups = create_safe_sliding_windows(wot_books, max_tokens)

    return {
        "harry_potter": (hp_dir, hp_groups),
        "asoiaf": (asoiaf_dir, asoiaf_groups),
        "wheel_of_time": (wot_dir, wot_groups)
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process EPUB series into text datasets with model-aware chunking.",
        epilog="""
Examples:
  # Process for Gemini (1M context window) - default
  python extract_epubs.py

  # Process for Claude models (200k context window)
  python extract_epubs.py --context-window 200000 --output-dir data/processed_books_claude_200k

  # Process for GPT-5.2 (400k context window)
  python extract_epubs.py --context-window 400000 --output-dir data/processed_books_gpt_400k

  # Dry run for testing
  python extract_epubs.py --dry-run --context-window 200000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first book of each series to verify configuration."
    )

    parser.add_argument(
        "--series",
        type=str,
        help="Specific series to process (harry_potter, asoiaf, wheel_of_time). If not specified, processes all series."
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=1_000_000,
        help="Maximum context window size in tokens for the target LLM. "
             "Examples: 1000000 (Gemini), 200000 (Claude Opus/Sonnet 4.5), 400000 (GPT-5.2). "
             "Default: 1000000"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_books",
        help="Directory to save processed text files. "
             "Use descriptive names like 'data/processed_books_claude_200k' or 'data/processed_books_gpt_400k' "
             "to distinguish different chunking strategies. "
             "Default: data/processed_books"
    )

    args = parser.parse_args()

    # Log the configuration for transparency
    logger.info(
        "epub_processor_initialized",
        context_window=args.context_window,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        series_filter=args.series or "all"
    )

    # Initialize processor with configured context window
    processor = EpubProcessor(
        output_dir=args.output_dir,
        target_context_window=args.context_window
    )

    # Get series definitions using the configured max token budget
    definitions = get_series_definitions(
        processor.data_dir,
        max_tokens=processor.max_content_tokens
    )

    # Process each series
    for series_name, (dir_path, groups) in definitions.items():
        if args.series and args.series != series_name:
            continue

        logger.info(
            "starting_series",
            series=series_name,
            num_groups=len(groups),
            context_window=args.context_window
        )
        processor.process_series(series_name, groups, dir_path, dry_run=args.dry_run)

    logger.info("epub_processing_complete", output_dir=args.output_dir)
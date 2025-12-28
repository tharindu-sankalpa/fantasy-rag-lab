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
    """

    def __init__(self, data_dir: str = "data", output_dir: str = "data/processed_books"):
        """Initializes the EpubProcessor.

        Args:
            data_dir: Path to the directory containing input EPUB files. 
                Defaults to "data".
            output_dir: Path to the directory where output text files will be stored.
                Defaults to "data/processed_books".
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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


def get_series_definitions(data_root: Path) -> Dict[str, Tuple[Path, List[List[str]]]]:
    """Defines the file groups and directory structures for specific book series.

    This configuration function maps series names to their directory paths and
    logical groupings of books. Groupings may overlap to provide context continuity
    for machine learning models.

    Args:
        data_root: The root path where series subdirectories are located.

    Returns:
        A dictionary where:
        - Keys are series names (str).
        - Values are tuples containing:
            1. The Path to the series directory.
            2. A List of Lists of strings (the filename groups).
    """
    # Harry Potter
    hp_dir = data_root / "harry_potter"
    hp_groups = [
        # Section 1: Books 1-4
        [
            "Harry Potter 1 - Harry Potter and the Sorcerer_s Stone - J. K. Rowling _ Mary Grandpre.epub",
            "Harry Potter 2 - Harry Potter and the Chamber of Secrets - J. K. Rowling _ Mary Grandpre.epub",
            "Harry Potter 3 - Harry Potter and the Prisoner of Azkaban - J. K. Rowling _ Mary Grandpre.epub",
            "Harry Potter 4 - Harry Potter and the Goblet of Fire - J. K. Rowling _ Mary Grandpre.epub",
        ],
        # Section 2: Books 4-7 (Overlap Book 4)
        [
            "Harry Potter 4 - Harry Potter and the Goblet of Fire - J. K. Rowling _ Mary Grandpre.epub",
            "Harry Potter 5 - Harry Potter and the Order of the Phoenix - J. K. Rowling _ Mary Grandpre.epub",
            "Harry Potter 6 - Harry Potter and the Half-Blood Prince - J. K. Rowling _ Mary Grandpre.epub",
            "Harry Potter 7 - Harry Potter and the Deathly Hallows - J. K. Rowling _ Mary Grandpre.epub",
        ]
    ]

    # ASOIAF (A Song of Ice and Fire) - Timeline Order
    asoiaf_dir = data_root / "song_of_ice_and_fire"
    asoiaf_groups = [
        # Section 1: Pre-Game of Thrones (History & Prequels)
        [
            "Fire _ Blood by George R.r. Martin.epub",
            "The Rise of the Dragon - An Illustrated History of the Targaryen Dynasty, Volume One (US Edition).epub",
            "A Knight of the Seven Kingdoms.epub",
        ],
        # Section 2: Main Series Start (overlap with Knight)
        [
            "A Knight of the Seven Kingdoms.epub",
            "A Game Of Thrones - George RR Martin.epub",
            "A Clash of Kings - George RR Martin.epub",
        ],
         # Section 3: Mid Series
        [
            "A Clash of Kings - George RR Martin.epub",
            "A Storm of Swords - George RR Martin.epub",
            "A Feast For Crows - George RR Martin.epub",
        ],
        # Section 4: Current End
        [
            "A Feast For Crows - George RR Martin.epub",
            "A Dance With Dragons - George RR Martin.epub",
        ]
    ]

    # Wheel of Time
    wot_dir = data_root / "wheel_of_time"
    wot_groups = [
        # Section 1: Books 0-3 (Including New Spring)
        [
            "00. New Spring.epub",
            "01. The Eye of the World.epub",
            "02. The Great Hunt.epub",
            "03. The Dragon Reborn.epub",
        ],
        # Section 2: Books 3-6
        [
            "03. The Dragon Reborn.epub",
            "04. The Shadow Rising.epub",
            "05. The Fires of Heaven.epub",
            "06. Lord of Chaos.epub",
        ],
        # Section 3: Books 6-9
        [
            "06. Lord of Chaos.epub",
            "07. A Crown of Swords.epub",
            "08. The Path of Daggers.epub",
            "09. Winter_s Heart.epub",
        ],
        # Section 4: Books 9-12
        [
            "09. Winter_s Heart.epub",
            "10. Crossroads of Twilight.epub",
            "11. Knife of Dreams.epub",
            "12. The Gathering Storm.epub",
        ],
        # Section 5: Books 12-14 + Companion
        [
            "12. The Gathering Storm.epub",
            "13. Towers of Midnight.epub",
            "14. A Memory of Light.epub",
            "The Wheel of Time Companion _ The People, Places and History of the Bestselling Series.epub"
        ]
    ]

    return {
        "harry_potter": (hp_dir, hp_groups),
        "asoiaf": (asoiaf_dir, asoiaf_groups),
        "wheel_of_time": (wot_dir, wot_groups)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process EPUB series into text datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Process only the first book of each series to verify.")
    parser.add_argument("--series", type=str, help="Specific series to process (optional).")
    args = parser.parse_args()

    processor = EpubProcessor()
    definitions = get_series_definitions(processor.data_dir)

    for series_name, (dir_path, groups) in definitions.items():
        if args.series and args.series != series_name:
            continue

        logger.info("starting_series", series=series_name)
        processor.process_series(series_name, groups, dir_path, dry_run=args.dry_run)
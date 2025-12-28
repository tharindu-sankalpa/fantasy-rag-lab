# dependencies: ebooklib, beautifulsoup4, structlog, tqdm

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

# Suppress ebooklib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ebooklib')
warnings.filterwarnings('ignore', category=FutureWarning, module='ebooklib')

class EpubProcessor:
    """
    Extracts text from EPUBs, cleans it, and groups it into large text blocks.
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "data/processed_books"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_epub(self, epub_path: Path) -> str:
        """
        Reads an EPUB file and extracts clean text from all distinct chapters/documents.
        """
        logger.info("extracting_epub", file=epub_path.name)
        try:
            book = epub.read_epub(str(epub_path))
            full_text = []
            
            # Iterate through items in spine order for narrative continuity
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                # We could filter by spine, but getting all documents is usually safer for full content
                # check if item is in spine
                # if item.get_name() not in [x[0] for x in book.spine]: continue
                
                content = item.get_content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "head", "title", "meta"]):
                    script.extract()
                
                # Get text
                text = soup.get_text(separator='\n\n')
                
                # Basic cleaning
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                if len(text) > 50: # Skip empty or tiny sections (often just page numbers or artifacts)
                    full_text.append(text)
                    
            return "\n\n" + ("="*50) + f"\nBOOK START: {epub_path.name}\n" + ("="*50) + "\n\n" + "\n\n".join(full_text)
            
        except Exception as e:
            logger.error("failed_to_extract_epub", file=epub_path.name, error=str(e))
            raise e

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def save_section(self, series_name: str, section_idx: int, text: str, book_names: List[str]):
        """Saves a processed section to a text file."""
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
        """
        Process a series defined by groups of filenames.
        
        groups: List of Lists, where each inner list contains filenames for one section.
        """
        logger.info("processing_series", series=series_name, dry_run=dry_run)
        
        for idx, group_files in enumerate(groups, 1):
            if dry_run and idx > 1:
                break # Only process first group in dry run
                
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

def get_series_definitions(data_root: Path) -> Dict:
    """
    Returns the file groups for each series.
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

    # ASOIAF
    asoiaf_dir = data_root / "song_of_ice_and_fire"
    asoiaf_groups = [
        # Section 1: Books 1-3
        [
            "A Game Of Thrones - George RR Martin.epub",
            "A Clash of Kings - George RR Martin.epub",
            "A Storm of Swords - George RR Martin.epub",
        ],
        # Section 2: Books 3-5 (Overlap 3)
        [
            "A Storm of Swords - George RR Martin.epub",
            "A Feast For Crows - George RR Martin.epub",
            "A Dance With Dragons - George RR Martin.epub",
        ],
        # Section 3: Books 5-8 (Overlap 5) + Companions
        [
            "A Dance With Dragons - George RR Martin.epub",
            "A Knight of the Seven Kingdoms.epub",
            "Fire _ Blood by George R.r. Martin.epub",
            "The Rise of the Dragon - An Illustrated History of the Targaryen Dynasty, Volume One (US Edition).epub"
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
    parser = argparse.ArgumentParser()
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

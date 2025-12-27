# Dependencies:
# pip install structlog langchain-community langchain-text-splitters langchain-core unstructured

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.logger import logger

@dataclass
class ChapterInfo:
    """
    Holds information about a chapter in the Wheel of Time series.

    Attributes:
        number (str): The chapter number, which can be a digit or special marker like "PROLOGUE".
        title (str): The chapter's title.
    """
    number: str
    title: str

class FantasyBookProcessor:
    """
    A comprehensive processor for fantasy book series that converts EPUB files
    into searchable text chunks with metadata.

    This class handles the entire pipeline of processing books:
    - Loading EPUB files
    - Splitting text into meaningful chunks
    - Tracking chapter information
    - Enriching chunks with metadata

    Attributes:
        universe (str): The name of the fantasy universe.
        chunk_size (int): Target size of text chunks.
        chunk_overlap (int): Overlap between chunks.
    """

    def __init__(self, universe: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the processor with customizable chunking parameters.

        Args:
            universe: The name of the fantasy universe (e.g., "Wheel of Time", "Harry Potter").
            chunk_size: Target size of text chunks in characters.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.universe = universe
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create a specialized text splitter optimized for the specific fantasy universe.

        Returns:
            RecursiveCharacterTextSplitter: A configured text splitter instance.
        """
        # Base separators common to all
        base_separators = [
            "\n\n",           # Paragraph breaks
            ". ",             # Sentence endings
            "! ",            # Exclamation endings
            "? ",            # Question endings
            '"\n',           # Dialogue line breaks
            '" ',            # Dialogue
            "\n",            # Basic line breaks
            " ",             # Word breaks
            ""               # Character-level fallback
        ]

        if self.universe.lower() == "wheel of time":
            # WoT specific logic (existing)
            section_patterns = [
                r'\nEPILOGUE\s*\n', r'\nepilogue\s*\n', r'\nEpilogue\s*\n',
                r'\nPROLOGUE\s*\n', r'\nprologue\s*\n', r'\nPrologue\s*\n',
                r'\nCHAPTER\s*\d*\s*\n', r'\nchapter\s*\d*\s*\n', r'\nChapter\s*\d*\s*\n',
            ]
            separators = section_patterns + ["\n*   *   *\n"] + base_separators

        elif self.universe.lower() == "harry potter":
             # HP Format: "CHAPTER ONE" or "Chapter 1"
            section_patterns = [
                 r'\nCHAPTER\s+[A-Za-z0-9]+\s*\n',
                 r'\nChapter\s+\d+\s*\n',
                 r'\nC\s*H\s*A\s*P\s*T\s*E\s*R\s+[A-Za-z0-9]+\s*\n' # Sometimes spaced out
            ]
            separators = section_patterns + base_separators
        
        elif self.universe.lower() == "song of ice and fire":
            # ASOIAF Format: Chapters are POV names (ARYA, BRAN, etc.)
            # This is harder to regex strictly without a list of names, but they are usually Headers on their own line.
            # We will use a more generic header detection for all caps lines which are likely headers.
            # And also specific known structure if possible.
            section_patterns = [
                r'\nPROLOGUE\s*\n',
                r'\nEPILOGUE\s*\n',
                r'\n[A-Z]{3,}\s*\n', # Heuristic: Lines with only uppercase words (minimum 3 chars) e.g. "ARYA"
            ]
            separators = section_patterns + base_separators
        
        else:
            # Generic default
            separators = base_separators

        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            is_separator_regex=True
        )

    def _extract_chapter_info(self, text: str) -> ChapterInfo:
        """
        Extract chapter info based on universe patterns.

        Args:
            text: The text content to analyze.

        Returns:
            ChapterInfo: An object containing extracted chapter number and title.
        """
        universe = self.universe.lower()
        
        if universe == "wheel of time":
            # Existing complex patterns for WoT
            patterns = [
                r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*\n+([^\n]+)',
                r'(?i)(?:CHAPTER|Chapter)\s*\n+(\d+)\s*\n+([^\n]+)',
                r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*\n+\s*\n+([^\n]+)',
                r'(?i)(?:CHAPTER|Chapter)\s*([A-Za-z0-9]+):\s*([^\n]+)',
                r'(?i)(?:CHAPTER|Chapter):\s*([^\n]+)',
                r'(?i)(?:PROLOGUE|Prologue|EPILOGUE|Epilogue)\s*\n+([^\n]+)',
                r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*$',
                r'(?i)(?:PROLOGUE|Prologue|EPILOGUE|Epilogue)\s*$'
            ]
            
        elif universe == "harry potter":
            # HP Patterns
            patterns = [
                r'(?i)CHAPTER\s+([A-Za-z0-9]+)\s*\n+([^\n]+)', # CHAPTER ONE \n The Boy Who Lived
                r'(?i)CHAPTER\s+([A-Za-z0-9]+)\s*$'
            ]
            
        elif universe == "song of ice and fire":
            # ASOIAF Patterns: POV Name is the chapter title. No numbers.
            # We look for lines that are just a name at the start of the chunk.
            patterns = [
                 r'^\s*(PROLOGUE|EPILOGUE)\s*\n',
                 r'^\s*([A-Z]+)\s*\n' # Uppercase name at start
            ]
        else:
            patterns = [r'(?i)(?:CHAPTER|Chapter)\s*(\d+)']

        # Generic Extraction Logic
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if universe == "song of ice and fire":
                    # Special case for ASOIAF where the match IS the title/chapter
                    return ChapterInfo(number="POV", title=groups[0].strip())
                
                if len(groups) >= 2:
                    return ChapterInfo(number=groups[0], title=groups[1].strip())
                elif len(groups) == 1:
                    if "PROLOGUE" in groups[0].upper() or "EPILOGUE" in groups[0].upper():
                         return ChapterInfo(number="N/A", title=groups[0].strip())
                    return ChapterInfo(number="N/A", title=groups[0].strip()) # Fallback

        return ChapterInfo(number="N/A", title="N/A")

    def process_epub(self, epub_path: str) -> List[Document]:
        """
        Process a single EPUB file into chunks with metadata.

        Args:
            epub_path: The absolute path to the EPUB file.

        Returns:
            List[Document]: A list of processed document chunks.
        """
        # Load the EPUB
        loader = UnstructuredEPubLoader(epub_path)
        data = loader.load()
        
        # Extract book name from path
        book_name = os.path.basename(epub_path).replace('.epub', '')
        
        # Create text splitter and split document
        text_splitter = self._create_text_splitter()
        chunks = text_splitter.split_documents(data)
        
        # Initialize chapter tracking
        current_chapter = ChapterInfo(number="N/A", title="N/A")
        
        # Process each chunk
        for chunk in chunks:
            # Check for new chapter information
            chapter_info = self._extract_chapter_info(chunk.page_content)
            
            # Logic to update current chapter:
            # If we explicitly found a new chapter header, update.
            # For ASOIAF, a POV name update is a new chapter.
            is_new_chapter = (chapter_info.number != "N/A" or chapter_info.title != "N/A")
            
            if is_new_chapter:
                current_chapter = chapter_info
            
            # Enrich chunk with metadata
            chunk.metadata.update({
                "universe": self.universe,
                "book_name": book_name,
                "chapter_number": current_chapter.number,
                "chapter_title": current_chapter.title
            })
        
        return chunks

    def process_series(self, epub_dir: str) -> List[Document]:
        """
        Process the entire fantasy series from a directory of EPUB files.

        This method processes all EPUB files in the specified directory, combining
        them into a single collection of chunks while maintaining proper book and
        chapter context for each chunk.

        Args:
            epub_dir: Directory containing EPUB files.

        Returns:
            List[Document]: List of all processed chunks across the entire series.
        """
        log = logger.bind(task="process_series", directory=epub_dir)
        all_chunks = []
        
        if not os.path.exists(epub_dir):
            log.error("directory_not_found")
            return []

        for filename in sorted(os.listdir(epub_dir)):
            if filename.endswith('.epub'):
                epub_path = os.path.join(epub_dir, filename)
                try:
                    chunks = self.process_epub(epub_path)
                    all_chunks.extend(chunks)
                    log.info("file_processed", filename=filename, chunks_created=len(chunks))
                except Exception as e:
                    log.exception("file_processing_failed", filename=filename)
        
        return all_chunks

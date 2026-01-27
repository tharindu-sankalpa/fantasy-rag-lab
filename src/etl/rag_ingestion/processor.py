# Dependencies:
# pip install structlog langchain-community langchain-text-splitters langchain-core unstructured

"""Fantasy book processor for RAG ingestion.

This module processes EPUB files from fantasy book series into searchable text
chunks with rich metadata including chapter information.
"""

import os
import re
from dataclasses import dataclass

from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import structlog


logger = structlog.get_logger(__name__)

# False positive patterns - matches that look like chapter titles but aren't
FALSE_POSITIVE_PATTERNS = [
    # Image file references (various formats)
    r"(?i)^image\s*$",
    r"(?i)^\d+\.(tiff|jpg|jpeg|png|gif|bmp)\s*$",
    r"(?i)^e\d+_i\d+\.(jpg|jpeg|png|gif|tiff|bmp)\s*$",  # e9781429960199_i0002.jpg
    r"(?i)^[a-z0-9_]+\.(jpg|jpeg|png|gif|tiff|bmp)\s*$",  # Any image filename
    r"(?i)^figure\s*\d*\s*$",
    r"(?i)^illustration\s*$",
    r"(?i)^map\s*$",
    # Book metadata / front/back matter
    r"(?i)^contents\s*$",
    r"(?i)^table\s+of\s+contents\s*$",
    r"(?i)^appendix\s*$",
    r"(?i)^acknowledgments?\s*$",
    r"(?i)^dedication\s*$",
    r"(?i)^about\s+the\s+author\s*$",
    r"(?i)^copyright\s*$",
    r"(?i)^cover\s*$",
    r"(?i)^title\s*page\s*$",
    # Publisher info
    r"(?i)^tor\s+books\s+by\s+",  # "Tor Books by Robert Jordan"
    r"(?i)^books\s+by\s+",
    r"(?i)^also\s+by\s+",
    r"(?i)^other\s+books\s+by\s+",
    # Preview/promotional content
    r"(?i)^a\s+preview\s+of\s+",  # "A preview of The Great Hunt"
    r"(?i)^preview\s*$",
    r"(?i)^excerpt\s+from\s+",
    r"(?i)^bonus\s+content\s*$",
    # Glossary and index
    r"(?i)^glossary\s*$",
    r"(?i)^index\s*$",
    r"(?i)^dramatis\s+personae\s*$",
    # Navigation artifacts
    r"(?i)^n/a\s*$",
]

# Known POV characters for A Song of Ice and Fire
ASOIAF_POV_CHARACTERS = frozenset([
    "PROLOGUE",
    "EPILOGUE",
    "ARYA",
    "BRAN",
    "CATELYN",
    "CERSEI",
    "DAENERYS",
    "DAVOS",
    "EDDARD",
    "JAIME",
    "JON",
    "SANSA",
    "THEON",
    "TYRION",
    "SAMWELL",
    "BRIENNE",
    "AERON",
    "AREO",
    "ARIANNE",
    "ASHA",
    "BARRISTAN",
    "CONNINGTON",
    "MELISANDRE",
    "QUENTYN",
    "VICTARION",
    # Also handle common variations
    "NED",  # Alias for Eddard
    "DANY",  # Alias for Daenerys
    "SAM",  # Alias for Samwell
])


def is_false_positive(text: str) -> bool:
    """Check if the extracted text is a false positive (not a real chapter title).

    Args:
        text: The candidate chapter title text.

    Returns:
        True if the text matches a known false positive pattern.
    """
    text = text.strip()
    for pattern in FALSE_POSITIVE_PATTERNS:
        if re.match(pattern, text):
            return True
    return False


def is_valid_asoiaf_pov(text: str) -> bool:
    """Check if the text is a valid ASOIAF POV character name.

    Args:
        text: The candidate POV character name.

    Returns:
        True if the text is a known POV character.
    """
    return text.strip().upper() in ASOIAF_POV_CHARACTERS


@dataclass
class ChapterInfo:
    """Holds information about a chapter.

    Attributes:
        number: The chapter number (digit or special marker like "PROLOGUE").
        title: The chapter's title.
    """

    number: str
    title: str


class FantasyBookProcessor:
    """Processor for fantasy book series that converts EPUB files into searchable chunks.

    This class handles the entire pipeline of processing books:
    - Loading EPUB files
    - Splitting text into meaningful chunks
    - Tracking chapter information
    - Enriching chunks with metadata

    Attributes:
        universe: The name of the fantasy universe.
        chunk_size: Target size of text chunks.
        chunk_overlap: Overlap between chunks.
    """

    def __init__(
        self, universe: str, chunk_size: int = 4000, chunk_overlap: int = 800
    ):
        """Initialize the processor with customizable chunking parameters.

        Args:
            universe: The name of the fantasy universe (e.g., "Wheel of Time").
            chunk_size: Target size of text chunks in characters.
                        Default 4000 chars ≈ 1000 tokens (1 token ≈ 4 chars).
            chunk_overlap: Number of characters to overlap between chunks.
                           Default 800 chars ≈ 200 tokens.
        """
        self.universe = universe
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a specialized text splitter optimized for the fantasy universe.

        Returns:
            A configured text splitter instance.
        """
        # Base separators common to all
        base_separators = [
            "\n\n",  # Paragraph breaks
            ". ",  # Sentence endings
            "! ",  # Exclamation endings
            "? ",  # Question endings
            '"\n',  # Dialogue line breaks
            '" ',  # Dialogue
            "\n",  # Basic line breaks
            " ",  # Word breaks
            "",  # Character-level fallback
        ]

        universe = self.universe.lower()

        if universe == "wheel of time":
            section_patterns = [
                r"\nEPILOGUE\s*\n",
                r"\nepilogue\s*\n",
                r"\nEpilogue\s*\n",
                r"\nPROLOGUE\s*\n",
                r"\nprologue\s*\n",
                r"\nPrologue\s*\n",
                r"\nCHAPTER\s*\d*\s*\n",
                r"\nchapter\s*\d*\s*\n",
                r"\nChapter\s*\d*\s*\n",
            ]
            separators = section_patterns + ["\n*   *   *\n"] + base_separators

        elif universe == "harry potter":
            section_patterns = [
                r"\nCHAPTER\s+[A-Za-z0-9]+\s*\n",
                r"\nChapter\s+\d+\s*\n",
                r"\nC\s*H\s*A\s*P\s*T\s*E\s*R\s+[A-Za-z0-9]+\s*\n",
            ]
            separators = section_patterns + base_separators

        elif universe == "song of ice and fire":
            # Build pattern that only matches known POV characters
            pov_pattern = "|".join(ASOIAF_POV_CHARACTERS)
            section_patterns = [
                r"\nPROLOGUE\s*\n",
                r"\nEPILOGUE\s*\n",
                rf"\n({pov_pattern})\s*\n",  # Only match known POV names
            ]
            separators = section_patterns + base_separators

        else:
            separators = base_separators

        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            is_separator_regex=True,
        )

    def _extract_chapter_info_wot(self, text: str) -> ChapterInfo:
        """Extract chapter info for Wheel of Time series.

        Uses patterns specifically designed for WoT formatting:
        - Standard format: "CHAPTER 8\\nWhirlpools of Color"
        - EPUB format with Image: "CHAPTER 8\\n\\nImage\\n\\nAn Empty Road"
        - Split format: "CHAPTER\\n39\\nThose Who Fight"
        - Special sections: "PROLOGUE\\nDragonmount"

        Args:
            text: The text content to analyze.

        Returns:
            ChapterInfo with chapter number and title.
        """
        # Patterns in order of specificity
        # Note: EPUB format often has "Image" text between chapter number and title
        chapter_patterns = [
            # EPUB format: "CHAPTER 8\n\nImage\n\nAn Empty Road"
            r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*\n+Image\s*\n+([^\n]+)',
            # PROLOGUE/EPILOGUE with Image: "PROLOGUE\n\nImage\n\nDragonmount"
            r'(?i)(PROLOGUE|EPILOGUE)\s*\n+Image\s*\n+([^\n]+)',
            # Standard format: "CHAPTER 8\nWhirlpools of Color"
            r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*\n+([^\n]+)',
            # Split format: "CHAPTER\n39\nThose Who Fight"
            r'(?i)(?:CHAPTER|Chapter)\s*\n+(\d+)\s*\n+([^\n]+)',
            # Multi-newline format: "CHAPTER 8\n\nWhirlpools of Color"
            r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*\n+\s*\n+([^\n]+)',
            # Written numbers: "CHAPTER EIGHT: The Title"
            r'(?i)(?:CHAPTER|Chapter)\s*([A-Za-z0-9]+):\s*([^\n]+)',
            # Simple title: "CHAPTER: The Title"
            r'(?i)(?:CHAPTER|Chapter):\s*([^\n]+)',
            # Special sections: "PROLOGUE\nThe Title" or "EPILOGUE\nThe Title"
            r'(?i)(?:PROLOGUE|Prologue|EPILOGUE|Epilogue)\s*\n+([^\n]+)',
            # Bare chapters: "CHAPTER 8"
            r'(?i)(?:CHAPTER|Chapter)\s*(\d+)\s*$',
            # Bare prologue/epilogue
            r'(?i)(?:PROLOGUE|Prologue|EPILOGUE|Epilogue)\s*$',
        ]

        for pattern in chapter_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()

                if len(groups) == 2:
                    first, second = groups[0].strip(), groups[1].strip()
                    # Filter out false positives for title
                    if is_false_positive(second):
                        continue

                    # PROLOGUE/EPILOGUE with title
                    if first.upper() in ("PROLOGUE", "EPILOGUE"):
                        return ChapterInfo(number=first.upper(), title=second)
                    # Chapter number with title
                    return ChapterInfo(number=str(first), title=second)

                elif len(groups) == 1:
                    val = groups[0].strip()
                    # Check if it's a number (chapter number without title)
                    if val.isdigit():
                        return ChapterInfo(number=val, title="N/A")
                    # Check if it's PROLOGUE/EPILOGUE
                    if val.upper() in ("PROLOGUE", "EPILOGUE"):
                        return ChapterInfo(number=val.upper(), title="N/A")
                    # Otherwise it's a title (from patterns like "CHAPTER: The Title")
                    if not is_false_positive(val):
                        return ChapterInfo(number="N/A", title=val)

                elif len(groups) == 0:
                    # Bare PROLOGUE or EPILOGUE match
                    matched_text = match.group(0).strip().upper()
                    if "PROLOGUE" in matched_text:
                        return ChapterInfo(number="PROLOGUE", title="N/A")
                    if "EPILOGUE" in matched_text:
                        return ChapterInfo(number="EPILOGUE", title="N/A")

        return ChapterInfo(number="N/A", title="N/A")

    def _extract_chapter_info_asoiaf(self, text: str) -> ChapterInfo:
        """Extract chapter info for A Song of Ice and Fire series.

        ASOIAF uses POV character names as chapter titles.

        Args:
            text: The text content to analyze.

        Returns:
            ChapterInfo with POV character as title.
        """
        # Build pattern that only matches known POV characters
        pov_pattern = "|".join(ASOIAF_POV_CHARACTERS)
        patterns = [
            rf"^\s*(PROLOGUE|EPILOGUE)\s*\n",
            rf"^\s*({pov_pattern})\s*\n",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate = match.group(1).strip().upper()
                if is_valid_asoiaf_pov(candidate):
                    return ChapterInfo(number="POV", title=candidate)

        return ChapterInfo(number="N/A", title="N/A")

    def _extract_chapter_info_hp(self, text: str) -> ChapterInfo:
        """Extract chapter info for Harry Potter series.

        Args:
            text: The text content to analyze.

        Returns:
            ChapterInfo with chapter number and title.
        """
        patterns = [
            # "CHAPTER ONE\nThe Boy Who Lived"
            r'(?i)CHAPTER\s+([A-Za-z0-9]+)\s*\n+([^\n]+)',
            # Just chapter number
            r'(?i)CHAPTER\s+([A-Za-z0-9]+)\s*$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    title = groups[1].strip()
                    if not is_false_positive(title):
                        return ChapterInfo(number=groups[0], title=title)
                elif len(groups) == 1:
                    return ChapterInfo(number=groups[0], title="N/A")

        return ChapterInfo(number="N/A", title="N/A")

    def _extract_chapter_info(self, text: str) -> ChapterInfo:
        """Extract chapter info based on universe-specific patterns.

        Delegates to the appropriate universe-specific extraction method.

        Args:
            text: The text content to analyze.

        Returns:
            An object containing extracted chapter number and title.
        """
        universe = self.universe.lower()

        if universe == "wheel of time":
            return self._extract_chapter_info_wot(text)
        elif universe == "song of ice and fire":
            return self._extract_chapter_info_asoiaf(text)
        elif universe == "harry potter":
            return self._extract_chapter_info_hp(text)
        else:
            # Generic fallback
            match = re.search(r'(?i)(?:CHAPTER|Chapter)\s*(\d+)', text)
            if match:
                return ChapterInfo(number=match.group(1), title="N/A")
            return ChapterInfo(number="N/A", title="N/A")

    def process_epub(self, epub_path: str) -> list[Document]:
        """Process a single EPUB file into chunks with metadata.

        Args:
            epub_path: The absolute path to the EPUB file.

        Returns:
            A list of processed document chunks.
        """
        # Load the EPUB
        loader = UnstructuredEPubLoader(epub_path)
        data = loader.load()

        # Extract book name from path
        book_name = os.path.basename(epub_path).replace(".epub", "")

        # Create text splitter and split document
        text_splitter = self._create_text_splitter()
        chunks = text_splitter.split_documents(data)

        # Initialize chapter tracking
        current_chapter = ChapterInfo(number="N/A", title="N/A")

        # Process each chunk
        for chunk in chunks:
            # Check for new chapter information
            chapter_info = self._extract_chapter_info(chunk.page_content)

            # Update current chapter if we found a new valid chapter header
            is_new_chapter = chapter_info.number != "N/A" or chapter_info.title != "N/A"

            if is_new_chapter:
                current_chapter = chapter_info

            # Enrich chunk with metadata
            chunk.metadata.update(
                {
                    "universe": self.universe,
                    "book_name": book_name,
                    "chapter_number": current_chapter.number,
                    "chapter_title": current_chapter.title,
                }
            )

        return chunks

    def process_series(self, epub_dir: str) -> list[Document]:
        """Process the entire fantasy series from a directory of EPUB files.

        This method processes all EPUB files in the specified directory, combining
        them into a single collection of chunks while maintaining proper book and
        chapter context for each chunk.

        Args:
            epub_dir: Directory containing EPUB files.

        Returns:
            List of all processed chunks across the entire series.
        """
        log = logger.bind(task="process_series", directory=epub_dir)
        all_chunks = []

        if not os.path.exists(epub_dir):
            log.error("directory_not_found")
            return []

        for filename in sorted(os.listdir(epub_dir)):
            if filename.endswith(".epub"):
                epub_path = os.path.join(epub_dir, filename)
                try:
                    chunks = self.process_epub(epub_path)
                    all_chunks.extend(chunks)
                    log.info(
                        "file_processed", filename=filename, chunks_created=len(chunks)
                    )
                except Exception as e:
                    log.exception("file_processing_failed", filename=filename)

        return all_chunks

"""
AI module for EvilFlowers WebDAV Book Import.

This module provides functionality for using OpenAI API to extract metadata from text files.
"""

import os
import re
import logging
import json
from typing import TypedDict, Optional, List, Dict, Any
import openai
from tqdm import tqdm
import tiktoken
from tiktoken import encoding_for_model

# Set up logging
logger = logging.getLogger(__name__)

# TypedDict for result
class BookMetadata(TypedDict):
    title: Optional[str]
    authors: List[str]
    publisher: Optional[str]
    year: Optional[str]
    isbn: Optional[str]
    doi: Optional[str]
    summary: str


class RegexExtractor:
    """Class for extracting metadata using regular expressions."""

    # Regex patterns for ISBN and DOI
    ISBN_PATTERN = re.compile(r"\b(?:ISBN(?:-13)?:?\s*)?((97[89][- ]?)?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?[\dXx])\b")
    DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)

    @classmethod
    def extract_isbn(cls, text: str) -> Optional[str]:
        """Extract ISBN from text using regex."""
        match = cls.ISBN_PATTERN.search(text)
        return match.group(1) if match else None

    @classmethod
    def extract_doi(cls, text: str) -> Optional[str]:
        """Extract DOI from text using regex."""
        match = cls.DOI_PATTERN.search(text)
        return match.group(0) if match else None


class TextChunker:
    """Class for chunking text into manageable pieces."""

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the text chunker.

        Args:
            model_name (str): The name of the model to use for token counting.
        """
        self.model_name = model_name
        self.encoder = encoding_for_model(model_name)

    def chunk_text(self, pages: Dict[int, str], max_tokens: int = 3500) -> List[str]:
        """
        Chunk text into manageable pieces based on token count.

        Args:
            pages (dict): Dictionary of page numbers to page content.
            max_tokens (int): Maximum number of tokens per chunk.

        Returns:
            List[str]: List of text chunks.
        """
        current, chunks = "", []
        sorted_pages = sorted(pages)
        total_pages = len(sorted_pages)

        # Create a progress bar for text chunking
        with tqdm(total=total_pages, desc="Chunking text", leave=False) as pbar:
            for page_num in sorted_pages:
                text = pages[page_num]
                if len(self.encoder.encode(current + text)) > max_tokens:
                    chunks.append(current)
                    current = ""
                current += text + "\n"
                pbar.update(1)

        if current:
            chunks.append(current)
        return chunks


class Summarizer:
    """Class for summarizing text using OpenAI API."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """
        Initialize the summarizer.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): The name of the model to use for summarization.
        """
        self.api_key = api_key
        self.model_name = model_name

    def summarize_chunk(self, chunk: str) -> str:
        """
        Summarize a single chunk of text.

        Args:
            chunk (str): Text chunk to summarize.

        Returns:
            str: Summarized text.
        """
        prompt = (
            "Summarize the following text in a concise paragraph, focusing on the main ideas and themes.\n\n"
            f"Text:\n\"\"\"\n{chunk}\n\"\"\"\n"
        )
        openai.api_key = self.api_key
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def final_summarize(self, text: str, target_words: int) -> str:
        """
        Perform a final summarization to reach the target word count.

        Args:
            text (str): Text to summarize.
            target_words (int): Target word count for the summary.

        Returns:
            str: Final summary with approximately the target word count.
        """
        prompt = (
            f"Summarize the following text in approximately {target_words} words, "
            f"focusing on the main ideas and themes.\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n"
        )
        openai.api_key = self.api_key
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def recursive_summarize(self, chunks: List[str], target_words: int = 120, max_depth: int = 3, current_depth: int = 0) -> str:
        """
        Recursively summarize text chunks until they can fit into a single prompt,
        then perform a final summarization to reach the target word count.

        Args:
            chunks (List[str]): List of text chunks to summarize.
            target_words (int): Target word count for the final summary.
            max_depth (int): Maximum recursion depth to prevent infinite recursion.
            current_depth (int): Current recursion depth.

        Returns:
            str: Final summary.
        """
        # Prevent infinite recursion
        if current_depth >= max_depth:
            logging.warning(f"Reached maximum recursion depth ({max_depth}). Returning current summary.")
            return "\n".join(chunks)

        # If we have only one chunk, we can proceed to final summarization
        if len(chunks) == 1:
            # Perform final summarization to reach target word count
            return self.final_summarize(chunks[0], target_words)

        # Create a progress bar for summarizing chunks
        with tqdm(total=len(chunks), desc="Summarizing chunks", leave=False) as pbar:
            summaries = []
            for chunk in chunks:
                summaries.append(self.summarize_chunk(chunk))
                pbar.update(1)

        # Combine summaries
        summary_text = "\n".join(summaries)

        # Check if we can fit this into a single prompt now
        if len(summaries) == 1:
            # We have a single summary, proceed to final summarization
            return self.final_summarize(summary_text, target_words)
        else:
            # Still need to reduce further, chunk again
            chunk_size = max(1, len(summaries) // 3)
            next_chunks = [
                "\n".join(summaries[i:i+chunk_size])
                for i in range(0, len(summaries), chunk_size)
            ]

            # Check if we're making progress in reducing the text
            if len(next_chunks) >= len(chunks):
                logging.warning("Summarization not reducing text size. Proceeding to final summarization.")
                return self.final_summarize(summary_text, target_words)

            # Create a progress bar for recursive summarization
            with tqdm(total=1, desc="Recursive summarization", leave=False) as pbar:
                result = self.recursive_summarize(next_chunks, target_words, max_depth, current_depth + 1)
                pbar.update(1)
            return result


class MetadataExtractor:
    """Class for extracting metadata using OpenAI API."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """
        Initialize the metadata extractor.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): The name of the model to use for metadata extraction.
        """
        self.api_key = api_key
        self.model_name = model_name

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text using OpenAI API.

        Args:
            text (str): Text to extract metadata from.

        Returns:
            Dict[str, Any]: Extracted metadata.
        """
        prompt = (
            "Given the text below, extract the following metadata as JSON:\n"
            "- title\n- authors (list)\n- publisher\n- year\n- ISBN\n- DOI\n\n"
            "If not found, return null for the field. Example:\n"
            '{"title": "...", "authors": ["..."], "publisher": "...", "year": "...", "isbn": "...", "doi": "..."}\n\n'
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n"
        )
        openai.api_key = self.api_key
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            # fallback: try to parse JSON from text if it's surrounded by extra stuff
            m = re.search(r"\{.*\}", response.choices[0].message.content, re.DOTALL)
            return json.loads(m.group(0)) if m else {}


class BookProcessor:
    """Class for processing books and extracting metadata."""

    def __init__(self, api_key: str):
        """
        Initialize the book processor.

        Args:
            api_key (str): OpenAI API key.
        """
        self.api_key = api_key
        self.regex_extractor = RegexExtractor()
        self.text_chunker = TextChunker()
        self.summarizer = Summarizer(api_key)
        self.metadata_extractor = MetadataExtractor(api_key)

    def process_book(self, pages: Dict[int, str]) -> BookMetadata:
        """
        Process a book and extract metadata.

        Args:
            pages (Dict[int, str]): Dictionary of page numbers to page content.

        Returns:
            BookMetadata: Extracted book metadata.
        """
        # Create a progress bar for the overall book processing
        with tqdm(total=4, desc="Processing book", leave=False) as pbar:
            # Use first 3 pages for metadata (usually enough)
            first_pages = "\n".join([pages[k] for k in sorted(pages)[:3]])
            pbar.update(1)

            # LLM-based metadata
            pbar.set_description("Extracting metadata")
            llm_metadata = self.metadata_extractor.extract_metadata(first_pages)
            pbar.update(1)

            # Regex fallback
            pbar.set_description("Extracting ISBN/DOI")
            text_full = "\n".join([pages[k] for k in sorted(pages)])
            isbn = self.regex_extractor.extract_isbn(text_full) or llm_metadata.get("isbn")
            doi = self.regex_extractor.extract_doi(text_full) or llm_metadata.get("doi")
            pbar.update(1)

            # Recursive summary
            pbar.set_description("Generating summary")
            chunks = self.text_chunker.chunk_text(pages)
            summary = self.summarizer.recursive_summarize(chunks)
            pbar.update(1)

        return BookMetadata(
            title=llm_metadata.get("title"),
            authors=llm_metadata.get("authors", []),
            publisher=llm_metadata.get("publisher"),
            year=llm_metadata.get("year"),
            isbn=isbn,
            doi=doi,
            summary=summary
        )

class AIExtractor:
    """AI-based metadata extractor using OpenAI API."""

    def __init__(self, api_key=None):
        """
        Initialize the AI extractor with OpenAI API.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or provide it as an argument.")
        self.book_processor = BookProcessor(self.api_key)

    def extract_metadata_from_directory(self, client, directory, progress_bar=None):
        """
        Extract book metadata from a directory using OpenAI API.

        Args:
            client: WebDAV client or local file system client
            directory (str): Directory path
            progress_bar (tqdm, optional): Progress bar to update. Defaults to None.

        Returns:
            dict: Extracted metadata
        """
        if progress_bar:
            progress_bar.set_description(f"Checking files in {os.path.basename(directory)}")

        # Initialize metadata with default values
        metadata = {
            'title': '',
            'authors': [],
            'publisher': '',
            'year': '',
            'isbn': '',
            'doi': '',
            'summary': '',
            'cover_image': ''
        }

        try:
            # Use the directory name as a fallback title
            metadata['title'] = os.path.basename(directory)

            # Check for cover image in Cover directory with _p.jpg postfix
            cover_dir = os.path.join(directory, 'Cover')
            if client.check(cover_dir):
                cover_files = client.list(cover_dir)
                cover_files = [f for f in cover_files if f.endswith('_p.jpg')]
                if cover_files:
                    metadata['cover_image'] = os.path.join(cover_dir, cover_files[0])
                    logger.info(f"Found cover image: {metadata['cover_image']}")

            # Check for text files in Kramerius/OPACID_*/*.txt
            # Dictionary to store page content with page number as key
            page_content = {}

            # Get the base directory name (e.g., CVI_OPACID_SJF_802271061_X)
            base_dir_name = os.path.basename(directory)

            # Extract the OPACID part from the directory name
            if "OPACID" in base_dir_name:
                opacid_part = base_dir_name.split("CVI_")[1] if base_dir_name.startswith("CVI_") else base_dir_name

                # Construct the path to the text files directory
                kramierius_dir = os.path.join(directory, 'Kramerius')
                text_files_dir = os.path.join(kramierius_dir, opacid_part)

                if client.check(text_files_dir):
                    # List text files in the directory
                    text_files = client.list(text_files_dir)
                    text_files = [f for f in text_files if f.endswith('.txt')]

                    if text_files:
                        # Sort text files by page number
                        text_files.sort()

                        # Update progress bar after checking files
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_description(f"Processing content from {os.path.basename(directory)}")

                        # Load all pages into the dictionary
                        with tqdm(total=len(text_files), desc="Loading text files", leave=False) as file_pbar:
                            for file in text_files:
                                try:
                                    # Extract page number from filename (assuming format like *_p0001.txt)
                                    page_num = int(file.split('_p')[-1].split('.')[0])

                                    file_path = os.path.join(text_files_dir, file)
                                    content = client.read(file_path)

                                    if isinstance(content, bytes):
                                        content = content.decode('utf-8', errors='ignore')

                                    # Store in dictionary with page number as key
                                    page_content[page_num] = content
                                    file_pbar.update(1)
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Could not parse page number from file {file}: {e}")
                                    file_pbar.update(1)
                else:
                    logger.warning(f"Text files directory not found: {text_files_dir}")
                    if progress_bar:
                        progress_bar.update(1)  # Skip to next step since no files found
            else:
                logger.warning(f"Could not extract OPACID part from directory name: {base_dir_name}")
                if progress_bar:
                    progress_bar.update(1)  # Skip to next step since no files found

            # If pages were found, extract metadata using the BookProcessor
            if page_content:
                try:
                    # Process the book using the BookProcessor
                    book_metadata = self.book_processor.process_book(page_content)

                    # Update metadata with extracted values
                    metadata.update(book_metadata)

                    logger.info("Successfully extracted metadata from text files")
                except Exception as e:
                    logger.error(f"Failed to process book: {e}")
            else:
                logger.warning(f"No text content found in directory: {directory}")

            logger.info(f"Metadata extracted for directory: {directory}")

            # Update progress bar for finalizing metadata
            if progress_bar:
                progress_bar.set_description(f"Finalizing metadata for {os.path.basename(directory)}")
                progress_bar.update(1)

            return metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata from directory {directory}: {e}")
            if progress_bar:
                # Ensure progress bar is updated even on error
                if progress_bar.n < progress_bar.total:
                    progress_bar.update(progress_bar.total - progress_bar.n)
            return metadata

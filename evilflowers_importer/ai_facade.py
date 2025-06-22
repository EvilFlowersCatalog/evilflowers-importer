"""
AI Facade module for EvilFlowers Book Import.

This module provides a facade for AI functionality, abstracting away the specific AI model implementation.
"""

import os
import re
import logging
import json
from typing import TypedDict, Optional, List, Dict, Any, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from evilflowers_importer.ai_models import AIModelInterface, AIModelFactory

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
    llm_isbn: Optional[str]  # ISBN extracted using LLM
    llm_doi: Optional[str]   # DOI extracted using LLM
    summary: str


class RegexExtractor:
    """Class for extracting metadata using regular expressions."""

    # Improved regex patterns for ISBN and DOI
    # ISBN pattern that matches both ISBN-10 and ISBN-13 formats with simpler approach
    ISBN_PATTERN = re.compile(r"\b(?:ISBN(?:-1[03])?:?\s*)?(?:97[89][-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d|(?:\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX]))\b", re.I)

    # DOI pattern with more comprehensive matching
    DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)

    @classmethod
    def extract_isbn(cls, text: str) -> Optional[str]:
        """
        Extract ISBN from text using regex.
        Handles both ISBN-10 and ISBN-13 formats with various separators.
        """
        match = cls.ISBN_PATTERN.search(text)
        if not match:
            return None

        # Clean up the ISBN by removing spaces and hyphens
        isbn = match.group(0)
        if "ISBN" in isbn:
            # Remove the "ISBN" prefix if present
            isbn = re.sub(r'^ISBN[-:]?\s*', '', isbn, flags=re.I)

        # Remove all spaces and hyphens
        isbn = re.sub(r'[\s-]', '', isbn)

        return isbn

    @classmethod
    def extract_doi(cls, text: str) -> Optional[str]:
        """
        Extract DOI from text using regex.
        Handles standard DOI format (10.xxxx/yyyy).
        """
        match = cls.DOI_PATTERN.search(text)
        if not match:
            return None

        # Return the cleaned DOI
        doi = match.group(0)
        # Ensure DOI is lowercase as per standard
        return doi.lower()


class TextChunker:
    """Class for chunking text into manageable pieces."""

    def __init__(self, ai_model: AIModelInterface, max_workers: int = None):
        """
        Initialize the text chunker.

        Args:
            ai_model (AIModelInterface): The AI model to use for token counting.
            max_workers (int, optional): Maximum number of worker threads for parallel processing.
                If None, it will use the default value from ThreadPoolExecutor.
        """
        self.ai_model = ai_model
        self.max_workers = max_workers
        # Calculate actual number of workers to use
        self.actual_workers = self.max_workers if self.max_workers is not None else os.cpu_count() or 4
        logger.info(f"TextChunker initialized with {self.actual_workers} worker threads")

    def _process_page_chunk(self, pages_chunk: List[Tuple[int, str]], max_tokens: int) -> List[str]:
        """
        Process a chunk of pages and create text chunks based on token count.

        Args:
            pages_chunk (List[Tuple[int, str]]): List of (page_num, text) tuples to process.
            max_tokens (int): Maximum number of tokens per chunk.

        Returns:
            List[str]: List of text chunks from the processed pages.
        """
        current, chunks = "", []

        for page_num, text in pages_chunk:
            if self.ai_model.count_tokens(current + text) > max_tokens:
                chunks.append(current)
                current = ""
            current += text + "\n"

        if current:
            chunks.append(current)

        return chunks

    def chunk_text_parallel(self, pages: Dict[int, str], max_tokens: int = 3500) -> List[str]:
        """
        Chunk text into manageable pieces based on token count using parallel processing.

        Args:
            pages (dict): Dictionary of page numbers to page content.
            max_tokens (int): Maximum number of tokens per chunk.

        Returns:
            List[str]: List of text chunks.
        """
        sorted_pages = sorted(pages)
        total_pages = len(sorted_pages)

        # Log information about the parallelism
        logger.info(f"Chunking {total_pages} pages using {self.actual_workers} worker threads")

        # Convert pages dict to list of (page_num, text) tuples
        pages_list = [(page_num, pages[page_num]) for page_num in sorted_pages]

        # Determine chunk size for parallel processing
        # Each worker will process approximately this many pages
        chunk_size = max(1, total_pages // self.actual_workers)

        # Split pages into chunks for parallel processing
        page_chunks = [pages_list[i:i + chunk_size] for i in range(0, len(pages_list), chunk_size)]

        # Log information about the chunks
        logger.info(f"Split {total_pages} pages into {len(page_chunks)} chunks (approx. {chunk_size} pages per worker)")

        all_chunks = []

        # Create a progress bar for parallel text chunking
        with tqdm(total=total_pages, desc=f"Chunking text with {self.actual_workers} workers", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks to the executor
                future_to_chunk = {
                    executor.submit(self._process_page_chunk, chunk, max_tokens): chunk 
                    for chunk in page_chunks
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        all_chunks.extend(result)
                        # Update progress bar based on the number of pages in this chunk
                        pbar.update(len(chunk))
                    except Exception as exc:
                        logger.error(f"Chunk processing generated an exception: {exc}")
                        # Still update the progress bar even on error
                        pbar.update(len(chunk))

        logger.info(f"Created {len(all_chunks)} text chunks from {total_pages} pages")
        return all_chunks

    def chunk_text(self, pages: Dict[int, str], max_tokens: int = 3500) -> List[str]:
        """
        Chunk text into manageable pieces based on token count.
        This method now uses parallel processing by default.

        Args:
            pages (dict): Dictionary of page numbers to page content.
            max_tokens (int): Maximum number of tokens per chunk.

        Returns:
            List[str]: List of text chunks.
        """
        # Use parallel chunking by default
        return self.chunk_text_parallel(pages, max_tokens)


class Summarizer:
    """Class for summarizing text using AI models."""

    def __init__(self, ai_model: AIModelInterface, max_workers: int = None):
        """
        Initialize the summarizer.

        Args:
            ai_model (AIModelInterface): The AI model to use for summarization.
            max_workers (int, optional): Maximum number of worker threads for parallel processing.
                If None, it will use the default value from ThreadPoolExecutor.
        """
        self.ai_model = ai_model
        self.max_workers = max_workers
        # Calculate actual number of workers to use
        self.actual_workers = self.max_workers if self.max_workers is not None else os.cpu_count() or 4
        logger.info(f"Summarizer initialized with {self.actual_workers} worker threads")

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
        response = self.ai_model.generate_text(prompt, temperature=0.2)

        # Check if the response is an error message
        if response.startswith("ERROR:"):
            logger.error(f"AI model returned an error during summarization: {response}")
            # Return a placeholder summary with the error message
            return f"[Unable to generate summary: AI model unavailable - {response.split('ERROR:', 1)[1].strip()}]"

        return response

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
        response = self.ai_model.generate_text(prompt, temperature=0.2)

        # Check if the response is an error message
        if response.startswith("ERROR:"):
            logger.error(f"AI model returned an error during final summarization: {response}")
            # Return a placeholder summary with the error message
            return f"[Unable to generate final summary: AI model unavailable - {response.split('ERROR:', 1)[1].strip()}]"

        return response

    def _summarize_chunks_parallel(self, chunks: List[str]) -> List[str]:
        """
        Summarize multiple chunks in parallel.

        Args:
            chunks (List[str]): List of text chunks to summarize.

        Returns:
            List[str]: List of summarized chunks.
        """
        logger.info(f"Summarizing {len(chunks)} chunks in parallel with {self.actual_workers} workers")
        summaries = []

        with tqdm(total=len(chunks), desc=f"Summarizing chunks with {self.actual_workers} workers", leave=False) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks to the executor
                future_to_chunk = {
                    executor.submit(self.summarize_chunk, chunk): i 
                    for i, chunk in enumerate(chunks)
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        # Store the result in the correct order
                        summaries.append((chunk_idx, result))
                        pbar.update(1)
                    except Exception as exc:
                        logger.error(f"Chunk {chunk_idx} summarization generated an exception: {exc}")
                        # Add an empty summary to maintain order
                        summaries.append((chunk_idx, ""))
                        pbar.update(1)

        # Sort summaries by chunk index and extract just the summaries
        summaries.sort(key=lambda x: x[0])
        return [summary for _, summary in summaries]

    def recursive_summarize(self, chunks: List[str], target_words: int = 120, max_depth: int = 3, current_depth: int = 0) -> str:
        """
        Recursively summarize text chunks until they can fit into a single prompt,
        then perform a final summarization to reach the target word count.
        Uses parallel processing for summarizing chunks.

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

        # Log information about the current recursion level
        logger.info(f"Recursive summarization depth {current_depth}: processing {len(chunks)} chunks")

        # Summarize chunks in parallel
        summaries = self._summarize_chunks_parallel(chunks)

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
            with tqdm(total=1, desc=f"Recursive summarization depth {current_depth+1}", leave=False) as pbar:
                result = self.recursive_summarize(next_chunks, target_words, max_depth, current_depth + 1)
                pbar.update(1)
            return result


class MetadataExtractor:
    """Class for extracting metadata using AI models."""

    def __init__(self, ai_model: AIModelInterface):
        """
        Initialize the metadata extractor.

        Args:
            ai_model (AIModelInterface): The AI model to use for metadata extraction.
        """
        self.ai_model = ai_model

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text using AI model.

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
        response = self.ai_model.generate_text(prompt, temperature=0.1)

        # Check if the response is an error message
        if response.startswith("ERROR:"):
            logger.error(f"AI model returned an error: {response}")
            # Return an empty metadata dictionary with an error message
            return {
                "title": f"Error: AI model unavailable - {response.split('ERROR:', 1)[1].strip()}",
                "authors": [],
                "publisher": None,
                "year": None,
                "isbn": None,
                "doi": None
            }

        try:
            return json.loads(response)
        except Exception:
            # fallback: try to parse JSON from text if it's surrounded by extra stuff
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    logger.error(f"Failed to parse JSON from response: {response}")

            # If all parsing attempts fail, return an empty dictionary with an error message
            logger.error(f"Failed to extract metadata from response: {response}")
            return {
                "title": "Error: Failed to extract metadata",
                "authors": [],
                "publisher": None,
                "year": None,
                "isbn": None,
                "doi": None
            }


class BookProcessor:
    """Class for processing books and extracting metadata."""

    def __init__(self, ai_model: AIModelInterface, max_workers: int = None):
        """
        Initialize the book processor.

        Args:
            ai_model (AIModelInterface): The AI model to use for processing.
            max_workers (int, optional): Maximum number of worker threads for parallel processing.
                If None, it will use the default value from ThreadPoolExecutor.
        """
        self.ai_model = ai_model
        self.max_workers = max_workers
        # Log information about the parallelism
        logger.info(f"BookProcessor initialized with max_workers={max_workers}")

        self.regex_extractor = RegexExtractor()
        self.text_chunker = TextChunker(ai_model, max_workers=max_workers)
        self.summarizer = Summarizer(ai_model, max_workers=max_workers)
        self.metadata_extractor = MetadataExtractor(ai_model)

    def process_book(self, pages: Dict[int, str]) -> BookMetadata:
        """
        Process a book and extract metadata.

        Args:
            pages (Dict[int, str]): Dictionary of page numbers to page content.

        Returns:
            BookMetadata: Extracted book metadata.
        """
        # Log information about the book being processed
        total_pages = len(pages)
        logger.info(f"Processing book with {total_pages} pages using {self.max_workers} worker threads")

        # Create a progress bar for the overall book processing
        with tqdm(total=4, desc=f"Processing book ({total_pages} pages)", leave=False) as pbar:
            # Use first 3 pages for metadata (usually enough)
            first_pages = "\n".join([pages[k] for k in sorted(pages)[:3]])
            pbar.update(1)

            # LLM-based metadata
            pbar.set_description(f"Extracting metadata from first 3 pages")
            llm_metadata = self.metadata_extractor.extract_metadata(first_pages)
            pbar.update(1)

            # Extract ISBN/DOI using regex from full text
            pbar.set_description(f"Extracting ISBN/DOI from full text")
            text_full = "\n".join([pages[k] for k in sorted(pages)])
            regex_isbn = self.regex_extractor.extract_isbn(text_full)
            regex_doi = self.regex_extractor.extract_doi(text_full)

            # Get LLM-extracted ISBN/DOI
            llm_isbn = llm_metadata.get("isbn")
            llm_doi = llm_metadata.get("doi")

            # Use regex as primary method with LLM as fallback for the main fields
            isbn = regex_isbn or llm_isbn
            doi = regex_doi or llm_doi
            pbar.update(1)

            # Recursive summary with parallel processing
            pbar.set_description(f"Generating summary with {self.max_workers} workers")
            chunks = self.text_chunker.chunk_text(pages)
            logger.info(f"Created {len(chunks)} chunks for summarization")
            summary = self.summarizer.recursive_summarize(chunks)
            pbar.update(1)

        return BookMetadata(
            title=llm_metadata.get("title"),
            authors=llm_metadata.get("authors", []),
            publisher=llm_metadata.get("publisher"),
            year=llm_metadata.get("year"),
            isbn=isbn,
            doi=doi,
            llm_isbn=llm_isbn,
            llm_doi=llm_doi,
            summary=summary
        )


class AIFacade:
    """Facade for AI-based metadata extraction."""

    def __init__(self, model_type: str = "openai", max_workers: int = None, **model_kwargs):
        """
        Initialize the AI facade.

        Args:
            model_type (str): The type of AI model to use ("openai" or "ollama").
            max_workers (int, optional): Maximum number of worker threads for parallel processing.
                If None, it will use the default value from ThreadPoolExecutor.
            **model_kwargs: Additional arguments to pass to the model constructor.
        """
        self.model_type = model_type
        self.max_workers = max_workers
        self.model_kwargs = model_kwargs

        # Calculate actual number of workers to use
        self.actual_workers = self.max_workers if self.max_workers is not None else os.cpu_count() or 4
        logger.info(f"AIFacade initialized with {self.actual_workers} worker threads")

        # Create the AI model
        self.ai_model = AIModelFactory.create_model(model_type, **model_kwargs)

        # Create the book processor
        self.book_processor = BookProcessor(self.ai_model, max_workers=max_workers)

    def extract_metadata_from_directory(self, client, directory, progress_bar=None):
        """
        Extract book metadata from a directory.

        Args:
            client: Local file system client
            directory (str): Directory path
            progress_bar (tqdm, optional): Progress bar to update. Defaults to None.

        Returns:
            dict: Extracted metadata
        """
        # Log information about the directory being processed
        logger.info(f"Extracting metadata from directory: {directory} with {self.actual_workers} worker threads")

        if progress_bar:
            progress_bar.set_description(f"Checking files in {os.path.basename(directory)} (workers: {self.actual_workers})")

        # Initialize metadata with default values
        metadata = {
            'title': '',
            'authors': [],
            'publisher': '',
            'year': '',
            'isbn': '',
            'doi': '',
            'llm_isbn': '',  # ISBN extracted using LLM
            'llm_doi': '',   # DOI extracted using LLM
            'summary': '',
            'cover_image': ''
        }

        try:
            # Use the directory name as a fallback title
            metadata['title'] = os.path.basename(directory)

            # Check for cover image in Cover directory with _p.jpg postfix
            # Build path to Cover directory
            cover_dir = os.path.join(directory, "Cover")
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
                kramierius_dir = os.path.join(directory, "Kramerius")
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
                            progress_bar.set_description(f"Processing content from {os.path.basename(directory)} (workers: {self.actual_workers})")

                        # Load all pages into the dictionary
                        with tqdm(total=len(text_files), desc="Loading text files", leave=False) as file_pbar:
                            for file in text_files:
                                try:
                                    # Extract page number from filename (assuming format like *_p0001.txt)
                                    page_num = int(file.split('_p')[-1].split('.')[0])

                                    # Build path to the text file
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
                progress_bar.set_description(f"Finalizing metadata for {os.path.basename(directory)} (workers: {self.actual_workers})")
                progress_bar.update(1)

            return metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata from directory {directory}: {e}")
            if progress_bar:
                # Ensure progress bar is updated even on error
                if progress_bar.n < progress_bar.total:
                    remaining_steps = progress_bar.total - progress_bar.n
                    progress_bar.set_description(f"Error processing {os.path.basename(directory)}")
                    progress_bar.update(remaining_steps)
                    logger.info(f"Updated progress bar by {remaining_steps} steps due to error")
            return metadata
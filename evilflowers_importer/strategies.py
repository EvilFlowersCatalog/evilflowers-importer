"""
Strategies module for EvilFlowers Book Import.

This module provides different strategies for content availability from various sources.
The strategies are responsible for listing input directories and processing individual items,
providing text streams and required static files to the rest of the application.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Iterator, BinaryIO
from evilflowers_importer.tui import RichProgressBar

# Set up logging
logger = logging.getLogger(__name__)


class ContentStrategy(ABC):
    """
    Abstract base class for content availability strategies.

    These strategies are responsible for listing input directories and processing individual items,
    providing text streams and required static files to the rest of the application.
    """

    @abstractmethod
    def list_items(self, client, base_dir: str) -> List[str]:
        """
        List all items to process in the base directory.

        Args:
            client: File system client
            base_dir (str): Base directory path

        Returns:
            List[str]: List of item identifiers (e.g., directory paths, URLs, database IDs)
        """
        pass

    @abstractmethod
    def process_item(self, client, item: str, progress_bar=None) -> Dict[str, Any]:
        """
        Process a single item and return its metadata.

        Args:
            client: Client for accessing data
            item (str): Item identifier (e.g., directory path, URL, database ID)
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[str, Any]: Item metadata including text content, cover path, and PDF path
        """
        pass

    @abstractmethod
    def get_text_content(self, client, item: str, progress_bar=None) -> Dict[int, str]:
        """
        Get text content from an item using a specific strategy.

        Args:
            client: Client for accessing data
            item (str): Item identifier (e.g., directory path, URL, database ID)
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[int, str]: Dictionary mapping page numbers to text content
        """
        pass

    @abstractmethod
    def get_cover_image_path(self, client, item: str) -> Optional[str]:
        """
        Get the path to the cover image for an item.

        Args:
            client: Client for accessing data
            item (str): Item identifier (e.g., directory path, URL, database ID)

        Returns:
            Optional[str]: Path to the cover image, or None if not found
        """
        pass

    @abstractmethod
    def get_pdf_path(self, client, item: str) -> Optional[str]:
        """
        Get the path to the PDF file for an item.

        Args:
            client: Client for accessing data
            item (str): Item identifier (e.g., directory path, URL, database ID)

        Returns:
            Optional[str]: Path to the PDF file, or None if not found
        """
        pass


class FileSystemKramerius(ContentStrategy):
    """
    Strategy for accessing content from Kramerius directory structure on the file system.

    This strategy expects the following directory structure:
    - Directory name contains OPACID
    - Cover images in Cover/ directory with _p.jpg postfix
    - Text files in Kramerius/OPACID_*/*.txt
    - PDF files in PDF/ directory
    """

    def list_items(self, client, base_dir: str) -> List[str]:
        """
        List all directories in the base directory.

        Args:
            client: File system client
            base_dir (str): Base directory path

        Returns:
            List[str]: List of directory paths
        """
        logger.info(f"Listing directories in {base_dir}")

        try:
            # Use the client's list_directories method if available
            if hasattr(client, 'list_directories'):
                directories = client.list_directories(base_dir)
                # Convert to full paths
                directories = [os.path.join(base_dir, d) if not os.path.isabs(d) else d for d in directories]
            else:
                # Fallback to manual listing
                items = client.list(base_dir)
                directories = []
                for item in items:
                    # Remove trailing slash if present
                    if item.endswith('/'):
                        item = item[:-1]
                    item_path = os.path.join(base_dir, item)
                    if client.check(item_path) and os.path.isdir(item_path):
                        directories.append(item_path)

            logger.info(f"Found {len(directories)} directories")
            return directories
        except Exception as e:
            logger.error(f"Failed to list directories: {e}")
            raise

    def process_item(self, client, item: str, progress_bar=None) -> Dict[str, Any]:
        """
        Process a single directory and return its metadata.

        Args:
            client: File system client
            item (str): Directory path
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[str, Any]: Directory metadata including text content, cover path, and PDF path
        """
        logger.info(f"Processing directory: {item}")

        # Get text content, cover image path, and PDF path
        text_content = self.get_text_content(client, item, progress_bar)
        cover_path = self.get_cover_image_path(client, item)
        pdf_path = self.get_pdf_path(client, item)

        # Return metadata
        return {
            'text_content': text_content,
            'cover_path': cover_path,
            'pdf_path': pdf_path
        }

    def get_text_content(self, client, item: str, progress_bar=None) -> Dict[int, str]:
        """
        Get text content from a directory using the Kramerius strategy.

        Args:
            client: File system client
            item (str): Directory path
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[int, str]: Dictionary mapping page numbers to text content
        """
        # Log information about the directory being processed
        logger.info(f"Getting text content using FileSystemKramerius strategy: {item}")

        if progress_bar:
            progress_bar.set_description(f"Checking files in {os.path.basename(item)} (FileSystemKramerius strategy)")

        # Dictionary to store page content with page number as key
        page_content = {}

        try:
            # Get the base directory name (e.g., CVI_OPACID_SJF_802271061_X)
            base_dir_name = os.path.basename(item)

            # Extract the OPACID part from the directory name
            if "OPACID" in base_dir_name:
                opacid_part = base_dir_name.split("CVI_")[1] if base_dir_name.startswith("CVI_") else base_dir_name

                # Construct the path to the text files directory
                kramierius_dir = os.path.join(item, "Kramerius")
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
                            progress_bar.set_description(f"Loading content from {os.path.basename(item)} (FileSystemKramerius strategy)")

                        # Load all pages into the dictionary
                        with RichProgressBar(total=len(text_files), description="Loading text files", leave=False) as file_pbar:
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

            return page_content
        except Exception as e:
            logger.error(f"Failed to get text content from directory {item}: {e}")
            if progress_bar and progress_bar.n < progress_bar.total:
                remaining_steps = progress_bar.total - progress_bar.n
                progress_bar.set_description(f"Error processing {os.path.basename(item)}")
                progress_bar.update(remaining_steps)
            return page_content

    def get_cover_image_path(self, client, item: str) -> Optional[str]:
        """
        Get the path to the cover image for a book using the Kramerius strategy.

        Args:
            client: File system client
            item (str): Directory path

        Returns:
            Optional[str]: Path to the cover image, or None if not found
        """
        logger.info(f"Getting cover image path using FileSystemKramerius strategy: {item}")

        # Check for cover image in Cover directory with _p.jpg postfix
        cover_dir = os.path.join(item, "Cover")
        if client.check(cover_dir):
            cover_files = client.list(cover_dir)
            cover_files = [f for f in cover_files if f.endswith('_p.jpg')]
            if cover_files:
                cover_path = os.path.join(cover_dir, cover_files[0])
                logger.info(f"Found cover image: {cover_path}")
                return cover_path

        logger.warning(f"No cover image found in directory: {item}")
        return None

    def get_pdf_path(self, client, item: str) -> Optional[str]:
        """
        Get the path to the PDF file for a book using the Kramerius strategy.

        Args:
            client: File system client
            item (str): Directory path

        Returns:
            Optional[str]: Path to the PDF file, or None if not found
        """
        logger.info(f"Getting PDF path using FileSystemKramerius strategy: {item}")

        # Check for PDF in PDF directory
        pdf_dir = os.path.join(item, "PDF")
        if client.check(pdf_dir):
            pdf_files = client.list(pdf_dir)
            pdf_files = [f for f in pdf_files if f.endswith('.pdf')]
            if pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_files[0])
                logger.info(f"Found PDF file: {pdf_path}")
                return pdf_path

        logger.warning(f"No PDF file found in directory: {item}")
        return None


class DummyStrategy(ContentStrategy):
    """
    Dummy strategy for testing and as a boilerplate for new strategies.

    This strategy doesn't actually access any real content but serves
    as a template for implementing new strategies. It can be used as a starting
    point for implementing strategies for different data sources like HTTP,
    PostgreSQL, WebDav, S3, etc.
    """

    def list_items(self, client, base_dir: str) -> List[str]:
        """
        List dummy items to process.

        Args:
            client: Client for accessing data
            base_dir (str): Base directory or connection string

        Returns:
            List[str]: List of dummy item identifiers
        """
        logger.info(f"Listing dummy items for {base_dir}")

        # Return 3 dummy items
        return [
            f"{base_dir}/dummy_item_1",
            f"{base_dir}/dummy_item_2",
            f"{base_dir}/dummy_item_3"
        ]

    def process_item(self, client, item: str, progress_bar=None) -> Dict[str, Any]:
        """
        Process a dummy item and return metadata.

        Args:
            client: Client for accessing data
            item (str): Item identifier
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[str, Any]: Dummy metadata
        """
        logger.info(f"Processing dummy item: {item}")

        # Get text content, cover image path, and PDF path
        text_content = self.get_text_content(client, item, progress_bar)
        cover_path = self.get_cover_image_path(client, item)
        pdf_path = self.get_pdf_path(client, item)

        # Return metadata
        return {
            'text_content': text_content,
            'cover_path': cover_path,
            'pdf_path': pdf_path
        }

    def get_text_content(self, client, item: str, progress_bar=None) -> Dict[int, str]:
        """
        Get dummy text content.

        Args:
            client: Client for accessing data
            item (str): Item identifier
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[int, str]: Dictionary mapping page numbers to dummy text content
        """
        logger.info(f"Getting dummy text content for item: {item}")

        if progress_bar:
            progress_bar.set_description(f"Processing {os.path.basename(item)} (Dummy strategy)")
            # Update progress bar to simulate work
            if progress_bar.n < progress_bar.total:
                progress_bar.update(progress_bar.total - progress_bar.n)

        # Return dummy text content for 3 pages
        return {
            1: f"This is page 1 of dummy content for {os.path.basename(item)}. This is a test.",
            2: f"This is page 2 of dummy content for {os.path.basename(item)}. The dummy strategy doesn't access real files.",
            3: f"This is page 3 of dummy content for {os.path.basename(item)}. It's just for testing purposes."
        }

    def get_cover_image_path(self, client, item: str) -> Optional[str]:
        """
        Get a dummy cover image path.

        Args:
            client: Client for accessing data
            item (str): Item identifier

        Returns:
            Optional[str]: Always returns None for the dummy strategy
        """
        logger.info(f"Getting dummy cover image path for item: {item}")
        return None

    def get_pdf_path(self, client, item: str) -> Optional[str]:
        """
        Get a dummy PDF path.

        Args:
            client: Client for accessing data
            item (str): Item identifier

        Returns:
            Optional[str]: Always returns None for the dummy strategy
        """
        logger.info(f"Getting dummy PDF path for item: {item}")
        return None


class StrategyFactory:
    """Factory for creating content strategies."""

    @staticmethod
    def create_strategy(strategy_name: str) -> ContentStrategy:
        """
        Create a content strategy based on the strategy name.

        Args:
            strategy_name (str): Name of the strategy to create

        Returns:
            ContentStrategy: The created strategy

        Raises:
            ValueError: If the strategy name is not recognized
        """
        strategies = {
            "kramerius": FileSystemKramerius,  # Renamed from KrameriusStrategy to FileSystemKramerius
            "dummy": DummyStrategy
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {', '.join(strategies.keys())}")

        return strategies[strategy_name]()

"""
Utilities module for EvilFlowers WebDAV Book Import.

This module provides common utility functions for the application.
"""

import os
import logging
import pandas as pd
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

def create_excel_file(directories_metadata, output_path):
    """
    Create an Excel file with book metadata.

    Args:
        directories_metadata (list): List of metadata dictionaries
        output_path (str): Path to the output Excel file

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Creating Excel file: {output_path}")

    try:
        # Create a progress bar for Excel file creation
        with tqdm(total=3, desc="Creating Excel file") as pbar:
            # Create a copy of the metadata without the full_text field
            clean_metadata = []

            # Use a nested progress bar for cleaning metadata
            with tqdm(total=len(directories_metadata), desc="Cleaning metadata", leave=False) as clean_pbar:
                for metadata in directories_metadata:
                    clean_metadata_item = metadata.copy()
                    # Remove full_text field if it exists to avoid large Excel files
                    if 'full_text' in clean_metadata_item:
                        del clean_metadata_item['full_text']
                    clean_metadata.append(clean_metadata_item)
                    clean_pbar.update(1)

            pbar.update(1)
            pbar.set_description("Creating DataFrame")

            # Create a DataFrame from the clean metadata
            df = pd.DataFrame(clean_metadata)
            pbar.update(1)

            pbar.set_description("Writing to Excel")
            # Write to Excel
            df.to_excel(output_path, index=False)
            pbar.update(1)

        logger.info(f"Excel file created successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create Excel file: {e}")
        raise

class LocalFileSystem:
    """
    A class to interact with the local file system in a way that's compatible with the WebDAV client.
    This allows us to use the same code for both WebDAV and local files.
    """

    def __init__(self, base_dir=None):
        """
        Initialize the local file system client.

        Args:
            base_dir (str, optional): Base directory for all operations. Defaults to None.
        """
        self.base_dir = base_dir or os.getcwd()

    def check(self, path):
        """
        Check if a path exists.

        Args:
            path (str): Path to check

        Returns:
            bool: True if the path exists, False otherwise
        """
        return os.path.exists(path)

    def list(self, path):
        """
        List files and directories at the specified path.

        Args:
            path (str): Path to list

        Returns:
            list: List of file and directory names
        """
        if not os.path.exists(path):
            return []

        items = os.listdir(path)
        # Add trailing slash to directories to be consistent with WebDAV
        result = []
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result.append(item + '/')
            else:
                result.append(item)
        return result

    def read(self, path):
        """
        Read the content of a file.

        Args:
            path (str): Path to the file

        Returns:
            bytes or str: Content of the file as bytes
        """
        try:
            # First try to open as binary to avoid encoding issues
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

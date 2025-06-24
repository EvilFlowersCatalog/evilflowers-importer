"""
Utilities module for EvilFlowers Book Import.

This module provides common utility functions for the application.
"""

import os
import logging
import pandas as pd
from tqdm import tqdm

def clean_metadata_item(item):
    """
    Clean metadata item by handling special fields.

    Args:
        item (dict): Metadata item to clean

    Returns:
        dict: Cleaned metadata item
    """
    clean_item = item.copy()

    # Remove full_text field if it exists to avoid large files
    if 'full_text' in clean_item:
        del clean_item['full_text']

    # Convert authors list to pipe-separated string
    if 'authors' in clean_item and isinstance(clean_item['authors'], list):
        clean_item['authors'] = '|'.join(clean_item['authors'])

    # Convert title list to string if it's a list
    if 'title' in clean_item and isinstance(clean_item['title'], list):
        clean_item['title'] = ' '.join(clean_item['title'])

    return clean_item

# Set up logging
logger = logging.getLogger(__name__)

def create_output_files(directories_metadata, output_dir):
    """
    Create Parquet and CSV files with book metadata in the specified directory.

    Args:
        directories_metadata (list): List of metadata dictionaries
        output_dir (str): Path to the output directory

    Returns:
        tuple: (parquet_path, csv_path) if successful
    """
    logger.info(f"Creating output files in directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    parquet_path = os.path.join(output_dir, "index.parquet")
    csv_path = os.path.join(output_dir, "index.csv")

    try:
        # Create a progress bar for file creation
        with tqdm(total=4, desc="Creating output files") as pbar:
            # Create a copy of the metadata without the full_text field
            clean_metadata = []

            # Use a nested progress bar for cleaning metadata
            with tqdm(total=len(directories_metadata), desc="Cleaning metadata", leave=False) as clean_pbar:
                for metadata in directories_metadata:
                    clean_metadata.append(clean_metadata_item(metadata))
                    clean_pbar.update(1)

            pbar.update(1)
            pbar.set_description("Creating DataFrame")

            # Create a DataFrame from the clean metadata
            df = pd.DataFrame(clean_metadata)

            # Ensure dirname (directory path) is the first column
            if 'dirname' in df.columns:
                cols = ['dirname'] + [col for col in df.columns if col != 'dirname']
                df = df[cols]

            pbar.update(1)

            pbar.set_description("Writing to Parquet")
            # Write to Parquet
            df.to_parquet(parquet_path, index=False)
            pbar.update(1)

            pbar.set_description("Writing to CSV")
            # Write to CSV
            df.to_csv(csv_path, index=False)
            pbar.update(1)

        logger.info(f"Output files created successfully: {parquet_path}, {csv_path}")
        return parquet_path, csv_path
    except Exception as e:
        logger.error(f"Failed to create output files: {e}")
        raise

def create_parquet_file(directories_metadata, output_path):
    """
    Create a Parquet file with book metadata.

    This function is maintained for backward compatibility.
    For new code, use create_output_files instead.

    Args:
        directories_metadata (list): List of metadata dictionaries
        output_path (str): Path to the output Parquet file

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Creating Parquet file: {output_path}")

    try:
        # Check if output_path is a directory
        if os.path.isdir(output_path) or output_path.endswith('/') or output_path.endswith('\\'):
            # Use the new function
            parquet_path, _ = create_output_files(directories_metadata, output_path)
            return True

        # Create a progress bar for Parquet file creation
        with tqdm(total=3, desc="Creating Parquet file") as pbar:
            # Create a copy of the metadata without the full_text field
            clean_metadata = []

            # Use a nested progress bar for cleaning metadata
            with tqdm(total=len(directories_metadata), desc="Cleaning metadata", leave=False) as clean_pbar:
                for metadata in directories_metadata:
                    clean_metadata.append(clean_metadata_item(metadata))
                    clean_pbar.update(1)

            pbar.update(1)
            pbar.set_description("Creating DataFrame")

            # Create a DataFrame from the clean metadata
            df = pd.DataFrame(clean_metadata)

            # Ensure dirname (directory path) is the first column
            if 'dirname' in df.columns:
                cols = ['dirname'] + [col for col in df.columns if col != 'dirname']
                df = df[cols]

            pbar.update(1)

            pbar.set_description("Writing to Parquet")
            # Write to Parquet
            df.to_parquet(output_path, index=False)
            pbar.update(1)

        logger.info(f"Parquet file created successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create Parquet file: {e}")
        raise

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
                    clean_metadata.append(clean_metadata_item(metadata))
                    clean_pbar.update(1)

            pbar.update(1)
            pbar.set_description("Creating DataFrame")

            # Create a DataFrame from the clean metadata
            df = pd.DataFrame(clean_metadata)

            # Ensure dirname (directory path) is the first column
            if 'dirname' in df.columns:
                cols = ['dirname'] + [col for col in df.columns if col != 'dirname']
                df = df[cols]

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
    A class to interact with the local file system.
    This allows us to work with local files in a consistent way.
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

    def list_directories(self, path=""):
        """
        List directories at the specified path.

        Args:
            path (str): Path to list directories from. Defaults to empty string (current directory).

        Returns:
            list: List of directory names
        """
        if not os.path.exists(path):
            return []

        items = os.listdir(path)
        result = []
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result.append(item)
        return result

    def test_connection(self, path=""):
        """
        Test the connection to the local file system.

        Args:
            path (str): Path to test. Defaults to empty string (current directory).

        Returns:
            dict: Dictionary with test results
                - auth_success: Always True for local file system
                - path_exists: True if the path exists
                - can_list: True if the client can list the contents of the path
                - error: Error message if any of the tests fail
        """
        result = {
            "auth_success": True,  # Always true for local file system
            "path_exists": False,
            "can_list": False,
            "error": None
        }

        try:
            # Test if the path exists
            logging.debug(f"Testing if path exists: {path}")
            path_exists = self.check(path)
            result["path_exists"] = path_exists

            if not path_exists:
                result["error"] = f"Path '{path}' does not exist"
                logging.debug(f"Path '{path}' does not exist")
                return result

            # Test if we can list the contents of the path
            logging.debug(f"Testing if we can list contents of path: {path}")
            try:
                self.list(path)
                result["can_list"] = True
                logging.debug(f"Successfully listed contents of path: {path}")
            except Exception as e:
                result["error"] = f"Cannot list contents of path '{path}': {str(e)}"
                logging.debug(f"Cannot list contents of path '{path}': {str(e)}")

        except Exception as e:
            result["error"] = f"Connection test failed: {str(e)}"
            logging.debug(f"Connection test failed: {str(e)}")

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

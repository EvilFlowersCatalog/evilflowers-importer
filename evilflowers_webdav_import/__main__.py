#!/usr/bin/env python3
"""
EvilFlowers WebDAV Book Import

A CLI application for extracting book metadata from WebDAV directories
and creating an Excel file with the results.
"""

import argparse
import os
import logging
from tqdm import tqdm

from evilflowers_webdav_import.webdav import WebDAVClient
from evilflowers_webdav_import.ai import AIExtractor
from evilflowers_webdav_import.utils import create_excel_file

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract book metadata from WebDAV directories and create an Excel file.'
    )

    parser.add_argument(
        '--webdav-url',
        required=True,
        help='URL of the WebDAV server'
    )

    parser.add_argument(
        '--username',
        required=True,
        help='WebDAV username'
    )

    parser.add_argument(
        '--password',
        required=True,
        help='WebDAV password'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Path to the output Excel file'
    )

    parser.add_argument(
        '--api-key',
        default=None,
        help='OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Connect to WebDAV server
        client = WebDAVClient(args.webdav_url, args.username, args.password, args.verbose)

        # List directories
        directories = client.list_directories()

        # Initialize AI extractor with OpenAI API
        ai_extractor = AIExtractor(args.api_key)

        # Extract metadata from each directory with progress bar
        directories_metadata = []

        # Create a top-level progress bar for processing files
        with tqdm(total=len(directories), desc="Processing directories") as pbar:
            for directory in directories:
                # Create a nested progress bar for downloading/processing
                with tqdm(total=1, desc=f"Processing {os.path.basename(directory)}", leave=False) as nested_pbar:
                    # Extract metadata
                    metadata = ai_extractor.extract_metadata_from_directory(client, directory, nested_pbar)

                    # Add directory path to metadata
                    metadata['dirname'] = directory

                    directories_metadata.append(metadata)

                # Update the top-level progress bar
                pbar.update(1)

        # Create Excel file
        create_excel_file(directories_metadata, args.output)

        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

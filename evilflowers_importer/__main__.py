#!/usr/bin/env python3
"""
EvilFlowers Book Import

A CLI application for extracting book metadata from local directories
and creating output files with the results.
"""

import argparse
import os
import logging
from tqdm import tqdm

from evilflowers_importer.ai import AIExtractor
from evilflowers_importer.utils import create_output_files, LocalFileSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract book metadata from local directories and create output files.'
    )

    parser.add_argument(
        '--input-dir',
        required=True,
        help='Path to the directory containing book directories'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Path to the output directory where index.parquet and index.csv will be created'
    )

    parser.add_argument(
        '--api-key',
        default=None,
        help='OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable'
    )

    parser.add_argument(
        '--model-type',
        choices=['openai', 'ollama'],
        default='openai',
        help='The type of AI model to use'
    )

    parser.add_argument(
        '--model-name',
        default=None,
        help='The name of the model to use. Defaults to "gpt-4o" for OpenAI and "mistral" for Ollama'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads for parallel processing. If not provided, uses the default based on CPU count.'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of directories to process. Useful for debugging.'
    )

    return parser.parse_args()


def list_directories(base_dir):
    """
    List all directories in the base directory.

    Args:
        base_dir (str): Base directory path

    Returns:
        list: List of directory paths
    """
    logger.info(f"Listing directories in {base_dir}")

    try:
        # Get all items in the base directory
        items = os.listdir(base_dir)

        # Filter out files and keep only directories
        directories = []
        for item in items:
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                directories.append(item_path)

        logger.info(f"Found {len(directories)} directories")
        return directories
    except Exception as e:
        logger.error(f"Failed to list directories: {e}")
        raise


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Create a local file system client
        client = LocalFileSystem(args.input_dir)

        logger.info(f"Using local directory: {args.input_dir}")

        # List directories
        directories = list_directories(args.input_dir)

        # Limit the number of directories if specified
        if args.limit is not None:
            logger.info(f"Limiting to {args.limit} directories")
            directories = directories[:args.limit]

        # Set default model name based on model type if not provided
        model_name = args.model_name
        if model_name is None:
            model_name = "gpt-4o" if args.model_type == "openai" else "mistral"

        # Initialize AI extractor with the specified model type
        ai_extractor = AIExtractor(
            api_key=args.api_key,
            max_workers=args.workers,
            model_type=args.model_type,
            model_name=model_name
        )

        # Extract metadata from each directory with progress bar
        directories_metadata = []

        # Log information about the parallelism
        workers_info = f" with {args.workers} workers" if args.workers else " with auto workers"
        logger.info(f"Processing {len(directories)} directories{workers_info}")

        # Create a top-level progress bar for processing files
        with tqdm(total=len(directories), desc=f"Processing directories{workers_info}") as pbar:
            for directory in directories:
                # Create a nested progress bar for processing
                # The total is set to 3 to represent: 1) checking files, 2) processing content, 3) finalizing metadata
                with tqdm(total=3, desc=f"Processing {os.path.basename(directory)}", leave=False) as nested_pbar:
                    # Extract metadata
                    metadata = ai_extractor.extract_metadata_from_directory(client, directory, nested_pbar)

                    # Add directory path to metadata
                    metadata['dirname'] = directory

                    directories_metadata.append(metadata)

                    # Ensure the nested progress bar is completed
                    if nested_pbar.n < nested_pbar.total:
                        nested_pbar.update(nested_pbar.total - nested_pbar.n)

                # Update the top-level progress bar
                pbar.update(1)

        # Create output files with a progress bar
        with tqdm(total=1, desc="Creating output files", leave=True) as output_pbar:
            parquet_path, csv_path = create_output_files(directories_metadata, args.output_dir)
            output_pbar.update(1)
            logger.info(f"Created output files: {parquet_path}, {csv_path}")

        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
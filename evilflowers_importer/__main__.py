#!/usr/bin/env python3
"""
EvilFlowers Book Import

A CLI application for extracting book metadata from local directories
and creating output files with the results.
"""

import argparse
import os
import logging
import pandas as pd
import time

from evilflowers_importer.ai import AIExtractor
from evilflowers_importer.utils import LocalFileSystem
from evilflowers_importer.tui import run_app, RichProgressBar, tui_app

# Set up logging
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
        '--results-file',
        required=True,
        help='Path to the JSON file where progress data will be stored'
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

    parser.add_argument(
        '--ignore-progress',
        action='store_true',
        help='Ignore existing progress and process all directories from scratch.'
    )

    parser.add_argument(
        '--strategy',
        choices=['kramerius', 'dummy'],
        default='kramerius',
        help='The import strategy to use. "kramerius" uses the current directory structure, "dummy" is a boilerplate strategy.'
    )

    return parser.parse_args()


# list_directories function has been removed and replaced with strategy.list_items


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create a local file system client
    client = LocalFileSystem(args.input_dir)

    logger.info(f"Using local directory: {args.input_dir}")

    # Set default model name based on model type if not provided
    model_name = args.model_name
    if model_name is None:
        model_name = "gpt-4o" if args.model_type == "openai" else "mistral"

    # Initialize AI extractor with the specified model type and strategy
    ai_extractor = AIExtractor(
        api_key=args.api_key,
        max_workers=args.workers,
        model_type=args.model_type,
        model_name=model_name,
        strategy=args.strategy
    )

    # List items to process using the strategy
    directories = ai_extractor.list_items(client, args.input_dir)

    # Load existing progress if available and not ignoring progress
    directories_metadata = []
    processed_dirs = set()

    if os.path.exists(args.results_file) and not args.ignore_progress:
        try:
            logger.info(f"Loading existing progress from {args.results_file}")
            progress_df = pd.read_json(args.results_file, orient="records")
            directories_metadata = progress_df.to_dict('records')

            # Check if 'dirname' column exists in the progress DataFrame
            if 'dirname' in progress_df.columns:
                processed_dirs = set(progress_df['dirname'].tolist())
                logger.info(f"Loaded progress for {len(processed_dirs)} directories")
            else:
                logger.warning("Progress file does not contain 'dirname' column. Starting with empty progress.")
                processed_dirs = set()
        except Exception as e:
            logger.error(f"Failed to load progress file: {e}")
            logger.info("Starting with empty progress")
    elif args.ignore_progress and os.path.exists(args.results_file):
        logger.info("Ignoring existing progress as requested")

    # Filter out already processed directories
    # Ensure we're comparing the same format of directory paths
    directories_to_process = []
    for d in directories:
        # Check if this directory has already been processed
        if d not in processed_dirs:
            directories_to_process.append(d)

    # Log information about directories being processed
    logger.debug(f"Total directories: {len(directories)}")
    logger.debug(f"Already processed: {len(processed_dirs)}")
    logger.debug(f"To be processed: {len(directories_to_process)}")

    # Limit the number of directories to process if specified
    if args.limit is not None:
        logger.info(f"Limiting to {args.limit} remaining directories")
        directories_to_process = directories_to_process[:args.limit]
        logger.debug(f"After limit: {len(directories_to_process)} to be processed")

    # Log information about the parallelism
    workers_info = f" with {args.workers} workers" if args.workers else " with auto workers"
    logger.info(f"Processing {len(directories_to_process)} directories{workers_info} ({len(processed_dirs)} already processed)")

    # Set up TUI app statistics and arguments
    tui_app.set_args(args, len(directories), len(processed_dirs))
    tui_app.total_directories = len(directories_to_process)
    tui_app.processed_directories = 0
    tui_app.start_time = time.time()

    # Create a top-level progress bar for processing files
    with RichProgressBar(total=len(directories_to_process), description=f"Processing directories{workers_info}") as pbar:
        for directory in directories_to_process:
            # Update current directory in TUI app
            tui_app.current_directory = directory
            # Create a nested progress bar for processing
            # The total is set to 3 to represent: 1) checking files, 2) processing content, 3) finalizing metadata
            with RichProgressBar(total=3, description=f"Processing {os.path.basename(directory)}", leave=False) as nested_pbar:
                # Process item using the strategy
                metadata = ai_extractor.process_item(client, directory, nested_pbar)

                # Add directory path to metadata
                metadata['dirname'] = directory

                directories_metadata.append(metadata)

                # Ensure the nested progress bar is completed
                if nested_pbar.n < nested_pbar.total:
                    nested_pbar.update(nested_pbar.total - nested_pbar.n)

            # Update the top-level progress bar
            pbar.update(1)

            # Update processed directories count in TUI app
            tui_app.processed_directories += 1

            # Save progress after each directory is processed
            with RichProgressBar(total=1, description=f"Saving progress for {os.path.basename(directory)}", leave=False) as save_pbar:
                # Create a DataFrame from the current metadata
                progress_df = pd.DataFrame(directories_metadata)

                # Ensure dirname (directory path) is the first column
                if 'dirname' in progress_df.columns:
                    cols = ['dirname'] + [col for col in progress_df.columns if col != 'dirname']
                    progress_df = progress_df[cols]

                # Save to JSON file
                try:
                    progress_df.to_json(args.results_file, orient="records", indent=4)
                except FileExistsError:
                    # This is expected since we're using the file to track progress
                    # Just overwrite the file
                    if os.path.exists(args.results_file):
                        os.remove(args.results_file)
                    progress_df.to_json(args.results_file, orient="records", indent=4)
                save_pbar.update(1)

    logger.info("Process completed successfully")

    return 0


if __name__ == "__main__":
    exit(run_app(main))

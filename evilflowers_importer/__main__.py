#!/usr/bin/env python3
"""
EvilFlowers Importer

A powerful tool for importing data from various sources through an AI extraction pipeline
to JSON files for the EvilFlowers OPDS catalog.
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
        description='Import data from various sources through an AI extraction pipeline to JSON files.'
    )

    parser.add_argument(
        '--input-dir',
        required=True,
        help='Path to the input directory or connection string for the data source'
    )

    parser.add_argument(
        '--results-file',
        required=True,
        help='Path to the JSON file where extracted data will be stored'
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
        help='Limit the number of items to process. Useful for debugging.'
    )

    parser.add_argument(
        '--ignore-progress',
        action='store_true',
        help='Ignore existing progress and process all items from scratch.'
    )

    parser.add_argument(
        '--strategy',
        choices=['kramerius', 'dummy'],
        default='kramerius',
        help='The import strategy to use. "kramerius" uses the file system structure, "dummy" is a boilerplate strategy.'
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
    items = ai_extractor.list_items(client, args.input_dir)

    # Load existing progress if available and not ignoring progress
    items_metadata = []
    processed_items = set()

    if os.path.exists(args.results_file) and not args.ignore_progress:
        try:
            logger.info(f"Loading existing progress from {args.results_file}")
            progress_df = pd.read_json(args.results_file, orient="records")
            items_metadata = progress_df.to_dict('records')

            # Check if 'dirname' column exists in the progress DataFrame
            if 'dirname' in progress_df.columns:
                processed_items = set(progress_df['dirname'].tolist())
                logger.info(f"Loaded progress for {len(processed_items)} items")
            else:
                logger.warning("Progress file does not contain 'dirname' column. Starting with empty progress.")
                processed_items = set()
        except Exception as e:
            logger.error(f"Failed to load progress file: {e}")
            logger.info("Starting with empty progress")
    elif args.ignore_progress and os.path.exists(args.results_file):
        logger.info("Ignoring existing progress as requested")

    # Filter out already processed items
    # Ensure we're comparing the same format of item paths
    items_to_process = []
    for item in items:
        # Check if this item has already been processed
        if item not in processed_items:
            items_to_process.append(item)

    # Log information about items being processed
    logger.debug(f"Total items: {len(items)}")
    logger.debug(f"Already processed: {len(processed_items)}")
    logger.debug(f"To be processed: {len(items_to_process)}")

    # Limit the number of items to process if specified
    if args.limit is not None:
        logger.info(f"Limiting to {args.limit} remaining items")
        items_to_process = items_to_process[:args.limit]
        logger.debug(f"After limit: {len(items_to_process)} to be processed")

    # Log information about the parallelism
    workers_info = f" with {args.workers} workers" if args.workers else " with auto workers"
    logger.info(f"Processing {len(items_to_process)} items{workers_info} ({len(processed_items)} already processed)")

    # Set up TUI app statistics and arguments
    tui_app.set_args(args, len(items), len(processed_items))
    tui_app.total_directories = len(items_to_process)  # Keep property name for compatibility
    tui_app.processed_directories = 0  # Keep property name for compatibility
    tui_app.start_time = time.time()

    # Create a top-level progress bar for processing files
    with RichProgressBar(total=len(items_to_process), description=f"Processing items{workers_info}") as pbar:
        for item in items_to_process:
            # Update current item in TUI app
            tui_app.current_directory = item  # Keep property name for compatibility
            # Create a nested progress bar for processing
            # The total is set to 3 to represent: 1) checking files, 2) processing content, 3) finalizing metadata
            with RichProgressBar(total=3, description=f"Processing {os.path.basename(item)}", leave=False) as nested_pbar:
                # Process item using the strategy
                metadata = ai_extractor.process_item(client, item, nested_pbar)

                # Add item path to metadata
                metadata['dirname'] = item  # Keep field name for compatibility

                items_metadata.append(metadata)

                # Ensure the nested progress bar is completed
                if nested_pbar.n < nested_pbar.total:
                    nested_pbar.update(nested_pbar.total - nested_pbar.n)

            # Update the top-level progress bar
            pbar.update(1)

            # Update processed items count in TUI app
            tui_app.processed_directories += 1  # Keep property name for compatibility

            # Save progress after each item is processed
            with RichProgressBar(total=1, description=f"Saving progress for {os.path.basename(item)}", leave=False) as save_pbar:
                # Create a DataFrame from the current metadata
                progress_df = pd.DataFrame(items_metadata)

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

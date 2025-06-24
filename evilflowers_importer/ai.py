"""
AI module for EvilFlowers Book Import.

This module provides functionality for using AI models to extract metadata from text files.
This is a compatibility layer that uses the new AI facade.
"""

import os
import logging
from typing import Dict, Any

from evilflowers_importer.ai_facade import AIFacade, BookMetadata
from evilflowers_importer.strategies import ContentStrategy, StrategyFactory

# Set up logging
logger = logging.getLogger(__name__)


class AIExtractor:
    """AI-based metadata extractor using AI models."""

    def __init__(self, api_key=None, max_workers=None, model_type="openai", strategy="kramerius", **model_kwargs):
        """
        Initialize the AI extractor.

        Args:
            api_key (str, optional): API key for the AI model. If not provided, it will be read from the environment variable.
            max_workers (int, optional): Maximum number of worker threads for parallel processing.
                If None, it will use the default value from ThreadPoolExecutor.
            model_type (str): The type of AI model to use ("openai" or "ollama").
            strategy (str): The import strategy to use ("kramerius" or "dummy").
            **model_kwargs: Additional arguments to pass to the model constructor.
        """
        # Set default model_kwargs if not provided
        if model_type == "openai" and "api_key" not in model_kwargs:
            model_kwargs["api_key"] = api_key

        self.max_workers = max_workers
        self.strategy_name = strategy
        # Calculate actual number of workers to use
        self.actual_workers = self.max_workers if self.max_workers is not None else os.cpu_count() or 4
        logger.info(f"AIExtractor initialized with {self.actual_workers} worker threads and {strategy} strategy")

        # Create the strategy
        self.strategy = StrategyFactory.create_strategy(strategy)

        # Create the AI facade
        self.facade = AIFacade(model_type=model_type, max_workers=max_workers, **model_kwargs)

    def list_items(self, client, base_dir):
        """
        List all items to process using the strategy.

        Args:
            client: Client for accessing data
            base_dir (str): Base directory or connection string

        Returns:
            List[str]: List of item identifiers
        """
        return self.strategy.list_items(client, base_dir)

    def process_items(self, client, items, progress_bar=None):
        """
        Process multiple items and return their metadata.

        Args:
            client: Client for accessing data
            items (List[str]): List of item identifiers
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            List[Dict[str, Any]]: List of item metadata
        """
        results = []
        for item in items:
            result = self.process_item(client, item, progress_bar)
            results.append(result)
        return results

    def process_item(self, client, item, progress_bar=None):
        """
        Process a single item and return its metadata.

        Args:
            client: Client for accessing data
            item (str): Item identifier
            progress_bar (RichProgressBar, optional): Progress bar to update

        Returns:
            Dict[str, Any]: Item metadata
        """
        # Get item content and metadata from the strategy
        item_data = self.strategy.process_item(client, item, progress_bar)

        # Extract components
        text_content = item_data.get('text_content', {})
        cover_path = item_data.get('cover_path')
        pdf_path = item_data.get('pdf_path')

        # Pass content to the facade for AI processing
        metadata = self.facade.extract_metadata_from_directory(
            client, item, text_content, cover_path, pdf_path, progress_bar
        )

        # Add item identifier to metadata
        metadata['dirname'] = item

        return metadata

    def extract_metadata_from_directory(self, client, directory, progress_bar=None):
        """
        Extract book metadata from a directory.

        This method is maintained for backward compatibility.
        For new code, use process_item instead.

        Args:
            client: Local file system client
            directory (str): Directory path
            progress_bar (RichProgressBar, optional): Progress bar to update. Defaults to None.

        Returns:
            dict: Extracted metadata
        """
        return self.process_item(client, directory, progress_bar)

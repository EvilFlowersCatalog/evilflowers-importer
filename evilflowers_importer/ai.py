"""
AI module for EvilFlowers Book Import.

This module provides functionality for using AI models to extract metadata from text files.
This is a compatibility layer that uses the new AI facade.
"""

import os
import logging
from typing import Dict, Any

from evilflowers_importer.ai_facade import AIFacade, BookMetadata

# Set up logging
logger = logging.getLogger(__name__)


class AIExtractor:
    """AI-based metadata extractor using AI models."""

    def __init__(self, api_key=None, max_workers=None, model_type="openai", **model_kwargs):
        """
        Initialize the AI extractor.

        Args:
            api_key (str, optional): API key for the AI model. If not provided, it will be read from the environment variable.
            max_workers (int, optional): Maximum number of worker threads for parallel processing.
                If None, it will use the default value from ThreadPoolExecutor.
            model_type (str): The type of AI model to use ("openai" or "ollama").
            **model_kwargs: Additional arguments to pass to the model constructor.
        """
        # Set default model_kwargs if not provided
        if model_type == "openai" and "api_key" not in model_kwargs:
            model_kwargs["api_key"] = api_key

        self.max_workers = max_workers
        # Calculate actual number of workers to use
        self.actual_workers = self.max_workers if self.max_workers is not None else os.cpu_count() or 4
        logger.info(f"AIExtractor initialized with {self.actual_workers} worker threads")

        # Create the AI facade
        self.facade = AIFacade(model_type=model_type, max_workers=max_workers, **model_kwargs)

    def extract_metadata_from_directory(self, client, directory, progress_bar=None):
        """
        Extract book metadata from a directory.

        Args:
            client: Local file system client
            directory (str): Directory path
            progress_bar (RichProgressBar, optional): Progress bar to update. Defaults to None.

        Returns:
            dict: Extracted metadata
        """
        # Delegate to the facade
        return self.facade.extract_metadata_from_directory(client, directory, progress_bar)

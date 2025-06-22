#!/usr/bin/env python3
"""
Test script for AI models in EvilFlowers WebDAV Book Import.

This script demonstrates how to use the AI models abstraction layer.
"""

import os
import argparse
import logging
from typing import Dict, Any

from evilflowers_webdav_import.ai_models import AIModelFactory
from evilflowers_webdav_import.ai_facade import AIFacade
from evilflowers_webdav_import.ai import AIExtractor
from evilflowers_webdav_import.utils import LocalFileSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_directly(model_type: str, prompt: str, **model_kwargs) -> None:
    """
    Test an AI model directly using the AIModelFactory.

    Args:
        model_type (str): The type of model to use ("openai" or "ollama").
        prompt (str): The prompt to generate text from.
        **model_kwargs: Additional arguments to pass to the model constructor.
    """
    logger.info(f"Testing {model_type} model directly")
    
    # Create the model
    model = AIModelFactory.create_model(model_type, **model_kwargs)
    
    # Generate text
    logger.info(f"Generating text with prompt: {prompt}")
    response = model.generate_text(prompt)
    
    # Print the response
    logger.info(f"Response from {model_type} model:")
    print(response)
    print("-" * 80)


def test_facade(model_type: str, directory: str, **model_kwargs) -> Dict[str, Any]:
    """
    Test the AI facade with a specific model type.

    Args:
        model_type (str): The type of model to use ("openai" or "ollama").
        directory (str): The directory to extract metadata from.
        **model_kwargs: Additional arguments to pass to the model constructor.

    Returns:
        Dict[str, Any]: The extracted metadata.
    """
    logger.info(f"Testing AI facade with {model_type} model")
    
    # Create the facade
    facade = AIFacade(model_type=model_type, **model_kwargs)
    
    # Create a local file system client
    client = LocalFileSystem()
    
    # Extract metadata
    logger.info(f"Extracting metadata from directory: {directory}")
    metadata = facade.extract_metadata_from_directory(client, directory)
    
    # Print the metadata
    logger.info(f"Metadata extracted with {model_type} model:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("-" * 80)
    
    return metadata


def test_extractor(model_type: str, directory: str, **model_kwargs) -> Dict[str, Any]:
    """
    Test the AI extractor with a specific model type.

    Args:
        model_type (str): The type of model to use ("openai" or "ollama").
        directory (str): The directory to extract metadata from.
        **model_kwargs: Additional arguments to pass to the model constructor.

    Returns:
        Dict[str, Any]: The extracted metadata.
    """
    logger.info(f"Testing AI extractor with {model_type} model")
    
    # Create the extractor
    extractor = AIExtractor(model_type=model_type, **model_kwargs)
    
    # Create a local file system client
    client = LocalFileSystem()
    
    # Extract metadata
    logger.info(f"Extracting metadata from directory: {directory}")
    metadata = extractor.extract_metadata_from_directory(client, directory)
    
    # Print the metadata
    logger.info(f"Metadata extracted with {model_type} model:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("-" * 80)
    
    return metadata


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test AI models in EvilFlowers WebDAV Book Import.'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['openai', 'ollama'],
        default='openai',
        help='The type of AI model to use'
    )
    
    parser.add_argument(
        '--api-key',
        default=None,
        help='API key for the AI model. If not provided, it will be read from the environment variable.'
    )
    
    parser.add_argument(
        '--model-name',
        default=None,
        help='The name of the model to use. Defaults to "gpt-4o" for OpenAI and "mistral" for Ollama.'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:11434',
        help='The URL of the Ollama API. Only used if model-type is "ollama".'
    )
    
    parser.add_argument(
        '--directory',
        default=None,
        help='Directory to extract metadata from. If not provided, only the direct model test will be run.'
    )
    
    parser.add_argument(
        '--prompt',
        default='Explain the concept of abstraction in software engineering.',
        help='Prompt to use for the direct model test.'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set default model name based on model type
    if args.model_name is None:
        args.model_name = "gpt-4o" if args.model_type == "openai" else "mistral"
    
    # Prepare model kwargs
    model_kwargs = {
        "model_name": args.model_name
    }
    
    if args.model_type == "openai":
        model_kwargs["api_key"] = args.api_key or os.environ.get("OPENAI_API_KEY")
    elif args.model_type == "ollama":
        model_kwargs["api_url"] = args.api_url
    
    # Test the model directly
    test_model_directly(args.model_type, args.prompt, **model_kwargs)
    
    # Test the facade and extractor if a directory is provided
    if args.directory:
        test_facade(args.model_type, args.directory, **model_kwargs)
        test_extractor(args.model_type, args.directory, **model_kwargs)
    
    return 0


if __name__ == "__main__":
    exit(main())
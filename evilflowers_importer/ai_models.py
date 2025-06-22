"""
AI Models module for EvilFlowers Book Import.

This module provides an abstraction layer for different AI models that can be used for metadata extraction.
"""

import os
import re
import logging
import json
from typing import Dict, List, Any, Optional, Protocol
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

class AIModelInterface(ABC):
    """Abstract base class for AI models."""

    @abstractmethod
    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generate text based on a prompt.

        Args:
            prompt (str): The prompt to generate text from.
            temperature (float): Controls randomness in generation. Lower is more deterministic.

        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        pass

class OpenAIModel(AIModelInterface):
    """Implementation of AIModelInterface for OpenAI API."""

    def __init__(self, api_key: str = None, model_name: str = "gpt-4o"):
        """
        Initialize the OpenAI model.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
            model_name (str): The name of the OpenAI model to use.
        """
        import openai
        import tiktoken

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or provide it as an argument.")

        self.model_name = model_name
        openai.api_key = self.api_key

        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found in tiktoken. Using cl100k_base encoding instead.")
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generate text using OpenAI API.

        Args:
            prompt (str): The prompt to generate text from.
            temperature (float): Controls randomness in generation. Lower is more deterministic.

        Returns:
            str: The generated text.
        """
        import openai

        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text using tiktoken.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        return len(self.encoder.encode(text))

class OllamaModel(AIModelInterface):
    """Implementation of AIModelInterface for Ollama API."""

    def __init__(self, model_name: str = "mistral", api_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama model.

        Args:
            model_name (str): The name of the Ollama model to use.
            api_url (str): The URL of the Ollama API.
        """
        import requests

        self.model_name = model_name
        self.api_url = api_url
        self.is_model_available = False
        self.is_ollama_running = False
        self.available_models = []

        # Check if Ollama is running
        try:
            response = requests.get(f"{api_url}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama API returned status code {response.status_code}. Make sure Ollama is running.")
            else:
                self.is_ollama_running = True
                # Check if the model is available
                models = response.json().get("models", [])
                self.available_models = [model.get("name") for model in models]
                if model_name not in self.available_models:
                    logger.warning(f"Model '{model_name}' not found in Ollama. Available models: {self.available_models}")
                    logger.info(f"You can pull the model using: ollama pull {model_name}")
                else:
                    self.is_model_available = True
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama API: {e}")
            logger.info("Make sure Ollama is installed and running. You can install it from https://ollama.ai/")

    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generate text using Ollama API.

        Args:
            prompt (str): The prompt to generate text from.
            temperature (float): Controls randomness in generation. Lower is more deterministic.

        Returns:
            str: The generated text.
        """
        import requests

        # Check if Ollama is running
        if not self.is_ollama_running:
            error_msg = (
                "Ollama is not running. Please install and start Ollama:\n"
                "1. Download from https://ollama.ai/\n"
                "2. Install and start the application\n"
                "3. Run your command again"
            )
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

        # Check if the model is available
        if not self.is_model_available:
            available_models_str = ", ".join(self.available_models) if self.available_models else "none"
            error_msg = (
                f"Model '{self.model_name}' is not available in your Ollama installation.\n"
                f"Available models: {available_models_str}\n"
                f"To install the model, run this command in your terminal:\n"
                f"    ollama pull {self.model_name}\n"
                f"Then run your command again."
            )
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )

            if response.status_code == 200:
                return response.json().get("response", "").strip()
            elif response.status_code == 404 and "model not found" in response.text.lower():
                # This can happen if the model was removed after we checked
                error_msg = (
                    f"Model '{self.model_name}' not found. Please install it with:\n"
                    f"    ollama pull {self.model_name}"
                )
                logger.error(f"Ollama API returned status code 404: {response.text}")
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
            else:
                logger.error(f"Ollama API returned status code {response.status_code}: {response.text}")
                return f"ERROR: Ollama API error: {response.text}"
        except Exception as e:
            error_msg = f"Failed to generate text with Ollama: {e}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        This is a rough estimate as Ollama doesn't provide a direct way to count tokens.
        We use a simple heuristic: 1 token ≈ 4 characters.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The estimated number of tokens.
        """
        # Simple heuristic: 1 token ≈ 4 characters
        return len(text) // 4 + 1

class AIModelFactory:
    """Factory class for creating AI models."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> AIModelInterface:
        """
        Create an AI model of the specified type.

        Args:
            model_type (str): The type of model to create ("openai" or "ollama").
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            AIModelInterface: An instance of the specified model type.

        Raises:
            ValueError: If the model type is not supported.
        """
        if model_type.lower() == "openai":
            return OpenAIModel(**kwargs)
        elif model_type.lower() == "ollama":
            return OllamaModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
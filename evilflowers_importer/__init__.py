"""
EvilFlowers Importer package.

A powerful tool for importing data from various sources through an AI extraction pipeline 
to JSON files for the EvilFlowers OPDS catalog.

The package provides a flexible and extensible architecture for:
- Extracting content from different sources (file systems, WebDAV, databases, etc.)
- Processing content using AI models (OpenAI, Ollama, etc.)
- Generating standardized JSON output for import into the EvilFlowers OPDS catalog
"""

import logging

# Set version
__version__ = "1.0.0"

# Disable INFO logging for httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

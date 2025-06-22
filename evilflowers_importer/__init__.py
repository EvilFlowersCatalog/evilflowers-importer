"""
EvilFlowers Book Import package.

A package for extracting book metadata from local directories
and creating output files with the results.
"""

import logging

# Disable INFO logging for httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
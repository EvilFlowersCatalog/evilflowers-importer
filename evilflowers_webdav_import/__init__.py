"""
EvilFlowers WebDAV Book Import package.
"""

import logging

# Disable INFO logging for httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

"""
WebDAV client module for EvilFlowers WebDAV Book Import.

This module provides functionality for connecting to and interacting with WebDAV servers.
"""

import os
import logging
from webdav3.client import Client
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class WebDAVClient:
    """WebDAV client for interacting with WebDAV servers."""

    def __init__(self, url, username, password, verbose=False):
        """
        Initialize the WebDAV client.

        Args:
            url (str): URL of the WebDAV server
            username (str): WebDAV username
            password (str): WebDAV password
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        self.url = url
        self.username = username
        self.password = password
        self.verbose = verbose
        self.client = self._connect()

    def _connect(self):
        """Connect to WebDAV server and return a client instance."""
        logger.info(f"Connecting to WebDAV server at {self.url}")

        options = {
            'webdav_hostname': self.url,
            'webdav_login': self.username,
            'webdav_password': self.password,
            'webdav_timeout': 30
        }

        client = Client(options)

        # Test connection
        try:
            client.list()
            logger.info("Successfully connected to WebDAV server")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to WebDAV server: {e}")
            raise

    def list_directories(self):
        """
        List all directories on the WebDAV server.
        
        Returns:
            list: List of directory paths
        """
        logger.info("Listing directories on WebDAV server")

        try:
            items = self.client.list()
            # Filter out files and keep only directories
            directories = [item for item in items if item.endswith('/')]

            # Remove trailing slash
            directories = [dir_path[:-1] for dir_path in directories]

            logger.info(f"Found {len(directories)} directories")
            return directories
        except Exception as e:
            logger.error(f"Failed to list directories: {e}")
            raise

    def check(self, path):
        """
        Check if a path exists on the WebDAV server.
        
        Args:
            path (str): Path to check
            
        Returns:
            bool: True if the path exists, False otherwise
        """
        return self.client.check(path)

    def list(self, path):
        """
        List files and directories at the specified path.
        
        Args:
            path (str): Path to list
            
        Returns:
            list: List of file and directory names
        """
        return self.client.list(path)

    def read(self, path):
        """
        Read the content of a file.
        
        Args:
            path (str): Path to the file
            
        Returns:
            str or bytes: Content of the file
        """
        return self.client.read(path)

    def download_directory(self, remote_path, local_path, progress_bar=None):
        """
        Download a directory from the WebDAV server.
        
        Args:
            remote_path (str): Path to the directory on the WebDAV server
            local_path (str): Local path to download to
            progress_bar (tqdm, optional): Progress bar to update. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if progress_bar:
                progress_bar.set_description(f"Downloading {os.path.basename(remote_path)}")
            
            # Create local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # List files and directories
            items = self.client.list(remote_path)
            
            for item in items:
                remote_item_path = os.path.join(remote_path, item)
                local_item_path = os.path.join(local_path, item)
                
                if item.endswith('/'):  # It's a directory
                    # Create local directory
                    os.makedirs(local_item_path, exist_ok=True)
                    # Download directory contents
                    self.download_directory(remote_item_path, local_item_path, progress_bar)
                else:  # It's a file
                    # Download file
                    self.client.download_file(remote_item_path, local_item_path)
            
            if progress_bar:
                progress_bar.update(1)
            
            return True
        except Exception as e:
            logger.error(f"Failed to download directory {remote_path}: {e}")
            return False
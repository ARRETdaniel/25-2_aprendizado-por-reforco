#!/usr/bin/env python

"""
Debug script to check the bridge directory paths and permissions.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_bridge_directory():
    """Check the bridge directory."""
    # Expected bridge directory
    bridge_dir = Path.home() / ".carla_drl_bridge"

    logger.info(f"Bridge directory path: {bridge_dir}")

    # Check if directory exists
    if bridge_dir.exists():
        logger.info(f"Bridge directory exists: {bridge_dir}")

        # Check permissions
        try:
            test_file_path = bridge_dir / "test_write.txt"
            with open(test_file_path, 'w') as f:
                f.write("Test write")
            logger.info(f"Successfully wrote to {test_file_path}")

            # Read the file
            with open(test_file_path, 'r') as f:
                content = f.read()
            logger.info(f"Successfully read from {test_file_path}: {content}")

            # Delete the test file
            test_file_path.unlink()
            logger.info(f"Successfully deleted {test_file_path}")
        except Exception as e:
            logger.error(f"Error accessing bridge directory: {e}")
    else:
        logger.warning(f"Bridge directory does not exist: {bridge_dir}")

        # Try to create the directory
        try:
            bridge_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Successfully created bridge directory: {bridge_dir}")
        except Exception as e:
            logger.error(f"Error creating bridge directory: {e}")

def list_directory_contents():
    """List the contents of the bridge directory."""
    bridge_dir = Path.home() / ".carla_drl_bridge"

    if bridge_dir.exists():
        logger.info(f"Listing contents of {bridge_dir}:")
        for item in bridge_dir.iterdir():
            if item.is_file():
                file_size = item.stat().st_size
                modified_time = item.stat().st_mtime
                logger.info(f"  {item.name} - Size: {file_size} bytes, Modified: {time.ctime(modified_time)}")
            else:
                logger.info(f"  {item.name}/ (directory)")
    else:
        logger.warning(f"Cannot list contents: {bridge_dir} does not exist")

if __name__ == "__main__":
    logger.info("Starting bridge path debug")

    # Check Python version
    logger.info(f"Python version: {sys.version}")

    # Check current working directory
    logger.info(f"Current working directory: {os.getcwd()}")

    # Check bridge directory
    check_bridge_directory()

    # List contents if it exists
    list_directory_contents()

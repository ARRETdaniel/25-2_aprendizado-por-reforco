"""
Development Setup Script

This script sets up the development environment for the RL-based autonomous driving project.
It ensures that the project modules are available in the Python path by creating a .pth file
in the site-packages directory.

Usage:
    python setup_development.py [--uninstall]
"""

import os
import sys
import site
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DevSetup")

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(os.path.abspath(__file__)).parent

def install_development_path() -> bool:
    """
    Create a .pth file in site-packages to add project root to Python path.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        project_root = get_project_root()
        site_packages_dir = site.getsitepackages()[0]
        pth_file_path = os.path.join(site_packages_dir, "rl_carla_dev.pth")

        logger.info(f"Adding project path to Python path: {project_root}")
        logger.info(f"Creating .pth file at: {pth_file_path}")

        # Create the .pth file with the project root path
        with open(pth_file_path, "w") as f:
            f.write(str(project_root))

        logger.info("Development path setup complete.")
        logger.info(f"The project at '{project_root}' can now be imported in Python.")
        return True

    except Exception as e:
        logger.error(f"Failed to set up development path: {e}")
        return False

def uninstall_development_path() -> bool:
    """
    Remove the .pth file from site-packages.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        site_packages_dir = site.getsitepackages()[0]
        pth_file_path = os.path.join(site_packages_dir, "rl_carla_dev.pth")

        if os.path.exists(pth_file_path):
            logger.info(f"Removing .pth file: {pth_file_path}")
            os.remove(pth_file_path)
            logger.info("Development path removed.")
            return True
        else:
            logger.info("Development path was not installed.")
            return True

    except Exception as e:
        logger.error(f"Failed to uninstall development path: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up development environment for RL-based autonomous driving project"
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall development path"
    )

    args = parser.parse_args()

    if args.uninstall:
        success = uninstall_development_path()
    else:
        success = install_development_path()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

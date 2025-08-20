"""
Setup script for validation system dependencies.

This script installs all required Python packages for running the validation system.

Usage:
    python setup_validation.py
"""

import os
import sys
import subprocess
import logging
import argparse
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("SetupValidation")

# Dependencies for validation
CORE_DEPENDENCIES = [
    "matplotlib",
    "numpy",
    "pandas",
    "tabulate"  # For markdown table formatting
]

# Optional dependencies for advanced features
OPTIONAL_DEPENDENCIES = {
    "visualization": [
        "seaborn",
        "plotly"
    ],
    "reporting": [
        "jinja2",
        "markdown"
    ],
    "dashboard": [
        "streamlit",
        "altair"
    ]
}

def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    major, minor, _ = sys.version_info[:3]

    # For CARLA 0.8.4, Python 3.6+ is preferred
    if major < 3 or (major == 3 and minor < 6):
        logger.warning(f"Python {major}.{minor} detected. Python 3.6+ is recommended for CARLA 0.8.4.")
        return False

    # For validation scripts, Python 3.8+ is recommended for advanced features
    if major < 3 or (major == 3 and minor < 8):
        logger.warning(f"Python {major}.{minor} detected. Python 3.8+ is recommended for validation scripts.")
        return False

    logger.info(f"Python version check passed: {major}.{minor}")
    return True

def check_pip() -> bool:
    """Check if pip is available."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("pip is available")
        return True
    except subprocess.CalledProcessError:
        logger.error("pip not found. Please install pip to continue.")
        return False

def install_dependencies(dependencies: List[str], upgrade: bool = False) -> bool:
    """
    Install dependencies using pip.

    Args:
        dependencies: List of package names to install
        upgrade: Whether to upgrade existing packages

    Returns:
        True if installation was successful, False otherwise
    """
    if not dependencies:
        return True

    logger.info(f"Installing {len(dependencies)} packages: {', '.join(dependencies)}")

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(dependencies)

    try:
        subprocess.check_call(cmd)
        logger.info("Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        return False

def check_dependencies(dependencies: List[str]) -> Dict[str, bool]:
    """
    Check if dependencies are installed.

    Args:
        dependencies: List of package names to check

    Returns:
        Dictionary with package names and their installation status
    """
    results = {}

    logger.info(f"Checking {len(dependencies)} packages")

    for package in dependencies:
        cmd = [sys.executable, "-c", f"import {package.split('==')[0].split('>=')[0]}"]
        try:
            subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results[package] = True
            logger.debug(f"Package {package} is installed")
        except subprocess.CalledProcessError:
            results[package] = False
            logger.debug(f"Package {package} is NOT installed")

    return results

def main() -> int:
    """Setup validation dependencies."""
    parser = argparse.ArgumentParser(description="Setup CARLA Environment Validation Dependencies")
    parser.add_argument("--all", action="store_true",
                      help="Install all dependencies including optional ones")
    parser.add_argument("--group", type=str, choices=list(OPTIONAL_DEPENDENCIES.keys()),
                      help="Install a specific optional dependency group")
    parser.add_argument("--upgrade", action="store_true",
                      help="Upgrade existing packages")
    parser.add_argument("--check-only", action="store_true",
                      help="Only check if dependencies are installed, don't install anything")

    args = parser.parse_args()

    # Check Python version
    if not check_python_version():
        if not args.check_only:
            logger.warning("Proceeding with installation despite Python version warning.")

    # Check pip
    if not check_pip():
        logger.error("Cannot proceed without pip.")
        return 1

    # Determine which packages to install
    packages_to_install = CORE_DEPENDENCIES.copy()

    if args.all:
        for group in OPTIONAL_DEPENDENCIES.values():
            packages_to_install.extend(group)
    elif args.group:
        packages_to_install.extend(OPTIONAL_DEPENDENCIES[args.group])

    # Check current installation status
    package_status = check_dependencies(packages_to_install)
    installed_packages = [p for p, status in package_status.items() if status]
    missing_packages = [p for p, status in package_status.items() if not status]

    logger.info(f"{len(installed_packages)} packages already installed")

    if missing_packages:
        logger.info(f"{len(missing_packages)} packages need to be installed: {', '.join(missing_packages)}")

        if args.check_only:
            logger.info("Check-only mode: Not installing missing packages")
        else:
            success = install_dependencies(missing_packages, args.upgrade)
            if not success:
                logger.error("Failed to install some packages")
                return 1
    else:
        logger.info("All required packages are installed")

    # Check if upgrade was requested for already installed packages
    if args.upgrade and not args.check_only and installed_packages:
        logger.info(f"Upgrading {len(installed_packages)} already installed packages")
        success = install_dependencies(installed_packages, True)
        if not success:
            logger.error("Failed to upgrade some packages")
            return 1

    logger.info("Setup completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())

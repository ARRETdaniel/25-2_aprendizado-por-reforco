#!/usr/bin/env python
"""
Script to verify CARLA import works properly.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print Python environment info
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")

# Project paths
project_root = Path(__file__).parent.parent
logger.info(f"Project root: {project_root}")

# Add potential CARLA paths to sys.path
paths_to_try = [
    project_root / "CarlaSimulator" / "PythonAPI",
    project_root / "CarlaSimulator" / "PythonAPI" / "carla",
    project_root / "CarlaSimulator" / "PythonAPI" / "carla" / "dist",
    project_root / "CarlaSimulator" / "PythonClient",
    Path("c:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/CarlaSimulator/PythonAPI"),
    Path("c:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/CarlaSimulator/PythonAPI/carla"),
    Path("c:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/CarlaSimulator/PythonAPI/carla/dist"),
    Path("c:/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/CarlaSimulator/PythonClient")
]

# Try to find the egg file in the dist folder
for path in paths_to_try:
    if path.exists():
        logger.info(f"Path exists: {path}")
        if path.name == "dist" or (path / "dist").exists():
            dist_path = path if path.name == "dist" else path / "dist"
            egg_files = list(dist_path.glob("*.egg"))
            for egg_file in egg_files:
                logger.info(f"Found egg file: {egg_file}")
                sys.path.insert(0, str(egg_file))
                logger.info(f"Added to sys.path: {egg_file}")
        else:
            sys.path.insert(0, str(path))
            logger.info(f"Added to sys.path: {path}")

# Print sys.path for debugging
logger.info("sys.path:")
for p in sys.path:
    logger.info(f"  {p}")

# Try to import CARLA
try:
    import carla
    logger.info("Successfully imported carla module!")
    logger.info(f"CARLA version: {carla.__version__}")
    logger.info(f"CARLA path: {carla.__file__}")
except ImportError as e:
    logger.error(f"Failed to import CARLA: {e}")

    # Look for egg files in the workspace
    logger.info("Searching for .egg files in the workspace...")
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".egg"):
                egg_path = os.path.join(root, file)
                logger.info(f"Found egg file: {egg_path}")

    logger.error("Make sure CARLA Python API is in your PYTHONPATH")
    sys.exit(1)

logger.info("Script completed successfully!")

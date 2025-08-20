#!/usr/bin/env python

"""
Simple script to test CARLA installation.
Just try to connect to the simulator.
"""

import os
import sys
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print Python info
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python version: {sys.version}")

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory: {current_dir}")

# Add the PythonAPI dir to the path
carla_pythonapi_path = os.path.abspath(os.path.join(current_dir, '..', 'CarlaSimulator', 'PythonAPI'))
sys.path.append(carla_pythonapi_path)
logger.info(f"Added to path: {carla_pythonapi_path}")

# Add the carla dir to the path
carla_path = os.path.join(carla_pythonapi_path, 'carla')
sys.path.append(carla_path)
logger.info(f"Added to path: {carla_path}")

# Add the examples dir to the path
examples_path = os.path.join(carla_pythonapi_path, 'examples')
sys.path.append(examples_path)
logger.info(f"Added to path: {examples_path}")

# Print sys.path
logger.info("Python sys.path:")
for p in sys.path:
    logger.info(f"  {p}")

# Try to find and append the CARLA .egg file
try:
    carla_egg_pattern = os.path.join(carla_path, 'dist', f'carla-*{sys.version_info.major}.{sys.version_info.minor}-*.egg')
    logger.info(f"Looking for CARLA egg file with pattern: {carla_egg_pattern}")
    carla_egg = glob.glob(carla_egg_pattern)
    if carla_egg:
        egg_path = carla_egg[0]
        sys.path.append(egg_path)
        logger.info(f"Found and added CARLA egg: {egg_path}")
    else:
        logger.warning(f"No CARLA egg file found with pattern: {carla_egg_pattern}")
except Exception as e:
    logger.error(f"Error finding CARLA egg: {e}")

# Try to import CARLA
try:
    import carla
    logger.info("Successfully imported CARLA module")
    logger.info(f"CARLA module path: {carla.__file__}")

    # Try to create a client
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        logger.info(f"CARLA Client created: {client}")

        world = client.get_world()
        logger.info(f"CARLA World obtained: {world}")

        logger.info("CARLA connection successful!")
    except Exception as e:
        logger.error(f"Failed to connect to CARLA: {e}")

except ImportError as e:
    logger.error(f"Failed to import CARLA module: {e}")

    # Recursive search for .egg files
    logger.info("Searching for .egg files:")
    for root, _, files in os.walk(os.path.abspath(os.path.join(current_dir, '..', 'CarlaSimulator'))):
        for file in files:
            if file.endswith('.egg'):
                logger.info(f"  Found egg file: {os.path.join(root, file)}")

    # Recursive search for any Python packages
    logger.info("Searching for Python packages:")
    for root, dirs, files in os.walk(os.path.abspath(os.path.join(current_dir, '..', 'CarlaSimulator'))):
        if '__init__.py' in files:
            logger.info(f"  Found Python package: {root}")

    sys.exit(1)

print("Script completed successfully!")

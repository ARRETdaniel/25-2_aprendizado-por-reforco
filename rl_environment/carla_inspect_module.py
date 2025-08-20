#!/usr/bin/env python

"""
Test script to inspect the CARLA module.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the PythonAPI dir to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
carla_pythonapi_path = os.path.abspath(os.path.join(current_dir, '..', 'CarlaSimulator', 'PythonAPI'))
sys.path.append(carla_pythonapi_path)
logger.info(f"Added to path: {carla_pythonapi_path}")

# Add the carla dir to the path
carla_path = os.path.join(carla_pythonapi_path, 'carla')
sys.path.append(carla_path)
logger.info(f"Added to path: {carla_path}")

# Try to import CARLA
try:
    import carla
    logger.info("Successfully imported CARLA module")

    # Print module info
    logger.info(f"CARLA module path: {getattr(carla, '__file__', None)}")
    logger.info(f"CARLA module dict keys: {dir(carla)}")

    # Try to import the source module
    try:
        import carla.source
        logger.info("Successfully imported carla.source")
        logger.info(f"carla.source dict keys: {dir(carla.source)}")
    except ImportError as e:
        logger.error(f"Failed to import carla.source: {e}")

    # Try to find the source module directly
    try:
        carla_source_path = os.path.join(carla_path, 'source')
        if os.path.exists(carla_source_path):
            logger.info(f"Found carla/source directory: {carla_source_path}")
            sys.path.append(carla_source_path)
            logger.info(f"Added to path: {carla_source_path}")

            # List files in the directory
            logger.info("Files in carla/source:")
            for root, dirs, files in os.walk(carla_source_path):
                for file in files:
                    logger.info(f"  {os.path.join(root, file)}")
    except Exception as e:
        logger.error(f"Error inspecting carla/source: {e}")

except ImportError as e:
    logger.error(f"Failed to import CARLA module: {e}")

print("Script completed!")

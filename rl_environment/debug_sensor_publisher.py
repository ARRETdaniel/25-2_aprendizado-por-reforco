#!/usr/bin/env python

"""
Debug script for the CARLA sensor publisher.
This script runs the sensor publisher with verbose debugging.
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory: {current_dir}")

# Import the sensor publisher
sensor_publisher_path = os.path.join(current_dir, 'carla_sensor_publisher.py')
if not os.path.exists(sensor_publisher_path):
    logger.error(f"Sensor publisher script not found: {sensor_publisher_path}")
    sys.exit(1)

sys.path.append(current_dir)
try:
    # Import dynamically to avoid import errors if Python version is wrong
    import importlib.util
    spec = importlib.util.spec_from_file_location("carla_sensor_publisher", sensor_publisher_path)
    sensor_publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sensor_publisher)
    logger.info("Successfully imported sensor publisher")
except Exception as e:
    logger.error(f"Failed to import sensor publisher: {e}")
    traceback.print_exc()
    sys.exit(1)

def check_carla_bridge_directory():
    """Check if the CARLA bridge directory exists and is writable."""
    bridge_dir = Path.home() / ".carla_drl_bridge"

    logger.info(f"Checking bridge directory: {bridge_dir}")

    if not bridge_dir.exists():
        logger.warning(f"Bridge directory does not exist, creating: {bridge_dir}")
        try:
            bridge_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create bridge directory: {e}")
            return False

    # Check if writable
    try:
        test_file = bridge_dir / "write_test.txt"
        with open(test_file, 'w') as f:
            f.write("Test write")
        test_file.unlink()  # Remove test file
        logger.info(f"Bridge directory is writable: {bridge_dir}")
        return True
    except Exception as e:
        logger.error(f"Bridge directory is not writable: {e}")
        return False

def run_debug_sensor_publisher():
    """Run the CARLA sensor publisher with additional debugging."""
    # Check Python version - CARLA needs Python 3.6
    logger.info(f"Running with Python {sys.version}")
    if sys.version_info.major != 3 or sys.version_info.minor != 6:
        logger.warning(f"CARLA requires Python 3.6, but running with Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check bridge directory
    if not check_carla_bridge_directory():
        logger.error("Bridge directory check failed")
        return False

    # Create sensor publisher instance with shorter run time for testing
    try:
        publisher = sensor_publisher.CarlaSensorPublisher(
            host='localhost',
            port=2000,
            town='Town01',
            fps=20
        )

        # Add debug hooks
        original_connect = publisher.connect
        original_spawn_vehicle = publisher.spawn_vehicle
        original_setup_cameras = publisher.setup_cameras
        original_publish_observation = publisher.publish_observation

        def debug_connect(*args, **kwargs):
            logger.debug("Calling connect() method")
            result = original_connect(*args, **kwargs)
            logger.debug(f"connect() returned: {result}")
            return result

        def debug_spawn_vehicle(*args, **kwargs):
            logger.debug("Calling spawn_vehicle() method")
            result = original_spawn_vehicle(*args, **kwargs)
            logger.debug(f"spawn_vehicle() returned: {result}")
            return result

        def debug_setup_cameras(*args, **kwargs):
            logger.debug("Calling setup_cameras() method")
            result = original_setup_cameras(*args, **kwargs)
            logger.debug(f"setup_cameras() returned: {result}")
            return result

        def debug_publish_observation(*args, **kwargs):
            logger.debug("Calling publish_observation() method")
            result = original_publish_observation(*args, **kwargs)
            logger.debug(f"publish_observation() returned: {result}")
            return result

        # Replace methods with debug versions
        publisher.connect = debug_connect
        publisher.spawn_vehicle = debug_spawn_vehicle
        publisher.setup_cameras = debug_setup_cameras
        publisher.publish_observation = debug_publish_observation

        # Run for a shorter time (10 steps)
        logger.info("Running sensor publisher for 10 steps")
        success = publisher.run(num_steps=10)

        logger.info(f"Sensor publisher completed with status: {success}")
        return success

    except Exception as e:
        logger.error(f"Error running sensor publisher: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting debug sensor publisher")
    success = run_debug_sensor_publisher()
    sys.exit(0 if success else 1)

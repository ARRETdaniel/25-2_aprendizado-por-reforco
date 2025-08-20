#!/usr/bin/env python

"""
Debug test script for CARLA DRL integration.
This script attempts to connect to the bridge and wait until data is available.
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

# Import file-based bridge
current_dir = os.path.dirname(os.path.abspath(__file__))
file_bridge_path = os.path.join(current_dir, 'ros_bridge.py')
sys.path.append(os.path.dirname(file_bridge_path))
try:
    from ros_bridge import DRLBridge
    logger.info("Successfully imported ROS bridge")
except ImportError as e:
    logger.error(f"Failed to import ROS bridge: {e}")
    sys.exit(1)

def wait_for_sensor_data(timeout=60, check_interval=1):
    """Wait for sensor data to be available.

    Args:
        timeout: Timeout in seconds
        check_interval: Check interval in seconds

    Returns:
        True if data becomes available, False if timeout
    """
    bridge = DRLBridge()
    logger.info("Initialized DRL bridge")

    start_time = time.time()
    logger.info(f"Waiting for sensor data (timeout: {timeout}s)")

    while time.time() - start_time < timeout:
        # Check if any data is available
        cameras, state, reward, done, info = bridge.get_latest_observation()

        # Check if we have any data
        has_rgb = cameras['rgb'] is not None
        has_depth = cameras['depth'] is not None
        has_semantic = cameras['semantic'] is not None
        has_state = state is not None

        logger.info(f"RGB: {has_rgb}, Depth: {has_depth}, Semantic: {has_semantic}, State: {has_state}")

        if has_rgb or has_depth or has_semantic or has_state:
            logger.info("Sensor data is available!")
            return True

        logger.info("No sensor data available yet, waiting...")
        time.sleep(check_interval)

    logger.error(f"Timeout waiting for sensor data ({timeout}s)")
    return False

def main():
    """Main function."""
    success = wait_for_sensor_data()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

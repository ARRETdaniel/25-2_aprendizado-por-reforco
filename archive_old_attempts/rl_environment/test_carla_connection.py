"""
CARLA Connection Test Script

This script tests the connection to CARLA and verifies that we can get
camera data through the ROS bridge.
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ROS bridge
try:
    from ros_bridge import DRLBridge
    logger.info("Successfully imported ROS bridge")
except ImportError as e:
    logger.error(f"Failed to import ROS bridge: {e}")
    logger.error("Make sure ros_bridge.py is in the same directory")
    sys.exit(1)

def main():
    """Main function."""
    # Initialize DRL bridge
    bridge = DRLBridge(use_ros=False)  # Use file-based communication for simplicity
    logger.info("Initialized DRL bridge")

    # Create window for visualization
    cv2.namedWindow("CARLA Camera RGB", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("CARLA Camera Depth (if available)", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("CARLA Camera Semantic (if available)", cv2.WINDOW_AUTOSIZE)

    # Main loop
    try:
        for i in range(100):  # Try for 10 seconds
            logger.info(f"Getting observation {i+1}/100...")

            # Get the latest observation
            cameras, state, reward, done, info = bridge.get_latest_observation()

            # Log the data we received
            logger.info(f"RGB Camera: {None if cameras['rgb'] is None else cameras['rgb'].shape}")
            logger.info(f"Depth Camera: {None if cameras['depth'] is None else cameras['depth'].shape}")
            logger.info(f"Semantic Camera: {None if cameras['semantic'] is None else cameras['semantic'].shape}")
            logger.info(f"State: {state}")
            logger.info(f"Reward: {reward}")
            logger.info(f"Done: {done}")
            logger.info(f"Info: {info}")

            # Display images if available
            if cameras['rgb'] is not None:
                cv2.imshow("CARLA Camera RGB", cameras['rgb'])

            if cameras['depth'] is not None:
                # Normalize depth for visualization
                depth_normalized = cv2.normalize(cameras['depth'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("CARLA Camera Depth (if available)", depth_normalized)

            if cameras['semantic'] is not None:
                # Normalize semantic for visualization
                semantic_normalized = cv2.normalize(cameras['semantic'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("CARLA Camera Semantic (if available)", semantic_normalized)

            # Wait for key press
            key = cv2.waitKey(100)
            if key == 27:  # ESC key
                break

            time.sleep(0.1)

    finally:
        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Shutdown the bridge
        bridge.shutdown()

        logger.info("Test completed")

if __name__ == "__main__":
    main()

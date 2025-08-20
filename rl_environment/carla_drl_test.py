#!/usr/bin/env python

"""
Test script for CARLA DRL integration.
This script receives data from CARLA via the ROS bridge
and sends actions back.
"""

import os
import sys
import time
import logging
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

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

# Initialize DRL bridge
bridge = DRLBridge()
logger.info("Initialized DRL bridge")

# Create a directory for storing test results
output_dir = os.path.join(current_dir, 'test_results')
os.makedirs(output_dir, exist_ok=True)

def save_image(image, filename):
    """Save an image to disk."""
    if image is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved image: {filename}")
    else:
        logger.warning(f"Cannot save {filename}, image is None")

def test_get_observations(num_steps=10, save_interval=5):
    """Test getting observations from the CARLA environment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(output_dir, f"test_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)

    # Save test configuration
    config = {
        'timestamp': timestamp,
        'num_steps': num_steps,
        'save_interval': save_interval
    }
    with open(os.path.join(test_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # Lists to store data
    rgb_images = []
    depth_images = []
    semseg_images = []
    states = []
    rewards = []
    dones = []
    infos = []

    for step in range(num_steps):
        logger.info(f"Getting observation {step+1}/{num_steps}...")

        # Get observation using the get_latest_observation method
        cameras, state, reward, done, info = bridge.get_latest_observation()

        # Extract data from the cameras dictionary
        rgb_image = cameras['rgb']
        depth_image = cameras['depth']
        semseg_image = cameras['semantic']

        # Log observation
        logger.info(f"RGB Camera: {rgb_image.shape if rgb_image is not None else None}")
        logger.info(f"Depth Camera: {depth_image.shape if depth_image is not None else None}")
        logger.info(f"Semantic Camera: {semseg_image.shape if semseg_image is not None else None}")
        logger.info(f"State: {state}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Done: {done}")
        logger.info(f"Info: {info}")

        # Store data
        rgb_images.append(rgb_image)
        depth_images.append(depth_image)
        semseg_images.append(semseg_image)
        states.append(state)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)

        # Save images at intervals
        if step % save_interval == 0:
            if rgb_image is not None:
                save_image(rgb_image, os.path.join(test_dir, f"rgb_{step}.png"))
            if depth_image is not None:
                save_image(depth_image, os.path.join(test_dir, f"depth_{step}.png"))
            if semseg_image is not None:
                save_image(semseg_image, os.path.join(test_dir, f"semseg_{step}.png"))

        # Send a random action back
        action = np.random.uniform(-1, 1, size=2)  # [throttle, steering]
        bridge.publish_action(action)

        # Short delay between steps
        time.sleep(0.2)

    # Save all data
    test_data = {
        'rgb_images': rgb_images,
        'depth_images': depth_images,
        'semseg_images': semseg_images,
        'states': states,
        'rewards': rewards,
        'dones': dones,
        'infos': infos
    }
    with open(os.path.join(test_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    logger.info("Test completed")

    # Return success status
    return any(x is not None for x in rgb_images[-3:] if len(rgb_images) >= 3)

def main():
    """Main function."""
    try:
        success = test_get_observations(num_steps=20, save_interval=2)
        logger.info("Test completed successfully!" if success else "Test failed: missing data")
        return 0 if success else 1

    except Exception as e:
        logger.error(f"Error in test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

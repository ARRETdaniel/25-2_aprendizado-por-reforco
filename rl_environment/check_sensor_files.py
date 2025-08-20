#!/usr/bin/env python

"""
Check if there are any sensor data files in the bridge directory.
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

def check_sensor_files():
    """Check if there are any sensor data files in the bridge directory."""
    bridge_dir = Path.home() / ".carla_drl_bridge"

    if not bridge_dir.exists():
        logger.error(f"Bridge directory does not exist: {bridge_dir}")
        return

    # List all files with specific prefixes
    camera_rgb_files = list(bridge_dir.glob("camera_rgb_*.jpg"))
    camera_depth_files = list(bridge_dir.glob("camera_depth_*.jpg"))
    camera_semantic_files = list(bridge_dir.glob("camera_semantic_*.jpg"))
    state_files = list(bridge_dir.glob("state_*.npy"))
    reward_files = list(bridge_dir.glob("reward_*.json"))
    info_files = list(bridge_dir.glob("info_*.json"))
    action_files = list(bridge_dir.glob("action_*.npy"))

    # Check metadata files
    camera_rgb_latest = bridge_dir / "camera_rgb_latest.json"
    camera_depth_latest = bridge_dir / "camera_depth_latest.json"
    camera_semantic_latest = bridge_dir / "camera_semantic_latest.json"
    state_latest = bridge_dir / "state_latest.txt"
    reward_latest = bridge_dir / "reward_latest.txt"
    info_latest = bridge_dir / "info_latest.txt"
    action_latest = bridge_dir / "action_latest.txt"

    # Report findings
    logger.info(f"Camera RGB files: {len(camera_rgb_files)}")
    logger.info(f"Camera depth files: {len(camera_depth_files)}")
    logger.info(f"Camera semantic files: {len(camera_semantic_files)}")
    logger.info(f"State files: {len(state_files)}")
    logger.info(f"Reward files: {len(reward_files)}")
    logger.info(f"Info files: {len(info_files)}")
    logger.info(f"Action files: {len(action_files)}")

    logger.info(f"Camera RGB latest exists: {camera_rgb_latest.exists()}")
    logger.info(f"Camera depth latest exists: {camera_depth_latest.exists()}")
    logger.info(f"Camera semantic latest exists: {camera_semantic_latest.exists()}")
    logger.info(f"State latest exists: {state_latest.exists()}")
    logger.info(f"Reward latest exists: {reward_latest.exists()}")
    logger.info(f"Info latest exists: {info_latest.exists()}")
    logger.info(f"Action latest exists: {action_latest.exists()}")

    # If no sensor data is found, but actions exist, there's a one-way communication
    if (len(camera_rgb_files) == 0 and len(camera_depth_files) == 0 and
        len(camera_semantic_files) == 0 and len(state_files) == 0 and
        len(reward_files) == 0 and len(info_files) == 0 and
        len(action_files) > 0):
        logger.warning("Only action files found, no sensor data. One-way communication issue detected.")

if __name__ == "__main__":
    logger.info("Checking for sensor data files")
    check_sensor_files()

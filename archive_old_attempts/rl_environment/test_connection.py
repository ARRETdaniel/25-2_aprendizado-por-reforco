#!/usr/bin/env python
"""
CARLA Environment Connection Test

This script tests the connection to the CARLA server using the
modified environment wrapper. It creates an environment instance,
resets it, takes a few steps, and then closes it, verifying that
the connection is established and released correctly.

Author: Autonomous Driving Research Team
Date: August 15, 2025
"""

import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ConnectionTest")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import environment wrapper
from rl_environment import CarlaEnvWrapper

def test_environment_connection():
    """Test environment connection to CARLA."""
    logger.info("Creating environment...")

    try:
        # Create environment with default parameters
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            image_shape=(84, 84, 3),  # Lower resolution for quick testing
        )
        logger.info("Environment created successfully!")

        # Test reset
        logger.info("Resetting environment...")
        initial_observation = env.reset()
        logger.info(f"Initial observation shape: {initial_observation.shape}")

        # Test a few steps
        logger.info("Taking 5 steps with random actions...")
        for i in range(5):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            logger.info(f"Step {i+1}: reward={reward:.2f}, done={done}")
            if done:
                logger.info("Episode terminated early, resetting...")
                env.reset()

        # Properly close the environment
        logger.info("Closing environment...")
        env.close()
        logger.info("Environment closed successfully!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting CARLA environment connection test...")

    # Wait a moment to ensure CARLA server is ready
    time.sleep(1)

    success = test_environment_connection()

    if success:
        logger.info("✅ Connection test passed!")
        sys.exit(0)
    else:
        logger.error("❌ Connection test failed!")
        sys.exit(1)

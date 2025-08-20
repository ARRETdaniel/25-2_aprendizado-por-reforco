#!/usr/bin/env python
"""
Simplified SAC implementation test for CARLA.

This script runs a quick test of the SAC implementation with the CARLA environment,
using a minimal configuration to quickly verify basic functionality works.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path

# Add project root to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment.environment import CarlaEnvWrapper
from rl_environment.examples.sac_agent import SAC, train_sac

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test_run(n_episodes=5, max_steps=200):
    """
    Run a quick test of the SAC implementation with minimal episodes and steps.

    Args:
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        True if the test was successful, False otherwise
    """
    try:
        # Initialize environment with minimal settings
        logger.info("Creating environment with minimal settings...")
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            city_name='Town01',
            image_size=(84, 84),
            frame_skip=1,
            max_episode_steps=max_steps,
            weather_id=0,
            quality_level='Low'
        )

        # Define state and action spaces
        logger.info("Setting up state and action spaces...")
        state_space = {
            'image': (84, 84, 3),
            'vehicle_state': (9,),
            'navigation': (3,),
            'detections': (10,)
        }
        action_space = (2,)  # Steering and throttle

        # Initialize agent with minimal config
        logger.info("Initializing SAC agent...")
        agent = SAC(
            state_space=state_space,
            action_dim=action_space[0],  # Use correct parameter name
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            tau=0.005,
            buffer_size=10000,  # Reduced for quick testing
            batch_size=64,
            feature_dim=128,    # Reduced for quick testing
            hidden_dim=128,     # Reduced for quick testing
            update_frequency=2,
            seed=42
        )

        # Train agent for a few episodes
        logger.info(f"Starting quick test run for {n_episodes} episodes...")
        stats = train_sac(
            env=env,
            agent=agent,
            n_episodes=n_episodes,
            max_steps=max_steps,
            checkpoint_dir='./test_checkpoints',
            checkpoint_freq=n_episodes,  # Save at the end only
            eval_freq=n_episodes,        # Evaluate at the end only
            eval_episodes=1
        )

        # Check if we have non-zero rewards and updates
        has_rewards = len(stats['episode_rewards']) > 0 and any(r != 0 for r in stats['episode_rewards'])
        has_updates = len(stats['critic_losses']) > 0

        # Close environment
        env.close()

        if has_rewards and has_updates:
            logger.info("Quick test successful: Agent received rewards and performed updates")
            return True
        else:
            logger.warning("Quick test suspicious: Agent didn't receive rewards or perform updates")
            return False

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        if 'env' in locals():
            env.close()
        return False
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        if 'env' in locals():
            env.close()
        return False

def main():
    """
    Main function to run quick test.
    """
    logger.info("Starting quick test of SAC implementation...")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")

    # Create checkpoints directory if it doesn't exist
    os.makedirs('./test_checkpoints', exist_ok=True)

    # Run quick test
    success = quick_test_run()

    if success:
        logger.info("Quick test completed successfully!")
    else:
        logger.warning("Quick test had issues - check logs for details")

    return success

if __name__ == "__main__":
    main()

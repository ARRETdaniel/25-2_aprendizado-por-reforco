"""
Integration test for the CARLA RL environment.

This script tests the complete integration of all components:
- Environment wrapper
- State processing
- Action processing
- Reward function

It runs a simple random agent for a few episodes and logs results.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegrationTest")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError as e:
    logger.error(f"Failed to import CarlaEnvWrapper: {e}")
    sys.exit(1)

def test_environment(episodes: int = 3, max_steps: int = 200) -> None:
    """
    Test the environment with a random agent.

    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    logger.info("Starting environment integration test with random agent")

    try:
        # Initialize environment
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            city_name='Town01',
            image_size=(84, 84),
            frame_skip=2,
            max_episode_steps=max_steps,
            weather_id=0,
            quality_level='Low',
            random_start=True
        )
        logger.info("Environment initialized successfully")

        # Run episodes with random actions
        for episode in range(1, episodes + 1):
            logger.info(f"Starting episode {episode}/{episodes}")

            # Reset environment
            observation = env.reset()
            logger.info(f"Environment reset. Observation keys: {list(observation.keys())}")

            # Episode loop
            total_reward = 0.0
            done = False
            step = 0

            while not done and step < max_steps:
                # Generate random action
                if env.action_space_type == 'continuous':
                    # Continuous actions: throttle, brake, steering
                    action = np.random.uniform(low=-1, high=1, size=3)
                    # Normalize throttle and brake to [0, 1]
                    action[0] = (action[0] + 1) / 2  # throttle
                    action[1] = (action[1] + 1) / 2  # brake
                else:
                    # Discrete action
                    action = np.random.randint(0, 9)  # 9 discrete actions

                # Step environment
                observation, reward, done, info = env.step(action)

                # Update metrics
                total_reward += reward
                step += 1

                # Log every 10 steps
                if step % 10 == 0:
                    logger.info(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
                    if 'vehicle_state' in observation:
                        speed = np.linalg.norm(observation['vehicle_state'][3:6])  # vel x,y,z
                        logger.info(f"Vehicle speed: {speed:.2f} m/s")

            # Episode summary
            logger.info(f"Episode {episode} finished: Steps = {step}, Total Reward = {total_reward:.3f}")
            logger.info(f"Termination reason: {info.get('termination_reason', 'unknown')}")

        # Close environment
        env.close()
        logger.info("Environment closed successfully")
        logger.info("Integration test completed successfully")

    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        test_environment()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

"""
Example script demonstrating how to use the CARLA environment wrapper.
"""

import os
import sys
import numpy as np
import time
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import rl_environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError:
    logger.error("Failed to import CarlaEnvWrapper. Make sure the environment is correctly installed.")
    sys.exit(1)


def random_agent_demo():
    """
    Demo using a random agent with the CARLA environment wrapper.
    """
    logger.info("Starting random agent demo...")

    # Create the environment
    try:
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            town='Town01',
            fps=30,
            image_size=(84, 84),
            frame_skip=2,
            max_episode_steps=1000
        )
        logger.info("Environment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    try:
        # Run for a few episodes
        for episode in range(3):
            logger.info(f"Starting episode {episode+1}")

            # Reset the environment
            state = env.reset()
            done = False
            total_reward = 0.0
            step = 0

            # Run episode
            while not done:
                # Generate random action
                # [throttle, brake, steering]
                action = np.array([
                    random.uniform(0.0, 1.0),   # Throttle
                    random.uniform(0.0, 0.3),   # Brake
                    random.uniform(-0.3, 0.3)   # Steering
                ])

                # Take a step
                next_state, reward, done, info = env.step(action)

                total_reward += reward
                step += 1

                if step % 10 == 0:
                    logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

                # Sleep to slow down the demo
                time.sleep(0.01)

            logger.info(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    except Exception as e:
        logger.error(f"Error during demo: {e}")
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        logger.info("Demo ended.")


def main():
    """
    Main function.
    """
    random_agent_demo()


if __name__ == "__main__":
    main()

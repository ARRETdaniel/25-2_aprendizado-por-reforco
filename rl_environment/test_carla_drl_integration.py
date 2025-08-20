"""
CARLA DRL Test Script

This script tests the integration of our DRL components with the CARLA simulator,
focusing on:
1. Camera visualization through ROS bridge
2. Basic state observation and action execution
3. Simple SAC agent interaction

Usage:
    python test_carla_drl_integration.py --quality Low
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import OpenCV for visualization
try:
    import cv2
    HAS_CV2 = True
    logger.info("Successfully imported OpenCV")
except ImportError:
    logger.warning("OpenCV not found, visualization will be disabled")
    HAS_CV2 = False

# Try to import ROS bridge
try:
    from ros_bridge import DRLBridge
    logger.info("Successfully imported ROS bridge")
except ImportError as e:
    logger.error(f"Failed to import ROS bridge: {e}")
    logger.error("Make sure ros_bridge.py is in the same directory")
    sys.exit(1)

# Try to import Enhanced SAC
try:
    from enhanced_sac import EnhancedSAC, SACConfig, ReplayBuffer
    logger.info("Successfully imported Enhanced SAC")
except ImportError as e:
    logger.error(f"Failed to import Enhanced SAC: {e}")
    logger.error("Make sure enhanced_sac.py is in the same directory")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test CARLA DRL integration")

    # General settings
    parser.add_argument('--use-ros', action='store_true', help='Use ROS 2 for communication')
    parser.add_argument('--quality', default='Low', choices=['Low', 'Epic'],
                        help='CARLA graphics quality')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--show', action='store_true', help='Show camera feeds')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to load checkpoint')

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()

    # Initialize DRL bridge
    bridge = DRLBridge(use_ros=args.use_ros)
    logger.info("Initialized DRL bridge")

    # Initialize SAC config
    sac_config = SACConfig(
        random_seed=42,
        device="auto",  # Auto-select CUDA if available
        checkpoint_dir="./checkpoints/sac_carla",
        log_dir="./logs/sac_carla",
        plot_dir="./plots",
        image_observation=True,
        vector_dim=9,  # Assuming a 9-dimensional vector state
        action_dim=2,  # Throttle and steering
        image_channels=3,
        image_height=84,
        image_width=84
    )

    # Create directories if they don't exist
    os.makedirs(sac_config.checkpoint_dir, exist_ok=True)
    os.makedirs(sac_config.log_dir, exist_ok=True)
    os.makedirs(sac_config.plot_dir, exist_ok=True)

    # Initialize SAC agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set device in config
    sac_config.device = str(device)

    agent = EnhancedSAC(sac_config)
    logger.info("Initialized Enhanced SAC agent")    # Load checkpoint if provided
    if args.checkpoint is not None:
        agent.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Initialize replay buffer
    buffer_size = 100000  # Define a reasonable buffer size
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        image_observation=sac_config.image_observation,
        image_shape=(sac_config.image_channels, sac_config.image_height, sac_config.image_width),
        vector_dim=sac_config.vector_dim,
        action_dim=sac_config.action_dim,
        device=device
    )

    # Wait for CARLA to start publishing data
    logger.info("Waiting for CARLA to start...")
    time.sleep(2.0)

    # Create a window for visualization if needed
    if args.show and HAS_CV2:
        cv2.namedWindow("CARLA Camera", cv2.WINDOW_AUTOSIZE)

    # Main loop
    total_rewards = []

    for episode in range(args.episodes):
        logger.info(f"Starting episode {episode+1}/{args.episodes}")

        # Reset episode variables
        step = 0
        total_reward = 0.0
        done = False

        # Reset environment
        # Send reset command to CARLA
        bridge.publish_control("reset", {"seed": episode})
        time.sleep(1.0)  # Wait for reset to complete

        # Get initial observation
        cameras, state, reward, done, info = bridge.get_latest_observation()

        # Main episode loop
        while not done and step < args.steps:
            # Process state
            if state is not None:
                vector_state = state
            else:
                # Default state if not available
                vector_state = np.zeros(sac_config.vector_dim, dtype=np.float32)

            # Process image
            if cameras['rgb'] is not None:
                # Resize image to match SAC configuration
                image = cv2.resize(cameras['rgb'], (sac_config.image_width, sac_config.image_height))

                # Normalize image
                image = image.astype(np.float32) / 255.0

                # Show image if required
                if args.show and HAS_CV2:
                    cv2.imshow("CARLA Camera", cameras['rgb'])
                    cv2.waitKey(1)
            else:
                # Default image if not available
                image = np.zeros((sac_config.image_height, sac_config.image_width, 3), dtype=np.float32)            # Select action using agent
            observation = {
                'image': np.transpose(image, (2, 0, 1)),  # Convert from HWC to CHW format
                'vector': vector_state
            }
            action = agent.select_action(observation)

            # Send action to CARLA
            bridge.publish_action(action)

            # Wait for CARLA to process action
            time.sleep(0.05)

            # Get next state
            next_cameras, next_state, reward, done, info = bridge.get_latest_observation()

            # Default reward if not available
            if reward is None:
                reward = 0.0

            # Process next state
            if next_state is not None:
                next_vector_state = next_state
            else:
                # Default state if not available
                next_vector_state = np.zeros(sac_config.vector_dim, dtype=np.float32)

            # Process next image
            if next_cameras['rgb'] is not None:
                # Resize image to match SAC configuration
                next_image = cv2.resize(next_cameras['rgb'], (sac_config.image_width, sac_config.image_height))

                # Normalize image
                next_image = next_image.astype(np.float32) / 255.0
            else:
                # Default image if not available
                next_image = np.zeros((sac_config.image_height, sac_config.image_width, 3), dtype=np.float32)            # Store transition in replay buffer
            observation = {'image': np.transpose(image, (2, 0, 1)), 'vector': vector_state}
            next_observation = {'image': np.transpose(next_image, (2, 0, 1)), 'vector': next_vector_state}

            replay_buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done
            )

            # Update state
            image = next_image
            vector_state = next_vector_state
            cameras = next_cameras
            state = next_state

            # Update counters
            step += 1
            total_reward += reward

            # Log progress
            if step % 10 == 0:
                logger.info(f"Episode {episode+1}, Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

        # End of episode
        logger.info(f"Episode {episode+1} finished after {step} steps, Total reward: {total_reward:.2f}")
        total_rewards.append(total_reward)

        # Check if we have enough data to train
        # Note: In this test script we're not training the agent
        # since the update_parameters method expects an internal replay buffer
        # In a full implementation, we would need to integrate the replay buffer with the agent
        if len(replay_buffer) > 0:
            logger.info(f"Collected {len(replay_buffer)} transitions")
            # We're not training in this test script
            # In a full implementation, we would do:
            # agent.replay_buffer = replay_buffer  # Set the replay buffer
            # for _ in range(min(step, 50)):  # Update up to 50 times
            #     agent.update_parameters(batch_size=256)

        # Save checkpoint after each episode
        checkpoint_path = os.path.join(sac_config.checkpoint_dir, f"sac_episode_{episode+1}")
        agent.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Close visualization window if open
    if args.show and HAS_CV2:
        cv2.destroyAllWindows()

    # Plot total rewards
    if len(total_rewards) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(total_rewards)+1), total_rewards, marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards")
        plt.grid(True)
        plt.savefig(os.path.join(sac_config.plot_dir, "episode_rewards.png"))
        plt.close()
        logger.info(f"Saved reward plot to {os.path.join(sac_config.plot_dir, 'episode_rewards.png')}")

    # Clean up
    bridge.shutdown()
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    main()

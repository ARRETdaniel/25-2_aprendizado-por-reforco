"""
CARLA Environment Wrapper for DRL

This module provides a Gym-like environment wrapper for the CARLA simulator,
optimized for deep reinforcement learning. It communicates with CARLA through
the ROS 2 bridge and provides a standardized interface for DRL algorithms.
"""

import os
import sys
import time
import logging
import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

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

# Try to import OpenCV for visualization
try:
    import cv2
    HAS_CV2 = True
    logger.info("Successfully imported OpenCV")
except ImportError:
    logger.warning("OpenCV not found, visualization will be disabled")
    HAS_CV2 = False


@dataclass
class CarlaEnvConfig:
    """Configuration for CARLA environment."""
    # General
    random_seed: int = 42
    timeout: float = 10.0  # Timeout for waiting for observations

    # State and action
    use_image_observations: bool = True  # Whether to use camera images in observations
    use_lidar: bool = False  # Whether to use LiDAR in observations
    action_smoothing: float = 0.9  # Action smoothing factor (0-1)

    # Environment parameters
    fps: int = 10  # Frames per second for environment stepping
    max_episode_steps: int = 1000  # Maximum steps per episode

    # Reward parameters
    reward_speed_factor: float = 0.2  # Reward factor for speed
    reward_collision_penalty: float = -100.0  # Penalty for collision
    reward_lane_penalty: float = -0.5  # Penalty for lane invasion
    reward_progress_factor: float = 1.0  # Reward factor for progress
    reward_steering_penalty: float = -0.1  # Penalty for excessive steering

    # Observation normalization
    normalize_obs: bool = True  # Whether to normalize observations

    # Visualization
    render: bool = True  # Whether to render the environment
    render_mode: str = 'human'  # Render mode ('human', 'rgb_array')

    # Communication
    use_ros: bool = True  # Whether to use ROS 2 for communication


class CarlaEnv(gym.Env):
    """
    A Gym-like environment wrapper for the CARLA simulator.

    This class provides a standardized interface for DRL algorithms to interact with
    the CARLA simulator through the ROS 2 bridge. It follows the OpenAI Gym interface
    with step(), reset(), render(), and close() methods.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: CarlaEnvConfig = None):
        """Initialize the CARLA environment.

        Args:
            config: Configuration for the environment
        """
        super().__init__()

        # Use default config if not provided
        self.config = config or CarlaEnvConfig()

        # Set random seed
        np.random.seed(self.config.random_seed)

        # Initialize ROS bridge
        self.ros_bridge = DRLBridge(use_ros=self.config.use_ros)

        # Initialize state variables
        self.current_obs = None
        self.current_image = None
        self.current_info = {}
        self.steps_in_episode = 0
        self.episode_reward = 0.0
        self.previous_location = None
        self.current_speed = 0.0
        self.collision_count = 0
        self.lane_invasion_count = 0
        self.last_action = np.zeros(2)  # [throttle, steer]

        # Define action and observation spaces
        # Action: [throttle, steer]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # Full brake, full left
            high=np.array([1.0, 1.0]),   # Full throttle, full right
            dtype=np.float32
        )

        # Observation space depends on whether images are used
        if self.config.use_image_observations:
            # Image observations (simplified)
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                'vector': spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            })
        else:
            # Vector observations only
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
            )

        # Wait for initial observations
        self._wait_for_observations(timeout=self.config.timeout)

        logger.info("CARLA environment initialized")

    def _wait_for_observations(self, timeout: float = 10.0) -> bool:
        """Wait for observations from ROS bridge.

        Args:
            timeout: Timeout in seconds

        Returns:
            Whether observations were received
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            camera, state, reward, done, info = self.ros_bridge.get_latest_observation()

            if state is not None:
                self.current_obs = state
                self.current_image = camera
                self.current_info = info or {}
                return True

            time.sleep(0.1)

        logger.warning(f"Timeout waiting for observations after {timeout} seconds")
        return False

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess camera image for RL.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Preprocessed image (84, 84, 3)
        """
        if image is None:
            return np.zeros((84, 84, 3), dtype=np.uint8)

        # Resize to standard size
        image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)

        return image

    def _normalize_observation(self, state: np.ndarray) -> np.ndarray:
        """Normalize state observation.

        Args:
            state: State array

        Returns:
            Normalized state array
        """
        if not self.config.normalize_obs:
            return state

        # Apply normalization to specific state dimensions
        # This is just an example - adapt to your specific state representation
        normalized = state.copy()

        # Normalize position within reasonable bounds
        normalized[0:3] = normalized[0:3] / 100.0  # Position (x, y, z)

        # Normalize rotation to [-1, 1]
        normalized[3:6] = normalized[3:6] / 180.0  # Rotation (pitch, yaw, roll)

        # Normalize speed to [0, 1] assuming max speed of 30 m/s
        normalized[6] = normalized[6] / 30.0  # Speed

        # Normalize collision values
        normalized[7:10] = np.clip(normalized[7:10] / 10.0, 0.0, 1.0)  # Collisions

        return normalized

    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> Tuple[float, bool]:
        """Compute reward and done flag.

        Args:
            state: State array
            action: Action array

        Returns:
            Tuple of (reward, done)
        """
        # Extract relevant state information
        pos_x, pos_y, pos_z = state[0:3]
        speed = state[6]  # m/s
        collision_vehicles = state[7]
        collision_pedestrians = state[8]
        collision_other = state[9]
        intersection_otherlane = state[10]
        intersection_offroad = state[11]

        # Initialize reward
        reward = 0.0
        done = False

        # Check for collisions
        collision_detected = (collision_vehicles + collision_pedestrians + collision_other) > 0
        if collision_detected:
            reward += self.config.reward_collision_penalty
            self.collision_count += 1
            done = True
            logger.info(f"Collision detected! Total collisions: {self.collision_count}")

        # Reward for speed (encourage moving forward)
        reward += self.config.reward_speed_factor * speed

        # Penalty for lane invasion
        lane_invasion = intersection_otherlane + intersection_offroad
        if lane_invasion > 0:
            reward += self.config.reward_lane_penalty * lane_invasion
            self.lane_invasion_count += 1

        # Penalty for excessive steering (to encourage smooth driving)
        steering = action[1] if action is not None else 0.0
        reward += self.config.reward_steering_penalty * abs(steering)

        # Reward for progress (distance traveled)
        if self.previous_location is not None:
            dx = pos_x - self.previous_location[0]
            dy = pos_y - self.previous_location[1]
            distance_traveled = np.sqrt(dx*dx + dy*dy)
            reward += self.config.reward_progress_factor * distance_traveled

        # Update previous location
        self.previous_location = (pos_x, pos_y, pos_z)

        # Store current speed
        self.current_speed = speed

        # Check if maximum steps reached
        if self.steps_in_episode >= self.config.max_episode_steps:
            done = True
            logger.info(f"Episode ended after reaching max steps: {self.config.max_episode_steps}")

        return reward, done

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action smoothing.

        Args:
            action: Action array [throttle, steer]

        Returns:
            Smoothed action array
        """
        if self.config.action_smoothing <= 0.0:
            return action

        # Apply exponential smoothing
        alpha = 1.0 - self.config.action_smoothing
        smoothed_action = alpha * action + (1.0 - alpha) * self.last_action
        self.last_action = smoothed_action

        return smoothed_action

    def _process_observation(self) -> Union[Dict, np.ndarray]:
        """Process observation for agent.

        Returns:
            Processed observation
        """
        if self.current_obs is None:
            if self.config.use_image_observations:
                return {
                    'image': np.zeros((84, 84, 3), dtype=np.uint8),
                    'vector': np.zeros(10, dtype=np.float32)
                }
            else:
                return np.zeros(17, dtype=np.float32)

        # Normalize state
        normalized_state = self._normalize_observation(self.current_obs)

        # Process image if using image observations
        if self.config.use_image_observations:
            processed_image = self._preprocess_image(self.current_image)

            # Extract key features for vector part (simplified for dict observation)
            vector_obs = np.array([
                normalized_state[3],   # pitch
                normalized_state[4],   # yaw
                normalized_state[5],   # roll
                normalized_state[6],   # speed
                normalized_state[7],   # collision_vehicles
                normalized_state[8],   # collision_pedestrians
                normalized_state[9],   # collision_other
                normalized_state[10],  # intersection_otherlane
                normalized_state[11],  # intersection_offroad
                normalized_state[12]   # steer
            ], dtype=np.float32)

            return {
                'image': processed_image,
                'vector': vector_obs
            }
        else:
            # Return full normalized state
            return normalized_state

    def step(self, action: np.ndarray) -> Tuple[Union[Dict, np.ndarray], float, bool, Dict]:
        """Take a step in the environment.

        Args:
            action: Action array [throttle, steer]

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Apply action smoothing
        smoothed_action = self._smooth_action(action)

        # Send action to CARLA via ROS bridge
        self.ros_bridge.publish_action(smoothed_action)

        # Wait for the environment to step (account for FPS)
        time.sleep(1.0 / self.config.fps)

        # Get new observation
        self._wait_for_observations(timeout=self.config.timeout)

        # Compute reward and check if done
        reward, done = self._compute_reward(self.current_obs, smoothed_action)

        # Process observation
        obs = self._process_observation()

        # Update step counter and episode reward
        self.steps_in_episode += 1
        self.episode_reward += reward

        # Prepare info dictionary
        info = {
            'speed': self.current_speed,
            'collision_count': self.collision_count,
            'lane_invasion_count': self.lane_invasion_count,
            'steps': self.steps_in_episode,
            'episode_reward': self.episode_reward
        }

        # Add any additional info from CARLA
        info.update(self.current_info)

        return obs, reward, done, info

    def reset(self) -> Union[Dict, np.ndarray]:
        """Reset the environment.

        Returns:
            Initial observation
        """
        # Reset state variables
        self.steps_in_episode = 0
        self.episode_reward = 0.0
        self.previous_location = None
        self.current_speed = 0.0
        self.last_action = np.zeros(2)  # [throttle, steer]

        # Reset CARLA via ROS bridge
        seed = np.random.randint(0, 1000000) if self.config.random_seed < 0 else self.config.random_seed
        self.ros_bridge.publish_control("reset", {"seed": seed})

        # Wait for initial observation
        if not self._wait_for_observations(timeout=self.config.timeout):
            logger.warning("Failed to get observation after reset")

        # Process and return initial observation
        return self._process_observation()

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment.

        Args:
            mode: Render mode ('human' or 'rgb_array')

        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        if not self.config.render:
            return None

        if self.current_image is None:
            return None

        if mode == 'rgb_array':
            return self.current_image

        if mode == 'human' and HAS_CV2:
            # Create a copy to avoid modifying the original image
            display_image = self.current_image.copy()

            # Add information overlay
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Add speed
            cv2.putText(display_image, f"Speed: {self.current_speed:.1f} m/s",
                        (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Add episode info
            cv2.putText(display_image, f"Step: {self.steps_in_episode}",
                        (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Add reward
            cv2.putText(display_image, f"Reward: {self.episode_reward:.1f}",
                        (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the image
            cv2.imshow('CARLA Environment', display_image)
            cv2.waitKey(1)

        return None

    def close(self):
        """Clean up resources."""
        # Close ROS bridge
        if self.ros_bridge is not None:
            self.ros_bridge.shutdown()

        # Close OpenCV windows
        if HAS_CV2 and self.config.render:
            cv2.destroyAllWindows()

        logger.info("CARLA environment closed")


# Test code
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CARLA environment wrapper")
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=100, help='Steps per episode')
    parser.add_argument('--no-ros', action='store_true', help='Disable ROS bridge')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create environment configuration
    config = CarlaEnvConfig(
        random_seed=args.seed,
        max_episode_steps=args.steps,
        render=args.render,
        use_ros=not args.no_ros
    )

    # Create environment
    env = CarlaEnv(config)

    try:
        # Run test episodes
        for episode in range(args.episodes):
            logger.info(f"Starting episode {episode+1}/{args.episodes}")

            observation = env.reset()
            episode_reward = 0.0

            for step in range(args.steps):
                # Sample a random action
                action = env.action_space.sample()

                # Take a step
                next_observation, reward, done, info = env.step(action)

                # Render if enabled
                env.render()

                # Update episode reward
                episode_reward += reward

                # Log step information
                logger.info(f"Step {step+1}: Reward = {reward:.2f}, Total = {episode_reward:.2f}")

                # Check if done
                if done:
                    logger.info(f"Episode {episode+1} finished after {step+1} steps with reward {episode_reward:.2f}")
                    break

            logger.info(f"Episode {episode+1} complete with reward {episode_reward:.2f}")

    finally:
        # Close environment
        env.close()

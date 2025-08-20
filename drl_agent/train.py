"""
DRL Agent Training Module for CARLA Autonomous Driving

This module implements PPO-based deep reinforcement learning for autonomous
driving in the CARLA simulator, using ROS 2 for communication.
"""
import os
import sys
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import yaml
from dataclasses import dataclass

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge

# DRL imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym
from gymnasium import spaces

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Configuration imports
sys.path.append(str(Path(__file__).parent.parent / "configs"))
from config_models import AlgorithmConfig, load_train_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CarlaObservation:
    """Structured observation from CARLA environment."""
    image: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    collision: bool
    lane_invasion: bool
    episode_status: str


class CarlaROS2Environment(gym.Env):
    """
    Gymnasium environment wrapper for CARLA via ROS 2.
    
    Receives sensor data from ROS 2 topics and sends control commands
    back to CARLA through the gateway node.
    """
    
    def __init__(self, config: AlgorithmConfig):
        """
        Initialize CARLA ROS 2 environment.
        
        Args:
            config: Training configuration object
        """
        super().__init__()
        
        self.config = config
        self.bridge = CvBridge()
        
        # Initialize ROS 2
        rclpy.init()
        self.node = Node('carla_drl_agent')
        
        # Environment state
        self.current_obs: Optional[CarlaObservation] = None
        self.episode_step = 0
        self.episode_return = 0.0
        self.last_progress = 0.0
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Setup ROS 2 communication
        self._setup_ros_communication()
        
        # Episode management
        self.max_episode_steps = config.environment.episode.max_steps
        
        logger.info("CARLA ROS 2 Environment initialized")
        
    def _setup_spaces(self) -> None:
        """Setup Gymnasium action and observation spaces."""
        # Action space: [throttle, steering] both in [-1, 1]
        # throttle: 0 to 1, steering: -1 to 1
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: RGB image + velocity + angular velocity
        img_config = self.config.environment.observation_space.image_size
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=img_config,
                dtype=np.uint8
            ),
            'velocity': spaces.Box(
                low=-50.0, high=50.0,
                shape=(3,),
                dtype=np.float32
            ),
            'angular_velocity': spaces.Box(
                low=-10.0, high=10.0,
                shape=(3,),
                dtype=np.float32
            )
        })
        
    def _setup_ros_communication(self) -> None:
        """Setup ROS 2 publishers and subscribers."""
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.camera_sub = self.node.create_subscription(
            Image, '/carla/ego_vehicle/camera/image',
            self._camera_callback, sensor_qos
        )
        
        self.imu_sub = self.node.create_subscription(
            Imu, '/carla/ego_vehicle/imu',
            self._imu_callback, sensor_qos
        )
        
        self.odom_sub = self.node.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry',
            self._odometry_callback, sensor_qos
        )
        
        self.collision_sub = self.node.create_subscription(
            Bool, '/carla/ego_vehicle/collision',
            self._collision_callback, reliable_qos
        )
        
        self.lane_invasion_sub = self.node.create_subscription(
            Bool, '/carla/ego_vehicle/lane_invasion',
            self._lane_invasion_callback, reliable_qos
        )
        
        self.status_sub = self.node.create_subscription(
            String, '/carla/ego_vehicle/status',
            self._status_callback, reliable_qos
        )
        
        # Publisher
        self.cmd_vel_pub = self.node.create_publisher(
            Twist, '/carla/ego_vehicle/cmd_vel', reliable_qos
        )
        
        # Initialize observation data
        self._reset_observation_data()
        
    def _reset_observation_data(self) -> None:
        """Reset observation data to default values."""
        self._latest_image = np.zeros((84, 84, 3), dtype=np.uint8)
        self._latest_velocity = np.zeros(3, dtype=np.float32)
        self._latest_angular_velocity = np.zeros(3, dtype=np.float32)
        self._collision_flag = False
        self._lane_invasion_flag = False
        self._episode_status = "running"
        
    def _camera_callback(self, msg: Image) -> None:
        """Process camera image messages."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Resize to expected size (84x84)
            self._latest_image = cv_image[:84, :84, :]
            
        except Exception as e:
            logger.error(f"Error processing camera image: {e}")
            
    def _imu_callback(self, msg: Imu) -> None:
        """Process IMU messages."""
        self._latest_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float32)
        
    def _odometry_callback(self, msg: Odometry) -> None:
        """Process odometry messages."""
        self._latest_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ], dtype=np.float32)
        
    def _collision_callback(self, msg: Bool) -> None:
        """Process collision events."""
        if msg.data:
            self._collision_flag = True
            logger.warning("Collision detected in environment")
            
    def _lane_invasion_callback(self, msg: Bool) -> None:
        """Process lane invasion events."""
        if msg.data:
            self._lane_invasion_flag = True
            logger.warning("Lane invasion detected in environment")
            
    def _status_callback(self, msg: String) -> None:
        """Process episode status updates."""
        self._episode_status = msg.data
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from sensor data."""
        # Spin ROS node to process callbacks
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # Normalize image if configured
        image = self._latest_image.astype(np.float32)
        if self.config.environment.observation_space.normalize_images:
            image = image / 255.0
            
        return {
            'image': image,
            'velocity': self._latest_velocity,
            'angular_velocity': self._latest_angular_velocity
        }
        
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        reward = 0.0
        reward_config = self.config.environment.reward_function
        
        # Speed reward (encourage forward movement)
        forward_speed = max(0.0, self._latest_velocity[0])
        reward += reward_config.speed_reward_weight * forward_speed
        
        # Collision penalty
        if self._collision_flag:
            reward += reward_config.collision_penalty
            
        # Lane invasion penalty
        if self._lane_invasion_flag:
            reward += reward_config.lane_invasion_penalty
            
        # Comfort penalty (penalize harsh acceleration/steering)
        angular_speed = abs(self._latest_angular_velocity[2])
        reward -= reward_config.comfort_weight * angular_speed
        
        # Progress reward (simplified - in real implementation, use waypoint progress)
        current_progress = self.episode_step / self.max_episode_steps
        progress_delta = current_progress - self.last_progress
        reward += reward_config.progress_reward_weight * progress_delta
        self.last_progress = current_progress
        
        return reward
        
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate."""
        # Collision or lane invasion
        if self._collision_flag or self._lane_invasion_flag:
            return True
            
        # Episode timeout
        if self.episode_step >= self.max_episode_steps:
            return True
            
        # Episode status from CARLA
        if self._episode_status in ["collision", "lane_invasion", "timeout"]:
            return True
            
        return False
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset episode state
        self.episode_step = 0
        self.episode_return = 0.0
        self.last_progress = 0.0
        
        # Reset flags
        self._reset_observation_data()
        
        # Wait for first observation
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5 second timeout
            obs = self._get_observation()
            if obs['image'].sum() > 0:  # Check if we received actual data
                break
            time.sleep(0.1)
        else:
            logger.warning("Timeout waiting for sensor data after reset")
            
        info = {
            'episode_step': self.episode_step,
            'episode_return': self.episode_return
        }
        
        return self._get_observation(), info
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute action and return next state."""
        # Convert action to control commands
        # action[0]: throttle/brake combined (-1 = full brake, +1 = full throttle)
        # action[1]: steering (-1 = full left, +1 = full right)
        
        throttle = max(0.0, action[0])
        brake = max(0.0, -action[0])
        steering = np.clip(action[1], -1.0, 1.0)
        
        # Send control command via ROS 2
        cmd_msg = Twist()
        cmd_msg.linear.x = throttle - brake  # Positive = throttle, negative = brake
        cmd_msg.angular.z = steering
        
        self.cmd_vel_pub.publish(cmd_msg)
        
        # Wait for next observation
        time.sleep(0.05)  # 20Hz control rate
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_return += reward
        
        # Check termination
        terminated = self._is_episode_done()
        truncated = False  # We handle truncation in terminated
        
        # Update episode step
        self.episode_step += 1
        
        # Info dictionary
        info = {
            'episode_step': self.episode_step,
            'episode_return': self.episode_return,
            'collision': self._collision_flag,
            'lane_invasion': self._lane_invasion_flag,
            'throttle': throttle,
            'steering': steering
        }
        
        return observation, reward, terminated, truncated, info
        
    def close(self) -> None:
        """Close environment and cleanup resources."""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        rclpy.shutdown()
        logger.info("CARLA ROS 2 Environment closed")


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for camera images.
    
    Based on the network architecture from the research papers,
    optimized for 84x84 RGB images.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # First conv layer: 32 filters, 8x8 kernel, stride 4
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Second conv layer: 64 filters, 4x4 kernel, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Third conv layer: 64 filters, 3x3 kernel, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Flatten
            nn.Flatten(),
        )
        
        # Calculate conv output size
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 84, 84)
            conv_output_size = self.cnn(sample_input).shape[1]
            
        # MLP layers for feature processing
        self.mlp = nn.Sequential(
            nn.Linear(conv_output_size + 6, 512),  # +6 for velocity and angular velocity
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self.features_dim = features_dim
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through CNN and MLP."""
        # Process image through CNN
        image_features = self.cnn(observations['image'])
        
        # Concatenate with velocity and angular velocity
        velocity_flat = observations['velocity'].view(observations['velocity'].shape[0], -1)
        angular_velocity_flat = observations['angular_velocity'].view(observations['angular_velocity'].shape[0], -1)
        
        combined_features = torch.cat([
            image_features, 
            velocity_flat, 
            angular_velocity_flat
        ], dim=1)
        
        # Process through MLP
        features = self.mlp(combined_features)
        
        return features


class CarlaTrainer:
    """
    Main training class for CARLA DRL agent.
    
    Handles PPO training, logging, checkpointing, and evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to training configuration YAML file
        """
        self.config = load_train_config(config_path)
        
        # Setup directories
        self.log_dir = Path(self.config.logging.tensorboard_log)
        self.checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Set seeds for reproducibility
        self._set_seeds()
        
        logger.info(f"Trainer initialized with config: {config_path}")
        
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        import random
        
        seed = self.config.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            if self.config.cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
        logger.info(f"Random seeds set to {seed}")
        
    def create_environment(self) -> CarlaROS2Environment:
        """Create CARLA ROS 2 environment."""
        env = CarlaROS2Environment(self.config)
        env = Monitor(env, self.log_dir / "monitor")
        return env
        
    def create_model(self, env: CarlaROS2Environment) -> PPO:
        """Create PPO model with custom CNN feature extractor."""
        
        # Custom policy kwargs
        policy_kwargs = {
            'features_extractor_class': CNNFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 512},
            'net_arch': [
                dict(pi=[256], vf=[256])  # Separate networks for policy and value
            ]
        }
        
        # Create PPO model
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=self.config.hyperparameters.learning_rate,
            n_steps=self.config.hyperparameters.n_steps,
            batch_size=self.config.hyperparameters.batch_size,
            n_epochs=self.config.hyperparameters.n_epochs,
            gamma=self.config.hyperparameters.gamma,
            gae_lambda=self.config.hyperparameters.gae_lambda,
            clip_range=self.config.hyperparameters.clip_range,
            clip_range_vf=self.config.hyperparameters.clip_range_vf,
            ent_coef=self.config.hyperparameters.ent_coef,
            vf_coef=self.config.hyperparameters.vf_coef,
            max_grad_norm=self.config.hyperparameters.max_grad_norm,
            target_kl=self.config.hyperparameters.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=self.config.logging.verbose,
            seed=self.config.random_seed,
            device='auto',
            tensorboard_log=str(self.log_dir)
        )
        
        logger.info("PPO model created with custom CNN feature extractor")
        return model
        
    def setup_callbacks(self, env: CarlaROS2Environment) -> List:
        """Setup training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.training.save_freq,
            save_path=str(self.checkpoint_dir),
            name_prefix='ppo_carla',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.checkpoint_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=self.config.training.eval_freq,
            n_eval_episodes=self.config.training.n_eval_episodes,
            deterministic=self.config.training.eval_deterministic,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        return callbacks
        
    def train(self) -> None:
        """Run training loop."""
        logger.info("Starting PPO training...")
        
        # Create environment and model
        env = self.create_environment()
        model = self.create_model(env)
        
        # Setup callbacks
        callbacks = self.setup_callbacks(env)
        
        try:
            # Start training
            model.learn(
                total_timesteps=self.config.training.total_timesteps,
                callback=callbacks,
                log_interval=self.config.logging.log_interval,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = self.checkpoint_dir / "final_model"
            model.save(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
            
        finally:
            # Cleanup
            env.close()
            self.writer.close()
            logger.info("Training completed and resources cleaned up")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DRL agent for CARLA')
    parser.add_argument(
        '--config', 
        type=str, 
        default='../configs/train.yaml',
        help='Path to training configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        trainer = CarlaTrainer(args.config)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

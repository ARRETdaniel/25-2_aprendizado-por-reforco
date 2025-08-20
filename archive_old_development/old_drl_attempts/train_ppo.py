"""
PPO Training Agent for CARLA Autonomous Driving

This module implements a complete PPO-based DRL training system with
real-time camera visualization, TensorBoard integration, and ROS 2 communication.

Author: GitHub Copilot  
Date: August 2025
Python Version: 3.12 (Modern DRL stack)
"""
import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import numpy as np
import cv2
import yaml

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, Empty
from cv_bridge import CvBridge

# DRL imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces

# TensorBoard and visualization
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Communication
import zmq
import msgpack

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarlaROS2Node(Node):
    """
    ROS 2 node for CARLA DRL training communication.
    
    Handles sensor data reception and control command publishing
    between the DRL agent and CARLA simulation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ROS 2 node.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__('carla_drl_agent')
        
        self.config = config
        self.bridge = CvBridge()
        
        # Data storage
        self.current_image = None
        self.current_vehicle_state = None
        self.current_reward = 0.0
        self.episode_done = False
        self.data_ready = False
        self.data_lock = threading.Lock()
        
        # Setup QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.control_publisher = self.create_publisher(
            Twist, '/carla/vehicle/control', qos_profile
        )
        self.reset_publisher = self.create_publisher(
            Empty, '/carla/episode/reset', qos_profile
        )
        
        # Subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/carla/camera/rgb', self.image_callback, qos_profile
        )
        self.reward_subscriber = self.create_subscription(
            Float32, '/carla/training/reward', self.reward_callback, qos_profile
        )
        self.done_subscriber = self.create_subscription(
            Bool, '/carla/training/done', self.done_callback, qos_profile
        )
        
        # ZeroMQ backup communication
        self.setup_zmq_backup()
        
        self.get_logger().info("CARLA ROS 2 node initialized")
        
    def setup_zmq_backup(self) -> None:
        """Setup ZeroMQ as backup communication method."""
        try:
            self.zmq_context = zmq.Context()
            
            # Subscriber for sensor data
            self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
            self.zmq_subscriber.connect("tcp://localhost:5555")
            self.zmq_subscriber.setsockopt_string(zmq.SUBSCRIBE, "sensor_data")
            self.zmq_subscriber.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            
            # Publisher for control commands
            self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
            self.zmq_publisher.bind("tcp://*:5556")
            
            self.get_logger().info("ZeroMQ backup communication setup")
            
        except Exception as e:
            self.get_logger().warning(f"ZeroMQ setup failed: {e}")
            self.zmq_context = None
            
    def image_callback(self, msg: Image) -> None:
        """Handle incoming camera images."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.data_lock:
                self.current_image = cv_image
                self.data_ready = True
                
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
            
    def reward_callback(self, msg: Float32) -> None:
        """Handle incoming reward values."""
        with self.data_lock:
            self.current_reward = msg.data
            
    def done_callback(self, msg: Bool) -> None:
        """Handle episode termination signals."""
        with self.data_lock:
            self.episode_done = msg.data
            
    def publish_control(self, throttle: float, steer: float, brake: float = 0.0) -> None:
        """
        Publish vehicle control commands.
        
        Args:
            throttle: Throttle value [0, 1]
            steer: Steering value [-1, 1]
            brake: Brake value [0, 1]
        """
        try:
            # ROS 2 message
            control_msg = Twist()
            control_msg.linear.x = float(throttle)
            control_msg.angular.z = float(steer)
            control_msg.linear.y = float(brake)  # Using y for brake
            
            self.control_publisher.publish(control_msg)
            
            # ZeroMQ backup
            if self.zmq_context:
                try:
                    control_data = {
                        'throttle': throttle,
                        'steer': steer,
                        'brake': brake,
                        'timestamp': time.time()
                    }
                    packed_data = msgpack.packb(control_data)
                    self.zmq_publisher.send_multipart([b"control", packed_data])
                except:
                    pass  # Silent fallback failure
                    
        except Exception as e:
            self.get_logger().error(f"Control publish error: {e}")
            
    def request_reset(self) -> None:
        """Request episode reset."""
        try:
            reset_msg = Empty()
            self.reset_publisher.publish(reset_msg)
            
            # Reset internal state
            with self.data_lock:
                self.episode_done = False
                self.current_reward = 0.0
                self.data_ready = False
                
        except Exception as e:
            self.get_logger().error(f"Reset request error: {e}")
            
    def get_observation(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, float, bool]]:
        """
        Get current observation with timeout.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            tuple: (image, reward, done) or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.data_lock:
                if self.data_ready and self.current_image is not None:
                    image = self.current_image.copy()
                    reward = self.current_reward
                    done = self.episode_done
                    return (image, reward, done)
                    
            # Check ZeroMQ backup if ROS 2 data not available
            if self.zmq_context:
                try:
                    topic, data = self.zmq_subscriber.recv_multipart(zmq.NOBLOCK)
                    message = msgpack.unpackb(data, raw=False)
                    
                    if 'camera_rgb' in message:
                        # Decode image
                        img_bytes = message['camera_rgb']
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        reward = message.get('reward', 0.0)
                        done = message.get('done', False)
                        
                        with self.data_lock:
                            self.current_image = image
                            self.current_reward = reward
                            self.episode_done = done
                            self.data_ready = True
                            
                        return (image, reward, done)
                        
                except zmq.Again:
                    pass  # No message available
                except Exception as e:
                    self.get_logger().warning(f"ZeroMQ receive error: {e}")
                    
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            
        return None


class CarlaGymEnvironment(gym.Env):
    """
    Gymnasium environment wrapper for CARLA via ROS 2.
    
    Provides a standard RL interface for training PPO agents
    with real-time camera visualization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CARLA Gym environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config
        
        # Initialize ROS 2 node
        try:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = CarlaROS2Node(config)
        except Exception as e:
            logger.error(f"Failed to initialize ROS 2: {e}")
            raise
            
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),    # [steer, throttle]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Image observation space
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),  # Resized RGB image
            dtype=np.uint8
        )
        
        # Environment state
        self.current_observation = None
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        
        # Visualization
        self.display_enabled = config.get('display_enabled', True)
        self.window_name = "CARLA DRL Training"
        
        # Performance tracking
        self.step_times = []
        self.reward_history = []
        
        logger.info("CARLA Gym environment initialized")
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess camera image for neural network input.
        
        Args:
            image: Raw camera image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if image is None:
            return np.zeros((84, 84, 3), dtype=np.uint8)
            
        # Resize to 84x84 (common for DRL)
        resized = cv2.resize(image, (84, 84))
        
        # Normalize to [0, 255] uint8 range
        if resized.dtype != np.uint8:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
            
        return resized
        
    def _display_training_info(self, image: np.ndarray, action: np.ndarray = None) -> np.ndarray:
        """
        Add training information overlay to image.
        
        Args:
            image: Input image
            action: Current action (optional)
            
        Returns:
            np.ndarray: Image with overlay
        """
        if not self.display_enabled or image is None:
            return image
            
        # Create display image (larger for visibility)
        display_img = cv2.resize(image, (400, 300))
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 2
        
        # Episode info
        cv2.putText(display_img, f"Episode Step: {self.episode_step_count}", 
                   (10, 25), font, font_scale, color, thickness)
        
        cv2.putText(display_img, f"Episode Reward: {self.episode_reward:.2f}", 
                   (10, 50), font, font_scale, color, thickness)
        
        # Action info
        if action is not None:
            steer_text = f"Steer: {action[0]:.3f}"
            throttle_text = f"Throttle: {action[1]:.3f}"
            cv2.putText(display_img, steer_text, (10, 75), font, font_scale, color, thickness)
            cv2.putText(display_img, throttle_text, (10, 100), font, font_scale, color, thickness)
            
        # Performance info
        if self.step_times:
            avg_step_time = np.mean(self.step_times[-100:])  # Last 100 steps
            fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 125), font, font_scale, color, thickness)
            
        return display_img
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed (optional)
            options: Reset options (optional)
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        logger.info("Resetting CARLA environment...")
        
        # Reset episode state
        self.episode_step_count = 0
        self.episode_reward = 0.0
        
        # Request reset from CARLA
        self.ros_node.request_reset()
        
        # Wait for new observation
        time.sleep(1.0)  # Allow simulation to reset
        
        obs_data = self.ros_node.get_observation(timeout=5.0)
        if obs_data is None:
            logger.warning("Failed to get initial observation, using zeros")
            image = np.zeros((600, 800, 3), dtype=np.uint8)
            reward = 0.0
            done = False
        else:
            image, reward, done = obs_data
            
        # Preprocess observation
        self.current_observation = self._preprocess_image(image)
        
        # Display
        if self.display_enabled:
            display_img = self._display_training_info(image)
            cv2.imshow(self.window_name, display_img)
            cv2.waitKey(1)
            
        info = {
            'episode_step': self.episode_step_count,
            'episode_reward': self.episode_reward,
            'reset_time': time.time()
        }
        
        return self.current_observation, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute [steer, throttle]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        step_start_time = time.time()
        
        # Parse action
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        
        # Send control command
        self.ros_node.publish_control(throttle, steer)
        
        # Get new observation
        obs_data = self.ros_node.get_observation(timeout=1.0)
        if obs_data is None:
            logger.warning("Failed to get observation, using previous")
            image = np.zeros((600, 800, 3), dtype=np.uint8)
            reward = -1.0  # Penalty for communication failure
            done = True
        else:
            image, reward, done = obs_data
            
        # Preprocess observation
        self.current_observation = self._preprocess_image(image)
        
        # Update episode state
        self.episode_step_count += 1
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = done
        truncated = self.episode_step_count >= self.max_episode_steps
        
        # Display
        if self.display_enabled:
            display_img = self._display_training_info(image, action)
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                terminated = True
                
        # Performance tracking
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        if len(self.step_times) > 1000:  # Keep last 1000 step times
            self.step_times.pop(0)
            
        if terminated or truncated:
            self.reward_history.append(self.episode_reward)
            
        info = {
            'episode_step': self.episode_step_count,
            'episode_reward': self.episode_reward,
            'step_time': step_time,
            'steer': steer,
            'throttle': throttle,
            'raw_reward': reward
        }
        
        return self.current_observation, reward, terminated, truncated, info
        
    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("Closing CARLA environment...")
        
        if self.display_enabled:
            cv2.destroyAllWindows()
            
        if hasattr(self, 'ros_node') and self.ros_node:
            self.ros_node.destroy_node()
            
        if rclpy.ok():
            rclpy.shutdown()


class TensorBoardVisualizationCallback(BaseCallback):
    """
    Custom callback for TensorBoard visualization during training.
    
    Logs training metrics, episode statistics, and creates
    real-time plots for monitoring training progress.
    """
    
    def __init__(self, log_dir: str, verbose: int = 1):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Performance tracking
        self.training_start_time = time.time()
        self.last_log_time = time.time()
        
        logger.info(f"TensorBoard logging to: {self.log_dir}")
        
    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Get current step info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Log step-level metrics
            self.writer.add_scalar('Environment/StepReward', 
                                 self.locals.get('rewards', [0])[0], 
                                 self.num_timesteps)
            
            # Log action distribution
            if 'actions' in self.locals:
                actions = self.locals['actions']
                if len(actions.shape) > 1 and actions.shape[1] >= 2:
                    self.writer.add_scalar('Actions/Steer_Mean', 
                                         np.mean(actions[:, 0]), 
                                         self.num_timesteps)
                    self.writer.add_scalar('Actions/Throttle_Mean', 
                                         np.mean(actions[:, 1]), 
                                         self.num_timesteps)
                                         
            # Episode tracking
            if 'episode_reward' in info:
                self.current_episode_reward = info['episode_reward']
                self.current_episode_length = info.get('episode_step', 0)
                
        # Log every 100 steps
        if self.num_timesteps % 100 == 0:
            current_time = time.time()
            
            # Training speed
            steps_per_sec = 100 / (current_time - self.last_log_time)
            self.writer.add_scalar('Performance/StepsPerSecond', 
                                 steps_per_sec, self.num_timesteps)
            
            # Memory usage (if available)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.writer.add_scalar('Performance/MemoryUsage_MB', 
                                     memory_mb, self.num_timesteps)
            except ImportError:
                pass
                
            self.last_log_time = current_time
            
        return True
        
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Log rollout statistics if available
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])
                self.writer.add_scalar('Episode/Reward', ep_info['r'], self.num_timesteps)
                
            if 'l' in ep_info:
                self.episode_lengths.append(ep_info['l'])
                self.writer.add_scalar('Episode/Length', ep_info['l'], self.num_timesteps)
                
            # Moving averages
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.writer.add_scalar('Episode/Reward_Avg10', avg_reward, self.num_timesteps)
                
            if len(self.episode_rewards) >= 100:
                avg_reward = np.mean(self.episode_rewards[-100:])
                self.writer.add_scalar('Episode/Reward_Avg100', avg_reward, self.num_timesteps)
                
    def _on_training_end(self) -> None:
        """Called when training ends."""
        training_time = time.time() - self.training_start_time
        self.writer.add_scalar('Training/TotalTime_Minutes', training_time / 60.0, self.num_timesteps)
        
        if self.writer:
            self.writer.close()
            
        logger.info(f"Training completed in {training_time/60:.1f} minutes")


class PPOTrainer:
    """
    Complete PPO training system for CARLA autonomous driving.
    
    Handles environment creation, model initialization, training loop,
    and real-time visualization with comprehensive monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Paths
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.total_timesteps = config.get('total_timesteps', 100000)
        self.save_interval = config.get('save_interval', 10000)
        self.eval_interval = config.get('eval_interval', 5000)
        
        # Model and environment
        self.env = None
        self.model = None
        
        # Monitoring
        self.training_start_time = None
        
        logger.info("PPO trainer initialized")
        
    def create_environment(self) -> CarlaGymEnvironment:
        """
        Create CARLA Gym environment.
        
        Returns:
            CarlaGymEnvironment: Created environment
        """
        env_config = self.config.get('environment', {})
        env_config.update({
            'display_enabled': self.config.get('display_enabled', True),
            'max_episode_steps': self.config.get('max_episode_steps', 1000)
        })
        
        env = CarlaGymEnvironment(env_config)
        
        # Wrap with Monitor for automatic logging
        env = Monitor(env, str(self.log_dir / "monitor"), allow_early_resets=True)
        
        return env
        
    def create_model(self, env: CarlaGymEnvironment) -> PPO:
        """
        Create PPO model.
        
        Args:
            env: Training environment
            
        Returns:
            PPO: Created model
        """
        # PPO hyperparameters
        model_config = self.config.get('ppo', {})
        
        # Policy architecture
        policy_kwargs = {
            'net_arch': model_config.get('net_arch', [dict(pi=[256, 256], vf=[256, 256])]),
            'activation_fn': torch.nn.ReLU,
            'features_extractor_class': None,  # Use default CNN for images
        }
        
        # Create PPO model
        model = PPO(
            policy="CnnPolicy",  # CNN policy for image observations
            env=env,
            learning_rate=model_config.get('learning_rate', 3e-4),
            n_steps=model_config.get('n_steps', 2048),
            batch_size=model_config.get('batch_size', 64),
            n_epochs=model_config.get('n_epochs', 10),
            gamma=model_config.get('gamma', 0.99),
            gae_lambda=model_config.get('gae_lambda', 0.95),
            clip_range=model_config.get('clip_range', 0.2),
            ent_coef=model_config.get('ent_coef', 0.01),
            vf_coef=model_config.get('vf_coef', 0.5),
            max_grad_norm=model_config.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(self.log_dir),
            verbose=1,
            seed=self.config.get('seed', 42)
        )
        
        logger.info(f"PPO model created with {model.policy.parameters().__len__()} parameters")
        return model
        
    def setup_callbacks(self) -> List[BaseCallback]:
        """
        Setup training callbacks.
        
        Returns:
            List[BaseCallback]: List of callbacks
        """
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_interval,
            save_path=str(self.checkpoint_dir),
            name_prefix="ppo_carla"
        )
        callbacks.append(checkpoint_callback)
        
        # TensorBoard visualization callback
        tb_callback = TensorBoardVisualizationCallback(
            log_dir=str(self.log_dir / "tensorboard"),
            verbose=1
        )
        callbacks.append(tb_callback)
        
        return callbacks
        
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
                
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.model = PPO.load(str(checkpoint_path), env=self.env)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
            
    def train(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Execute training loop.
        
        Args:
            checkpoint_path: Optional checkpoint to resume from
        """
        logger.info("Starting PPO training...")
        self.training_start_time = time.time()
        
        try:
            # Create environment
            logger.info("Creating environment...")
            self.env = self.create_environment()
            
            # Create or load model
            if checkpoint_path and self.load_checkpoint(checkpoint_path):
                logger.info("Resumed from checkpoint")
            else:
                logger.info("Creating new model...")
                self.model = self.create_model(self.env)
                
            # Setup callbacks
            callbacks = self.setup_callbacks()
            
            # Start training
            logger.info(f"Training for {self.total_timesteps} timesteps...")
            
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                tb_log_name="ppo_carla_training",
                reset_num_timesteps=checkpoint_path is None
            )
            
            # Save final model
            final_path = self.checkpoint_dir / "ppo_carla_final.zip"
            self.model.save(str(final_path))
            logger.info(f"Final model saved to: {final_path}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.model:
                interrupt_path = self.checkpoint_dir / "ppo_carla_interrupted.zip"
                self.model.save(str(interrupt_path))
                logger.info(f"Model saved to: {interrupt_path}")
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
        finally:
            if self.env:
                self.env.close()
                
            training_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {training_time/3600:.2f} hours")
            
    def evaluate(self, checkpoint_path: str, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            num_episodes: Number of evaluation episodes
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating model: {checkpoint_path}")
        
        try:
            # Create evaluation environment
            eval_config = self.config.get('environment', {})
            eval_config['display_enabled'] = True  # Always show during evaluation
            eval_env = CarlaGymEnvironment(eval_config)
            
            # Load model
            model = PPO.load(checkpoint_path, env=eval_env)
            
            # Run evaluation episodes
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(num_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}")
                
            eval_env.close()
            
            # Calculate statistics
            results = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'std_length': np.std(episode_lengths),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards)
            }
            
            logger.info(f"Evaluation results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def main():
    """Main entry point for PPO training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CARLA PPO Training')
    parser.add_argument('--config', type=str, 
                       default='configs/complete_system_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation mode')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes for evaluation')
    parser.add_argument('--timesteps', type=int, help='Override total timesteps')
    parser.add_argument('--display', action='store_true', help='Enable display')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.timesteps:
        config['total_timesteps'] = args.timesteps
    if args.display:
        config['display_enabled'] = True
    if args.no_display:
        config['display_enabled'] = False
        
    # Create trainer
    trainer = PPOTrainer(config)
    
    try:
        if args.evaluate:
            if not args.checkpoint:
                logger.error("Checkpoint required for evaluation")
                sys.exit(1)
            trainer.evaluate(args.checkpoint, args.episodes)
        else:
            trainer.train(args.checkpoint)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

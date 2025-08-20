"""
CARLA-ROS2 Environment Wrapper for DRL Training

This module provides a Gym-compatible environment wrapper that interfaces
with the CARLA-ROS2 bridge for reinforcement learning training. It handles
observation preprocessing, action execution, and reward computation.

The environment receives sensor data from the ROS2 bridge and sends control
commands back through the same interface.
"""

import gym
import numpy as np
import cv2
import logging
import time
import zmq
import msgpack
import threading
from typing import Dict, Tuple, Any, Optional, List, Union
from collections import deque
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROS 2 imports (conditional)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float32MultiArray, Bool
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
    logger.info("ROS 2 available for environment")
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ROS 2 not available - using ZeroMQ communication only")


class ObservationProcessor:
    """Processes raw observations into ML-ready format."""
    
    def __init__(self, config: Dict):
        """Initialize observation processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.image_size = tuple(config.get('image_size', [84, 84]))
        self.normalize_images = config.get('normalize_images', True)
        self.stack_frames = config.get('stack_frames', 4)
        
        # Frame stacking
        self.frame_stack = deque(maxlen=self.stack_frames)
        
        # Statistics for normalization
        self.vector_stats = {
            'mean': np.zeros(config.get('vector_dim', 10)),
            'std': np.ones(config.get('vector_dim', 10)),
            'count': 0
        }
        
        logger.info(f"ObservationProcessor initialized - "
                   f"Image size: {self.image_size}, Stack frames: {self.stack_frames}")
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process raw image observation.
        
        Args:
            image: Raw image array (H, W, C) or (H, W)
            
        Returns:
            Processed image array
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        if self.normalize_images:
            image = image.astype(np.float32) / 255.0
        
        # Add to frame stack
        self.frame_stack.append(image)
        
        # Pad if not enough frames
        while len(self.frame_stack) < self.stack_frames:
            self.frame_stack.append(image)
        
        # Stack frames (C, H, W)
        stacked = np.stack(list(self.frame_stack), axis=0)
        
        return stacked
    
    def process_vector(self, vector: np.ndarray) -> np.ndarray:
        """Process vector observation.
        
        Args:
            vector: Raw vector observation
            
        Returns:
            Processed vector observation
        """
        vector = np.array(vector, dtype=np.float32)
        
        # Update statistics (running average)
        if self.config.get('normalize_vectors', True):
            batch_mean = np.mean(vector)
            batch_var = np.var(vector)
            
            # Update running statistics
            self.vector_stats['count'] += 1
            alpha = 1.0 / self.vector_stats['count']
            
            self.vector_stats['mean'] = (
                (1 - alpha) * self.vector_stats['mean'] + alpha * batch_mean
            )
            self.vector_stats['std'] = np.sqrt(
                (1 - alpha) * self.vector_stats['std']**2 + alpha * batch_var
            )
            
            # Normalize
            vector = (vector - self.vector_stats['mean']) / (self.vector_stats['std'] + 1e-8)
        
        return vector
    
    def reset(self):
        """Reset processor state."""
        self.frame_stack.clear()


class RewardCalculator:
    """Calculates rewards for autonomous driving task."""
    
    def __init__(self, config: Dict):
        """Initialize reward calculator.
        
        Args:
            config: Reward configuration
        """
        self.config = config
        
        # Reward weights
        self.weights = {
            'speed': config.get('speed_weight', 1.0),
            'lane_keeping': config.get('lane_weight', 2.0),
            'collision': config.get('collision_weight', -10.0),
            'smoothness': config.get('smoothness_weight', 0.5),
            'goal_progress': config.get('progress_weight', 5.0),
            'efficiency': config.get('efficiency_weight', 0.1)
        }
        
        # Previous action for smoothness
        self.prev_action = None
        
        # Target speed
        self.target_speed = config.get('target_speed', 30.0)  # km/h
        
        logger.info(f"RewardCalculator initialized with weights: {self.weights}")
    
    def calculate_reward(self,
                        obs: Dict[str, Any],
                        action: np.ndarray,
                        info: Dict[str, Any]) -> float:
        """Calculate reward for current step.
        
        Args:
            obs: Current observations
            action: Action taken
            info: Additional information
            
        Returns:
            Reward value
        """
        reward = 0.0
        reward_components = {}
        
        # Speed reward
        current_speed = info.get('speed', 0.0)
        speed_reward = self._speed_reward(current_speed)
        reward += self.weights['speed'] * speed_reward
        reward_components['speed'] = speed_reward
        
        # Lane keeping reward
        lane_deviation = info.get('lane_deviation', 0.0)
        lane_reward = self._lane_keeping_reward(lane_deviation)
        reward += self.weights['lane_keeping'] * lane_reward
        reward_components['lane_keeping'] = lane_reward
        
        # Collision penalty
        if info.get('collision', False):
            collision_reward = self.weights['collision']
            reward += collision_reward
            reward_components['collision'] = collision_reward
        
        # Smoothness reward
        if self.prev_action is not None:
            smoothness_reward = self._smoothness_reward(action, self.prev_action)
            reward += self.weights['smoothness'] * smoothness_reward
            reward_components['smoothness'] = smoothness_reward
        
        # Goal progress reward
        progress = info.get('goal_progress', 0.0)
        progress_reward = self.weights['goal_progress'] * progress
        reward += progress_reward
        reward_components['goal_progress'] = progress_reward
        
        # Efficiency reward (fuel/energy)
        efficiency = info.get('efficiency', 0.0)
        efficiency_reward = self.weights['efficiency'] * efficiency
        reward += efficiency_reward
        reward_components['efficiency'] = efficiency_reward
        
        # Store previous action
        self.prev_action = action.copy()
        
        # Store reward components in info
        info['reward_components'] = reward_components
        
        return reward
    
    def _speed_reward(self, current_speed: float) -> float:
        """Calculate speed-based reward.
        
        Args:
            current_speed: Current vehicle speed (km/h)
            
        Returns:
            Speed reward
        """
        # Reward for maintaining target speed
        speed_diff = abs(current_speed - self.target_speed)
        speed_reward = max(0.0, 1.0 - speed_diff / self.target_speed)
        
        return speed_reward
    
    def _lane_keeping_reward(self, lane_deviation: float) -> float:
        """Calculate lane keeping reward.
        
        Args:
            lane_deviation: Distance from lane center (meters)
            
        Returns:
            Lane keeping reward
        """
        # Exponential decay for lane deviation
        max_deviation = 2.0  # meters
        lane_reward = np.exp(-lane_deviation / max_deviation)
        
        return lane_reward
    
    def _smoothness_reward(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """Calculate action smoothness reward.
        
        Args:
            action: Current action
            prev_action: Previous action
            
        Returns:
            Smoothness reward
        """
        # Penalize large action changes
        action_diff = np.linalg.norm(action - prev_action)
        smoothness_reward = np.exp(-action_diff)
        
        return smoothness_reward
    
    def reset(self):
        """Reset reward calculator state."""
        self.prev_action = None


class CarlaROS2Environment(gym.Env):
    """Gym environment for CARLA autonomous driving with ROS2 integration."""
    
    def __init__(self, config_path: str):
        """Initialize CARLA ROS2 environment.
        
        Args:
            config_path: Path to environment configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Environment setup
        self.episode_length = self.config.get('max_episode_steps', 1000)
        self.current_step = 0
        
        # Initialize processors
        self.obs_processor = ObservationProcessor(self.config['observation'])
        self.reward_calculator = RewardCalculator(self.config['reward'])
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Communication setup
        self._setup_communication()
        
        # State tracking
        self.current_obs = None
        self.episode_stats = {
            'total_reward': 0.0,
            'steps': 0,
            'collisions': 0,
            'avg_speed': 0.0
        }
        
        # Initialize ROS2 if available
        if ROS2_AVAILABLE and self.config.get('use_ros2', True):
            self._setup_ros2()
        
        logger.info("CarlaROS2Environment initialized successfully")
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [throttle, steer, brake] all in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Observation space
        image_shape = (
            self.obs_processor.stack_frames,
            *self.obs_processor.image_size
        )
        vector_dim = self.config['observation']['vector_dim']
        
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0.0 if self.obs_processor.normalize_images else 0,
                high=1.0 if self.obs_processor.normalize_images else 255,
                shape=image_shape,
                dtype=np.float32
            ),
            'vector': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(vector_dim,),
                dtype=np.float32
            )
        })
        
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
    
    def _setup_communication(self):
        """Setup communication with CARLA bridge."""
        # ZeroMQ setup
        self.context = zmq.Context()
        
        # Socket for receiving observations
        self.obs_socket = self.context.socket(zmq.SUB)
        obs_port = self.config['communication']['obs_port']
        self.obs_socket.connect(f"tcp://localhost:{obs_port}")
        self.obs_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.obs_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        
        # Socket for sending actions
        self.action_socket = self.context.socket(zmq.PUB)
        action_port = self.config['communication']['action_port']
        self.action_socket.bind(f"tcp://*:{action_port}")
        
        # Socket for control messages
        self.control_socket = self.context.socket(zmq.REQ)
        control_port = self.config['communication']['control_port']
        self.control_socket.connect(f"tcp://localhost:{control_port}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        
        logger.info("ZeroMQ communication setup complete")
    
    def _setup_ros2(self):
        """Setup ROS2 communication."""
        if not ROS2_AVAILABLE:
            logger.warning("ROS2 not available for setup")
            return
        
        # Initialize ROS2
        rclpy.init()
        
        # Create ROS2 node
        self.ros_node = CarlaEnvironmentNode(
            name='carla_env_node',
            config=self.config['ros2']
        )
        
        # Start ROS2 spinner thread
        self.ros_thread = threading.Thread(
            target=self._ros2_spin,
            daemon=True
        )
        self.ros_thread.start()
        
        logger.info("ROS2 communication setup complete")
    
    def _ros2_spin(self):
        """Spin ROS2 node in separate thread."""
        try:
            rclpy.spin(self.ros_node)
        except Exception as e:
            logger.error(f"ROS2 spinning error: {e}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode.
        
        Returns:
            Initial observation
        """
        # Reset processors
        self.obs_processor.reset()
        self.reward_calculator.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'steps': 0,
            'collisions': 0,
            'avg_speed': 0.0
        }
        
        # Send reset command to CARLA
        try:
            reset_msg = {'command': 'reset'}
            self.control_socket.send(msgpack.packb(reset_msg))
            response = self.control_socket.recv()
            response = msgpack.unpackb(response)
            
            if not response.get('success', False):
                logger.warning("CARLA reset failed")
        except zmq.Again:
            logger.warning("Timeout waiting for reset response")
        
        # Get initial observation
        obs = self._get_observation()
        self.current_obs = obs
        
        logger.debug("Environment reset complete")
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute action in environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Send action to CARLA
        self._send_action(action)
        
        # Wait for next observation
        obs = self._get_observation()
        
        # Get additional info
        info = self._get_info()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(obs, action, info)
        
        # Check termination conditions
        done = self._check_done(info)
        
        # Update episode stats
        self.episode_stats['total_reward'] += reward
        self.episode_stats['steps'] = self.current_step
        self.episode_stats['avg_speed'] = (
            (self.episode_stats['avg_speed'] * self.current_step + info.get('speed', 0)) /
            (self.current_step + 1)
        )
        if info.get('collision', False):
            self.episode_stats['collisions'] += 1
        
        # Update step counter
        self.current_step += 1
        
        # Add episode stats to info
        info['episode_stats'] = self.episode_stats.copy()
        
        self.current_obs = obs
        
        return obs, reward, done, info
    
    def _send_action(self, action: np.ndarray):
        """Send action to CARLA bridge.
        
        Args:
            action: Action array [throttle, steer, brake]
        """
        # Convert action to control message
        action_msg = {
            'throttle': float(max(0.0, action[0])),  # Only positive throttle
            'steer': float(action[1]),
            'brake': float(max(0.0, -action[0])),  # Negative throttle becomes brake
            'timestamp': time.time()
        }
        
        # Send via ZeroMQ
        try:
            self.action_socket.send(msgpack.packb(action_msg))
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
        
        # Send via ROS2 if available
        if hasattr(self, 'ros_node'):
            self.ros_node.publish_control(action_msg)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation from CARLA bridge.
        
        Returns:
            Processed observation dictionary
        """
        try:
            # Receive from ZeroMQ
            raw_obs = self.obs_socket.recv()
            obs_data = msgpack.unpackb(raw_obs)
            
            # Process image
            image = np.array(obs_data['image'], dtype=np.uint8)
            processed_image = self.obs_processor.process_image(image)
            
            # Process vector data
            vector = np.array(obs_data['vector'], dtype=np.float32)
            processed_vector = self.obs_processor.process_vector(vector)
            
            observation = {
                'image': processed_image,
                'vector': processed_vector
            }
            
            return observation
            
        except zmq.Again:
            logger.warning("Timeout receiving observation")
            # Return previous observation if available
            if self.current_obs is not None:
                return self.current_obs
            else:
                # Return dummy observation
                return {
                    'image': np.zeros(self.observation_space['image'].shape, dtype=np.float32),
                    'vector': np.zeros(self.observation_space['vector'].shape, dtype=np.float32)
                }
        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            return self.current_obs or {
                'image': np.zeros(self.observation_space['image'].shape, dtype=np.float32),
                'vector': np.zeros(self.observation_space['vector'].shape, dtype=np.float32)
            }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information from CARLA.
        
        Returns:
            Information dictionary
        """
        try:
            # Request info from CARLA
            info_msg = {'command': 'get_info'}
            self.control_socket.send(msgpack.packb(info_msg))
            response = self.control_socket.recv()
            info = msgpack.unpackb(response)
            
            return info
            
        except zmq.Again:
            logger.warning("Timeout getting info")
            return {}
        except Exception as e:
            logger.error(f"Failed to get info: {e}")
            return {}
    
    def _check_done(self, info: Dict[str, Any]) -> bool:
        """Check if episode should terminate.
        
        Args:
            info: Information dictionary
            
        Returns:
            Whether episode is done
        """
        # Maximum steps reached
        if self.current_step >= self.episode_length:
            return True
        
        # Collision occurred
        if info.get('collision', False):
            return True
        
        # Goal reached
        if info.get('goal_reached', False):
            return True
        
        # Vehicle stuck (very low speed for too long)
        if info.get('stuck', False):
            return True
        
        return False
    
    def close(self):
        """Clean up environment resources."""
        # Close ZeroMQ sockets
        if hasattr(self, 'obs_socket'):
            self.obs_socket.close()
        if hasattr(self, 'action_socket'):
            self.action_socket.close()
        if hasattr(self, 'control_socket'):
            self.control_socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        
        # Shutdown ROS2
        if hasattr(self, 'ros_node'):
            self.ros_node.destroy_node()
            rclpy.shutdown()
        
        logger.info("Environment closed successfully")


class CarlaEnvironmentNode(Node):
    """ROS2 node for CARLA environment communication."""
    
    def __init__(self, name: str, config: Dict):
        """Initialize ROS2 node.
        
        Args:
            name: Node name
            config: ROS2 configuration
        """
        super().__init__(name)
        
        self.config = config
        self.cv_bridge = CvBridge()
        
        # Publishers
        self.control_pub = self.create_publisher(
            Twist,
            config['control_topic'],
            10
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            config['image_topic'],
            self.image_callback,
            10
        )
        
        self.state_sub = self.create_subscription(
            Float32MultiArray,
            config['state_topic'],
            self.state_callback,
            10
        )
        
        # Latest data
        self.latest_image = None
        self.latest_state = None
        
        logger.info(f"ROS2 node {name} initialized")
    
    def image_callback(self, msg: Image):
        """Handle incoming image messages.
        
        Args:
            msg: Image message
        """
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            logger.error(f"Failed to convert image: {e}")
    
    def state_callback(self, msg: Float32MultiArray):
        """Handle incoming state messages.
        
        Args:
            msg: State message
        """
        self.latest_state = np.array(msg.data)
    
    def publish_control(self, action_msg: Dict[str, float]):
        """Publish control command.
        
        Args:
            action_msg: Action message dictionary
        """
        twist = Twist()
        twist.linear.x = action_msg['throttle'] - action_msg['brake']
        twist.angular.z = action_msg['steer']
        
        self.control_pub.publish(twist)
    
    def get_latest_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get latest sensor data.
        
        Returns:
            Tuple of (image, state)
        """
        return self.latest_image, self.latest_state


def create_carla_environment(config_path: str) -> CarlaROS2Environment:
    """Create CARLA ROS2 environment from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Environment instance
    """
    env = CarlaROS2Environment(config_path)
    return env

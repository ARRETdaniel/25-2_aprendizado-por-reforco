"""
ROS 2 Communication Layer for CARLA-DRL Bridge.

This module provides a ROS 2 communication interface between the CARLA simulator
(running in Python 3.6) and the DRL agent (running in Python 3.12).
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ROS 2 packages
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray, String
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge
    HAS_ROS2 = True
    logger.info("ROS 2 modules imported successfully")
except ImportError:
    logger.warning("ROS 2 not found, falling back to file-based communication")
    HAS_ROS2 = False


class ROSBridge:
    """Base class for ROS 2 communication."""

    def __init__(self, node_name: str, use_ros: bool = True):
        """Initialize the ROS bridge.

        Args:
            node_name: Name of the ROS 2 node
            use_ros: Whether to use ROS 2 or fall back to file-based communication
        """
        self.use_ros = use_ros and HAS_ROS2
        self.node_name = node_name
        self.node = None
        self.bridge = None

        if self.use_ros:
            try:
                # Initialize ROS 2
                rclpy.init()
                self.node = Node(node_name)
                self.bridge = CvBridge()
                logger.info(f"ROS 2 node '{node_name}' initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ROS 2: {e}")
                self.use_ros = False

        # If ROS 2 initialization failed or not requested, set up file-based communication
        if not self.use_ros:
            self.comm_dir = Path.home() / ".carla_drl_bridge"
            self.comm_dir.mkdir(exist_ok=True)
            logger.info(f"Using file-based communication in {self.comm_dir}")

    def shutdown(self):
        """Clean shutdown of ROS resources."""
        if self.use_ros and self.node is not None:
            self.node.destroy_node()
            rclpy.shutdown()
            logger.info(f"ROS 2 node '{self.node_name}' shut down")


class CARLABridge(ROSBridge):
    """ROS 2 bridge for CARLA simulator (Python 3.6 side)."""

    def __init__(self, use_ros: bool = True):
        """Initialize the CARLA bridge.

        Args:
            use_ros: Whether to use ROS 2 or fall back to file-based communication
        """
        super().__init__("carla_bridge", use_ros)

        # Publishers for sensor data
        if self.use_ros:
            # Create QoS profile for sensor data (optimized for real-time)
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            # Create QoS profile for control commands (reliable)
            control_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )

            # Publishers
            self.camera_rgb_pub = self.node.create_publisher(
                Image, 'carla/camera/rgb', sensor_qos)
            self.camera_depth_pub = self.node.create_publisher(
                Image, 'carla/camera/depth', sensor_qos)
            self.camera_semantic_pub = self.node.create_publisher(
                Image, 'carla/camera/semantic', sensor_qos)
            self.state_pub = self.node.create_publisher(
                Float32MultiArray, 'carla/state', sensor_qos)
            self.reward_pub = self.node.create_publisher(
                Float32MultiArray, 'carla/reward', sensor_qos)
            self.info_pub = self.node.create_publisher(
                String, 'carla/info', control_qos)

            # Subscribers
            self.action_sub = self.node.create_subscription(
                Float32MultiArray, 'drl/action', self.action_callback, control_qos)
            self.control_sub = self.node.create_subscription(
                String, 'drl/control', self.control_callback, control_qos)

            # Background spinner
            self.spin_thread = None
            self._start_spin_thread()

    def _start_spin_thread(self):
        """Start a background thread to spin the ROS 2 node."""
        if self.use_ros:
            import threading
            self.spin_thread = threading.Thread(target=self._spin_node)
            self.spin_thread.daemon = True
            self.spin_thread.start()

    def _spin_node(self):
        """Spin the ROS 2 node in a background thread."""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.01)  # Limit CPU usage

    def publish_camera(self, image_array: np.ndarray, camera_type: str = "rgb", timestamp: float = None):
        """Publish camera image.

        Args:
            image_array: Image as numpy array (H, W, 3) for RGB or (H, W, 1) for depth/semantic
            camera_type: Type of camera ('rgb', 'depth', 'semantic')
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        encoding = 'rgb8'
        if camera_type == 'depth':
            # Convert depth to normalized float32 if needed
            if image_array.dtype != np.float32:
                image_array = image_array.astype(np.float32)
            encoding = '32FC1'
            if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                image_array = image_array[:, :, 0]  # Take first channel
        elif camera_type == 'semantic':
            encoding = 'mono8'
            if len(image_array.shape) == 3 and image_array.shape[2] > 1:
                image_array = image_array[:, :, 0]  # Take first channel

        if self.use_ros:
            try:
                # Convert numpy array to ROS Image message
                img_msg = self.bridge.cv2_to_imgmsg(image_array, encoding=encoding)
                img_msg.header.stamp = self.node.get_clock().now().to_msg()

                # Publish to the appropriate topic
                if camera_type == 'rgb':
                    self.camera_rgb_pub.publish(img_msg)
                elif camera_type == 'depth':
                    self.camera_depth_pub.publish(img_msg)
                elif camera_type == 'semantic':
                    self.camera_semantic_pub.publish(img_msg)
                else:
                    logger.warning(f"Unknown camera type: {camera_type}")
            except Exception as e:
                logger.error(f"Failed to publish camera image ({camera_type}): {e}")
        else:
            # Save image to file (for debugging, not efficient for real-time)
            import cv2
            file_path = str(self.comm_dir / f"camera_{camera_type}_{timestamp:.6f}.jpg")

            if camera_type == 'rgb':
                cv2.imwrite(file_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            elif camera_type == 'depth':
                # Normalize depth for visualization
                norm_depth = (image_array / np.max(image_array) * 255).astype(np.uint8) if np.max(image_array) > 0 else np.zeros_like(image_array, dtype=np.uint8)
                cv2.imwrite(file_path, norm_depth)
            elif camera_type == 'semantic':
                cv2.imwrite(file_path, image_array)

            # Write metadata
            metadata = {
                "timestamp": timestamp,
                "shape": image_array.shape,
                "type": camera_type
            }
            with open(self.comm_dir / f"camera_{camera_type}_latest.json", 'w') as f:
                json.dump(metadata, f)

    def publish_state(self, state: np.ndarray, timestamp: float = None):
        """Publish state observation.

        Args:
            state: State observation as numpy array
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        if self.use_ros:
            try:
                # Convert numpy array to Float32MultiArray message
                msg = Float32MultiArray()
                msg.data = state.flatten().tolist()
                self.state_pub.publish(msg)
            except Exception as e:
                logger.error(f"Failed to publish state: {e}")
        else:
            # Save state to file
            np.save(self.comm_dir / f"state_{timestamp:.6f}.npy", state)
            with open(self.comm_dir / "state_latest.txt", 'w') as f:
                f.write(f"{timestamp:.6f}")

    def publish_structured_state(self, state_dict: Dict[str, np.ndarray], timestamp: float = None):
        """Publish structured state observation with named components.

        Args:
            state_dict: Dictionary of named state components (e.g., 'position', 'velocity', etc.)
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        # Convert all state components to flattened arrays and include their shapes
        state_info = {
            "components": {},
            "timestamp": timestamp
        }

        # Flatten and concatenate all state components
        all_values = []
        for name, value in state_dict.items():
            start_idx = len(all_values)
            all_values.extend(value.flatten().tolist())
            end_idx = len(all_values)

            state_info["components"][name] = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "shape": value.shape
            }

        # Create the combined state array
        combined_state = np.array(all_values, dtype=np.float32)

        # Publish using the standard state method
        self.publish_state(combined_state, timestamp)

        # Also publish the state info via the info channel
        self.publish_info({"state_structure": state_info})

    def publish_reward(self, reward: float, done: bool, timestamp: float = None):
        """Publish reward and episode status.

        Args:
            reward: Reward value
            done: Whether the episode is done
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()

        if self.use_ros:
            try:
                # Package reward and done flag into a Float32MultiArray
                msg = Float32MultiArray()
                msg.data = [reward, 1.0 if done else 0.0]
                self.reward_pub.publish(msg)
            except Exception as e:
                logger.error(f"Failed to publish reward: {e}")
        else:
            # Save reward to file
            with open(self.comm_dir / f"reward_{timestamp:.6f}.json", 'w') as f:
                json.dump({"reward": reward, "done": done, "timestamp": timestamp}, f)
            with open(self.comm_dir / "reward_latest.txt", 'w') as f:
                f.write(f"{timestamp:.6f}")

    def publish_info(self, info: Dict[str, Any]):
        """Publish additional information.

        Args:
            info: Dictionary of additional information
        """
        if self.use_ros:
            try:
                msg = String()
                msg.data = json.dumps(info)
                self.info_pub.publish(msg)
            except Exception as e:
                logger.error(f"Failed to publish info: {e}")
        else:
            # Save info to file
            timestamp = time.time()
            with open(self.comm_dir / f"info_{timestamp:.6f}.json", 'w') as f:
                json.dump(info, f)
            with open(self.comm_dir / "info_latest.txt", 'w') as f:
                f.write(f"{timestamp:.6f}")

    def action_callback(self, msg):
        """Callback for receiving actions from the DRL agent."""
        # This will be overridden by the user
        pass

    def control_callback(self, msg):
        """Callback for receiving control commands from the DRL agent."""
        # This will be overridden by the user
        pass

    def get_latest_action(self) -> Optional[np.ndarray]:
        """Get the latest action when using file-based communication.

        Returns:
            Latest action as numpy array, or None if no action is available
        """
        if self.use_ros:
            return None  # Actions come through callbacks in ROS mode

        try:
            latest_file = self.comm_dir / "action_latest.txt"
            if not latest_file.exists():
                return None

            with open(latest_file, 'r') as f:
                timestamp = float(f.read().strip())

            action_file = self.comm_dir / f"action_{timestamp:.6f}.npy"
            if action_file.exists():
                return np.load(action_file)

        except Exception as e:
            logger.error(f"Failed to get latest action: {e}")

        return None


class DRLBridge(ROSBridge):
    """ROS 2 bridge for DRL agent (Python 3.12 side)."""

    def __init__(self, use_ros: bool = True):
        """Initialize the DRL bridge.

        Args:
            use_ros: Whether to use ROS 2 or fall back to file-based communication
        """
        super().__init__("drl_bridge", use_ros)

        # Last received data
        self.latest_camera_rgb = None
        self.latest_camera_depth = None
        self.latest_camera_semantic = None
        self.latest_state = None
        self.structured_state = {}  # Dictionary of named state components
        self.latest_reward = None
        self.latest_done = False
        self.latest_info = {}

        if self.use_ros:
            # Create QoS profiles
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

            control_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )

            # Publishers
            self.action_pub = self.node.create_publisher(
                Float32MultiArray, 'drl/action', control_qos)
            self.control_pub = self.node.create_publisher(
                String, 'drl/control', control_qos)

            # Subscribers
            self.camera_rgb_sub = self.node.create_subscription(
                Image, 'carla/camera/rgb', self.camera_rgb_callback, sensor_qos)
            self.camera_depth_sub = self.node.create_subscription(
                Image, 'carla/camera/depth', self.camera_depth_callback, sensor_qos)
            self.camera_semantic_sub = self.node.create_subscription(
                Image, 'carla/camera/semantic', self.camera_semantic_callback, sensor_qos)
            self.state_sub = self.node.create_subscription(
                Float32MultiArray, 'carla/state', self.state_callback, sensor_qos)
            self.reward_sub = self.node.create_subscription(
                Float32MultiArray, 'carla/reward', self.reward_callback, sensor_qos)
            self.info_sub = self.node.create_subscription(
                String, 'carla/info', self.info_callback, control_qos)

            # Background spinner
            self.spin_thread = None
            self._start_spin_thread()

    def _start_spin_thread(self):
        """Start a background thread to spin the ROS 2 node."""
        if self.use_ros:
            import threading
            self.spin_thread = threading.Thread(target=self._spin_node)
            self.spin_thread.daemon = True
            self.spin_thread.start()

    def _spin_node(self):
        """Spin the ROS 2 node in a background thread."""
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            time.sleep(0.01)  # Limit CPU usage

    def camera_rgb_callback(self, msg):
        """Callback for receiving RGB camera images."""
        try:
            self.latest_camera_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            logger.error(f"Failed to process RGB camera image: {e}")

    def camera_depth_callback(self, msg):
        """Callback for receiving depth camera images."""
        try:
            self.latest_camera_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            logger.error(f"Failed to process depth camera image: {e}")

    def camera_semantic_callback(self, msg):
        """Callback for receiving semantic segmentation camera images."""
        try:
            self.latest_camera_semantic = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            logger.error(f"Failed to process semantic camera image: {e}")

    def state_callback(self, msg):
        """Callback for receiving state observations."""
        try:
            self.latest_state = np.array(msg.data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to process state: {e}")

    def reward_callback(self, msg):
        """Callback for receiving rewards."""
        try:
            self.latest_reward = msg.data[0]
            self.latest_done = bool(msg.data[1] > 0.5)
        except Exception as e:
            logger.error(f"Failed to process reward: {e}")

    def info_callback(self, msg):
        """Callback for receiving additional information."""
        try:
            info_data = json.loads(msg.data)

            # Check if this is structured state information
            if "state_structure" in info_data:
                self._process_structured_state(info_data["state_structure"])

            self.latest_info = info_data
        except Exception as e:
            logger.error(f"Failed to process info: {e}")

    def _process_structured_state(self, state_structure):
        """Process structured state information.

        Args:
            state_structure: Dictionary containing state component details
        """
        if self.latest_state is not None and "components" in state_structure:
            self.structured_state = {}

            # Extract each component from the combined state array
            for name, info in state_structure["components"].items():
                start_idx = info["start_idx"]
                end_idx = info["end_idx"]
                shape = tuple(info["shape"])

                # Extract and reshape the component
                component_data = self.latest_state[start_idx:end_idx].reshape(shape)
                self.structured_state[name] = component_data

    def publish_action(self, action: np.ndarray):
        """Publish action to CARLA.

        Args:
            action: Action as numpy array
        """
        if self.use_ros:
            try:
                msg = Float32MultiArray()
                msg.data = action.flatten().tolist()
                self.action_pub.publish(msg)
            except Exception as e:
                logger.error(f"Failed to publish action: {e}")
        else:
            # Save action to file
            timestamp = time.time()
            np.save(self.comm_dir / f"action_{timestamp:.6f}.npy", action)
            with open(self.comm_dir / "action_latest.txt", 'w') as f:
                f.write(f"{timestamp:.6f}")

    def publish_control(self, command: str, params: Dict[str, Any] = None):
        """Publish control command to CARLA.

        Args:
            command: Command name
            params: Optional parameters
        """
        if params is None:
            params = {}

        data = {
            "command": command,
            "params": params,
            "timestamp": time.time()
        }

        if self.use_ros:
            try:
                msg = String()
                msg.data = json.dumps(data)
                self.control_pub.publish(msg)
            except Exception as e:
                logger.error(f"Failed to publish control: {e}")
        else:
            # Save control to file
            timestamp = time.time()
            with open(self.comm_dir / f"control_{timestamp:.6f}.json", 'w') as f:
                json.dump(data, f)
            with open(self.comm_dir / "control_latest.txt", 'w') as f:
                f.write(f"{timestamp:.6f}")

    def get_structured_observation(self) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, np.ndarray], Optional[float], bool, Dict]:
        """Get the latest observation with structured state components.

        Returns:
            Tuple of (cameras_dict, structured_state, reward, done, info)

            cameras_dict is a dictionary with keys 'rgb', 'depth', 'semantic'
            structured_state is a dictionary with named state components
        """
        cameras, state, reward, done, info = self.get_latest_observation()
        return cameras, self.structured_state, reward, done, info

    def get_latest_observation(self) -> Tuple[Dict[str, Optional[np.ndarray]], Optional[np.ndarray], Optional[float], bool, Dict]:
        """Get the latest observation when using file-based communication.

        Returns:
            Tuple of (cameras_dict, state, reward, done, info)

            cameras_dict is a dictionary with keys 'rgb', 'depth', 'semantic'
        """
        if self.use_ros:
            cameras = {
                'rgb': self.latest_camera_rgb,
                'depth': self.latest_camera_depth,
                'semantic': self.latest_camera_semantic
            }
            return (cameras, self.latest_state,
                    self.latest_reward, self.latest_done, self.latest_info)

        # File-based communication
        cameras = {
            'rgb': None,
            'depth': None,
            'semantic': None
        }
        state = None
        reward = None
        done = False
        info = {}

        try:
            import cv2

            # Get latest RGB camera
            rgb_meta_file = self.comm_dir / "camera_rgb_latest.json"
            if rgb_meta_file.exists():
                with open(rgb_meta_file, 'r') as f:
                    rgb_meta = json.load(f)
                    timestamp = rgb_meta["timestamp"]

                rgb_file = self.comm_dir / f"camera_rgb_{timestamp:.6f}.jpg"
                if rgb_file.exists():
                    cameras['rgb'] = cv2.imread(str(rgb_file))
                    cameras['rgb'] = cv2.cvtColor(cameras['rgb'], cv2.COLOR_BGR2RGB)

            # Get latest depth camera
            depth_meta_file = self.comm_dir / "camera_depth_latest.json"
            if depth_meta_file.exists():
                with open(depth_meta_file, 'r') as f:
                    depth_meta = json.load(f)
                    timestamp = depth_meta["timestamp"]

                depth_file = self.comm_dir / f"camera_depth_{timestamp:.6f}.jpg"
                if depth_file.exists():
                    cameras['depth'] = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)

            # Get latest semantic camera
            semantic_meta_file = self.comm_dir / "camera_semantic_latest.json"
            if semantic_meta_file.exists():
                with open(semantic_meta_file, 'r') as f:
                    semantic_meta = json.load(f)
                    timestamp = semantic_meta["timestamp"]

                semantic_file = self.comm_dir / f"camera_semantic_{timestamp:.6f}.jpg"
                if semantic_file.exists():
                    cameras['semantic'] = cv2.imread(str(semantic_file), cv2.IMREAD_GRAYSCALE)

            # Get latest state
            state_meta_file = self.comm_dir / "state_latest.txt"
            if state_meta_file.exists():
                with open(state_meta_file, 'r') as f:
                    timestamp = float(f.read().strip())

                state_file = self.comm_dir / f"state_{timestamp:.6f}.npy"
                if state_file.exists():
                    state = np.load(state_file)

            # Get latest reward
            reward_meta_file = self.comm_dir / "reward_latest.txt"
            if reward_meta_file.exists():
                with open(reward_meta_file, 'r') as f:
                    timestamp = float(f.read().strip())

                reward_file = self.comm_dir / f"reward_{timestamp:.6f}.json"
                if reward_file.exists():
                    with open(reward_file, 'r') as f:
                        reward_data = json.load(f)
                        reward = reward_data["reward"]
                        done = reward_data["done"]

            # Get latest info
            info_meta_file = self.comm_dir / "info_latest.txt"
            if info_meta_file.exists():
                with open(info_meta_file, 'r') as f:
                    timestamp = float(f.read().strip())

                info_file = self.comm_dir / f"info_{timestamp:.6f}.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)

        except Exception as e:
            logger.error(f"Failed to get latest observation: {e}")

        return cameras, state, reward, done, info
# If this module is run directly, run a simple test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ROS 2 bridge")
    parser.add_argument("--mode", choices=["carla", "drl"], required=True,
                        help="Whether to run as CARLA or DRL bridge")
    parser.add_argument("--use-ros", action="store_true",
                        help="Use ROS 2 (if available)")
    args = parser.parse_args()

    if args.mode == "carla":
        bridge = CARLABridge(use_ros=args.use_ros)

        # Publish test data
        for i in range(10):
            # Create test images for different camera types
            test_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            test_rgb[:, :, 0] = 255  # Red
            bridge.publish_camera(test_rgb, camera_type='rgb')

            # Create a depth test image
            test_depth = np.ones((480, 640), dtype=np.float32) * i * 0.1  # Increasing depth
            bridge.publish_camera(test_depth, camera_type='depth')

            # Create a semantic segmentation test image
            test_semantic = np.zeros((480, 640), dtype=np.uint8)
            test_semantic[100:200, 100:200] = 1  # Road
            test_semantic[300:400, 300:400] = 2  # Car
            bridge.publish_camera(test_semantic, camera_type='semantic')

            # Create a test state (standard)
            test_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            bridge.publish_state(test_state)

            # Create a structured test state
            if i % 2 == 0:  # Every other iteration
                test_structured_state = {
                    'position': np.array([10.0 + i, 20.0, 30.0], dtype=np.float32),
                    'velocity': np.array([1.0, 2.0 - i*0.1, 0.0], dtype=np.float32),
                    'orientation': np.array([0.0, 0.0, i*0.1], dtype=np.float32)
                }
                bridge.publish_structured_state(test_structured_state)

            # Create a test reward
            bridge.publish_reward(i * 0.1, i == 9)

            # Create test info
            bridge.publish_info({"step": i, "message": "Test message"})

            time.sleep(1.0)

    elif args.mode == "drl":
        bridge = DRLBridge(use_ros=args.use_ros)

        # Receive and print data
        for i in range(10):
            cameras, state, reward, done, info = bridge.get_latest_observation()

            print(f"Step {i}:")
            print(f"  RGB Camera: {None if cameras['rgb'] is None else cameras['rgb'].shape}")
            print(f"  Depth Camera: {None if cameras['depth'] is None else cameras['depth'].shape}")
            print(f"  Semantic Camera: {None if cameras['semantic'] is None else cameras['semantic'].shape}")
            print(f"  State: {state}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            print(f"  Info: {info}")

            # Try to get structured observation
            try:
                cameras, structured_state, reward, done, info = bridge.get_structured_observation()
                print("  Structured State:")
                for key, value in structured_state.items():
                    print(f"    {key}: {value.shape if value is not None else None}")
            except Exception as e:
                print(f"  No structured state available: {e}")            # Send a test action
            test_action = np.array([0.5, -0.5], dtype=np.float32)
            bridge.publish_action(test_action)

            # Send a test control
            bridge.publish_control("reset", {"seed": 42})

            time.sleep(1.0)

    # Shutdown
    if args.mode == "carla":
        bridge.shutdown()
    else:
        bridge.shutdown()

    print("Done!")

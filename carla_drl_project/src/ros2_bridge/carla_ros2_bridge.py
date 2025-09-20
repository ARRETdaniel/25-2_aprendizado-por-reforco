#!/usr/bin/env python3
"""
CARLA ROS 2 Bridge for Deep Reinforcement Learning

This module implements a professional ROS 2 bridge connecting CARLA simulator
with TD3 agents. Designed following clean code principles and best practices.

Architecture:
    CarlaRos2Bridge: Main bridge coordinating data flow
    MessageConverter: CARLA ↔ ROS message conversion utilities
    PublisherManager: Organized ROS 2 publisher management
    SubscriberManager: Organized ROS 2 subscriber management

Design Principles:
    - Single Responsibility: Each class has one clear purpose
    - Dependency Injection: ROS nodes and configuration passed explicitly
    - Error Handling: Graceful failure with comprehensive logging
    - Performance: Real-time capable with configurable rates
    - Maintainability: Clear interfaces and extensive documentation

Author: DRL CARLA Project Team
Created: September 2025
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from threading import Lock, Event
import numpy as np

# ROS 2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    
    # ROS 2 message types
    from sensor_msgs.msg import Image, CompressedImage
    from geometry_msgs.msg import Twist, Vector3
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64, Header
    from builtin_interfaces.msg import Time
    
    print("✅ ROS 2 modules loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import ROS 2: {e}")
    print("Please ensure ROS 2 Foxy is installed and sourced")
    exit(1)

# Import our CARLA client
try:
    from carla_interface.carla_client import CarlaClient, CarlaConfig
    print("✅ CARLA client imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CARLA client: {e}")
    print("Please ensure carla_client.py is available")
    exit(1)


@dataclass
class BridgeConfig:
    """
    Configuration for CARLA-ROS 2 bridge operation.
    
    Centralizes all configuration parameters for maintainability
    and clear separation between environment and code.
    """
    # Node configuration
    node_name: str = 'carla_ros2_bridge'
    namespace: str = '/carla'
    
    # Publishing rates (Hz)
    camera_rate: float = 10.0  # Match CARLA sensor tick
    vehicle_state_rate: float = 20.0  # Smooth control feedback
    control_timeout: float = 0.1  # Max time without control commands
    
    # Topic names
    camera_topic: str = '/carla/camera/image_raw'
    compressed_camera_topic: str = '/carla/camera/image_raw/compressed'
    vehicle_state_topic: str = '/carla/vehicle/odometry'
    control_topic: str = '/carla/vehicle/cmd_vel'
    reward_topic: str = '/carla/reward'
    
    # Message configuration
    camera_frame_id: str = 'carla_camera'
    vehicle_frame_id: str = 'carla_vehicle'
    use_compressed_images: bool = True  # Optimize bandwidth
    
    # Performance settings
    max_queue_size: int = 10
    enable_reliability_best_effort: bool = True  # For real-time performance


class MessageConverter:
    """
    Utility class for converting between CARLA and ROS 2 message formats.
    
    This class handles the conversion logic, ensuring data integrity
    and optimal performance for real-time applications.
    """
    
    @staticmethod
    def carla_image_to_ros_image(carla_image_array: np.ndarray, 
                                frame_id: str = "carla_camera",
                                timestamp: Optional[float] = None) -> Image:
        """
        Convert CARLA image array to ROS 2 Image message.
        
        Args:
            carla_image_array: BGR image array from CARLA
            frame_id: Frame identifier for the image
            timestamp: Optional timestamp (uses current time if None)
            
        Returns:
            ROS 2 Image message
        """
        image_msg = Image()
        
        # Set header information
        image_msg.header.frame_id = frame_id
        if timestamp:
            # Convert timestamp to ROS 2 Time
            sec = int(timestamp)
            nanosec = int((timestamp - sec) * 1e9)
            image_msg.header.stamp = Time(sec=sec, nanosec=nanosec)
        
        # Set image properties
        height, width, channels = carla_image_array.shape
        image_msg.height = height
        image_msg.width = width
        image_msg.encoding = 'bgr8'  # CARLA uses BGR format
        image_msg.is_bigendian = False
        image_msg.step = width * channels
        
        # Set image data
        image_msg.data = carla_image_array.flatten().tobytes()
        
        return image_msg
    
    @staticmethod
    def carla_vehicle_state_to_odometry(location, velocity, timestamp: Optional[float] = None) -> Odometry:
        """
        Convert CARLA vehicle state to ROS 2 Odometry message.
        
        Args:
            location: CARLA Location object
            velocity: CARLA Vector3D object  
            timestamp: Optional timestamp
            
        Returns:
            ROS 2 Odometry message
        """
        odom_msg = Odometry()
        
        # Set header
        odom_msg.header.frame_id = "carla_world"
        odom_msg.child_frame_id = "carla_vehicle"
        if timestamp:
            sec = int(timestamp)
            nanosec = int((timestamp - sec) * 1e9)
            odom_msg.header.stamp = Time(sec=sec, nanosec=nanosec)
        
        # Set position (CARLA uses UE4 coordinate system)
        odom_msg.pose.pose.position.x = location.x
        odom_msg.pose.pose.position.y = -location.y  # Convert to ROS convention
        odom_msg.pose.pose.position.z = location.z
        
        # Set velocity
        odom_msg.twist.twist.linear.x = velocity.x
        odom_msg.twist.twist.linear.y = -velocity.y  # Convert to ROS convention
        odom_msg.twist.twist.linear.z = velocity.z
        
        return odom_msg
    
    @staticmethod
    def ros_twist_to_carla_control(twist_msg: Twist) -> Dict[str, float]:
        """
        Convert ROS 2 Twist message to CARLA vehicle control.
        
        Args:
            twist_msg: ROS 2 Twist message with control commands
            
        Returns:
            Dictionary with throttle, steer, brake values
        """
        # Extract control values from twist message
        # Linear.x = throttle/brake combined (-1 to 1)
        # Angular.z = steering (-1 to 1)
        
        linear_x = np.clip(twist_msg.linear.x, -1.0, 1.0)
        angular_z = np.clip(twist_msg.angular.z, -1.0, 1.0)
        
        # Convert to CARLA control format
        if linear_x >= 0:
            throttle = linear_x
            brake = 0.0
        else:
            throttle = 0.0
            brake = -linear_x
        
        steer = angular_z
        
        return {
            'throttle': float(throttle),
            'steer': float(steer), 
            'brake': float(brake)
        }


class CarlaRos2Bridge(Node):
    """
    Main ROS 2 bridge node for CARLA DRL integration.
    
    This node coordinates data flow between CARLA simulator and ROS 2,
    providing a clean interface for TD3 agents to interact with the simulation.
    
    Responsibilities:
    - Publish CARLA sensor data as ROS 2 messages
    - Subscribe to control commands from DRL agents
    - Maintain synchronization and timing
    - Handle errors gracefully with logging
    """
    
    def __init__(self, bridge_config: BridgeConfig, carla_config: CarlaConfig):
        """
        Initialize ROS 2 bridge node.
        
        Args:
            bridge_config: Bridge-specific configuration
            carla_config: CARLA client configuration
        """
        super().__init__(bridge_config.node_name, namespace=bridge_config.namespace)
        
        # Store configurations
        self._bridge_config = bridge_config
        self._carla_config = carla_config
        
        # Initialize CARLA client
        self._carla_client = CarlaClient(carla_config)
        
        # State management
        self._is_running = False
        self._last_control_time = time.time()
        self._bridge_lock = Lock()
        self._shutdown_event = Event()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize ROS 2 components
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_timers()
        
        self.get_logger().info("🔗 CARLA-ROS 2 Bridge initialized")
    
    def _setup_logging(self) -> None:
        """Configure logging for the bridge."""
        self._logger = self.get_logger()
        self._logger.info("📝 Bridge logging configured")
    
    def _setup_publishers(self) -> None:
        """Setup ROS 2 publishers for CARLA data."""
        # QoS profiles for different data types
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT if self._bridge_config.enable_reliability_best_effort else QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self._bridge_config.max_queue_size
        )
        
        control_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Only latest control command matters
        )
        
        # Create publishers
        if self._bridge_config.use_compressed_images:
            self._camera_publisher = self.create_publisher(
                CompressedImage, 
                self._bridge_config.compressed_camera_topic, 
                sensor_qos
            )
        else:
            self._camera_publisher = self.create_publisher(
                Image, 
                self._bridge_config.camera_topic, 
                sensor_qos
            )
        
        self._vehicle_state_publisher = self.create_publisher(
            Odometry, 
            self._bridge_config.vehicle_state_topic, 
            control_qos
        )
        
        self._reward_publisher = self.create_publisher(
            Float64, 
            self._bridge_config.reward_topic, 
            control_qos
        )
        
        self._logger.info("📡 ROS 2 publishers configured")
    
    def _setup_subscribers(self) -> None:
        """Setup ROS 2 subscribers for control commands."""
        self._control_subscriber = self.create_subscription(
            Twist,
            self._bridge_config.control_topic,
            self._control_callback,
            10
        )
        
        self._logger.info("📥 ROS 2 subscribers configured")
    
    def _setup_timers(self) -> None:
        """Setup periodic timers for data publishing."""
        # Camera data timer
        camera_period = 1.0 / self._bridge_config.camera_rate
        self._camera_timer = self.create_timer(camera_period, self._publish_camera_data)
        
        # Vehicle state timer
        state_period = 1.0 / self._bridge_config.vehicle_state_rate
        self._state_timer = self.create_timer(state_period, self._publish_vehicle_state)
        
        # Control timeout checker
        self._control_timer = self.create_timer(0.1, self._check_control_timeout)
        
        self._logger.info("⏰ Timers configured")
    
    def start_bridge(self) -> bool:
        """
        Start the CARLA-ROS 2 bridge operation.
        
        Returns:
            True if bridge started successfully, False otherwise
        """
        try:
            self._logger.info("🚀 Starting CARLA-ROS 2 bridge...")
            
            # Connect to CARLA
            if not self._carla_client.connect():
                self._logger.error("❌ Failed to connect to CARLA")
                return False
            
            # Spawn vehicle
            if not self._carla_client.spawn_vehicle():
                self._logger.error("❌ Failed to spawn vehicle")
                return False
            
            # Setup sensors
            if not self._carla_client.setup_sensors():
                self._logger.error("❌ Failed to setup sensors")
                return False
            
            # Wait for sensor data
            self._logger.info("⏳ Waiting for sensor initialization...")
            timeout = time.time() + 5.0
            while not self._carla_client.data_manager.has_camera_data() and time.time() < timeout:
                time.sleep(0.1)
            
            if not self._carla_client.data_manager.has_camera_data():
                self._logger.error("❌ Sensor data not available")
                return False
            
            self._is_running = True
            self._logger.info("✅ CARLA-ROS 2 bridge started successfully")
            
            return True
            
        except Exception as e:
            self._logger.error(f"❌ Bridge start failed: {e}")
            return False
    
    def _publish_camera_data(self) -> None:
        """Publish camera data from CARLA to ROS 2."""
        if not self._is_running:
            return
        
        try:
            # Get latest camera frame
            frame = self._carla_client.data_manager.get_latest_camera_frame()
            if frame is None:
                return
            
            # Convert to ROS 2 message
            if self._bridge_config.use_compressed_images:
                # TODO: Implement compressed image conversion
                pass
            else:
                image_msg = MessageConverter.carla_image_to_ros_image(
                    frame, 
                    self._bridge_config.camera_frame_id,
                    time.time()
                )
                self._camera_publisher.publish(image_msg)
            
        except Exception as e:
            self._logger.error(f"❌ Camera data publishing failed: {e}")
    
    def _publish_vehicle_state(self) -> None:
        """Publish vehicle state from CARLA to ROS 2."""
        if not self._is_running or not self._carla_client.vehicle:
            return
        
        try:
            # Get vehicle state
            location = self._carla_client.vehicle.get_location()
            velocity = self._carla_client.vehicle.get_velocity()
            
            # Convert to ROS 2 message
            odom_msg = MessageConverter.carla_vehicle_state_to_odometry(
                location, velocity, time.time()
            )
            
            self._vehicle_state_publisher.publish(odom_msg)
            
        except Exception as e:
            self._logger.error(f"❌ Vehicle state publishing failed: {e}")
    
    def _control_callback(self, msg: Twist) -> None:
        """
        Handle control commands from DRL agent.
        
        Args:
            msg: ROS 2 Twist message with control commands
        """
        try:
            with self._bridge_lock:
                # Convert ROS message to CARLA control
                control_dict = MessageConverter.ros_twist_to_carla_control(msg)
                
                # Apply control to vehicle
                self._carla_client.apply_control(**control_dict)
                
                # Update control timestamp
                self._last_control_time = time.time()
                
        except Exception as e:
            self._logger.error(f"❌ Control callback failed: {e}")
    
    def _check_control_timeout(self) -> None:
        """Check for control command timeout and apply safety stop."""
        if not self._is_running:
            return
        
        time_since_control = time.time() - self._last_control_time
        if time_since_control > self._bridge_config.control_timeout:
            # Apply safety stop
            self._carla_client.apply_control(throttle=0.0, steer=0.0, brake=1.0)
    
    def stop_bridge(self) -> None:
        """Stop the bridge and cleanup resources."""
        self._logger.info("🛑 Stopping CARLA-ROS 2 bridge...")
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cleanup CARLA client
        self._carla_client.cleanup()
        
        self._logger.info("✅ Bridge stopped successfully")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get bridge performance metrics."""
        return {
            'camera_frames': self._carla_client.data_manager.get_frame_count(),
            'is_running': self._is_running,
            'last_control_time': self._last_control_time,
            'vehicle_active': self._carla_client.vehicle is not None
        }


def main():
    """
    Main function to run CARLA-ROS 2 bridge.
    
    This function demonstrates proper usage of the bridge with
    clean initialization and resource management.
    """
    # Initialize ROS 2
    rclpy.init()
    
    try:
        # Create configurations
        bridge_config = BridgeConfig()
        carla_config = CarlaConfig()
        
        # Create bridge node
        bridge = CarlaRos2Bridge(bridge_config, carla_config)
        
        # Start bridge
        if not bridge.start_bridge():
            bridge.get_logger().error("❌ Failed to start bridge")
            return 1
        
        bridge.get_logger().info("🎮 CARLA-ROS 2 Bridge Ready!")
        bridge.get_logger().info("=" * 50)
        bridge.get_logger().info("📋 Bridge is now active:")
        bridge.get_logger().info(f"  • Camera topic: {bridge_config.camera_topic}")
        bridge.get_logger().info(f"  • Control topic: {bridge_config.control_topic}")
        bridge.get_logger().info(f"  • Vehicle state: {bridge_config.vehicle_state_topic}")
        bridge.get_logger().info("  • Ready for TD3 agent connection")
        bridge.get_logger().info("=" * 50)
        
        # Run ROS 2 spin
        try:
            rclpy.spin(bridge)
        except KeyboardInterrupt:
            bridge.get_logger().info("🛑 Interrupted by user")
        
        return 0
        
    except Exception as e:
        print(f"❌ Bridge execution failed: {e}")
        return 1
    
    finally:
        # Cleanup
        if 'bridge' in locals():
            bridge.stop_bridge()
            bridge.destroy_node()
        
        rclpy.shutdown()


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

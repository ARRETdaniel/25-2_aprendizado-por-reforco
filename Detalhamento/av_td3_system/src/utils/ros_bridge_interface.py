#!/usr/bin/env python3
"""
ROS 2 Bridge Interface for CARLA Vehicle Control

This module provides a minimal interface to publish vehicle control commands
via ROS 2 topics to the external CARLA ROS Bridge using standard geometry_msgs/Twist.

Architecture:
    Training Container (Ubuntu 22.04) → Native rclpy → ROS 2 Topics →
    Twist to Control Node (ROS Bridge) → CARLA Python API → CARLA Server

Why Twist instead of CarlaEgoVehicleControl?
    - geometry_msgs/Twist is a STANDARD ROS 2 message (no extra packages needed)
    - CARLA ROS Bridge includes carla_twist_to_control node that converts Twist → VehicleControl
    - This keeps our training container lightweight (no need to build carla_msgs)

Reference:
    - CARLA Twist to Control: https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_twist_to_control/
    - geometry_msgs/Twist: https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/Twist.html
    - ROS 2 Python Tutorial: https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html

Topic:
    /carla/ego_vehicle/twist (geometry_msgs/msg/Twist)

Message mapping (Twist → CARLA Control):
    - twist.linear.x > 0 → throttle (normalized by MAX_LON_ACCELERATION=10)
    - twist.linear.x < 0 → reverse throttle
    - twist.angular.z → steering (normalized by max_steering_angle)

Author: Daniel Terra
Date: 2025-01-25 (Phase 5 - Native ROS 2 Migration)
"""

import logging
from typing import Optional
import carla

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, Vector3
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("[ROS BRIDGE] rclpy not available - ROS 2 control disabled")


class ROSBridgeInterface:
    """
    Minimal ROS 2 publisher for CARLA vehicle control using geometry_msgs/Twist.

    Publishes Twist messages to the CARLA Twist to Control node which converts
    them to CARLA VehicleControl commands. This approach uses standard ROS messages,
    eliminating the need for carla_msgs package in the training container.

    Conversion Logic (from carla_twist_to_control.py):
        - twist.linear.x > 0  → throttle = min(10, x) / 10
        - twist.linear.x < 0  → reverse = True, throttle = max(-10, x) / -10
        - twist.angular.z     → steer = -z / max_steering_angle (normalized)

    Usage:
        # Initialize (once per environment)
        ros_interface = ROSBridgeInterface(node_name='carla_env_controller')

        # Publish control commands (every step)
        ros_interface.publish_control(throttle=0.5, steer=0.1, brake=0.0)

        # Cleanup
        ros_interface.destroy()
    """

    # Maximum longitudinal acceleration for Twist conversion
    # This matches the carla_twist_to_control.py implementation
    MAX_LON_ACCELERATION = 10.0

    # Approximate max steering angle (radians) for Tesla Model 3
    # This will be updated if we receive vehicle info from ROS Bridge
    DEFAULT_MAX_STEER_ANGLE = 1.22  # ~70 degrees

    def __init__(
        self,
        node_name: str = 'carla_vehicle_controller',
        role_name: str = 'ego_vehicle',
        use_docker_exec: bool = False  # Deprecated parameter (kept for backward compatibility)
    ):
        """
        Initialize ROS 2 publisher for vehicle control via Twist messages.

        Args:
            node_name: Name of the ROS 2 node
            role_name: Role name of the ego vehicle (default: 'ego_vehicle')
            use_docker_exec: DEPRECATED - no longer used (native rclpy always)

        Raises:
            RuntimeError: If ROS 2 is not available
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Debug: Log initialization attempt
        self.logger.info(f"[ROS BRIDGE] Attempting to initialize ROSBridgeInterface...")
        self.logger.info(f"[ROS BRIDGE] Node name: {node_name}, Role: {role_name}")

        if not ROS2_AVAILABLE:
            error_msg = (
                "rclpy not available. "
                "Ensure ROS 2 Humble is installed and sourced: "
                "source /opt/ros/humble/setup.bash"
            )
            self.logger.error(f"[ROS BRIDGE] {error_msg}")
            raise RuntimeError(error_msg)

        self.logger.info(f"[ROS BRIDGE] rclpy module found, proceeding with initialization...")

        self.node_name = node_name
        self.role_name = role_name
        self.node: Optional[Node] = None
        self.publisher = None
        self.max_steering_angle = self.DEFAULT_MAX_STEER_ANGLE
        self._message_count = 0  # Track published messages for diagnostic logging

        # Initialize ROS 2
        self.logger.info(f"[ROS BRIDGE] Calling _initialize_ros()...")
        self._initialize_ros()
        self.logger.info(f"[ROS BRIDGE] _initialize_ros() completed successfully!")

        self.logger.info(f"[ROS BRIDGE] Initialized native rclpy Twist publisher")
        self.logger.info(f"[ROS BRIDGE] Node name: {node_name}")
        self.logger.info(f"[ROS BRIDGE] Topic: /carla/{role_name}/twist")
        self.logger.info(f"[ROS BRIDGE] Using geometry_msgs/Twist (standard ROS 2 message)")

    def _initialize_ros(self):
        """Initialize ROS 2 node and Twist publisher."""
        try:
            self.logger.info("[ROS BRIDGE] Step 1: Checking rclpy context...")
            # Initialize rclpy (if not already initialized)
            if not rclpy.ok():
                self.logger.info("[ROS BRIDGE] Step 2: Initializing rclpy context...")
                rclpy.init()
                self.logger.info("[ROS BRIDGE] Step 3: rclpy context initialized successfully!")
            else:
                self.logger.info("[ROS BRIDGE] Step 2: rclpy already initialized, skipping init()")

            # Create ROS 2 node
            self.logger.info(f"[ROS BRIDGE] Step 4: Creating ROS 2 node '{self.node_name}'...")
            self.node = Node(self.node_name)
            self.logger.info(f"[ROS BRIDGE] Step 5: Node created successfully!")

            # Create publisher for Twist commands
            # Topic: /carla/ego_vehicle/twist
            # Message type: geometry_msgs/msg/Twist (standard ROS 2)
            # QoS: Queue size 10 (standard for control commands)
            topic_name = f'/carla/{self.role_name}/twist'
            self.logger.info(f"[ROS BRIDGE] Step 6: Creating publisher on topic: {topic_name}")

            self.publisher = self.node.create_publisher(
                Twist,
                topic_name,
                10  # QoS queue size
            )

            self.logger.info(f"[ROS BRIDGE] Step 7: Publisher created successfully!")
            self.logger.info(f"[ROS BRIDGE] Topic: {topic_name}")
            self.logger.info(f"[ROS BRIDGE] Message type: geometry_msgs/msg/Twist")

        except Exception as e:
            self.logger.error(f"[ROS BRIDGE] FAILED to initialize ROS 2: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"[ROS BRIDGE] Traceback:\n{traceback.format_exc()}")
            raise

    def publish_control(
        self,
        control: Optional[carla.VehicleControl] = None,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
        gear: int = 0,
        manual_gear_shift: bool = False
    ) -> bool:
        """
        Publish CARLA vehicle control as geometry_msgs/Twist message.

        Converts CARLA control commands to Twist format for the ROS Bridge.

        Conversion Logic (matches carla_twist_to_control.py):
            - throttle/brake → twist.linear.x (m/s acceleration)
            - steering → twist.angular.z (radians)

        Supports two calling patterns:
        1. publish_control(control=carla.VehicleControl(...))
        2. publish_control(throttle=0.5, steer=0.1, brake=0.0)

        Args:
            control: CARLA VehicleControl object (if provided, other params ignored)
            throttle: Throttle value [0.0, 1.0]
            steer: Steering value [-1.0, 1.0]
            brake: Brake value [0.0, 1.0]
            hand_brake: Hand brake enabled (triggers full stop in Twist)
            reverse: Reverse gear enabled
            gear: Gear number (unused in Twist conversion)
            manual_gear_shift: Manual gear shift (unused in Twist conversion)

        Returns:
            True if published successfully, False otherwise
        """
        if not self.publisher:
            self.logger.error("[ROS BRIDGE] Publisher not initialized")
            return False

        try:
            # Extract control values
            if control is not None:
                # Use VehicleControl object
                throttle_val = float(control.throttle)
                steer_val = float(control.steer)
                brake_val = float(control.brake)
                hand_brake_val = bool(control.hand_brake)
                reverse_val = bool(control.reverse)
            else:
                # Use individual parameters
                throttle_val = float(throttle)
                steer_val = float(steer)
                brake_val = float(brake)
                hand_brake_val = bool(hand_brake)
                reverse_val = bool(reverse)

            # Create Twist message
            msg = Twist()

            # Handle special case: hand brake or full brake → full stop
            if hand_brake_val or brake_val >= 0.99:
                msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
                msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
            else:
                # Convert throttle/brake to longitudinal velocity command
                # Positive linear.x → forward throttle
                # Negative linear.x → reverse throttle
                # Scale by MAX_LON_ACCELERATION (10 m/s²)
                if brake_val > 0.01:
                    # Braking: map brake [0,1] → velocity [-10, 0]
                    # Simplified: just send negative acceleration
                    linear_x = -brake_val * self.MAX_LON_ACCELERATION
                elif reverse_val:
                    # Reverse: negative velocity
                    linear_x = -throttle_val * self.MAX_LON_ACCELERATION
                else:
                    # Forward: positive velocity
                    linear_x = throttle_val * self.MAX_LON_ACCELERATION

                # Convert steering to angular velocity (radians)
                # CARLA steer ∈ [-1, 1] → angular_z ∈ [-max_angle, +max_angle]
                # NOTE: The carla_twist_to_control.py uses NEGATIVE mapping
                angular_z = -steer_val * self.max_steering_angle

                msg.linear = Vector3(x=linear_x, y=0.0, z=0.0)
                msg.angular = Vector3(x=0.0, y=0.0, z=angular_z)

            # Publish message
            self.publisher.publish(msg)

            # Spin once to process callbacks (non-blocking)
            rclpy.spin_once(self.node, timeout_sec=0.0)

            # Increment message counter
            self._message_count += 1

            # Log first 10 messages at INFO level for diagnostics
            if self._message_count <= 10:
                self.logger.info(
                    f"[ROS BRIDGE] Published Twist #{self._message_count}: "
                    f"linear.x={msg.linear.x:.2f} m/s, angular.z={msg.angular.z:.3f} rad "
                    f"(throttle={throttle_val:.2f}, steer={steer_val:.3f}, brake={brake_val:.2f})"
                )
            elif self._message_count % 100 == 0:
                # Log every 100th message thereafter
                self.logger.debug(
                    f"[ROS BRIDGE] Published Twist #{self._message_count}: "
                    f"linear.x={msg.linear.x:.2f} m/s, angular.z={msg.angular.z:.3f} rad"
                )
            else:
                # Normal debug logging
                self.logger.debug(
                    f"[ROS BRIDGE] Published Twist: "
                    f"linear.x={msg.linear.x:.2f} m/s, angular.z={msg.angular.z:.3f} rad "
                    f"(throttle={throttle_val:.2f}, steer={steer_val:.3f}, brake={brake_val:.2f})"
                )

            return True

        except Exception as e:
            self.logger.error(f"[ROS BRIDGE] Failed to publish Twist: {e}")
            return False

    def wait_for_topics(self, timeout: float = 10.0) -> bool:
        """
        Wait for ROS Bridge topics to become available.

        This verifies that the external CARLA ROS Bridge with Twist to Control
        node is running and ready to receive Twist commands.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if topics are available, False on timeout
        """
        import time

        if not self.node:
            self.logger.error("[ROS BRIDGE] Node not initialized")
            return False

        self.logger.info(f"[ROS BRIDGE] Waiting for Twist topic (timeout: {timeout}s)...")

        start_time = time.time()
        topic_name = f'/carla/{self.role_name}/twist'

        while (time.time() - start_time) < timeout:
            # Get list of current topics
            topic_names_and_types = self.node.get_topic_names_and_types()
            topic_names = [name for name, _ in topic_names_and_types]

            # Check if our Twist topic exists
            # NOTE: The twist topic is created by OUR publisher, so it will exist
            # We should check for the ROS Bridge topics instead:
            # - /carla/ego_vehicle/vehicle_info (indicates bridge is running)
            # - /carla/ego_vehicle/vehicle_control_cmd (indicates twist_to_control is running)
            check_topics = [
                f'/carla/{self.role_name}/vehicle_info',
                f'/carla/{self.role_name}/vehicle_control_cmd'
            ]

            found_count = sum(1 for t in check_topics if t in topic_names)

            if found_count >= 1:  # At least vehicle_info should exist
                self.logger.info(
                    f"[ROS BRIDGE] ROS Bridge topics found ({found_count}/{len(check_topics)})!"
                )
                return True

            # Wait a bit before checking again
            time.sleep(0.5)
            rclpy.spin_once(self.node, timeout_sec=0.0)

        self.logger.warning(
            f"[ROS BRIDGE] Timeout waiting for ROS Bridge topics\n"
            f"[ROS BRIDGE] Expected topics: {check_topics}\n"
            f"[ROS BRIDGE] Available topics (sample): {topic_names[:10]}\n"
            f"[ROS BRIDGE] Make sure CARLA ROS Bridge is running with carla_twist_to_control!"
        )
        return False

    def destroy(self):
        """Cleanup ROS 2 resources."""
        try:
            if self.node:
                self.node.destroy_node()
                self.logger.info("[ROS BRIDGE] Node destroyed")

            # Note: We don't call rclpy.shutdown() here because other nodes
            # might be using the same rclpy context

        except Exception as e:
            self.logger.warning(f"[ROS BRIDGE] Cleanup warning: {e}")

    def __del__(self):
        """Destructor - ensure cleanup."""
        self.destroy()

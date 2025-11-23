#!/usr/bin/env python3
"""
Minimal ROS Bridge Test - Following Official Documentation
Phase 2.2 - Simplified Vehicle Control Test

Based on official CARLA ROS Bridge documentation:
https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/

Key Findings from Documentation:
1. Manual control node creates "manual override mode" that blocks automatic control
2. We should NOT use carla_ros_bridge_with_example_ego_vehicle.launch.py for automated control
3. We should use carla_ros_bridge.launch.py + spawn objects separately
4. Default mode reads from /carla/<ROLE NAME>/vehicle_control_cmd (normal mode)

This test follows the minimal setup pattern from official docs.

Date: 2025-11-22
Phase: 2.2 - ROS Bridge Vehicle Control Verification
"""

import sys
import time
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Bool


class MinimalBridgeController(Node):
    """
    Minimal test controller following official ROS Bridge patterns.

    Architecture (from official docs):
    - Normal mode: Reads from /carla/<ROLE NAME>/vehicle_control_cmd
    - Manual mode: Reads from /carla/<ROLE NAME>/vehicle_control_cmd_manual
    - Toggle: Publish to /carla/<ROLE NAME>/vehicle_control_manual_override

    We will NOT launch manual_control node to avoid override mode.
    """

    def __init__(self):
        super().__init__('minimal_bridge_controller')

        # State tracking
        self.initial_position = None
        self.current_position = None
        self.odometry_received = False
        self.test_phase = 'waiting'  # waiting -> brake_release -> testing -> stopping -> done
        self.test_start_time = None
        self.brake_release_start = None

        # ROS 2 Subscribers
        self.odometry_subscriber = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )

        # ROS 2 Publishers
        self.control_publisher = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',  # Normal mode (NOT manual)
            10
        )

        # Explicitly disable manual override (ensure normal mode)
        self.manual_override_publisher = self.create_publisher(
            Bool,
            '/carla/ego_vehicle/vehicle_control_manual_override',
            10
        )

        # Timer for control loop (20 Hz = 0.05s)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('🚀 Minimal ROS Bridge Controller initialized')
        self.get_logger().info('📡 Subscribing to: /carla/ego_vehicle/odometry')
        self.get_logger().info('📤 Publishing to: /carla/ego_vehicle/vehicle_control_cmd')

    def odometry_callback(self, msg: Odometry):
        """Callback for odometry data."""
        self.current_position = msg.pose.pose.position

        if not self.odometry_received:
            self.initial_position = self.current_position
            self.odometry_received = True
            self.get_logger().info(
                f'✅ Odometry received! Initial position: '
                f'x={self.initial_position.x:.2f}, '
                f'y={self.initial_position.y:.2f}, '
                f'z={self.initial_position.z:.2f}'
            )

            # Explicitly disable manual override
            override_msg = Bool()
            override_msg.data = False
            self.manual_override_publisher.publish(override_msg)
            self.get_logger().info('🔓 Manual override DISABLED (normal mode active)')

    def calculate_distance_moved(self):
        """Calculate distance from initial position."""
        if self.initial_position is None or self.current_position is None:
            return 0.0

        dx = self.current_position.x - self.initial_position.x
        dy = self.current_position.y - self.initial_position.y
        dz = self.current_position.z - self.initial_position.z

        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def control_loop(self):
        """Main control loop (20 Hz)."""

        # Phase 1: Wait for odometry
        if self.test_phase == 'waiting':
            if self.odometry_received:
                self.test_phase = 'brake_release'
                self.brake_release_start = time.time()
                self.get_logger().info('🔓 Releasing initial brake...')
            return

        # Phase 2: Release brake for 1 second before testing
        elif self.test_phase == 'brake_release':
            elapsed = time.time() - self.brake_release_start

            # Publish control command with NO throttle, NO brake, MANUAL GEAR=1
            msg = CarlaEgoVehicleControl()
            msg.throttle = 0.0
            msg.steer = 0.0
            msg.brake = 0.0  # Release brake
            msg.hand_brake = False
            msg.reverse = False
            msg.manual_gear_shift = True  # ENABLE manual gear control - CRITICAL FIX
            msg.gear = 1  # First gear (now respected because manual_gear_shift=True)
            self.control_publisher.publish(msg)

            # After 1 second, start throttle test
            if elapsed >= 1.0:
                self.test_phase = 'testing'
                self.test_start_time = time.time()
                self.get_logger().info('🏁 Starting test: throttle=0.5, gear=1 (manual) for 5 seconds')
            return

        # Phase 3: Apply throttle for 5 seconds
        elif self.test_phase == 'testing':
            elapsed = time.time() - self.test_start_time

            # Publish control command with MANUAL GEAR
            msg = CarlaEgoVehicleControl()
            msg.throttle = 0.5
            msg.steer = 0.0
            msg.brake = 0.0
            msg.hand_brake = False
            msg.reverse = False
            msg.manual_gear_shift = True  # Keep manual gear control enabled
            msg.gear = 1  # Maintain first gear
            self.control_publisher.publish(msg)

            # Log progress every second
            if int(elapsed) > int(elapsed - 0.05):
                distance = self.calculate_distance_moved()
                self.get_logger().info(f'  t={elapsed:.1f}s: distance = {distance:.2f}m')

            # After 5 seconds, stop
            if elapsed >= 5.0:
                self.test_phase = 'stopping'
                self.get_logger().info('🛑 Test complete, applying brake')

        # Phase 4: Apply brake for 1 second
        elif self.test_phase == 'stopping':
            msg = CarlaEgoVehicleControl()
            msg.throttle = 0.0
            msg.steer = 0.0
            msg.brake = 1.0
            msg.hand_brake = False
            msg.reverse = False
            msg.manual_gear_shift = False
            msg.gear = 1
            self.control_publisher.publish(msg)

            elapsed = time.time() - self.test_start_time
            if elapsed >= 6.0:
                self.test_phase = 'done'

        # Phase 5: Report results
        elif self.test_phase == 'done':
            distance = self.calculate_distance_moved()
            self.get_logger().info('')
            self.get_logger().info('='*60)
            self.get_logger().info('📊 TEST RESULTS')
            self.get_logger().info('='*60)
            self.get_logger().info(f'Initial position: x={self.initial_position.x:.2f}, '
                                  f'y={self.initial_position.y:.2f}, '
                                  f'z={self.initial_position.z:.2f}')
            self.get_logger().info(f'Final position:   x={self.current_position.x:.2f}, '
                                  f'y={self.current_position.y:.2f}, '
                                  f'z={self.current_position.z:.2f}')
            self.get_logger().info(f'📏 Distance moved: {distance:.2f} meters')
            self.get_logger().info('='*60)

            if distance > 0.5:
                self.get_logger().info('✅ ✅ ✅ SUCCESS! ✅ ✅ ✅')
                self.get_logger().info('Vehicle responded to control commands!')
                exit_code = 0
            else:
                self.get_logger().error('❌ ❌ ❌ FAILED! ❌ ❌ ❌')
                self.get_logger().error('Vehicle did NOT respond to control commands')
                self.get_logger().error('Expected > 0.5m movement with 5s of throttle=0.5')
                exit_code = 1

            self.get_logger().info('='*60)

            # Stop the test
            time.sleep(1.0)
            sys.exit(exit_code)


def main(args=None):
    rclpy.init(args=args)

    controller = MinimalBridgeController()

    exit_code = 0
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Test interrupted by user')
        exit_code = 130
    except Exception as e:
        controller.get_logger().error(f'Test failed with exception: {e}')
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        controller.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

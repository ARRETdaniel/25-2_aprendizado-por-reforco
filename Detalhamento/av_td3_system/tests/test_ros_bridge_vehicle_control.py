#!/usr/bin/env python3
"""
Test ROS 2 Bridge Vehicle Control - Phase 2.2

This is THE CRITICAL TEST that native ROS 2 FAILED in our comprehensive testing.

We will:
1. Subscribe to /carla/ego_vehicle/odometry (verify sensor data)
2. Publish to /carla/ego_vehicle/vehicle_control_cmd (TEST CONTROL)
3. Monitor vehicle movement to confirm control works

Expected Result: Vehicle MOVES (unlike native ROS 2 where it didn't)

Date: 2025-11-22
Phase: 2.2 - ROS Bridge Vehicle Control Verification
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import time
import math


class VehicleControlTester(Node):
    """Test vehicle control via ROS 2 Bridge."""

    def __init__(self):
        super().__init__('vehicle_control_tester')

        # Publisher for vehicle control commands
        self.control_publisher = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )

        # Subscriber for odometry (to verify movement)
        self.odometry_subscriber = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )

        # State tracking
        self.initial_position = None
        self.current_position = None
        self.odometry_received = False
        self.test_start_time = None
        self.test_duration = 5.0  # seconds

        self.get_logger().info('Vehicle Control Tester initialized')
        self.get_logger().info('Waiting for odometry data...')

    def odometry_callback(self, msg: Odometry):
        """Store odometry data."""
        self.current_position = msg.pose.pose.position

        if not self.odometry_received:
            self.initial_position = self.current_position
            self.odometry_received = True
            self.get_logger().info(
                f'Initial position: x={self.initial_position.x:.2f}, '
                f'y={self.initial_position.y:.2f}, z={self.initial_position.z:.2f}'
            )

    def calculate_distance_moved(self):
        """Calculate 3D distance from initial position."""
        if self.initial_position is None or self.current_position is None:
            return 0.0

        dx = self.current_position.x - self.initial_position.x
        dy = self.current_position.y - self.initial_position.y
        dz = self.current_position.z - self.initial_position.z

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def send_control_command(self, throttle=0.0, steer=0.0, brake=0.0):
        """Publish vehicle control command."""
        msg = CarlaEgoVehicleControl()
        msg.throttle = throttle
        msg.steer = steer
        msg.brake = brake
        msg.hand_brake = False
        msg.reverse = False
        msg.manual_gear_shift = False
        msg.gear = 1

        self.control_publisher.publish(msg)

    def run_test(self):
        """Run the vehicle control test."""
        # Wait for odometry
        self.get_logger().info('Step 1: Waiting for odometry data (max 10s)...')
        timeout = 10.0
        start = time.time()

        while not self.odometry_received and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.odometry_received:
            self.get_logger().error('❌ FAILED: No odometry received!')
            return False

        self.get_logger().info('✅ Odometry received!')

        # Apply throttle
        self.get_logger().info('')
        self.get_logger().info('Step 2: Applying throttle=0.5 for 5 seconds...')
        self.get_logger().info('⏰ THIS IS THE CRITICAL TEST THAT NATIVE ROS 2 FAILED!')
        self.get_logger().info('')

        self.test_start_time = time.time()

        while (time.time() - self.test_start_time) < self.test_duration:
            # Send control command
            self.send_control_command(throttle=0.5, steer=0.0, brake=0.0)

            # Update odometry
            rclpy.spin_once(self, timeout_sec=0.05)

            # Log progress every second
            elapsed = time.time() - self.test_start_time
            if int(elapsed) != int(elapsed - 0.05):  # New second
                distance = self.calculate_distance_moved()
                self.get_logger().info(
                    f'  t={elapsed:.1f}s: distance moved = {distance:.2f}m'
                )

        # Stop vehicle
        self.get_logger().info('')
        self.get_logger().info('Step 3: Stopping vehicle (brake=1.0)...')
        for _ in range(20):  # 1 second at 20Hz
            self.send_control_command(throttle=0.0, steer=0.0, brake=1.0)
            rclpy.spin_once(self, timeout_sec=0.05)

        # Calculate final distance
        final_distance = self.calculate_distance_moved()

        self.get_logger().info('')
        self.get_logger().info('='*70)
        self.get_logger().info('TEST RESULTS - ROS 2 Bridge Vehicle Control')
        self.get_logger().info('='*70)
        self.get_logger().info(f'Initial position: x={self.initial_position.x:.2f}, '
                              f'y={self.initial_position.y:.2f}, z={self.initial_position.z:.2f}')
        self.get_logger().info(f'Final position:   x={self.current_position.x:.2f}, '
                              f'y={self.current_position.y:.2f}, z={self.current_position.z:.2f}')
        self.get_logger().info(f'Distance moved: {final_distance:.2f} meters')
        self.get_logger().info('')

        # Determine success
        success = final_distance > 0.5  # Should move at least 0.5m

        if success:
            self.get_logger().info('✅ SUCCESS! Vehicle control via ROS 2 Bridge WORKS!')
            self.get_logger().info('   This confirms ROS Bridge is MANDATORY for control.')
            self.get_logger().info('   Native ROS 2 (--ros2 flag) is sensor-only.')
        else:
            self.get_logger().error('❌ FAILED! Vehicle did not move!')
            self.get_logger().error(f'   Expected >0.5m, got {final_distance:.2f}m')

        self.get_logger().info('='*70)

        return success


def main(args=None):
    rclpy.init(args=args)

    tester = VehicleControlTester()

    try:
        success = tester.run_test()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        tester.get_logger().info('Test interrupted by user')
        exit_code = 1
    except Exception as e:
        tester.get_logger().error(f'Test failed with exception: {e}')
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        tester.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == '__main__':
    import sys
    sys.exit(main())

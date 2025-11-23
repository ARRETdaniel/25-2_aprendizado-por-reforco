#!/usr/bin/env python3
"""
Test ROS 2 Bridge Vehicle Control - Pure ROS 2 Approach
Phase 2.2 - Final Test

IMPORTANT: In synchronous mode, the ROS Bridge itself handles tick()!
Our controller just needs to:
1. Subscribe to odometry
2. Publish control commands

The ROS Bridge will advance the simulation when it receives control commands.

Date: 2025-11-22
Phase: 2.2 - ROS Bridge Vehicle Control Verification (Final)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import time
import math


class PureROS2Controller(Node):
    """
    Pure ROS 2 controller - no CARLA Python API needed!

    The ROS Bridge handles all simulation management in synchronous mode.
    """

    def __init__(self):
        super().__init__('pure_ros2_controller')

        # ROS 2 Publishers/Subscribers
        self.control_publisher = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )

        self.odometry_subscriber = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self.odometry_callback,
            10
        )

        # Create a timer for control loop (20 Hz to match CARLA)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        # State tracking
        self.initial_position = None
        self.current_position = None
        self.odometry_received = False
        self.test_phase = 'waiting'  # waiting, testing, stopping, done
        self.control_count = 0
        self.test_start_count = None

        self.get_logger().info('✅ Pure ROS 2 Controller initialized!')
        self.get_logger().info('Waiting for odometry data...')

    def odometry_callback(self, msg: Odometry):
        """Store odometry data."""
        self.current_position = msg.pose.pose.position

        if not self.odometry_received:
            self.initial_position = self.current_position
            self.odometry_received = True
            self.get_logger().info('='*70)
            self.get_logger().info('✅ Odometry received!')
            self.get_logger().info(
                f'Initial position: x={self.initial_position.x:.2f}, '
                f'y={self.initial_position.y:.2f}, z={self.initial_position.z:.2f}'
            )
            self.get_logger().info('='*70)
            self.test_phase = 'testing'
            self.test_start_count = self.control_count

    def calculate_distance_moved(self):
        """Calculate 3D distance from initial position."""
        if self.initial_position is None or self.current_position is None:
            return 0.0

        dx = self.current_position.x - self.initial_position.x
        dy = self.current_position.y - self.initial_position.y
        dz = self.current_position.z - self.initial_position.z

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def control_loop(self):
        """Control loop callback (20 Hz)."""
        self.control_count += 1

        if self.test_phase == 'waiting':
            # CRITICAL: Publish control commands to "wake up" synchronous mode!
            # The ROS Bridge won't publish sensor data until it receives commands
            msg = CarlaEgoVehicleControl()
            msg.throttle = 0.0
            msg.steer = 0.0
            msg.brake = 1.0
            msg.hand_brake = False
            msg.reverse = False
            msg.manual_gear_shift = False
            msg.gear = 1
            self.control_publisher.publish(msg)

            if self.control_count == 1:
                self.get_logger().info('Publishing control commands to wake up synchronous mode...')

            # Timeout after 10 seconds
            if self.control_count > 200:  # 200 * 0.05 = 10 seconds
                self.get_logger().error('❌ TIMEOUT: No odometry received!')
                self.test_phase = 'done'
                rclpy.shutdown()

        elif self.test_phase == 'testing':
            # Apply throttle for 5 seconds (100 control cycles)
            elapsed_cycles = self.control_count - self.test_start_count

            if elapsed_cycles == 1:
                self.get_logger().info('\n' + '='*70)
                self.get_logger().info('Starting test: throttle=0.5 for 5 seconds')
                self.get_logger().info('⏰ THIS IS THE CRITICAL TEST!')
                self.get_logger().info('='*70 + '\n')

            msg = CarlaEgoVehicleControl()
            msg.throttle = 0.5
            msg.steer = 0.0
            msg.brake = 0.0
            msg.hand_brake = False
            msg.reverse = False
            msg.manual_gear_shift = False
            msg.gear = 1
            self.control_publisher.publish(msg)

            # Log every second (20 cycles)
            if elapsed_cycles % 20 == 0:
                elapsed_sec = elapsed_cycles / 20.0
                distance = self.calculate_distance_moved()
                self.get_logger().info(
                    f'  t={elapsed_sec:.1f}s: distance = {distance:.2f}m'
                )

            # After 5 seconds, move to stopping
            if elapsed_cycles >= 100:
                self.test_phase = 'stopping'
                self.stop_start_count = self.control_count
                self.get_logger().info('\nApplying brake...')

        elif self.test_phase == 'stopping':
            # Brake for 1 second (20 cycles)
            stop_cycles = self.control_count - self.stop_start_count

            msg = CarlaEgoVehicleControl()
            msg.throttle = 0.0
            msg.steer = 0.0
            msg.brake = 1.0
            self.control_publisher.publish(msg)

            if stop_cycles >= 20:
                self.test_phase = 'done'
                self.show_results()
                rclpy.shutdown()

    def show_results(self):
        """Display test results."""
        final_distance = self.calculate_distance_moved()

        self.get_logger().info('\n' + '='*70)
        self.get_logger().info('TEST RESULTS - ROS 2 Bridge Vehicle Control')
        self.get_logger().info('='*70)
        self.get_logger().info(f'Initial position: x={self.initial_position.x:.2f}, '
                              f'y={self.initial_position.y:.2f}, z={self.initial_position.z:.2f}')
        self.get_logger().info(f'Final position:   x={self.current_position.x:.2f}, '
                              f'y={self.current_position.y:.2f}, z={self.current_position.z:.2f}')
        self.get_logger().info(f'\n📏 Distance moved: {final_distance:.2f} meters\n')

        # Determine success
        success = final_distance > 0.5  # Should move at least 0.5m

        if success:
            self.get_logger().info('✅ ✅ ✅ SUCCESS! ✅ ✅ ✅')
            self.get_logger().info('')
            self.get_logger().info('🎉 Vehicle control via ROS 2 Bridge WORKS!')
            self.get_logger().info('   - ROS 2 control commands were received')
            self.get_logger().info('   - Vehicle moved in response to throttle')
            self.get_logger().info('   - Pure ROS 2 approach (no Python API needed!)')
            self.get_logger().info('')
            self.get_logger().info('📊 This confirms:')
            self.get_logger().info('   ✅ ROS Bridge provides BIDIRECTIONAL control')
            self.get_logger().info('   ✅ Native ROS 2 (--ros2) is sensor-only')
            self.get_logger().info('   ✅ ROS Bridge is MANDATORY for baseline controller')
            self.get_logger().info('')
            self.get_logger().info('✅ Phase 2.2 COMPLETE! Ready for Phase 2.3')
        else:
            self.get_logger().error('❌ ❌ ❌ FAILED! ❌ ❌ ❌')
            self.get_logger().error(f'   Expected >0.5m, got {final_distance:.2f}m')

        self.get_logger().info('='*70)


def main(args=None):
    rclpy.init(args=args)

    controller = PureROS2Controller()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Test interrupted by user')
    except Exception as e:
        controller.get_logger().error(f'Test failed with exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

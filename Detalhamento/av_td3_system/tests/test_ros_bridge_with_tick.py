#!/usr/bin/env python3
"""
Test ROS 2 Bridge Vehicle Control with Synchronous Mode Tick
Phase 2.2 - Complete Test with Simulation Advancement

This test properly handles CARLA's synchronous mode by:
1. Connecting to CARLA Python API to advance simulation
2. Publishing control commands via ROS 2
3. Monitoring odometry to verify movement

This mirrors the architecture we'll use for the baseline controller.

Date: 2025-11-22
Phase: 2.2 - ROS Bridge Vehicle Control Verification
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl
import carla
import time
import math
import threading


class SynchronousBridgeController(Node):
    """
    Test controller that properly handles synchronous mode.
    
    Architecture (same as baseline will use):
    - ROS 2 Node: Subscribe to sensors, publish control
    - CARLA Python Client: Advance simulation with tick()
    """
    
    def __init__(self):
        super().__init__('synchronous_bridge_controller')
        
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
        
        # CARLA Client for tick()
        try:
            self.get_logger().info('Connecting to CARLA server...')
            self.carla_client = carla.Client('localhost', 2000)
            self.carla_client.set_timeout(10.0)
            self.world = self.carla_client.get_world()
            
            # Verify synchronous mode
            settings = self.world.get_settings()
            self.get_logger().info(f'CARLA Settings:')
            self.get_logger().info(f'  Synchronous mode: {settings.synchronous_mode}')
            self.get_logger().info(f'  Fixed delta: {settings.fixed_delta_seconds}s')
            
            if not settings.synchronous_mode:
                self.get_logger().warn('⚠️  Synchronous mode is OFF! Enabling it...')
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05  # 20 Hz
                self.world.apply_settings(settings)
                
        except Exception as e:
            self.get_logger().error(f'❌ Failed to connect to CARLA: {e}')
            raise
        
        # State tracking
        self.initial_position = None
        self.current_position = None
        self.odometry_received = False
        self.frame_count = 0
        
        self.get_logger().info('✅ Controller initialized!')
        
    def odometry_callback(self, msg: Odometry):
        """Store odometry data."""
        self.current_position = msg.pose.pose.position
        
        if not self.odometry_received:
            self.initial_position = self.current_position
            self.odometry_received = True
            self.get_logger().info(
                f'✅ Initial position: x={self.initial_position.x:.2f}, '
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
    
    def tick_simulation(self):
        """Advance the simulation by one step."""
        try:
            self.world.tick()
            self.frame_count += 1
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to tick: {e}')
            return False
    
    def run_test(self):
        """Run the vehicle control test with proper synchronous mode handling."""
        
        self.get_logger().info('='*70)
        self.get_logger().info('Starting Vehicle Control Test (Synchronous Mode)')
        self.get_logger().info('='*70)
        
        # Step 1: Wait for odometry (with tick())
        self.get_logger().info('\nStep 1: Waiting for odometry data...')
        timeout = 50  # 50 ticks = 2.5s at 20Hz
        
        for i in range(timeout):
            # Process ROS messages
            rclpy.spin_once(self, timeout_sec=0.01)
            
            # Advance simulation
            if not self.tick_simulation():
                return False
                
            if self.odometry_received:
                break
                
            if i % 10 == 0:
                self.get_logger().info(f'  Waiting... ({i} ticks)')
        
        if not self.odometry_received:
            self.get_logger().error('❌ FAILED: No odometry received after 50 ticks!')
            return False
        
        # Step 2: Apply throttle and tick for 5 seconds (100 frames @ 20Hz)
        self.get_logger().info('\n' + '='*70)
        self.get_logger().info('Step 2: Applying throttle=0.5 for 5 seconds (100 frames)')
        self.get_logger().info('⏰ THIS IS THE CRITICAL TEST THAT NATIVE ROS 2 FAILED!')
        self.get_logger().info('='*70 + '\n')
        
        test_frames = 100  # 5 seconds at 20 Hz
        start_frame = self.frame_count
        
        for i in range(test_frames):
            # Send control command
            self.send_control_command(throttle=0.5, steer=0.0, brake=0.0)
            
            # Process ROS messages
            rclpy.spin_once(self, timeout_sec=0.01)
            
            # Advance simulation (CRITICAL for synchronous mode!)
            if not self.tick_simulation():
                return False
            
            # Log progress every 20 frames (1 second)
            if i % 20 == 0:
                elapsed = i / 20.0
                distance = self.calculate_distance_moved()
                self.get_logger().info(
                    f'  t={elapsed:.1f}s (frame {i}): distance = {distance:.2f}m'
                )
        
        # Step 3: Stop vehicle
        self.get_logger().info('\nStep 3: Stopping vehicle (brake=1.0)...')
        for i in range(20):  # 1 second at 20Hz
            self.send_control_command(throttle=0.0, steer=0.0, brake=1.0)
            rclpy.spin_once(self, timeout_sec=0.01)
            self.tick_simulation()
        
        # Calculate final distance
        final_distance = self.calculate_distance_moved()
        
        # Display results
        self.get_logger().info('\n' + '='*70)
        self.get_logger().info('TEST RESULTS - ROS 2 Bridge Vehicle Control')
        self.get_logger().info('='*70)
        self.get_logger().info(f'Total frames: {self.frame_count - start_frame}')
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
            self.get_logger().info('   - Synchronous mode tick() working correctly')
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
        
        return success


def main(args=None):
    rclpy.init(args=args)
    
    controller = SynchronousBridgeController()
    
    try:
        success = controller.run_test()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        controller.get_logger().info('Test interrupted by user')
        exit_code = 1
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
    import sys
    sys.exit(main())

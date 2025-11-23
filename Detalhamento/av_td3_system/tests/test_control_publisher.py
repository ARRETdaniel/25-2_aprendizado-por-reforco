#!/usr/bin/env python3
"""
Test CARLA native ROS 2 vehicle control by publishing commands via Python.

This script creates a simple ROS 2 publisher node to send control commands
to the CARLA vehicle and observe if it responds.

We'll test different possible topic formats:
1. /carla/ego/vehicle_control_cmd
2. /carla//ego/vehicle_control_cmd  
3. /carla/ego/control
4. Check what subscriber topics CARLA actually creates

Author: Generated for Phase 2.2 - Vehicle Control Testing
Date: November 22, 2025
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SimpleVehicleController(Node):
    """
    Simple ROS 2 node to test vehicle control.
    
    We'll use geometry_msgs/Twist first (standard ROS message)
    to see if CARLA responds. If not, we'll need custom CARLA messages.
    """
    
    def __init__(self):
        super().__init__('simple_vehicle_controller')
        
        # Try different possible topic names
        self.topic_variations = [
            '/carla/ego/cmd_vel',  # Standard ROS convention
            '/carla/ego/vehicle_control_cmd',  # Expected from docs
            '/carla//ego/cmd_vel',  # Double slash like sensors
            '/cmd_vel',  # Generic
        ]
        
        # Create publishers for all variations
        self.publishers = {}
        for topic in self.topic_variations:
            pub = self.create_publisher(Twist, topic, 10)
            self.publishers[topic] = pub
            self.get_logger().info(f'Created publisher for: {topic}')
        
        # Create timer for sending control commands
        self.timer = self.create_timer(0.5, self.send_control_command)
        self.command_count = 0
        
    def send_control_command(self):
        """
        Send a simple forward command.
        
        Twist message:
          linear.x = forward velocity (m/s)
          angular.z = turning rate (rad/s)
        """
        msg = Twist()
        msg.linear.x = 5.0  # 5 m/s forward (~18 km/h)
        msg.angular.z = 0.0  # No turning
        
        # Publish to all topic variations
        for topic, pub in self.publishers.items():
            pub.publish(msg)
        
        self.command_count += 1
        
        if self.command_count % 4 == 0:  # Log every 2 seconds
            self.get_logger().info(f'Published {self.command_count} control commands (linear.x=5.0)')

def main(args=None):
    """Main function."""
    print("=" * 80)
    print("CARLA Native ROS 2 - Vehicle Control Publisher Test")
    print("=" * 80)
    print()
    print("This node will publish Twist messages to test vehicle control.")
    print("If the vehicle moves, we've found the correct topic!")
    print()
    print("Expected behavior:")
    print("  - Vehicle should move forward at ~18 km/h")
    print("  - No turning (straight line)")
    print()
    print("To monitor vehicle position in CARLA:")
    print("  1. Open another terminal")
    print("  2. Connect to CARLA Python API")
    print("  3. Get vehicle transform and print periodically")
    print()
    print("=" * 80)
    print()
    
    rclpy.init(args=args)
    
    try:
        controller = SimpleVehicleController()
        print("✅ Controller node started")
        print("📤 Publishing control commands every 0.5 seconds...")
        print("⏳ Running for 30 seconds (press Ctrl+C to stop)...")
        print()
        
        # Run for 30 seconds
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time < 30):
            rclpy.spin_once(controller, timeout_sec=0.1)
        
        print()
        print("=" * 80)
        print(f"Published {controller.command_count} total commands")
        print("Check CARLA simulation to see if vehicle moved!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

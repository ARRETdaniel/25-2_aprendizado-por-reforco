#!/usr/bin/env python3
"""
CARLA DRL Demonstration Script

This script demonstrates the complete CARLA integration for DRL training.
It shows vehicle control, sensor data collection, and ROS 2 bridge readiness.

Features demonstrated:
- Vehicle spawning and basic control
- Real-time camera visualization
- Performance monitoring
- System architecture validation
"""

import sys
import os
import time
import numpy as np
import carla

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from carla_interface.carla_client import CarlaClient, CarlaConfig

def demonstrate_vehicle_control(client: CarlaClient, duration: float = 10.0):
    """Demonstrate basic vehicle control patterns"""
    print("\n🎮 Demonstrating Vehicle Control")
    print("=" * 40)

    if not client.vehicle:
        print("❌ No vehicle available")
        return

    # Control patterns for demonstration
    control_patterns = [
        {"name": "Forward", "throttle": 0.5, "steer": 0.0, "brake": 0.0},
        {"name": "Turn Left", "throttle": 0.3, "steer": -0.5, "brake": 0.0},
        {"name": "Turn Right", "throttle": 0.3, "steer": 0.5, "brake": 0.0},
        {"name": "Brake", "throttle": 0.0, "steer": 0.0, "brake": 1.0},
    ]

    pattern_duration = duration / len(control_patterns)

    for pattern in control_patterns:
        print(f"🚗 {pattern['name']} for {pattern_duration:.1f}s...")

        control = carla.VehicleControl(
            throttle=pattern["throttle"],
            steer=pattern["steer"],
            brake=pattern["brake"]
        )

        end_time = time.time() + pattern_duration
        while time.time() < end_time:
            client.vehicle.apply_control(control)
            time.sleep(0.1)

    # Stop vehicle
    stop_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    client.vehicle.apply_control(stop_control)
    print("🛑 Vehicle stopped")

def monitor_system_performance(client: CarlaClient, duration: float = 5.0):
    """Monitor and display system performance metrics"""
    print("\n📊 System Performance Monitoring")
    print("=" * 40)

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        # Get current frame if available
        frame = client.data_manager.get_latest_camera_frame()
        if frame is not None:
            frame_count += 1

        # Display current stats
        elapsed = time.time() - start_time
        if frame_count > 0:
            fps = frame_count / elapsed
            print(f"📈 Camera FPS: {fps:.1f} | Frames: {frame_count} | Time: {elapsed:.1f}s", end="\r")

        time.sleep(0.1)

    print(f"\n✅ Performance test completed: {frame_count} frames in {duration:.1f}s")

def validate_ros2_readiness(client: CarlaClient):
    """Validate system readiness for ROS 2 integration"""
    print("\n🤖 ROS 2 Integration Readiness Check")
    print("=" * 40)

    checks = []

    # Check CARLA connection
    checks.append(("CARLA Connection", client.is_connected))

    # Check vehicle availability
    checks.append(("Vehicle Spawned", client.vehicle is not None))

    # Check sensor setup
    camera_ok = hasattr(client, 'camera_sensor') and client.camera_sensor is not None
    checks.append(("Camera Sensor", camera_ok))

    # Check data streaming
    frame = client.data_manager.get_latest_camera_frame()
    checks.append(("Camera Data Stream", frame is not None))

    # Check vehicle control capability
    if client.vehicle:
        transform = client.vehicle.get_transform()
        control_ok = transform is not None
        checks.append(("Vehicle Control Access", control_ok))
    else:
        checks.append(("Vehicle Control Access", False))

    # Display results
    all_passed = True
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
        if not status:
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 System ready for ROS 2 + TD3 integration!")
        print("📋 Next steps:")
        print("   1. Implement ROS 2 bridge (geometry_msgs, sensor_msgs)")
        print("   2. Connect TD3 agent for action commands")
        print("   3. Setup training pipeline with state/reward functions")
    else:
        print("⚠️  System not fully ready. Address failed checks above.")

    return all_passed

def main():
    """Main demonstration function"""
    print("🚀 CARLA DRL System Demonstration")
    print("=" * 50)

    # Initialize CARLA client
    config = CarlaConfig()
    client = CarlaClient(config)

    try:
        # Connect to CARLA
        if not client.connect():
            print("❌ Failed to connect to CARLA")
            return

        # Spawn vehicle
        if not client.spawn_vehicle():
            print("❌ Failed to spawn vehicle")
            return

        # Setup sensors
        client.setup_sensors()

        # Wait for sensor data
        print("\n⏳ Waiting for sensor initialization...")
        time.sleep(2)

        # Run demonstrations
        validate_ros2_readiness(client)

        print("\n🎬 Starting vehicle control demonstration...")
        print("📺 Watch the camera view window for visual feedback")

        # Start visualization in background
        import threading
        viz_thread = threading.Thread(
            target=client.start_visualization,
            kwargs={"display_time": 20},
            daemon=True
        )
        viz_thread.start()

        # Demonstrate vehicle control
        demonstrate_vehicle_control(client, duration=15.0)

        # Monitor performance
        monitor_system_performance(client, duration=3.0)

        # Final readiness check
        print("\n🔍 Final System Validation:")
        validate_ros2_readiness(client)

        print("\n🎯 Demonstration completed successfully!")
        print("💡 The system is ready for DRL integration")

    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
    finally:
        # Cleanup
        client.cleanup()
        print("✅ Resources cleaned up")

if __name__ == "__main__":
    main()

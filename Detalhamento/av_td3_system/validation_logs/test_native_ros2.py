#!/usr/bin/env python3
"""
Test script to verify CARLA native ROS 2 support.

According to CARLA documentation, when CARLA is launched with --ros2 flag,
setting the 'ros_name' attribute on actors should automatically create
ROS 2 publishers and subscribers for that actor.

Expected behavior:
- Vehicle control subscriber: /carla/{ros_name}/vehicle_control_cmd
- Sensor publishers: /carla/{ros_name}/{sensor_name}/{type}
- Clock publisher: /carla/clock

This script will:
1. Connect to CARLA server (should be running with --ros2 flag)
2. Spawn a vehicle with ros_name='test_vehicle'
3. Attach a camera sensor with ros_name='front_camera'
4. Print success and expected topics
5. Keep vehicle alive for 30 seconds to allow external topic inspection

Author: Generated for Phase 2.2 ROS 2 verification
Date: 2025-01-XX
"""

import carla
import time
import sys

def main():
    print("=" * 80)
    print("CARLA Native ROS 2 Verification Test")
    print("=" * 80)

    # Connect to CARLA
    print("\n[1/5] Connecting to CARLA server...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        version = client.get_server_version()
        print(f"✅ Connected to CARLA {version}")
    except Exception as e:
        print(f"❌ Failed to connect to CARLA: {e}")
        print("\nMake sure CARLA is running with:")
        print("  docker run -d --name carla-server --runtime=nvidia \\")
        print("    --net=host carlasim/carla:0.9.16 \\")
        print("    bash CarlaUE4.sh --ros2 -RenderOffScreen -nosound")
        sys.exit(1)

    # Get world
    world = client.get_world()
    bp_library = world.get_blueprint_library()

    # Spawn vehicle with ros_name attribute
    print("\n[2/5] Spawning vehicle with ros_name='test_vehicle'...")
    try:
        vehicle_bp = bp_library.filter('vehicle.lincoln.mkz_2020')[0]

        # CRITICAL: Set ros_name attribute for native ROS 2
        if vehicle_bp.has_attribute('ros_name'):
            vehicle_bp.set_attribute('ros_name', 'test_vehicle')
            print("✅ Set ros_name='test_vehicle' on vehicle blueprint")
        else:
            print("❌ WARNING: 'ros_name' attribute not available!")
            print("   This suggests native ROS 2 support is NOT compiled in.")
            print("   The --ros2 flag may be silently ignored.")

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✅ Spawned vehicle ID: {vehicle.id}")

    except Exception as e:
        print(f"❌ Failed to spawn vehicle: {e}")
        sys.exit(1)

    # Attach camera sensor with ros_name
    print("\n[3/5] Attaching camera sensor with ros_name='front_camera'...")
    try:
        camera_bp = bp_library.find('sensor.camera.rgb')

        # Set ros_name for sensor
        if camera_bp.has_attribute('ros_name'):
            camera_bp.set_attribute('ros_name', 'front_camera')
            print("✅ Set ros_name='front_camera' on camera blueprint")
        else:
            print("❌ WARNING: 'ros_name' attribute not available on sensor!")

        # Set image resolution
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')

        # Attach to vehicle
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print(f"✅ Attached camera ID: {camera.id}")

        # CRITICAL: Enable ROS 2 publisher for this sensor
        try:
            camera.enable_for_ros()
            print("✅ Enabled ROS 2 publisher for camera (enable_for_ros() succeeded)")
        except AttributeError:
            print("❌ WARNING: enable_for_ros() method not available!")
            print("   Native ROS 2 support may not be fully compiled.")

    except Exception as e:
        print(f"❌ Failed to attach camera: {e}")
        vehicle.destroy()
        sys.exit(1)

    # Print expected ROS 2 topics
    print("\n[4/5] If native ROS 2 is working, these topics should exist:")
    print("-" * 80)
    print("Control subscriber:")
    print("  • /carla/test_vehicle/vehicle_control_cmd")
    print("    Type: carla_msgs/msg/CarlaEgoVehicleControl")
    print()
    print("Sensor publishers:")
    print("  • /carla/test_vehicle/front_camera/image")
    print("    Type: sensor_msgs/msg/Image")
    print()
    print("Clock publisher:")
    print("  • /carla/clock")
    print("    Type: rosgraph_msgs/msg/Clock")
    print("-" * 80)

    # Check for ROS 2 topics (if ros2 CLI is available)
    print("\n[5/5] Verification methods:")
    print()
    print("Method 1: Using ros2 CLI (if installed)")
    print("  $ ros2 topic list")
    print("  $ ros2 topic echo /carla/test_vehicle/vehicle_control_cmd")
    print()
    print("Method 2: Using FastDDS discovery")
    print("  $ docker exec carla-server lsof -i :7400-7500")
    print()
    print("Method 3: Check CARLA logs for ROS 2 initialization")
    print("  $ docker logs carla-server 2>&1 | grep -i ros")
    print()

    # Keep actors alive
    print("\n" + "=" * 80)
    print("Test vehicle will stay active for 30 seconds...")
    print("Press Ctrl+C to cleanup and exit early")
    print("=" * 80)

    try:
        for remaining in range(30, 0, -1):
            print(f"\rTime remaining: {remaining}s ", end='', flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Cleanup
    print("\n\nCleaning up...")
    camera.destroy()
    vehicle.destroy()
    print("✅ Cleanup complete")

    print("\n" + "=" * 80)
    print("VERIFICATION RESULT:")
    print("-" * 80)
    if vehicle_bp.has_attribute('ros_name'):
        print("✅ ros_name attribute EXISTS on blueprints")
        print("   → Native ROS 2 support is likely compiled in")
        print("   → Check external tools to verify topic publication")
    else:
        print("❌ ros_name attribute DOES NOT EXIST")
        print("   → Native ROS 2 support is NOT compiled in")
        print("   → The --ros2 flag is being silently ignored")
        print("   → Recommendation: Use external ROS bridge instead")
    print("=" * 80)

if __name__ == '__main__':
    main()

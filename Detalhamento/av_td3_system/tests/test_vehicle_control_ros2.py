#!/usr/bin/env python3
"""
Test script to verify CARLA native ROS 2 vehicle control.

This script tests if we can control a CARLA vehicle via ROS 2 topics using
the native ROS 2 interface (--ros2 flag).

Based on CARLA 0.9.16 official example: /workspace/PythonAPI/examples/ros2/ros2_native.py

Test procedure:
1. Spawn a vehicle with ros_name='ego'
2. Spawn a camera sensor with ros_name='front_camera' and enable ROS 2
3. Try to control the vehicle using ROS 2 topics from a separate ROS 2 node
4. Verify vehicle responds to control commands

Expected ROS 2 topics:
- Control subscriber: /carla/ego/vehicle_control_cmd (or similar)
- Camera publisher: /carla//front_camera/image (confirmed working)
- Odometry publisher: /carla/ego/odometry (to be tested)

Author: Generated for Phase 2.2 - Vehicle Control Testing
Date: November 22, 2025
"""

import carla
import time
import sys
import argparse

def spawn_ego_vehicle(world, spawn_point_index=0):
    """
    Spawn ego vehicle with ROS 2 configuration.
    
    Args:
        world: CARLA world object
        spawn_point_index: Index of spawn point to use
        
    Returns:
        tuple: (vehicle_actor, vehicle_id, ros_name)
    """
    print("\n[2/6] Spawning ego vehicle with ROS 2 configuration...")
    
    bp_library = world.get_blueprint_library()
    
    # Get vehicle blueprint (Lincoln MKZ 2020 - same as TD3 agent)
    vehicle_bp = bp_library.filter('vehicle.lincoln.mkz_2020')[0]
    
    # Set ROS 2 attributes (critical for native ROS 2)
    ros_name = 'ego'
    vehicle_bp.set_attribute('role_name', ros_name)
    vehicle_bp.set_attribute('ros_name', ros_name)
    
    print(f"   Vehicle blueprint: {vehicle_bp.id}")
    print(f"   ROS name: {ros_name}")
    print(f"   Role name: {ros_name}")
    
    # Get spawn point
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("   ❌ ERROR: No spawn points available on map!")
        return None, None, None
        
    spawn_point = spawn_points[spawn_point_index % len(spawn_points)]
    print(f"   Spawn point: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})")
    
    # Spawn vehicle
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"   ✅ Vehicle spawned successfully (ID: {vehicle.id})")
        return vehicle, vehicle.id, ros_name
    except Exception as e:
        print(f"   ❌ Failed to spawn vehicle: {e}")
        return None, None, None

def attach_camera_sensor(world, vehicle, ros_name='front_camera'):
    """
    Attach camera sensor to vehicle with ROS 2 enabled.
    
    Args:
        world: CARLA world object
        vehicle: Vehicle actor to attach sensor to
        ros_name: ROS name for the sensor
        
    Returns:
        sensor_actor or None
    """
    print("\n[3/6] Attaching camera sensor with ROS 2 enabled...")
    
    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find('sensor.camera.rgb')
    
    # Set ROS 2 attributes
    camera_bp.set_attribute('ros_name', ros_name)
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    
    print(f"   Camera ROS name: {ros_name}")
    print(f"   Resolution: 800x600")
    
    # Attach to vehicle
    camera_transform = carla.Transform(
        carla.Location(x=2.5, z=0.7),  # Front bumper, eye level
        carla.Rotation(pitch=0.0)
    )
    
    try:
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print(f"   ✅ Camera attached (ID: {camera.id})")
        
        # CRITICAL: Enable ROS 2 publisher
        camera.enable_for_ros()
        print("   ✅ ROS 2 publisher enabled (enable_for_ros() called)")
        
        return camera
    except Exception as e:
        print(f"   ❌ Failed to attach camera: {e}")
        return None

def configure_synchronous_mode(world, delta_seconds=0.05):
    """
    Configure CARLA to run in synchronous mode for deterministic testing.
    
    Args:
        world: CARLA world object
        delta_seconds: Fixed time step (0.05 = 20 Hz)
        
    Returns:
        original_settings for restoration
    """
    print("\n[4/6] Configuring synchronous mode...")
    
    original_settings = world.get_settings()
    settings = world.get_settings()
    
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = delta_seconds
    
    world.apply_settings(settings)
    
    print(f"   ✅ Synchronous mode: {settings.synchronous_mode}")
    print(f"   ✅ Fixed delta seconds: {settings.fixed_delta_seconds} ({1.0/delta_seconds:.0f} Hz)")
    
    return original_settings

def print_expected_topics(ros_name):
    """Print expected ROS 2 topics for verification."""
    print("\n[5/6] Expected ROS 2 Topics:")
    print("=" * 80)
    print("\n📤 PUBLISHERS (CARLA → ROS 2):")
    print(f"   • /carla//{ros_name}_camera/image")
    print(f"     Type: sensor_msgs/msg/Image")
    print(f"     Description: Front camera RGB image")
    print()
    print(f"   • /carla//{ros_name}_camera/camera_info")
    print(f"     Type: sensor_msgs/msg/CameraInfo")
    print(f"     Description: Camera calibration info")
    print()
    print("   • /clock")
    print("     Type: rosgraph_msgs/msg/Clock")
    print("     Description: Simulation time")
    print()
    print("   • /tf")
    print("     Type: tf2_msgs/msg/TFMessage")
    print("     Description: Transform tree")
    print()
    
    print("📥 SUBSCRIBERS (ROS 2 → CARLA):")
    print(f"   • /carla/{ros_name}/vehicle_control_cmd (?)")
    print("     Type: carla_msgs/msg/CarlaEgoVehicleControl")
    print("     Description: Vehicle throttle/steering/brake commands")
    print("     ⚠️  TOPIC FORMAT TO BE VERIFIED")
    print()
    print("=" * 80)

def verify_topics_with_ros2(duration_seconds=10):
    """
    Instructions for verifying topics using ROS 2 CLI tools.
    
    Args:
        duration_seconds: How long to keep vehicle alive for testing
    """
    print("\n[6/6] Topic Verification Instructions:")
    print("=" * 80)
    print()
    print("While this script is running, open a NEW terminal and run:")
    print()
    print("1. List all ROS 2 topics:")
    print("   $ docker run --rm --net=host ros:humble-ros-core ros2 topic list")
    print()
    print("2. Check topic details:")
    print("   $ docker run --rm --net=host ros:humble-ros-core ros2 topic info /carla//front_camera/image")
    print()
    print("3. Monitor image topic (verify publishing):")
    print("   $ docker run --rm --net=host ros:humble-ros-core ros2 topic hz /carla//front_camera/image")
    print()
    print("4. Test vehicle control (send throttle command):")
    print("   $ docker run --rm --net=host ros:humble-ros-core ros2 topic pub --once \\")
    print("       /carla/ego/vehicle_control_cmd carla_msgs/msg/CarlaEgoVehicleControl \\")
    print("       '{throttle: 0.3, steer: 0.0, brake: 0.0, hand_brake: false, reverse: false, manual_gear_shift: false, gear: 1}'")
    print()
    print("   ⚠️  Note: This might fail if carla_msgs are not installed in ROS container.")
    print("   ⚠️  We'll test control via Python script instead.")
    print()
    print("=" * 80)
    print(f"\n⏳ Keeping vehicle alive for {duration_seconds} seconds for manual testing...")
    print("   Press Ctrl+C to cleanup and exit early\n")

def main(args):
    """Main test procedure."""
    
    print("=" * 80)
    print("CARLA Native ROS 2 - Vehicle Control Test")
    print("=" * 80)
    
    world = None
    vehicle = None
    camera = None
    original_settings = None
    
    try:
        # Connect to CARLA
        print("\n[1/6] Connecting to CARLA server...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        try:
            version = client.get_server_version()
            print(f"   ✅ Connected to CARLA {version}")
        except Exception as e:
            print(f"   ❌ Failed to connect: {e}")
            print("\n   Make sure CARLA is running with:")
            print("   docker run -d --name carla-server --runtime=nvidia \\")
            print("     --net=host --env=NVIDIA_VISIBLE_DEVICES=all \\")
            print("     carlasim/carla:0.9.16 bash CarlaUE4.sh --ros2 -RenderOffScreen -nosound")
            sys.exit(1)
        
        world = client.get_world()
        
        # Spawn vehicle
        vehicle, vehicle_id, ros_name = spawn_ego_vehicle(world, args.spawn_point)
        if not vehicle:
            sys.exit(1)
        
        # Attach camera
        camera = attach_camera_sensor(world, vehicle, 'front_camera')
        if not camera:
            vehicle.destroy()
            sys.exit(1)
        
        # Configure synchronous mode
        original_settings = configure_synchronous_mode(world, args.delta_seconds)
        
        # Tick world to activate sensors
        print("\n   Ticking world to activate ROS 2 publishers...")
        world.tick()
        print("   ✅ World tick complete")
        
        # Print expected topics
        print_expected_topics(ros_name)
        
        # Verify topics
        verify_topics_with_ros2(args.duration)
        
        # Keep vehicle alive for testing
        try:
            for remaining in range(args.duration, 0, -1):
                print(f"\r   Time remaining: {remaining}s ", end='', flush=True)
                time.sleep(1)
                
                # Tick world in synchronous mode
                if args.synchronous:
                    world.tick()
                    
        except KeyboardInterrupt:
            print("\n\n   Interrupted by user")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n\nCleaning up...")
        
        if camera:
            camera.destroy()
            print("   ✅ Camera destroyed")
            
        if vehicle:
            vehicle.destroy()
            print("   ✅ Vehicle destroyed")
            
        if original_settings and world:
            world.apply_settings(original_settings)
            print("   ✅ World settings restored")
        
        print("\n" + "=" * 80)
        print("Test complete!")
        print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Native ROS 2 Vehicle Control Test')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--spawn-point', type=int, default=0, help='Spawn point index')
    parser.add_argument('--duration', type=int, default=30, help='Duration to keep vehicle alive (seconds)')
    parser.add_argument('--delta-seconds', type=float, default=0.05, help='Fixed delta seconds for synchronous mode')
    parser.add_argument('--synchronous', action='store_true', default=True, help='Use synchronous mode')
    
    args = parser.parse_args()
    main(args)

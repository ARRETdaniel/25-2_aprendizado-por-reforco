#!/usr/bin/env python3
"""
Critical test: Does vehicle.enable_for_ros() exist?

This test will determine if CARLA's native ROS 2 supports vehicle control
by checking if the vehicle actor has an enable_for_ros() method like sensors do.

Result will determine our architecture:
- YES → Pure ROS 2 architecture (2 containers)
- NO → Hybrid architecture (ROS 2 sensors + Python API control)

Author: Generated for Phase 2.2 - Architecture Decision
Date: November 22, 2025
"""

import carla
import sys
import time

def main():
    print("=" * 80)
    print("CRITICAL TEST: vehicle.enable_for_ros() Existence Check")
    print("=" * 80)
    print()
    
    try:
        # Connect to CARLA
        print("[1/3] Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        version = client.get_server_version()
        print(f"   ✅ Connected to CARLA {version}")
        
        world = client.get_world()
        bp_library = world.get_blueprint_library()
        
        # Spawn vehicle
        print("\n[2/3] Spawning test vehicle...")
        vehicle_bp = bp_library.filter('vehicle.lincoln.mkz_2020')[0]
        vehicle_bp.set_attribute('role_name', 'test')
        vehicle_bp.set_attribute('ros_name', 'test')
        
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"   ✅ Vehicle spawned (ID: {vehicle.id})")
        
        # CRITICAL TEST: Check for enable_for_ros() method
        print("\n[3/3] Testing vehicle.enable_for_ros()...")
        print("   Checking if method exists...")
        
        has_method = hasattr(vehicle, 'enable_for_ros')
        is_callable = callable(getattr(vehicle, 'enable_for_ros', None))
        
        print(f"   hasattr(vehicle, 'enable_for_ros'): {has_method}")
        print(f"   callable: {is_callable}")
        
        if has_method and is_callable:
            print("\n   ✅ Method exists! Attempting to call...")
            try:
                vehicle.enable_for_ros()
                print("   ✅✅✅ SUCCESS! vehicle.enable_for_ros() WORKS!")
                print()
                print("=" * 80)
                print("RESULT: NATIVE ROS 2 VEHICLE CONTROL IS AVAILABLE")
                print("=" * 80)
                print()
                print("Next steps:")
                print("1. Wait 5 seconds and check ROS 2 topics")
                print("2. Look for vehicle control subscriber topic")
                print("3. Test sending control commands via ROS 2")
                print()
                print("Expected new topics:")
                print("   • /carla/test/vehicle_control_cmd")
                print("   • /carla/test/odometry") 
                print("   • /carla/test/vehicle_status")
                print()
                
                # Wait and let user check topics
                print("Waiting 10 seconds for topic discovery...")
                print("Run this in another terminal:")
                print("  $ docker run --rm --net=host ros:humble-ros-core ros2 topic list | grep -i test")
                print()
                
                world.tick()  # Activate ROS 2
                time.sleep(10)
                
                result_code = 0  # Success
                
            except Exception as e:
                print(f"   ❌ Method exists but failed to execute: {e}")
                result_code = 2
        else:
            print("\n   ❌ Method DOES NOT exist on vehicle actor")
            print()
            print("=" * 80)
            print("RESULT: NATIVE ROS 2 VEHICLE CONTROL NOT AVAILABLE")
            print("=" * 80)
            print()
            print("CONCLUSION:")
            print("CARLA 0.9.16 native ROS 2 supports:")
            print("   ✅ Sensor data publishing (cameras, GNSS, IMU, etc.)")
            print("   ❌ Vehicle control commands (no ROS 2 subscriber)")
            print()
            print("RECOMMENDED ARCHITECTURE:")
            print("   → Hybrid approach (ROS 2 sensors + Python API control)")
            print("   → Sensor data: Native ROS 2 publishers")
            print("   → Vehicle control: Direct Python API (carla.VehicleControl)")
            print()
            result_code = 1  # Method not found
        
        # Cleanup
        vehicle.destroy()
        print("\nCleanup complete")
        
        return result_code
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 3  # Error

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

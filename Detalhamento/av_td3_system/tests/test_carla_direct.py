#!/usr/bin/env python3
"""
Quick test script to verify CARLA server is running and accepting connections.
This bypasses ROS Bridge completely to isolate infrastructure issues.

Date: 2025-01-24
"""

import sys
import time

try:
    import carla
    print("✓ CARLA Python API imported successfully")
except ImportError as e:
    print(f"✗ CARLA Python API not found: {e}")
    print("Install with: pip install carla==0.9.16")
    sys.exit(1)

def test_carla_connection(host='localhost', port=2000, timeout=10.0):
    """Test connection to CARLA server"""
    print(f"\n{'='*60}")
    print(f"Testing CARLA Connection: {host}:{port}")
    print(f"{'='*60}\n")
    
    try:
        # Create client
        print(f"[1/5] Creating CARLA client...")
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        print("    ✓ Client created")
        
        # Get server version
        print(f"\n[2/5] Getting server version...")
        version = client.get_server_version()
        print(f"    ✓ Server version: {version}")
        
        # Get world
        print(f"\n[3/5] Getting world...")
        world = client.get_world()
        world_name = world.get_map().name
        print(f"    ✓ World loaded: {world_name}")
        
        # Get actors
        print(f"\n[4/5] Checking actors...")
        actors = world.get_actors()
        print(f"    ✓ Total actors: {len(actors)}")
        
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')
        print(f"    - Vehicles: {len(vehicles)}")
        print(f"    - Pedestrians: {len(pedestrians)}")
        
        # Spawn test vehicle
        print(f"\n[5/5] Spawning test vehicle...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        if len(spawn_points) > 0:
            spawn_point = spawn_points[0]
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"    ✓ Test vehicle spawned (ID: {vehicle.id})")
            
            # Wait a moment
            time.sleep(1)
            
            # Get vehicle location
            location = vehicle.get_location()
            print(f"    - Location: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")
            
            # Apply control
            print(f"\n[TEST] Applying throttle control...")
            control = carla.VehicleControl(throttle=0.3, steer=0.0)
            vehicle.apply_control(control)
            print(f"    ✓ Control applied successfully")
            
            time.sleep(2)
            
            # Cleanup
            print(f"\n[CLEANUP] Destroying test vehicle...")
            vehicle.destroy()
            print(f"    ✓ Vehicle destroyed")
        else:
            print(f"    ✗ No spawn points available")
            return False
        
        print(f"\n{'='*60}")
        print(f"✓ ALL TESTS PASSED - CARLA is running correctly!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check if CARLA server is running:")
        print(f"   docker ps | grep carla")
        print(f"2. Check if port {port} is listening:")
        print(f"   netstat -tuln | grep :{port}")
        print(f"3. Check CARLA logs:")
        print(f"   docker logs carla-server")
        return False

if __name__ == "__main__":
    success = test_carla_connection()
    sys.exit(0 if success else 1)

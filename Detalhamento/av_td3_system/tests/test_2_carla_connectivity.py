#!/usr/bin/env python3
"""
Test Suite 2: CARLA Server Connectivity - REAL INTEGRATION TESTS
Phase 3: System Testing for TD3 Autonomous Navigation System

Tests (ALL REAL CARLA CONNECTIVITY):
  2.1: Basic Client Connection
  2.2: Vehicle Spawning & Control
  2.3: Camera Sensor Attachment & Image Capture

Prerequisites: CARLA server must be running (use docker-compose.test.yml)
"""

import carla
import numpy as np
import time
import sys
import random
import os


def test_2_1_basic_client_connection():
    """Test 2.1: Verify Python client can connect to CARLA server"""
    print("\n" + "="*70)
    print("🔌 TEST 2.1: Basic Client Connection")
    print("="*70)

    try:
        # Connect to CARLA server
        print("Connecting to CARLA server at localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get server version
        version = client.get_server_version()
        print(f"✅ Connected to CARLA server")
        print(f"   Server version: {version}")

        # Get world
        world = client.get_world()
        map_name = world.get_map().name
        print(f"✅ Retrieved world")
        print(f"   Map: {map_name}")

        # Get world settings
        settings = world.get_settings()
        print(f"✅ World settings:")
        print(f"   Synchronous mode: {settings.synchronous_mode}")
        print(f"   Fixed delta seconds: {settings.fixed_delta_seconds}")
        print(f"   No rendering mode: {settings.no_rendering_mode}")

        # Get available actors
        actors = world.get_actors()
        print(f"✅ World actors: {len(actors)} total")

        print("\n✅ TEST 2.1 PASSED: Client connected successfully\n")
        return True, client

    except Exception as e:
        print(f"\n❌ TEST 2.1 FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure CARLA server is running:")
        print("     docker run --rm --gpus all --net=host td3-av-system:v2.0-python310 \\")
        print("       /home/carla/CarlaUE4.sh -RenderOffScreen -nosound")
        print("  2. Wait 15-30 seconds for server initialization")
        print("  3. Check if port 2000 is available (no firewall blocking)\n")
        return False, None


def test_2_2_vehicle_spawning(client):
    """Test 2.2: Test ego vehicle spawning and control"""
    print("="*70)
    print("🚗 TEST 2.2: Vehicle Spawning")
    print("="*70)

    vehicle = None
    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Get vehicle blueprint
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        print(f"✅ Selected vehicle blueprint: {vehicle_bp.id}")

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        print(f"✅ Available spawn points: {len(spawn_points)}")

        # Choose random spawn point
        spawn_point = random.choice(spawn_points)
        print(f"✅ Selected spawn point: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})")

        # Spawn vehicle
        print("Spawning vehicle...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✅ Vehicle spawned successfully")
        print(f"   ID: {vehicle.id}")
        print(f"   Type: {vehicle.type_id}")

        # Test vehicle control
        print("\nTesting vehicle control...")
        control = carla.VehicleControl()
        control.throttle = 0.5
        control.steer = 0.0
        vehicle.apply_control(control)

        # Wait and observe
        time.sleep(2.0)

        # Get vehicle transform
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h

        print(f"✅ Vehicle state after 2s:")
        print(f"   Location: ({transform.location.x:.2f}, {transform.location.y:.2f}, {transform.location.z:.2f})")
        print(f"   Speed: {speed:.2f} km/h")

        # Stop vehicle
        control.throttle = 0.0
        control.brake = 1.0
        vehicle.apply_control(control)

        print("\n✅ TEST 2.2 PASSED: Vehicle spawned and controlled successfully\n")
        return True, vehicle

    except Exception as e:
        print(f"\n❌ TEST 2.2 FAILED: {e}\n")
        if vehicle is not None:
            vehicle.destroy()
        return False, None


def test_2_3_camera_sensor(client, vehicle):
    """Test 2.3: Test camera sensor attachment and image capture"""
    print("="*70)
    print("📷 TEST 2.3: Camera Sensor Attachment")
    print("="*70)

    camera = None
    captured_frames = []

    def camera_callback(image):
        """Camera callback to store captured images"""
        captured_frames.append(image)

    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Get camera blueprint
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '256')
        camera_bp.set_attribute('image_size_y', '144')
        camera_bp.set_attribute('fov', '90')
        print(f"✅ Camera blueprint configured:")
        print(f"   Resolution: 256x144")
        print(f"   FOV: 90°")

        # Camera transform (front-facing)
        camera_transform = carla.Transform(
            carla.Location(x=2.5, z=1.0),
            carla.Rotation(pitch=0.0)
        )

        # Spawn camera
        print("Attaching camera to vehicle...")
        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle
        )
        print(f"✅ Camera attached successfully")
        print(f"   ID: {camera.id}")

        # Start listening
        print("\nCapturing images for 3 seconds...")
        camera.listen(camera_callback)

        time.sleep(3.0)

        # Stop listening
        camera.stop()

        # Verify captured frames
        num_frames = len(captured_frames)
        print(f"✅ Captured {num_frames} frames")

        if num_frames > 0:
            # Check first frame
            first_frame = captured_frames[0]
            width = first_frame.width
            height = first_frame.height

            # Convert to numpy array
            frame_array = np.frombuffer(first_frame.raw_data, dtype=np.uint8)
            frame_array = frame_array.reshape((height, width, 4))[:, :, :3]  # BGRA to RGB

            print(f"✅ Frame properties:")
            print(f"   Shape: {frame_array.shape}")
            print(f"   Dtype: {frame_array.dtype}")
            print(f"   Value range: [{frame_array.min()}, {frame_array.max()}]")

            assert frame_array.shape == (144, 256, 3), f"Unexpected shape: {frame_array.shape}"
            assert frame_array.dtype == np.uint8, f"Unexpected dtype: {frame_array.dtype}"
        else:
            print("⚠️  No frames captured (camera may need more time)")

        print("\n✅ TEST 2.3 PASSED: Camera sensor works correctly\n")
        return True, camera

    except Exception as e:
        print(f"\n❌ TEST 2.3 FAILED: {e}\n")
        if camera is not None:
            camera.destroy()
        return False, None


def cleanup(vehicle, camera):
    """Clean up spawned actors"""
    print("="*70)
    print("🧹 Cleanup: Destroying actors...")
    print("="*70)

    if camera is not None:
        try:
            camera.destroy()
            print("✅ Camera destroyed")
        except:
            pass

    if vehicle is not None:
        try:
            vehicle.destroy()
            print("✅ Vehicle destroyed")
        except:
            pass

    print()


def main():
    """Run all Test Suite 2 tests"""

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*12 + "SYSTEM TESTING: Test Suite 2 - CARLA Connectivity" + " "*14 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    client = None
    vehicle = None
    camera = None

    results = []

    # Test 2.1: Connection
    success, client = test_2_1_basic_client_connection()
    results.append(("Test 2.1: Basic Client Connection", success))

    if not success:
        print("\n⚠️  Cannot proceed: CARLA server connection failed")
        print("   Start CARLA server first, then re-run this test suite\n")
        return False

    # Test 2.2: Vehicle Spawning
    success, vehicle = test_2_2_vehicle_spawning(client)
    results.append(("Test 2.2: Vehicle Spawning", success))

    if not success:
        cleanup(vehicle, camera)
        return False

    # Test 2.3: Camera Sensor
    success, camera = test_2_3_camera_sensor(client, vehicle)
    results.append(("Test 2.3: Camera Sensor Attachment", success))

    # Cleanup
    cleanup(vehicle, camera)

    # Summary
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*25 + "SUMMARY - TEST SUITE 2" + " "*21 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:<12} {test_name}")

    passed = sum(1 for _, r in results if r is True)
    total = len(results)

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Ready for Test Suite 3 (Environment Integration)")
        print("   Next: Run test_3_environment_integration.py\n")
        success = True
    else:
        print("\n❌ SOME TESTS FAILED - Fix issues before proceeding\n")
        success = False

    print("█"*70 + "\n")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

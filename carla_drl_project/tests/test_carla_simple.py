#!/usr/bin/env python3
"""
Simple CARLA Connection Test

This script tests basic CARLA connectivity and spawns a vehicle with camera visualization.
Optimized for development and debugging.
"""

import carla
import time
import numpy as np
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_carla_connection():
    """Test basic CARLA connection and operations"""
    client = None
    vehicle = None
    camera = None

    try:
        # Connect to CARLA
        logger.info("🔄 Connecting to CARLA...")
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(20.0)  # Increased timeout

        # Get version info
        version = client.get_server_version()
        logger.info(f"✅ Connected to CARLA {version}")

        # Get world (don't change map, use current one)
        world = client.get_world()
        current_map = world.get_map().name
        logger.info(f"🗺️  Current map: {current_map}")

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()
        logger.info(f"📍 Available spawn points: {len(spawn_points)}")

        if not spawn_points:
            logger.error("❌ No spawn points available")
            return False

        # Get vehicle blueprint
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        logger.info(f"🚗 Available vehicles: {len(vehicle_bps)}")

        # Choose a simple vehicle (not necessarily truck)
        vehicle_bp = None
        for bp in vehicle_bps:
            if 'tesla' in bp.id.lower() or 'lincoln' in bp.id.lower():
                vehicle_bp = bp
                break

        if not vehicle_bp:
            vehicle_bp = vehicle_bps[0]  # Use first available

        logger.info(f"🚙 Selected vehicle: {vehicle_bp.id}")

        # Spawn vehicle
        spawn_point = spawn_points[0]
        logger.info(f"🎯 Spawning at: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        if not vehicle:
            logger.error("❌ Failed to spawn vehicle")
            return False

        logger.info("✅ Vehicle spawned successfully")

        # Setup camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')

        # Camera transform (front of vehicle)
        camera_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.5),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logger.info("📹 Camera attached successfully")

        # Setup camera callback
        image_queue = []

        def camera_callback(data):
            """Simple camera callback"""
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))  # BGRA
            array = array[:, :, [2, 1, 0]]  # Convert to RGB for OpenCV
            image_queue.append(array)
            if len(image_queue) > 5:  # Keep only last 5 frames
                image_queue.pop(0)

        camera.listen(camera_callback)
        logger.info("🎥 Camera listening started")

        # Test control
        logger.info("🎮 Testing vehicle control...")
        for i in range(10):
            # Apply simple forward control
            control = carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)
            vehicle.apply_control(control)
            time.sleep(0.1)

        # Stop vehicle
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        vehicle.apply_control(control)

        # Wait for camera data
        logger.info("📸 Waiting for camera data...")
        timeout = time.time() + 5
        while len(image_queue) == 0 and time.time() < timeout:
            time.sleep(0.1)

        if image_queue:
            logger.info(f"✅ Received {len(image_queue)} camera frames")

            # Display one frame
            frame = image_queue[-1]
            cv2.imshow('CARLA Camera Test', frame)
            logger.info("📺 Displaying camera frame (press any key to continue)")
            cv2.waitKey(2000)  # Wait 2 seconds
            cv2.destroyAllWindows()
        else:
            logger.warning("⚠️  No camera data received")

        logger.info("✅ Test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")

        if camera and camera.is_alive:
            camera.stop()
            camera.destroy()

        if vehicle and vehicle.is_alive:
            vehicle.destroy()

        cv2.destroyAllWindows()
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    print("🚀 CARLA Connection Test")
    print("=" * 40)

    success = test_carla_connection()

    if success:
        print("\n🎉 All tests passed!")
        print("✅ CARLA client is working correctly")
    else:
        print("\n❌ Tests failed")
        print("Check CARLA server and configuration")

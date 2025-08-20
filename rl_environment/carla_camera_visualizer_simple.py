#!/usr/bin/env python
"""
CARLA camera visualization (simplified).

This script connects to a running CARLA server, spawns a vehicle with a camera,
and displays the camera feed using OpenCV.
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import random
import argparse
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Find CARLA module
try:
    # Try relative path first (running from a dir outside CarlaSimulator)
    carla_path = '../CarlaSimulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')
    logger.info(f"Looking for CARLA egg at: {carla_path}")
    sys.path.append(glob.glob(carla_path)[0])
except IndexError:
    try:
        # Try from the script's location
        project_root = Path(__file__).parent.parent
        carla_path = project_root / 'CarlaSimulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')
        logger.info(f"Looking for CARLA egg at: {carla_path}")

        # Try an absolute path as a last resort
        abs_path = r"c:\Users\danie\Documents\Documents\MESTRADO\25-2_aprendizado-por-reforco\CarlaSimulator\PythonAPI\carla\dist\carla-*%d.%d-%s.egg" % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')
        logger.info(f"Looking for CARLA egg at: {abs_path}")
        sys.path.append(glob.glob(abs_path)[0])
    except IndexError:
        logger.warning("Could not find CARLA egg file. Adding fallback paths...")
        # Add fallback paths
        project_root = Path(__file__).parent.parent
        carla_api_path = project_root / "CarlaSimulator" / "PythonAPI"
        sys.path.append(str(carla_api_path))
        logger.info(f"Added CARLA PythonAPI path: {carla_api_path}")

        carla_path = carla_api_path / "carla"
        if carla_path.exists():
            sys.path.append(str(carla_path))
            logger.info(f"Added CARLA path: {carla_path}")

# Import CARLA
try:
    import carla
    logger.info("CARLA imported successfully")
except ImportError as e:
    logger.error(f"Failed to import CARLA: {e}")
    logger.error("Make sure CARLA Python API is in your PYTHONPATH")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CARLA camera visualization")

    # CARLA connection
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')

    # Camera settings
    parser.add_argument('--width', type=int, default=800, help='Camera width')
    parser.add_argument('--height', type=int, default=600, help='Camera height')

    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    try:
        # Connect to CARLA server
        logger.info(f"Connecting to CARLA server at {args.host}:{args.port}")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # seconds
        logger.info("Connected successfully")

        # Get world
        world = client.get_world()
        logger.info(f"Got world: {world}")

        # Get map
        carla_map = world.get_map()
        logger.info(f"Got map: {carla_map.name}")

        # Get blueprint library
        blueprint_library = world.get_blueprint_library()
        logger.info("Got blueprint library")

        # Get vehicle blueprint
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        logger.info(f"Got vehicle blueprint: {vehicle_bp}")

        # Get spawn points
        spawn_points = carla_map.get_spawn_points()
        logger.info(f"Found {len(spawn_points)} spawn points")

        if not spawn_points:
            logger.error("No spawn points found")
            return False

        # Choose random spawn point
        spawn_point = random.choice(spawn_points)
        logger.info(f"Selected spawn point: {spawn_point}")

        # Spawn vehicle
        logger.info("Spawning vehicle...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"Vehicle spawned: {vehicle}")

        # Get camera blueprint
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '90')
        logger.info(f"Configured camera blueprint: {camera_bp}")

        # Set camera position
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
        logger.info(f"Camera transform: {camera_transform}")

        # Create OpenCV window
        cv2.namedWindow('CARLA Camera', cv2.WINDOW_AUTOSIZE)
        logger.info("Created OpenCV window")

        # Image data storage
        image_data = {'frame': None}

        # Define image callback function
        def image_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Convert to RGB
            image_data['frame'] = array

        # Spawn camera
        logger.info("Spawning camera...")
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        logger.info(f"Camera spawned: {camera}")

        # Register callback
        logger.info("Registering camera callback...")
        camera.listen(image_callback)
        logger.info("Camera callback registered")

        # Enable autopilot
        logger.info("Enabling autopilot...")
        vehicle.set_autopilot(True)
        logger.info("Autopilot enabled")

        # Main loop
        logger.info("Starting main loop...")
        try:
            frame_counter = 0
            start_time = time.time()

            while True:
                if image_data['frame'] is not None:
                    # Add FPS counter
                    img_display = image_data['frame'].copy()
                    elapsed_time = time.time() - start_time
                    fps = frame_counter / elapsed_time if elapsed_time > 0 else 0

                    cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Display the image
                    cv2.imshow('CARLA Camera', img_display)
                    frame_counter += 1

                # Check for exit key (ESC or Q)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    break

                time.sleep(0.01)  # Small sleep to prevent CPU hogging

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            # Cleanup
            logger.info("Cleaning up...")
            if 'camera' in locals():
                camera.stop()
                camera.destroy()
                logger.info("Camera destroyed")

            if 'vehicle' in locals():
                vehicle.destroy()
                logger.info("Vehicle destroyed")

            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")

        return True

    except Exception as e:
        logger.exception(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Run main
    success = main()
    sys.exit(0 if success else 1)

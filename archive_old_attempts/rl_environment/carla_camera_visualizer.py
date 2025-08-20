#!/usr/bin/env python
"""
CARLA camera visualization.

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
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add CARLA PythonAPI to path
project_root = Path(__file__).parent.parent
carla_path = project_root / "CarlaSimulator" / "PythonAPI"
sys.path.insert(0, str(carla_path))
carla_client_path = project_root / "CarlaSimulator" / "PythonClient"
sys.path.insert(0, str(carla_client_path))

# Import CARLA
try:
    import carla
    from carla import Client, Transform, Location, Rotation
    logger.info("CARLA imported successfully")
except ImportError as e:
    logger.error(f"Failed to import CARLA: {e}")
    logger.error("Make sure CARLA Python API is in your PYTHONPATH")
    sys.exit(1)

class CameraVisualizer:
    """Class to handle CARLA camera visualization with OpenCV."""

    def __init__(self, args):
        """Initialize the visualizer."""
        self.args = args
        self.client = None
        self.world = None
        self.camera = None
        self.vehicle = None
        self.current_image = None
        self.running = True

        # Create display window
        cv2.namedWindow('CARLA Camera', cv2.WINDOW_AUTOSIZE)

    def connect_to_carla(self):
        """Connect to CARLA server."""
        try:
            self.client = Client(self.args.host, self.args.port)
            self.client.set_timeout(10.0)  # seconds
            self.world = self.client.get_world()
            logger.info(f"Connected to CARLA server at {self.args.host}:{self.args.port}")

            # Set synchronous mode if needed
            settings = self.world.get_settings()
            if self.args.sync:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)

            return True
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False

    def spawn_vehicle(self):
        """Spawn a vehicle in the world."""
        try:
            # Get blueprint for vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

            # Find a valid spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                logger.error("No spawn points found")
                return False

            spawn_point = random.choice(spawn_points)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            logger.info(f"Vehicle spawned at {spawn_point.location}")

            # Let it settle
            time.sleep(2)

            return True
        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            return False

    def setup_camera(self):
        """Attach a camera to the vehicle."""
        try:
            # Get blueprint for camera
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')

            # Set camera attributes
            camera_bp.set_attribute('image_size_x', str(self.args.width))
            camera_bp.set_attribute('image_size_y', str(self.args.height))
            camera_bp.set_attribute('fov', '90')

            # Define camera position relative to the vehicle
            camera_transform = Transform(Location(x=1.5, z=2.4), Rotation(pitch=-15))

            # Spawn the camera and attach it to our vehicle
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

            # Register a callback function to save each image
            self.camera.listen(lambda image: self.process_camera_data(image))

            logger.info("Camera attached to vehicle")
            return True
        except Exception as e:
            logger.error(f"Failed to set up camera: {e}")
            return False

    def process_camera_data(self, image):
        """Process camera data and save the image."""
        try:
            # Convert CARLA raw image to OpenCV format
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Convert to RGB

            # Save the current image
            self.current_image = array
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")

    def setup_autopilot(self):
        """Enable autopilot for the vehicle."""
        if not self.vehicle:
            logger.error("Vehicle not spawned")
            return False

        try:
            self.vehicle.set_autopilot(True)
            logger.info("Autopilot enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable autopilot: {e}")
            return False

    def run(self):
        """Run the camera visualization."""
        try:
            frame_counter = 0
            start_time = time.time()

            while self.running:
                # Display the current camera image
                if self.current_image is not None:
                    # Add FPS counter
                    img_display = self.current_image.copy()
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
                        self.running = False
                        break

                # For synchronous mode
                if self.args.sync and self.world:
                    self.world.tick()
                else:
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging

            return True
        except KeyboardInterrupt:
            logger.info("Visualization interrupted by user")
            return True
        except Exception as e:
            logger.error(f"Error in run loop: {e}")
            return False

    def cleanup(self):
        """Clean up resources."""
        try:
            logger.info("Cleaning up resources...")

            if self.camera:
                self.camera.stop()
                self.camera.destroy()
                logger.info("Camera destroyed")

            if self.vehicle:
                self.vehicle.destroy()
                logger.info("Vehicle destroyed")

            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")

            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CARLA camera visualization")

    # CARLA connection
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')

    # Camera settings
    parser.add_argument('--width', type=int, default=800, help='Camera width')
    parser.add_argument('--height', type=int, default=600, help='Camera height')

    # Other settings
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')

    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Create the visualizer
    visualizer = CameraVisualizer(args)

    try:
        # Connect to CARLA
        if not visualizer.connect_to_carla():
            return False

        # Spawn vehicle
        if not visualizer.spawn_vehicle():
            visualizer.cleanup()
            return False

        # Setup camera
        if not visualizer.setup_camera():
            visualizer.cleanup()
            return False

        # Setup autopilot
        if not visualizer.setup_autopilot():
            visualizer.cleanup()
            return False

        # Run visualization
        visualizer.run()

        # Cleanup
        visualizer.cleanup()

        return True
    except KeyboardInterrupt:
        logger.info("Visualization interrupted by user")
        visualizer.cleanup()
        return True
    except Exception as e:
        logger.error(f"Error in main: {e}")
        visualizer.cleanup()
        return False

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Run main
    success = main()
    sys.exit(0 if success else 1)

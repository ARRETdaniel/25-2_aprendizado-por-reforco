#!/usr/bin/env python3
"""
Script to initialize CARLA cameras and sensors and publish data to the ROS bridge.

This script needs to be run with Python 3.6 since it directly interfaces with CARLA.
"""

import os
import sys
import time
import math
import logging
import random
import pickle
import threading
import numpy as np
from queue import Queue, Empty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current directory: {current_dir}")

# Add the PythonAPI dir to the path
carla_pythonapi_path = os.path.abspath(os.path.join(current_dir, '..', 'CarlaSimulator', 'PythonAPI'))
sys.path.append(carla_pythonapi_path)
logger.info(f"Added to path: {carla_pythonapi_path}")

# Add the carla dir to the path
carla_path = os.path.join(carla_pythonapi_path, 'carla')
sys.path.append(carla_path)
logger.info(f"Added to path: {carla_path}")

# Add the examples dir to the path
examples_path = os.path.join(carla_pythonapi_path, 'examples')
sys.path.append(examples_path)
logger.info(f"Added to path: {examples_path}")

# Try to find and append the CARLA .egg file
try:
    import glob
    carla_egg_pattern = os.path.join(carla_path, 'dist', f'carla-*{sys.version_info.major}.{sys.version_info.minor}-*.egg')
    logger.info(f"Looking for CARLA egg file with pattern: {carla_egg_pattern}")
    carla_egg = glob.glob(carla_egg_pattern)
    if carla_egg:
        egg_path = carla_egg[0]
        sys.path.append(egg_path)
        logger.info(f"Found and added CARLA egg: {egg_path}")
    else:
        logger.warning(f"No CARLA egg file found with pattern: {carla_egg_pattern}")
except Exception as e:
    logger.error(f"Error finding CARLA egg: {e}")

# Try to import CARLA
try:
    import carla
    logger.info("Successfully imported CARLA module")

except ImportError as e:
    logger.error(f"Failed to import CARLA module: {e}")
    sys.exit(1)

# Import file-based bridge
file_bridge_path = os.path.join(current_dir, 'ros_bridge.py')
sys.path.append(os.path.dirname(file_bridge_path))
try:
    from ros_bridge import CARLABridge
    logger.info("Successfully imported ROS bridge")
except ImportError as e:
    logger.error(f"Failed to import ROS bridge: {e}")
    sys.exit(1)

class CarlaSensorPublisher:
    """
    Class to set up CARLA sensors and publish data to the ROS bridge.
    """

    def __init__(self, host='localhost', port=2000, town='Town01', fps=20):
        """Initialize the CARLA sensor publisher."""
        self.host = host
        self.port = port
        self.town = town
        self.fps = fps
        self.delta_seconds = 1.0 / fps

        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle = None
        self.camera_rgb = None
        self.camera_depth = None
        self.camera_semseg = None
        self.rgb_queue = Queue()
        self.depth_queue = Queue()
        self.semseg_queue = Queue()

        # Initialize ROS bridge
        self.bridge = CARLABridge()

        # Storage for last observation
        self.last_rgb = None
        self.last_depth = None
        self.last_semseg = None
        self.last_state = None

    def connect(self):
        """Connect to CARLA server."""
        try:
            logger.info(f"Connecting to CARLA server at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)

            # Load the desired map
            logger.info(f"Loading map {self.town}")
            self.world = self.client.get_world()
            if self.world.get_map().name != self.town:
                logger.info(f"Loading map {self.town}")
                self.world = self.client.load_world(self.town)

            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.delta_seconds
            self.world.apply_settings(settings)

            # Get the blueprint library
            self.blueprint_library = self.world.get_blueprint_library()

            logger.info("Successfully connected to CARLA server")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            return False

    def spawn_vehicle(self):
        """Spawn a vehicle in the world."""
        try:
            # Get a random vehicle blueprint
            bp = random.choice(self.blueprint_library.filter('vehicle.*'))

            # Get a random spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)

            # Spawn the vehicle
            logger.info(f"Spawning vehicle {bp.id}")
            self.vehicle = self.world.spawn_actor(bp, spawn_point)

            # Set autopilot
            self.vehicle.set_autopilot(True)

            logger.info(f"Vehicle spawned: {self.vehicle}")
            return True

        except Exception as e:
            logger.error(f"Failed to spawn vehicle: {e}")
            return False

    def setup_cameras(self):
        """Set up RGB, depth, and semantic segmentation cameras."""
        try:
            # Camera setup parameters
            cam_location = carla.Location(x=1.5, z=2.4)
            cam_rotation = carla.Rotation(pitch=-15)
            cam_transform = carla.Transform(cam_location, cam_rotation)

            # RGB camera
            rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
            rgb_bp.set_attribute('image_size_x', '800')
            rgb_bp.set_attribute('image_size_y', '600')
            rgb_bp.set_attribute('fov', '90')
            self.camera_rgb = self.world.spawn_actor(
                rgb_bp, cam_transform, attach_to=self.vehicle)
            self.camera_rgb.listen(lambda image: self._process_rgb_image(image))

            # Depth camera
            depth_bp = self.blueprint_library.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', '800')
            depth_bp.set_attribute('image_size_y', '600')
            depth_bp.set_attribute('fov', '90')
            self.camera_depth = self.world.spawn_actor(
                depth_bp, cam_transform, attach_to=self.vehicle)
            self.camera_depth.listen(lambda image: self._process_depth_image(image))

            # Semantic segmentation camera
            semseg_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            semseg_bp.set_attribute('image_size_x', '800')
            semseg_bp.set_attribute('image_size_y', '600')
            semseg_bp.set_attribute('fov', '90')
            self.camera_semseg = self.world.spawn_actor(
                semseg_bp, cam_transform, attach_to=self.vehicle)
            self.camera_semseg.listen(lambda image: self._process_semseg_image(image))

            logger.info("Cameras set up successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to set up cameras: {e}")
            return False

    def _process_rgb_image(self, image):
        """Process RGB image."""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Extract RGB
            self.last_rgb = array
            self.rgb_queue.put(array)
        except Exception as e:
            logger.error(f"Error processing RGB image: {e}")

    def _process_depth_image(self, image):
        """Process depth image."""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Extract RGB channels

            # Convert depth
            depth_array = array[:, :, 0] + array[:, :, 1] * 256 + array[:, :, 2] * 256 * 256
            depth_array = depth_array.astype(np.float32) / (256 * 256 * 256 - 1)
            depth_array = depth_array * 1000  # Convert to meters

            # Normalize to [0, 255] for visualization
            normalized = (depth_array * 255 / np.max(depth_array)).astype(np.uint8)
            colored = np.stack([normalized, normalized, normalized], axis=-1)

            self.last_depth = colored
            self.depth_queue.put(colored)
        except Exception as e:
            logger.error(f"Error processing depth image: {e}")

    def _process_semseg_image(self, image):
        """Process semantic segmentation image."""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Extract RGB
            self.last_semseg = array
            self.semseg_queue.put(array)
        except Exception as e:
            logger.error(f"Error processing semantic segmentation image: {e}")

    def get_vehicle_state(self):
        """Get the vehicle state."""
        try:
            if self.vehicle is None:
                return None

            # Get vehicle transform
            transform = self.vehicle.get_transform()

            # Get vehicle velocity
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

            # Get vehicle control
            control = self.vehicle.get_control()

            # Construct state dictionary
            state = {
                'location': {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                },
                'rotation': {
                    'pitch': transform.rotation.pitch,
                    'yaw': transform.rotation.yaw,
                    'roll': transform.rotation.roll
                },
                'velocity': {
                    'x': velocity.x,
                    'y': velocity.y,
                    'z': velocity.z,
                    'speed': speed
                },
                'control': {
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake,
                    'hand_brake': control.hand_brake,
                    'reverse': control.reverse
                }
            }

            self.last_state = state
            return state

        except Exception as e:
            logger.error(f"Error getting vehicle state: {e}")
            return None

    def calculate_reward(self, state):
        """Calculate a reward based on the vehicle state."""
        if state is None:
            return 0.0

        # Simple reward function based on speed and steering
        speed = state['velocity']['speed']
        steer = abs(state['control']['steer'])

        # Reward for moving at a reasonable speed
        speed_reward = min(speed / 50.0, 1.0) if speed < 50.0 else 1.0 - min((speed - 50.0) / 50.0, 1.0)

        # Penalty for excessive steering
        steer_penalty = steer * 0.5

        # Combine rewards
        total_reward = speed_reward - steer_penalty

        return float(total_reward)

    def publish_observation(self):
        """Publish the current observation to the ROS bridge."""
        try:
            # Get the current state
            state = self.get_vehicle_state()

            # Calculate reward
            reward = self.calculate_reward(state)

            # Check if we have images
            rgb_image = self.last_rgb
            depth_image = self.last_depth
            semseg_image = self.last_semseg

            if rgb_image is None or depth_image is None or semseg_image is None:
                logger.warning("Missing camera images, waiting for sensors...")
                return False

            # Publish observation
            logger.info("Publishing observation to ROS bridge")
            info = {'frame': self.world.get_snapshot().frame}
            done = False

            self.bridge.publish_cameras(rgb_image, depth_image, semseg_image)
            self.bridge.publish_state(state)
            self.bridge.publish_reward(reward)
            self.bridge.publish_done(done)
            self.bridge.publish_info(info)

            logger.info(f"Published observation: reward={reward}, info={info}")
            return True

        except Exception as e:
            logger.error(f"Error publishing observation: {e}")
            return False

    def run(self, num_steps=1000):
        """Run the main loop."""
        try:
            # Connect to CARLA
            if not self.connect():
                return False

            # Spawn vehicle
            if not self.spawn_vehicle():
                return False

            # Setup cameras
            if not self.setup_cameras():
                return False

            logger.info(f"Running for {num_steps} steps")

            # Main loop
            for step in range(num_steps):
                logger.info(f"Step {step+1}/{num_steps}")

                # Tick the world
                self.world.tick()

                # Wait for sensors to provide data
                time.sleep(0.1)

                # Publish observation
                self.publish_observation()

                # Process any actions received from the DRL agent
                # This would read actions from the bridge and apply to vehicle

            return True

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            return False

        finally:
            # Clean up actors
            if self.camera_rgb:
                self.camera_rgb.destroy()
            if self.camera_depth:
                self.camera_depth.destroy()
            if self.camera_semseg:
                self.camera_semseg.destroy()
            if self.vehicle:
                self.vehicle.destroy()

            # Restore world settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            logger.info("Cleaned up CARLA actors")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CARLA Sensor Publisher')

    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='CARLA server host'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=2000,
        help='CARLA server port'
    )

    parser.add_argument(
        '--town',
        type=str,
        default='Town01',
        help='CARLA town/map'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Simulation frames per second'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of steps to run'
    )

    args = parser.parse_args()

    # Create and run the sensor publisher
    publisher = CarlaSensorPublisher(
        host=args.host,
        port=args.port,
        town=args.town,
        fps=args.fps
    )

    success = publisher.run(num_steps=args.steps)

    return 0 if success else 1

if __name__ == "__main__":
    import argparse
    sys.exit(main())

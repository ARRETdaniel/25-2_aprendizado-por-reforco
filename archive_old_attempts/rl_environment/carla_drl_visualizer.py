#!/usr/bin/env python
"""
CARLA Deep Reinforcement Learning with camera visualization.

This script connects to a running CARLA server, sets up a camera,
and uses OpenCV to display the view while the agent learns.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
import cv2
import torch
import random
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure CARLA paths are in sys.path
carla_api_path = project_root / "CarlaSimulator" / "PythonAPI"
sys.path.insert(0, str(carla_api_path))
carla_client_path = project_root / "CarlaSimulator" / "PythonClient"
sys.path.insert(0, str(carla_client_path))

# Try to import CARLA from PythonClient (Coursera version 0.8.4)
try:
    from carla.client import make_carla_client, VehicleControl
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.tcp import TCPConnectionError
    from carla.util import print_over_same_line
    logger.info("CARLA modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import CARLA: {e}")
    logger.error("Make sure CARLA Python API is in your PYTHONPATH")
    sys.exit(1)

# Import our SAC implementation
try:
    from rl_environment.environment import CarlaEnvWrapper
    from rl_environment.simple_sac import SimpleSAC
    logger.info("DRL modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import DRL modules: {e}")
    sys.exit(1)

class CarlaDRLVisualizer:
    """
    Class to handle CARLA environment visualization with OpenCV.
    """
    def __init__(self, args):
        """Initialize the visualizer."""
        self.args = args
        self.client = None
        self.world = None
        self.camera = None
        self.vehicle = None
        self.frame_counter = 0
        self.current_image = None
        self.running = True
        self.env = None
        self.agent = None

        # Create display windows
        cv2.namedWindow('CARLA Camera', cv2.WINDOW_AUTOSIZE)

    def setup_carla_client(self):
        """Connect to CARLA server and setup the world."""
        try:
            with make_carla_client(self.args.host, self.args.port, timeout=10.0) as client:
                self.client = client
                logger.info(f"Connected to CARLA server at {self.args.host}:{self.args.port}")

                # Create CarlaSettings
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=self.args.sync,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=2,
                    NumberOfPedestrians=0,
                    WeatherId=0,
                    QualityLevel='Low'
                )

                # Load the scene
                scene = self.client.load_settings(settings)
                self.world = scene

            return True
        except TCPConnectionError as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting up CARLA client: {e}")
            return False    def setup_vehicle(self):
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
        """Process camera data."""
        try:
            # Convert CARLA raw image to OpenCV format
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
            array = array[:, :, :3]  # Convert to RGB

            # Save the current image
            self.current_image = array
            self.frame_counter += 1

            # Display the image with OpenCV (in main loop)
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")

    def setup_drl(self):
        """Initialize the DRL environment and agent."""
        try:
            # Initialize environment with CARLA connection
            self.env = CarlaEnvWrapper(
                host=self.args.host,
                port=self.args.port,
                city_name='Town01',
                image_size=(self.args.height, self.args.width),
                frame_skip=1,
                max_episode_steps=1000,
                weather_id=0,
                quality_level='Low'
            )

            # Reset environment to get state dimensions
            state = self.env.reset()
            state_dims = {key: value.shape for key, value in state.items()}
            action_dim = self.env.action_space.shape[0]

            logger.info(f"State dimensions: {state_dims}")
            logger.info(f"Action dimension: {action_dim}")

            # Initialize SAC agent
            self.agent = SimpleSAC(
                state_dims=state_dims,
                action_dim=action_dim,
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                buffer_size=100000,
                batch_size=64,
                feature_dim=128,
                hidden_dim=256,
                update_freq=2,
                seed=42
            )

            return True
        except Exception as e:
            logger.error(f"Failed to setup DRL: {e}")
            return False

    def run_training_loop(self):
        """Run the DRL training loop with visualization."""
        if not self.env or not self.agent:
            logger.error("Environment or agent not initialized")
            return False

        try:
            episode = 0
            max_episodes = self.args.episodes

            while episode < max_episodes and self.running:
                logger.info(f"Starting episode {episode + 1}/{max_episodes}")

                # Reset the environment
                state = self.env.reset()
                episode_reward = 0
                step = 0
                done = False

                # Episode loop
                while not done and self.running:
                    # Select action
                    action = self.agent.select_action(state)

                    # Take step in environment
                    next_state, reward, done, info = self.env.step(action)

                    # Process step in agent
                    self.agent.step(state, action, reward, next_state, done)

                    # Update state
                    state = next_state
                    episode_reward += reward
                    step += 1

                    # Display the current camera image
                    if self.current_image is not None:
                        # Add text with episode info
                        img_display = self.current_image.copy()
                        cv2.putText(img_display,
                                  f"Episode: {episode+1}, Step: {step}, Reward: {reward:.2f}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        cv2.imshow('CARLA Camera', img_display)

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

                # Episode ended
                logger.info(f"Episode {episode + 1} ended with reward {episode_reward:.2f} after {step} steps")
                episode += 1

                # Save model periodically
                if episode % 10 == 0:
                    save_path = os.path.join(self.args.save_dir, f"sac_episode_{episode}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.agent.save(save_path)
                    logger.info(f"Model saved to {save_path}")

            # Final save
            save_path = os.path.join(self.args.save_dir, "sac_final")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.agent.save(save_path)
            logger.info(f"Final model saved to {save_path}")

            return True
        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            import traceback
            traceback.print_exc()
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

            if self.env:
                self.env.close()
                logger.info("Environment closed")

            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")

            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CARLA DRL with camera visualization")

    # CARLA connection
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')

    # Camera settings
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--save-dir', default='./checkpoints', help='Directory to save models')

    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Create the visualizer
    visualizer = CarlaDRLVisualizer(args)

    try:
        # Setup CARLA client
        if not visualizer.setup_carla_client():
            logger.error("Failed to set up CARLA client")
            return False

        # Setup vehicle
        if not visualizer.setup_vehicle():
            logger.error("Failed to set up vehicle")
            visualizer.cleanup()
            return False

        # Setup camera
        if not visualizer.setup_camera():
            logger.error("Failed to set up camera")
            visualizer.cleanup()
            return False

        # Setup DRL
        if not visualizer.setup_drl():
            logger.error("Failed to set up DRL")
            visualizer.cleanup()
            return False

        # Run training loop
        visualizer.run_training_loop()

        # Cleanup
        visualizer.cleanup()

        logger.info("Training completed successfully")
        return True

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        visualizer.cleanup()
        return True
    except Exception as e:
        logger.error(f"Error in main: {e}")
        visualizer.cleanup()
        return False

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run main
    success = main()
    sys.exit(0 if success else 1)

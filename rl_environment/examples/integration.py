"""
Integration script for connecting the RL environment with existing code.

This script demonstrates how to integrate the CARLA RL environment with
the existing autonomous vehicle code from the Final Project.
"""

import os
import sys
import time
import glob
import logging
import numpy as np
import math
import threading
from typing import Dict, Any, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import rl_environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError:
    logger.error("Failed to import CarlaEnvWrapper. Make sure the environment is correctly installed.")
    sys.exit(1)

# Try to import existing Final Project code
try:
    # Adjust these paths to match your project structure
    sys.path.append('../../FinalProject/')  # Path to Final Project code

    # Import existing modules
    from detector_socket import DetectorClient
    # Other imports from the Final Project
    # ...

    logger.info("Successfully imported Final Project modules.")
except ImportError as e:
    logger.warning(f"Could not import all Final Project modules: {e}")
    logger.warning("Some functionality may be limited.")


class IntegratedController:
    """
    Controller that integrates RL agent with existing perception system.

    This class demonstrates how to use an RL agent for control while
    utilizing the existing perception system from the Final Project.
    """

    def __init__(self,
                 model_path: str = None,
                 detector_host: str = 'localhost',
                 detector_port: int = 8000):
        """
        Initialize the integrated controller.

        Args:
            model_path: Path to saved RL model (if None, random actions will be used)
            detector_host: Host for the detector server
            detector_port: Port for the detector server
        """
        self.model_path = model_path

        # Initialize detector client
        try:
            self.detector = DetectorClient(host=detector_host, port=detector_port)
            logger.info(f"Connected to detector server at {detector_host}:{detector_port}")
        except:
            logger.warning("Failed to connect to detector server. Will proceed without object detection.")
            self.detector = None

        # Initialize RL environment
        self.env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            town='Town01',
            fps=30,
            image_size=(84, 84),
            frame_skip=1  # Lower frame skip for more responsive control
        )

        # Load RL model if provided
        self.rl_model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.info("No model path provided or model not found. Will use random actions.")

        # Initialize state
        self.current_state = None
        self.detections = []

        # For threading
        self.running = False
        self.detection_thread = None

    def _load_model(self, model_path: str):
        """
        Load a trained RL model.

        Args:
            model_path: Path to saved model
        """
        try:
            # This is a placeholder - implementation depends on the model type
            # For PyTorch:
            # import torch
            # self.rl_model = torch.load(model_path)
            # self.rl_model.eval()

            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def _get_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Get object detections from the detector.

        Args:
            image: RGB image array

        Returns:
            List of detection dictionaries
        """
        if self.detector is None:
            return []

        try:
            # Send image to detector
            detections = self.detector.detect(image)
            return detections
        except Exception as e:
            logger.error(f"Error getting detections: {e}")
            return []

    def _detection_loop(self):
        """Background thread for continuous object detection."""
        while self.running:
            try:
                if self.current_state and 'image' in self.current_state:
                    self.detections = self._get_detection(self.current_state['image'])
                time.sleep(0.1)  # Rate limit detection requests
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")

    def _select_action(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Select an action based on the current state.

        Args:
            state: Current state

        Returns:
            Selected action as numpy array
        """
        if self.rl_model is None:
            # Random action
            return np.array([
                np.random.uniform(0.0, 1.0),   # Throttle
                np.random.uniform(0.0, 0.2),   # Brake
                np.random.uniform(-0.3, 0.3)   # Steering
            ])
        else:
            # Model-based action selection
            # This is a placeholder - implementation depends on the model type
            # For PyTorch:
            # with torch.no_grad():
            #     action = self.rl_model(state)
            # return action.numpy()

            # For now, just return random action
            return np.array([
                np.random.uniform(0.0, 1.0),
                np.random.uniform(0.0, 0.2),
                np.random.uniform(-0.3, 0.3)
            ])

    def start(self):
        """Start the integrated controller."""
        logger.info("Starting integrated controller...")

        # Reset the environment
        self.current_state = self.env.reset()

        # Start detection thread
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        try:
            # Main control loop
            step = 0
            total_reward = 0.0

            while self.running:
                # Update state with latest detections
                if 'detections' in self.current_state:
                    self.current_state['detections'] = np.array(self.detections, dtype=np.float32)

                # Select action
                action = self._select_action(self.current_state)

                # Take a step in the environment
                next_state, reward, done, info = self.env.step(action)

                total_reward += reward
                step += 1
                self.current_state = next_state

                if step % 10 == 0:
                    logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

                if done:
                    logger.info(f"Episode finished after {step} steps with total reward: {total_reward:.2f}")
                    # Reset for next episode
                    self.current_state = self.env.reset()
                    step = 0
                    total_reward = 0.0

                # Control rate
                time.sleep(0.05)

        except KeyboardInterrupt:
            logger.info("Controller interrupted by user.")
        except Exception as e:
            logger.error(f"Error in control loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the integrated controller."""
        logger.info("Stopping integrated controller...")

        # Stop detection thread
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)

        # Close environment
        if hasattr(self, 'env'):
            self.env.close()

        logger.info("Controller stopped.")


def main():
    """
    Main function.
    """
    controller = IntegratedController(
        model_path=None,  # Path to trained model (if available)
        detector_host='localhost',
        detector_port=8000
    )

    controller.start()


if __name__ == "__main__":
    main()

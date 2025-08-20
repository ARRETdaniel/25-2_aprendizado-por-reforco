"""
Integration test for the CARLA environment wrapper.

This script demonstrates how to use the CARLA environment wrapper with the
existing CARLA 0.8.4 implementation and detector system.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import rl_environment
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Add FinalProject directory to path for detector imports
final_project_path = os.path.join(parent_dir, 'CarlaSimulator', 'PythonClient', 'FinalProject')
sys.path.append(final_project_path)
detector_path = os.path.join(final_project_path, 'detector_socket')
sys.path.append(detector_path)

# Import the environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError:
    logger.error("Failed to import CarlaEnvWrapper. Make sure the environment is correctly installed.")
    sys.exit(1)

# Import detector client if available
try:
    from detector_socket.detector_client import DetectionClient
    HAS_DETECTOR = True
    logger.info("Detector client imported successfully.")
except ImportError:
    logger.warning("Could not import detector client. Will proceed without object detection.")
    HAS_DETECTOR = False


def run_test(args):
    """
    Run test with the CARLA environment wrapper.
    """
    logger.info("Starting environment wrapper test...")

    # Create the environment
    try:
        env = CarlaEnvWrapper(
            host=args.host,
            port=args.port,
            city_name=args.map,
            image_size=(84, 84),
            frame_skip=1,
            max_episode_steps=args.steps,
            weather_id=args.weather,
            quality_level=args.quality_level
        )
        logger.info("Environment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    # Initialize detector if available
    detector = None
    if HAS_DETECTOR and args.use_detector:
        try:
            detector = DetectionClient(host=args.detector_host, port=args.detector_port)
            logger.info(f"Connected to detector server at {args.detector_host}:{args.detector_port}")
        except Exception as e:
            logger.warning(f"Could not connect to detector server: {e}")

    try:
        # Run for one episode
        logger.info("Resetting environment...")
        state = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        logger.info("Starting episode...")
        while not done:
            # Take a simple action (go forward with slight steering)
            action = np.array([0.5, 0.0, 0.0])  # throttle=0.5, brake=0, steer=0

            # Take a step
            next_state, reward, done, info = env.step(action)

            # If detector is available, run detection
            if detector:
                if 'image' in next_state:
                    detections = detector.detect(next_state['image'])
                    logger.info(f"Detections: {detections}")

            total_reward += reward
            step += 1

            logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

            # Sleep to slow down the test
            time.sleep(0.01)

            # Break early for testing
            if args.debug and step >= 10:
                logger.info("Debug mode: breaking early after 10 steps")
                break

        logger.info(f"Episode finished after {step} steps with total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'env' in locals():
            logger.info("Closing environment...")
            env.close()
        logger.info("Test ended.")


def main():
    """
    Main function.
    """
    argparser = argparse.ArgumentParser(description="Test CARLA Environment Wrapper")
    argparser.add_argument('--host', default='localhost',
                          help='IP of the host server (default: localhost)')
    argparser.add_argument('-p', '--port', default=2000, type=int,
                          help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-m', '--map', default='Town01',
                          help='City/map to use (default: Town01)')
    argparser.add_argument('-s', '--steps', default=1000, type=int,
                          help='Maximum steps per episode (default: 1000)')
    argparser.add_argument('-w', '--weather', default=0, type=int,
                          help='Weather preset ID (default: 0)')
    argparser.add_argument('-q', '--quality-level', choices=['Low', 'Epic'],
                          default='Low', help='Graphics quality level')
    argparser.add_argument('-d', '--use-detector', action='store_true',
                          help='Use object detector')
    argparser.add_argument('--detector-host', default='localhost',
                          help='Detector server host (default: localhost)')
    argparser.add_argument('--detector-port', default=5555, type=int,
                          help='Detector server port (default: 5555)')
    argparser.add_argument('--debug', action='store_true',
                          help='Run in debug mode (limited steps)')

    args = argparser.parse_args()
    run_test(args)


if __name__ == '__main__':
    main()

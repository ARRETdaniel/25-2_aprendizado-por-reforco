"""
CARLA Camera Visualizer with ROS 2 Bridge

This module provides a CARLA client that captures camera feeds and publishes them
using the ROS 2 bridge. It's designed to run in a Python 3.6 environment and
interfaces with the CARLA simulator.
"""

import os
import sys
import time
import math
import logging
import argparse
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add PythonAPI path to import CARLA modules
try:
    carla_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'CarlaSimulator', 'PythonAPI')
    sys.path.append(carla_path)
    from carla.client import make_carla_client, VehicleControl
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.tcp import TCPConnectionError
    from carla.image_converter import to_rgb_array

    logger.info("Successfully imported CARLA modules")
except ImportError as e:
    logger.error(f"Failed to import CARLA modules: {e}")
    logger.error("Make sure to run this script with Python 3.6")
    sys.exit(1)

# Import ROS bridge
try:
    from ros_bridge import CARLABridge
    logger.info("Successfully imported ROS bridge")
except ImportError as e:
    logger.error(f"Failed to import ROS bridge: {e}")
    logger.error("Make sure ros_bridge.py is in the same directory")
    sys.exit(1)

# Import OpenCV for visualization
try:
    import cv2
    HAS_CV2 = True
    logger.info("Successfully imported OpenCV")
except ImportError:
    logger.warning("OpenCV not found, visualization will be disabled")
    HAS_CV2 = False


# CARLA simulator configuration
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 2000
DEFAULT_TIMEOUT = 10
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180


class CARLACameraVisualizer:
    """
    CARLA client that captures camera feeds and visualizes them.

    This class connects to the CARLA simulator, sets up cameras, and
    provides methods to capture and visualize camera feeds. It also
    publishes the camera feeds using the ROS bridge.
    """

    def __init__(self, args):
        """
        Initialize the CARLA Camera Visualizer.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.client = None
        self.scene = None
        self.player = None
        self.measurements = None
        self.ros_bridge = None
        self.frame = 0
        self.rgb_image = None
        self.depth_image = None
        self.seg_image = None
        self.running = True

        # Initialize ROS bridge
        self.ros_bridge = CARLABridge(use_ros=not args.no_ros)

        # Override action and control callbacks
        self.ros_bridge.action_callback = self._handle_action
        self.ros_bridge.control_callback = self._handle_control

        # Initialize display windows if OpenCV is available
        if HAS_CV2 and not args.no_display:
            cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth Camera', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Segmentation Camera', cv2.WINDOW_AUTOSIZE)

    def _handle_action(self, msg):
        """Handle action messages from DRL agent."""
        if hasattr(msg, 'data'):
            action = np.array(msg.data, dtype=np.float32)
            logger.debug(f"Received action: {action}")

            # Convert action to control command
            # Assuming action is [throttle, steering]
            if len(action) >= 2:
                throttle = float(np.clip(action[0], 0.0, 1.0))
                steer = float(np.clip(action[1], -1.0, 1.0))
                brake = 0.0

                # If action has a third component, use it as brake
                if len(action) >= 3:
                    brake = float(np.clip(action[2], 0.0, 1.0))

                self._send_control(throttle, steer, brake)

    def _handle_control(self, msg):
        """Handle control messages from DRL agent."""
        if hasattr(msg, 'data'):
            try:
                import json
                control_data = json.loads(msg.data)
                command = control_data.get("command", "")
                params = control_data.get("params", {})

                if command == "reset":
                    logger.info("Received reset command")
                    self._reset_episode(seed=params.get("seed"))
                elif command == "exit":
                    logger.info("Received exit command")
                    self.running = False
                else:
                    logger.warning(f"Unknown command: {command}")
            except Exception as e:
                logger.error(f"Failed to parse control message: {e}")

    def connect(self):
        """Connect to the CARLA server."""
        try:
            logger.info(f"Connecting to CARLA server at {self.args.host}:{self.args.port}")
            self.client = make_carla_client(self.args.host, self.args.port)
            self.client.set_timeout(self.args.timeout)

            logger.info("Connected to CARLA server")
            return True
        except TCPConnectionError as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            return False

    def setup_carla(self):
        """Set up the CARLA environment."""
        logger.info("Setting up CARLA environment")

        # Create a CarlaSettings object
        settings = CarlaSettings()

        # Set up synchronous mode
        settings.set(SynchronousMode=True,
                     SendNonPlayerAgentsInfo=True,
                     NumberOfVehicles=self.args.vehicles,
                     NumberOfPedestrians=self.args.pedestrians,
                     WeatherId=self.args.weather,
                     QualityLevel=self.args.quality)

        # Set up cameras
        # Main RGB camera
        camera0 = Camera('CameraRGB')
        camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        camera0.set_position(2.0, 0.0, 1.4)  # Forward position, car hood
        camera0.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera0)

        # Depth camera
        camera1 = Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
        camera1.set_position(2.0, 0.0, 1.4)
        camera1.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera1)

        # Semantic segmentation camera
        camera2 = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
        camera2.set_position(2.0, 0.0, 1.4)
        camera2.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera2)

        return settings

    def start_episode(self):
        """Start a new episode."""
        logger.info("Starting new episode")
        self.frame = 0

        # Setup CARLA settings
        settings = self.setup_carla()

        # Start new episode
        self.scene = self.client.load_settings(settings)
        self.client.start_episode(self.args.start_pos)

        # Get initial measurements
        self.measurements, sensor_data = self.client.read_data()

        # Process initial sensor data
        self._process_sensor_data(sensor_data)

        # Publish initial state
        self._publish_state()

        return True

    def _reset_episode(self, seed=None):
        """Reset the episode."""
        if seed is not None:
            logger.info(f"Resetting episode with seed {seed}")
            # CARLA doesn't have a direct seed setting, so we'll use this as a flag
        else:
            logger.info("Resetting episode")

        # Setup CARLA settings
        settings = self.setup_carla()

        # Restart episode
        self.scene = self.client.load_settings(settings)
        self.client.start_episode(self.args.start_pos)

        # Reset frame counter
        self.frame = 0

        # Get initial measurements
        self.measurements, sensor_data = self.client.read_data()

        # Process initial sensor data
        self._process_sensor_data(sensor_data)

        # Publish initial state
        self._publish_state()

    def _process_sensor_data(self, sensor_data):
        """Process sensor data from CARLA."""
        if 'CameraRGB' in sensor_data:
            self.rgb_image = to_rgb_array(sensor_data['CameraRGB'])

        if 'CameraDepth' in sensor_data:
            self.depth_image = sensor_data['CameraDepth']
            # Normalize depth for visualization
            depth_array = np.array(self.depth_image.data)
            depth_array = depth_array.reshape((self.depth_image.height, self.depth_image.width, 4))
            depth_array = depth_array[:, :, 0:3]  # Remove alpha channel
            normalized_depth = np.dot(depth_array, [1.0, 0.0, 0.0])  # Keep only the R channel
            normalized_depth = normalized_depth / np.max(normalized_depth) * 255 if np.max(normalized_depth) > 0 else normalized_depth
            self.depth_image = normalized_depth.astype(np.uint8)

        if 'CameraSegmentation' in sensor_data:
            self.seg_image = sensor_data['CameraSegmentation']
            # Process segmentation for visualization
            seg_array = np.array(self.seg_image.data)
            seg_array = seg_array.reshape((self.seg_image.height, self.seg_image.width, 4))
            self.seg_image = seg_array[:, :, 0:3].astype(np.uint8)  # Remove alpha channel

    def _publish_state(self):
        """Publish state information using ROS bridge."""
        if self.measurements is None:
            return

        # Extract state information from measurements
        player = self.measurements.player_measurements
        transform = player.transform
        control = player.autopilot_control

        # Basic state information
        state = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll,
            player.forward_speed,  # m/s
            player.collision_vehicles,
            player.collision_pedestrians,
            player.collision_other,
            float(player.intersection_otherlane),
            float(player.intersection_offroad),
            control.steer,
            control.throttle,
            control.brake,
            control.hand_brake,
            control.reverse
        ], dtype=np.float32)

        # Publish state via ROS bridge
        self.ros_bridge.publish_state(state)

        # Calculate reward (simple example)
        reward = player.forward_speed - \
                 10.0 * (player.collision_vehicles + player.collision_pedestrians + player.collision_other) - \
                 player.intersection_otherlane - player.intersection_offroad

        # Check if episode is done
        done = False
        if player.collision_vehicles + player.collision_pedestrians + player.collision_other > 0:
            done = True

        # Publish reward and done flag
        self.ros_bridge.publish_reward(reward, done)

        # Publish additional info
        info = {
            "frame": self.frame,
            "timestamp": self.measurements.game_timestamp,
            "platform_timestamp": self.measurements.platform_timestamp,
            "location": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z
            },
            "rotation": {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll
            },
            "speed": player.forward_speed,
            "collisions": {
                "vehicles": player.collision_vehicles,
                "pedestrians": player.collision_pedestrians,
                "other": player.collision_other
            },
            "intersection_otherlane": player.intersection_otherlane,
            "intersection_offroad": player.intersection_offroad
        }
        self.ros_bridge.publish_info(info)

    def _send_control(self, throttle, steer, brake, hand_brake=False, reverse=False):
        """Send control commands to CARLA."""
        control = VehicleControl()
        control.throttle = float(throttle)
        control.steer = float(steer)
        control.brake = float(brake)
        control.hand_brake = hand_brake
        control.reverse = reverse

        self.client.send_control(control)

    def update(self):
        """Update the visualizer with the latest data from CARLA."""
        if self.client is None:
            return False

        try:
            # Read data from CARLA
            self.measurements, sensor_data = self.client.read_data()

            # Process sensor data
            self._process_sensor_data(sensor_data)

            # Publish state
            self._publish_state()

            # Publish camera images if available
            if self.rgb_image is not None:
                self.ros_bridge.publish_camera(self.rgb_image)

            # Display images if OpenCV is available and display is enabled
            if HAS_CV2 and not self.args.no_display:
                if self.rgb_image is not None:
                    cv2.imshow('RGB Camera', cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR))

                if self.depth_image is not None:
                    cv2.imshow('Depth Camera', self.depth_image)

                if self.seg_image is not None:
                    cv2.imshow('Segmentation Camera', self.seg_image)

                # Process OpenCV events
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    return False

            # Check for actions from the DRL agent
            action = self.ros_bridge.get_latest_action()
            if action is not None:
                # Convert action to control command
                # Assuming action is [throttle, steering]
                if len(action) >= 2:
                    throttle = float(np.clip(action[0], 0.0, 1.0))
                    steer = float(np.clip(action[1], -1.0, 1.0))
                    brake = 0.0

                    # If action has a third component, use it as brake
                    if len(action) >= 3:
                        brake = float(np.clip(action[2], 0.0, 1.0))

                    self._send_control(throttle, steer, brake)

            # Increment frame counter
            self.frame += 1

            return True
        except Exception as e:
            logger.error(f"Error during update: {e}")
            return False

    def run(self):
        """Run the visualizer."""
        logger.info("Running CARLA Camera Visualizer")

        if not self.connect():
            return False

        if not self.start_episode():
            return False

        logger.info("Starting main loop")
        try:
            while self.running:
                if not self.update():
                    break

                # Sleep to control the update rate
                time.sleep(0.01)  # 100 Hz max update rate
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up")

        # Close OpenCV windows
        if HAS_CV2 and not self.args.no_display:
            cv2.destroyAllWindows()

        # Shutdown ROS bridge
        if self.ros_bridge is not None:
            self.ros_bridge.shutdown()


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CARLA Camera Visualizer")
    parser.add_argument('--host', default=DEFAULT_HOST, help='CARLA server host')
    parser.add_argument('--port', default=DEFAULT_PORT, type=int, help='CARLA server port')
    parser.add_argument('--timeout', default=DEFAULT_TIMEOUT, type=float, help='CARLA server timeout')
    parser.add_argument('--quality', default='Low', choices=['Low', 'Epic'], help='Graphics quality')
    parser.add_argument('--start-pos', default=0, type=int, help='Player start position')
    parser.add_argument('--vehicles', default=0, type=int, help='Number of vehicles')
    parser.add_argument('--pedestrians', default=0, type=int, help='Number of pedestrians')
    parser.add_argument('--weather', default=0, type=int, help='Weather preset ID')
    parser.add_argument('--no-display', action='store_true', help='Disable OpenCV visualization')
    parser.add_argument('--no-ros', action='store_true', help='Disable ROS bridge')

    args = parser.parse_args()

    # Run the visualizer
    visualizer = CARLACameraVisualizer(args)
    success = visualizer.run()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

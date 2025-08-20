"""
Enhanced CARLA Client for DRL Integration with Real-time Visualization
=======================================================================

This module enhances the existing module_7.py functionality with:
- ROS 2 bridge integration via ZeroMQ
- Real-time camera visualization using cv2.imshow
- Sensor data processing and publishing
- Vehicle control command handling
- Episode management and reward calculation

Author: CARLA DRL Team
Date: August 2025
Python Version: 3.6 (Required for CARLA 0.8.4)
"""

from __future__ import print_function, division
import sys
import os
import time
import json
import logging
import threading
import queue
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# CARLA imports - Updated paths for 0.8.4
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CarlaSimulator', 'PythonClient')))
from carla.client import make_carla_client, VehicleControl
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.image_converter import to_rgb_array
from carla.util import print_over_same_line

# Computer vision and communication
import cv2
import numpy as np
import zmq
try:
    import msgpack
except ImportError:
    print("Warning: msgpack not available, using JSON fallback")
    msgpack = None

# Configuration
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
ZMQ_PUB_PORT = 5555
ZMQ_SUB_PORT = 5556
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30
WEATHER_PRESETS = {
    'CLEAR': 1,
    'CLOUDY': 2,
    'WET': 3,
    'HARD_RAIN': 6,
    'SOFT_RAIN': 7
}

# Configure logging for Python 3.6
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedCarlaClient:
    """
    Enhanced CARLA client that builds upon module_7.py functionality.

    Integrates with ROS 2 bridge for DRL training while maintaining
    compatibility with existing perception and control systems.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize enhanced CARLA client.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # CARLA components
        self.client = None
        self.game = None
        self.scene = None
        self.camera_rgb = None
        self.camera_depth = None
        self.measurements = None

        # Vehicle state
        self.autopilot = False
        self.control = VehicleControl()

        # Sensor data
        self.image_rgb = None
        self.image_depth = None
        self.sensor_data = {}
        self.data_lock = threading.Lock()

        # Episode management
        self.episode_start_time = None
        self.episode_steps = 0
        self.total_reward = 0.0
        self.collision_detected = False
        self.done = False

        # Communication
        self.zmq_context = None
        self.publisher_socket = None
        self.subscriber_socket = None
        self.running = False

        # Real-time visualization
        self.display_enabled = True
        self.window_names = {
            'rgb': 'CARLA Camera (RGB)',
            'depth': 'CARLA Camera (Depth)',
            'control': 'Vehicle Control Status'
        }

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'carla_host': 'localhost',
            'carla_port': 2000,
            'timeout': 10.0,
            'town': 'Town01',
            'weather': 1,  # Clear noon
            'vehicle_filter': 'vehicle.tesla.model3',
            'camera_width': 800,
            'camera_height': 600,
            'camera_fov': 90.0,
            'zmq_image_port': 5555,
            'zmq_control_port': 5556,
            'display_sensors': True,
            'sync_mode': True,
            'fps': 30
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config.get('carla', {}))
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        return default_config

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def setup_communication(self) -> bool:
        """
        Setup ZeroMQ communication for ROS 2 bridge.

        Returns:
            bool: True if setup successful
        """
        try:
            self.zmq_context = zmq.Context()

            # Publisher for sensor data (to ROS 2)
            self.publisher_socket = self.zmq_context.socket(zmq.PUB)
            self.publisher_socket.bind(f"tcp://*:{self.config['zmq_image_port']}")

            # Subscriber for control commands (from ROS 2)
            self.subscriber_socket = self.zmq_context.socket(zmq.SUB)
            self.subscriber_socket.connect(f"tcp://localhost:{self.config['zmq_control_port']}")
            self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "control")
            self.subscriber_socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout

            logger.info("ZeroMQ communication setup successful")
            return True

        except Exception as e:
            logger.error(f"Failed to setup communication: {e}")
            return False

    def connect_to_carla(self) -> bool:
        """
        Connect to CARLA server and setup simulation.

        Returns:
            bool: True if connection successful
        """
        try:
            logger.info("Connecting to CARLA server...")

            # Use context manager pattern like in module_7.py
            with make_carla_client(
                self.config['carla_host'],
                self.config['carla_port']
            ) as client:

                # Store client reference
                self.client = client

                # Create settings
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=self.config['sync_mode'],
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=2,
                    NumberOfPedestrians=0,
                    WeatherId=self.config['weather'],
                    QualityLevel='Low'  # For better performance
                )

                # Setup cameras
                camera_rgb = Camera('CameraRGB')
                camera_rgb.set_image_size(
                    self.config['camera_width'],
                    self.config['camera_height']
                )
                camera_rgb.set_field_of_view(self.config['camera_fov'])
                camera_rgb.set_position(2.0, 0.0, 1.4)  # Front bumper position
                camera_rgb.set_rotation(0.0, 0.0, 0.0)
                settings.add_sensor(camera_rgb)

                camera_depth = Camera('CameraDepth', PostProcessing='Depth')
                camera_depth.set_image_size(
                    self.config['camera_width'],
                    self.config['camera_height']
                )
                camera_depth.set_field_of_view(self.config['camera_fov'])
                camera_depth.set_position(2.0, 0.0, 1.4)
                camera_depth.set_rotation(0.0, 0.0, 0.0)
                settings.add_sensor(camera_depth)

                # Load settings and start episode
                scene = client.load_settings(settings)
                client.start_episode(0)  # Random start position

                logger.info("CARLA connection established successfully")
                return True

        except Exception as e:
            logger.error(f"Unexpected error connecting to CARLA: {e}")
            return False

            logger.info(f"Connected to CARLA - Town: {self.config['town']}")
            return True

        except TCPConnectionError as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to CARLA: {e}")
            return False

    def setup_sensors(self) -> None:
        """Setup sensor data processing."""
        if self.scene:
            self.camera_rgb = None
            self.camera_depth = None

            for name, measurement in self.scene.sensor_data.items():
                if 'CameraRGB' in name:
                    self.camera_rgb = measurement
                elif 'CameraDepth' in name:
                    self.camera_depth = measurement

    def process_measurements(self) -> Dict[str, Any]:
        """
        Process current measurements and sensor data.

        Returns:
            dict: Processed measurements
        """
        if not self.scene:
            return {}

        try:
            # Get measurements
            measurements = self.scene.player_measurements

            # Process vehicle state
            vehicle_state = {
                'position': {
                    'x': measurements.transform.location.x,
                    'y': measurements.transform.location.y,
                    'z': measurements.transform.location.z
                },
                'rotation': {
                    'pitch': measurements.transform.rotation.pitch,
                    'yaw': measurements.transform.rotation.yaw,
                    'roll': measurements.transform.rotation.roll
                },
                'velocity': {
                    'forward': measurements.forward_speed,
                    'x': getattr(measurements, 'velocity_x', 0.0),
                    'y': getattr(measurements, 'velocity_y', 0.0)
                },
                'collision': measurements.collision_vehicles + measurements.collision_pedestrians + measurements.collision_other > 0,
                'lane_invasion': measurements.intersection_offroad > 0.5 or measurements.intersection_otherlane > 0.5
            }

            # Process camera data
            sensor_data = {}

            if self.camera_rgb is not None:
                rgb_array = to_rgb_array(self.camera_rgb)
                sensor_data['camera_rgb'] = rgb_array

            if self.camera_depth is not None:
                depth_array = to_rgb_array(self.camera_depth)
                sensor_data['camera_depth'] = depth_array

            # Update internal state
            with self.data_lock:
                self.sensor_data = sensor_data
                self.collision_detected = vehicle_state['collision']

            # Calculate reward
            reward = self._calculate_reward(vehicle_state, measurements)

            return {
                'vehicle_state': vehicle_state,
                'sensor_data': sensor_data,
                'reward': reward,
                'measurements': measurements,
                'episode_steps': self.episode_steps,
                'total_reward': self.total_reward,
                'done': self._check_done(vehicle_state, measurements)
            }

        except Exception as e:
            logger.error(f"Error processing measurements: {e}")
            return {}

    def _calculate_reward(self, vehicle_state: Dict, measurements) -> float:
        """
        Calculate reward based on driving performance.

        Args:
            vehicle_state: Current vehicle state
            measurements: CARLA measurements

        Returns:
            float: Calculated reward
        """
        reward = 0.0

        # Forward progress reward
        forward_speed = vehicle_state['velocity']['forward']
        if forward_speed > 0:
            reward += min(forward_speed / 10.0, 1.0)  # Normalize to max 1.0

        # Lane keeping reward
        if not vehicle_state['lane_invasion']:
            reward += 0.5

        # Collision penalty
        if vehicle_state['collision']:
            reward -= 10.0

        # Speed maintenance (penalize too slow or too fast)
        target_speed = 8.0  # m/s (~30 km/h)
        speed_diff = abs(forward_speed - target_speed)
        reward -= speed_diff * 0.1

        self.total_reward += reward
        return reward

    def _check_done(self, vehicle_state: Dict, measurements) -> bool:
        """
        Check if episode should terminate.

        Args:
            vehicle_state: Current vehicle state
            measurements: CARLA measurements

        Returns:
            bool: True if episode should end
        """
        # Collision
        if vehicle_state['collision']:
            logger.info("Episode ended: Collision detected")
            return True

        # Off-road for too long
        if measurements.intersection_offroad > 0.8:
            logger.info("Episode ended: Vehicle off-road")
            return True

        # Maximum episode length
        max_steps = 3000  # ~100 seconds at 30 FPS
        if self.episode_steps >= max_steps:
            logger.info("Episode ended: Maximum steps reached")
            return True

        # Goal reached (if waypoints available)
        # This would need integration with waypoint system from module_7.py

        return False

    def publish_sensor_data(self, data: Dict[str, Any]) -> None:
        """
        Publish sensor data via ZeroMQ to ROS 2 bridge.

        Args:
            data: Sensor and vehicle data to publish
        """
        if not self.publisher_socket:
            return

        try:
            # Prepare message
            message = {
                'timestamp': time.time(),
                'frame_id': self.frame_count,
                'vehicle_state': data.get('vehicle_state', {}),
                'reward': data.get('reward', 0.0),
                'done': data.get('done', False),
                'episode_steps': self.episode_steps
            }

            # Add camera images if available
            if 'sensor_data' in data:
                sensor_data = data['sensor_data']
                if 'camera_rgb' in sensor_data:
                    # Encode image for transmission
                    rgb_img = sensor_data['camera_rgb']
                    _, encoded = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    message['camera_rgb'] = encoded.tobytes()
                    message['image_shape'] = rgb_img.shape

                if 'camera_depth' in sensor_data:
                    depth_img = sensor_data['camera_depth']
                    _, encoded = cv2.imencode('.png', depth_img)
                    message['camera_depth'] = encoded.tobytes()

            # Serialize and send
            packed_data = msgpack.packb(message, use_bin_type=True)
            self.publisher_socket.send_multipart([b"sensor_data", packed_data])

        except Exception as e:
            logger.error(f"Failed to publish sensor data: {e}")

    def receive_control_commands(self) -> Optional[VehicleControl]:
        """
        Receive control commands from ROS 2 bridge.

        Returns:
            VehicleControl: Control commands or None if no message
        """
        if not self.subscriber_socket:
            return None

        try:
            # Non-blocking receive
            topic, data = self.subscriber_socket.recv_multipart(zmq.NOBLOCK)
            message = msgpack.unpackb(data, raw=False)

            # Create control command
            control = VehicleControl()
            control.throttle = max(0.0, min(1.0, message.get('throttle', 0.0)))
            control.brake = max(0.0, min(1.0, message.get('brake', 0.0)))
            control.steer = max(-1.0, min(1.0, message.get('steer', 0.0)))
            control.hand_brake = message.get('hand_brake', False)
            control.reverse = message.get('reverse', False)

            return control

        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            logger.error(f"Failed to receive control commands: {e}")
            return None

    def display_sensor_data(self, data: Dict[str, Any]) -> None:
        """
        Display sensor data using OpenCV for real-time visualization.

        Args:
            data: Sensor data to display
        """
        if not self.display_enabled or 'sensor_data' not in data:
            return

        try:
            sensor_data = data['sensor_data']

            # Display RGB camera
            if 'camera_rgb' in sensor_data:
                rgb_img = sensor_data['camera_rgb']

                # Add overlay information
                overlay_img = rgb_img.copy()

                # Add FPS counter
                cv2.putText(overlay_img, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add episode info
                episode_text = f"Episode Step: {self.episode_steps}"
                cv2.putText(overlay_img, episode_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add reward info
                reward_text = f"Total Reward: {self.total_reward:.2f}"
                cv2.putText(overlay_img, reward_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Add vehicle status
                vehicle_state = data.get('vehicle_state', {})
                speed = vehicle_state.get('velocity', {}).get('forward', 0.0)
                speed_text = f"Speed: {speed:.1f} m/s"
                cv2.putText(overlay_img, speed_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show collision warning
                if vehicle_state.get('collision', False):
                    cv2.putText(overlay_img, "COLLISION!", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                cv2.imshow(self.window_names['rgb'], overlay_img)

            # Display depth camera (optional)
            if 'camera_depth' in sensor_data:
                depth_img = sensor_data['camera_depth']
                cv2.imshow(self.window_names['depth'], depth_img)

            # Handle window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord(' '):
                self.autopilot = not self.autopilot
                logger.info(f"Autopilot: {'ON' if self.autopilot else 'OFF'}")

        except Exception as e:
            logger.error(f"Error displaying sensor data: {e}")

    def update_fps(self) -> None:
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        self.frame_count += 1

    def reset_episode(self) -> Dict[str, Any]:
        """
        Reset episode and return initial observation.

        Returns:
            dict: Initial observation data
        """
        logger.info("Resetting episode...")

        # Reset episode state
        self.episode_steps = 0
        self.total_reward = 0.0
        self.collision_detected = False
        self.done = False
        self.episode_start_time = time.time()

        # Reset vehicle position (restart episode)
        if self.client:
            try:
                self.scene = self.client.start_episode(0)  # Random position
                time.sleep(0.5)  # Allow simulation to stabilize
            except Exception as e:
                logger.error(f"Failed to reset episode: {e}")

        # Get initial observation
        initial_data = self.process_measurements()
        return initial_data

    def step(self, control_override: VehicleControl = None) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Args:
            control_override: Optional control override

        Returns:
            dict: Step observation data
        """
        if not self.scene:
            return {}

        try:
            # Determine control command
            if control_override:
                control = control_override
            else:
                # Check for external control commands
                received_control = self.receive_control_commands()
                if received_control:
                    control = received_control
                elif self.autopilot:
                    control = self.scene.player_measurements.autopilot_control
                else:
                    control = self.control  # Use default/last control

            # Send control command
            self.client.send_control(control)

            # Process frame
            data = self.process_measurements()

            # Update state
            self.episode_steps += 1
            self.done = data.get('done', False)

            # Publish data
            self.publish_sensor_data(data)

            # Display visualization
            self.display_sensor_data(data)

            # Update performance metrics
            self.update_fps()

            return data

        except Exception as e:
            logger.error(f"Error during step: {e}")
            return {'done': True}

    def run_episode(self, max_steps: int = 3000) -> None:
        """
        Run a complete episode.

        Args:
            max_steps: Maximum steps per episode
        """
        logger.info("Starting new episode...")

        # Reset episode
        data = self.reset_episode()

        # Episode loop
        step_count = 0
        while self.running and not self.done and step_count < max_steps:
            data = self.step()

            if data.get('done', False):
                break

            step_count += 1

            # Synchronous mode tick
            if self.config['sync_mode']:
                time.sleep(1.0 / self.config['fps'])

        logger.info(f"Episode completed - Steps: {step_count}, Total Reward: {self.total_reward:.2f}")

    def cleanup(self) -> None:
        """Cleanup resources gracefully."""
        logger.info("Cleaning up CARLA client...")

        self.running = False

        # Close OpenCV windows
        cv2.destroyAllWindows()

        # Close ZeroMQ sockets
        if self.publisher_socket:
            self.publisher_socket.close()
        if self.subscriber_socket:
            self.subscriber_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        # Disconnect from CARLA
        if self.client:
            try:
                self.client.disconnect()
            except:
                pass

        logger.info("Cleanup completed")

    def run(self) -> None:
        """Main execution loop."""
        logger.info("Starting Enhanced CARLA Client...")

        try:
            # Setup communication
            if not self.setup_communication():
                logger.error("Failed to setup communication")
                return

            # Connect to CARLA
            if not self.connect_to_carla():
                logger.error("Failed to connect to CARLA")
                return

            # Setup sensors
            self.setup_sensors()

            self.running = True
            logger.info("Enhanced CARLA Client ready - Running episodes...")

            # Run episodes continuously
            episode_count = 0
            while self.running:
                try:
                    episode_count += 1
                    logger.info(f"Starting Episode {episode_count}")

                    self.run_episode()

                    if not self.running:
                        break

                    # Brief pause between episodes
                    time.sleep(2.0)

                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in episode {episode_count}: {e}")
                    time.sleep(5.0)  # Wait before retry

        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced CARLA Client')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--host', type=str, default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--town', type=str, default='Town01', help='CARLA town')
    parser.add_argument('--weather', type=int, default=1, help='Weather ID')
    parser.add_argument('--autopilot', action='store_true', help='Enable autopilot')
    parser.add_argument('--no-display', action='store_true', help='Disable display')

    args = parser.parse_args()

    # Create client
    client = EnhancedCarlaClient(args.config)

    # Override config with command line arguments
    if args.host:
        client.config['carla_host'] = args.host
    if args.port:
        client.config['carla_port'] = args.port
    if args.town:
        client.config['town'] = args.town
    if args.weather:
        client.config['weather'] = args.weather
    if args.autopilot:
        client.autopilot = True
    if args.no_display:
        client.display_enabled = False

    # Run client
    try:
        client.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.cleanup()


if __name__ == '__main__':
    main()

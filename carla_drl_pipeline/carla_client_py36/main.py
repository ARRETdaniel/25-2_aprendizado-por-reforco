#!/usr/bin/env python3
"""
CARLA Client for DRL Pipeline (Python 3.6 Compatible)

This module provides a CARLA client that connects to the CARLA simulator,
manages sensors and vehicle control, and communicates with the ROS 2 gateway
through a high-performance IPC bridge.

Based on the existing module_7.py implementation with enhanced DRL integration.
"""

from __future__ import print_function
from __future__ import division

import os
import sys
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add CARLA Python API to path
CARLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(CARLA_ROOT, 'CarlaSimulator', 'PythonClient'))

try:
    import numpy as np
    from carla.client import make_carla_client, VehicleControl
    from carla.settings import CarlaSettings
    from carla.sensor import Camera, Lidar
    from carla.tcp import TCPConnectionError
    from carla.image_converter import to_rgb_array, to_bgra_array
    from carla.util import print_over_same_line
    logger.info("CARLA modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import CARLA modules: {e}")
    logger.error("Ensure CARLA Python API is properly installed")
    sys.exit(1)

# Import local modules
try:
    from sensor_manager import SensorManager
    from vehicle_controller import VehicleController
    from communication_bridge import CommunicationBridge
    logger.info("Local modules imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import local modules: {e}")
    logger.warning("Some functionality may be limited")


class CarlaClientConfig:
    """Configuration for CARLA client."""
    
    def __init__(self):
        # CARLA Server Settings
        self.host = "localhost"
        self.port = 2000
        self.timeout = 10.0
        self.quality_level = "Low"
        self.synchronous_mode = True
        self.fixed_delta_seconds = 0.033  # 30 FPS
        
        # Environment Settings
        self.town = "Town01"
        self.weather_id = 1
        self.number_of_vehicles = 15
        self.number_of_pedestrians = 10
        
        # Vehicle Settings
        self.player_start_index = 0
        self.autopilot_enabled = False
        
        # Sensor Settings
        self.camera_rgb_enabled = True
        self.camera_depth_enabled = True
        self.lidar_enabled = False
        self.image_width = 800
        self.image_height = 600
        
        # Communication Settings
        self.bridge_enabled = True
        self.bridge_address = "tcp://localhost:5555"
        
        # Logging Settings
        self.log_level = "INFO"
        self.save_images = False
        self.display_camera = True


class CarlaClient:
    """Enhanced CARLA client for DRL pipeline."""
    
    def __init__(self, config: CarlaClientConfig):
        """Initialize CARLA client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.client = None
        self.sensor_manager = None
        self.vehicle_controller = None
        self.communication_bridge = None
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_start_time = None
        self.last_measurement = None
        
        # Control flags
        self.running = False
        self.episode_done = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        
        logger.info("CARLA client initialized")
    
    def connect(self) -> bool:
        """Connect to CARLA server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to CARLA server at {self.config.host}:{self.config.port}")
            
            # Create CARLA client
            self.client = make_carla_client(self.config.host, self.config.port)
            self.client.connect(timeout=self.config.timeout)
            
            # Initialize components
            self._initialize_components()
            
            logger.info("Successfully connected to CARLA server")
            return True
            
        except TCPConnectionError as e:
            logger.error(f"Failed to connect to CARLA server: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize client components."""
        # Initialize sensor manager
        self.sensor_manager = SensorManager(self.config)
        
        # Initialize vehicle controller
        self.vehicle_controller = VehicleController(self.config)
        
        # Initialize communication bridge if enabled
        if self.config.bridge_enabled:
            self.communication_bridge = CommunicationBridge(
                address=self.config.bridge_address
            )
            self.communication_bridge.connect()
        
        logger.info("Client components initialized")
    
    def setup_episode(self) -> bool:
        """Setup a new episode.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info(f"Setting up episode {self.episode_count}")
            
            # Create CARLA settings
            settings = self._create_carla_settings()
            
            # Load settings and start episode
            scene = self.client.load_settings(settings)
            
            # Select player start position
            if self.config.player_start_index >= 0:
                player_start = self.config.player_start_index
            else:
                import random
                player_start = random.randint(0, len(scene.player_start_spots) - 1)
            
            # Start episode
            self.client.start_episode(player_start)
            
            # Reset tracking variables
            self.step_count = 0
            self.episode_start_time = time.time()
            self.episode_done = False
            
            logger.info(f"Episode {self.episode_count} started at position {player_start}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup episode: {e}")
            return False
    
    def _create_carla_settings(self) -> CarlaSettings:
        """Create CARLA settings for the episode.
        
        Returns:
            CarlaSettings object
        """
        settings = CarlaSettings()
        
        # Basic settings
        settings.set(
            SynchronousMode=self.config.synchronous_mode,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=self.config.number_of_vehicles,
            NumberOfPedestrians=self.config.number_of_pedestrians,
            WeatherId=self.config.weather_id,
            QualityLevel=self.config.quality_level
        )
        
        if self.config.synchronous_mode:
            settings.set(FixedDeltaSeconds=self.config.fixed_delta_seconds)
        
        # Add sensors
        if self.config.camera_rgb_enabled:
            camera_rgb = Camera('CameraRGB')
            camera_rgb.set_image_size(self.config.image_width, self.config.image_height)
            camera_rgb.set_position(2.0, 0.0, 1.4)
            camera_rgb.set_rotation(0.0, 0.0, 0.0)
            settings.add_sensor(camera_rgb)
        
        if self.config.camera_depth_enabled:
            camera_depth = Camera('CameraDepth', PostProcessing='Depth')
            camera_depth.set_image_size(self.config.image_width, self.config.image_height)
            camera_depth.set_position(2.0, 0.0, 1.4)
            camera_depth.set_rotation(0.0, 0.0, 0.0)
            settings.add_sensor(camera_depth)
        
        if self.config.lidar_enabled:
            lidar = Lidar('Lidar32')
            lidar.set_position(0.0, 0.0, 2.5)
            lidar.set_rotation(0.0, 0.0, 0.0)
            lidar.set(
                Channels=32,
                Range=50.0,
                PointsPerSecond=56000,
                RotationFrequency=10.0,
                UpperFovLimit=10.0,
                LowerFovLimit=-30.0
            )
            settings.add_sensor(lidar)
        
        # Randomize seeds for reproducibility
        settings.randomize_seeds()
        
        return settings
    
    def step(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute one simulation step.
        
        Returns:
            Tuple of (step_success, step_data)
        """
        try:
            # Read sensor data and measurements
            measurements, sensor_data = self.client.read_data()
            self.last_measurement = measurements
            
            # Process sensor data
            processed_data = self._process_sensor_data(sensor_data, measurements)
            
            # Handle vehicle control
            control_command = self._get_control_command()
            if control_command:
                self.client.send_control(control_command)
            
            # Update step counter
            self.step_count += 1
            
            # Update FPS counter
            self._update_fps()
            
            # Check episode termination
            episode_info = self._check_episode_termination(measurements)
            processed_data.update(episode_info)
            
            return True, processed_data
            
        except Exception as e:
            logger.error(f"Error during step: {e}")
            return False, {}
    
    def _process_sensor_data(self, sensor_data: Dict, measurements) -> Dict[str, Any]:
        """Process sensor data and measurements.
        
        Args:
            sensor_data: Raw sensor data from CARLA
            measurements: Vehicle measurements from CARLA
            
        Returns:
            Processed sensor data dictionary
        """
        processed_data = {
            'timestamp': time.time(),
            'episode': self.episode_count,
            'step': self.step_count
        }
        
        # Process camera data
        if 'CameraRGB' in sensor_data:
            rgb_image = to_rgb_array(sensor_data['CameraRGB'])
            processed_data['camera_rgb'] = rgb_image
            
            # Display camera if enabled
            if self.config.display_camera:
                self._display_image(rgb_image, 'RGB Camera')
        
        if 'CameraDepth' in sensor_data:
            depth_image = to_bgra_array(sensor_data['CameraDepth'])
            processed_data['camera_depth'] = depth_image
        
        # Process LiDAR data
        if 'Lidar32' in sensor_data:
            lidar_data = np.array(sensor_data['Lidar32'].data)
            processed_data['lidar'] = lidar_data
        
        # Process vehicle measurements
        player_measurements = measurements.player_measurements
        processed_data.update({
            'position': {
                'x': player_measurements.transform.location.x,
                'y': player_measurements.transform.location.y,
                'z': player_measurements.transform.location.z
            },
            'rotation': {
                'pitch': player_measurements.transform.rotation.pitch,
                'yaw': player_measurements.transform.rotation.yaw,
                'roll': player_measurements.transform.rotation.roll
            },
            'velocity': {
                'forward': player_measurements.forward_speed,
                'acceleration': {
                    'x': player_measurements.acceleration.x,
                    'y': player_measurements.acceleration.y,
                    'z': player_measurements.acceleration.z
                }
            },
            'collision': {
                'vehicles': player_measurements.collision_vehicles,
                'pedestrians': player_measurements.collision_pedestrians,
                'other': player_measurements.collision_other
            },
            'lane_invasion': {
                'other_lane': player_measurements.intersection_otherlane,
                'off_road': player_measurements.intersection_offroad
            }
        })
        
        # Send data through communication bridge
        if self.communication_bridge:
            self.communication_bridge.publish_sensor_data(processed_data)
        
        return processed_data
    
    def _get_control_command(self) -> Optional[VehicleControl]:
        """Get vehicle control command.
        
        Returns:
            VehicleControl command or None
        """
        if self.communication_bridge:
            # Get control from bridge (DRL agent)
            control_data = self.communication_bridge.get_control_command()
            if control_data:
                control = VehicleControl()
                control.throttle = float(control_data.get('throttle', 0.0))
                control.brake = float(control_data.get('brake', 0.0))
                control.steer = float(control_data.get('steer', 0.0))
                control.hand_brake = bool(control_data.get('hand_brake', False))
                control.reverse = bool(control_data.get('reverse', False))
                return control
        
        # Default: no control (autopilot or manual)
        if self.config.autopilot_enabled and self.last_measurement:
            return self.last_measurement.player_measurements.autopilot_control
        
        return None
    
    def _check_episode_termination(self, measurements) -> Dict[str, Any]:
        """Check if episode should terminate.
        
        Args:
            measurements: Vehicle measurements
            
        Returns:
            Episode termination info
        """
        player_measurements = measurements.player_measurements
        
        # Check collision
        collision_detected = (
            player_measurements.collision_vehicles > 0 or
            player_measurements.collision_pedestrians > 0 or
            player_measurements.collision_other > 0
        )
        
        # Check time limit
        episode_time = time.time() - self.episode_start_time
        time_limit_reached = episode_time > 120.0  # 2 minutes max
        
        # Check step limit
        step_limit_reached = self.step_count > 2000
        
        # Determine if episode is done
        episode_done = collision_detected or time_limit_reached or step_limit_reached
        
        if episode_done and not self.episode_done:
            self.episode_done = True
            logger.info(f"Episode {self.episode_count} terminated - "
                       f"Collision: {collision_detected}, "
                       f"Time: {time_limit_reached}, "
                       f"Steps: {step_limit_reached}")
        
        return {
            'episode_done': episode_done,
            'collision_detected': collision_detected,
            'time_limit_reached': time_limit_reached,
            'step_limit_reached': step_limit_reached,
            'episode_time': episode_time
        }
    
    def _display_image(self, image: np.ndarray, window_name: str):
        """Display image using OpenCV if available.
        
        Args:
            image: Image array
            window_name: Window name for display
        """
        try:
            import cv2
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
        except ImportError:
            pass  # OpenCV not available
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_timer >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_timer)
            logger.debug(f"FPS: {fps:.1f}")
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def reset_episode(self):
        """Reset current episode."""
        self.episode_count += 1
        self.setup_episode()
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a complete episode.
        
        Returns:
            Episode statistics
        """
        if not self.setup_episode():
            return {}
        
        episode_data = []
        
        while not self.episode_done:
            success, step_data = self.step()
            if not success:
                break
            
            episode_data.append(step_data)
            
            # Check for reset command from bridge
            if self.communication_bridge and self.communication_bridge.should_reset():
                break
        
        # Calculate episode statistics
        episode_stats = self._calculate_episode_stats(episode_data)
        
        return episode_stats
    
    def _calculate_episode_stats(self, episode_data: List[Dict]) -> Dict[str, Any]:
        """Calculate episode statistics.
        
        Args:
            episode_data: List of step data
            
        Returns:
            Episode statistics
        """
        if not episode_data:
            return {}
        
        stats = {
            'episode_number': self.episode_count,
            'total_steps': len(episode_data),
            'episode_time': episode_data[-1]['episode_time'],
            'final_position': episode_data[-1]['position'],
            'collision_detected': episode_data[-1]['collision_detected'],
            'average_speed': np.mean([data['velocity']['forward'] for data in episode_data]),
            'max_speed': np.max([data['velocity']['forward'] for data in episode_data])
        }
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up CARLA client")
        
        if self.communication_bridge:
            self.communication_bridge.disconnect()
        
        if self.client:
            self.client.disconnect()
        
        # Close OpenCV windows
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass
        
        logger.info("Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CARLA DRL Client')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--quality', choices=['Low', 'Epic'], default='Low',
                       help='Graphics quality level')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--autopilot', action='store_true',
                       help='Enable autopilot')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Display camera feed')
    
    args = parser.parse_args()
    
    # Create configuration
    config = CarlaClientConfig()
    config.host = args.host
    config.port = args.port
    config.quality_level = args.quality
    config.autopilot_enabled = args.autopilot
    config.display_camera = args.display
    
    # Create and run client
    client = CarlaClient(config)
    
    try:
        if not client.connect():
            logger.error("Failed to connect to CARLA server")
            return 1
        
        logger.info(f"Running {args.episodes} episodes")
        
        for episode in range(args.episodes):
            logger.info(f"Starting episode {episode + 1}/{args.episodes}")
            episode_stats = client.run_episode()
            
            if episode_stats:
                logger.info(f"Episode completed: {episode_stats}")
            else:
                logger.warning("Episode failed or was interrupted")
        
        logger.info("All episodes completed")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        client.cleanup()


if __name__ == '__main__':
    sys.exit(main())

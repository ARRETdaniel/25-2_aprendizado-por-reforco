"""
CARLA Client for Python 3.6 - Main entry point for CARLA simulation.

This module connects to CARLA 0.8.4, spawns ego vehicle with sensors,
and publishes sensor data via ZeroMQ to the ROS 2 gateway.
"""
import sys
import os
import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List
import signal

# Python 3.6 compatible imports
try:
    import carla
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CarlaSimulator', 'PythonAPI', 'carla'))
    import carla

import zmq
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarlaClient:
    """
    CARLA client that manages simulation, sensors, and communication.
    
    Handles vehicle spawning, sensor attachment, and real-time data
    streaming to ROS 2 via ZeroMQ bridge.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CARLA client with configuration.
        
        Args:
            config: Dictionary containing simulation configuration
        """
        self.config = config
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.sensors: Dict[str, carla.Sensor] = {}
        self.sensor_data: Dict[str, Any] = {}
        self.zmq_context: Optional[zmq.Context] = None
        self.zmq_socket: Optional[zmq.Socket] = None
        self.running: bool = False
        self.data_lock = threading.Lock()
        
        # Vehicle control
        self.vehicle_control = carla.VehicleControl()
        
        # Episode management
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.episode_start_time = 0.0
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def connect_to_carla(self) -> bool:
        """
        Connect to CARLA server and load world.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Connect to CARLA server
            self.client = carla.Client(
                self.config['carla_host'], 
                self.config['carla_port']
            )
            self.client.set_timeout(self.config['timeout'])
            
            # Load specified town
            town_name = self.config['town']
            logger.info(f"Loading world: {town_name}")
            self.world = self.client.load_world(town_name)
            
            # Set synchronous mode
            if self.config['sync_mode']:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config['fixed_delta_seconds']
                self.world.apply_settings(settings)
                
            # Set weather
            weather_config = self.config['weather']
            weather = carla.WeatherParameters(
                cloudiness=weather_config['cloudiness'],
                precipitation=weather_config['precipitation'],
                sun_altitude_angle=weather_config['sun_altitude_angle'],
                fog_density=weather_config['fog_density']
            )
            self.world.set_weather(weather)
            
            logger.info("Successfully connected to CARLA")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False
            
    def spawn_ego_vehicle(self) -> bool:
        """
        Spawn ego vehicle in the world.
        
        Returns:
            bool: True if spawn successful, False otherwise
        """
        try:
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(self.config['blueprint'])
            
            # Set spawn point
            spawn_config = self.config['spawn_point']
            spawn_point = carla.Transform(
                carla.Location(
                    x=spawn_config['x'],
                    y=spawn_config['y'], 
                    z=spawn_config['z']
                ),
                carla.Rotation(yaw=spawn_config['yaw'])
            )
            
            # Spawn vehicle
            self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            if self.ego_vehicle is None:
                logger.error("Failed to spawn ego vehicle")
                return False
                
            logger.info(f"Spawned ego vehicle with ID: {self.ego_vehicle.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn ego vehicle: {e}")
            return False
            
    def attach_sensors(self) -> bool:
        """
        Attach all configured sensors to ego vehicle.
        
        Returns:
            bool: True if all sensors attached successfully
        """
        try:
            blueprint_library = self.world.get_blueprint_library()
            
            # RGB Camera
            camera_config = self.config['sensors']['camera']
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(camera_config['width']))
            camera_bp.set_attribute('image_size_y', str(camera_config['height']))
            camera_bp.set_attribute('fov', str(camera_config['fov']))
            camera_bp.set_attribute('sensor_tick', str(camera_config['sensor_tick']))
            
            camera_transform = carla.Transform(
                carla.Location(*camera_config['location']),
                carla.Rotation(*camera_config['rotation'])
            )
            
            camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.ego_vehicle
            )
            camera.listen(lambda data: self._on_camera_data(data))
            self.sensors['camera'] = camera
            
            # IMU Sensor
            imu_config = self.config['sensors']['imu']
            imu_bp = blueprint_library.find('sensor.other.imu')
            imu_bp.set_attribute('sensor_tick', str(imu_config['sensor_tick']))
            
            imu_transform = carla.Transform(
                carla.Location(*imu_config['location']),
                carla.Rotation(*imu_config['rotation'])
            )
            
            imu = self.world.spawn_actor(
                imu_bp, imu_transform, attach_to=self.ego_vehicle
            )
            imu.listen(lambda data: self._on_imu_data(data))
            self.sensors['imu'] = imu
            
            # Collision Sensor
            collision_config = self.config['sensors']['collision']
            collision_bp = blueprint_library.find('sensor.other.collision')
            
            collision_transform = carla.Transform(
                carla.Location(*collision_config['location'])
            )
            
            collision_sensor = self.world.spawn_actor(
                collision_bp, collision_transform, attach_to=self.ego_vehicle
            )
            collision_sensor.listen(lambda data: self._on_collision_data(data))
            self.sensors['collision'] = collision_sensor
            
            # Lane Invasion Sensor
            lane_config = self.config['sensors']['lane_invasion']
            lane_bp = blueprint_library.find('sensor.other.lane_invasion')
            
            lane_transform = carla.Transform(
                carla.Location(*lane_config['location'])
            )
            
            lane_sensor = self.world.spawn_actor(
                lane_bp, lane_transform, attach_to=self.ego_vehicle
            )
            lane_sensor.listen(lambda data: self._on_lane_invasion_data(data))
            self.sensors['lane_invasion'] = lane_sensor
            
            logger.info(f"Attached {len(self.sensors)} sensors to ego vehicle")
            return True
            
        except Exception as e:
            logger.error(f"Failed to attach sensors: {e}")
            return False
            
    def _on_camera_data(self, image) -> None:
        """Process camera sensor data."""
        try:
            # Convert CARLA image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA
            array = array[:, :, :3]  # Remove alpha, convert to BGR
            array = array[:, :, ::-1]  # BGR to RGB
            
            with self.data_lock:
                self.sensor_data['camera'] = {
                    'timestamp': image.timestamp,
                    'frame': image.frame,
                    'width': image.width,
                    'height': image.height,
                    'data': array.tolist()  # Convert to list for JSON serialization
                }
                
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")
            
    def _on_imu_data(self, imu_data) -> None:
        """Process IMU sensor data."""
        try:
            with self.data_lock:
                self.sensor_data['imu'] = {
                    'timestamp': imu_data.timestamp,
                    'accelerometer': {
                        'x': imu_data.accelerometer.x,
                        'y': imu_data.accelerometer.y,
                        'z': imu_data.accelerometer.z
                    },
                    'gyroscope': {
                        'x': imu_data.gyroscope.x,
                        'y': imu_data.gyroscope.y,
                        'z': imu_data.gyroscope.z
                    },
                    'compass': imu_data.compass
                }
                
        except Exception as e:
            logger.error(f"Error processing IMU data: {e}")
            
    def _on_collision_data(self, collision_data) -> None:
        """Process collision sensor data."""
        try:
            self.collision_detected = True
            logger.warning("Collision detected!")
            
            with self.data_lock:
                self.sensor_data['collision'] = {
                    'timestamp': collision_data.timestamp,
                    'other_actor_id': collision_data.other_actor.id,
                    'other_actor_type': collision_data.other_actor.type_id,
                    'normal_impulse': {
                        'x': collision_data.normal_impulse.x,
                        'y': collision_data.normal_impulse.y,
                        'z': collision_data.normal_impulse.z
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing collision data: {e}")
            
    def _on_lane_invasion_data(self, lane_data) -> None:
        """Process lane invasion sensor data."""
        try:
            self.lane_invasion_detected = True
            logger.warning("Lane invasion detected!")
            
            with self.data_lock:
                self.sensor_data['lane_invasion'] = {
                    'timestamp': lane_data.timestamp,
                    'crossed_lane_markings': [
                        {
                            'type': marking.type.name,
                            'color': marking.color.name,
                            'lane_change': marking.lane_change.name
                        }
                        for marking in lane_data.crossed_lane_markings
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error processing lane invasion data: {e}")
            
    def setup_zmq_publisher(self) -> bool:
        """
        Setup ZeroMQ publisher for sensor data.
        
        Returns:
            bool: True if setup successful
        """
        try:
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            
            bind_address = self.config['communication']['zmq_bind_address']
            self.zmq_socket.bind(bind_address)
            
            logger.info(f"ZeroMQ publisher bound to: {bind_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup ZeroMQ publisher: {e}")
            return False
            
    def publish_sensor_data(self) -> None:
        """Publish current sensor data via ZeroMQ."""
        try:
            # Get vehicle state
            transform = self.ego_vehicle.get_transform()
            velocity = self.ego_vehicle.get_velocity()
            angular_velocity = self.ego_vehicle.get_angular_velocity()
            
            # Compile message
            with self.data_lock:
                message = {
                    'timestamp': time.time(),
                    'vehicle_state': {
                        'position': {
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
                            'z': velocity.z
                        },
                        'angular_velocity': {
                            'x': angular_velocity.x,
                            'y': angular_velocity.y,
                            'z': angular_velocity.z
                        }
                    },
                    'sensors': self.sensor_data.copy(),
                    'episode_status': {
                        'collision': self.collision_detected,
                        'lane_invasion': self.lane_invasion_detected,
                        'elapsed_time': time.time() - self.episode_start_time
                    }
                }
            
            # Send message
            json_message = json.dumps(message)
            self.zmq_socket.send_string(json_message, zmq.NOBLOCK)
            
        except zmq.Again:
            # Non-blocking send failed, skip this frame
            pass
        except Exception as e:
            logger.error(f"Error publishing sensor data: {e}")
            
    def apply_vehicle_control(self, throttle: float, steering: float, brake: float = 0.0) -> None:
        """
        Apply control commands to ego vehicle.
        
        Args:
            throttle: Throttle value [0.0, 1.0]
            steering: Steering value [-1.0, 1.0]
            brake: Brake value [0.0, 1.0]
        """
        try:
            self.vehicle_control.throttle = max(0.0, min(1.0, throttle))
            self.vehicle_control.steer = max(-1.0, min(1.0, steering))
            self.vehicle_control.brake = max(0.0, min(1.0, brake))
            
            self.ego_vehicle.apply_control(self.vehicle_control)
            
        except Exception as e:
            logger.error(f"Error applying vehicle control: {e}")
            
    def reset_episode(self) -> None:
        """Reset episode state."""
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.episode_start_time = time.time()
        
        # Reset vehicle to spawn point
        spawn_config = self.config['spawn_point']
        spawn_point = carla.Transform(
            carla.Location(
                x=spawn_config['x'],
                y=spawn_config['y'], 
                z=spawn_config['z']
            ),
            carla.Rotation(yaw=spawn_config['yaw'])
        )
        self.ego_vehicle.set_transform(spawn_point)
        
        # Clear sensor data
        with self.data_lock:
            self.sensor_data.clear()
            
        logger.info("Episode reset")
        
    def run_simulation(self) -> None:
        """Main simulation loop."""
        logger.info("Starting simulation loop")
        self.running = True
        self.episode_start_time = time.time()
        
        publish_rate = self.config['communication']['message_rate_hz']
        sleep_time = 1.0 / publish_rate
        
        try:
            while self.running:
                # Tick world in synchronous mode
                if self.config['sync_mode']:
                    self.world.tick()
                
                # Publish sensor data
                self.publish_sensor_data()
                
                # Check episode termination conditions
                if self.collision_detected or self.lane_invasion_detected:
                    logger.info("Episode terminated, resetting...")
                    self.reset_episode()
                
                # Sleep to maintain publishing rate
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
        finally:
            self.running = False
            
    def cleanup(self) -> None:
        """Clean up resources and destroy actors."""
        logger.info("Cleaning up CARLA client...")
        
        try:
            # Destroy sensors
            for sensor_name, sensor in self.sensors.items():
                if sensor is not None and sensor.is_alive:
                    sensor.destroy()
                    logger.info(f"Destroyed sensor: {sensor_name}")
            
            # Destroy ego vehicle
            if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
                self.ego_vehicle.destroy()
                logger.info("Destroyed ego vehicle")
            
            # Reset world settings if in sync mode
            if self.world is not None and self.config['sync_mode']:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                
            # Close ZeroMQ connection
            if self.zmq_socket is not None:
                self.zmq_socket.close()
                
            if self.zmq_context is not None:
                self.zmq_context.term()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'sim.yaml')
        config = load_config(config_path)
        
        # Create and initialize client
        client = CarlaClient(config['simulation'])
        
        # Connect to CARLA
        if not client.connect_to_carla():
            logger.error("Failed to connect to CARLA")
            return
            
        # Spawn ego vehicle
        if not client.spawn_ego_vehicle():
            logger.error("Failed to spawn ego vehicle")
            client.cleanup()
            return
            
        # Attach sensors
        if not client.attach_sensors():
            logger.error("Failed to attach sensors")
            client.cleanup()
            return
            
        # Setup ZeroMQ publisher
        if not client.setup_zmq_publisher():
            logger.error("Failed to setup ZeroMQ publisher")
            client.cleanup()
            return
            
        # Run simulation
        client.run_simulation()
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if 'client' in locals():
            client.cleanup()


if __name__ == "__main__":
    main()

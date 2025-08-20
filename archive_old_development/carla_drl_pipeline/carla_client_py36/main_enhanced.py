#!/usr/bin/env python3
"""
Enhanced CARLA Client with ROS 2 Integration
Building upon existing module_7.py with advanced DRL pipeline integration

Author: GitHub Copilot  
Date: 2025-01-26
Based on: CarlaSimulator/PythonClient/FinalProject/module_7.py
"""

import os
import sys
import glob
import time
import logging
import argparse
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import yaml

# Add CARLA Python API to path
try:
    carla_path = glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(carla_path)
except IndexError:
    # Fallback paths for different CARLA installations
    possible_paths = [
        'C:\\CARLA_0.8.4\\PythonAPI',
        'C:\\CARLA\\PythonAPI',
        '../carla/PythonAPI',
        './PythonAPI'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break

try:
    import carla
    print("✅ CARLA Python API loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import CARLA: {e}")
    sys.exit(1)

# ROS 2 Communication Bridge
from ros_communication import ROS2CommunicationBridge
from sensor_manager import EnhancedSensorManager
from yolo_integration import YOLODetectionIntegration
from performance_tracker import PerformanceTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carla_client_enhanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedCarlaClient:
    """Enhanced CARLA client with ROS 2 and DRL integration"""
    
    def __init__(self, config_path: str = None):
        """Initialize enhanced CARLA client"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors = {}
        self.sensor_data = {}
        
        # Communication and integration components
        self.ros_bridge = None
        self.sensor_manager = None
        self.yolo_detector = None
        self.performance_tracker = None
        
        # Control and state
        self.vehicle_control = carla.VehicleControl()
        self.episode_info = {
            'episode_id': 0,
            'step': 0,
            'total_reward': 0.0,
            'done': False,
            'info': {}
        }
        
        # Threading and synchronization
        self.is_running = False
        self.sensor_lock = threading.Lock()
        self.control_queue = queue.Queue(maxsize=10)
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        
        logger.info("Enhanced CARLA client initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load system configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'carla': {
                'server': {'host': '127.0.0.1', 'port': 2000, 'timeout': 10.0},
                'world': {'map': 'Town01', 'weather': 'ClearNoon', 'synchronous_mode': True, 'fixed_delta_seconds': 0.033},
                'vehicle': {'blueprint': 'vehicle.tesla.model3', 'spawn_point': 'random'},
                'sensors': {
                    'camera_rgb': {'enabled': True, 'width': 800, 'height': 600, 'fov': 90, 'position': [2.0, 0.0, 1.4]},
                    'camera_depth': {'enabled': True, 'width': 800, 'height': 600, 'fov': 90, 'position': [2.0, 0.0, 1.4]}
                }
            },
            'ros2': {
                'domain_id': 42,
                'topics': {
                    'camera_rgb': '/carla/ego_vehicle/camera/rgb/image_raw',
                    'vehicle_status': '/carla/ego_vehicle/vehicle_status',
                    'vehicle_control': '/carla/ego_vehicle/vehicle_control_cmd'
                }
            },
            'visualization': {
                'camera_display': {'enabled': True, 'update_rate': 30}
            }
        }
    
    def connect_to_carla(self) -> bool:
        """Connect to CARLA server"""
        try:
            logger.info("🔌 Connecting to CARLA server...")
            
            carla_config = self.config['carla']['server']
            self.client = carla.Client(carla_config['host'], carla_config['port'])
            self.client.set_timeout(carla_config['timeout'])
            
            # Test connection
            version = self.client.get_server_version()
            logger.info(f"✅ Connected to CARLA server version: {version}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to CARLA: {e}")
            return False
    
    def setup_world(self) -> bool:
        """Setup CARLA world and environment"""
        try:
            logger.info("🌍 Setting up CARLA world...")
            
            world_config = self.config['carla']['world']
            
            # Load map
            if world_config['map'] != 'current':
                self.client.load_world(world_config['map'])
            
            self.world = self.client.get_world()
            
            # Configure world settings
            settings = self.world.get_settings()
            settings.synchronous_mode = world_config['synchronous_mode']
            settings.fixed_delta_seconds = world_config['fixed_delta_seconds']
            settings.no_rendering_mode = False  # Keep rendering for visualization
            self.world.apply_settings(settings)
            
            # Set weather
            weather = getattr(carla.WeatherParameters, world_config['weather'])
            self.world.set_weather(weather)
            
            logger.info(f"✅ World setup complete: {world_config['map']} with {world_config['weather']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup world: {e}")
            return False
    
    def spawn_vehicle(self) -> bool:
        """Spawn ego vehicle in the world"""
        try:
            logger.info("🚗 Spawning ego vehicle...")
            
            vehicle_config = self.config['carla']['vehicle']
            
            # Get vehicle blueprint
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find(vehicle_config['blueprint'])
            
            # Set color (optional)
            if vehicle_bp.has_attribute('color'):
                color = '255,0,0'  # Red color for visibility
                vehicle_bp.set_attribute('color', color)
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            
            if vehicle_config['spawn_point'] == 'random':
                spawn_point = np.random.choice(spawn_points)
            else:
                spawn_point = spawn_points[0]  # Default to first spawn point
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            if self.vehicle is None:
                raise Exception("Failed to spawn vehicle")
            
            logger.info(f"✅ Vehicle spawned: {self.vehicle.type_id} at {spawn_point}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn vehicle: {e}")
            return False
    
    def setup_sensors(self) -> bool:
        """Setup and attach sensors to the vehicle"""
        try:
            logger.info("📷 Setting up sensors...")
            
            # Initialize sensor manager
            self.sensor_manager = EnhancedSensorManager(
                self.world, 
                self.vehicle,
                self.config['carla']['sensors']
            )
            
            # Setup sensors and attach to vehicle
            sensor_configs = self.config['carla']['sensors']
            
            for sensor_name, sensor_config in sensor_configs.items():
                if sensor_config.get('enabled', False):
                    sensor = self.sensor_manager.create_sensor(sensor_name, sensor_config)
                    if sensor:
                        self.sensors[sensor_name] = sensor
                        # Register callback for sensor data
                        sensor.listen(lambda data, name=sensor_name: self._on_sensor_data(name, data))
                        logger.info(f"  ✅ {sensor_name} sensor attached")
            
            logger.info(f"✅ {len(self.sensors)} sensors configured")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup sensors: {e}")
            return False
    
    def _on_sensor_data(self, sensor_name: str, data):
        """Handle incoming sensor data"""
        with self.sensor_lock:
            # Process sensor data based on type
            if 'camera' in sensor_name:
                processed_data = self.sensor_manager.process_camera_data(data)
                self.sensor_data[sensor_name] = processed_data
                
                # Display camera feed if visualization enabled
                if self.config['visualization']['camera_display']['enabled']:
                    self._display_camera_feed(sensor_name, processed_data)
            
            # Update frame count for performance tracking
            self.frame_count += 1
            
            # Publish to ROS 2 bridge
            if self.ros_bridge:
                self.ros_bridge.publish_sensor_data(sensor_name, data)
    
    def _display_camera_feed(self, sensor_name: str, image_data: np.ndarray):
        """Display camera feed in OpenCV window"""
        try:
            if image_data is not None and len(image_data.shape) == 3:
                # Resize for display
                display_image = cv2.resize(image_data, (400, 300))
                
                # Add sensor name overlay
                cv2.putText(display_image, sensor_name.upper(), (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add frame info
                fps = self._calculate_fps()
                cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display
                cv2.imshow(f"CARLA - {sensor_name}", display_image)
                cv2.waitKey(1)
                
        except Exception as e:
            logger.warning(f"Display error for {sensor_name}: {e}")
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0.0
    
    def setup_communication_bridge(self) -> bool:
        """Setup ROS 2 communication bridge"""
        try:
            logger.info("🌉 Setting up ROS 2 communication bridge...")
            
            # Initialize ROS 2 bridge
            self.ros_bridge = ROS2CommunicationBridge(self.config['ros2'])
            
            # Setup control command subscription
            self.ros_bridge.subscribe_vehicle_control(self._on_control_command)
            
            # Initialize YOLO detection integration
            self.yolo_detector = YOLODetectionIntegration()
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker()
            
            logger.info("✅ Communication bridge setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup communication bridge: {e}")
            return False
    
    def _on_control_command(self, control_msg):
        """Handle incoming vehicle control commands from DRL agent"""
        try:
            # Convert ROS control message to CARLA control
            control = carla.VehicleControl()
            
            # Map control values (assuming normalized [-1, 1] range)
            linear_x = control_msg.linear.x  # Forward velocity
            angular_z = control_msg.angular.z  # Steering
            
            # Convert to CARLA control format
            if linear_x >= 0:
                control.throttle = float(np.clip(linear_x, 0.0, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-linear_x, 0.0, 1.0))
            
            control.steer = float(np.clip(angular_z, -1.0, 1.0))
            control.hand_brake = False
            control.reverse = False
            
            # Queue control command for next simulation step
            if not self.control_queue.full():
                self.control_queue.put(control)
            
        except Exception as e:
            logger.error(f"❌ Control command processing error: {e}")
    
    def get_vehicle_state(self) -> Dict:
        """Get current vehicle state"""
        if not self.vehicle:
            return {}
        
        try:
            # Get vehicle transform and velocity
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            angular_velocity = self.vehicle.get_angular_velocity()
            acceleration = self.vehicle.get_acceleration()
            
            # Calculate derived values
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
            
            vehicle_state = {
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
                    'z': velocity.z,
                    'speed_kmh': speed
                },
                'acceleration': {
                    'x': acceleration.x,
                    'y': acceleration.y,
                    'z': acceleration.z
                },
                'angular_velocity': {
                    'x': angular_velocity.x,
                    'y': angular_velocity.y,
                    'z': angular_velocity.z
                }
            }
            
            return vehicle_state
            
        except Exception as e:
            logger.error(f"❌ Error getting vehicle state: {e}")
            return {}
    
    def detect_collisions(self) -> bool:
        """Check for vehicle collisions"""
        try:
            # Get collision sensor if available
            if hasattr(self, 'collision_sensor') and self.collision_sensor:
                # Check collision history
                return len(self.collision_sensor.get_collision_history()) > 0
            
            # Fallback: Check for nearby objects
            world_snapshot = self.world.get_snapshot()
            ego_location = self.vehicle.get_location()
            
            for actor_snapshot in world_snapshot:
                actor = self.world.get_actor(actor_snapshot.id)
                if actor and actor.id != self.vehicle.id:
                    distance = ego_location.distance(actor.get_location())
                    if distance < 2.0:  # 2 meter collision threshold
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Collision detection error: {e}")
            return False
    
    def calculate_reward(self) -> float:
        """Calculate reward for current state"""
        try:
            reward = 0.0
            vehicle_state = self.get_vehicle_state()
            
            if not vehicle_state:
                return -10.0  # Penalty for invalid state
            
            # Speed reward (encourage maintaining target speed)
            target_speed = 30.0  # km/h
            current_speed = vehicle_state['velocity']['speed_kmh']
            speed_diff = abs(current_speed - target_speed)
            speed_reward = max(0, 1.0 - speed_diff / target_speed)
            reward += speed_reward
            
            # Collision penalty
            if self.detect_collisions():
                reward -= 100.0
                self.episode_info['done'] = True
                logger.warning("💥 Collision detected!")
            
            # Lane keeping reward (simplified)
            # In a full implementation, this would use lane detection
            reward += 0.1  # Small baseline reward for staying alive
            
            return reward
            
        except Exception as e:
            logger.error(f"❌ Reward calculation error: {e}")
            return -1.0
    
    def step_simulation(self) -> bool:
        """Execute one simulation step"""
        try:
            # Apply queued control commands
            if not self.control_queue.empty():
                control = self.control_queue.get_nowait()
                self.vehicle.apply_control(control)
            
            # Tick the world (synchronous mode)
            if self.config['carla']['world']['synchronous_mode']:
                self.world.tick()
            
            # Update episode info
            self.episode_info['step'] += 1
            
            # Calculate reward
            step_reward = self.calculate_reward()
            self.episode_info['total_reward'] += step_reward
            
            # Publish vehicle state and reward
            if self.ros_bridge:
                vehicle_state = self.get_vehicle_state()
                self.ros_bridge.publish_vehicle_state(vehicle_state)
                self.ros_bridge.publish_reward(step_reward)
                
                # Publish episode info
                self.ros_bridge.publish_episode_info(self.episode_info)
            
            # Performance tracking
            if self.performance_tracker:
                self.performance_tracker.update_metrics({
                    'fps': self._calculate_fps(),
                    'episode_step': self.episode_info['step'],
                    'total_reward': self.episode_info['total_reward']
                })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Simulation step error: {e}")
            return False
    
    def reset_episode(self):
        """Reset episode for new training iteration"""
        try:
            logger.info(f"🔄 Resetting episode {self.episode_info['episode_id']}")
            
            # Reset vehicle position
            spawn_points = self.world.get_map().get_spawn_points()
            new_spawn_point = np.random.choice(spawn_points)
            self.vehicle.set_transform(new_spawn_point)
            
            # Reset vehicle control
            self.vehicle.apply_control(carla.VehicleControl())
            
            # Reset episode info
            self.episode_info = {
                'episode_id': self.episode_info['episode_id'] + 1,
                'step': 0,
                'total_reward': 0.0,
                'done': False,
                'info': {}
            }
            
            # Clear sensor data
            with self.sensor_lock:
                self.sensor_data.clear()
            
            # Clear control queue
            while not self.control_queue.empty():
                self.control_queue.get_nowait()
            
            # Publish reset signal
            if self.ros_bridge:
                self.ros_bridge.publish_episode_reset()
            
            logger.info(f"✅ Episode reset complete")
            
        except Exception as e:
            logger.error(f"❌ Episode reset error: {e}")
    
    def run_main_loop(self):
        """Main simulation loop"""
        logger.info("🚀 Starting main simulation loop...")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Execute simulation step
                if not self.step_simulation():
                    logger.error("Simulation step failed, stopping...")
                    break
                
                # Check episode termination
                max_steps = 1000  # Maximum steps per episode
                if (self.episode_info['done'] or 
                    self.episode_info['step'] >= max_steps):
                    
                    logger.info(f"Episode {self.episode_info['episode_id']} completed: "
                              f"{self.episode_info['step']} steps, "
                              f"reward: {self.episode_info['total_reward']:.2f}")
                    
                    self.reset_episode()
                
                # Small delay to prevent overloading
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("⚠️ Keyboard interrupt received")
        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        self.is_running = False
        
        # Destroy sensors
        for sensor in self.sensors.values():
            if sensor and sensor.is_alive:
                sensor.destroy()
        
        # Destroy vehicle
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        # Cleanup communication bridge
        if self.ros_bridge:
            self.ros_bridge.cleanup()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("✅ Cleanup complete")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced CARLA Client for DRL Training')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run')
    
    args = parser.parse_args()
    
    # Initialize enhanced CARLA client
    client = EnhancedCarlaClient(args.config)
    
    try:
        # Setup system components
        if not client.connect_to_carla():
            logger.error("Failed to connect to CARLA")
            return 1
        
        if not client.setup_world():
            logger.error("Failed to setup world")
            return 1
        
        if not client.spawn_vehicle():
            logger.error("Failed to spawn vehicle")
            return 1
        
        if not client.setup_sensors():
            logger.error("Failed to setup sensors")
            return 1
        
        if not client.setup_communication_bridge():
            logger.error("Failed to setup communication bridge")
            return 1
        
        logger.info("🎉 Enhanced CARLA client ready!")
        
        # Run main simulation loop
        client.run_main_loop()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    finally:
        client.cleanup()

if __name__ == "__main__":
    sys.exit(main())

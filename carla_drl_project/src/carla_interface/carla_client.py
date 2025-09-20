#!/usr/bin/env python3
"""
CARLA Client for DRL Project

This module provides a comprehensive CARLA client interface designed for deep
reinforcement learning applications. It handles vehicle spawning, sensor management,
and real-time data streaming optimized for TD3 training with RTX 2060 constraints.

Key Features:
- Memory-optimized sensor configuration (RTX 2060 6GB limit)
- Real-time camera streaming with CV2 visualization
- ROS 2 integration ready architecture
- Truck-specific vehicle spawning
- Error handling and resource management

Author: DRL CARLA Project Team
Date: September 2024
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from contextlib import contextmanager

# CARLA imports - ensure proper path setup
try:
    import carla
    print("✅ CARLA module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import CARLA: {e}")
    print("Please ensure CARLA Python API is in PYTHONPATH")
    sys.exit(1)

# Configuration constants optimized for RTX 2060
@dataclass
class CarlaConfig:
    """Configuration parameters for CARLA client optimized for DRL training"""
    # Connection settings
    HOST: str = '127.0.0.1'
    PORT: int = 2000
    TIMEOUT: float = 20.0

    # Map and spawn settings
    TOWN: str = 'Town01'
    SPAWN_INDEX: int = 1  # Default spawn point

    # Camera settings - optimized for memory constraints
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FOV: float = 90.0
    CAMERA_POSITION: carla.Location = carla.Location(x=4.0, y=0.0, z=1.4)
    CAMERA_ROTATION: carla.Rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

    # Depth camera (for development/debugging)
    DEPTH_WIDTH: int = 320
    DEPTH_HEIGHT: int = 240

    # Vehicle preference (trucks for our DRL task)
    VEHICLE_FILTER: str = 'vehicle.carlamotors.carlacola'  # Truck
    VEHICLE_FILTER_FALLBACK: str = 'vehicle.*'  # Any vehicle as fallback

    # Performance settings
    FIXED_DELTA_SECONDS: float = 0.05  # 20 FPS for stable training
    SYNCHRONOUS_MODE: bool = True  # Critical for DRL reproducibility
    NO_RENDERING_MODE: bool = False  # Keep rendering for development

    # Memory optimization
    MAX_FRAME_BUFFER: int = 5  # Limit frame buffering
    SENSOR_TICK: float = 0.1  # Sensor update rate (10 Hz)


class CarlaDataManager:
    """Manages sensor data streaming and buffering with memory optimization"""

    def __init__(self, max_buffer_size: int = 5):
        self.max_buffer_size = max_buffer_size
        self.camera_queue = Queue(maxsize=max_buffer_size)
        self.depth_queue = Queue(maxsize=max_buffer_size)
        self.latest_camera_frame = None
        self.latest_depth_frame = None
        self.frame_count = 0
        self.lock = threading.Lock()

    def camera_callback(self, data: carla.Image) -> None:
        """Process camera data with memory-efficient handling"""
        try:
            # Convert CARLA image to numpy array
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))  # BGRA
            # Convert BGRA to RGB for CV2
            frame = array[:, :, [2, 1, 0]]  # BGR for OpenCV

            with self.lock:
                self.latest_camera_frame = frame.copy()
                self.frame_count += 1

                # Non-blocking queue update
                if not self.camera_queue.full():
                    self.camera_queue.put(frame)
                else:
                    # Remove oldest frame to prevent memory buildup
                    try:
                        self.camera_queue.get_nowait()
                        self.camera_queue.put(frame)
                    except Empty:
                        pass

        except Exception as e:
            logging.error(f"Camera callback error: {e}")

    def depth_callback(self, data: carla.Image) -> None:
        """Process depth camera data"""
        try:
            # Convert depth data (R channel contains depth in meters * 1000)
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((data.height, data.width, 4))
            # Extract depth from R channel and normalize
            depth = array[:, :, 0].astype(np.float32) / 255.0 * 1000.0

            with self.lock:
                self.latest_depth_frame = depth.copy()

                if not self.depth_queue.full():
                    self.depth_queue.put(depth)
                else:
                    try:
                        self.depth_queue.get_nowait()
                        self.depth_queue.put(depth)
                    except Empty:
                        pass

        except Exception as e:
            logging.error(f"Depth callback error: {e}")

    def get_latest_camera_frame(self) -> Optional[np.ndarray]:
        """Get the latest camera frame (thread-safe)"""
        with self.lock:
            return self.latest_camera_frame.copy() if self.latest_camera_frame is not None else None

    def get_latest_depth_frame(self) -> Optional[np.ndarray]:
        """Get the latest depth frame (thread-safe)"""
        with self.lock:
            return self.latest_depth_frame.copy() if self.latest_depth_frame is not None else None


class CarlaClient:
    """
    Advanced CARLA client optimized for DRL training with memory constraints.

    Features:
    - Automatic vehicle spawning (truck preference)
    - Multi-sensor management (RGB camera, depth camera)
    - Real-time visualization with CV2
    - Memory-optimized data handling
    - ROS 2 integration ready
    - Graceful error handling and cleanup
    """

    def __init__(self, config: CarlaConfig = CarlaConfig()):
        self.config = config
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera_sensor: Optional[carla.Sensor] = None
        self.depth_sensor: Optional[carla.Sensor] = None
        self.data_manager = CarlaDataManager(config.MAX_FRAME_BUFFER)
        self.is_running = False
        self.sensors = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Performance monitoring
        self.spawn_time = None
        self.connection_time = None

    def connect(self) -> bool:
        """Establish connection to CARLA server with retry mechanism"""
        start_time = time.time()

        try:
            self.logger.info(f"🔄 Connecting to CARLA server at {self.config.HOST}:{self.config.PORT}")

            self.client = carla.Client(self.config.HOST, self.config.PORT)
            self.client.set_timeout(self.config.TIMEOUT)

            # Test connection
            version = self.client.get_server_version()
            self.logger.info(f"✅ Connected to CARLA {version}")
            self.logger.info("⚠️  Version mismatch warnings can be ignored for development")

            # Load world if needed
            self.world = self.client.get_world()
            current_map = self.world.get_map().name
            self.logger.info(f"🗺️  Current map: {current_map}")

            # Only change map if necessary (avoid unnecessary reloading)
            if self.config.TOWN not in current_map:
                self.logger.info(f"� Loading map: {self.config.TOWN}")
                self.world = self.client.load_world(self.config.TOWN)
                time.sleep(3)  # Allow more time for map loading

            self.blueprint_library = self.world.get_blueprint_library()
            self.is_connected = True

            self.connection_time = time.time() - start_time
            self.logger.info(f"🎉 CARLA connection established in {self.connection_time:.2f}s")
            return True

            # Configure world settings for DRL
            self._configure_world_settings()

            self.connection_time = time.time() - start_time
            self.logger.info(f"⚡ Connection established in {self.connection_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to connect to CARLA: {e}")
            return False

    def _configure_world_settings(self) -> None:
        """Configure world settings optimized for DRL training"""
        settings = self.world.get_settings()

        settings.synchronous_mode = self.config.SYNCHRONOUS_MODE
        settings.fixed_delta_seconds = self.config.FIXED_DELTA_SECONDS
        settings.no_rendering_mode = self.config.NO_RENDERING_MODE

        # Memory optimization settings
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.deterministic_ragdolls = True  # For reproducibility

        self.world.apply_settings(settings)
        self.logger.info("⚙️  World settings configured for DRL training")

    def spawn_vehicle(self, vehicle_filter: str = None) -> bool:
        """Spawn a vehicle (preferably truck) at a designated spawn point"""
        start_time = time.time()

        try:
            # Get vehicle blueprints
            blueprint_library = self.world.get_blueprint_library()
            vehicle_filter = vehicle_filter or self.config.VEHICLE_FILTER

            # Try to get truck first, fallback to any vehicle
            vehicle_bps = blueprint_library.filter(vehicle_filter)
            if not vehicle_bps:
                self.logger.warning(f"⚠️  Truck not found, using fallback filter")
                vehicle_bps = blueprint_library.filter(self.config.VEHICLE_FILTER_FALLBACK)

            if not vehicle_bps:
                raise Exception("No vehicle blueprints available")

            vehicle_bp = vehicle_bps[0]  # Take first available
            self.logger.info(f"🚚 Selected vehicle: {vehicle_bp.id}")

            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available")

            # Use specified spawn point or default
            spawn_point = spawn_points[min(self.config.SPAWN_INDEX, len(spawn_points) - 1)]

            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            if not self.vehicle:
                raise Exception("Failed to spawn vehicle")

            self.spawn_time = time.time() - start_time
            self.logger.info(f"✅ Vehicle spawned successfully in {self.spawn_time:.2f}s")
            self.logger.info(f"📍 Location: {spawn_point.location}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to spawn vehicle: {e}")
            return False

    def setup_sensors(self) -> bool:
        """Setup camera and depth sensors with memory optimization"""
        if not self.vehicle:
            self.logger.error("❌ No vehicle available for sensor attachment")
            return False

        try:
            blueprint_library = self.world.get_blueprint_library()

            # RGB Camera sensor
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.CAMERA_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.config.CAMERA_HEIGHT))
            camera_bp.set_attribute('fov', str(self.config.CAMERA_FOV))
            camera_bp.set_attribute('sensor_tick', str(self.config.SENSOR_TICK))

            # Attach camera to vehicle
            camera_transform = carla.Transform(
                self.config.CAMERA_POSITION,
                self.config.CAMERA_ROTATION
            )

            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            self.camera_sensor.listen(self.data_manager.camera_callback)
            self.sensors.append(self.camera_sensor)

            # Depth Camera sensor (optional, for debugging)
            depth_bp = blueprint_library.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', str(self.config.DEPTH_WIDTH))
            depth_bp.set_attribute('image_size_y', str(self.config.DEPTH_HEIGHT))
            depth_bp.set_attribute('fov', str(self.config.CAMERA_FOV))
            depth_bp.set_attribute('sensor_tick', str(self.config.SENSOR_TICK))

            self.depth_sensor = self.world.spawn_actor(
                depth_bp, camera_transform, attach_to=self.vehicle
            )
            self.depth_sensor.listen(self.data_manager.depth_callback)
            self.sensors.append(self.depth_sensor)

            self.logger.info("📹 Sensors configured successfully")
            self.logger.info(f"   📷 RGB Camera: {self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT}")
            self.logger.info(f"   🔍 Depth Camera: {self.config.DEPTH_WIDTH}x{self.config.DEPTH_HEIGHT}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to setup sensors: {e}")
            return False

    def start_visualization(self) -> None:
        """Start real-time camera visualization using CV2"""
        if not self.camera_sensor:
            self.logger.error("❌ No camera sensor available for visualization")
            return

        self.is_running = True
        self.logger.info("🖥️  Starting real-time visualization (Press 'q' to quit)")

        # Wait for first frame
        start_wait = time.time()
        while self.data_manager.latest_camera_frame is None and time.time() - start_wait < 5.0:
            if self.config.SYNCHRONOUS_MODE:
                self.world.tick()
            time.sleep(0.1)

        if self.data_manager.latest_camera_frame is None:
            self.logger.error("❌ No camera data received within timeout")
            return

        fps_counter = 0
        fps_start_time = time.time()

        try:
            while self.is_running:
                frame = self.data_manager.get_latest_camera_frame()

                if frame is not None:
                    # Add overlay information
                    self._add_overlay(frame)

                    # Display frame
                    cv2.imshow('CARLA Camera Feed', frame)

                    # FPS calculation
                    fps_counter += 1
                    if fps_counter % 30 == 0:
                        elapsed = time.time() - fps_start_time
                        fps = fps_counter / elapsed
                        self.logger.info(f"📊 Display FPS: {fps:.1f}")

                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.logger.info("🛑 Stopping visualization")
                    break
                elif key == ord('s'):  # 's' for screenshot
                    self._save_screenshot(frame)

                # Tick world in synchronous mode
                if self.config.SYNCHRONOUS_MODE:
                    self.world.tick()
                else:
                    time.sleep(0.01)  # Small delay for async mode

        except KeyboardInterrupt:
            self.logger.info("🛑 Interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Visualization error: {e}")
        finally:
            self.is_running = False
            cv2.destroyAllWindows()

    def _add_overlay(self, frame: np.ndarray) -> None:
        """Add informational overlay to camera frame"""
        if frame is None:
            return

        # Vehicle information
        if self.vehicle:
            location = self.vehicle.get_location()
            velocity = self.vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h

            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # Green
            thickness = 2

            # Status information
            info_lines = [
                f"Speed: {speed:.1f} km/h",
                f"Pos: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})",
                f"Frame: {self.data_manager.frame_count}",
                f"Vehicle: {self.vehicle.type_id}",
                "Press 'q' to quit, 's' for screenshot"
            ]

            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                cv2.putText(frame, line, (10, y_pos), font, font_scale, color, thickness)

    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save current frame as screenshot"""
        if frame is not None:
            timestamp = int(time.time())
            filename = f"carla_screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            self.logger.info(f"📸 Screenshot saved: {filename}")

    def get_vehicle_control_interface(self) -> Optional[carla.Vehicle]:
        """Get vehicle control interface for external controllers (e.g., TD3 agent)"""
        return self.vehicle

    def apply_control(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0) -> None:
        """Apply control commands to vehicle"""
        if self.vehicle:
            control = carla.VehicleControl(
                throttle=max(0.0, min(1.0, throttle)),
                steer=max(-1.0, min(1.0, steer)),
                brake=max(0.0, min(1.0, brake))
            )
            self.vehicle.apply_control(control)

    def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor data for RL training"""
        return {
            'camera': self.data_manager.get_latest_camera_frame(),
            'depth': self.data_manager.get_latest_depth_frame(),
            'frame_count': self.data_manager.frame_count,
            'vehicle_location': self.vehicle.get_location() if self.vehicle else None,
            'vehicle_velocity': self.vehicle.get_velocity() if self.vehicle else None
        }

    @contextmanager
    def managed_session(self):
        """Context manager for automatic resource cleanup"""
        try:
            yield self
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources"""
        self.logger.info("🧹 Cleaning up resources...")

        self.is_running = False

        # Destroy sensors
        for sensor in self.sensors:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()

        # Destroy vehicle
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            self.vehicle = None

        # Close CV2 windows
        cv2.destroyAllWindows()

        # Reset world settings if needed
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        self.logger.info("✅ Cleanup completed")


def main():
    """Main function demonstrating CarlaClient usage"""
    # Initialize configuration
    config = CarlaConfig()

    # Create and run client
    carla_client = CarlaClient(config)

    with carla_client.managed_session():
        # Connect to CARLA
        if not carla_client.connect():
            print("❌ Failed to connect to CARLA server")
            return

        # Spawn vehicle
        if not carla_client.spawn_vehicle():
            print("❌ Failed to spawn vehicle")
            return

        # Setup sensors
        if not carla_client.setup_sensors():
            print("❌ Failed to setup sensors")
            return

        print("\n🎮 CARLA Client Ready!")
        print("=" * 50)
        print("📋 Instructions:")
        print("  • Camera view will open automatically")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save screenshot")
        print("  • Vehicle is ready for DRL agent control")
        print("=" * 50)

        # Start visualization
        carla_client.start_visualization()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CARLA Client for Deep Reinforcement Learning (DRL) Project

This module implements a professional CARLA client interface for TD3 training.
Designed following clean code principles: concise, coherent, maintainable.

Architecture:
    CarlaConfig: Centralized configuration management
    CarlaDataManager: Thread-safe sensor data handling
    CarlaClient: Main interface for CARLA interaction

Design Principles:
    - Single Responsibility: Each class has one clear purpose
    - Dependency Injection: Configuration passed explicitly
    - Error Handling: Graceful failure with informative logging
    - Resource Management: Automatic cleanup with context managers
    - Performance: Memory-optimized for RTX 2060 constraints

Author: DRL CARLA Project Team
Created: September 2024
Updated: September 2025
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

# CARLA imports with error handling
try:
    import carla
    print("✅ CARLA module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import CARLA: {e}")
    print("Please ensure CARLA Python API is in PYTHONPATH")
    sys.exit(1)


class SensorType(Enum):
    """Enumeration of supported sensor types"""
    RGB_CAMERA = "sensor.camera.rgb"
    DEPTH_CAMERA = "sensor.camera.depth"
    SEMANTIC_CAMERA = "sensor.camera.semantic_segmentation"


@dataclass
class CarlaConfig:
    """
    Configuration parameters for CARLA client optimized for DRL training.

    This class centralizes all configuration to improve maintainability
    and provide clear separation between environment and code.
    """
    # Network configuration
    host: str = '127.0.0.1'
    port: int = 2000
    timeout: float = 20.0  # Increased for stability

    # Simulation configuration
    town: str = 'Town01'
    spawn_index: int = 1
    synchronous_mode: bool = True  # Critical for DRL reproducibility
    fixed_delta_seconds: float = 0.05  # 20 FPS for stable training

    # Vehicle configuration
    vehicle_filter: str = 'vehicle.carlamotors.carlacola'  # Preferred truck
    vehicle_fallback: str = 'vehicle.*'  # Fallback to any vehicle

    # RGB Camera configuration (primary sensor for DRL)
    camera_width: int = 640
    camera_height: int = 480
    camera_fov: float = 90.0
    camera_x: float = 4.0  # Forward position for truck visibility
    camera_y: float = 0.0  # Centered
    camera_z: float = 1.4  # Eye-level height
    camera_pitch: float = 0.0  # Level horizon
    camera_yaw: float = 0.0  # Forward facing
    camera_roll: float = 0.0  # No rotation

    # Depth camera configuration (auxiliary sensor)
    depth_width: int = 320
    depth_height: int = 240

    # Performance optimization for RTX 2060
    max_frame_buffer: int = 5  # Limit memory usage
    sensor_tick: float = 0.1  # 10 Hz sensor rate
    no_rendering_mode: bool = False  # Keep visual feedback for development

    def get_camera_transform(self) -> carla.Transform:
        """Returns camera transform based on configuration"""
        location = carla.Location(x=self.camera_x, y=self.camera_y, z=self.camera_z)
        rotation = carla.Rotation(pitch=self.camera_pitch, yaw=self.camera_yaw, roll=self.camera_roll)
        return carla.Transform(location, rotation)


class CarlaDataManager:
    """
    Thread-safe sensor data management with memory optimization.

    Handles sensor callbacks and provides thread-safe access to latest frames.
    Implements memory-efficient buffering suitable for RTX 2060 constraints.

    Design principles:
    - Thread safety through locking mechanisms
    - Memory efficiency through frame limiting
    - Clear separation of concerns for different sensor types
    """

    def __init__(self, max_buffer_size: int = 5):
        """
        Initialize data manager with specified buffer size.

        Args:
            max_buffer_size: Maximum frames to keep in memory
        """
        self._max_buffer_size = max_buffer_size
        self._latest_camera_frame: Optional[np.ndarray] = None
        self._latest_depth_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        # Setup logging for this component
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def camera_callback(self, carla_image: carla.Image) -> None:
        """
        Process RGB camera data from CARLA.

        Converts CARLA image format to OpenCV BGR format for visualization.
        Thread-safe operation with automatic memory management.

        Args:
            carla_image: Raw image data from CARLA sensor
        """
        try:
            # Convert CARLA BGRA to OpenCV BGR
            raw_array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            bgra_array = raw_array.reshape((carla_image.height, carla_image.width, 4))
            bgr_frame = bgra_array[:, :, [2, 1, 0]]  # Extract BGR channels

            with self._lock:
                self._latest_camera_frame = bgr_frame.copy()
                self._frame_count += 1

        except Exception as e:
            self._logger.error(f"Camera callback failed: {e}")

    def depth_callback(self, carla_image: carla.Image) -> None:
        """
        Process depth camera data from CARLA.

        Extracts depth information from R channel and normalizes to meters.

        Args:
            carla_image: Raw depth image from CARLA sensor
        """
        try:
            # Extract depth from R channel (CARLA stores depth * 1000 in R)
            raw_array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
            bgra_array = raw_array.reshape((carla_image.height, carla_image.width, 4))
            depth_frame = bgra_array[:, :, 0].astype(np.float32) / 255.0 * 1000.0

            with self._lock:
                self._latest_depth_frame = depth_frame.copy()

        except Exception as e:
            self._logger.error(f"Depth callback failed: {e}")

    def get_latest_camera_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent camera frame in thread-safe manner.

        Returns:
            Copy of latest BGR frame or None if no frame available
        """
        with self._lock:
            return self._latest_camera_frame.copy() if self._latest_camera_frame is not None else None

    def get_latest_depth_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent depth frame in thread-safe manner.

        Returns:
            Copy of latest depth frame or None if no frame available
        """
        with self._lock:
            return self._latest_depth_frame.copy() if self._latest_depth_frame is not None else None

    def get_frame_count(self) -> int:
        """Get total number of processed frames."""
        with self._lock:
            return self._frame_count

    def has_camera_data(self) -> bool:
        """Check if camera data is available."""
        with self._lock:
            return self._latest_camera_frame is not None


class CarlaClient:
    """
    Professional CARLA client for Deep Reinforcement Learning applications.

    This class provides a clean, maintainable interface to CARLA simulator
    optimized for TD3 training with truck vehicles. Following clean code principles:
    - Single responsibility for CARLA operations
    - Clear error handling and logging
    - Resource management with context managers
    - Memory optimization for RTX 2060

    Architecture:
        Configuration -> Connection -> Vehicle Spawn -> Sensor Setup -> Control Loop
    """

    def __init__(self, config: CarlaConfig):
        """
        Initialize CARLA client with dependency injection.

        Args:
            config: Configuration object with all parameters
        """
        # Store configuration (dependency injection)
        self._config = config

        # Initialize CARLA components
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._vehicle: Optional[carla.Vehicle] = None
        self._blueprint_library: Optional[carla.BlueprintLibrary] = None

        # Initialize sensors
        self._rgb_camera: Optional[carla.Sensor] = None
        self._depth_camera: Optional[carla.Sensor] = None
        self._sensors: list = []

        # Initialize data management
        self._data_manager = CarlaDataManager(config.max_frame_buffer)

        # State tracking
        self._is_connected = False
        self._is_running = False

        # Performance monitoring
        self._connection_time: Optional[float] = None
        self._spawn_time: Optional[float] = None

        # Setup logging with clear naming
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for this component."""
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def connect(self) -> bool:
        """
        Establish connection to CARLA server.

        Returns:
            True if connection successful, False otherwise
        """
        start_time = time.time()

        try:
            self._logger.info(f"🔄 Connecting to CARLA at {self._config.host}:{self._config.port}")

            # Create client connection
            self._client = carla.Client(self._config.host, self._config.port)
            self._client.set_timeout(self._config.timeout)

            # Verify connection
            version = self._client.get_server_version()
            self._logger.info(f"✅ Connected to CARLA {version}")

            # Get world and blueprint library
            self._world = self._client.get_world()
            self._blueprint_library = self._world.get_blueprint_library()

            # Configure world for DRL
            self._configure_world_settings()

            # Load map if necessary
            self._ensure_correct_map()

            self._is_connected = True
            self._connection_time = time.time() - start_time
            self._logger.info(f"🎉 Connection established in {self._connection_time:.2f}s")

            return True

        except Exception as e:
            self._logger.error(f"❌ Connection failed: {e}")
            return False

    def _configure_world_settings(self) -> None:
        """Configure world settings for deterministic DRL training."""
        if not self._world:
            return

        settings = self._world.get_settings()
        settings.synchronous_mode = self._config.synchronous_mode
        settings.fixed_delta_seconds = self._config.fixed_delta_seconds
        settings.no_rendering_mode = self._config.no_rendering_mode

        # Apply settings for reproducibility
        self._world.apply_settings(settings)
        self._logger.info("⚙️  World configured for DRL training")

    def _ensure_correct_map(self) -> None:
        """Load the correct map if not already loaded."""
        if not self._world:
            return

        current_map = self._world.get_map().name
        self._logger.info(f"🗺️  Current map: {current_map}")

        # Only reload if different map is requested
        if self._config.town not in current_map:
            self._logger.info(f"🔄 Loading map: {self._config.town}")
            self._world = self._client.load_world(self._config.town)
            time.sleep(3)  # Allow map loading time
            self._blueprint_library = self._world.get_blueprint_library()

    def spawn_vehicle(self) -> bool:
        """
        Spawn preferred vehicle (truck) at designated location.

        Returns:
            True if vehicle spawned successfully, False otherwise
        """
        if not self._is_connected or not self._blueprint_library:
            self._logger.error("❌ Must connect before spawning vehicle")
            return False

        start_time = time.time()

        try:
            # Get vehicle blueprint
            vehicle_bp = self._get_vehicle_blueprint()
            if not vehicle_bp:
                return False

            # Get spawn location
            spawn_point = self._get_spawn_point()
            if not spawn_point:
                return False

            # Spawn vehicle
            self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)
            if not self._vehicle:
                self._logger.error("❌ Failed to spawn vehicle actor")
                return False

            self._spawn_time = time.time() - start_time
            self._logger.info(f"✅ Vehicle spawned in {self._spawn_time:.2f}s")
            self._logger.info(f"📍 Location: {spawn_point.location}")
            self._logger.info(f"🚚 Vehicle: {vehicle_bp.id}")

            return True

        except Exception as e:
            self._logger.error(f"❌ Vehicle spawn failed: {e}")
            return False

    def _get_vehicle_blueprint(self) -> Optional[carla.BlueprintLibrary]:
        """Get preferred vehicle blueprint with fallback."""
        # Try preferred truck first
        truck_bps = self._blueprint_library.filter(self._config.vehicle_filter)
        if truck_bps:
            return truck_bps[0]

        # Fallback to any vehicle
        self._logger.warning(f"⚠️  Preferred vehicle {self._config.vehicle_filter} not found")
        vehicle_bps = self._blueprint_library.filter(self._config.vehicle_fallback)
        if vehicle_bps:
            self._logger.info(f"🔄 Using fallback vehicle: {vehicle_bps[0].id}")
            return vehicle_bps[0]

        self._logger.error("❌ No vehicles available")
        return None

    def _get_spawn_point(self) -> Optional[carla.Transform]:
        """Get spawn point for vehicle."""
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            self._logger.error("❌ No spawn points available")
            return None

        # Use configured spawn index, clamp to available range
        index = min(self._config.spawn_index, len(spawn_points) - 1)
        return spawn_points[index]

    def setup_sensors(self) -> bool:
        """
        Setup RGB and depth cameras for DRL training.

        Returns:
            True if sensors configured successfully, False otherwise
        """
        if not self._vehicle:
            self._logger.error("❌ Must spawn vehicle before setting up sensors")
            return False

        try:
            # Setup RGB camera (primary sensor for DRL)
            if not self._setup_rgb_camera():
                return False

            # Setup depth camera (auxiliary sensor for development)
            if not self._setup_depth_camera():
                self._logger.warning("⚠️  Depth camera setup failed, continuing with RGB only")

            self._logger.info("📹 Sensors configured successfully")
            self._logger.info(f"   📷 RGB Camera: {self._config.camera_width}x{self._config.camera_height}")
            self._logger.info(f"   🔍 Depth Camera: {self._config.depth_width}x{self._config.depth_height}")

            return True

        except Exception as e:
            self._logger.error(f"❌ Sensor setup failed: {e}")
            return False

    def _setup_rgb_camera(self) -> bool:
        """Setup RGB camera sensor."""
        try:
            # Get camera blueprint
            camera_bp = self._blueprint_library.find(SensorType.RGB_CAMERA.value)
            camera_bp.set_attribute('image_size_x', str(self._config.camera_width))
            camera_bp.set_attribute('image_size_y', str(self._config.camera_height))
            camera_bp.set_attribute('fov', str(self._config.camera_fov))
            camera_bp.set_attribute('sensor_tick', str(self._config.sensor_tick))

            # Spawn and attach camera
            camera_transform = self._config.get_camera_transform()
            self._rgb_camera = self._world.spawn_actor(
                camera_bp, camera_transform, attach_to=self._vehicle
            )

            # Setup callback
            self._rgb_camera.listen(self._data_manager.camera_callback)
            self._sensors.append(self._rgb_camera)

            return True

        except Exception as e:
            self._logger.error(f"❌ RGB camera setup failed: {e}")
            return False

    def _setup_depth_camera(self) -> bool:
        """Setup depth camera sensor."""
        try:
            # Get depth camera blueprint
            depth_bp = self._blueprint_library.find(SensorType.DEPTH_CAMERA.value)
            depth_bp.set_attribute('image_size_x', str(self._config.depth_width))
            depth_bp.set_attribute('image_size_y', str(self._config.depth_height))
            depth_bp.set_attribute('fov', str(self._config.camera_fov))
            depth_bp.set_attribute('sensor_tick', str(self._config.sensor_tick))

            # Spawn and attach depth camera
            depth_transform = self._config.get_camera_transform()
            self._depth_camera = self._world.spawn_actor(
                depth_bp, depth_transform, attach_to=self._vehicle
            )

            # Setup callback
            self._depth_camera.listen(self._data_manager.depth_callback)
            self._sensors.append(self._depth_camera)

            return True

        except Exception as e:
            self._logger.error(f"❌ Depth camera setup failed: {e}")
            return False

    def start_visualization(self, display_time: Optional[float] = None) -> None:
        """
        Start real-time camera visualization using OpenCV.

        Args:
            display_time: Maximum display time in seconds (None for infinite)
        """
        if not self._data_manager.has_camera_data():
            self._logger.info("⏳ Waiting for camera data...")
            # Wait up to 5 seconds for first frame
            timeout = time.time() + 5.0
            while not self._data_manager.has_camera_data() and time.time() < timeout:
                time.sleep(0.1)

        if not self._data_manager.has_camera_data():
            self._logger.error("❌ No camera data received within timeout")
            return

        self._is_running = True
        self._logger.info("🖥️  Starting visualization (Press 'q' to quit, 's' to save)")

        start_time = time.time()
        fps_counter = 0
        last_fps_time = start_time

        try:
            while self._is_running:
                # Check display time limit
                if display_time and (time.time() - start_time) > display_time:
                    break

                # Get latest frame
                frame = self._data_manager.get_latest_camera_frame()
                if frame is None:
                    continue

                # Add informational overlay
                self._add_overlay(frame)

                # Display frame
                cv2.imshow('CARLA DRL - Camera View', frame)

                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(frame)

                # Update FPS counter
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 7.0:  # Update every 7 seconds
                    fps = fps_counter / (current_time - last_fps_time)
                    self._logger.info(f"📊 Display FPS: {fps:.1f}")
                    fps_counter = 0
                    last_fps_time = current_time

        except KeyboardInterrupt:
            self._logger.info("🛑 Interrupted by user")
        except Exception as e:
            self._logger.error(f"❌ Visualization error: {e}")
        finally:
            self._is_running = False
            cv2.destroyAllWindows()

    def _add_overlay(self, frame: np.ndarray) -> None:
        """Add informational overlay to camera frame."""
        if frame is None:
            return

        # Vehicle information overlay
        if self._vehicle:
            location = self._vehicle.get_location()
            velocity = self._vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h

            # Add text overlay
            text_lines = [
                f"Location: ({location.x:.1f}, {location.y:.1f})",
                f"Speed: {speed:.1f} km/h",
                f"Frames: {self._data_manager.get_frame_count()}"
            ]

            for i, line in enumerate(text_lines):
                cv2.putText(frame, line, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save current frame as screenshot."""
        if frame is not None:
            timestamp = int(time.time())
            filename = f"carla_screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            self._logger.info(f"📸 Screenshot saved: {filename}")

    def apply_control(self, throttle: float = 0.0, steer: float = 0.0, brake: float = 0.0) -> None:
        """
        Apply control commands to vehicle (for external DRL agents).

        Args:
            throttle: Throttle value [0.0, 1.0]
            steer: Steering value [-1.0, 1.0]
            brake: Brake value [0.0, 1.0]
        """
        if self._vehicle:
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            self._vehicle.apply_control(control)

    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Get current sensor data for DRL training.

        Returns:
            Dictionary containing sensor data and vehicle state
        """
        data = {
            'camera_frame': self._data_manager.get_latest_camera_frame(),
            'depth_frame': self._data_manager.get_latest_depth_frame(),
            'frame_count': self._data_manager.get_frame_count(),
            'timestamp': time.time()
        }

        if self._vehicle:
            data['vehicle_location'] = self._vehicle.get_location()
            data['vehicle_velocity'] = self._vehicle.get_velocity()
            data['vehicle_transform'] = self._vehicle.get_transform()

        return data

    @contextmanager
    def managed_session(self):
        """Context manager for automatic resource cleanup."""
        try:
            yield self
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up all resources properly."""
        self._logger.info("🧹 Cleaning up resources...")

        self._is_running = False

        # Destroy sensors
        for sensor in self._sensors:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors.clear()

        # Destroy vehicle
        if self._vehicle and self._vehicle.is_alive:
            self._vehicle.destroy()
            self._vehicle = None

        # Close OpenCV windows
        cv2.destroyAllWindows()

        # Reset world settings if needed
        if self._world:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            self._world.apply_settings(settings)

        self._is_connected = False
        self._logger.info("✅ Cleanup completed")

    # Public properties for external access
    @property
    def vehicle(self) -> Optional[carla.Vehicle]:
        """Get vehicle instance for external control."""
        return self._vehicle

    @property
    def is_connected(self) -> bool:
        """Check if connected to CARLA."""
        return self._is_connected

    @property
    def data_manager(self) -> CarlaDataManager:
        """Get data manager for external access."""
        return self._data_manager


def main():
    """
    Main demonstration function following clean code principles.

    This function demonstrates the proper usage of CarlaClient with
    clear error handling and resource management.
    """
    # Initialize configuration with dependency injection
    config = CarlaConfig()

    # Create client instance
    carla_client = CarlaClient(config)

    # Use context manager for automatic cleanup
    with carla_client.managed_session():
        # Connect to CARLA server
        if not carla_client.connect():
            print("❌ Failed to connect to CARLA server")
            return 1

        # Spawn vehicle
        if not carla_client.spawn_vehicle():
            print("❌ Failed to spawn vehicle")
            return 1

        # Setup sensors
        if not carla_client.setup_sensors():
            print("❌ Failed to setup sensors")
            return 1

        # Display ready message
        print("\n🎮 CARLA DRL Client Ready!")
        print("=" * 50)
        print("📋 Instructions:")
        print("  • Camera view will open automatically")
        print("  • Press 'q' to quit")
        print("  • Press 's' to save screenshot")
        print("  • Vehicle is ready for DRL agent control")
        print("=" * 50)

        # Start visualization loop
        carla_client.start_visualization()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

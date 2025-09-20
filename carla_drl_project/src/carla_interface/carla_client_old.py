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

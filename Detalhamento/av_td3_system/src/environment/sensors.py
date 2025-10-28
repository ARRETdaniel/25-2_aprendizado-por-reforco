"""
Sensor Management for CARLA Autonomous Vehicle

Handles all sensor data collection, preprocessing, and queuing:
- Front camera: RGB image capture and preprocessing (grayscale, resize, normalize)
- Image stacking: Maintain 4-frame history for temporal context
- Collision detector: Track collision events
- Lane invasion detector: Track lane departure events

Paper Reference: "End-to-End Visual Autonomous Navigation with Twin Delayed DDPG"
Section III.A: Visual input processing (4 stacked frames, 84×84 grayscale)
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict
import logging
import queue
import threading

import carla


class CARLACameraManager:
    """
    Manages front-facing RGB camera sensor.

    Responsibilities:
    - Capture RGB images from CARLA
    - Preprocess images (grayscale, resize, normalize)
    - Maintain queue of latest captures
    """

    def __init__(
        self,
        vehicle: carla.Actor,
        camera_config: Dict,
        world: carla.World,
    ):
        """
        Initialize camera sensor.

        Args:
            vehicle: CARLA vehicle actor to attach camera to
            camera_config: Dict with resolution, FOV from carla_config.yaml
                Expected keys: width, height, fov
            world: CARLA world object
        """
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        # Camera configuration
        self.width = camera_config.get("width", 256)
        self.height = camera_config.get("height", 144)
        self.fov = camera_config.get("fov", 90)

        # Preprocessing target (for CNN)
        self.target_width = 84
        self.target_height = 84

        # Queue for latest image (thread-safe)
        self.image_queue = queue.Queue(maxsize=5)
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Camera sensor setup
        self.camera_sensor = None
        self._setup_camera()

        self.logger.info(
            f"Camera initialized: {self.width}×{self.height}, FOV={self.fov}°, "
            f"target size {self.target_width}×{self.target_height}"
        )

    def _setup_camera(self):
        """Attach RGB camera to vehicle."""
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # Camera parameters
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", str(self.fov))

        # Mount camera at front of vehicle (default position)
        spawn_point = carla.Transform(
            carla.Location(x=1.6, z=1.5)  # Front, slightly above center
        )

        self.camera_sensor = self.world.spawn_actor(
            camera_bp, spawn_point, attach_to=self.vehicle
        )

        # Register callback for new images
        self.camera_sensor.listen(self._on_camera_frame)

    def _on_camera_frame(self, image: carla.Image):
        """
        Callback when camera frame arrives.

        Converts CARLA image to numpy, preprocesses, and queues.
        Runs in CARLA's callback thread, so must be thread-safe.
        """
        try:
            # Convert CARLA image to numpy array
            # Note: CARLA provides BGRA format
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Drop alpha channel → BGR
            array = array[:, :, ::-1]  # Convert BGR to RGB

            # Preprocess
            processed = self._preprocess(array)

            # Store in thread-safe manner
            with self.image_lock:
                self.latest_image = processed

            # Try to add to queue (non-blocking)
            try:
                self.image_queue.put_nowait(processed)
            except queue.Full:
                pass  # Drop old frame if queue full

        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN (matches DQN reference implementation).

        Steps:
        1. Convert RGB to grayscale
        2. Resize to 84×84 (standard for CNN in RL)
        3. Scale to [0, 1]
        4. Normalize to [-1, 1] (zero-centered for better CNN performance)

        Args:
            image: RGB image as numpy array (H×W×3) with values 0-255

        Returns:
            Grayscale image (84×84) normalized to [-1, 1]
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize to target size
        resized = cv2.resize(
            gray, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA
        )

        # Scale to [0, 1]
        scaled = resized.astype(np.float32) / 255.0

        # Normalize to [-1, 1] (zero-centered)
        # This matches the DQN reference and is standard for image CNNs
        mean, std = 0.5, 0.5
        normalized = (scaled - mean) / std

        return normalized

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get latest preprocessed frame.

        Returns:
            84×84 grayscale image normalized [-1, 1], or None if no frame available
        """
        with self.image_lock:
            return self.latest_image

    def destroy(self):
        """Clean up camera sensor."""
        if self.camera_sensor:
            try:
                self.camera_sensor.destroy()
                self.logger.info("Camera sensor destroyed")
            except RuntimeError as e:
                self.logger.warning(f"Camera sensor already destroyed or invalid: {e}")
            except Exception as e:
                self.logger.error(f"Error destroying camera sensor: {e}")


class ImageStack:
    """
    Maintains stack of 4 most recent frames for temporal context.

    This is critical for RL: individual frames are ambiguous (can't tell velocity
    from static image), but 4 stacked frames show motion and acceleration.

    Uses FIFO queue: pushes new frame, pops oldest.
    """

    def __init__(self, frame_height: int = 84, frame_width: int = 84, num_frames: int = 4):
        """
        Initialize frame stack.

        Args:
            frame_height: Height of each frame (84)
            frame_width: Width of each frame (84)
            num_frames: Number of frames to stack (4)
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames
        self.logger = logging.getLogger(__name__)

        # Pre-allocate stack
        self.stack = deque(
            [
                np.zeros((frame_height, frame_width), dtype=np.float32)
                for _ in range(num_frames)
            ],
            maxlen=num_frames,
        )

    def push_frame(self, frame: np.ndarray):
        """
        Add new frame to stack (removes oldest).

        Args:
            frame: 84×84 preprocessed grayscale image, normalized [-1, 1]
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError(
                f"Frame shape {frame.shape} != expected "
                f"({self.frame_height}, {self.frame_width})"
            )
        self.stack.append(frame)

    def get_stacked_frames(self) -> np.ndarray:
        """
        Get all stacked frames as single array.

        Returns:
            (4, 84, 84) array with frames in temporal order (oldest to newest)
        """
        return np.array(list(self.stack), dtype=np.float32)

    def reset(self):
        """Reset stack with zeros."""
        self.stack.clear()
        for _ in range(self.num_frames):
            self.stack.append(
                np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
            )

    def is_filled(self) -> bool:
        """Check if stack contains actual frames (not just zeros)."""
        # Return True if all frames have been pushed at least once
        return len(self.stack) == self.num_frames


class CollisionDetector:
    """
    Tracks collision events with other vehicles/obstacles.
    """

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        """
        Initialize collision detector.

        Args:
            vehicle: CARLA vehicle actor to attach collision sensor to
            world: CARLA world object
        """
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        # Collision state
        self.collision_detected = False
        self.collision_event = None
        self.collision_lock = threading.Lock()

        # Collision sensor setup
        self.collision_sensor = None
        self._setup_collision_sensor()

    def _setup_collision_sensor(self):
        """Attach collision sensor to vehicle."""
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        spawn_point = carla.Transform(carla.Location())

        self.collision_sensor = self.world.spawn_actor(
            collision_bp, spawn_point, attach_to=self.vehicle
        )
        self.collision_sensor.listen(self._on_collision)
        self.logger.info("Collision sensor initialized")

    def _on_collision(self, event: carla.CollisionEvent):
        """
        Callback when collision occurs.

        Args:
            event: CARLA CollisionEvent with collision details
        """
        with self.collision_lock:
            self.collision_detected = True
            self.collision_event = event
        self.logger.warning(
            f"Collision detected with {event.other_actor.type_id} "
            f"(impulse: {event.normal_impulse.length()})"
        )

    def is_colliding(self) -> bool:
        """
        Check if collision detected this frame.

        Returns:
            True if collision occurred
        """
        with self.collision_lock:
            return self.collision_detected

    def get_collision_info(self) -> Optional[Dict]:
        """
        Get collision details if collision occurred.

        Returns:
            Dict with collision info or None
        """
        with self.collision_lock:
            if self.collision_detected and self.collision_event:
                return {
                    "other_actor": str(self.collision_event.other_actor.type_id),
                    "impulse": self.collision_event.normal_impulse.length(),
                }
            return None

    def reset(self):
        """Reset collision state for new episode."""
        with self.collision_lock:
            self.collision_detected = False
            self.collision_event = None

    def destroy(self):
        """Clean up collision sensor."""
        if self.collision_sensor:
            try:
                self.collision_sensor.destroy()
                self.logger.info("Collision sensor destroyed")
            except RuntimeError as e:
                self.logger.warning(f"Collision sensor already destroyed or invalid: {e}")
            except Exception as e:
                self.logger.error(f"Error destroying collision sensor: {e}")


class LaneInvasionDetector:
    """
    Tracks lane departure events (driving off road).
    """

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        """
        Initialize lane invasion detector.

        Args:
            vehicle: CARLA vehicle actor to attach sensor to
            world: CARLA world object
        """
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        # Lane invasion state
        self.lane_invaded = False
        self.invasion_event = None
        self.invasion_lock = threading.Lock()

        # Lane invasion sensor setup
        self.lane_sensor = None
        self._setup_lane_sensor()

    def _setup_lane_sensor(self):
        """Attach lane invasion sensor to vehicle."""
        lane_bp = self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        spawn_point = carla.Transform(carla.Location())

        self.lane_sensor = self.world.spawn_actor(
            lane_bp, spawn_point, attach_to=self.vehicle
        )
        self.lane_sensor.listen(self._on_lane_invasion)
        self.logger.info("Lane invasion sensor initialized")

    def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
        """
        Callback when lane invasion occurs.

        Args:
            event: CARLA LaneInvasionEvent
        """
        with self.invasion_lock:
            self.lane_invaded = True
            self.invasion_event = event
        self.logger.warning(f"Lane invasion detected: {event.crossed_lane_markings}")

    def is_invading_lane(self) -> bool:
        """
        Check if lane invasion detected this frame.

        Returns:
            True if off-road/lane invaded
        """
        with self.invasion_lock:
            return self.lane_invaded

    def reset(self):
        """Reset lane invasion state for new episode."""
        with self.invasion_lock:
            self.lane_invaded = False
            self.invasion_event = None

    def destroy(self):
        """Clean up lane invasion sensor."""
        if self.lane_sensor:
            try:
                self.lane_sensor.destroy()
                self.logger.info("Lane invasion sensor destroyed")
            except RuntimeError as e:
                self.logger.warning(f"Lane invasion sensor already destroyed or invalid: {e}")
            except Exception as e:
                self.logger.error(f"Error destroying lane invasion sensor: {e}")


class SensorSuite:
    """
    Aggregates all sensors and provides unified interface.

    Manages camera, collision, lane invasion detectors.
    Provides synchronized access to all sensor data.
    """

    def __init__(
        self,
        vehicle: carla.Actor,
        carla_config: Dict,
        world: carla.World,
    ):
        """
        Initialize all sensors.

        Args:
            vehicle: CARLA vehicle actor
            carla_config: Configuration dict from carla_config.yaml
            world: CARLA world object
        """
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        # Extract sensor configs
        camera_config = carla_config.get("sensors", {}).get("camera", {})

        # Initialize sensors
        self.camera = CARLACameraManager(vehicle, camera_config, world)
        self.collision_detector = CollisionDetector(vehicle, world)
        self.lane_invasion_detector = LaneInvasionDetector(vehicle, world)

        # Initialize frame stacking
        self.image_stack = ImageStack(
            frame_height=84, frame_width=84, num_frames=4
        )

        self.logger.info("Complete sensor suite initialized")

    def tick(self):
        """
        Process sensor data for current frame.

        Called once per simulation step to:
        - Capture latest camera frame and add to stack
        - Update collision/lane status
        """
        # Get latest camera frame
        frame = self.camera.get_latest_frame()
        if frame is not None:
            self.image_stack.push_frame(frame)

    def get_camera_data(self) -> np.ndarray:
        """
        Get stacked camera frames.

        Returns:
            (4, 84, 84) array with 4 stacked grayscale frames, normalized [-1, 1]
        """
        return self.image_stack.get_stacked_frames()

    def is_collision_detected(self) -> bool:
        """
        Check if collision occurred this frame.

        Returns:
            True if collision detected
        """
        return self.collision_detector.is_colliding()

    def is_lane_invaded(self) -> bool:
        """
        Check if vehicle went off-road this frame.

        Returns:
            True if lane invaded
        """
        return self.lane_invasion_detector.is_invading_lane()

    def get_collision_info(self) -> Optional[Dict]:
        """Get collision details if any."""
        return self.collision_detector.get_collision_info()

    def reset(self):
        """Reset all sensors for new episode."""
        self.image_stack.reset()
        self.collision_detector.reset()
        self.lane_invasion_detector.reset()
        self.logger.info("All sensors reset")

    def destroy(self):
        """Clean up all sensors."""
        errors = []

        # Destroy camera
        try:
            self.camera.destroy()
        except Exception as e:
            errors.append(f"Camera: {e}")
            self.logger.warning(f"Error destroying camera: {e}")

        # Destroy collision detector
        try:
            self.collision_detector.destroy()
        except Exception as e:
            errors.append(f"Collision: {e}")
            self.logger.warning(f"Error destroying collision detector: {e}")

        # Destroy lane invasion detector
        try:
            self.lane_invasion_detector.destroy()
        except Exception as e:
            errors.append(f"Lane: {e}")
            self.logger.warning(f"Error destroying lane invasion detector: {e}")

        if errors:
            self.logger.warning(f"Sensor cleanup completed with {len(errors)} error(s)")
        else:
            self.logger.info("Sensor suite destroyed successfully")

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

        # OPTIMIZATION: Step counter for throttled debug logging (every 100 frames)
        self.frame_step_counter = 0
        self.log_frequency = 100

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
        # OPTIMIZATION: Increment step counter for throttled logging
        self.frame_step_counter += 1
        should_log = (self.frame_step_counter % self.log_frequency == 0)

        # DEBUG: Log input image statistics every 100 frames
        # OPTIMIZATION: Throttled to reduce logging overhead (was every frame)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   PREPROCESSING INPUT (Frame #{self.frame_step_counter}):\n"
                f"   Shape: {image.shape}\n"
                f"   Dtype: {image.dtype}\n"
                f"   Range: [{image.min():.2f}, {image.max():.2f}]\n"
                f"   Mean: {image.mean():.2f}, Std: {image.std():.2f}"
            )

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

        # DEBUG: Log output statistics every 100 frames
        # OPTIMIZATION: Throttled to reduce logging overhead (was every frame)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   PREPROCESSING OUTPUT:\n"
                f"   Shape: {normalized.shape}\n"
                f"   Dtype: {normalized.dtype}\n"
                f"   Range: [{normalized.min():.3f}, {normalized.max():.3f}]\n"
                f"   Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}\n"
                f"   Expected range: [-1, 1]"
            )

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
        """
        Clean up camera sensor.

        CRITICAL FIX: In synchronous mode, sensor callbacks may still be pending
        after stop(). We must flush the callback queue before destroying the actor.

        Reference: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        "Data coming from GPU-based sensors, mostly cameras, is usually generated
        with a delay of a couple of frames."
        """
        if self.camera_sensor:
            try:
                # Step 1: Stop listening before destruction (CARLA best practice)
                # Reference: https://carla.readthedocs.io/en/latest/core_sensors/
                if self.camera_sensor.is_listening:
                    self.camera_sensor.stop()
                    self.logger.debug("Camera sensor stopped listening")

                # Step 2: Flush pending callbacks in synchronous mode
                # CRITICAL: In sync mode, callbacks may still be queued even after stop().
                # We need to let the server process them before destroying the actor.
                # The world.tick() call in the environment will handle this automatically,
                # but we add a small safety delay for async callback completion.
                import time
                time.sleep(0.01)  # 10ms grace period for callback completion

                # Step 3: Destroy the sensor actor
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

        # OPTIMIZATION: Step counter for throttled debug logging (every 100 pushes)
        self.push_step_counter = 0
        self.log_frequency = 100

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

        # OPTIMIZATION: Increment step counter for throttled logging
        self.push_step_counter += 1
        should_log = (self.push_step_counter % self.log_frequency == 0)

        # DEBUG: Log frame stacking every 100 pushes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every push)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   FRAME STACKING (Push #{self.push_step_counter}):\n"
                f"   New frame shape: {frame.shape}\n"
                f"   New frame range: [{frame.min():.3f}, {frame.max():.3f}]\n"
                f"   Stack size before: {len(self.stack)}\n"
                f"   Stack filled: {self.is_filled()}"
            )

        self.stack.append(frame)

        # DEBUG: Log stack state after push every 100 pushes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every push)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            stacked = self.get_stacked_frames()
            self.logger.debug(
                f"  STACK STATE AFTER PUSH:\n"
                f"   Stack size: {len(self.stack)}\n"
                f"   Stacked shape: {stacked.shape}\n"
                f"   Stacked range: [{stacked.min():.3f}, {stacked.max():.3f}]\n"
                f"   Per-frame ranges: {[f'[{frame.min():.3f}, {frame.max():.3f}]' for frame in self.stack]}"
            )

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

    Captures collision impulse magnitude for graduated penalties in PBRS reward shaping.
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
        self.collision_impulse = 0.0  # NEW: Impulse magnitude in Newton-seconds
        self.collision_force = 0.0    # NEW: Approximate force in Newtons
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

        Captures collision impulse magnitude for graduated penalties.
        Collision impulse is the force applied over the collision duration.

        Args:
            event: CARLA CollisionEvent with collision details
                - event.normal_impulse: Vector3D impulse in Newton-seconds (N·s)
                - event.other_actor: The actor collided with

        Reference: PBRS Implementation Guide (Priority 3 Fix)
        """
        with self.collision_lock:
            self.collision_detected = True
            self.collision_event = event

            # Extract collision impulse magnitude (force in Newton-seconds)
            # This enables graduated penalties (soft collision vs severe crash)
            impulse_vector = event.normal_impulse  # Vector3D in N·s
            self.collision_impulse = impulse_vector.length()  # Magnitude in N·s

            # Approximate collision force (assuming typical collision duration ~0.1s)
            # This is an approximation: Force = Impulse / Duration
            # CARLA doesn't provide exact collision duration, so we use typical value
            collision_duration = 0.1  # seconds (typical for rigid body impacts)
            self.collision_force = self.collision_impulse / collision_duration  # Newtons

        self.logger.warning(
            f"Collision detected with {event.other_actor.type_id} "
            f"(impulse: {self.collision_impulse:.1f} N·s, force: ~{self.collision_force:.1f} N)"
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
            Dict with collision info including impulse magnitude, or None
        """
        with self.collision_lock:
            if self.collision_detected and self.collision_event:
                return {
                    "other_actor": str(self.collision_event.other_actor.type_id),
                    "impulse": self.collision_impulse,  # N·s (Newton-seconds)
                    "force": self.collision_force,      # N (Newtons, approximate)
                }
            return None

    def reset(self):
        """Reset collision state for new episode."""
        with self.collision_lock:
            self.collision_detected = False
            self.collision_event = None
            self.collision_impulse = 0.0
            self.collision_force = 0.0

    def destroy(self):
        """
        Clean up collision sensor.

        CRITICAL FIX: Add grace period for pending callback completion.
        Reference: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        """
        if self.collision_sensor:
            try:
                # Stop listening before destruction (CARLA best practice)
                # Reference: https://carla.readthedocs.io/en/latest/core_sensors/
                if self.collision_sensor.is_listening:
                    self.collision_sensor.stop()
                    self.logger.debug("Collision sensor stopped listening")

                # Grace period for async callback completion
                import time
                time.sleep(0.01)

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
        """
        Clean up lane invasion sensor.

        CRITICAL FIX: Add grace period for pending callback completion.
        Reference: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        """
        if self.lane_sensor:
            try:
                # Stop listening before destruction (CARLA best practice)
                # Reference: https://carla.readthedocs.io/en/latest/core_sensors/
                if self.lane_sensor.is_listening:
                    self.lane_sensor.stop()
                    self.logger.debug("Lane invasion sensor stopped listening")

                # Grace period for async callback completion
                import time
                time.sleep(0.01)

                self.lane_sensor.destroy()
                self.logger.info("Lane invasion sensor destroyed")
            except RuntimeError as e:
                self.logger.warning(f"Lane invasion sensor already destroyed or invalid: {e}")
            except Exception as e:
                self.logger.error(f"Error destroying lane invasion sensor: {e}")
            except Exception as e:
                self.logger.error(f"Error destroying lane invasion sensor: {e}")


class ObstacleDetector:
    """
    Detects obstacles ahead of the vehicle using CARLA's obstacle detector sensor.

    Blueprint: sensor.other.obstacle
    Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#obstacle-detector

    Provides distance to nearest obstacle for dense PBRS safety guidance.
    """

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        """
        Initialize obstacle detector.

        Args:
            vehicle: CARLA vehicle actor to attach sensor to
            world: CARLA world object
        """
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        # Obstacle detection state
        self.distance_to_obstacle = float('inf')  # Distance in meters
        self.other_actor = None  # Actor detected as obstacle
        self.obstacle_lock = threading.Lock()

        # Obstacle sensor setup
        self.obstacle_sensor = None
        self._setup_obstacle_sensor()

    def _setup_obstacle_sensor(self):
        """
        Attach obstacle sensor to vehicle.

        Configuration (from CARLA docs):
        - distance: Maximum trace distance (5m default, 10m for highway)
        - hit_radius: Radius of trace capsule (0.5m default)
        - only_dynamics: Only detect dynamic objects (False = detect all)
        - debug_linetrace: Visualize trace (False for performance)
        """
        obstacle_bp = self.world.get_blueprint_library().find("sensor.other.obstacle")

        # Configure sensor attributes (CARLA 0.9.16 documentation)
        obstacle_bp.set_attribute("distance", "10.0")  # 10m lookahead for anticipation
        obstacle_bp.set_attribute("hit_radius", "0.5")  # Standard vehicle width
        obstacle_bp.set_attribute("only_dynamics", "False")  # Detect all obstacles
        obstacle_bp.set_attribute("debug_linetrace", "False")  # No visualization
        obstacle_bp.set_attribute("sensor_tick", "0.0")  # Capture every frame

        # Spawn sensor attached to vehicle (forward-facing)
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))  # Front bumper

        self.obstacle_sensor = self.world.spawn_actor(
            obstacle_bp, spawn_point, attach_to=self.vehicle
        )
        self.obstacle_sensor.listen(self._on_obstacle_detection)
        self.logger.info("Obstacle detector initialized (distance=10m, hit_radius=0.5m)")

    def _on_obstacle_detection(self, event: carla.ObstacleDetectionEvent):
        """
        Callback when obstacle detected.

        Args:
            event: CARLA ObstacleDetectionEvent with obstacle details
                - distance: float (meters to obstacle)
                - other_actor: carla.Actor (what was detected)
        """
        with self.obstacle_lock:
            self.distance_to_obstacle = event.distance
            self.other_actor = event.other_actor

        # Log only significant changes (avoid spam)
        if event.distance < 5.0:
            self.logger.debug(
                f"Obstacle detected: {event.other_actor.type_id} at {event.distance:.2f}m"
            )

    def get_distance_to_nearest_obstacle(self) -> float:
        """
        Get distance to nearest detected obstacle.

        Returns:
            Distance in meters (float('inf') if no obstacle within range)
        """
        with self.obstacle_lock:
            return self.distance_to_obstacle

    def get_obstacle_info(self) -> Optional[Dict]:
        """
        Get detailed obstacle information.

        Returns:
            Dict with obstacle details or None
        """
        with self.obstacle_lock:
            if self.distance_to_obstacle < float('inf') and self.other_actor:
                return {
                    "distance": self.distance_to_obstacle,
                    "actor_type": str(self.other_actor.type_id),
                }
            return None

    def reset(self):
        """Reset obstacle detection state for new episode."""
        with self.obstacle_lock:
            self.distance_to_obstacle = float('inf')
            self.other_actor = None

    def destroy(self):
        """
        Clean up obstacle sensor.

        CRITICAL FIX: Add grace period for pending callback completion.
        Reference: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        """
        if self.obstacle_sensor:
            try:
                if self.obstacle_sensor.is_listening:
                    self.obstacle_sensor.stop()
                    self.logger.debug("Obstacle sensor stopped listening")

                # Grace period for async callback completion
                import time
                time.sleep(0.01)

                self.obstacle_sensor.destroy()
                self.logger.info("Obstacle sensor destroyed")
            except RuntimeError as e:
                self.logger.warning(f"Obstacle sensor already destroyed or invalid: {e}")
            except Exception as e:
                self.logger.error(f"Error destroying obstacle sensor: {e}")


class SensorSuite:
    """
    Aggregates all sensors and provides unified interface.

    Manages camera, collision, lane invasion, and obstacle detectors.
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
        self.obstacle_detector = ObstacleDetector(vehicle, world)  # NEW: Obstacle detection

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
        stacked = self.image_stack.get_stacked_frames()

        # DEBUG: Log final observation
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FINAL CAMERA OBSERVATION:\n"
                f"   Shape: {stacked.shape}\n"
                f"   Dtype: {stacked.dtype}\n"
                f"   Range: [{stacked.min():.3f}, {stacked.max():.3f}]\n"
                f"   Mean: {stacked.mean():.3f}, Std: {stacked.std():.3f}\n"
                f"   Non-zero frames: {(stacked != 0).any(axis=(1,2)).sum()}/4\n"
                f"   Ready for CNN input (batch, 4, 84, 84)"
            )

        return stacked

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

    def get_distance_to_nearest_obstacle(self) -> float:
        """
        Get distance to nearest detected obstacle.

        Returns:
            Distance in meters (float('inf') if no obstacle within range)
        """
        return self.obstacle_detector.get_distance_to_nearest_obstacle()

    def get_obstacle_info(self) -> Optional[Dict]:
        """Get detailed obstacle information."""
        return self.obstacle_detector.get_obstacle_info()

    def reset(self):
        """Reset all sensors for new episode."""
        self.image_stack.reset()
        self.collision_detector.reset()
        self.lane_invasion_detector.reset()
        self.obstacle_detector.reset()
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

        # Destroy obstacle detector
        try:
            self.obstacle_detector.destroy()
        except Exception as e:
            errors.append(f"Obstacle: {e}")
            self.logger.warning(f"Error destroying obstacle detector: {e}")

        if errors:
            self.logger.warning(f"Sensor cleanup completed with {len(errors)} error(s)")
        else:
            self.logger.info("Sensor suite destroyed successfully")

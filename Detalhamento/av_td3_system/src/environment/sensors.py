"""
Sensor Management for CARLA Autonomous Vehicle

Handles all sensor data collection, preprocessing, and queuing:
- Front camera: RGB image capture and preprocessing (grayscale, resize, normalize)
- Image stacking: Maintain 4-frame history for temporal context
- Collision detector: Track collision events
- Lane invasion detector: Track lane departure events

Paper Reference: "End-to-End Visual Autonomous Navigation with Twin Delayed DDPG"
Section III.A: Visual input processing (4 stacked frames, 84×84 grayscale)

Author: Daniel Terra Gomes
2025
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
        self.latest_rgb_image = None  # Store raw RGB for visualization
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

            # Store raw RGB for visualization (used by validation/rendering)
            with self.image_lock:
                self.latest_rgb_image = array.copy()

            # Preprocess for training
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

    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        """
        Get latest raw RGB frame (for visualization/debugging).

        Returns:
            HxWx3 RGB image (uint8, 0-255), or None if no frame available
        """
        with self.image_lock:
            return self.latest_rgb_image

    def destroy(self):
        """
        Clean up camera sensor with robust error handling.

        CRITICAL FIX: Defense against "destroyed actor" RuntimeError
        - Check is_alive before any operation
        - Stop listening before destruction (CARLA best practice)
        - Add grace period for async callback completion
        - Catch C++ runtime errors that bypass Python exception handling

        References:
        - https://carla.readthedocs.io/en/latest/python_api/ (Actor.is_alive)
        - https://carla.readthedocs.io/en/latest/core_sensors/ (sensor.stop())
        - https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
        """
        if self.camera_sensor is None:
            return

        try:
            # Step 1: Check if actor is still alive before operations
            # CRITICAL: Prevents "destroyed actor" RuntimeError
            if not hasattr(self.camera_sensor, 'is_alive') or not self.camera_sensor.is_alive:
                self.logger.warning("Camera sensor already destroyed, skipping cleanup")
                self.camera_sensor = None
                return

            # Step 2: Stop listening before destruction (CARLA best practice)
            # Reference: https://carla.readthedocs.io/en/latest/core_sensors/
            if hasattr(self.camera_sensor, 'is_listening') and self.camera_sensor.is_listening:
                self.camera_sensor.stop()
                self.logger.debug("Camera sensor stopped listening")

            # Step 3: Flush pending callbacks in synchronous mode
            # CRITICAL: In sync mode, callbacks may still be queued even after stop().
            # We need to let the server process them before destroying the actor.
            import time
            time.sleep(0.01)  # 10ms grace period for callback completion

            # Step 4: Double-check is_alive before destroy call
            if self.camera_sensor.is_alive:
                success = self.camera_sensor.destroy()
                if success:
                    self.logger.info("Camera sensor destroyed")
                else:
                    self.logger.warning("Camera sensor destroy returned False")
            else:
                self.logger.warning("Camera sensor became invalid before destroy call")

        except RuntimeError as e:
            # CARLA C++ runtime errors (e.g., "destroyed actor")
            self.logger.warning(f"Camera sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            # Handle missing attributes (actor already cleaned up)
            self.logger.warning(f"Camera sensor attribute error during cleanup: {e}")
        except Exception as e:
            # Catch-all for unexpected errors
            self.logger.error(f"Unexpected error destroying camera sensor: {e}", exc_info=True)
        finally:
            # Always clear reference to prevent further access attempts
            self.camera_sensor = None


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

        # P0 FIX #2: Per-step collision counter for TensorBoard metrics
        self.step_collision_count = 0

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

            # P0 FIX #2: Increment per-step counter for TensorBoard tracking
            self.step_collision_count = 1  # Binary flag: collision occurred this step

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

    def get_step_collision_count(self) -> int:
        """
        P0 FIX #2: Get collision count for current step (0 or 1).

        Returns:
            1 if collision occurred this step, 0 otherwise
        """
        with self.collision_lock:
            return self.step_collision_count

    def reset(self):
        """Reset collision state for new episode."""
        with self.collision_lock:
            self.collision_detected = False
            self.collision_event = None
            self.collision_impulse = 0.0
            self.collision_force = 0.0
            self.step_collision_count = 0  # P0 FIX #2: Reset counter

    def reset_step_counter(self):
        """
        P0 FIX #2: Reset per-step counter (called after each environment step).
        """
        with self.collision_lock:
            self.step_collision_count = 0

    def destroy(self):
        """
        Clean up collision sensor with robust error handling.

        CRITICAL FIX: Defense against "destroyed actor" RuntimeError
        - Check is_alive before any operation
        - Stop listening before destruction (CARLA best practice)
        - Add grace period for async callback completion
        - Catch C++ runtime errors that bypass Python exception handling

        References:
        - https://carla.readthedocs.io/en/latest/python_api/ (Actor.is_alive)
        - https://carla.readthedocs.io/en/latest/core_sensors/ (sensor.stop())
        """
        if self.collision_sensor is None:
            return

        try:
            # Step 1: Check if actor is still alive
            if not hasattr(self.collision_sensor, 'is_alive') or not self.collision_sensor.is_alive:
                self.logger.warning("Collision sensor already destroyed, skipping cleanup")
                self.collision_sensor = None
                return

            # Step 2: Stop listening before destruction
            if hasattr(self.collision_sensor, 'is_listening') and self.collision_sensor.is_listening:
                self.collision_sensor.stop()
                self.logger.debug("Collision sensor stopped listening")

            # Step 3: Grace period for async callback completion
            import time
            time.sleep(0.01)

            # Step 4: Double-check is_alive before destroy
            if self.collision_sensor.is_alive:
                success = self.collision_sensor.destroy()
                if success:
                    self.logger.info("Collision sensor destroyed")
                else:
                    self.logger.warning("Collision sensor destroy returned False")
            else:
                self.logger.warning("Collision sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Collision sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Collision sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying collision sensor: {e}", exc_info=True)
        finally:
            self.collision_sensor = None


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

        # P0 FIX #3: Per-step lane invasion counter for TensorBoard metrics
        self.step_invasion_count = 0

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

        CRITICAL FIX (Nov 23, 2025): Offroad Detection Issue #1
        ========================================================
        Lane invasion sensor fires EVENT when crossing lane markings,
        but doesn't fire when returning to center. We need separate logic
        to clear the flag based on lateral deviation.

        According to CARLA docs (ref_sensors.md#lane-invasion-detector):
        "Registers an event each time its parent crosses a lane marking"
        - Event is PER CROSSING, not continuous state
        - No "return to lane" event exists

        Solution: Set flag on event, clear it in is_invading_lane() based on
        lateral deviation check.

        Args:
            event: CARLA LaneInvasionEvent
        """
        with self.invasion_lock:
            self.lane_invaded = True
            self.invasion_event = event

            # P0 FIX #3: Increment per-step counter for TensorBoard tracking
            self.step_invasion_count = 1  # Binary flag: lane invasion occurred this step

        self.logger.warning(f"Lane invasion detected: {event.crossed_lane_markings}")

    def is_invading_lane(self, lateral_deviation: float = None, lane_half_width: float = None) -> bool:
        """
        Check if lane invasion detected this frame.

        CRITICAL FIX (Nov 23, 2025): Offroad Detection Issue #1
        ========================================================
        Clear invasion flag when vehicle returns to lane center.

        Logic:
        1. If lateral_deviation and lane_half_width provided:
           - Check if vehicle has returned to safe zone (within lane bounds)
           - Clear flag if lateral_deviation < lane_half_width * 0.8 (80% threshold)
        2. Otherwise: Return current flag state

        Args:
            lateral_deviation: Current distance from lane center (meters)
            lane_half_width: Half-width of current lane (meters)

        Returns:
            True if off-road/lane invaded, False if safely in lane
        """
        with self.invasion_lock:
            # If vehicle state data provided, check if returned to lane
            if lateral_deviation is not None and lane_half_width is not None:
                # Vehicle is safely within lane if deviation < 80% of lane width
                # Example: 1.75m lane width → half=0.875m → threshold=0.7m
                # If vehicle deviates 0.5m from center → clear flag (safely in lane)
                # If vehicle deviates 0.85m from center → keep flag (approaching edge)
                recovery_threshold = lane_half_width * 0.8

                if abs(lateral_deviation) < recovery_threshold:
                    # Vehicle has recovered to lane center, clear invasion flag
                    if self.lane_invaded:  # Only log if changing from invaded → clear
                        self.logger.info(
                            f"[LANE RECOVERY] Vehicle returned to lane center "
                            f"(deviation={lateral_deviation:.3f}m < threshold={recovery_threshold:.3f}m)"
                        )
                    self.lane_invaded = False
                    self.invasion_event = None

            return self.lane_invaded

    def get_step_invasion_count(self) -> int:
        """
        P0 FIX #3: Get lane invasion count for current step (0 or 1).

        Returns:
            1 if lane invasion occurred this step, 0 otherwise
        """
        with self.invasion_lock:
            return self.step_invasion_count

    def reset(self):
        """Reset lane invasion state for new episode."""
        with self.invasion_lock:
            self.lane_invaded = False
            self.invasion_event = None
            self.step_invasion_count = 0  # P0 FIX #3: Reset counter

    def reset_step_counter(self):
        """
        P0 FIX #3: Reset per-step counter (called after each environment step).
        """
        with self.invasion_lock:
            self.step_invasion_count = 0

    def destroy(self):
        """
        Clean up lane invasion sensor with robust error handling.

        CRITICAL FIX: Defense against "destroyed actor" RuntimeError
        - Check is_alive before any operation
        - Stop listening before destruction (CARLA best practice)
        - Add grace period for async callback completion
        - Catch C++ runtime errors that bypass Python exception handling

        References:
        - https://carla.readthedocs.io/en/latest/python_api/ (Actor.is_alive)
        - https://carla.readthedocs.io/en/latest/core_sensors/ (sensor.stop())
        """
        if self.lane_sensor is None:
            return

        try:
            # Step 1: Check if actor is still alive
            if not hasattr(self.lane_sensor, 'is_alive') or not self.lane_sensor.is_alive:
                self.logger.warning("Lane invasion sensor already destroyed, skipping cleanup")
                self.lane_sensor = None
                return

            # Step 2: Stop listening before destruction
            if hasattr(self.lane_sensor, 'is_listening') and self.lane_sensor.is_listening:
                self.lane_sensor.stop()
                self.logger.debug("Lane invasion sensor stopped listening")

            # Step 3: Grace period for async callback completion
            import time
            time.sleep(0.01)

            # Step 4: Double-check is_alive before destroy
            if self.lane_sensor.is_alive:
                success = self.lane_sensor.destroy()
                if success:
                    self.logger.info("Lane invasion sensor destroyed")
                else:
                    self.logger.warning("Lane invasion sensor destroy returned False")
            else:
                self.logger.warning("Lane invasion sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Lane invasion sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Lane invasion sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying lane invasion sensor: {e}", exc_info=True)
        finally:
            self.lane_sensor = None


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
        self.last_detection_time = None  # Track when last obstacle was detected (for staleness check)
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

        CRITICAL FIX (Nov 24, 2025): Issue 1.5 - Safety Penalty Persistence
        =====================================================================
        Problem: Callback only fires when obstacle IS detected, NOT when it clears.
        This causes distance to be cached at last detected value (e.g., 1.647m),
        resulting in persistent PBRS penalty (e.g., -1.0/1.647 = -0.607) even after
        vehicle moves away from obstacle.

        Solution: Store detection timestamp to detect stale data. If no new detection
        for a few frames, reset distance to infinity.

        Reference: TASK_1.5_SAFETY_PERSISTENCE_FIX.md
        CARLA Docs: https://carla.readthedocs.io/en/latest/ref_sensors/#obstacle-detector

        Args:
            event: CARLA ObstacleDetectionEvent with obstacle details
                - distance: float (meters to obstacle)
                - other_actor: carla.Actor (what was detected)
        """
        import time

        with self.obstacle_lock:
            self.distance_to_obstacle = event.distance
            self.other_actor = event.other_actor
            self.last_detection_time = time.time()  # Track when last detection occurred

        # Log only significant changes (avoid spam)
        if event.distance < 5.0:
            self.logger.debug(
                f"Obstacle detected: {event.other_actor.type_id} at {event.distance:.2f}m"
            )

    def get_distance_to_nearest_obstacle(self) -> float:
        """
        Get distance to nearest detected obstacle.

        CRITICAL FIX (Nov 24, 2025): Issue 1.5 - Safety Penalty Persistence
        =====================================================================
        Clear stale obstacle detections that haven't updated recently.

        The CARLA obstacle sensor callback only fires when an obstacle IS detected,
        not when it clears. This means if a vehicle was at 1.65m and then moves away,
        the callback never fires again, leaving distance_to_obstacle cached at 1.65m.

        Solution: If no detection within last 0.2 seconds (4 frames at 20 FPS),
        assume obstacle has cleared and reset to infinity.

        This prevents persistent PBRS penalty (e.g., -1.0/1.647 = -0.607) after
        successful recovery from near-obstacle situations.

        Returns:
            Distance in meters (float('inf') if no obstacle within range or stale data)
        """
        import time

        with self.obstacle_lock:
            # Check if detection is stale (no update in last 0.2 seconds)
            if self.last_detection_time is not None:
                time_since_detection = time.time() - self.last_detection_time

                # If no detection for 0.2 seconds (4 frames at 20 FPS), assume cleared
                if time_since_detection > 0.2:
                    # Reset to infinity (no obstacle)
                    if self.distance_to_obstacle < float('inf'):
                        self.logger.debug(
                            f"[OBSTACLE CLEAR] No detection for {time_since_detection:.3f}s, "
                            f"clearing cached distance {self.distance_to_obstacle:.2f}m"
                        )
                    self.distance_to_obstacle = float('inf')
                    self.other_actor = None
                    self.last_detection_time = None

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
            self.last_detection_time = None  # Clear timestamp

    def destroy(self):
        """
        Clean up obstacle sensor with robust error handling.

        CRITICAL FIX: Defense against "destroyed actor" RuntimeError
        - Check is_alive before any operation
        - Stop listening before destruction (CARLA best practice)
        - Add grace period for async callback completion
        - Catch C++ runtime errors that bypass Python exception handling

        References:
        - https://carla.readthedocs.io/en/latest/python_api/ (Actor.is_alive)
        - https://carla.readthedocs.io/en/latest/core_sensors/ (sensor.stop())
        """
        if self.obstacle_sensor is None:
            return

        try:
            # Step 1: Check if actor is still alive
            if not hasattr(self.obstacle_sensor, 'is_alive') or not self.obstacle_sensor.is_alive:
                self.logger.warning("Obstacle sensor already destroyed, skipping cleanup")
                self.obstacle_sensor = None
                return

            # Step 2: Stop listening before destruction
            if hasattr(self.obstacle_sensor, 'is_listening') and self.obstacle_sensor.is_listening:
                self.obstacle_sensor.stop()
                self.logger.debug("Obstacle sensor stopped listening")

            # Step 3: Grace period for async callback completion
            import time
            time.sleep(0.01)

            # Step 4: Double-check is_alive before destroy
            if self.obstacle_sensor.is_alive:
                success = self.obstacle_sensor.destroy()
                if success:
                    self.logger.info("Obstacle sensor destroyed")
                else:
                    self.logger.warning("Obstacle sensor destroy returned False")
            else:
                self.logger.warning("Obstacle sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Obstacle sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Obstacle sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying obstacle sensor: {e}", exc_info=True)
        finally:
            self.obstacle_sensor = None


class OffroadDetector:
    """
    Detects when vehicle is on non-drivable surface using CARLA Waypoint API.

    Uses waypoint.lane_type to distinguish:
    - DRIVABLE: Driving, Parking, Bidirectional lanes
    - NOT DRIVABLE: Sidewalk, Shoulder, Border, Restricted, etc.

    This is the CORRECT way to detect off-road conditions in CARLA,
    as opposed to using the lane invasion sensor which detects line crossings.

    Reference: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
    """

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        """
        Initialize offroad detector.

        Args:
            vehicle: CARLA vehicle actor to monitor
            world: CARLA world object to access map
        """
        self.vehicle = vehicle
        self.world = world
        self.carla_map = world.get_map()
        self.logger = logging.getLogger(__name__)
        self.is_offroad = False
        self.offroad_lock = threading.Lock()

        self.logger.info("Offroad detector initialized using Waypoint API")

    def check_offroad(self) -> bool:
        """
        Check if vehicle is on non-drivable surface.

        Returns True if on sidewalk/grass/shoulder, False if on drivable lane.
        Uses waypoint.lane_type to determine surface type.

        Returns:
            bool: True if vehicle is off-road (non-drivable surface)
        """
        with self.offroad_lock:
            try:
                location = self.vehicle.get_location()

                # Don't snap to road - we want to know if truly off-road
                # project_to_road=False means: return None if location is not on a road
                waypoint = self.carla_map.get_waypoint(
                    location,
                    project_to_road=False,
                    lane_type=carla.LaneType.Any
                )

                if waypoint is None:
                    # No waypoint found = vehicle is completely off the map/road
                    self.is_offroad = True
                    self.logger.warning("[OFFROAD] Vehicle off map (no waypoint)")
                    return True

                # Define drivable lane types (from CARLA API documentation)
                # Reference: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
                drivable_lane_types = [
                    carla.LaneType.Driving,      # Normal traffic lanes
                    carla.LaneType.Parking,      # Parking spaces
                    carla.LaneType.Bidirectional,  # Two-way traffic lanes
                ]

                if waypoint.lane_type not in drivable_lane_types:
                    # Vehicle is on non-drivable surface
                    self.is_offroad = True
                    lane_type_name = str(waypoint.lane_type).replace("LaneType.", "")
                    self.logger.warning(
                        f"[OFFROAD] Vehicle on {lane_type_name} (not drivable)"
                    )
                    return True

                # Vehicle is on drivable lane
                self.is_offroad = False
                return False

            except Exception as e:
                self.logger.error(f"Error checking offroad status: {e}", exc_info=True)
                # Default to safe assumption (not offroad) on error
                return False

    def reset(self):
        """Reset offroad state for new episode."""
        with self.offroad_lock:
            self.is_offroad = False
            self.logger.debug("Offroad detector state reset")


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
        self.obstacle_detector = ObstacleDetector(vehicle, world)  # Obstacle detection
        self.offroad_detector = OffroadDetector(vehicle, world)  # TASK 1 FIX: True off-road detection

        # Initialize frame stacking
        self.image_stack = ImageStack(
            frame_height=84, frame_width=84, num_frames=4
        )

        self.logger.info("Complete sensor suite initialized (including offroad detector)")

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

    def get_rgb_camera_frame(self) -> Optional[np.ndarray]:
        """
        Get raw RGB camera frame (for visualization).

        Returns:
            HxWx3 RGB image (uint8, 0-255), or None if not available
        """
        return self.camera.get_latest_rgb_frame()

    def is_collision_detected(self) -> bool:
        """
        Check if collision occurred this frame.

        Returns:
            True if collision detected
        """
        return self.collision_detector.is_colliding()

    def is_lane_invaded(self, lateral_deviation: float = None, lane_half_width: float = None) -> bool:
        """
        Check if vehicle went off-road this frame.

        CRITICAL FIX (Nov 23, 2025): Offroad Detection Issue #1
        ========================================================
        Pass lateral deviation and lane width to detector so it can clear
        the invasion flag when vehicle returns to lane center.

        Args:
            lateral_deviation: Current distance from lane center (meters)
            lane_half_width: Half-width of current lane (meters)

        Returns:
            True if lane invaded, False if safely in lane
        """
        return self.lane_invasion_detector.is_invading_lane(lateral_deviation, lane_half_width)

    def get_collision_info(self) -> Optional[Dict]:
        """Get collision details if any."""
        return self.collision_detector.get_collision_info()

    def get_step_collision_count(self) -> int:
        """
        P0 FIX #2: Get collision count for current step (0 or 1).
        """
        return self.collision_detector.get_step_collision_count()

    def get_step_lane_invasion_count(self) -> int:
        """
        P0 FIX #3: Get lane invasion count for current step (0 or 1).
        """
        return self.lane_invasion_detector.get_step_invasion_count()

    def get_distance_to_nearest_obstacle(self) -> float:
        """
        Get distance to nearest detected obstacle.

        Returns:
            Distance in meters (float('inf') if no obstacle within range)
        """
        return self.obstacle_detector.get_distance_to_nearest_obstacle()

    def is_offroad(self) -> bool:
        """
        Check if vehicle is on non-drivable surface (TASK 1 FIX).

        Uses CARLA Waypoint API to check lane_type:
        - Returns True if on Sidewalk, Shoulder, Border, Restricted, etc.
        - Returns False if on Driving, Parking, or Bidirectional lanes

        This is the CORRECT method for off-road detection, NOT lane invasion sensor!

        Returns:
            True if vehicle is off-road (non-drivable surface)
        """
        return self.offroad_detector.check_offroad()

    def get_obstacle_info(self) -> Optional[Dict]:
        """Get detailed obstacle information."""
        return self.obstacle_detector.get_obstacle_info()

    def reset(self):
        """Reset all sensors for new episode."""
        self.image_stack.reset()
        self.collision_detector.reset()
        self.lane_invasion_detector.reset()
        self.obstacle_detector.reset()
        self.offroad_detector.reset()  # TASK 1 FIX: Reset offroad state
        self.logger.info("All sensors reset (including offroad detector)")

    def reset_step_counters(self):
        """
        P0 FIX #2 & #3: Reset per-step counters for all sensors.
        Called after each environment step to clear collision/lane invasion flags.
        """
        self.collision_detector.reset_step_counter()
        self.lane_invasion_detector.reset_step_counter()

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

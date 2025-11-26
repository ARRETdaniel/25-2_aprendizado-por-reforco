"""
CARLA Autonomous Vehicle Environment (Gym Interface)

Implements OpenAI Gym interface for CARLA vehicle control with:
- TD3-compatible observation and action spaces
- Multi-component reward function
- Full sensor integration (camera, collision, lane invasion)
- Waypoint-based navigation
- Episode management and termination logic

Paper Reference: "End-to-End Visual Autonomous Navigation with Twin Delayed DDPG"
Section III: System Architecture and Environment Setup
"""

import numpy as np
import logging
import math  # For yaw calculation from waypoint direction
import time
import os
from typing import Dict, Tuple, Optional, Any
from collections import deque
import yaml

import carla
from gymnasium import Env, spaces

# Relative imports
from .sensors import SensorSuite
from .waypoint_manager import WaypointManager
from .dynamic_route_manager import DynamicRouteManager
from .reward_functions import RewardCalculator


class CARLANavigationEnv(Env):
    """
    CARLA autonomous vehicle navigation environment.

    Gym interface for training DRL agents (TD3/DDPG) on autonomous driving in CARLA.

    State Space:
    - Visual: 4 stacked 84×84 grayscale frames (from front camera)
    - Kinematic: velocity, lateral deviation, heading error
    - Navigation: 10 future waypoints (local coordinates)
    - Total: Dict with 'image' (4×84×84) + 'vector' (535-dim)

    Action Space:
    - Continuous 2D: [steering, throttle_brake]
    - Ranges: [-1, 1] for both
    - Maps to CARLA: steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]

    Reward:
    - Multi-component: efficiency, lane-keeping, comfort, safety
    - Weighted sum of components
    """

    def __init__(
        self,
        carla_config_path: str,
        td3_config_path: str,
        training_config_path: str,
        host: str = "localhost",
        port: int = 2000,
        headless: bool = True,
        tm_port: Optional[int] = None,
        use_ros_bridge: bool = False,
    ):
        """
        Initialize CARLA environment.

        Args:
            carla_config_path: Path to carla_config.yaml
            td3_config_path: Path to td3_config.yaml
            training_config_path: Path to training_config.yaml
            host: CARLA server host (default localhost)
            port: CARLA server port (default 2000)
            headless: Whether to run without rendering (default True)
            tm_port: Traffic Manager port (None = use default 8000)
                     Use different ports for training vs evaluation environments
                     to avoid Traffic Manager registry conflicts.
                     Reference: EVALUATION_BUG_ANALYSIS.md
            use_ros_bridge: Whether to publish control commands via ROS 2 Bridge
                           topics instead of direct CARLA API. Default: False.
                           Set to True for ROS integration testing (Phase 5).

        Raises:
            RuntimeError: If cannot connect to CARLA server or invalid config
        """
        self.logger = logging.getLogger(__name__)

        # ROS Bridge control (Phase 5 integration)
        self.use_ros_bridge = use_ros_bridge
        self.ros_interface = None

        if self.use_ros_bridge:
            try:
                print("[ROS BRIDGE] Importing ROSBridgeInterface...")  # DEBUG
                self.logger.info("[ROS BRIDGE] Importing ROSBridgeInterface...")
                from src.utils.ros_bridge_interface import ROSBridgeInterface, ROS2_AVAILABLE

                print(f"[ROS BRIDGE] ROS2_AVAILABLE = {ROS2_AVAILABLE}")  # DEBUG

                if not ROS2_AVAILABLE:
                    print("[ROS BRIDGE] WARNING: rclpy not available, skipping ROS initialization")
                    raise RuntimeError("rclpy not available")

                print("[ROS BRIDGE] Creating ROSBridgeInterface instance...")  # DEBUG
                self.logger.info("[ROS BRIDGE] Creating ROSBridgeInterface instance...")
                self.ros_interface = ROSBridgeInterface(
                    node_name='carla_env_controller',
                    use_docker_exec=False
                )
                print("[ROS BRIDGE] ROSBridgeInterface created successfully!")  # DEBUG
                self.logger.info("[ROS BRIDGE] Initialized ROS Bridge interface for vehicle control")

                # Wait for topics to be available
                print("[ROS BRIDGE] Waiting for ROS topics to be available (10s timeout)...")  # DEBUG
                self.logger.info("[ROS BRIDGE] Waiting for ROS topics to be available (10s timeout)...")
                if not self.ros_interface.wait_for_topics(timeout=10.0):
                    print("[ROS BRIDGE] WARNING: ROS topics not available, but continuing with direct API + ROS publishing")  # DEBUG
                    self.logger.warning("[ROS BRIDGE] ROS topics not available, but continuing with direct API + ROS publishing")
                    # Don't disable ROS bridge - we still want to publish for monitoring
                else:
                    print("[ROS BRIDGE] Successfully connected to ROS topics")  # DEBUG
                    self.logger.info("[ROS BRIDGE] Successfully connected to ROS topics")
            except Exception as e:
                print(f"[ROS BRIDGE] EXCEPTION: {type(e).__name__}: {e}")  # DEBUG
                self.logger.error(f"[ROS BRIDGE] Failed to initialize ROS Bridge: {type(e).__name__}: {e}")
                import traceback
                self.logger.error(f"[ROS BRIDGE] Traceback:\n{traceback.format_exc()}")
                self.logger.warning("[ROS BRIDGE] Falling back to direct CARLA API control")
                self.use_ros_bridge = False
                self.ros_interface = None

        # Load configurations
        with open(carla_config_path, "r") as f:
            self.carla_config = yaml.safe_load(f)
        with open(td3_config_path, "r") as f:
            self.td3_config = yaml.safe_load(f)
        with open(training_config_path, "r") as f:
            self.training_config = yaml.safe_load(f)

        # Connection parameters
        self.host = host
        self.port = port
        self.tm_port = tm_port  # Custom TM port (None = use default 8000)
        self.client = None
        self.world = None
        self.traffic_manager = None

        # Initialize CARLA connection
        self._connect_to_carla()

        # Map and route setup
        map_name = self.carla_config.get("simulation", {}).get("map", "Town01")
        self.logger.info(f"Loading map: {map_name}")
        self.world = self.client.load_world(map_name)

        # Store original world settings BEFORE any modifications
        # Reference: CLOSE_ANALYSIS.md - Optional Improvement #2
        # Enables full settings restoration in close() for persistent CARLA servers
        self._original_settings = self.world.get_settings()
        self.logger.debug(
            f"Stored original world settings: "
            f"sync={self._original_settings.synchronous_mode}, "
            f"delta={self._original_settings.fixed_delta_seconds}"
        )

        # Synchronous mode setup
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0 / self.carla_config.get("simulation", {}).get("fps", 20)
        )
        self.world.apply_settings(settings)

        # Store fixed_delta_seconds for jerk computation (Critical Fix: Comfort Reward)
        self.fixed_delta_seconds = settings.fixed_delta_seconds

        self.logger.info(f"Synchronous mode enabled: delta={settings.fixed_delta_seconds}s")

        # Disable rendering if headless
        if headless:
            self.world.unload_map_layer(carla.MapLayer.All)

        # Get spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in map")

        # Initialize vehicle and sensors
        self.vehicle = None
        self.sensors = None
        self.npcs = []

        # Waypoint manager (Legacy - for extracting start/end)
        waypoints_file = self.carla_config.get("route", {}).get(
            "waypoints_file", "waypoints.txt"
        )

        # Get CARLA map for proper lateral deviation calculation
        carla_map = self.world.get_map()

        legacy_waypoint_manager = WaypointManager(
            waypoints_file=waypoints_file,
            lookahead_distance=self.carla_config.get("route", {}).get(
                "lookahead_distance", 50.0
            ), #previous change to get 53 dimensions
            num_waypoints_ahead=self.carla_config.get("route", {}).get(
                "num_waypoints_ahead", 25
            ),
            carla_map=carla_map,  # Pass CARLA map for intersection-aware lateral deviation
        )

        # Extract start and end locations from legacy waypoints file
        # These provide fixed, reproducible start/end points for training
        route_start_legacy = legacy_waypoint_manager.waypoints[0]  # (x, y, z)
        route_end_legacy = legacy_waypoint_manager.waypoints[-1]  # (x, y, z)

        self.logger.info(
            f"Route definition (from {waypoints_file}):\n"
            f"  Start: ({route_start_legacy[0]:.2f}, {route_start_legacy[1]:.2f}, {route_start_legacy[2]:.2f})\n"
            f"  End: ({route_end_legacy[0]:.2f}, {route_end_legacy[1]:.2f}, {route_end_legacy[2]:.2f})"
        )

        # Dynamic Route Manager (uses CARLA's GlobalRoutePlanner API)
        # Generates waypoints dynamically using road topology
        self.use_dynamic_routes = self.carla_config.get("route", {}).get(
            "use_dynamic_generation", True
        )

        if self.use_dynamic_routes:
            # Initialize DynamicRouteManager (needs world/map to be loaded)
            self._route_start_location = route_start_legacy
            self._route_end_location = route_end_legacy

            sampling_resolution = self.carla_config.get("route", {}).get(
                "sampling_resolution", 2.0
            )

            try:
                self.route_manager = DynamicRouteManager(
                    carla_world=self.world,
                    start_location=self._route_start_location,
                    end_location=self._route_end_location,
                    sampling_resolution=sampling_resolution,
                    logger=self.logger
                )
                # Create adapter to make route_manager compatible with WaypointManager interface
                self.waypoint_manager = self._create_waypoint_manager_adapter()
                self.logger.info(
                    f"DynamicRouteManager initialized: {len(self.route_manager.waypoints)} waypoints, "
                    f"~{self.route_manager.get_route_length():.0f}m route length"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize DynamicRouteManager: {e}\n"
                    f"Falling back to legacy waypoint manager"
                )
                self.use_dynamic_routes = False
                self.waypoint_manager = legacy_waypoint_manager
                self.route_manager = None
        else:
            # Fallback: use legacy waypoint manager
            self.waypoint_manager = legacy_waypoint_manager
            self.route_manager = None
            self.logger.info("Using legacy static waypoints from file")

        # Reward calculator
        reward_config = self.training_config.get("reward", {})
        self.reward_calculator = RewardCalculator(reward_config)

        # Episode state
        self.current_step = 0
        self.episode_start_time = None
        self.episode_count = 0  # Track episode number for diagnostics

        # FIX #3.2: Goal bonus flag (Jan 26, 2025)
        # Ensures +100.0 goal bonus awarded only ONCE per episode
        # Prevents reward inflation from multiple bonuses near goal
        self.goal_bonus_awarded = False

        #  FIX: Read max_time_steps from config (not max_duration_seconds)
        # Config has episode.max_time_steps (5000) directly in steps
        self.max_episode_steps = self.carla_config.get("episode", {}).get("max_time_steps", 1000)

        # Action/observation spaces
        self._setup_spaces()

        # Closed state flag for idempotency
        # Reference: CLOSE_ANALYSIS.md - Optional Improvement #1
        # Provides explicit closed state tracking per Gymnasium best practices
        self._closed = False

        self.logger.info(
            f"CARLA environment initialized: {map_name}, {self.max_episode_steps} steps/episode"
        )

    def _create_waypoint_manager_adapter(self):
        """
        Create adapter to make DynamicRouteManager compatible with WaypointManager interface.

        This allows existing code to continue working while using dynamic route generation.

         FIX: Dynamically calculates num_waypoints_ahead to match lookahead_distance
        and actual sampling_resolution, preventing spacing mismatch bug.

        Returns:
            Object with waypoint_manager interface (waypoints attribute, reset() method, etc.)
        """
        class WaypointManagerAdapter:
            def __init__(self, route_manager, lookahead_distance, sampling_resolution, carla_map):
                self.route_manager = route_manager
                self.lookahead_distance = lookahead_distance
                self.sampling_resolution = sampling_resolution
                self.carla_map = carla_map  # Store CARLA map for lateral deviation

                #  FIX: Calculate num_waypoints dynamically to match actual spacing
                # Before: num_waypoints_ahead was hardcoded (10), assuming 5m spacing
                # After: num_waypoints = lookahead_distance / sampling_resolution
                # Example: 50m / 2m = 25 waypoints (correct for 2m spacing)
                self.num_waypoints_ahead = int(np.ceil(lookahead_distance / sampling_resolution))

                self._current_waypoint_index = 0

                # Log the calculated value for verification
                logger = logging.getLogger(__name__)
                logger.info(
                    f"WaypointManagerAdapter initialized:\n"
                    f"  Lookahead distance: {lookahead_distance}m\n"
                    f"  Sampling resolution: {sampling_resolution}m\n"
                    f"  Calculated waypoints ahead: {self.num_waypoints_ahead}\n"
                    f"  Expected coverage: {self.num_waypoints_ahead * sampling_resolution}m"
                )

            @property
            def waypoints(self):
                """Return waypoints array from route manager."""
                return self.route_manager.get_waypoints()

            def reset(self):
                """Reset waypoint tracking."""
                self._current_waypoint_index = 0

            def get_next_waypoints(self, vehicle_location, current_index=None):
                """
                Get next waypoints ahead of vehicle.

                Args:
                    vehicle_location: carla.Location of vehicle
                    current_index: Optional current waypoint index

                Returns:
                    Array of next waypoints (N, 3) shape
                """
                if current_index is None:
                    current_index = self._current_waypoint_index

                # Update index based on vehicle position
                self._current_waypoint_index = self.route_manager.get_next_waypoint_index(
                    vehicle_location,
                    current_index
                )

                # Get next N waypoints
                start_idx = self._current_waypoint_index
                end_idx = min(
                    start_idx + self.num_waypoints_ahead,
                    len(self.waypoints)
                )

                return self.waypoints[start_idx:end_idx]

            def get_lateral_deviation(self, vehicle_location) -> float:
                """
                Get lateral deviation from lane center using CARLA's OpenDRIVE projection.
                Delegates to the same proper calculation used by WaypointManager.

                Args:
                    vehicle_location: Current vehicle location

                Returns:
                    Lateral deviation in meters
                """
                if not self.carla_map:
                    return 0.0  # Fallback if no map available

                # Convert to carla.Location if needed
                if hasattr(vehicle_location, 'x'):  # Already carla.Location
                    loc = vehicle_location
                else:  # Tuple (x, y, z)
                    loc = carla.Location(x=vehicle_location[0],
                                        y=vehicle_location[1],
                                        z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)

                # Get waypoint at lane center (follows road curvature)
                waypoint = self.carla_map.get_waypoint(
                    loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )

                if waypoint is None:
                    # Vehicle is off-road
                    return float('inf')

                # Calculate 2D distance from vehicle to lane center
                lane_center = waypoint.transform.location
                lateral_deviation = math.sqrt(
                    (loc.x - lane_center.x)**2 +
                    (loc.y - lane_center.y)**2
                )

                return lateral_deviation

        # FIX: Pass sampling_resolution instead of hardcoded num_waypoints_ahead
        sampling_resolution = self.carla_config.get("route", {}).get("sampling_resolution", 2.0)

        # Get CARLA map for lateral deviation calculation
        carla_map = self.world.get_map()

        return WaypointManagerAdapter(
            route_manager=self.route_manager,
            lookahead_distance=self.carla_config.get("route", {}).get("lookahead_distance", 50.0),
            sampling_resolution=sampling_resolution,
            carla_map=carla_map  # Pass map for intersection-aware lateral deviation
        )

    def _connect_to_carla(self, max_retries: int = 5):
        """
        Connect to CARLA server with retry logic.

        Args:
            max_retries: Maximum connection attempts

        Raises:
            RuntimeError: If cannot connect after max_retries
        """
        for attempt in range(max_retries):
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(5.0)
                # Test connection
                self.client.get_server_version()
                self.logger.info(f"Connected to CARLA server at {self.host}:{self.port}")
                return
            except Exception as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Failed to connect to CARLA server after {max_retries} attempts"
                    )

    def _setup_spaces(self):
        """
        Define observation and action spaces (Gym API).

        Observation space:
        - Dict with 'image' (4×84×84 float32 [-1,1]) and 'vector' (kinematic+waypoint state)

        FIX: Dynamically calculates vector size based on actual waypoint count.
        FIX BUG #8: Image space now matches preprocessing output range [-1,1].

        Action space:
        - Box: 2D continuous [-1, 1]
        """
        # Image observation: 4 stacked frames, 84×84, normalized to [-1, 1]
        # FIX BUG #8: Match sensors.py preprocessing output (zero-centered normalization)
        image_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4, 84, 84),
            dtype=np.float32,
        )

        # FIX: Calculate vector observation size dynamically
        # Components: velocity (1) + lateral_dev (1) + heading_err (1) + waypoints (num_waypoints × 2)
        lookahead_distance = self.carla_config.get("route", {}).get("lookahead_distance", 50.0)
        sampling_resolution = self.carla_config.get("route", {}).get("sampling_resolution", 2.0)
        num_waypoints_ahead = int(np.ceil(lookahead_distance / sampling_resolution))

        # Vector size = 3 (kinematic) + (num_waypoints × 2) (x, y coordinates per waypoint)
        vector_size = 3 + (num_waypoints_ahead * 2)

        self.logger.info(
            f"Observation space configuration:\n"
            f"  Image: (4, 84, 84)\n"
            f"  Vector: ({vector_size},) = 3 kinematic + {num_waypoints_ahead} waypoints × 2\n"
            f"  Lookahead: {lookahead_distance}m / {sampling_resolution}m = {num_waypoints_ahead} waypoints"
        )

        vector_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(vector_size,),
            dtype=np.float32,
        )

        # Dict space combining image and vector
        self.observation_space = spaces.Dict(
            {
                "image": image_space,
                "vector": vector_space,
            }
        )

        # Action space: [steering, throttle/brake] in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.logger.info(
            f"Observation space: {self.observation_space}\n"
            f"Action space: {self.action_space}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for new episode.

        Steps:
        1. Clean up previous episode (vehicle, NPCs, sensors)
        2. Spawn new ego vehicle
        3. Attach sensors
        4. Spawn NPC traffic
        5. Initialize state
        6. Return initial observation

        Args:
            seed: Random seed for reproducibility (optional, currently not used)
            options: Additional reset configuration (optional, currently not used)

        Returns:
            observation: Initial observation dict with 'image' and 'vector'
            info: Diagnostic information dict with:
                - episode: Episode number
                - route_length_m: Total route length in meters
                - npc_count: Number of NPC vehicles spawned
                - spawn_location: Vehicle spawn coordinates and heading
                - observation_shapes: Observation tensor shapes

        Raises:
            RuntimeError: If spawn/setup fails
        """
        # Increment episode counter
        self.episode_count += 1

        self.logger.info(f"Resetting environment (episode {self.episode_count})...")

        # Clean up previous episode
        self._cleanup_episode()

        # DYNAMIC ROUTE GENERATION: Use CARLA's GlobalRoutePlanner
        if self.use_dynamic_routes and self.route_manager is not None:
            # Get spawn transform from DynamicRouteManager
            # This uses CARLA's road topology for correct position and orientation
            spawn_point = self.route_manager.get_start_transform()

            self.logger.info(
                f"Using DYNAMIC route (GlobalRoutePlanner):\n"
                f"   Start: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})\n"
                f"   Heading: {spawn_point.rotation.yaw:.2f}°\n"
                f"   Route length: ~{self.route_manager.get_route_length():.0f}m\n"
                f"   Waypoints: {len(self.route_manager.waypoints)}"
            )
        else:
            # FALLBACK: Legacy waypoint-based spawn (for backward compatibility)
            # Get the first waypoint as spawn location
            route_start = self.waypoint_manager.waypoints[0]  # (x, y, z)

            # Calculate initial heading from first two waypoints
            if len(self.waypoint_manager.waypoints) >= 2:
                wp0 = self.waypoint_manager.waypoints[0]
                wp1 = self.waypoint_manager.waypoints[1]
                dx = wp1[0] - wp0[0]  # X-component (East in CARLA)
                dy = wp1[1] - wp0[1]  # Y-component (South in CARLA, +Y direction)

                # FIX BUG #10: CARLA uses LEFT-HANDED coordinate system (Unreal Engine)
                # Standard math: +Y = North (right-handed), atan2(dy, dx) assumes this
                # CARLA/Unreal: +Y = SOUTH (left-handed), 90° yaw points to +Y (South)
                # Solution: Flip Y-axis by negating dy to convert between coordinate systems
                # Reference: https://carla.readthedocs.io/en/latest/python_api/#carlarotation
                # "CARLA uses the Unreal Engine coordinates system. This is a Z-up left-handed system."
                # "Yaw mapping: 0° = East (+X), 90° = South (+Y), 180° = West (-X), 270° = North (-Y)"
                #If we get spawning error fallback:heading_rad = math.atan2(dy, dx)

                heading_rad = math.atan2(-dy, dx)  # Negate dy to flip Y-axis for left-handed system
                initial_yaw = math.degrees(heading_rad)
            else:
                initial_yaw = 0.0
                self.logger.warning("Only one waypoint available, using default yaw=0")

            # Get proper ground-level Z coordinate from CARLA map
            carla_map = self.world.get_map()
            road_location = carla.Location(x=route_start[0], y=route_start[1], z=0.0)
            road_waypoint = carla_map.get_waypoint(
                road_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            if road_waypoint is not None:
                spawn_z = road_waypoint.transform.location.z + 0.5
                self.logger.info(f"Using CARLA map Z: {spawn_z:.2f}m (original: {route_start[2]:.2f}m)")
            else:
                spawn_z = route_start[2] + 0.5
                self.logger.warning(f"Could not get road waypoint, using Z + 0.5m: {spawn_z:.2f}m")

            spawn_point = carla.Transform(
                carla.Location(x=route_start[0], y=route_start[1], z=spawn_z),
                carla.Rotation(pitch=0.0, yaw=initial_yaw, roll=0.0)
            )

            self.logger.info(
                f"Using LEGACY static waypoints:\n"
                f"   Total waypoints in route: {len(self.waypoint_manager.waypoints)}\n"
                f"   Spawn location: ({route_start[0]:.2f}, {route_start[1]:.2f}, {spawn_z:.2f})\n"
                f"   Spawn heading: {initial_yaw:.2f}°\n"
                f"   First 5 waypoints (X, Y, Z):\n" +
                "\n".join([f"      WP{i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})"
                          for i, wp in enumerate(self.waypoint_manager.waypoints[:5])]) +
                f"\n   Route direction: dx={wp1[0]-wp0[0]:.2f}, dy={wp1[1]-wp0[1]:.2f} → yaw={initial_yaw:.2f}°"
            )

        # Spawn ego vehicle (Tesla Model 3)
        vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")

        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.logger.info(f"Ego vehicle spawned successfully at ({route_start[0]:.2f}, {route_start[1]:.2f}, {spawn_z:.2f})")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to spawn ego vehicle: {e}")

        # Attach sensors
        self.sensors = SensorSuite(self.vehicle, self.carla_config, self.world)

        # Spawn NPC traffic
        self._spawn_npc_traffic()

        # Initialize state tracking
        self.current_step = 0
        self.episode_start_time = time.time()

        # FIX #3.2: Reset goal bonus flag (Jan 26, 2025)
        # Ensures each episode can award goal bonus exactly once
        self.goal_bonus_awarded = False

        self.waypoint_manager.reset()
        self.reward_calculator.reset()

        # Tick simulation to initialize sensors AND settle vehicle physics
        self.world.tick()
        self.sensors.tick()

        #  SPAWN VERIFICATION (after physics settled)
        # Reference: ISSUE_1_CORRECTED_ANALYSIS.md - Fix spawn verification timing
        actual_transform = self.vehicle.get_transform()
        forward_vec = actual_transform.get_forward_vector()

        # Calculate expected forward vector from route direction
        if len(self.waypoint_manager.waypoints) >= 2:
            wp0 = self.waypoint_manager.waypoints[0]
            wp1 = self.waypoint_manager.waypoints[1]
            expected_dx = wp1[0] - wp0[0]
            expected_dy = wp1[1] - wp0[1]
            expected_mag = math.sqrt(expected_dx**2 + expected_dy**2)
            expected_fwd = [expected_dx/expected_mag, expected_dy/expected_mag, 0.0] if expected_mag > 0 else [1.0, 0.0, 0.0]

            self.logger.info(
                f"SPAWN VERIFICATION (post-tick):\n"
                f"   Requested spawn yaw: {spawn_point.rotation.yaw:.2f}°\n"
                f"   Actual vehicle yaw: {actual_transform.rotation.yaw:.2f}°\n"
                f"   Actual forward vector: [{forward_vec.x:.3f}, {forward_vec.y:.3f}, {forward_vec.z:.3f}]\n"
                f"   Expected forward (route): [{expected_fwd[0]:.3f}, {expected_fwd[1]:.3f}, {expected_fwd[2]:.3f}]\n"
                f"   Yaw difference: {abs(actual_transform.rotation.yaw - spawn_point.rotation.yaw):.2f}°\n"
                f"   Alignment: {' ALIGNED' if abs(forward_vec.x - expected_fwd[0]) < 0.1 and abs(forward_vec.y - expected_fwd[1]) < 0.1 else ' MISALIGNED'}"
            )

        # Get initial observation
        observation = self._get_observation()

        # Build info dict with diagnostic data (Gymnasium v0.25+ compliance)
        # Reference: RESET_FUNCTION_ANALYSIS.md - Fix 2 (Full Compliance with Info Dict)
        info = {
            "episode": self.episode_count,
            "route_length_m": self.waypoint_manager.get_total_distance() if hasattr(self.waypoint_manager, 'get_total_distance') else len(self.waypoint_manager.waypoints) * 2.0,
            "npc_count": len(self.npcs),
            "spawn_location": {
                "x": spawn_point.location.x,
                "y": spawn_point.location.y,
                "z": spawn_point.location.z,
                "yaw": spawn_point.rotation.yaw,
            },
            "observation_shapes": {
                "image": list(observation['image'].shape),
                "vector": list(observation['vector'].shape),
            },
        }

        self.logger.info(
            f"Episode {info['episode']} reset. "
            f"Route: {info['route_length_m']:.0f}m, "
            f"NPCs: {info['npc_count']}, "
            f"Obs shapes: image {tuple(info['observation_shapes']['image'])}, "
            f"vector {tuple(info['observation_shapes']['vector'])}"
        )

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 2D array [steering, throttle/brake] in [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            - observation: Dict with 'image' and 'vector'
            - reward: Float reward for this step
            - terminated: Boolean, True if episode ended naturally (collision, goal reached)
            - truncated: Boolean, True if episode ended due to time/step limit
            - info: Dict with additional metrics
        """
        # Apply action to vehicle
        self._apply_action(action)

        # Tick CARLA simulation
        self.world.tick()
        self.sensors.tick()

        #  DEBUG: Verify simulation is advancing (first 10 steps)
        if self.current_step < 10:
            snapshot = self.world.get_snapshot()
            self.logger.info(
                f" DEBUG Step {self.current_step} - World State After Tick:\n"
                f"   Frame: {snapshot.frame}\n"
                f"   Timestamp: {snapshot.timestamp.elapsed_seconds:.3f}s\n"
                f"   Delta: {snapshot.timestamp.delta_seconds:.3f}s"
            )

        # Get new state
        observation = self._get_observation()
        vehicle_state = self._get_vehicle_state()

        # Get progress metrics for reward calculation
        vehicle_location = self.vehicle.get_location()

        # FIX #2: Use ROUTE DISTANCE instead of Euclidean to prevent off-road shortcuts
        # Changed from: get_distance_to_goal() (Euclidean, rewards diagonal shortcuts)
        # Changed to: get_route_distance_to_goal() (route-following, prevents shortcuts)
        # Reference: #file:DIAGNOSIS_RIGHT_TURN_BIAS.md, #file:SYSTEMATIC_FIX_ANALYSIS.md
        distance_to_goal = self.waypoint_manager.get_route_distance_to_goal(vehicle_location)

        waypoint_reached = self.waypoint_manager.check_waypoint_reached()

        # FIX #3: Align goal detection threshold with termination threshold (Jan 26, 2025)
        # ==============================================================================
        # Problem: check_goal_reached() default threshold=5.0m, but is_route_finished() uses 3.0m
        # Result: Vehicle gets +100.0 bonus every step from 5.0m to 3.0m (17 times, +1700 total!)
        # This creates reward inflation and perverse incentive to linger near goal.
        #
        # Fix: Pass threshold=3.0 to match is_route_finished() (300 segments = 3.0m)
        # Ensures goal bonus awarded only when vehicle is truly at goal (within 3.0m)
        # Aligns with TD3 terminal state semantics: terminal rewards given once, not continuously
        #
        # Reference:
        # - TD3 paper: y(r,s',d) = r + γ(1-d)min(Q₁,Q₂) → terminal rewards NOT bootstrapped
        # - Gymnasium API: "terminated=True when agent reaches goal state"
        # - GOAL_TERMINATION_BUG_ANALYSIS.md - Investigation of multiple goal bonuses
        goal_detected = self.waypoint_manager.check_goal_reached(vehicle_location, threshold=3.0)

        # FIX #3.2: Only award goal bonus ONCE per episode (Jan 26, 2025)
        # Even with aligned thresholds, vehicle could take multiple steps in 3.0m zone
        # before termination due to Euclidean vs route distance divergence on curves.
        # Solution: Track if bonus already awarded this episode using flag.
        goal_reached = goal_detected and not self.goal_bonus_awarded
        if goal_reached:
            self.goal_bonus_awarded = True
            self.logger.info("[GOAL] First goal detection - bonus will be awarded (flag set)")        # ========================================================================
        # PRIORITY 1 & 3: Get dense safety metrics for PBRS guidance
        # ========================================================================
        # Get distance to nearest obstacle from obstacle detector sensor
        distance_to_nearest_obstacle = self.sensors.get_distance_to_nearest_obstacle()

        # Calculate time-to-collision (TTC)
        # TTC = distance / velocity (undefined if not moving or no obstacle)
        time_to_collision = None
        if (distance_to_nearest_obstacle < float('inf') and
            vehicle_state["velocity"] > 0.1):  # Moving threshold
            time_to_collision = distance_to_nearest_obstacle / vehicle_state["velocity"]

        # Get collision impulse magnitude for graduated penalties
        collision_info = self.sensors.get_collision_info()
        collision_impulse = None
        if collision_info is not None and "impulse" in collision_info:
            collision_impulse = collision_info["impulse"]

        # CRITICAL FIX (Nov 19, 2025): Get per-step sensor counts BEFORE reward calculation
        # Previously lane_invasion_count was only tracked in info dict but NOT passed to reward calculator
        collision_count = self.sensors.get_step_collision_count()
        lane_invasion_count = self.sensors.get_step_lane_invasion_count()

        # Calculate reward with new dense safety guidance
        reward_dict = self.reward_calculator.calculate(
            velocity=vehicle_state["velocity"],
            lateral_deviation=vehicle_state["lateral_deviation"],
            heading_error=vehicle_state["heading_error"],
            acceleration=vehicle_state["acceleration"],
            acceleration_lateral=vehicle_state["acceleration_lateral"],
            collision_detected=self.sensors.is_collision_detected(),
            # TASK 1 FIX (Nov 23, 2025): Use correct offroad detection
            # OLD BUG: Used is_lane_invaded() which detects LINE CROSSINGS (legal lane changes!)
            # NEW FIX: Use is_offroad() which checks WAYPOINT.LANE_TYPE (Sidewalk vs Driving)
            # Reference: https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType
            offroad_detected=self.sensors.is_offroad(),  # TRUE off-road detection (grass, sidewalk)
            wrong_way=vehicle_state["wrong_way"],

            # CRITICAL FIX (Nov 24, 2025): Issue 1.6 - Lane Invasion Penalty Inconsistency
            # =================================================================================
            # OLD BUG: Used is_lane_invaded() which returns PERSISTENT flag that gets cleared
            #          by recovery logic. If vehicle returns to center before reward calculation,
            #          flag is already False even though callback fired this step.
            # NEW FIX: Use get_step_lane_invasion_count() which returns per-step counter (0 or 1)
            #          that accurately tracks whether invasion occurred THIS step, regardless of
            #          whether vehicle has already recovered.
            # Reference: TASK_1.6_LANE_INVASION_INCONSISTENCY_FIX.md
            # CARLA Docs: https://carla.readthedocs.io/en/latest/ref_sensors/#lane-invasion-detector
            lane_invasion_detected=bool(self.sensors.get_step_lane_invasion_count()),  # Per-step detection
            distance_to_goal=distance_to_goal,
            waypoint_reached=waypoint_reached,
            goal_reached=goal_reached,
            lane_half_width=vehicle_state["lane_half_width"],
            dt=vehicle_state["dt"],
            # NEW: Dense safety metrics (Priority 1 & 3 fixes)
            distance_to_nearest_obstacle=distance_to_nearest_obstacle,
            time_to_collision=time_to_collision,
            collision_impulse=collision_impulse,
            # FIX #3: Pass wrong-way penalty value (not just boolean)
            wrong_way_penalty=vehicle_state.get("wrong_way_penalty", 0.0),
        )

        reward = reward_dict["total"]

        # FIX: Increment step counter BEFORE checking termination
        # This ensures timeout check uses correct step count
        self.current_step += 1

        # Check termination conditions
        done, termination_reason = self._check_termination(vehicle_state)

        # Gymnasium API: split done into terminated and truncated
        # terminated: episode ended naturally (collision, goal, off-road)
        # truncated: episode ended due to time/step limit
        truncated = (self.current_step >= self.max_episode_steps) and not done
        terminated = done and not truncated

        # DIAGNOSTIC: Log when approaching max steps
        if self.current_step >= self.max_episode_steps - 10 and not done:
            self.logger.warning(
                f"[DIAGNOSTIC] Approaching max_episode_steps! "
                f"Step {self.current_step}/{self.max_episode_steps}, "
                f"will truncate in {self.max_episode_steps - self.current_step} steps"
            )

        # DIAGNOSTIC: Log truncation
        if truncated:
            self.logger.warning(
                f"[TERMINATION] Episode TRUNCATED at step {self.current_step} "
                f"(reached max_episode_steps={self.max_episode_steps})"
            )

        # Prepare info dict (using counts retrieved before reward calculation)
        info = {
            "step": self.current_step,
            "reward_breakdown": reward_dict["breakdown"],
            # Add validation-friendly flat format for manual control script
            "reward_components": {
                "total": reward,
                "efficiency": reward_dict["breakdown"]["efficiency"][2],  # weighted value
                "lane_keeping": reward_dict["breakdown"]["lane_keeping"][2],
                "comfort": reward_dict["breakdown"]["comfort"][2],
                "safety": reward_dict["breakdown"]["safety"][2],
                "progress": reward_dict["breakdown"]["progress"][2],
            },
            # Add state metrics for validation HUD
            "state": {
                "velocity": vehicle_state["velocity"],
                "lateral_deviation": vehicle_state["lateral_deviation"],
                "heading_error": vehicle_state["heading_error"],
                "distance_to_goal": distance_to_goal,
            },
            "termination_reason": termination_reason,
            "vehicle_state": vehicle_state,
            "collision_info": collision_info,  # Already retrieved above
            "collision_count": collision_count,  # P0 FIX #2: Per-step collision count (retrieved before reward calc)
            "lane_invasion_count": lane_invasion_count,  # P0 FIX #3 + CRITICAL FIX (Nov 19): Used in reward & logged
            "distance_to_goal": distance_to_goal,
            "progress_percentage": self.waypoint_manager.get_progress_percentage(),
            "current_waypoint_idx": self.waypoint_manager.get_current_waypoint_index(),
            "waypoint_reached": waypoint_reached,
            "goal_reached": goal_reached,
            "success": 1 if (termination_reason == "route_completed" and goal_reached) else 0,  # FIX: For evaluation script compatibility
            # NEW: Dense safety metrics for analysis
            "distance_to_nearest_obstacle": distance_to_nearest_obstacle,
            "time_to_collision": time_to_collision,
            "collision_impulse": collision_impulse,
        }

        # P0 FIX #2 & #3: Reset per-step counters after collecting data
        self.sensors.reset_step_counters()

        if terminated or truncated:
            self.logger.info(
                f"Episode ended: {termination_reason} after {self.current_step} steps "
                f"(terminated={terminated}, truncated={truncated})"
            )

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray):
        """
        Apply action to vehicle.

        Maps [-1,1] action to CARLA controls:
        - action[0]: steering ∈ [-1,1] → direct CARLA steering
        - action[1]: throttle/brake ∈ [-1,1]
          - negative: brake (throttle=0, brake=-action[1])
          - positive: throttle (throttle=action[1], brake=0)

        Args:
            action: 2D array [steering, throttle/brake]
        """
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        # Separate throttle and brake
        if throttle_brake > 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # Create CARLA VehicleControl object
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steering,
            hand_brake=False,
            reverse=False,
        )

        # ALWAYS apply control to our vehicle via direct CARLA API
        # This ensures our spawned vehicle moves regardless of ROS Bridge status
        self.vehicle.apply_control(control)

        # ADDITIONALLY publish via ROS 2 topics if ROS Bridge is enabled
        # This allows external ROS 2 nodes to monitor/log our control commands
        if self.use_ros_bridge and self.ros_interface is not None:
            self.ros_interface.publish_control(
                throttle=throttle,
                steer=steering,
                brake=brake,
                hand_brake=False,
                reverse=False
            )

        # DEBUG: Log control application and vehicle response (first 10 steps)
        if self.current_step < 100:
            # Get vehicle velocity
            velocity = self.vehicle.get_velocity()
            speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            speed_kmh = speed_mps * 3.6

            # Get applied control to verify
            applied_control = self.vehicle.get_control()

            self.logger.info(
                f"DEBUG Step {self.current_step}:\n"
                f"   Input Action: steering={action[0]:+.4f}, throttle/brake={action[1]:+.4f}\n"
                f"   Sent Control: throttle={throttle:.4f}, brake={brake:.4f}, steer={steering:.4f}\n"
                f"   Applied Control: throttle={applied_control.throttle:.4f}, brake={applied_control.brake:.4f}, steer={applied_control.steer:.4f}\n"
                f"   Speed: {speed_kmh:.2f} km/h ({speed_mps:.2f} m/s)\n"
                f"   Hand Brake: {applied_control.hand_brake}, Reverse: {applied_control.reverse}, Gear: {applied_control.gear}"
            )

    def _handle_tick_timeout(self):
        """
        Handle CARLA tick timeout by gracefully terminating episode.

        This prevents silent freezes when CARLA simulator hangs (sensor queue
        overflow, Traffic Manager deadlock, physics engine lock, etc.).

        Returns:
            Tuple compatible with step() return: (obs, reward, terminated, truncated, info)
        """
        self.logger.warning(
            f"   Forcing episode termination due to CARLA tick timeout\n"
            f"   Episode: {self.episode_count}, Step: {self.current_step}\n"
            f"   Last waypoint: {self.waypoint_manager.current_waypoint_index}/{len(self.waypoint_manager.waypoints)}\n"
            f"   Recommendation: Check CARLA server logs for deadlock/error"
        )

        # Get last known observation (may be stale)
        try:
            observation = self._get_observation()
        except Exception as e:
            # If even observation fails, return zero observation
            self.logger.error(f"Failed to get observation during timeout recovery: {e}")
            observation = {
                "image": np.zeros((4, 84, 84), dtype=np.float32),
                "vector": np.zeros(53, dtype=np.float32),
            }

        # Terminate episode with penalty
        reward = -100.0  # Heavy timeout penalty
        terminated = True
        truncated = False

        info = {
            "step": self.current_step,
            "episode": self.episode_count,
            "termination_reason": "carla_tick_timeout",
            "timeout_duration": 10.0,
            "reward_total": reward,
            "reward_components": {"timeout_penalty": -100.0},
        }

        # Increment timeout counter for monitoring
        if not hasattr(self, 'timeout_count'):
            self.timeout_count = 0
        self.timeout_count += 1

        self.logger.warning(
            f"   Total timeouts in session: {self.timeout_count}\n"
            f"   If timeouts persist, consider:\n"
            f"   1. Reducing NPC density\n"
            f"   2. Lowering sensor resolution\n"
            f"   3. Restarting CARLA server"
        )

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construct observation from sensors and state.

        FIX BUG #4: Handles variable-length waypoint arrays near route end by padding.
        FIX BUG #9: Normalizes all vector features to comparable scales [-1, 1].

        Returns:
            Dict with:
            - 'image': (4, 84, 84) stacked frames, normalized [-1,1]
            - 'vector': (53,) kinematic + waypoint state (FIXED SIZE, NORMALIZED)
        """
        # Get camera data (4 stacked frames)
        image_obs = self.sensors.get_camera_data()

        # Get vehicle state for vector observation
        vehicle_state = self._get_vehicle_state()

        # Get next waypoints in vehicle frame
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        # Convert CARLA yaw (degrees) to radians for waypoint manager
        vehicle_heading_radians = np.radians(vehicle_transform.rotation.yaw)
        next_waypoints = self.waypoint_manager.get_next_waypoints(
            vehicle_location, vehicle_heading_radians
        )

        # FIX BUG #4: Handle variable-length waypoint arrays near route end
        # Expected: num_waypoints_ahead waypoints (e.g., 25 with 2m spacing)
        # Near route end: may return fewer waypoints
        # Solution: Pad with last waypoint to maintain fixed observation size
        expected_num_waypoints = self.waypoint_manager.num_waypoints_ahead

        if len(next_waypoints) < expected_num_waypoints:
            # Pad with last waypoint to maintain fixed size
            if len(next_waypoints) > 0:
                last_waypoint = next_waypoints[-1]
                padding = np.tile(last_waypoint, (expected_num_waypoints - len(next_waypoints), 1))
                next_waypoints = np.vstack([next_waypoints, padding])
            else:
                # No waypoints available (route finished), use zeros
                next_waypoints = np.zeros((expected_num_waypoints, 2), dtype=np.float32)

        # FIX BUG #9: Normalize all vector features to comparable scales
        # This prevents large-magnitude features (waypoints ~50m) from dominating
        # small-magnitude features (heading error ~π), which was causing training failure.

        # Normalize velocity by max expected urban speed (30 m/s = 108 km/h)
        # Output range: [0, ~1] (can exceed 1 if vehicle goes faster than expected)
        velocity_normalized = vehicle_state["velocity"] / 30.0

        # Normalize lateral deviation by standard lane width (3.5m)
        # Output range: typically [-1, 1] for staying within lane
        lateral_deviation_normalized = vehicle_state["lateral_deviation"] / 3.5

        # Normalize heading error by π radians (maximum possible error)
        # Output range: [-1, 1]
        heading_error_normalized = vehicle_state["heading_error"] / np.pi

        # Normalize waypoints by lookahead distance (50m)
        # Output range: typically [-1, 1] for waypoints within lookahead
        lookahead_distance = self.carla_config.get("route", {}).get("lookahead_distance", 50.0)
        waypoints_normalized = next_waypoints / lookahead_distance

        # Construct normalized vector observation
        # Expected size: 1 (velocity) + 1 (lateral_dev) + 1 (heading_err) + (num_waypoints * 2)
        vector_obs = np.concatenate(
            [
                [velocity_normalized],
                [lateral_deviation_normalized],
                [heading_error_normalized],
                waypoints_normalized.flatten(),
            ]
        ).astype(np.float32)

        # DEBUG: Log observation details every 100 steps (throttled logging)
        if self.current_step % 100 == 0:
            self.logger.info(
                f"OBSERVATION (Step {self.current_step}):\n"
                f"   Vehicle State (Raw):\n"
                f"      Velocity: {vehicle_state['velocity']:.2f} m/s ({vehicle_state['velocity']*3.6:.1f} km/h)\n"
                f"      Lateral deviation: {vehicle_state['lateral_deviation']:.3f} m\n"
                f"      Heading error: {np.degrees(vehicle_state['heading_error']):.2f}° ({vehicle_state['heading_error']:.3f} rad)\n"
                f"   Waypoints (Raw, vehicle frame):\n"
                f"      Total waypoints: {len(next_waypoints)}\n"
                f"      First 3 waypoints: {next_waypoints[:3].tolist() if len(next_waypoints) > 0 else 'None'}\n"
                f"      Lookahead distance: {lookahead_distance:.1f} m\n"
                f"   Normalized Vector Features (passed to TD3/CNN):\n"
                f"      Velocity (normalized): {velocity_normalized:.4f} (÷30.0)\n"
                f"      Lateral dev (normalized): {lateral_deviation_normalized:.4f} (÷3.5)\n"
                f"      Heading err (normalized): {heading_error_normalized:.4f} (÷π)\n"
                f"      Waypoints (normalized): shape={waypoints_normalized.shape}, range=[{waypoints_normalized.min():.3f}, {waypoints_normalized.max():.3f}] (÷{lookahead_distance})\n"
                f"   Final Observation Shapes:\n"
                f"      Image: {image_obs.shape} (dtype={image_obs.dtype}, range=[{image_obs.min():.2f}, {image_obs.max():.2f}])\n"
                f"      Vector: {vector_obs.shape} (dtype={vector_obs.dtype}, sum={vector_obs.sum():.3f})"
            )

        return {
            "image": image_obs,
            "vector": vector_obs,
        }

    def _get_vehicle_state(self) -> Dict[str, float]:
        """
        Get current vehicle kinematics and navigation state.

        Returns:
            Dict with:
            - velocity: m/s
            - acceleration: m/s² (longitudinal)
            - acceleration_lateral: m/s² (lateral)
            - lateral_deviation: m from route
            - heading_error: rad from route heading
            - wrong_way: bool if driving backwards
            - lane_half_width: half of current lane width from CARLA (m)
            - dt: time step since last measurement (seconds) [NEW]
        """
        # Get velocity
        velocity_vec = self.vehicle.get_velocity()
        velocity = np.sqrt(
            velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2
        )

        # Get acceleration
        accel_vec = self.vehicle.get_acceleration()
        acceleration = np.sqrt(
            accel_vec.x**2 + accel_vec.y**2 + accel_vec.z**2
        )

        # Get angular velocity (for lateral acceleration)
        # CARLA returns angular velocity in deg/s, but centripetal acceleration formula requires rad/s
        angular_vel = self.vehicle.get_angular_velocity()
        omega_z_rad = np.radians(angular_vel.z)  # Convert deg/s → rad/s
        acceleration_lateral = abs(velocity * omega_z_rad) if velocity > 0.1 else 0.0

        # Get location and heading
        location = self.vehicle.get_location()
        heading = self.vehicle.get_transform().rotation.yaw

        # Route-relative state
        lateral_deviation = self.waypoint_manager.get_lateral_deviation(location)
        target_heading = self.waypoint_manager.get_target_heading(location)
        heading_error = np.arctan2(
            np.sin(np.radians(heading) - target_heading),
            np.cos(np.radians(heading) - target_heading),
        )

        # ENHANCEMENT: Get lane width from CARLA waypoint API
        # This enables dynamic normalization instead of fixed config tolerance
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is not None:
            # Use actual lane geometry from CARLA (e.g., 2.5m urban, 3.5m highway)
            lane_half_width = waypoint.lane_width / 2.0
        else:
            # Fallback to config value if vehicle is off-road or waypoint not found
            lane_half_width = self.reward_calculator.lateral_tolerance

        # FIX #3 (Nov 24, 2025): Wrong-Way Detection
        # ============================================
        # ISSUE: No wrong-way penalty triggered despite backward movement (Steps 95-96)
        #
        # ROOT CAUSE: Current implementation checks velocity direction, not heading vs. route
        #   - Vehicle facing 180° from goal but stationary → wrong_way=False (no penalty)
        #   - Vehicle moving slowly backward → wrong_way=False (velocity < 0.1 m/s threshold)
        #   - Checks: dot(forward_vec, velocity_vec) < -0.5
        #   - Problem: Physics-based (velocity) not navigation-based (heading vs. route)
        #
        # SOLUTION: Check heading relative to route direction
        #   - Get intended route direction from next waypoint
        #   - Calculate heading error: vehicle_yaw - route_direction
        #   - Wrong-way if: |heading_error| > 90° AND moving (velocity > 0.5 m/s)
        #   - Scale penalty by severity: 90° = -1.0, 180° = -5.0
        #
        # Expected Impact:
        #   - Step 95 (backward, heading ~180° from route): -3.0 to -5.0 penalty
        #   - Total reward: -0.83 → -3.83 to -5.83 (strong discouragement)
        #
        # Reference: CORRECTED_ANALYSIS_SUMMARY.md - Issue #3
        wrong_way_penalty = self._check_wrong_way_penalty(velocity)
        wrong_way = wrong_way_penalty != 0.0  # Boolean for state dict

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "acceleration_lateral": acceleration_lateral,
            "lateral_deviation": lateral_deviation,
            "heading_error": float(heading_error),
            "wrong_way": wrong_way,
            "wrong_way_penalty": wrong_way_penalty,  # NEW: Actual penalty value for reward
            "lane_half_width": lane_half_width,  # NEW: CARLA lane width
            "dt": self.fixed_delta_seconds,  # NEW: Time step for jerk computation
        }

    def _check_wrong_way_penalty(self, velocity: float) -> float:
        """
        Check if vehicle is facing wrong direction relative to route and return penalty.

        FIX #3 (Nov 24, 2025): Wrong-Way Detection Based on Heading vs. Route
        ======================================================================
        FIX #1 (Jan 26, 2025): Use Corrected Heading Calculation (Route Tangent)
        =========================================================================

        Previous Implementation (BUGGY):
            v1 (Before Nov 24): Checked velocity direction vs. vehicle heading
            v2 (Nov 24 - Jan 25): Used vehicle→waypoint bearing (same bug as efficiency!)

            Problem: Calculated route_direction = atan2(next_waypoint - vehicle)
                    - This is vehicle→waypoint bearing, NOT route tangent
                    - Small position changes → large heading calculation errors
                    - Same root cause as efficiency reward bug

        New Implementation (CORRECT):
            - Uses waypoint_manager.get_target_heading() for route tangent
            - Consistent with efficiency reward calculation
            - Single source of truth for route direction

        Algorithm:
            1. Get route tangent direction from waypoint_manager (corrected method)
            2. Calculate heading error: vehicle_yaw - route_tangent
            3. Normalize to [-180°, 180°]
            4. If |heading_error| > 90° AND velocity > 0.5 m/s:
                - Base penalty: -1.0 (at 90°) to -5.0 (at 180°)
                - Scale by velocity (0-1): stationary = 0%, full speed = 100%
            5. Return penalty (negative float)

        Expected Impact:
            - Consistent heading regardless of vehicle position in segment
            - Wrong-way penalty triggers correctly for backward driving
            - No false positives from lateral deviations

        Reference:
            - docs/day-24/ROOT_CAUSE_ANALYSIS_HEADING_ERROR.md
            - CORRECTED_ANALYSIS_SUMMARY.md - Issue #3

        Args:
            velocity: Current vehicle velocity magnitude (m/s)

        Returns:
            Penalty value (0.0 if correct direction, -1.0 to -5.0 if wrong-way)

        Literature Support:
            - Chen et al. (2019): Traffic rule violations need explicit constraints
            - Safety-critical RL: Hard safety constraints require large penalties
        """
        # Early exit if no route plan available
        if not hasattr(self, 'waypoint_manager') or self.waypoint_manager is None:
            return 0.0

        waypoints = self.waypoint_manager.waypoints
        if waypoints is None or len(waypoints) < 2:
            return 0.0

        # Get current vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = vehicle_transform.rotation.yaw  # degrees, [-180, 180]

        #  FIX: Use corrected heading calculation (route tangent, not bearing)
        # This uses the same method as efficiency reward for consistency
        route_direction_rad = self.waypoint_manager.get_target_heading(vehicle_location)
        route_direction_deg = np.degrees(route_direction_rad)  # Convert to degrees

        # Calculate heading error: vehicle_yaw - route_direction
        heading_error = vehicle_yaw - route_direction_deg

        # Normalize to [-180, 180]
        while heading_error > 180.0:
            heading_error -= 360.0
        while heading_error < -180.0:
            heading_error += 360.0

        abs_heading_error = abs(heading_error)

        # Wrong-way threshold: >90° means facing away from route
        if abs_heading_error > 90.0:
            # Calculate severity: 0.0 (at 90°) to 1.0 (at 180°)
            severity = (abs_heading_error - 90.0) / 90.0  # [0, 1]

            # Base penalty scales with severity
            # 90° → -1.0 (slightly wrong)
            # 135° → -3.0 (moderately wrong)
            # 180° → -5.0 (completely backward)
            base_penalty = -1.0 - severity * 4.0

            # Scale by velocity: stationary = 0% penalty, moving = up to 100%
            # Threshold: 0.5 m/s (1.8 km/h) - allows stopped vehicle to recover
            velocity_scale = min(velocity / 2.0, 1.0)  # [0, 1]

            penalty = base_penalty * velocity_scale
            penalty = max(penalty, -5.0)  # Cap at -5.0

            self.logger.warning(
                f"[WRONG-WAY] Heading error: {heading_error:.1f}° "
                f"(vehicle: {vehicle_yaw:.1f}°, route: {route_direction_deg:.1f}°), "
                f"Velocity: {velocity:.2f} m/s, "
                f"Severity: {severity:.2f}, "
                f"Penalty: {penalty:.2f}"
            )

            return penalty

        # Correct direction - no penalty
        return 0.0

    def _check_termination(self, vehicle_state: Dict) -> Tuple[bool, str]:
        """
        Check if episode should terminate naturally (within MDP).

        FIX BUG #11: This function returns TRUE only for NATURAL MDP terminations.
        Time limits are NOT MDP terminations - they are handled as TRUNCATION in step().

        CRITICAL FIX (Lane Invasion Bug): Based on research paper:
        "End-to-End Deep Reinforcement Learning for Lane Keeping Assist"
        Finding: "We concluded that the more we put termination conditions, the slower convergence time to learn"

        Previous Bug: Terminated immediately on ANY lane marking touch → prevented recovery learning
        Fix: Only terminate if COMPLETELY off-road (> 2.0m lateral deviation from lane center)

        Natural MDP Termination Conditions (terminated=True):
        1. Collision detected → immediate termination
        2. Completely off-road (lateral deviation > 2.0m) → safety violation
        3. Route completion (reached goal) → success

        NOT Termination Conditions (allow learning/recovery):
        - Lane marking touch → penalty via reward function, continue episode
        - Small lateral deviations (< 2.0m) → allow correction behavior

        NOT Included (handled as truncation in step()):
        - Max steps / time limit → truncated=True, terminated=False

        This distinction is CRITICAL for TD3 bootstrapping:
        - If terminated=True: Target Q = r + 0 (no future value)
        - If truncated=True: Target Q = r + γ*V(s') (has future value)

        Per Gymnasium API v0.26+ and official TD3 implementation.

        Args:
            vehicle_state: Current vehicle state dict

        Returns:
            Tuple of (done: bool, reason: str)
            - done=True only for natural MDP termination
            - reason: "collision", "off_road", "route_completed", or "running"
        """
        # Collision: immediate termination
        if self.sensors.is_collision_detected():
            self.logger.warning(f"[TERMINATION] Collision detected at step {self.current_step}")
            return True, "collision"

        # FIXED: Off-road detection based on lateral deviation threshold
        # Only terminate if COMPLETELY off-road (> 2.0m from lane center)
        # Lane marking touches are penalized via reward function but do NOT terminate episode
        # This allows agent to learn recovery behavior from mistakes
        lateral_deviation = abs(vehicle_state.get("lateral_deviation", 0.0))
        if lateral_deviation > 2.5:  # meters from lane center
            self.logger.warning(
                f"[TERMINATION] Off-road at step {self.current_step}: "
                f"lateral_deviation={lateral_deviation:.3f}m > 2.0m threshold"
            )
            return True, "off_road"

        # Wrong way: penalize but don't terminate immediately
        # (penalty is in reward function)

        # Route completion
        if self.waypoint_manager.is_route_finished():
            self.logger.info(
                f"[TERMINATION] Route completed at step {self.current_step}! "
                f"Dense waypoint {self.waypoint_manager.get_current_waypoint_index()}/{len(self.waypoint_manager.dense_waypoints)-1}"
            )
            return True, "route_completed"

        # FIX BUG #11: Max steps is NOT an MDP termination condition
        # Time limits should be handled as TRUNCATION in step(), not TERMINATION here.
        # Per official TD3 implementation (main.py line 133):
        #   done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        # This pattern explicitly sets done_bool=0 at time limits to prevent incorrect bootstrapping.
        #
        # Gymnasium API specification (v0.26+):
        #   - terminated: Natural MDP termination (collision, goal, death) → V(s')=0
        #   - truncated: Artificial termination (time limit, bounds) → V(s')≠0
        #
        # REMOVED: if self.current_step >= self.max_episode_steps: return True, "max_steps"
        # Time limit handling is now in step() lines 602-604 as truncation.

        return False, "running"

    def _spawn_npc_traffic(self):
        """
        Spawn NPC vehicles for traffic.

        Uses traffic manager for autonomous control.
        NPC count from training_config.yaml scenarios list.
        """
        # Get configured NPC count from scenarios list
        scenarios = self.training_config.get("scenarios", [])

        # DEBUG: Print config contents
        self.logger.debug(f"training_config keys: {self.training_config.keys()}")
        self.logger.debug(f"scenarios type: {type(scenarios)}, length: {len(scenarios) if isinstance(scenarios, list) else 'N/A'}")

        # Use first scenario by default (for testing), or get from environment variable
        scenario_idx = int(os.getenv('CARLA_SCENARIO_INDEX', '0'))
        self.logger.debug(f"CARLA_SCENARIO_INDEX from environment: {scenario_idx}")

        if isinstance(scenarios, list) and len(scenarios) > 0:
            scenario_idx = min(scenario_idx, len(scenarios) - 1)
            scenario = scenarios[scenario_idx]
            npc_count = scenario.get("num_vehicles", 50)
            self.logger.info(f"Using scenario: {scenario.get('name', 'unknown')} (index {scenario_idx})")
        else:
            npc_count = 50
            self.logger.warning(f"No scenarios found in config, using default NPC count: {npc_count}")

        self.logger.info(f"Spawning {npc_count} NPC vehicles...")

        try:
            # Get or create Traffic Manager on specified port
            # Reference: EVALUATION_BUG_ANALYSIS.md - Option A (Separate TM Ports)
            # Training and evaluation environments MUST use different TM ports
            # to avoid registry conflicts when destroying/spawning NPCs
            if self.tm_port is None:
                # Default behavior: use default TM port (8000)
                self.traffic_manager = self.client.get_trafficmanager()
                self.logger.info("Using default Traffic Manager port (8000)")
            else:
                # Custom port specified (e.g., 8050 for evaluation environment)
                self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
                self.logger.info(f"Using custom Traffic Manager port ({self.tm_port})")

            # Configure Traffic Manager for synchronous mode
            self.traffic_manager.set_synchronous_mode(True)

            # CRITICAL FIX: Set deterministic seed for reproducibility
            # Reference: CARLA Traffic Manager Documentation
            # https://carla.readthedocs.io/en/latest/adv_traffic_manager/
            seed = self.training_config.get("seed", 42)
            self.traffic_manager.set_random_device_seed(seed)
            tm_port = self.traffic_manager.get_port()
            self.logger.info(f"Traffic Manager configured: synchronous=True, seed={seed}, port={tm_port}")

            # Spawn NPCs
            vehicle_bp = self.world.get_blueprint_library().filter("vehicle")
            spawn_points = np.random.choice(
                self.spawn_points, min(npc_count, len(self.spawn_points)), replace=False
            )

            spawn_attempts = 0
            spawn_successes = 0
            ego_location = self.vehicle.get_location()

            for spawn_point in spawn_points:
                # IMPROVED: Increased safety distance from 10.0m to 20.0m
                # Reference: NPC_TRAFFIC_SPAWNING_ANALYSIS.md Section 8.3
                if spawn_point.location.distance(ego_location) < 20.0:
                    continue

                spawn_attempts += 1
                try:
                    npc = self.world.spawn_actor(
                        np.random.choice(vehicle_bp), spawn_point
                    )

                    # CRITICAL FIX: Activate autopilot - NPCs must be registered with Traffic Manager
                    # Without this, NPCs spawn but never move (stationary obstacles)
                    # Reference: CARLA Traffic Manager Documentation
                    # https://carla.readthedocs.io/en/latest/adv_traffic_manager/
                    npc.set_autopilot(True, tm_port)

                    # Configure behavior
                    self.traffic_manager.update_vehicle_lights(npc, True)
                    self.traffic_manager.auto_lane_change(npc, True)
                    self.npcs.append(npc)
                    spawn_successes += 1

                except RuntimeError as e:
                    # IMPROVED: Better error handling with specific collision detection
                    if "collision" in str(e).lower():
                        self.logger.debug(f"Spawn collision at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
                    else:
                        self.logger.warning(f"NPC spawn failed: {e}")

            # IMPROVED: Validate spawn success rate
            success_rate = spawn_successes / spawn_attempts if spawn_attempts > 0 else 0
            self.logger.info(f"NPC spawning complete: {spawn_successes}/{spawn_attempts} successful ({success_rate*100:.1f}%)")

            if success_rate < 0.8 and spawn_attempts > 0:
                self.logger.warning(f"Low NPC spawn success rate: {success_rate*100:.1f}% (target: 80%)")

        except Exception as e:
            self.logger.warning(f"NPC traffic spawning failed: {e}")

    def _cleanup_episode(self):
        """
        Clean up vehicles and sensors from previous episode.

        Cleanup order follows CARLA best practices:
        1. Sensors (children) before vehicle (parent)
        2. NPCs (independent actors) last

        Reference: CARLA 0.9.16 Actor Destruction Best Practices
        https://carla.readthedocs.io/en/latest/core_actors/
        """
        cleanup_errors = []

        # STEP 1: Destroy sensors first (children before parent)
        # Sensors are attached to vehicle, destroy children first per CARLA docs
        if self.sensors:
            try:
                self.logger.debug("Destroying sensor suite...")
                success = self.sensors.destroy()
                # Note: SensorSuite.destroy() returns None, handles its own logging
                self.sensors = None
                self.logger.debug("Sensor suite destroyed successfully")

            except Exception as e:
                cleanup_errors.append(f"Sensor cleanup error: {e}")
                self.logger.error(f"Error during sensor cleanup: {e}", exc_info=True)
                self.sensors = None  # Clear reference anyway

        # STEP 2: Destroy ego vehicle (parent after children)
        if self.vehicle:
            try:
                self.logger.debug("Destroying ego vehicle...")

                # Check if actor is still alive before operations
                if not hasattr(self.vehicle, 'is_alive') or not self.vehicle.is_alive:
                    self.logger.warning("Ego vehicle already destroyed, skipping cleanup")
                    self.vehicle = None
                else:
                    # Vehicle is alive, proceed with destruction
                    success = self.vehicle.destroy()
                    if success:
                        self.logger.debug("Ego vehicle destroyed successfully")
                    else:
                        cleanup_errors.append("Ego vehicle destruction returned False")
                        self.logger.warning("Ego vehicle destruction failed")
                    self.vehicle = None

            except RuntimeError as e:
                cleanup_errors.append(f"Vehicle cleanup RuntimeError: {e}")
                self.logger.warning(f"Vehicle destruction RuntimeError: {e}")
                self.vehicle = None  # Clear reference anyway
            except AttributeError as e:
                cleanup_errors.append(f"Vehicle attribute error: {e}")
                self.logger.warning(f"Vehicle attribute error during cleanup: {e}")
                self.vehicle = None
            except Exception as e:
                cleanup_errors.append(f"Vehicle cleanup error: {e}")
                self.logger.error(f"Unexpected error during vehicle cleanup: {e}", exc_info=True)
                self.vehicle = None  # Clear reference anyway

        # STEP 3: Destroy NPCs (independent actors, non-critical)
        npc_failures = 0
        for i, npc in enumerate(self.npcs):
            try:
                # Check if NPC is still alive before destroying
                if not hasattr(npc, 'is_alive') or not npc.is_alive:
                    self.logger.debug(f"NPC {i} already destroyed")
                    continue

                success = npc.destroy()
                if not success:
                    npc_failures += 1
                    self.logger.debug(f"NPC {i} destruction returned False")

            except RuntimeError as e:
                npc_failures += 1
                self.logger.debug(f"NPC {i} RuntimeError during destruction: {e}")
            except AttributeError as e:
                npc_failures += 1
                self.logger.debug(f"NPC {i} AttributeError during destruction: {e}")
            except Exception as e:
                npc_failures += 1
                self.logger.debug(f"Failed to destroy NPC {i}: {e}")

        if npc_failures > 0:
            self.logger.debug(f"{npc_failures}/{len(self.npcs)} NPCs failed to destroy")

        self.npcs = []

        # Report accumulated critical errors
        if cleanup_errors:
            error_msg = f"Critical cleanup issues encountered: {cleanup_errors}"
            self.logger.warning(error_msg)
        else:
            self.logger.debug("Episode cleanup completed successfully")

    def close(self):
        """
        Shut down environment and disconnect from CARLA.

        Cleanup sequence:
        1. Destroy actors (sensors, vehicle, NPCs)
        2. Disable Traffic Manager synchronous mode
        3. Restore original world settings
        4. Clear client reference

        Idempotent: Safe to call multiple times.

        References:
        - CLOSE_ANALYSIS.md - Complete analysis with documentation backing
        - CARLA Traffic Manager: https://carla.readthedocs.io/en/latest/adv_traffic_manager/
        - Gymnasium Env.close(): https://gymnasium.farama.org/api/env/#gymnasium.Env.close
        """
        # Idempotency guard - Optional Improvement #1
        # Prevents duplicate cleanup and provides explicit state tracking
        if getattr(self, '_closed', False):
            self.logger.debug("Environment already closed, skipping cleanup")
            return

        self.logger.info("Closing CARLA environment...")

        # Close ROS Bridge interface if active (Phase 5)
        if self.ros_interface is not None:
            try:
                self.logger.info("[ROS BRIDGE] Closing ROS interface...")
                self.ros_interface.destroy()
                self.ros_interface = None
            except Exception as e:
                self.logger.warning(f"[ROS BRIDGE] Error closing ROS interface: {e}")

        # Phase 1: Destroy actors (sensors, vehicle, NPCs)
        self._cleanup_episode()

        # CRITICAL FIX: Allow CARLA to complete pending operations
        # After destroying actors, CARLA may have pending callbacks/ticks
        # Give it time to finish before disabling synchronous mode
        if self.world:
            try:
                # Perform one final tick to ensure all callbacks complete
                self.world.tick()
                time.sleep(0.02)  # 20ms grace period for callback completion
                self.logger.debug("Final world tick completed, callbacks flushed")
            except Exception as e:
                self.logger.warning(f"Final world tick failed: {e}")

        # Phase 2: Disable Traffic Manager synchronous mode
        # CRITICAL: Must be done AFTER world sync mode per CARLA docs
        # Reference: https://carla.readthedocs.io/en/latest/adv_traffic_manager/
        if self.traffic_manager:
            try:
                self.traffic_manager.set_synchronous_mode(False)
                self.logger.debug("Traffic Manager synchronous mode disabled")
            except Exception as e:
                self.logger.warning(f"Failed to disable TM sync mode: {e}")
            finally:
                self.traffic_manager = None

        # Phase 3: Restore original world settings - Optional Improvement #2
        # Restores all settings (not just sync mode) for persistent CARLA servers
        if self.world and hasattr(self, '_original_settings'):
            try:
                self.world.apply_settings(self._original_settings)
                self.logger.debug("World settings restored to original state")
            except Exception as e:
                self.logger.warning(f"Failed to restore world settings: {e}")

        # Phase 4: Clear client reference
        if self.client:
            self.client = None

        # Mark as closed
        self._closed = True
        self.logger.info("CARLA environment closed")

    @property
    def is_closed(self):
        """
        Check if environment is closed.

        Returns:
            bool: True if environment is closed, False otherwise.

        Reference: CLOSE_ANALYSIS.md - Optional Improvement #1
        """
        return getattr(self, '_closed', False)

    def render(self, mode: str = "human"):
        """Not implemented (CARLA runs headless for efficiency)."""
        pass

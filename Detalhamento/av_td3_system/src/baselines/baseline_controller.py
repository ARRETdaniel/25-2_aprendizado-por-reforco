"""
Combined Baseline Controller for Autonomous Vehicle Navigation.

This module integrates PID (longitudinal) and Pure Pursuit (lateral) controllers
to provide complete vehicle control. It interfaces with CARLA 0.9.16 Python API
and serves as a classical baseline for comparison with deep reinforcement learning agents.

Controller Architecture:
-----------------------
1. **Longitudinal Control (PID)**: Maintains target speed
   - Proportional term: Immediate response to speed error
   - Integral term: Eliminates steady-state error
   - Derivative term: Dampens overshoot and oscillations

2. **Lateral Control (Pure Pursuit)**: Follows waypoint path
   - Geometric path tracking algorithm
   - Speed-adaptive lookahead distance
   - Bicycle kinematic model steering

Design Philosophy:
-----------------
This baseline represents a well-tuned classical controller that:
- Uses proven control algorithms (PID + Pure Pursuit)
- Requires no training (deterministic)
- Provides interpretable behavior (white-box)
- Serves as performance lower bound for learning-based methods

The controller parameters are based on Course1FinalProject/controller2d.py
which demonstrated smooth, stable path following in CARLA simulations.

Performance Expectations:
------------------------
- Smooth path following with minimal oscillations
- Speed tracking within ±1 m/s of target
- Lateral deviation < 0.6m on straight sections
- Heading error < 6° during normal operation
- Stable up to 20 m/s (72 km/h)

Author: Daniel Terra Gomes
2025
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import carla

from src.baselines.pid_controller import PIDController
from src.baselines.pure_pursuit_controller import PurePursuitController


class BaselineController:
    """
    Combined PID + Pure Pursuit controller for autonomous vehicle navigation.

    This controller provides complete vehicle control by integrating:
    - **PID Controller**: Longitudinal control (throttle/brake for speed tracking)
    - **Pure Pursuit Controller**: Lateral control (steering for path following)

    The integration is straightforward:
    1. Both controllers operate independently (decoupled control)
    2. PID uses only speed feedback
    3. Pure Pursuit uses only position/heading feedback
    4. Outputs are combined into a single carla.VehicleControl command

    This decoupled approach works well because:
    - Longitudinal and lateral dynamics are weakly coupled at low speeds
    - Independent tuning simplifies controller design
    - Computational efficiency (no coupled optimization)

    Attributes:
        pid_controller (PIDController): Longitudinal speed controller
        pure_pursuit_controller (PurePursuitController): Lateral path tracking controller
        target_speed (float): Default target speed in m/s (converted from km/h)

    Example:
        >>> controller = BaselineController(target_speed=30.0)  # 30 km/h
        >>> waypoints = [(0,0,8.33), (10,0,8.33), (20,5,8.33)]
        >>> control = controller.compute_control(vehicle, waypoints, dt=0.05)
        >>> vehicle.apply_control(control)
    """

    def __init__(
        self,
        # PID parameters (longitudinal control)
        pid_kp: float = 0.50,
        pid_ki: float = 0.30,
        pid_kd: float = 0.13,
        integrator_min: float = 0.0,
        integrator_max: float = 10.0,
        # Pure Pursuit parameters (lateral control)
        kp_lookahead: float = 0.8,
        min_lookahead: float = 10.0,
        wheelbase: float = 3.0,
        # General parameters
        target_speed: float = 30.0  # km/h (will be converted to m/s)
    ):
        """
        Initialize the combined baseline controller with classical control parameters.

        Parameter Selection Rationale:
        ------------------------------

        **PID Parameters** (from Course4FinalProject/controller2d.py):
        - kp=0.50: Moderate proportional gain for responsive but stable speed control
        - ki=0.30: Integral gain to eliminate steady-state speed error
        - kd=0.13: Derivative gain to dampen speed oscillations
        - Tuned for CARLA vehicle dynamics (mass ~2000kg)

        **Pure Pursuit Parameters** (from Course1FinalProject/controller2d.py):
        - kp_lookahead=0.8: Speed-adaptive gain (L_d = 0.8 × v)
          * At 10 m/s: lookahead = 10m (1.0s preview)
          * At 15 m/s: lookahead = 12m (0.8s preview)
        - min_lookahead=10.0m: Minimum preview distance
          * Ensures adequate path preview even at low speeds
          * Prevents instability when stopped or moving slowly
        - wheelbase=3.0m: Typical passenger car wheelbase
          * CARLA default vehicles: ~2.5-3.5m
          * Used in bicycle kinematic model

        **Target Speed**:
        - Default 30 km/h (8.33 m/s): Safe urban driving speed
        - Can be overridden per-call in compute_control()
        - Waypoints can also specify per-segment speeds

        Args:
            pid_kp: PID proportional gain (default: 0.50, validated in Course4)
            pid_ki: PID integral gain (default: 0.30, validated in Course4)
            pid_kd: PID derivative gain (default: 0.13, validated in Course4)
            integrator_min: Minimum PID integrator value (anti-windup)
            integrator_max: Maximum PID integrator value (anti-windup)
            kp_lookahead: Pure Pursuit lookahead gain (default: 0.8, from Course1)
            min_lookahead: Minimum lookahead distance in meters (default: 10.0m)
            wheelbase: Vehicle wheelbase in meters (default: 3.0m)
            target_speed: Default target speed in km/h (default: 30.0 km/h = 8.33 m/s)

        Design Notes:
        ------------
        - PID parameters remain from Course4 (already well-tuned)
        - Pure Pursuit parameters now match Course1 (fixed zigzag issue)
        - Controller is stateless except for PID integrator
        - No learning or adaptation (purely classical control)
        """
        # Initialize longitudinal controller (PID)
        # Speed tracking via throttle/brake commands
        self.pid_controller = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            integrator_min=integrator_min,
            integrator_max=integrator_max
        )

        # Initialize lateral controller (Pure Pursuit)
        # Path following via steering commands
        # FIXED: Now uses true Pure Pursuit instead of Stanley
        self.pure_pursuit_controller = PurePursuitController(
            kp_lookahead=kp_lookahead,
            min_lookahead=min_lookahead,
            wheelbase=wheelbase,
            debug_log=True  # Enable debug logging for investigation
        )

        # Convert target speed from km/h to m/s (CARLA uses m/s internally)
        # Example: 30 km/h ÷ 3.6 = 8.33 m/s
        self.target_speed = target_speed / 3.6

        # Debug tracking
        self.step_count = 0
        self.debug_log = False  # Enable debug logging

    def reset(self) -> None:
        """
        Reset controller state between episodes.

        Should be called at the start of each new evaluation episode to ensure:
        - PID integrator is cleared (prevents carry-over from previous episode)
        - No residual state affects initial behavior
        - Consistent starting conditions for fair evaluation

        Pure Pursuit is stateless (purely geometric), so no reset needed.

        Example:
            >>> controller = BaselineController()
            >>> for episode in range(num_episodes):
            ...     controller.reset()  # Fresh start
            ...     # Run episode...
        """
        self.pid_controller.reset()
        self.step_count = 0  # Reset debug counter
        # Pure Pursuit is stateless, no reset needed

    def _get_target_speed_from_waypoints(
        self,
        current_x: float,
        current_y: float,
        waypoints: List[Tuple[float, float, float]],
        lookahead_distance: float = 20.0,
        max_decel: float = 1.5  # m/s² - comfortable braking
    ) -> float:
        """
        Extract target speed with LOOKAHEAD capability for safe braking.

        CRITICAL IMPROVEMENT (2025-11-23):
        Instead of using only the CLOSEST waypoint (reactive), this method looks
        AHEAD to find upcoming speed transitions and returns the MINIMUM speed
        within the lookahead distance. This ensures the vehicle starts braking
        EARLY ENOUGH before reaching slow zones (intersections, curves).

        Physics-Based Lookahead:
        - Lookahead distance: 20m (default)
        - This provides ~2.6 seconds at 30 km/h for driver comfort
        - At 30 km/h → 9 km/h transition, requires ~19m braking distance
        - Ensures smooth deceleration at comfortable 1.5 m/s²

        Why This Fix is Needed:
        Without lookahead, the vehicle only reacts when it REACHES the speed
        transition point. For example:
        - Vehicle at X=111m, speed=8.333 m/s (30 km/h)
        - Speed transition at X=98.59m to 2.5 m/s (9 km/h)
        - Closest waypoint is still high-speed until X≈100m
        - Only ~6m to brake from 30→9 km/h = TOO LATE!
        - Results in lane invasion at intersection

        With lookahead:
        - Vehicle at X=111m looks ahead 20m
        - Sees the 2.5 m/s waypoint at X=98.59m (13m ahead)
        - Immediately starts braking to 2.5 m/s
        - Reaches safe speed BEFORE the turn = NO LANE INVASION ✅

        Args:
            current_x: Vehicle X position in meters
            current_y: Vehicle Y position in meters
            waypoints: List of (x, y, speed_m_s) tuples
            lookahead_distance: How far ahead to check for speed changes (meters)
            max_decel: Maximum comfortable deceleration (m/s²)

        Returns:
            target_speed: MINIMUM speed found within lookahead distance (m/s)

        Example:
            >>> waypoints = [
            ...     (116.91, 129.49, 8.333),  # High speed
            ...     (110.77, 129.49, 8.333),  # High speed
            ...     (104.62, 129.49, 8.333),  # High speed
            ...     (98.59, 129.22, 2.5),     # SLOW ZONE! (intersection)
            ...     (95.98, 127.76, 2.5)      # Turn
            ... ]
            >>> # WITHOUT lookahead (old approach):
            >>> speed = closest_waypoint_speed(111, 129.49, waypoints)
            >>> print(speed)  # 8.333 m/s - closest is WP #2, too late to brake!
            >>>
            >>> # WITH lookahead (new approach):
            >>> speed = self._get_target_speed_from_waypoints(111, 129.49, waypoints)
            >>> print(speed)  # 2.5 m/s - sees slow zone ahead, starts braking NOW!

        References:
            - Physics: d = (v_f² - v_i²) / (2a) for braking distance
            - GitHub's VelocityPlanner uses similar lookahead (nominal_profile)
            - Our implementation: Simplified but effective

        Bug Fix History:
            2025-11-23 #1: Added speed extraction from waypoints (vs fixed speed)
            2025-11-23 #2: Added lookahead capability (THIS FIX)
                          - Prevents late braking at speed transitions
                          - Ensures safe intersection navigation
        """
        if len(waypoints) == 0:
            return self.target_speed  # Fallback to default

        waypoints_np = np.array(waypoints)

        # Find closest waypoint to vehicle (for reference)
        distances = np.sqrt(
            (waypoints_np[:, 0] - current_x)**2 +
            (waypoints_np[:, 1] - current_y)**2
        )
        closest_index = np.argmin(distances)

        # Start with speed from closest waypoint
        min_speed = waypoints_np[closest_index, 2]

        # Look ahead from closest waypoint to find upcoming slow zones
        for i in range(closest_index, len(waypoints_np)):
            # Calculate distance from vehicle to this waypoint
            wp_dist = np.sqrt(
                (waypoints_np[i, 0] - current_x)**2 +
                (waypoints_np[i, 1] - current_y)**2
            )

            # Stop searching beyond lookahead distance
            if wp_dist > lookahead_distance:
                break

            # Track the MINIMUM speed found within lookahead
            # This ensures we start braking for upcoming slow zones
            wp_speed = waypoints_np[i, 2]
            min_speed = min(min_speed, wp_speed)

        # Debug logging (if enabled)
        if self.debug_log and self.step_count % 10 == 0:
            closest_dist = distances[closest_index]
            print(f"[SPEED-LOOKAHEAD] Step {self.step_count}: "
                  f"Pos=({current_x:.2f}, {current_y:.2f}) | "
                  f"Closest WP: idx={closest_index}, dist={closest_dist:.2f}m, "
                  f"speed={waypoints_np[closest_index, 2]:.3f} m/s | "
                  f"Min speed in {lookahead_distance}m lookahead: {min_speed:.3f} m/s")

        return min_speed

    def _filter_waypoints_ahead(
        self,
        current_x: float,
        current_y: float,
        waypoints: List[Tuple[float, float, float]],
        lookahead_distance: float = 20.0
    ) -> List[Tuple[float, float, float]]:
        """
        Filter waypoints to only include those ahead of vehicle within lookahead.

        This is CRITICAL for Pure Pursuit! The GitHub implementation filters waypoints
        to only send a subset within lookahead distance. Without this, the controller
        receives waypoints behind the vehicle, causing it to try steering backward.

        Algorithm (from module_7.py):
        1. Find closest waypoint to vehicle
        2. Include 1 waypoint behind (for smooth transition)
        3. Include waypoints ahead until total distance > lookahead
        4. This subset is what Pure Pursuit sees

        Args:
            current_x: Vehicle X position in meters
            current_y: Vehicle Y position in meters
            waypoints: Full list of (x, y, speed) waypoints
            lookahead_distance: Distance ahead to include waypoints (meters)

        Returns:
            Filtered list of waypoints ahead of vehicle

        Bug Fix (2025-11-23):
            This method was added to match GitHub's working implementation.
            Previously, we passed ALL waypoints to Pure Pursuit, causing it
            to select waypoints behind the vehicle when it got far along the path.
        """
        if len(waypoints) == 0:
            return waypoints

        waypoints_np = np.array(waypoints)

        # Find closest waypoint index
        distances = np.sqrt(
            (waypoints_np[:, 0] - current_x)**2 +
            (waypoints_np[:, 1] - current_y)**2
        )
        closest_index = np.argmin(distances)

        # Start from 1 waypoint behind (or 0 if at start)
        start_index = max(0, closest_index - 1)

        # Find last index within lookahead distance
        end_index = closest_index
        total_distance = 0.0

        for i in range(closest_index, len(waypoints) - 1):
            # Distance from waypoint i to waypoint i+1
            wp_dist = np.sqrt(
                (waypoints_np[i+1, 0] - waypoints_np[i, 0])**2 +
                (waypoints_np[i+1, 1] - waypoints_np[i, 1])**2
            )
            total_distance += wp_dist
            end_index = i + 1

            if total_distance >= lookahead_distance:
                break

        # Return subset of waypoints
        return waypoints[start_index:end_index+1]


    def compute_control(
        self,
        vehicle: carla.Vehicle,
        waypoints: List[Tuple[float, float, float]],
        dt: float,
        target_speed: Optional[float] = None
    ) -> carla.VehicleControl:
        """
        Compute vehicle control commands from current state and waypoints.

        Control Loop Architecture:
        -------------------------
        1. **Extract State**: Read vehicle position, heading, and speed from CARLA
        2. **Longitudinal Control**: PID computes throttle/brake for speed tracking
        3. **Lateral Control**: Pure Pursuit computes steering for path following
        4. **Combine Commands**: Create CARLA VehicleControl with both outputs

        This method serves as the bridge between CARLA's API and our controllers,
        handling:
        - Coordinate system conversions (CARLA uses different conventions)
        - Unit conversions (degrees ↔ radians, Vector3D ↔ scalar speed)
        - Control command packaging (separate throttle/brake/steer)

        Implementation Details:
        ----------------------

        **Position Extraction**:
        - CARLA provides position in UE4 coordinate system (cm, left-handed)
        - We use meters for controller (standard SI units)
        - Transform.location gives vehicle center (not rear axle)
        - Pure Pursuit internally computes rear axle position

        **Heading Extraction**:
        - CARLA provides rotation.yaw in degrees [-180, 180]
        - Convert to radians for controller math
        - Pure Pursuit normalizes internally to [-π, π]

        **Speed Extraction**:
        - CARLA provides velocity as Vector3D in m/s
        - Compute scalar speed: ||v|| = √(vx² + vy² + vz²)
        - Ignore vertical component for ground vehicles (vz ≈ 0)

        **Control Output**:
        - PID outputs: throttle ∈ [0,1], brake ∈ [0,1]
        - Pure Pursuit outputs: steer ∈ [-1,1]
        - Package into carla.VehicleControl for application

        Args:
            vehicle: CARLA vehicle actor
                    Provides: get_transform() → location, rotation
                             get_velocity() → 3D velocity vector

            waypoints: List of (x, y, speed) tuples defining reference path
                      x, y: Position in meters (CARLA world coordinates)
                      speed: Target speed in m/s (optional, can use default)

            dt: Time step in seconds (should match CARLA's fixed_delta_seconds)
                Used by PID for derivative/integral calculations
                Typical value: 0.05s (20 Hz) for synchronous mode

            target_speed: Optional override for target speed in m/s
                         If None, uses self.target_speed from initialization
                         If provided, overrides default for this step only

        Returns:
            carla.VehicleControl command with:
            - throttle: [0, 1] - Accelerator pedal position
            - brake: [0, 1] - Brake pedal position
            - steer: [-1, 1] - Steering wheel angle (left negative, right positive)
            - hand_brake: False - Parking brake (not used)
            - reverse: False - Reverse gear (not used for forward driving)

        Example:
            >>> controller = BaselineController(target_speed=30.0)
            >>> waypoints = [(0, 0, 8.33), (10, 0, 8.33), (20, 5, 8.33)]
            >>>
            >>> # In CARLA simulation loop (synchronous mode):
            >>> world.tick()  # Advance simulation
            >>> control = controller.compute_control(
            ...     vehicle=ego_vehicle,
            ...     waypoints=waypoints,
            ...     dt=0.05  # 20 Hz update rate
            ... )
            >>> vehicle.apply_control(control)

        Performance Notes:
        -----------------
        - Computation time: < 1ms (pure Python, no ML inference)
        - Deterministic: Same state → same output (no randomness)
        - Real-time capable: Suitable for 100+ Hz control loops
        """
        # ====================================================================
        # STEP 1: Extract vehicle state from CARLA API
        # ====================================================================

        # Get vehicle transform (position and orientation)
        transform = vehicle.get_transform()

        # Get velocity vector
        velocity = vehicle.get_velocity()

        # Position in meters (vehicle center point)
        # CARLA's Transform.location is in centimeters, but Python API converts to meters
        current_x = transform.location.x
        current_y = transform.location.y

        # Heading angle in radians (convert from CARLA's degrees)
        # CARLA convention: 0° = North (Y-axis), 90° = East (X-axis)
        # Controller expects radians: 0 = East, π/2 = North
        current_yaw = np.radians(transform.rotation.yaw)

        # Speed in m/s (scalar magnitude of velocity vector)
        # Compute Euclidean norm: ||v|| = √(vx² + vy² + vz²)
        # For ground vehicles, vz ≈ 0, so mainly horizontal speed
        current_speed = np.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )

        # ====================================================================
        # STEP 2: Determine target speed for this control step
        # ====================================================================

        # Extract speed from waypoints (matching GitHub implementation)
        # This enables dynamic speed profiles:
        #   - 8.333 m/s (30 km/h) on straight sections
        #   - 2.5 m/s (9 km/h) at intersections/curves
        # Without this, the vehicle uses a fixed speed everywhere,
        # causing lane invasions at turns.
        if target_speed is None:
            target_speed = self._get_target_speed_from_waypoints(
                current_x=current_x,
                current_y=current_y,
                waypoints=waypoints  # Full waypoint list (not filtered yet)
            )

        # ====================================================================
        # STEP 3: Compute longitudinal control (PID for speed tracking)
        # ====================================================================

        # PID controller computes throttle and brake to track target speed
        # Output: throttle ∈ [0,1], brake ∈ [0,1]
        # Logic:
        #   - If speed < target: throttle > 0, brake = 0
        #   - If speed > target: throttle = 0, brake > 0
        #   - Integral term eliminates steady-state error
        #   - Derivative term dampens oscillations
        throttle, brake = self.pid_controller.update(
            current_speed=current_speed,
            target_speed=target_speed,  # ← Now varies with position!
            dt=dt
        )

        # ====================================================================
        # STEP 4: Filter waypoints to those ahead of vehicle
        # ====================================================================

        # CRITICAL FIX (2025-11-23): Filter waypoints to match GitHub implementation
        # The GitHub code only passes waypoints AHEAD of the vehicle within lookahead.
        # Without this, Pure Pursuit receives waypoints BEHIND the vehicle,
        # causing it to try steering backward (alpha = -180°) which manifests as drift.
        filtered_waypoints = self._filter_waypoints_ahead(
            current_x=current_x,
            current_y=current_y,
            waypoints=waypoints,
            lookahead_distance=20.0  # GitHub uses 20m lookahead for waypoint filtering
        )

        # ====================================================================
        # STEP 5: Compute lateral control (Pure Pursuit for tepath following)
        # ====================================================================

        # Pure Pursuit controller computes steering to follow waypoint path
        # Output: steer ∈ [-1,1]
        # Algorithm:
        #   1. Compute rear axle position (bicycle model reference)
        #   2. Calculate speed-adaptive lookahead: L_d = max(15, 1.0×v)
        #   3. Find "carrot" waypoint at lookahead distance
        #   4. Calculate steering: δ = atan2(2L×sin(α), L_d)
        steer = self.pure_pursuit_controller.update(
            current_x=current_x,
            current_y=current_y,
            current_yaw=current_yaw,
            current_speed=current_speed,
            waypoints=filtered_waypoints  # Use filtered waypoints!
        )

        # ====================================================================
        # STEP 6: Package control commands into CARLA format
        # ====================================================================

        # Debug logging (similar to Pure Pursuit debug output)
        self.step_count += 1
        if self.debug_log and (
            self.step_count % 10 == 0 or
            (1340 <= self.step_count <= 1380) or  # Around intersection
            (120 <= self.step_count <= 150)       # Early stage (from PP debug)
        ):
            speed_error = target_speed - current_speed
            print(
                f"[CTRL-DEBUG Step {self.step_count:4d}] "
                f"Pos=({current_x:.2f}, {current_y:.2f}) | "
                f"Speed={current_speed:.2f} m/s | "
                f"Target={target_speed:.2f} m/s ({target_speed*3.6:.1f} km/h) | "
                f"Error={speed_error:+.2f} m/s | "
                f"Throttle={throttle:.3f} Brake={brake:.3f} Steer={steer:+.4f}",
                flush=True
            )

        # Create CARLA VehicleControl message
        # This will be applied to the vehicle in the next simulation step
        control = carla.VehicleControl(
            throttle=float(throttle),      # Accelerator [0,1]
            steer=float(steer),            # Steering [-1,1] (left neg, right pos)
            brake=float(brake),            # Brake [0,1]
            hand_brake=False,              # Parking brake (not used)
            reverse=False,                 # Reverse gear (forward driving only)
            manual_gear_shift=False,       # Automatic transmission
            gear=0                         # Let CARLA handle gears
        )

        return control

    def get_debug_info(self, vehicle: carla.Vehicle) -> Dict[str, Any]:
        """
        Get debug information about current controller state.

        Useful for logging and debugging.

        Args:
            vehicle: CARLA vehicle actor

        Returns:
            Dictionary with debug information
        """
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        return {
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
            'speed_m_s': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            'pid_state': {
                'integral': self.pid_controller.v_error_integral,
                'prev_error': self.pid_controller.v_error_prev
            },
            'target_speed_m_s': self.target_speed,
            'target_speed_km_h': self.target_speed * 3.6
        }

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"BaselineController(\n"
            f"  PID: {self.pid_controller},\n"
            f"  PurePursuit: {self.pure_pursuit_controller},\n"
            f"  target_speed={self.target_speed:.2f} m/s ({self.target_speed * 3.6:.2f} km/h)\n"
            f")"
        )

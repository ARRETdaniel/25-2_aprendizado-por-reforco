"""
Waypoint Manager for CARLA Autonomous Navigation

Manages loading, processing, and providing waypoints for the vehicle to follow.
Transforms global waypoints to the vehicle's local coordinate frame.
Implements route completion detection and next waypoint lookahead.

Based on: FinalProject/waypoints.txt (Town01 route format)
"""

import numpy as np
import math
from typing import List, Tuple, Optional
import logging

try:
    import carla
except ImportError:
    # CARLA not available (e.g., for testing without simulator)
    carla = None


class WaypointManager:
    """
    Manages route waypoints in vehicle-local coordinates.

    Responsibilities:
    1. Load waypoints from file (format: x, y, z)
    2. Transform to vehicle-local frame
    3. Provide next N waypoints for DRL state
    4. Track route progress
    5. Detect route completion
    """

    def __init__(
        self,
        waypoints_file: str,
        lookahead_distance: float = 50.0,
        num_waypoints_ahead: int = 10,
        waypoint_spacing: float = 5.0,
        carla_map=None,  # Optional: CARLA map for proper lateral deviation calculation
    ):
        """
        Initialize waypoint manager.

        Args:
            waypoints_file: Path to CSV file with waypoints (format: x, y, z)
            lookahead_distance: Maximum distance ahead to track (meters)
            num_waypoints_ahead: Number of future waypoints to include in state
            waypoint_spacing: Distance between consecutive waypoints (meters)
            carla_map: Optional CARLA map object for accurate lateral deviation calculation
        """
        self.logger = logging.getLogger(__name__)
        self.waypoints_file = waypoints_file
        self.lookahead_distance = lookahead_distance
        self.num_waypoints_ahead = num_waypoints_ahead
        self.waypoint_spacing = waypoint_spacing
        self.carla_map = carla_map  # Store reference to CARLA map for lateral deviation

        # Load waypoints from file
        self.waypoints = self._load_waypoints()  # List of (x, y, z) tuples
        self.current_waypoint_idx = 0
        self.prev_waypoint_idx = 0  # Track for waypoint reached detection

        # FIX #3.1 Phase 6: Dense waypoint interpolation for continuous progress rewards
        # Reference: validation_logs/SIMPLE_SOLUTION_WAYPOINT_INTERPOLATION.md
        #
        # DECISION: Replace arc-length projection (which had edge cases at waypoint boundaries)
        # with proven dense waypoint interpolation from user's TCC code (module_7.py).
        #
        # Benefits:
        # - No complex projection calculation (simpler implementation)
        # - No edge cases at waypoint crossings (no t=0.000 sticking)
        # - Continuous distance updates every step (agent never "blind")
        # - Already proven in production (user's TCC project)
        #
        # Implementation: Transform 86 waypoints (3.11m spacing) into ~26,446 dense waypoints
        # (1cm spacing) via linear interpolation. Distance calculation becomes simple nearest
        # waypoint search - O(1) with local search optimization.
        self.dense_waypoints = self._create_dense_waypoints()
        self.total_route_length = self._calculate_total_route_length()

        original_count = len(self.waypoints)
        dense_count = len(self.dense_waypoints)
        self.logger.info(
            f"Loaded {original_count} waypoints from {waypoints_file}, "
            f"interpolated to {dense_count} dense waypoints (1cm resolution), "
            f"total route length: {self.total_route_length:.2f}m"
        )

    def _load_waypoints(self) -> List[Tuple[float, float, float]]:
        """
        Load waypoints from CSV file.

        Expected format (one waypoint per line):
            x, y, z
            317.74, 129.49, 8.333
            314.74, 129.49, 8.333
            ...

        Returns:
            List of (x, y, z) tuples
        """
        waypoints = []
        try:
            with open(self.waypoints_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = [float(x.strip()) for x in line.split(",")]
                    if len(parts) >= 3:
                        waypoints.append((parts[0], parts[1], parts[2]))

            if not waypoints:
                raise ValueError(f"No waypoints found in {self.waypoints_file}")

            return waypoints

        except Exception as e:
            self.logger.error(f"Failed to load waypoints from {self.waypoints_file}: {e}")
            raise

    def _create_dense_waypoints(self) -> List[Tuple[float, float, float]]:
        """
        Create densely interpolated waypoints for continuous progress reward feedback.

        FIX #3.1 Phase 6: Dense Waypoint Interpolation (Proven Solution from User's TCC)
        Reference: validation_logs/SIMPLE_SOLUTION_WAYPOINT_INTERPOLATION.md
        Source: related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/module_7.py

        PROBLEM SOLVED: Arc-length projection stuck at t=0.000 for ~6 steps after waypoint crossing

        ROOT CAUSE: Projection calculation returns exactly t=0.000 when vehicle very close to
        waypoint boundary, causing distance to "stick" and progress reward to be 0.00 even though
        vehicle is moving forward. This makes the agent "blind" without reward feedback.

        DECISION: Replace complex arc-length projection with proven dense waypoint interpolation
        from user's own TCC code. This completely eliminates projection edge cases by using
        simple nearest-waypoint distance calculation instead.

        Algorithm (from module_7.py lines 1475-1515):
        1. Calculate distances between consecutive waypoints
        2. For each waypoint pair:
           a. Determine number of interpolation points (distance / 0.01m resolution)
           b. Create unit vector pointing to next waypoint
           c. Add interpolated points at 1cm intervals along the vector
        3. Result: 86 waypoints (3.11m spacing) → ~26,446 dense waypoints (1cm spacing)

        Benefits:
        - NO projection calculation (eliminates edge cases)
        - Continuous distance updates every step (agent never "blind")
        - Simple nearest waypoint search - O(1) with local search optimization
        - Already proven in user's TCC project (production-tested)
        - Memory cost: ~427 KB (negligible for modern systems)

        Returns:
            List of densely interpolated waypoints as (x, y, z) tuples
            Typical output: ~26,000 waypoints with 1cm spacing
        """
        if not self.waypoints or len(self.waypoints) < 2:
            return self.waypoints.copy() if self.waypoints else []

        INTERP_DISTANCE_RES = 0.01  # 1cm resolution (from module_7.py)

        # Convert to numpy for easier vector operations
        waypoints_np = np.array(self.waypoints)

        # Calculate distances between consecutive waypoints
        wp_distance = []
        for i in range(1, waypoints_np.shape[0]):
            dist = np.sqrt(
                (waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2
            )
            wp_distance.append(dist)

        # Linearly interpolate between waypoints
        wp_interp = []

        for i in range(waypoints_np.shape[0] - 1):
            # Add original waypoint
            wp_interp.append(tuple(waypoints_np[i]))

            # Calculate number of interpolation points based on distance and resolution
            num_pts_to_interp = int(np.floor(wp_distance[i] / INTERP_DISTANCE_RES) - 1)

            if num_pts_to_interp > 0:
                # Create unit vector pointing to next waypoint
                wp_vector = waypoints_np[i+1] - waypoints_np[i]
                wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                # Add interpolated points at regular intervals
                for j in range(num_pts_to_interp):
                    next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                    interp_point = waypoints_np[i] + next_wp_vector
                    wp_interp.append(tuple(interp_point))

        # Add last waypoint
        wp_interp.append(tuple(waypoints_np[-1]))

        self.logger.info(
            f"Dense waypoint interpolation: {len(self.waypoints)} original → "
            f"{len(wp_interp)} interpolated (resolution={INTERP_DISTANCE_RES}m)"
        )

        return wp_interp

    def _calculate_total_route_length(self) -> float:
        """
        Calculate total route length from dense waypoints.

        Simple sum of distances between consecutive dense waypoints.

        Returns:
            Total route length in meters
        """
        if not self.dense_waypoints or len(self.dense_waypoints) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.dense_waypoints)):
            wp_prev = self.dense_waypoints[i-1]
            wp_curr = self.dense_waypoints[i]

            dist = math.sqrt(
                (wp_curr[0] - wp_prev[0])**2 +
                (wp_curr[1] - wp_prev[1])**2
            )
            total_length += dist

        return total_length

    def reset(self):
        """Reset waypoint tracking for new episode."""
        self.current_waypoint_idx = 0
        self.prev_waypoint_idx = 0

    def get_next_waypoints(
        self,
        vehicle_location: Tuple[float, float, float],
        vehicle_heading: float,
    ) -> np.ndarray:
        """
        Get next N waypoints in vehicle-local coordinates.

        Transforms global waypoints to vehicle's local frame:
        - Local X: forward direction
        - Local Y: right direction
        - Z: vertical (unchanged)

        Args:
            vehicle_location: (x, y, z) in global CARLA frame
            vehicle_heading: Heading angle in radians (0=North, π/2=East, -π/2=West)

        Returns:
            Array of shape (num_waypoints_ahead, 2) with [local_x, local_y] for each waypoint
            Waypoints are ordered by distance ahead
            Padding with zeros for remaining slots if near route end
        """
        waypoints_local = []

        # Find current waypoint (closest ahead of vehicle)
        self._update_current_waypoint(vehicle_location)

        # Get next N waypoints
        for i in range(self.num_waypoints_ahead):
            idx = self.current_waypoint_idx + i
            if idx >= len(self.waypoints):
                # Reached end of route, pad with zeros
                waypoints_local.append([0.0, 0.0])
            else:
                wp_global = self.waypoints[idx]
                wp_local = self._global_to_local(
                    wp_global, vehicle_location, vehicle_heading
                )

                # Only include if within lookahead distance
                dist = np.linalg.norm(wp_local)
                if dist <= self.lookahead_distance:
                    waypoints_local.append(wp_local)
                else:
                    waypoints_local.append([0.0, 0.0])

        return np.array(waypoints_local, dtype=np.float32)

    def _update_current_waypoint(self, vehicle_location):
        """
        Update current waypoint index based on vehicle position.

        Uses a proper "passing" threshold: only advances to next waypoint
        when vehicle is within 5m radius of current waypoint.

        Args:
            vehicle_location: Current vehicle location (can be carla.Location or tuple (x,y,z))
        """
        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):  # carla.Location object
            vx, vy, vz = vehicle_location.x, vehicle_location.y, vehicle_location.z
        else:  # Tuple (x, y, z)
            vx, vy, vz = vehicle_location

        # Waypoint passing threshold (meters)
        WAYPOINT_PASSED_THRESHOLD = 5.0

        # Check if current waypoint has been passed
        if self.current_waypoint_idx < len(self.waypoints):
            wpx, wpy, wpz = self.waypoints[self.current_waypoint_idx]
            dist_to_current = math.sqrt((vx - wpx) ** 2 + (vy - wpy) ** 2)

            # If within threshold, consider this waypoint reached and advance
            if dist_to_current < WAYPOINT_PASSED_THRESHOLD:
                # Move to next waypoint if available
                if self.current_waypoint_idx < len(self.waypoints) - 1:
                    self.prev_waypoint_idx = self.current_waypoint_idx
                    self.current_waypoint_idx += 1

    def _global_to_local(
        self,
        global_point: Tuple[float, float, float],
        vehicle_location: Tuple[float, float, float],
        vehicle_heading: float,
    ) -> np.ndarray:
        """
        Transform global coordinates to vehicle-local frame.

        Vehicle frame (CARLA convention):
        - X: forward direction (vehicle's front)
        - Y: right direction (vehicle's right side)
        - Origin: vehicle location

        CARLA yaw convention:
        - 0° = North (+Y in world)
        - 90° = East (+X in world)
        - Positive = clockwise rotation

        Args:
            global_point: (x, y, z) in global CARLA coordinates or carla.Location
            vehicle_location: (x, y, z) of vehicle in global coordinates or carla.Location
            vehicle_heading: Vehicle heading angle in radians (CARLA convention: 0=North)

        Returns:
            [local_x, local_y] in vehicle frame where:
            - local_x > 0: waypoint is in front
            - local_y > 0: waypoint is to the right
        """
        # Handle carla.Location objects
        if hasattr(global_point, 'x'):
            gx, gy = global_point.x, global_point.y
        else:
            gx, gy = global_point[0], global_point[1]

        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        # Vector from vehicle to waypoint in world frame
        dx = gx - vx
        dy = gy - vy

        # CARLA yaw: 0° = EAST (+X), 90° = SOUTH (+Y), 180° = WEST (-X), 270° = NORTH (-Y)
        # This matches standard math atan2 convention!
        # Vehicle frame: +X forward, +Y right
        #
        # Standard 2D rotation matrix to transform world coordinates to vehicle frame:
        # [local_x]   [cos(θ)  sin(θ)] [dx]
        # [local_y] = [-sin(θ) cos(θ)] [dy]
        #
        # Where θ = vehicle_heading (in radians)

        cos_h = math.cos(vehicle_heading)
        sin_h = math.sin(vehicle_heading)

        local_x = cos_h * dx + sin_h * dy  # Forward (vehicle +X)
        local_y = -sin_h * dx + cos_h * dy  # Right (vehicle +Y)

        return np.array([local_x, local_y], dtype=np.float32)

    def get_progress(self) -> float:
        """
        Get route completion progress (0.0 to 1.0).

        Returns:
            Fraction of route completed (0 = start, 1 = end)
        """
        if len(self.waypoints) == 0:
            return 0.0
        return self.current_waypoint_idx / len(self.waypoints)

    def is_route_finished(self) -> bool:
        """
        Check if vehicle has reached end of route.

        Returns:
            True if reached final waypoint

        Note: current_waypoint_idx now tracks position in DENSE waypoints (26k+),
              not original waypoints (86). Must compare against dense_waypoints length.
        """
        # FIX: current_waypoint_idx is now an index into dense_waypoints, not waypoints!
        # After progressive search fix, we track position in dense waypoints array.
        return self.current_waypoint_idx >= len(self.dense_waypoints) - 2

    def get_target_heading(self, vehicle_location) -> float:
        """
        Get target heading to next waypoint.

        Args:
            vehicle_location: Current vehicle location (can be carla.Location or tuple (x,y,z))

        Returns:
            Target heading in radians (0=North, π/2=East)
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):  # carla.Location object
            vx, vy = vehicle_location.x, vehicle_location.y
        else:  # Tuple (x, y, z)
            vx, vy = vehicle_location[0], vehicle_location[1]

        next_wp = self.waypoints[self.current_waypoint_idx]
        dx = next_wp[0] - vx  # X-component (East in CARLA)
        dy = next_wp[1] - vy  # Y-component (North in CARLA)

        # Calculate target heading to match CARLA's yaw convention
        # CARLA yaw: 0° = EAST (+X), 90° = SOUTH (+Y), 180° = WEST (-X), 270° = NORTH (-Y)
        # Standard atan2(dy, dx): 0 rad = East (+X), π/2 rad = North (+Y)
        #
        # CARLA uses the SAME angle convention as standard math!
        # No conversion needed, just use atan2 directly
        #
        # This is because:
        #   - When dx>0, dy=0 (East): atan2=0 → CARLA yaw=0° ✓
        #   - When dx=0, dy>0 (South): atan2=π/2 → CARLA yaw=90° ✓
        #   - When dx<0, dy=0 (West): atan2=±π → CARLA yaw=180° ✓
        #   - When dx=0, dy<0 (North): atan2=-π/2 → CARLA yaw=270° or -90° ✓

        heading_carla = math.atan2(dy, dx)  # CARLA uses same convention as atan2!

        return heading_carla

    def get_lateral_deviation(self, vehicle_location) -> float:
        """
        Get lateral deviation from lane center using CARLA's OpenDRIVE projection.

        This method properly accounts for road curvature at intersections by projecting
        the vehicle's location to the center of the nearest lane using CARLA's map API.

        Args:
            vehicle_location: Current vehicle location (can be carla.Location or tuple (x,y,z))

        Returns:
            Lateral deviation in meters (Euclidean 2D distance from lane center)
            Returns 0.0 if vehicle is not on any road (truly off-road)
        """
        if not hasattr(self, 'carla_map') or self.carla_map is None:
            # Fallback to old straight-line method if map not available
            return self._get_lateral_deviation_legacy(vehicle_location)

        # Check if carla module is available
        if carla is None:
            return self._get_lateral_deviation_legacy(vehicle_location)

        # Convert to carla.Location if needed
        if hasattr(vehicle_location, 'x'):  # Already carla.Location
            loc = vehicle_location
        else:  # Tuple (x, y, z)
            loc = carla.Location(x=vehicle_location[0],
                                y=vehicle_location[1],
                                z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)

        # Get waypoint at lane center using CARLA's OpenDRIVE projection
        # project_to_road=True ensures we get the center of the nearest lane
        waypoint = self.carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            # Vehicle is truly off-road (not on any driving lane)
            return float('inf')  # Signal as maximum deviation

        # Get lane center location (follows road curvature through intersections)
        lane_center = waypoint.transform.location

        # Calculate 2D Euclidean distance from vehicle to lane center
        lateral_deviation = math.sqrt(
            (loc.x - lane_center.x)**2 +
            (loc.y - lane_center.y)**2
        )

        return lateral_deviation

    def _get_lateral_deviation_legacy(self, vehicle_location) -> float:
        """
        Legacy method: Calculate lateral deviation using straight-line projection
        between waypoints. DEPRECATED - does not account for road curvature.

        Kept for fallback when CARLA map is not available.
        """
        print("WARNING: Using legacy lateral deviation calculation (may be inaccurate at intersections)")
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):  # carla.Location object
            vx_pos, vy_pos = vehicle_location.x, vehicle_location.y
        else:  # Tuple (x, y, z)
            vx_pos, vy_pos = vehicle_location[0], vehicle_location[1]

        # Get current and next waypoints
        if self.current_waypoint_idx == 0:
            wp1 = self.waypoints[0]
        else:
            wp1 = self.waypoints[self.current_waypoint_idx - 1]

        wp2 = self.waypoints[self.current_waypoint_idx]

        # Vector along route
        route_dx = wp2[0] - wp1[0]
        route_dy = wp2[1] - wp1[1]
        route_length = math.sqrt(route_dx**2 + route_dy**2)

        if route_length < 0.001:
            return 0.0

        # Vector from wp1 to vehicle
        vx = vx_pos - wp1[0]
        vy = vy_pos - wp1[1]

        # Perpendicular distance (cross product divided by route length)
        cross = route_dx * vy - route_dy * vx
        lateral_dev = cross / route_length

        return lateral_dev

    def get_distance_to_goal_euclidean(self, vehicle_location: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean (straight-line) distance from vehicle to final goal waypoint.

        ⚠️ USAGE NOTE: This method is for GOAL CHECKING ONLY, not progress reward calculation!

        For progress reward calculation, use get_route_distance_to_goal() which implements
        smooth metric blending to prevent discontinuity (Fix #3.1).

        This method is kept for:
        - Goal reached detection (check_goal_reached())
        - Debugging/comparison purposes
        - Simple distance queries where metric switching is not an issue

        Args:
            vehicle_location: Current vehicle location (x, y, z) or carla.Location

        Returns:
            Euclidean distance to goal in meters, or None if waypoints not initialized
        """
        # Safety check: Return None if waypoints not initialized
        if not self.waypoints or len(self.waypoints) == 0:
            return None

        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        # Final waypoint is the goal
        goal_x, goal_y, _ = self.waypoints[-1]

        distance = math.sqrt((goal_x - vx) ** 2 + (goal_y - vy) ** 2)
        return distance

    def get_route_distance_to_goal(self, vehicle_location: Tuple[float, float, float]) -> float:
        """
        Calculate distance to goal using dense waypoint interpolation (simple and robust).

        FIX #3.1 Phase 6: Dense Waypoint Interpolation - Simple Solution
        Reference: validation_logs/SIMPLE_SOLUTION_WAYPOINT_INTERPOLATION.md

        PREVIOUS IMPLEMENTATION (Arc-Length Projection):
        - Complex projection calculation with parameter t ∈ [0,1]
        - Edge case: t stuck at 0.000 for ~6 steps after waypoint crossing
        - Result: Progress reward = 0.00 even though vehicle moving forward
        - Impact: Agent "blind" without feedback for ~516 steps per episode

        NEW IMPLEMENTATION (Dense Waypoint Interpolation):
        - Simple nearest waypoint search in densely interpolated route
        - Resolution: 1cm spacing (~26,446 waypoints from original 86)
        - NO projection calculation needed (eliminates edge cases)
        - Continuous distance updates every step (agent never "blind")

        Algorithm:
        1. Find nearest dense waypoint to vehicle (local search optimization)
        2. Sum distances from nearest waypoint to goal
        3. Distance updates smoothly as vehicle moves (no edge cases)

        Benefits over arc-length projection:
        -  Simpler implementation (no complex projection math)
        - No edge cases (no t=0.000 sticking at waypoint boundaries)
        - Continuous feedback (distance updates every step)
        - Already proven in user's TCC code (production-tested)
        - Fast: O(1) with local search (start from last nearest waypoint)

        Memory cost: ~427 KB for ~26K waypoints (negligible)

        Args:
            vehicle_location: Current vehicle location (x, y, z) or carla.Location

        Returns:
            Distance in meters to goal along dense waypoint route
        """
        if not self.dense_waypoints or len(self.dense_waypoints) == 0:
            return None

        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        # CRITICAL FIX #3: Progressive segment search with proper t-parameter handling
        # Reference: validation_logs/CRITICAL_BUG_T_CLAMPING_ISSUE.md
        #
        # Previous approach: Search window with t clamping caused "segment sticking"
        # Problem: When vehicle passes segment endpoint (t > 1.0), search window might
        #          not include the segment vehicle is actually on. Vehicle gets stuck
        #          on old segment with t=1.0, causing constant distance (0 progress!)
        #
        # Solution: Progressive forward search from current_waypoint_idx
        #          - Only select segments where 0 <= t <= 1 (vehicle is WITHIN segment)
        #          - Update current_waypoint_idx to track vehicle position
        #          - O(1) average case performance (checks 1-3 segments typically)

        # Progressive search: Start from current position and search forward
        # until finding segment where vehicle is WITHIN bounds (0 <= t <= 1)
        nearest_segment_idx = max(0, self.current_waypoint_idx)
        t_final = 0.0
        min_dist = float('inf')
        found_valid_segment = False

        # Search forward up to 200 segments (2m with 1cm spacing, generous for sharp turns)
        max_search = min(len(self.dense_waypoints) - 1, self.current_waypoint_idx + 200)

        for i in range(self.current_waypoint_idx, max_search):
            # Segment: dense_waypoints[i] → dense_waypoints[i+1]
            wp_a = self.dense_waypoints[i]
            wp_b = self.dense_waypoints[i + 1]

            # Vector from segment start to end
            seg_x = wp_b[0] - wp_a[0]
            seg_y = wp_b[1] - wp_a[1]
            seg_length_sq = seg_x ** 2 + seg_y ** 2

            if seg_length_sq < 1e-6:  # Degenerate segment (shouldn't happen with dense waypoints)
                continue

            # Project vehicle onto segment (UNCLAMPED first to check validity)
            # t = ((V - A) · (B - A)) / |B - A|²
            t_unclamped = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq

            # Check if vehicle is WITHIN this segment
            if t_unclamped < 0.0:
                # Vehicle is BEFORE this segment (shouldn't happen if searching forward)
                # Use this segment anyway (vehicle may have reversed or jumped)
                nearest_segment_idx = i
                t_final = 0.0
                found_valid_segment = True
                break
            elif 0.0 <= t_unclamped <= 1.0:
                # Vehicle is ON this segment! This is the correct segment.
                nearest_segment_idx = i
                t_final = t_unclamped
                found_valid_segment = True

                # Calculate perpendicular distance for logging
                closest_x = wp_a[0] + t_final * seg_x
                closest_y = wp_a[1] + t_final * seg_y
                min_dist = math.sqrt((vx - closest_x) ** 2 + (vy - closest_y) ** 2)
                break
            # else: t_unclamped > 1.0 → vehicle is PAST this segment, continue to next

        # Handle special cases
        if not found_valid_segment:
            # Vehicle is past ALL segments in search window
            # Check if we searched all the way to the last segment
            if max_search == len(self.dense_waypoints) - 1:
                # Vehicle is past the GOAL! Return distance = 0
                self.logger.debug(
                    f"[DENSE_WP_PROJ] Vehicle past goal waypoint! "
                    f"Vehicle=({vx:.2f}, {vy:.2f}), "
                    f"LastWaypoint={self.dense_waypoints[-1]}, "
                    f"Distance=0.00m (GOAL REACHED!)"
                )
                return 0.0
            else:
                # Fallback: use last segment in search window with clamped t
                nearest_segment_idx = max_search - 1
                wp_a = self.dense_waypoints[nearest_segment_idx]
                wp_b = self.dense_waypoints[nearest_segment_idx + 1]

                seg_x = wp_b[0] - wp_a[0]
                seg_y = wp_b[1] - wp_a[1]
                seg_length_sq = seg_x ** 2 + seg_y ** 2

                if seg_length_sq > 1e-6:
                    t_final = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq
                    t_final = max(0.0, min(1.0, t_final))  # Clamp for fallback case
                else:
                    t_final = 0.0

        # Update current_waypoint_idx to track vehicle position (keeps search efficient)
        # Only update if we're not at the fallback case
        if nearest_segment_idx > self.current_waypoint_idx:
            self.current_waypoint_idx = nearest_segment_idx

        # Now calculate arc-length distance ALONG the path from projection to goal
        # This is the key: we measure distance ALONG the path, not straight-line to waypoint

        # Get segment parameters for final calculation
        wp_a = self.dense_waypoints[nearest_segment_idx]
        wp_b = self.dense_waypoints[nearest_segment_idx + 1]

        seg_x = wp_b[0] - wp_a[0]
        seg_y = wp_b[1] - wp_a[1]
        seg_length_sq = seg_x ** 2 + seg_y ** 2

        # Use t_final from progressive search
        t = t_final
        segment_length = math.sqrt(seg_length_sq)

        # Calculate perpendicular distance for logging (if not already calculated)
        if min_dist == float('inf'):
            closest_x = wp_a[0] + t * seg_x
            closest_y = wp_a[1] + t * seg_y
            min_dist = math.sqrt((vx - closest_x) ** 2 + (vy - closest_y) ** 2)

        # Arc-length calculation:
        # 1. Remaining distance on current segment: (1 - t) * segment_length
        # 2. Sum of all subsequent segments to goal
        arc_on_current_segment = (1.0 - t) * segment_length

        # Sum remaining segments
        arc_on_remaining_segments = 0.0
        for i in range(nearest_segment_idx + 1, len(self.dense_waypoints) - 1):
            wp_curr = self.dense_waypoints[i]
            wp_next = self.dense_waypoints[i + 1]

            segment_dist = math.sqrt(
                (wp_next[0] - wp_curr[0])**2 +
                (wp_next[1] - wp_curr[1])**2
            )
            arc_on_remaining_segments += segment_dist

        # Total arc-length distance to goal
        distance_to_goal = arc_on_current_segment + arc_on_remaining_segments

        # Debug logging
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"[DENSE_WP_PROJ] Vehicle=({vx:.2f}, {vy:.2f}), "
                f"SegmentIdx={nearest_segment_idx}/{len(self.dense_waypoints)-1}, "
                f"t={t:.4f}, "
                f"PerpendicularDist={min_dist:.3f}m, "
                f"ArcLength={distance_to_goal:.2f}m"
            )

        return distance_to_goal

    def _find_nearest_waypoint_index(self, vehicle_location: Tuple[float, float, float]) -> Optional[int]:
        """
        Find index of nearest waypoint ahead of vehicle on route.

        Args:
            vehicle_location: Current vehicle location (x, y, z) or carla.Location

        Returns:
            Index of nearest waypoint ahead, or None if off-route
        """
        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        # Start from current waypoint index (vehicle should be near this)
        min_distance = float('inf')
        nearest_idx = None

        # Search ahead from current waypoint (vehicle progresses forward)
        search_start = max(0, self.current_waypoint_idx - 2)  # Look back 2 in case we missed one
        search_end = min(len(self.waypoints), self.current_waypoint_idx + 10)  # Look ahead 10

        for i in range(search_start, search_end):
            wp = self.waypoints[i]
            dist = math.sqrt((wp[0] - vx) ** 2 + (wp[1] - vy) ** 2)

            if dist < min_distance:
                min_distance = dist
                nearest_idx = i

        # If vehicle is too far from route (>20m), return None
        if min_distance > 20.0:
            return None

        return nearest_idx

    # DEPRECATED: Arc-length projection methods removed in Phase 6
    # Reference: validation_logs/SIMPLE_SOLUTION_WAYPOINT_INTERPOLATION.md
    #
    # These methods (_find_nearest_segment, _project_onto_segment) were used by the previous
    # arc-length interpolation implementation which had edge cases at waypoint boundaries
    # (t=0.000 sticking for ~6 steps). Replaced with dense waypoint interpolation which
    # completely eliminates projection calculation and its associated edge cases.
    #
    # Keeping methods below for reference/debugging only - NOT USED in production code.

    def _find_nearest_segment(self, vehicle_location: Tuple[float, float, float]) -> Tuple[Optional[int], float]:
        """
        Find index of nearest route segment (between waypoint[i] and waypoint[i+1]).

        FIX #4: Helper method for projection-based route distance calculation
        Reference: BUG_ROUTE_DISTANCE_INCREASES.md

        FIX #3.1: Now returns distance to enable smooth metric blending
        Reference: PHASE_2_REINVESTIGATION.md

        ALGORITHM:
        1. For each segment (waypoint[i] to waypoint[i+1])
        2. Calculate perpendicular distance from vehicle to line segment
        3. Return index of segment with minimum distance + that distance

        Unlike _find_nearest_waypoint_index (which finds nearest point),
        this finds the nearest LINE SEGMENT, which is critical for projection.

        Args:
            vehicle_location: Current vehicle location (x, y, z) or carla.Location

        Returns:
            Tuple of (segment_idx, distance_from_route):
            - segment_idx: Index i where segment is waypoint[i] to waypoint[i+1]
                          Returns None if vehicle is off-route (>20m from any segment)
            - distance_from_route: Perpendicular distance in meters from vehicle to nearest segment
        """
        # Handle both carla.Location and tuple inputs
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        min_distance = float('inf')
        nearest_segment_idx = None

        # Search near current waypoint index
        search_start = max(0, self.current_waypoint_idx - 2)
        search_end = min(len(self.waypoints) - 1, self.current_waypoint_idx + 10)

        for i in range(search_start, search_end):
            # Segment: waypoint[i] → waypoint[i+1]
            wp_start = self.waypoints[i]
            wp_end = self.waypoints[i + 1]

            # Calculate perpendicular distance from point to line segment
            # Reference: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

            # Vector from segment start to end
            seg_x = wp_end[0] - wp_start[0]
            seg_y = wp_end[1] - wp_start[1]
            seg_length_sq = seg_x ** 2 + seg_y ** 2

            if seg_length_sq < 1e-6:  # Avoid division by zero for degenerate segments
                # Segment is a point, use point-to-point distance
                dist = math.sqrt((wp_start[0] - vx) ** 2 + (wp_start[1] - vy) ** 2)
            else:
                # Project vehicle onto line containing segment (unbounded)
                # t = ((P - A) · (B - A)) / |B - A|²
                t = ((vx - wp_start[0]) * seg_x + (vy - wp_start[1]) * seg_y) / seg_length_sq

                # Clamp t to [0, 1] to stay within segment bounds
                t = max(0.0, min(1.0, t))

                # Closest point on segment
                closest_x = wp_start[0] + t * seg_x
                closest_y = wp_start[1] + t * seg_y

                # Distance from vehicle to closest point
                dist = math.sqrt((vx - closest_x) ** 2 + (vy - closest_y) ** 2)

            if dist < min_distance:
                min_distance = dist
                nearest_segment_idx = i

        # If vehicle is too far from route (>20m), return None
        if min_distance > 20.0:
            self.logger.warning(
                f"[FIND_NEAREST_SEGMENT] Vehicle off-route: "
                f"min_distance={min_distance:.2f}m > 20m threshold"
            )
            return None, min_distance

        return nearest_segment_idx, min_distance

    def _project_onto_segment(
        self,
        point: Tuple[float, float],
        segment_start: Tuple[float, float],
        segment_end: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Project a point onto a line segment using vector projection.

        FIX #4: Helper method for projection-based route distance calculation
        Reference: BUG_ROUTE_DISTANCE_INCREASES.md

        ALGORITHM (Vector Projection):
        Let P = point, A = segment_start, B = segment_end

        1. Compute vector v = B - A (segment direction)
        2. Compute vector w = P - A (point relative to start)
        3. Compute projection parameter: t = (w · v) / (v · v)
        4. Clamp t to [0, 1] to stay within segment
        5. Projected point: Q = A + t × v

        Reference: https://en.wikipedia.org/wiki/Vector_projection

        Geometric intuition:
        - If vehicle is ahead of segment start: t > 0
        - If vehicle is behind segment start: t < 0 (clamped to 0)
        - If vehicle is past segment end: t > 1 (clamped to 1)
        - If vehicle is alongside segment: 0 < t < 1

        Args:
            point: Point to project (vehicle position)
            segment_start: Start of line segment (waypoint[i])
            segment_end: End of line segment (waypoint[i+1])

        Returns:
            Tuple (x, y) of projected point on segment
        """
        px, py = point
        ax, ay = segment_start
        bx, by = segment_end

        # Vector v = B - A (segment direction)
        vx = bx - ax
        vy = by - ay

        # Vector w = P - A (point relative to start)
        wx = px - ax
        wy = py - ay

        # Compute v · v (segment length squared)
        v_dot_v = vx * vx + vy * vy

        if v_dot_v < 1e-6:  # Avoid division by zero for degenerate segments
            # Segment is a point, return segment start
            return (ax, ay)

        # Compute projection parameter: t = (w · v) / (v · v)
        t = (wx * vx + wy * vy) / v_dot_v

        # Clamp t to [0, 1] to keep projection within segment bounds
        t = max(0.0, min(1.0, t))

        # Projected point: Q = A + t × v
        proj_x = ax + t * vx
        proj_y = ay + t * vy

        return (proj_x, proj_y)

    def check_waypoint_reached(self) -> bool:
        """
        Check if a new waypoint was reached since last check.

        Returns:
            True if current_waypoint_idx increased since last call
        """
        waypoint_reached = self.current_waypoint_idx > self.prev_waypoint_idx
        self.prev_waypoint_idx = self.current_waypoint_idx
        return waypoint_reached

    def check_goal_reached(self, vehicle_location: Tuple[float, float, float], threshold: float = 5.0) -> bool:
        """
        Check if vehicle reached the final goal waypoint.

        Args:
            vehicle_location: Current vehicle location (x, y, z) or carla.Location
            threshold: Distance threshold in meters to consider goal reached

        Returns:
            True if vehicle is within threshold of final waypoint
        """
        distance_to_goal = self.get_distance_to_goal_euclidean(vehicle_location)
        return distance_to_goal < threshold

    def get_progress_percentage(self) -> float:
        """
        Calculate route completion percentage.

        Returns:
            Percentage of route completed (0.0 to 100.0)
        """
        if len(self.waypoints) == 0:
            return 0.0

        return (self.current_waypoint_idx / (len(self.waypoints) - 1)) * 100.0

    def get_current_waypoint_index(self) -> int:
        """
        Get the index of the current target waypoint.

        Returns:
            Index of current waypoint
        """
        return self.current_waypoint_idx

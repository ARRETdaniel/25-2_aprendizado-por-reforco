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
    ):
        """
        Initialize waypoint manager.

        Args:
            waypoints_file: Path to CSV file with waypoints (format: x, y, z)
            lookahead_distance: Maximum distance ahead to track (meters)
            num_waypoints_ahead: Number of future waypoints to include in state
            waypoint_spacing: Distance between consecutive waypoints (meters)
        """
        self.logger = logging.getLogger(__name__)
        self.waypoints_file = waypoints_file
        self.lookahead_distance = lookahead_distance
        self.num_waypoints_ahead = num_waypoints_ahead
        self.waypoint_spacing = waypoint_spacing

        # Load waypoints from file
        self.waypoints = self._load_waypoints()  # List of (x, y, z) tuples
        self.current_waypoint_idx = 0
        self.prev_waypoint_idx = 0  # Track for waypoint reached detection

        self.logger.info(
            f"Loaded {len(self.waypoints)} waypoints from {waypoints_file}"
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
        """
        return self.current_waypoint_idx >= len(self.waypoints) - 1

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
        Get lateral deviation from route (perpendicular distance to waypoint).

        Args:
            vehicle_location: Current vehicle location (can be carla.Location or tuple (x,y,z))

        Returns:
            Lateral deviation in meters (positive = right of route)
        """
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

    def get_distance_to_goal(self, vehicle_location: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance from vehicle to final goal waypoint.

        Args:
            vehicle_location: Current vehicle location (x, y, z) or carla.Location

        Returns:
            Distance to goal in meters, or None if waypoints not initialized
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
        distance_to_goal = self.get_distance_to_goal(vehicle_location)
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

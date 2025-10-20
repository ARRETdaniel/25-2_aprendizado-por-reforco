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

    def _update_current_waypoint(self, vehicle_location: Tuple[float, float, float]):
        """
        Update current waypoint index based on vehicle position.

        Finds the next waypoint ahead of the vehicle.
        """
        vx, vy, vz = vehicle_location

        # Find closest waypoint ahead
        min_dist = float("inf")
        closest_idx = self.current_waypoint_idx

        for idx in range(self.current_waypoint_idx, len(self.waypoints)):
            wpx, wpy, wpz = self.waypoints[idx]

            # Calculate distance
            dist = math.sqrt((vx - wpx) ** 2 + (vy - wpy) ** 2)

            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

            # Stop if we've moved significantly past this waypoint
            if idx > self.current_waypoint_idx and dist > min_dist + 10.0:
                break

        self.current_waypoint_idx = closest_idx

    def _global_to_local(
        self,
        global_point: Tuple[float, float, float],
        vehicle_location: Tuple[float, float, float],
        vehicle_heading: float,
    ) -> np.ndarray:
        """
        Transform global coordinates to vehicle-local frame.

        Vehicle frame:
        - X: forward direction
        - Y: right direction (perpendicular to X)
        - Origin: vehicle location

        Args:
            global_point: (x, y, z) in global CARLA coordinates
            vehicle_location: (x, y, z) of vehicle in global coordinates
            vehicle_heading: Vehicle heading angle in radians

        Returns:
            [local_x, local_y] in vehicle frame
        """
        # Vector from vehicle to waypoint
        dx = global_point[0] - vehicle_location[0]
        dy = global_point[1] - vehicle_location[1]

        # Rotate to vehicle frame
        # Heading = 0 is North, positive is counter-clockwise
        # In CARLA, this corresponds to -heading for rotation matrix
        cos_h = math.cos(-vehicle_heading)
        sin_h = math.sin(-vehicle_heading)

        local_x = cos_h * dx - sin_h * dy
        local_y = sin_h * dx + cos_h * dy

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

    def get_target_heading(
        self, vehicle_location: Tuple[float, float, float]
    ) -> float:
        """
        Get target heading towards next waypoint.

        Args:
            vehicle_location: Current vehicle (x, y, z)

        Returns:
            Target heading in radians (0=North, π/2=East)
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        next_wp = self.waypoints[self.current_waypoint_idx]
        dx = next_wp[0] - vehicle_location[0]
        dy = next_wp[1] - vehicle_location[1]

        # Calculate heading (0=North, π/2=East in CARLA)
        heading = math.atan2(dx, dy)  # Note: CARLA uses y as "north"
        return heading

    def get_lateral_deviation(
        self, vehicle_location: Tuple[float, float, float]
    ) -> float:
        """
        Get lateral deviation from route (perpendicular distance to waypoint).

        Args:
            vehicle_location: Current vehicle (x, y, z)

        Returns:
            Lateral deviation in meters (positive = right of route)
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

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
        vx = vehicle_location[0] - wp1[0]
        vy = vehicle_location[1] - wp1[1]

        # Perpendicular distance (cross product divided by route length)
        cross = route_dx * vy - route_dy * vx
        lateral_dev = cross / route_length

        return lateral_dev

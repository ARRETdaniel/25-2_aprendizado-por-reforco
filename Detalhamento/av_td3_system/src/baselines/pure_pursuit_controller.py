"""
Pure Pursuit Controller for Lateral Vehicle Control.

This module implements a Pure Pursuit controller with heading error compensation
for path following. The implementation is adapted from the TCC controller2d.py
for compatibility with CARLA 0.9.16 Python API.

The controller uses Stanley's approach: steering is based on both the heading error
(angle to the path) and the crosstrack error (lateral distance from the path).

Author: GitHub Copilot Agent
Date: 2025-01-20
Based on: related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/controller2d.py
"""

from typing import List, Tuple
import numpy as np


class PurePursuitController:
    """
    Pure Pursuit controller for lateral (steering) control.

    This controller computes steering commands to follow a path defined by waypoints.
    It combines:
    - Heading error: angle difference between vehicle and path direction
    - Crosstrack error: lateral distance from the vehicle to the path

    The implementation uses Stanley's formula:
        steer = heading_error + atan(k * crosstrack_error / speed)

    Attributes:
        lookahead_distance (float): Distance ahead on path to target (meters)
        kp_heading (float): Gain for heading error term
        k_speed_crosstrack (float): Speed-dependent crosstrack gain
        cross_track_deadband (float): Deadband to reduce oscillations
        conv_rad_to_steer (float): Conversion factor from radians to steering [-1, 1]
    """

    def __init__(
        self,
        lookahead_distance: float = 2.0,
        kp_heading: float = 8.00,
        k_speed_crosstrack: float = 0.00,
        cross_track_deadband: float = 0.01
    ):
        """
        Initialize Pure Pursuit controller.

        Args:
            lookahead_distance: Distance to lookahead point on path in meters
                               (default: 2.0 from controller2d.py)
            kp_heading: Gain for heading error (default: 8.00 from controller2d.py)
            k_speed_crosstrack: Speed-dependent gain for crosstrack error
                               (default: 0.00 from controller2d.py)
            cross_track_deadband: Minimum crosstrack error to react to
                                 (default: 0.01 from controller2d.py)
        """
        self.lookahead_distance = lookahead_distance
        self.kp_heading = kp_heading
        self.k_speed_crosstrack = k_speed_crosstrack
        self.cross_track_deadband = cross_track_deadband

        # Conversion factor from controller2d.py
        # Converts radians to steering range [-1, 1]
        self.conv_rad_to_steer = 180.0 / 70.0 / np.pi

        # Angle normalization constants
        self.pi = np.pi
        self.two_pi = 2.0 * np.pi

    def _get_lookahead_index(
        self,
        current_x: float,
        current_y: float,
        waypoints: List[Tuple[float, float, float]]
    ) -> int:
        """
        Find the index of the waypoint closest to the lookahead distance.

        This method:
        1. Finds the closest waypoint to the vehicle
        2. Accumulates distance along the path until reaching lookahead_distance

        Args:
            current_x: Current vehicle X position (meters)
            current_y: Current vehicle Y position (meters)
            waypoints: List of (x, y, speed) tuples defining the path

        Returns:
            Index of the lookahead waypoint
        """
        # Find closest waypoint
        min_idx = 0
        min_dist = float("inf")

        for i in range(len(waypoints)):
            dist = np.sqrt(
                (waypoints[i][0] - current_x)**2 +
                (waypoints[i][1] - current_y)**2
            )
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        # Accumulate distance from closest waypoint until lookahead distance
        total_dist = min_dist
        lookahead_idx = min_idx

        for i in range(min_idx + 1, len(waypoints)):
            if total_dist >= self.lookahead_distance:
                break

            # Distance from previous waypoint to current
            dist = np.sqrt(
                (waypoints[i][0] - waypoints[i-1][0])**2 +
                (waypoints[i][1] - waypoints[i-1][1])**2
            )
            total_dist += dist
            lookahead_idx = i

        return lookahead_idx

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-π, π] range.

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in [-π, π]
        """
        return (angle + self.pi) % self.two_pi - self.pi

    def update(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        current_speed: float,
        waypoints: List[Tuple[float, float, float]]
    ) -> float:
        """
        Compute steering command based on vehicle state and waypoints.

        The steering is computed using Stanley's formula:
            steer = heading_error + atan(kp * crosstrack_error / (speed + k))

        Args:
            current_x: Current vehicle X position in meters (from carla.Transform)
            current_y: Current vehicle Y position in meters (from carla.Transform)
            current_yaw: Current vehicle yaw in radians (from carla.Transform)
            current_speed: Current vehicle speed in m/s (from carla.Vehicle.get_velocity())
            waypoints: List of (x, y, speed) tuples defining the reference path

        Returns:
            Steering command in [-1.0, 1.0] range

        Example:
            >>> controller = PurePursuitController()
            >>> waypoints = [(0, 0, 5), (10, 0, 5), (20, 5, 5)]
            >>> steer = controller.update(x=5, y=0.5, yaw=0.1, speed=5.0, waypoints=waypoints)
            >>> print(f"Steering: {steer:.3f}")
        """
        # Get lookahead waypoint index
        lookahead_idx = self._get_lookahead_index(current_x, current_y, waypoints)

        # Compute crosstrack error (lateral deviation from path)
        # This is the perpendicular distance from the vehicle to the lookahead point
        crosstrack_vector = np.array([
            waypoints[lookahead_idx][0] - current_x - self.lookahead_distance * np.cos(current_yaw),
            waypoints[lookahead_idx][1] - current_y - self.lookahead_distance * np.sin(current_yaw)
        ])
        crosstrack_error = np.linalg.norm(crosstrack_vector)

        # Apply deadband to reduce oscillations for small errors
        if crosstrack_error < self.cross_track_deadband:
            crosstrack_error = 0.0

        # Determine sign of crosstrack error (left or right of path)
        crosstrack_heading = np.arctan2(crosstrack_vector[1], crosstrack_vector[0])
        crosstrack_heading_error = self._normalize_angle(crosstrack_heading - current_yaw)
        crosstrack_sign = np.sign(crosstrack_heading_error)

        # Compute trajectory heading (direction of path at lookahead point)
        if lookahead_idx < len(waypoints) - 1:
            # Vector from current lookahead to next waypoint
            vect_wp0_to_wp1 = np.array([
                waypoints[lookahead_idx + 1][0] - waypoints[lookahead_idx][0],
                waypoints[lookahead_idx + 1][1] - waypoints[lookahead_idx][1]
            ])
        else:
            # At last waypoint, loop back to start (for circular paths)
            # Or use previous segment direction
            vect_wp0_to_wp1 = np.array([
                waypoints[0][0] - waypoints[-1][0],
                waypoints[0][1] - waypoints[-1][1]
            ])

        trajectory_heading = np.arctan2(vect_wp0_to_wp1[1], vect_wp0_to_wp1[0])

        # Compute heading error (angle difference between vehicle and path)
        heading_error = self._normalize_angle(trajectory_heading - current_yaw)

        # Stanley controller formula:
        # steer = heading_error + atan(k * crosstrack_error / speed)
        steer_rad = heading_error + np.arctan(
            self.kp_heading * crosstrack_sign * crosstrack_error /
            (current_speed + self.k_speed_crosstrack)
        )

        # Convert radians to normalized steering [-1, 1]
        steer_normalized = self.conv_rad_to_steer * steer_rad
        steer_normalized = np.clip(steer_normalized, -1.0, 1.0)

        return steer_normalized

    def __repr__(self) -> str:
        """String representation of the controller."""
        return (
            f"PurePursuitController(lookahead={self.lookahead_distance}m, "
            f"kp_heading={self.kp_heading}, k_speed_crosstrack={self.k_speed_crosstrack})"
        )

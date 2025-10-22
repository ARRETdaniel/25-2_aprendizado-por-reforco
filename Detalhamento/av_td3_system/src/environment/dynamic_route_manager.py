"""
Dynamic Route Manager for CARLA Navigation

Uses CARLA's GlobalRoutePlanner API to generate waypoints dynamically
instead of relying on static waypoints.txt file.

Benefits:
- Topology-aware route planning
- Correct Z-coordinates at road surface
- Scalable to any CARLA map
- Maintains fixed start/end for reproducibility
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
import carla

# Import GlobalRoutePlanner from CARLA agents package
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    logging.warning(
        "Could not import GlobalRoutePlanner from agents.navigation. "
        "Ensure CARLA Python API is installed and PythonAPI/carla/agents is in PYTHONPATH."
    )
    GlobalRoutePlanner = None


class DynamicRouteManager:
    """
    Manages dynamic route generation using CARLA's GlobalRoutePlanner.
    
    Instead of using static waypoints from a file, this class generates
    waypoints on-the-fly using CARLA's road topology and planning algorithms.
    """
    
    def __init__(
        self,
        carla_world: carla.World,
        start_location: Tuple[float, float, float],
        end_location: Tuple[float, float, float],
        sampling_resolution: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize dynamic route manager.
        
        Args:
            carla_world: CARLA world object
            start_location: Start position (x, y, z) - z will be corrected by CARLA
            end_location: End position (x, y, z) - z will be corrected by CARLA
            sampling_resolution: Distance between waypoints in meters (default 2m)
            logger: Logger instance (optional)
        """
        self.world = carla_world
        self.map = carla_world.get_map()
        self.sampling_resolution = sampling_resolution
        self.logger = logger or logging.getLogger(__name__)
        
        # Store start/end locations
        self._start_location_raw = start_location
        self._end_location_raw = end_location
        
        # Get proper waypoints at road surface
        self.start_waypoint = self._get_road_waypoint(start_location)
        self.end_waypoint = self._get_road_waypoint(end_location)
        
        if self.start_waypoint is None or self.end_waypoint is None:
            raise RuntimeError(
                f"Could not find road waypoints for start {start_location} or end {end_location}"
            )
        
        # Initialize GlobalRoutePlanner
        if GlobalRoutePlanner is None:
            raise ImportError(
                "GlobalRoutePlanner not available. Ensure CARLA agents package is installed."
            )
        
        self.route_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)
        
        # Generate initial route
        self.route = []  # List of (carla.Waypoint, RoadOption) tuples
        self.waypoints = []  # NumPy array of waypoint positions (N, 3)
        self._generate_route()
        
        self.logger.info(
            f"DynamicRouteManager initialized:\n"
            f"  Start: ({self.start_waypoint.transform.location.x:.2f}, "
            f"{self.start_waypoint.transform.location.y:.2f}, "
            f"{self.start_waypoint.transform.location.z:.2f})\n"
            f"  End: ({self.end_waypoint.transform.location.x:.2f}, "
            f"{self.end_waypoint.transform.location.y:.2f}, "
            f"{self.end_waypoint.transform.location.z:.2f})\n"
            f"  Total waypoints: {len(self.waypoints)}\n"
            f"  Sampling resolution: {self.sampling_resolution}m"
        )
    
    def _get_road_waypoint(self, location: Tuple[float, float, float]) -> Optional[carla.Waypoint]:
        """
        Get waypoint at road surface for given (x, y, z) location.
        
        Uses CARLA's map.get_waypoint() with project_to_road=True to find
        the nearest driving lane and get the correct Z-coordinate.
        
        Args:
            location: (x, y, z) tuple - z will be ignored and corrected
            
        Returns:
            carla.Waypoint at road surface, or None if not found
        """
        carla_location = carla.Location(x=location[0], y=location[1], z=0.0)
        
        # Get waypoint projected to nearest driving lane
        waypoint = self.map.get_waypoint(
            carla_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if waypoint is None:
            self.logger.warning(
                f"No road waypoint found at ({location[0]:.2f}, {location[1]:.2f})"
            )
        
        return waypoint
    
    def _generate_route(self):
        """
        Generate route using CARLA's GlobalRoutePlanner.
        
        Creates a route from start_waypoint to end_waypoint using the
        road topology. Stores both the full route (with RoadOptions) and
        a simplified waypoints array.
        """
        # Use GlobalRoutePlanner to compute route
        self.route = self.route_planner.trace_route(
            self.start_waypoint.transform.location,
            self.end_waypoint.transform.location
        )
        
        if not self.route:
            raise RuntimeError(
                f"GlobalRoutePlanner could not find route from "
                f"({self.start_waypoint.transform.location.x:.2f}, "
                f"{self.start_waypoint.transform.location.y:.2f}) to "
                f"({self.end_waypoint.transform.location.x:.2f}, "
                f"{self.end_waypoint.transform.location.y:.2f})"
            )
        
        # Extract waypoints as numpy array (N, 3) - [x, y, z]
        self.waypoints = np.array([
            [
                wp.transform.location.x,
                wp.transform.location.y,
                wp.transform.location.z
            ]
            for wp, _ in self.route  # route is list of (waypoint, RoadOption) tuples
        ])
        
        self.logger.info(
            f"Route generated with {len(self.waypoints)} waypoints "
            f"(~{len(self.waypoints) * self.sampling_resolution:.0f}m total distance)"
        )
    
    def get_waypoints(self) -> np.ndarray:
        """
        Get waypoints array compatible with old WaypointManager.
        
        Returns:
            NumPy array of shape (N, 3) with [x, y, z] coordinates
        """
        return self.waypoints
    
    def get_start_transform(self) -> carla.Transform:
        """
        Get spawn transform at route start.
        
        Returns transform at the first waypoint with:
        - Location at road surface + 0.5m offset (avoid Z-collision)
        - Rotation aligned with road heading
        
        Returns:
            carla.Transform for vehicle spawn
        """
        # Get first waypoint's transform
        first_wp = self.route[0][0]  # (waypoint, RoadOption) tuple
        
        # Use waypoint's transform but add +0.5m Z offset
        spawn_location = carla.Location(
            x=first_wp.transform.location.x,
            y=first_wp.transform.location.y,
            z=first_wp.transform.location.z + 0.5  # Avoid Z-collision
        )
        
        # Use waypoint's rotation (aligned with road)
        spawn_rotation = first_wp.transform.rotation
        
        return carla.Transform(spawn_location, spawn_rotation)
    
    def get_next_waypoint_index(
        self,
        vehicle_location: carla.Location,
        current_index: int = 0
    ) -> int:
        """
        Find the index of the next waypoint ahead of the vehicle.
        
        Args:
            vehicle_location: Current vehicle location
            current_index: Last known waypoint index (for optimization)
            
        Returns:
            Index of the next waypoint to target
        """
        # Convert to numpy for distance calculation
        vehicle_pos = np.array([
            vehicle_location.x,
            vehicle_location.y,
            vehicle_location.z
        ])
        
        # Search forward from current index
        min_distance = float('inf')
        best_index = current_index
        
        # Look ahead from current index (optimization)
        search_start = max(0, current_index - 5)
        search_end = min(len(self.waypoints), current_index + 20)
        
        for i in range(search_start, search_end):
            distance = np.linalg.norm(self.waypoints[i] - vehicle_pos)
            
            # Find closest waypoint ahead
            if distance < min_distance and i >= current_index:
                min_distance = distance
                best_index = i
        
        return best_index
    
    def regenerate_route(self, new_end_location: Optional[Tuple[float, float, float]] = None):
        """
        Regenerate route with optional new destination.
        
        Useful for:
        - Generating varied routes between episodes
        - Handling route completion
        - Dynamic goal updates
        
        Args:
            new_end_location: New destination (x, y, z), or None to use original
        """
        if new_end_location is not None:
            self.end_waypoint = self._get_road_waypoint(new_end_location)
            if self.end_waypoint is None:
                self.logger.warning(
                    f"Could not find road waypoint for new end {new_end_location}, "
                    f"keeping original destination"
                )
                return
        
        self._generate_route()
        self.logger.info("Route regenerated")
    
    def get_route_length(self) -> float:
        """
        Calculate total route length in meters.
        
        Returns:
            Approximate route length based on waypoint distances
        """
        if len(self.waypoints) < 2:
            return 0.0
        
        # Sum distances between consecutive waypoints
        total_length = 0.0
        for i in range(len(self.waypoints) - 1):
            total_length += np.linalg.norm(
                self.waypoints[i + 1] - self.waypoints[i]
            )
        
        return total_length
    
    def is_route_complete(
        self,
        vehicle_location: carla.Location,
        threshold_distance: float = 5.0
    ) -> bool:
        """
        Check if vehicle has reached the route destination.
        
        Args:
            vehicle_location: Current vehicle location
            threshold_distance: Distance threshold in meters
            
        Returns:
            True if vehicle is within threshold of destination
        """
        vehicle_pos = np.array([
            vehicle_location.x,
            vehicle_location.y,
            vehicle_location.z
        ])
        
        distance_to_end = np.linalg.norm(self.waypoints[-1] - vehicle_pos)
        
        return distance_to_end < threshold_distance

"""
Utility functions for CARLA environment wrapper.

This module provides helper functions for processing sensor data and
other utility functions for the CARLA environment wrapper.
"""

import numpy as np
import cv2
import time
import math
from typing import Dict, Tuple, List, Any, Union

def transform_sensor_data(sensor_data: Dict) -> Dict:
    """
    Transform raw sensor data from CARLA into a more usable format.

    Args:
        sensor_data: Raw sensor data from CARLA

    Returns:
        Transformed sensor data dictionary
    """
    transformed = {}

    # Process camera image data
    if 'CameraRGB' in sensor_data:
        # Convert from CARLA format to numpy array
        # This implementation depends on the specific CARLA version
        try:
            # For CARLA 0.8.x
            image_data = sensor_data['CameraRGB'].data
            height = sensor_data['CameraRGB'].height
            width = sensor_data['CameraRGB'].width

            # Reshape data into image array
            image = np.frombuffer(image_data, dtype=np.uint8)
            image = image.reshape((height, width, 4))  # RGBA format
            image = image[:, :, :3]  # Convert to RGB

            transformed['image'] = image
        except:
            # If extraction fails, provide a dummy image
            transformed['image'] = np.zeros((84, 84, 3), dtype=np.uint8)

    # Extract vehicle measurements
    if 'measurements' in sensor_data:
        measurements = sensor_data['measurements']
        vehicle = {}

        # Extract position, rotation, velocity
        if hasattr(measurements, 'player_measurements'):
            player = measurements.player_measurements

            # Position
            vehicle['position'] = (
                player.transform.location.x,
                player.transform.location.y,
                player.transform.location.z
            )

            # Orientation (roll, pitch, yaw in radians)
            vehicle['orientation'] = (
                math.radians(player.transform.rotation.roll),
                math.radians(player.transform.rotation.pitch),
                math.radians(player.transform.rotation.yaw)
            )

            # Velocity
            vehicle['velocity'] = (
                player.forward_speed * math.cos(math.radians(player.transform.rotation.yaw)),
                player.forward_speed * math.sin(math.radians(player.transform.rotation.yaw)),
                0.0  # Assuming no vertical velocity
            )

            # Additional vehicle state information
            vehicle['acceleration'] = player.acceleration.x, player.acceleration.y, player.acceleration.z
            vehicle['collision'] = (player.collision_vehicles > 0 or
                                   player.collision_pedestrians > 0 or
                                   player.collision_other > 0)

            transformed['vehicle'] = vehicle

    return transformed

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (84, 84)) -> np.ndarray:
    """
    Preprocess an image for RL input.

    Args:
        image: Raw RGB image array
        target_size: Target size (height, width) for the image

    Returns:
        Preprocessed image as numpy array
    """
    if image is None:
        return np.zeros(target_size + (3,), dtype=np.uint8)

    # Resize
    resized = cv2.resize(image, (target_size[1], target_size[0]))

    # Convert to grayscale if needed
    # grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized

def calculate_distance(point1: Tuple[float, float, float],
                       point2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two 3D points.

    Args:
        point1: First 3D point (x, y, z)
        point2: Second 3D point (x, y, z)

    Returns:
        Euclidean distance
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def calculate_angle(vector1: Tuple[float, float],
                    vector2: Tuple[float, float]) -> float:
    """
    Calculate angle between two 2D vectors.

    Args:
        vector1: First 2D vector (x, y)
        vector2: Second 2D vector (x, y)

    Returns:
        Angle in radians
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Prevent division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Ensure dot/(norm1*norm2) is in the range [-1, 1]
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Determine direction (sign of angle)
    cross = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
    if cross < 0:
        angle = -angle

    return angle

def extract_navigation_info(vehicle_location: Tuple[float, float, float],
                           waypoints: List[Tuple[float, float, float]]) -> Dict[str, float]:
    """
    Extract navigation information relative to waypoints.

    Args:
        vehicle_location: Vehicle position (x, y, z)
        waypoints: List of waypoints (x, y, z)

    Returns:
        Dictionary with navigation information
    """
    if not waypoints:
        return {'distance': 0.0, 'angle': 0.0, 'curvature': 0.0}

    # Find closest waypoint
    distances = [calculate_distance(vehicle_location, wp) for wp in waypoints]
    closest_idx = np.argmin(distances)
    closest_wp = waypoints[closest_idx]

    # Calculate distance to closest waypoint
    distance = distances[closest_idx]

    # Calculate angle to waypoint (in 2D, ignoring z)
    vehicle_2d = (vehicle_location[0], vehicle_location[1])
    waypoint_2d = (closest_wp[0], closest_wp[1])

    # Vehicle forward vector (assuming it's available or calculable)
    # For demonstration, use a default vector
    vehicle_forward = (1.0, 0.0)

    # Vector to waypoint
    to_waypoint = (waypoint_2d[0] - vehicle_2d[0], waypoint_2d[1] - vehicle_2d[1])

    # Normalize
    norm = math.sqrt(to_waypoint[0]**2 + to_waypoint[1]**2)
    if norm > 0:
        to_waypoint = (to_waypoint[0]/norm, to_waypoint[1]/norm)

    # Calculate angle
    angle = calculate_angle(vehicle_forward, to_waypoint)

    # Calculate road curvature (from waypoints)
    curvature = 0.0
    if len(waypoints) >= 3 and closest_idx + 1 < len(waypoints):
        # Very basic curvature calculation
        # For a proper calculation, use circle fitting or other methods
        p1 = waypoints[max(0, closest_idx - 1)]
        p2 = waypoints[closest_idx]
        p3 = waypoints[min(len(waypoints) - 1, closest_idx + 1)]

        # Convert to 2D for simplicity
        p1_2d = (p1[0], p1[1])
        p2_2d = (p2[0], p2[1])
        p3_2d = (p3[0], p3[1])

        # Calculate vectors
        v1 = (p2_2d[0] - p1_2d[0], p2_2d[1] - p1_2d[1])
        v2 = (p3_2d[0] - p2_2d[0], p3_2d[1] - p2_2d[1])

        # Calculate angle between vectors as a simple curvature measure
        angle_between = calculate_angle(v1, v2)

        # Distance between points
        d1 = calculate_distance(p1, p2)
        d2 = calculate_distance(p2, p3)

        if d1 > 0 and d2 > 0:
            # Curvature is approximately angle change per unit distance
            curvature = angle_between / ((d1 + d2) / 2)

    return {
        'distance': distance,
        'angle': angle,
        'curvature': curvature
    }

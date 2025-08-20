"""
State Processor for CARLA environment.

This module processes raw sensor data into a format suitable for RL algorithms.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Any

class StateProcessor:
    """
    Process raw sensor data into a state representation for RL.

    This class handles the transformation of raw sensor data (images,
    vehicle measurements, etc.) into a structured state representation
    suitable for reinforcement learning algorithms.

    Attributes:
        image_size: Tuple[int, int], target size of processed images
    """

    def __init__(self, image_size: Tuple[int, int] = (84, 84)):
        """
        Initialize the state processor.

        Args:
            image_size: Target size (height, width) for processed images
        """
        self.image_size = image_size

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a raw camera image.

        Args:
            image: Raw RGB image array from camera

        Returns:
            Processed image as numpy array
        """
        if image is None or image.size == 0:
            return np.zeros(self.image_size + (3,), dtype=np.uint8)

        # Resize image
        processed = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        # Convert to RGB if necessary
        if len(processed.shape) == 2:  # If grayscale
            processed = np.stack([processed] * 3, axis=2)
        elif processed.shape[2] == 4:  # If RGBA
            processed = processed[:, :, :3]

        # Normalize to [0, 1]
        processed = processed.astype(np.float32) / 255.0

        return processed

    def process_vehicle_state(self,
                              position: Tuple[float, float, float],
                              velocity: Tuple[float, float, float],
                              orientation: Tuple[float, float, float]) -> np.ndarray:
        """
        Process vehicle state information.

        Args:
            position: (x, y, z) position of vehicle
            velocity: (vx, vy, vz) velocity of vehicle
            orientation: (roll, pitch, yaw) orientation in radians

        Returns:
            Processed vehicle state as numpy array
        """
        # Combine and normalize
        state = np.array([
            *position,
            *velocity,
            *orientation
        ], dtype=np.float32)

        return state

    def process_navigation(self,
                           distance_to_waypoint: float,
                           angle_to_waypoint: float,
                           road_curvature: float) -> np.ndarray:
        """
        Process navigation information.

        Args:
            distance_to_waypoint: Distance to next waypoint in meters
            angle_to_waypoint: Angle to next waypoint in radians
            road_curvature: Curvature of the road ahead

        Returns:
            Processed navigation info as numpy array
        """
        # Normalize
        distance_normalized = np.clip(distance_to_waypoint / 100.0, 0, 1)  # Assuming 100m is max relevant distance
        angle_normalized = angle_to_waypoint / np.pi  # Normalize to [-1, 1]
        curvature_normalized = np.clip(road_curvature / 0.1, -1, 1)  # Assuming 0.1 is high curvature

        nav = np.array([
            distance_normalized,
            angle_normalized,
            curvature_normalized
        ], dtype=np.float32)

        return nav

    def process_detections(self, detections: List[Dict]) -> np.ndarray:
        """
        Process object detection information.

        Args:
            detections: List of detection dictionaries with information about
                        detected objects (class, position, etc.)

        Returns:
            Processed detections as numpy array
        """
        # Create a fixed-size array for detections
        # Each detection has: class, distance, angle, relative velocity, confidence
        max_detections = 2  # Maximum number of detections to consider
        detection_size = 5   # Size of each detection features

        detection_array = np.zeros((max_detections, detection_size), dtype=np.float32)

        # Process each detection up to max_detections
        for i, detection in enumerate(detections[:max_detections]):
            if i >= max_detections:
                break

            # Extract and normalize detection features
            object_class = detection.get('class_id', 0) / 10.0  # Normalize class ID
            distance = detection.get('distance', 100.0) / 100.0  # Normalize distance
            angle = detection.get('angle', 0.0) / np.pi  # Normalize angle
            rel_velocity = detection.get('relative_velocity', 0.0) / 30.0  # Normalize velocity
            confidence = detection.get('confidence', 0.0)  # Already normalized

            detection_array[i] = [object_class, distance, angle, rel_velocity, confidence]

        # Flatten the array
        flattened = detection_array.flatten()

        return flattened

    def process(self, raw_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process a complete raw state into a structured state representation.

        Args:
            raw_state: Dictionary containing raw sensor data

        Returns:
            Processed state as a dictionary with numpy arrays
        """
        processed_state = {}

        # Process image if available
        if 'image' in raw_state:
            processed_state['image'] = self.process_image(raw_state['image'])
        else:
            processed_state['image'] = np.zeros(self.image_size + (3,), dtype=np.float32)

        # Process vehicle state
        if 'vehicle' in raw_state:
            vehicle = raw_state['vehicle']
            processed_state['vehicle_state'] = self.process_vehicle_state(
                position=vehicle.get('position', (0, 0, 0)),
                velocity=vehicle.get('velocity', (0, 0, 0)),
                orientation=vehicle.get('orientation', (0, 0, 0))
            )
        else:
            processed_state['vehicle_state'] = np.zeros(9, dtype=np.float32)

        # Process navigation info
        if 'navigation' in raw_state:
            nav = raw_state['navigation']
            processed_state['navigation'] = self.process_navigation(
                distance_to_waypoint=nav.get('distance', 0.0),
                angle_to_waypoint=nav.get('angle', 0.0),
                road_curvature=nav.get('curvature', 0.0)
            )
        else:
            processed_state['navigation'] = np.zeros(3, dtype=np.float32)

        # Process detections
        if 'detections' in raw_state:
            processed_state['detections'] = self.process_detections(raw_state['detections'])
        else:
            processed_state['detections'] = np.zeros(10, dtype=np.float32)

        return processed_state

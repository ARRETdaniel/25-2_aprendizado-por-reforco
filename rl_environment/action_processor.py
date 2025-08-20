"""
Action Processor for CARLA environment.

This module processes RL agent actions into CARLA vehicle controls.
"""

import numpy as np
from typing import Tuple, Union, Dict, Any

class ActionProcessor:
    """
    Process RL agent actions into CARLA vehicle controls.

    This class handles the transformation of actions from the RL agent
    into control commands suitable for the CARLA vehicle controller.

    Attributes:
        action_space_type: Type of action space ('continuous' or 'discrete')
        discrete_actions: List of discrete actions if using discrete action space
    """

    def __init__(self, action_space_type: str = 'continuous'):
        """
        Initialize the action processor.

        Args:
            action_space_type: Type of action space to use
                               ('continuous' or 'discrete')
        """
        self.action_space_type = action_space_type

        # Define discrete actions if needed
        if action_space_type == 'discrete':
            self.discrete_actions = [
                {'throttle': 0.0, 'brake': 0.0, 'steer': 0.0},  # Idle
                {'throttle': 0.5, 'brake': 0.0, 'steer': 0.0},  # Accelerate
                {'throttle': 1.0, 'brake': 0.0, 'steer': 0.0},  # Full acceleration
                {'throttle': 0.0, 'brake': 0.5, 'steer': 0.0},  # Brake
                {'throttle': 0.0, 'brake': 1.0, 'steer': 0.0},  # Full brake
                {'throttle': 0.5, 'brake': 0.0, 'steer': -0.5},  # Accelerate and steer left
                {'throttle': 0.5, 'brake': 0.0, 'steer': 0.5},   # Accelerate and steer right
                {'throttle': 0.0, 'brake': 0.0, 'steer': -0.5},  # Steer left
                {'throttle': 0.0, 'brake': 0.0, 'steer': 0.5}    # Steer right
            ]

    def process(self, action: Union[np.ndarray, int]) -> Dict[str, float]:
        """
        Process an RL agent action into vehicle control.

        Args:
            action: Either a continuous action array [throttle, brake, steer]
                   or a discrete action index

        Returns:
            Dictionary with throttle, brake, and steer values
        """
        if self.action_space_type == 'continuous':
            return self._process_continuous(action)
        else:
            return self._process_discrete(action)

    def _process_continuous(self, action: np.ndarray) -> Dict[str, float]:
        """
        Process a continuous action.

        Args:
            action: Array with [throttle, brake, steer] values

        Returns:
            Dictionary with throttle, brake, and steer values
        """
        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Clip values to appropriate ranges
        throttle = np.clip(action[0], 0.0, 1.0)
        brake = np.clip(action[1], 0.0, 1.0)
        steer = np.clip(action[2], -1.0, 1.0)

        # Return control dictionary
        return {
            'throttle': float(throttle),
            'brake': float(brake),
            'steer': float(steer)
        }

    def _process_discrete(self, action_idx: int) -> Dict[str, float]:
        """
        Process a discrete action.

        Args:
            action_idx: Index into the discrete action space

        Returns:
            Dictionary with throttle, brake, and steer values
        """
        # Ensure valid action index
        if action_idx >= len(self.discrete_actions) or action_idx < 0:
            action_idx = 0  # Default to idle if invalid

        # Return the corresponding discrete action
        return self.discrete_actions[action_idx]

    def get_carla_control(self, control_dict: Dict[str, float]) -> Any:
        """
        Convert control dictionary to CARLA VehicleControl object.

        Args:
            control_dict: Dictionary with throttle, brake, and steer values

        Returns:
            CARLA VehicleControl object

        Note:
            This method requires the CARLA VehicleControl class to be imported.
            For implementation without importing CARLA, it returns the dictionary.
        """
        # Try to import CARLA and create a VehicleControl object
        try:
            from carla.client import VehicleControl
            return VehicleControl(
                throttle=control_dict['throttle'],
                brake=control_dict['brake'],
                steer=control_dict['steer']
            )
        except ImportError:
            # If CARLA is not available, just return the dictionary
            return control_dict

"""
RL Environment package for CARLA simulator.

This package provides a reinforcement learning environment
wrapper for the CARLA simulator, implementing a gym-like interface.
"""

# Import components
from .environment import CarlaEnvWrapper
from .state_processor import StateProcessor
from .action_processor import ActionProcessor
from .reward_function import RewardFunction
from .utils import transform_sensor_data, preprocess_image

__all__ = [
    'CarlaEnvWrapper',
    'StateProcessor',
    'ActionProcessor',
    'RewardFunction',
    'transform_sensor_data',
    'preprocess_image'
]

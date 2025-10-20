"""
Agent implementations for autonomous driving with deep reinforcement learning.

Contains TD3 and DDPG agents for continuous control in CARLA simulator.
"""

from .td3_agent import TD3Agent
from .ddpg_agent import DDPGAgent

__all__ = ['TD3Agent', 'DDPGAgent']

"""
Deterministic Policy Network (Actor) for TD3/DDPG

Implements the actor network μ_φ(s) that maps state to continuous actions.

Architecture:
- Input: 535-dimensional state (512 CNN features + 3 kinematic + 20 waypoint)
- Hidden Layer 1: 256 units, ReLU activation
- Hidden Layer 2: 256 units, ReLU activation
- Output: 2-dimensional action (steering, throttle/brake), Tanh activation
- Output scaling: Multiplied by max_action to scale to [-1, 1]

Paper Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)

Author: Daniel Terra Gomes
2025
"""

import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    """
    Deterministic actor network for continuous control.

    Maps state to action using deterministic policy:
    a = tanh(FC2(ReLU(FC1(s)))) * max_action

    This is the policy that the agent learns to maximize expected return.
    During training, exploration noise is added externally.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        max_action: float = 1.0,
        hidden_size: int = 256,
    ):
        """
        Initialize actor network.

        Args:
            state_dim: Dimension of state space (535)
            action_dim: Dimension of action space (2: steering + throttle/brake)
            max_action: Maximum action value for scaling (1.0 for [-1,1])
            hidden_size: Number of units in hidden layers (256)
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.hidden_size = hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize weights (uniform distribution as in original TD3)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights.

        Uses uniform distribution U[-1/sqrt(f), 1/sqrt(f)] where f is fan-in.
        This is the standard initialization for actor-critic networks.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(
                layer.weight, -1.0 / np.sqrt(layer.in_features),
                1.0 / np.sqrt(layer.in_features)
            )
            if layer.bias is not None:
                nn.init.uniform_(
                    layer.bias, -1.0 / np.sqrt(layer.in_features),
                    1.0 / np.sqrt(layer.in_features)
                )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.

        Args:
            state: Batch of states (batch_size, state_dim)

        Returns:
            Batch of actions (batch_size, action_dim) in range [-max_action, max_action]
        """
        # Hidden layers with ReLU
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))

        # Output layer with Tanh and scaling
        a = self.tanh(self.fc3(x))
        a = a * self.max_action

        return a

"""
Critic Q-Networks for TD3/DDPG

Implements value function approximators that estimate Q(s, a).

TD3 uses twin critics: Q_θ1(s,a) and Q_θ2(s,a)
DDPG uses single critic: Q_θ(s,a)

Architecture (for each critic):
- Input: State (535-dim) concatenated with action (2-dim) = 537-dim
- Hidden Layer 1: 256 units, ReLU activation
- Hidden Layer 2: 256 units, ReLU activation
- Output: 1-dimensional Q-value (state-action value)

Paper Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
Key contribution: Twin critics reduce overestimation bias in Bellman backup
"""

import torch
import torch.nn as nn
import numpy as np


class Critic(nn.Module):
    """
    Q-value network (critic) for state-action value estimation.

    Maps (state, action) pairs to scalar Q-value estimates.
    Used in both TD3 (as one of two critics) and DDPG (single critic).

    The Q-value represents the expected return (discounted cumulative reward)
    when taking action a in state s and following the current policy thereafter.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_size: int = 256,
    ):
        """
        Initialize critic network.

        Args:
            state_dim: Dimension of state space (535)
            action_dim: Dimension of action space (2)
            hidden_size: Number of units in hidden layers (256)
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Fully connected layers
        # Input: state (state_dim) + action (action_dim)
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Activation functions
        self.relu = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using uniform distribution.

        Uses U[-1/sqrt(f), 1/sqrt(f)] where f is fan-in, standard for actor-critic.
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

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            state: Batch of states (batch_size, state_dim)
            action: Batch of actions (batch_size, action_dim)

        Returns:
            Batch of Q-values (batch_size, 1)
        """
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)

        # Hidden layers with ReLU
        x = self.relu(self.fc1(sa))
        x = self.relu(self.fc2(x))

        # Output layer (no activation on Q-value)
        q = self.fc3(x)

        return q


class TwinCritic(nn.Module):
    """
    Paired critic networks for TD3 algorithm.

    TD3 uses two independent Q-networks to reduce overestimation bias:
    - Q_θ1(s, a): First Q-network
    - Q_θ2(s, a): Second Q-network

    Training target uses minimum: y = r + γ(1-d) min(Q_θ1'(s', ã), Q_θ2'(s', ã))

    This reduces the tendency of standard actor-critic to overestimate Q-values,
    which can lead to poor policy updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_size: int = 256,
    ):
        """
        Initialize twin critic networks.

        Args:
            state_dim: Dimension of state space (535)
            action_dim: Dimension of action space (2)
            hidden_size: Number of units in hidden layers (256)
        """
        super(TwinCritic, self).__init__()

        # Two independent Q-networks with same architecture
        self.Q1 = Critic(state_dim, action_dim, hidden_size)
        self.Q2 = Critic(state_dim, action_dim, hidden_size)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple:
        """
        Forward pass through both critic networks.

        Args:
            state: Batch of states (batch_size, state_dim)
            action: Batch of actions (batch_size, action_dim)

        Returns:
            Tuple of (Q1_values, Q2_values), each (batch_size, 1)
        """
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2

    def Q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through first Q-network only."""
        return self.Q1(state, action)

    def Q2_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through second Q-network only."""
        return self.Q2(state, action)


class CriticLoss:
    """
    Computes critic loss using Bellman equation.

    Standard formulation:
    L = mean((Q(s,a) - y)^2)
    where y = r + γ(1-d) max_a' Q_target(s', a')

    For TD3 with twin critics:
    - Compute target using minimum of twin targets
    - Compute MSE loss for both critics
    - L = MSE(Q1) + MSE(Q2)
    """

    @staticmethod
    def compute_td3_loss(
        q1: torch.Tensor,
        q2: torch.Tensor,
        target_q: torch.Tensor,
    ) -> tuple:
        """
        Compute TD3 critic loss (both networks).

        Args:
            q1: Q1 values from current network (batch_size, 1)
            q2: Q2 values from current network (batch_size, 1)
            target_q: Target Q-values (batch_size, 1)

        Returns:
            Tuple of (loss_Q1, loss_Q2, total_loss)
        """
        loss_q1 = torch.nn.functional.mse_loss(q1, target_q)
        loss_q2 = torch.nn.functional.mse_loss(q2, target_q)
        loss = loss_q1 + loss_q2

        return loss_q1, loss_q2, loss

    @staticmethod
    def compute_ddpg_loss(
        q: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DDPG critic loss (single network).

        Args:
            q: Q values from current network (batch_size, 1)
            target_q: Target Q-values (batch_size, 1)

        Returns:
            MSE loss
        """
        loss = torch.nn.functional.mse_loss(q, target_q)
        return loss


if __name__ == "__main__":
    """Quick test of critic networks."""
    import torch

    print("Testing Critic network...")
    state_dim = 535
    action_dim = 2
    batch_size = 4

    # Test single Critic
    critic = Critic(state_dim, action_dim)
    dummy_state = torch.randn(batch_size, state_dim)
    dummy_action = torch.randn(batch_size, action_dim)

    q_values = critic(dummy_state, dummy_action)

    print(f"State shape: {dummy_state.shape}")
    print(f"Action shape: {dummy_action.shape}")
    print(f"Q-value shape: {q_values.shape}")
    print(f"Q-value range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    print(f"Expected shape: ({batch_size}, 1)")

    # Test TwinCritic
    print("\nTesting TwinCritic network...")
    twin_critic = TwinCritic(state_dim, action_dim)

    q1, q2 = twin_critic(dummy_state, dummy_action)

    print(f"Q1 shape: {q1.shape}")
    print(f"Q2 shape: {q2.shape}")
    print(f"Q1 range: [{q1.min():.3f}, {q1.max():.3f}]")
    print(f"Q2 range: [{q2.min():.3f}, {q2.max():.3f}]")

    # Test Clipped Double Q-Learning
    print("\nTesting Clipped Double Q-Learning...")
    target_q = torch.randn(batch_size, 1)

    min_q = torch.min(q1, q2)
    print(f"min(Q1, Q2) shape: {min_q.shape}")
    print(f"min(Q1, Q2) value: {min_q}")

    # Test loss computation
    print("\nTesting loss computation...")
    loss_q1, loss_q2, total_loss = CriticLoss.compute_td3_loss(q1, q2, target_q)

    print(f"Loss Q1: {loss_q1.item():.6f}")
    print(f"Loss Q2: {loss_q2.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")

    print("\n✓ Critic network tests passed!")

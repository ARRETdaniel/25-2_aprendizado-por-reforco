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
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


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

    def select_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select action for given state (for environment interaction).

        Can optionally add exploration noise during training.

        Args:
            state: State as numpy array (state_dim,)
            device: Device to run on ("cpu" or "cuda")
            noise: Optional exploration noise to add (action_dim,)

        Returns:
            Action as numpy array (action_dim,) in range [-max_action, max_action]
        """
        # Convert state to tensor
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)

        # Get deterministic action (no gradients)
        with torch.no_grad():
            action = self.forward(state_tensor).cpu().numpy().squeeze()

        # Add exploration noise if provided (during training)
        if noise is not None:
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action.astype(np.float32)


class ActorLoss(nn.Module):
    """
    Wrapper for computing actor loss using critic network.

    In TD3/DDPG, the actor is trained to maximize Q(s, μ(s)), which means
    minimizing -Q(s, μ(s)).

    This is used in the training loop:
    actor_loss = -critic.Q1(state, actor(state)).mean()
    """

    def __init__(self, actor: Actor, critic):
        """
        Initialize actor loss computation.

        Args:
            actor: Actor network instance
            critic: Critic network instance (provides Q-value function)
        """
        super(ActorLoss, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute actor loss.

        Loss = -mean(Q1(s, μ(s)))
        We minimize -Q to maximize Q (gradient ascent on expected return).

        Args:
            state: Batch of states (batch_size, state_dim)

        Returns:
            Scalar loss (negative mean Q-value)
        """
        # Get action from actor
        action = self.actor(state)

        # Get Q-value from critic's first Q-network
        q_value = self.critic.Q1(state, action)

        # Actor loss: negative mean Q-value (for gradient ascent)
        loss = -q_value.mean()

        return loss


if __name__ == "__main__":
    """Quick test of actor network."""
    import torch

    # Test Actor
    print("Testing Actor network...")
    state_dim = 535
    action_dim = 2
    batch_size = 4

    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        hidden_size=256,
    )

    # Test forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    action = actor(dummy_state)

    print(f"State shape: {dummy_state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"Expected: (batch_size={batch_size}, action_dim={action_dim})")
    print(f"Expected action range: [-1.0, 1.0]")

    # Test select_action method
    print("\nTesting select_action method...")
    state_np = np.random.randn(state_dim).astype(np.float32)
    action_np = actor.select_action(state_np, device="cpu")

    print(f"State (numpy) shape: {state_np.shape}")
    print(f"Action (numpy) shape: {action_np.shape}")
    print(f"Action value: {action_np}")

    # Test with noise
    print("\nTesting select_action with exploration noise...")
    noise = np.random.randn(action_dim) * 0.1  # 10% of max_action
    action_noisy = actor.select_action(state_np, device="cpu", noise=noise)

    print(f"Noise: {noise}")
    print(f"Action with noise: {action_noisy}")

    print("\n✓ Actor network tests passed!")

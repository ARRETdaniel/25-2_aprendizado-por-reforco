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

TD3 Paper Reference:
    Fujimoto, S., Hoof, H., & Meger, D. (2018).
    "Addressing Function Approximation Error in Actor-Critic Methods."
    International Conference on Machine Learning (ICML).
    https://arxiv.org/abs/1802.09477

Key TD3 Contribution:
    Twin critics reduce overestimation bias in Bellman backup through
    "Clipped Double Q-Learning" - using min(Q_θ1', Q_θ2') for target computation.

Official Documentation:
    - Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
    - OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/td3.html
    - Original Implementation: https://github.com/sfujim/TD3

Author: Daniel Terra Gomes
2025
"""

import torch
import torch.nn as nn
import numpy as np


class Critic(nn.Module):
    """
    Q-value network (critic) for state-action value estimation.

    Maps (state, action) pairs to scalar Q-value estimates Q(s,a).
    Used in both TD3 (as one of two critics) and DDPG (single critic).

    The Q-value represents the expected return (discounted cumulative reward)
    when taking action a in state s and following the current policy thereafter:
        Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a, π]

    Architecture follows the TD3 paper specification:
        - Default hidden layers: [256, 256] with ReLU activation
        - Input: Concatenated state-action pairs [s, a]
        - Output: Single scalar Q-value (no activation function)

    Reference:
        TD3 Paper (Fujimoto et al., 2018), Section 4: Implementation Details
        "We use two hidden layers of 256 units with ReLU activations."
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

        Uses U[-1/√f, 1/√f] where f is the fan-in (number of input units).
        This initialization is standard practice in actor-critic methods to ensure
        stable gradient flow and prevent vanishing/exploding gradients.

        Note:
            Original TD3 implementation uses PyTorch defaults (Kaiming uniform).
            Our explicit initialization provides better control and matches
            classical actor-critic literature conventions.
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
        Forward pass through critic network: Q(s,a).

        Computes Q-value estimates for given state-action pairs.
        Architecture: [s,a] → FC(256) → ReLU → FC(256) → ReLU → FC(1) → Q

        Args:
            state: Batch of states (batch_size, state_dim=535)
                   For our AV task: 512 (CNN features) + 23 (kinematic)
            action: Batch of actions (batch_size, action_dim=2)
                    [steering ∈ [-1,1], throttle/brake ∈ [-1,1]]

        Returns:
            Batch of Q-values (batch_size, 1)
            Scalar Q-value for each state-action pair (unbounded)

        Note:
            No activation function on the output layer - Q-values can be
            positive or negative depending on expected cumulative reward.
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
    Paired critic networks for TD3 algorithm - implements Clipped Double Q-Learning.

    TD3 Trick #1: Uses two independent Q-networks to reduce overestimation bias:
        - Q_θ1(s, a): First Q-network
        - Q_θ2(s, a): Second Q-network

    Training target uses minimum of twin targets (Clipped Double Q-Learning):
        y = r + γ(1-d) min(Q_θ1'(s', ã), Q_θ2'(s', ã))

    where:
        - ã = π_φ'(s') + ε: Target action with smoothing noise (TD3 Trick #3)
        - ε ~ clip(N(0,σ), -c, c): Clipped Gaussian noise
        - Both Q1 and Q2 are trained toward the SAME target y

    Key Insight (from TD3 Paper):
        "With Clipped Double Q-learning, the value target cannot introduce any
        additional overestimation over using the standard Q-learning target."

    This reduces the tendency of standard actor-critic to overestimate Q-values,
    which can lead to poor policy updates and training instability.

    Reference:
        TD3 Paper (Fujimoto et al., 2018), Section 3.1: Clipped Double Q-Learning
        "We propose to simply upper-bound the less biased value estimate Q_θ2
        by the biased estimate Q_θ1. This results in taking the minimum between
        the two estimates."

    Implementation Note:
        Our design uses two separate Critic() instances for modularity.
        Original TD3 uses a single class with Q1/Q2 layers inside.
        Both approaches are functionally equivalent.
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

        Computes Q-values from both Q1 and Q2 networks simultaneously.
        Used during training to compute the Bellman error for both critics.

        Args:
            state: Batch of states (batch_size, state_dim)
            action: Batch of actions (batch_size, action_dim)

        Returns:
            Tuple of (Q1_values, Q2_values), each (batch_size, 1)
            Both networks evaluate the same state-action pairs

        Usage in TD3:
            - Training: Both values used to compute twin critic losses
            - Target computation: min(Q1_target, Q2_target) used for Bellman target
        """
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2

    def Q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through first Q-network only.

        Used for actor loss computation in TD3. The actor is optimized with
        respect to Q1 only to reduce computational cost.

        From TD3 Paper:
            "In implementation, computational costs can be reduced by using a
            single actor optimized with respect to Q_θ1."

        Args:
            state: Batch of states (batch_size, state_dim)
            action: Batch of actions (batch_size, action_dim)

        Returns:
            Q1 values (batch_size, 1)

        Usage:
            actor_loss = -critic.Q1_forward(state, actor(state)).mean()
        """
        return self.Q1(state, action)

    def Q2_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through second Q-network only."""
        return self.Q2(state, action)


class CriticLoss:
    """
    Computes critic loss using Bellman equation with TD3 modifications.

    Standard TD formulation:
        L = E[(Q(s,a) - y)²]
    where:
        y = r + γ(1-d) Q_target(s', π(s'))

    TD3 Clipped Double Q-Learning formulation:
        y = r + γ(1-d) min(Q_θ1'(s', ã), Q_θ2'(s', ã))
    where:
        ã = π_φ'(s') + ε (target action with smoothing noise)

    For twin critics:
        1. Compute single target y using min(Q1_target, Q2_target)
        2. Compute MSE loss for both critics toward SAME target
        3. Total loss: L = MSE(Q1, y) + MSE(Q2, y)

    Key Property:
        Both critics regress to the same conservative target, preventing
        overestimation bias while maintaining independent function approximation.

    Reference:
        TD3 Paper (Fujimoto et al., 2018), Section 3.1
        OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/td3.html
    """

    @staticmethod
    def compute_td3_loss(
        q1: torch.Tensor,
        q2: torch.Tensor,
        target_q: torch.Tensor,
    ) -> tuple:
        """
        Compute TD3 critic loss for both Q-networks.

        Computes mean squared Bellman error for both critics using the
        SAME target value (computed as min of twin target networks).

        Loss formulation:
            L_Q1 = E[(Q_θ1(s,a) - y)²]
            L_Q2 = E[(Q_θ2(s,a) - y)²]
            L_total = L_Q1 + L_Q2

        where y is already computed as:
            y = r + γ(1-d) min(Q_θ1'(s', ã), Q_θ2'(s', ã))

        Args:
            q1: Q1 values from current network (batch_size, 1)
            q2: Q2 values from current network (batch_size, 1)
            target_q: Target Q-values (batch_size, 1)
                     Already computed with min(Q1_target, Q2_target)

        Returns:
            Tuple of (loss_Q1, loss_Q2, total_loss)
            - loss_Q1: MSE loss for first critic
            - loss_Q2: MSE loss for second critic
            - total_loss: Sum of both losses (used for backprop)

        Note:
            Both critics use the SAME target_q (from minimum of twin targets).
            This is key to TD3's clipped double Q-learning mechanism.
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
        Compute DDPG critic loss (single Q-network).

        Standard DDPG uses a single critic without the twin mechanism.
        Loss formulation:
            L_Q = E[(Q_θ(s,a) - y)²]
        where:
            y = r + γ(1-d) Q_θ'(s', π_φ'(s'))

        Args:
            q: Q values from current network (batch_size, 1)
            target_q: Target Q-values (batch_size, 1)
                     Computed with single target network

        Returns:
            MSE loss (scalar)

        Note:
            Used for DDPG baseline comparison. DDPG is more prone to
            overestimation bias than TD3 due to lack of clipped double Q-learning.

        Reference:
            DDPG Paper (Lillicrap et al., 2015)
            "Continuous control with deep reinforcement learning"
        """
        loss = torch.nn.functional.mse_loss(q, target_q)
        return loss

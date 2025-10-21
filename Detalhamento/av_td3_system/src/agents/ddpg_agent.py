"""
Deep Deterministic Policy Gradient (DDPG) Agent

Implements the DDPG algorithm for continuous control in autonomous driving.
DDPG is an off-policy actor-critic algorithm that learns a deterministic policy
by directly optimizing the policy gradient. It uses experience replay and target
networks for stability.

Reference: "Continuous Control with Deep Reinforcement Learning"
           (Lillicrap et al., ICLR 2016)

Note: This implementation uses IDENTICAL hyperparameters and architecture to TD3
for FAIR ALGORITHMIC COMPARISON. The only differences are:
- Single Critic network (not twin)
- No delayed policy updates (policy_freq=1)
- No target policy smoothing (policy_noise=0.0)

Author: Daniel Terra
Date: 2024
"""

import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.networks.actor import Actor
from src.networks.critic import Critic  # Single critic for DDPG
from src.utils.replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    DDPG agent for autonomous vehicle control.

    The agent learns a deterministic policy that maps states (visual features +
    kinematic data + waypoints) to continuous actions (steering + throttle/brake).

    DDPG Mechanisms:
    - Off-policy learning with experience replay
    - Deterministic policy gradient
    - Single Q-network for value estimation
    - Target networks for stable learning (soft updates)
    - Exploration via action noise (not policy smoothing)

    Key Attributes:
        actor: Policy network μ_φ(s)
        actor_target: Target policy for stable learning
        critic: Q-network Q_θ(s,a)
        critic_target: Target Q-network for stable learning
        replay_buffer: Experience replay buffer
        total_it: Total training iterations
    """

    def __init__(
        self,
        state_dim: int = 535,
        action_dim: int = 2,
        max_action: float = 1.0,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize DDPG agent with networks and hyperparameters.

        Args:
            state_dim: Dimension of state space (default: 535)
                      = 512 (CNN features) + 3 (kinematic) + 20 (waypoints)
            action_dim: Dimension of action space (default: 2)
                       = [steering, throttle/brake]
            max_action: Maximum absolute value of actions (default: 1.0)
            config: Dictionary with DDPG hyperparameters (if None, loads from file)
            config_path: Path to YAML config file (default: config/ddpg_config.yaml)
        """
        # Load configuration
        if config is None:
            if config_path is None:
                config_path = "config/ddpg_config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        # Store dimensions and config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.config = config

        # Extract hyperparameters from config
        algo_config = config['algorithm']
        self.discount = algo_config['gamma']  # Discount factor γ
        self.tau = algo_config['tau']  # Soft update rate for target networks
        # Note: DDPG does NOT use target policy smoothing
        self.actor_lr = algo_config['actor_lr']
        self.critic_lr = algo_config['critic_lr']

        # Training config
        training_config = config['training']
        self.batch_size = training_config['batch_size']
        self.buffer_size = training_config['buffer_size']
        self.start_timesteps = training_config['start_timesteps']

        # Exploration config
        exploration_config = config['exploration']
        self.expl_noise = exploration_config['expl_noise']

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DDPGAgent initialized on device: {self.device}")

        # Initialize actor networks
        network_config = config['networks']['actor']
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_sizes=network_config['hidden_sizes']
        ).to(self.device)

        # Create target actor as deep copy
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr
        )

        # Initialize SINGLE critic network (unlike TD3's twin critics)
        critic_config = config['networks']['critic']
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=critic_config['hidden_sizes']
        ).to(self.device)

        # Create target critic as deep copy
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=self.buffer_size,
            device=self.device
        )

        # Training iteration counter
        self.total_it = 0

        print(f"DDPGAgent initialized with:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Actor hidden sizes: {network_config['hidden_sizes']}")
        print(f"  Critic hidden sizes: {critic_config['hidden_sizes']}")
        print(f"  Discount γ: {self.discount}, Tau τ: {self.tau}")
        print(f"  Exploration noise: {self.expl_noise}")
        print(f"  Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
        print(f"\n  DDPG vs TD3 Configuration:")
        print(f"    - Using SINGLE Critic (not Twin)")
        print(f"    - Policy updated EVERY step (policy_freq=1, not delayed)")
        print(f"    - No target policy smoothing (noise applied to exploration only)")

    def select_action(
        self,
        state: np.ndarray,
        noise: Optional[float] = None
    ) -> np.ndarray:
        """
        Select action from current policy with optional exploration noise.

        During training, Gaussian noise is added for exploration. During evaluation,
        the deterministic policy is used (noise=0).

        Args:
            state: Current state observation (535-dim numpy array)
            noise: Std dev of Gaussian exploration noise. If None, uses self.expl_noise

        Returns:
            action: 2-dim numpy array [steering, throttle/brake] ∈ [-1, 1]²
        """
        # Convert state to tensor
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # Get deterministic action from actor
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()

        # Add exploration noise if specified
        if noise is not None and noise > 0:
            noise_sample = np.random.normal(0, noise, size=self.action_dim)
            action = action + noise_sample
            # Clip to valid action range
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Perform one DDPG training iteration.

        Implements the complete DDPG algorithm:
        1. Sample mini-batch from replay buffer
        2. Compute target Q-value (unlike TD3, no target smoothing, no twin minimum)
        3. Update critic network to minimize TD error
        4. Update actor and target networks (EVERY step, unlike TD3's delayed updates)

        Args:
            batch_size: Size of mini-batch to sample. If None, uses self.batch_size

        Returns:
            Dictionary with training metrics:
                - critic_loss: TD error of critic network
                - actor_loss: Mean negative Q-value under current policy
                - q_value: Mean Q prediction
        """
        self.total_it += 1

        if batch_size is None:
            batch_size = self.batch_size

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            # For DDPG: Use target actor WITHOUT noise smoothing (unlike TD3)
            # This is the key difference: direct target action from target actor
            next_action = self.actor_target(next_state)

            # IMPORTANT: DDPG uses SINGLE Q-network (not twin minimum)
            # Compute target Q-value: y = r + γ * Q_θ'(s', μ_φ'(s'))
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimate from critic
        current_Q = self.critic(state, action)

        # Compute critic loss (MSE)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # DDPG: Update actor and target networks EVERY step (policy_freq=1)
        # Unlike TD3 which only updates every policy_freq=2 steps
        # Compute actor loss: -Q(s, μ_φ(s))
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks: θ' ← τθ + (1-τ)θ'
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Prepare metrics
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_Q.mean().item()
        }

        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint to disk.

        Saves actor, critic networks and their optimizers in a single file.

        Args:
            filepath: Path to save checkpoint (e.g., 'checkpoints/ddpg_100k.pth')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'total_it': self.total_it,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent checkpoint from disk.

        Restores networks, optimizers, and training state. Also recreates
        target networks from loaded weights.

        Args:
            filepath: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Recreate target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Restore optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Restore training state
        self.total_it = checkpoint['total_it']

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Resumed at iteration: {self.total_it}")

    def get_stats(self) -> Dict[str, any]:
        """
        Get current agent statistics.

        Returns:
            Dictionary with agent state information
        """
        return {
            'total_iterations': self.total_it,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_full': self.replay_buffer.is_full(),
            'device': str(self.device)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing DDPGAgent...")
    print("\n" + "="*60)
    print("DDPG BASELINE AGENT FOR COMPARISON WITH TD3")
    print("="*60)

    # Initialize agent (will load config from file)
    agent = DDPGAgent(
        state_dim=535,
        action_dim=2,
        max_action=1.0
    )

    print("\nAgent stats:", agent.get_stats())

    # Test action selection
    dummy_state = np.random.randn(535)
    action = agent.select_action(dummy_state, noise=0.1)
    print(f"\nSelected action: {action}")
    print(f"  Steering: {action[0]:.3f}")
    print(f"  Throttle/Brake: {action[1]:.3f}")

    # Add some transitions to replay buffer
    print("\nFilling replay buffer...")
    for i in range(1000):
        state = np.random.randn(535)
        action = np.random.randn(2)
        next_state = np.random.randn(535)
        reward = np.random.randn()
        done = (i % 100 == 0)

        agent.replay_buffer.add(state, action, next_state, reward, done)

    print(f"Buffer size: {len(agent.replay_buffer)}")

    # Test training
    print("\nPerforming 5 training steps...")
    for step in range(5):
        metrics = agent.train(batch_size=32)
        print(f"Step {step+1}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    # Test checkpoint save/load
    checkpoint_path = "/tmp/test_ddpg_checkpoint.pth"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    agent.save_checkpoint(checkpoint_path)

    print("Loading checkpoint...")
    agent.load_checkpoint(checkpoint_path)

    print("\n✓ DDPGAgent tests passed!")
    print("\n" + "="*60)
    print("KEY DIFFERENCES FROM TD3:")
    print("="*60)
    print("1. Single Critic instead of Twin Critics")
    print("2. No target policy smoothing (immediate target actions)")
    print("3. Policy updated EVERY step (not delayed)")
    print("4. Same hyperparameters for FAIR COMPARISON")
    print("="*60)

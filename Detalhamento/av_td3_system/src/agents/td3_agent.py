"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

Implements the TD3 algorithm for continuous control in autonomous driving.
TD3 improves upon DDPG with three key mechanisms:
1. Twin Critics: Uses two Q-networks and takes minimum for target value
2. Delayed Policy Updates: Updates actor less frequently than critics
3. Target Policy Smoothing: Adds noise to target actions for regularization

Reference: "Addressing Function Approximation Error in Actor-Critic Methods"
           (Fujimoto et al., ICML 2018)

Author: Daniel Terra
Date: 2024
"""

import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.networks.actor import Actor
from src.networks.critic import TwinCritic
from src.utils.replay_buffer import ReplayBuffer


class TD3Agent:
    """
    TD3 agent for autonomous vehicle control.
    
    The agent learns a deterministic policy that maps states (visual features +
    kinematic data + waypoints) to continuous actions (steering + throttle/brake).
    
    Key Attributes:
        actor: Policy network μ_φ(s)
        actor_target: Target policy for stable learning
        critic: Twin Q-networks Q_θ1(s,a) and Q_θ2(s,a)
        critic_target: Target Q-networks for stable learning
        replay_buffer: Experience replay buffer
        total_it: Total training iterations (for delayed policy updates)
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
        Initialize TD3 agent with networks and hyperparameters.

        Args:
            state_dim: Dimension of state space (default: 535)
                      = 512 (CNN features) + 3 (kinematic) + 20 (waypoints)
            action_dim: Dimension of action space (default: 2)
                       = [steering, throttle/brake]
            max_action: Maximum absolute value of actions (default: 1.0)
            config: Dictionary with TD3 hyperparameters (if None, loads from file)
            config_path: Path to YAML config file (default: config/td3_config.yaml)
        """
        # Load configuration
        if config is None:
            if config_path is None:
                config_path = "config/td3_config.yaml"
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
        self.policy_noise = algo_config['policy_noise']  # Noise for target smoothing
        self.noise_clip = algo_config['noise_clip']  # Clip range for target noise
        self.policy_freq = algo_config['policy_freq']  # Delayed policy update frequency
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
        print(f"TD3Agent initialized on device: {self.device}")

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

        # Initialize twin critic networks
        critic_config = config['networks']['critic']
        self.critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=critic_config['hidden_sizes']
        ).to(self.device)
        
        # Create target critics as deep copy
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

        # Training iteration counter for delayed updates
        self.total_it = 0

        print(f"TD3Agent initialized with:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Actor hidden sizes: {network_config['hidden_sizes']}")
        print(f"  Critic hidden sizes: {critic_config['hidden_sizes']}")
        print(f"  Discount γ: {self.discount}, Tau τ: {self.tau}")
        print(f"  Policy freq: {self.policy_freq}, Policy noise: {self.policy_noise}")
        print(f"  Exploration noise: {self.expl_noise}")
        print(f"  Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")

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
        Perform one TD3 training iteration.

        Implements the complete TD3 algorithm:
        1. Sample mini-batch from replay buffer
        2. Compute target Q-value with twin minimum and target smoothing
        3. Update both critic networks to minimize TD error
        4. (Every policy_freq steps) Update actor and target networks

        Args:
            batch_size: Size of mini-batch to sample. If None, uses self.batch_size

        Returns:
            Dictionary with training metrics:
                - critic_loss: Mean TD error of both critics
                - actor_loss: Mean Q-value under current policy (if policy updated)
                - q1_value: Mean Q1 prediction
                - q2_value: Mean Q2 prediction
        """
        self.total_it += 1
        
        if batch_size is None:
            batch_size = self.batch_size

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to target policy with added smoothing noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value: y = r + γ * min_i Q_θ'i(s', μ_φ'(s'))
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss (MSE on both Q-networks)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Prepare metrics
        metrics = {
            'critic_loss': critic_loss.item(),
            'q1_value': current_Q1.mean().item(),
            'q2_value': current_Q2.mean().item()
        }

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss: -Q1(s, μ_φ(s))
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

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

            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint to disk.

        Saves actor, critic networks and their optimizers in a single file.

        Args:
            filepath: Path to save checkpoint (e.g., 'checkpoints/td3_100k.pth')
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
    print("Testing TD3Agent...")
    
    # Initialize agent (will load config from file)
    agent = TD3Agent(
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
    print("\nPerforming training step...")
    metrics = agent.train(batch_size=32)
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test checkpoint save/load
    checkpoint_path = "/tmp/test_td3_checkpoint.pth"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    agent.save_checkpoint(checkpoint_path)
    
    print("Loading checkpoint...")
    agent.load_checkpoint(checkpoint_path)
    
    print("\n✓ TD3Agent tests passed!")

"""
Proximal Policy Optimization (PPO) Implementation for CARLA

This module provides a high-performance PPO implementation optimized for
autonomous driving in CARLA. It includes multimodal observations (images + vectors),
curriculum learning support, and comprehensive logging.

Based on the research papers showing excellent results with PPO in CARLA environments.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from networks import PolicyNetwork, ValueNetwork, FeatureExtractor
    from environment_wrapper import CarlaROS2Environment
    logger.info("Successfully imported local modules")
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error("Ensure all required modules are available")


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    # Algorithm parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    # Training parameters
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    eval_episodes: int = 5
    save_freq: int = 25000
    log_interval: int = 1
    
    # Device and performance
    device: str = "auto"
    n_envs: int = 1
    
    # Paths
    tensorboard_log: str = "./logs/tensorboard"
    checkpoint_dir: str = "./checkpoints"
    model_save_path: str = "./models"


class PPOBuffer:
    """Experience buffer for PPO algorithm."""
    
    def __init__(self,
                 n_steps: int,
                 n_envs: int,
                 obs_space: Dict,
                 action_space: torch.Size,
                 device: torch.device,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99):
        """Initialize PPO buffer.
        
        Args:
            n_steps: Number of steps per rollout
            n_envs: Number of parallel environments
            obs_space: Observation space specification
            action_space: Action space shape
            device: Torch device
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
        """
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        # Buffer storage
        self.observations = {}
        for key, space in obs_space.items():
            self.observations[key] = torch.zeros(
                (n_steps, n_envs) + space,
                dtype=torch.float32,
                device=device
            )
        
        self.actions = torch.zeros(
            (n_steps, n_envs) + action_space,
            dtype=torch.float32,
            device=device
        )
        self.log_probs = torch.zeros(
            n_steps, n_envs,
            dtype=torch.float32,
            device=device
        )
        self.values = torch.zeros(
            n_steps, n_envs,
            dtype=torch.float32,
            device=device
        )
        self.rewards = torch.zeros(
            n_steps, n_envs,
            dtype=torch.float32,
            device=device
        )
        self.dones = torch.zeros(
            n_steps, n_envs,
            dtype=torch.bool,
            device=device
        )
        
        # GAE computation
        self.advantages = torch.zeros(
            n_steps, n_envs,
            dtype=torch.float32,
            device=device
        )
        self.returns = torch.zeros(
            n_steps, n_envs,
            dtype=torch.float32,
            device=device
        )
        
        self.pos = 0
        self.full = False
    
    def add(self,
            obs: Dict[str, torch.Tensor],
            action: torch.Tensor,
            log_prob: torch.Tensor,
            value: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor):
        """Add experience to buffer.
        
        Args:
            obs: Observation dictionary
            action: Action taken
            log_prob: Log probability of action
            value: Value estimate
            reward: Reward received
            done: Episode done flag
        """
        for key, obs_tensor in obs.items():
            self.observations[key][self.pos] = obs_tensor.clone()
        
        self.actions[self.pos] = action.clone()
        self.log_probs[self.pos] = log_prob.clone()
        self.values[self.pos] = value.clone()
        self.rewards[self.pos] = reward.clone()
        self.dones[self.pos] = done.clone()
        
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True
    
    def compute_returns_and_advantages(self, last_values: torch.Tensor):
        """Compute returns and advantages using GAE.
        
        Args:
            last_values: Value estimates for final observations
        """
        if not self.full:
            raise ValueError("Buffer must be full before computing returns")
        
        # Compute advantages using GAE
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step] +
                self.gamma * next_values * next_non_terminal -
                self.values[step]
            )
            
            self.advantages[step] = last_gae_lam = (
                delta +
                self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        
        # Compute returns
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: int):
        """Get batched experiences for training.
        
        Args:
            batch_size: Size of training batches
            
        Yields:
            Batched experience tuples
        """
        if not self.full:
            raise ValueError("Buffer must be full before sampling")
        
        # Flatten buffer data
        batch_obs = {}
        for key, obs_tensor in self.observations.items():
            batch_obs[key] = obs_tensor.reshape((-1,) + obs_tensor.shape[2:])
        
        batch_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        batch_log_probs = self.log_probs.reshape(-1)
        batch_values = self.values.reshape(-1)
        batch_advantages = self.advantages.reshape(-1)
        batch_returns = self.returns.reshape(-1)
        
        # Create random indices for batching
        total_samples = self.n_steps * self.n_envs
        indices = torch.randperm(total_samples, device=self.device)
        
        # Yield batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'observations': {key: obs[batch_indices] for key, obs in batch_obs.items()},
                'actions': batch_actions[batch_indices],
                'old_log_probs': batch_log_probs[batch_indices],
                'old_values': batch_values[batch_indices],
                'advantages': batch_advantages[batch_indices],
                'returns': batch_returns[batch_indices]
            }
    
    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False


class PPOAgent:
    """PPO agent for CARLA autonomous driving."""
    
    def __init__(self,
                 config: PPOConfig,
                 obs_space: Dict,
                 action_space: Tuple[int],
                 device: Optional[torch.device] = None):
        """Initialize PPO agent.
        
        Args:
            config: PPO configuration
            obs_space: Observation space specification
            action_space: Action space shape
            device: Torch device
        """
        self.config = config
        self.obs_space = obs_space
        self.action_space = action_space
        
        # Set device
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = device
        
        logger.info(f"PPO agent using device: {self.device}")
        
        # Initialize networks
        self._initialize_networks()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=config.learning_rate
        )
        
        # Initialize buffer
        self.buffer = PPOBuffer(
            n_steps=config.n_steps,
            n_envs=config.n_envs,
            obs_space=obs_space,
            action_space=action_space,
            device=self.device,
            gae_lambda=config.gae_lambda,
            gamma=config.gamma
        )
        
        # Training tracking
        self.n_updates = 0
        self.total_timesteps = 0
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("PPO agent initialized successfully")
    
    def _initialize_networks(self):
        """Initialize policy and value networks."""
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            image_shape=self.obs_space['image'],
            vector_dim=self.obs_space['vector'][0]
        ).to(self.device)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_obs = {
                'image': torch.zeros((1,) + self.obs_space['image'], device=self.device),
                'vector': torch.zeros((1,) + self.obs_space['vector'], device=self.device)
            }
            feature_dim = self.feature_extractor(dummy_obs).shape[1]
        
        # Policy network
        self.policy_net = PolicyNetwork(
            feature_dim=feature_dim,
            action_dim=self.action_space[0]
        ).to(self.device)
        
        # Value network
        self.value_net = ValueNetwork(
            feature_dim=feature_dim
        ).to(self.device)
        
        logger.info(f"Networks initialized - Feature dim: {feature_dim}, "
                   f"Action dim: {self.action_space[0]}")
    
    def get_action(self,
                   obs: Dict[str, torch.Tensor],
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy.
        
        Args:
            obs: Observation dictionary
            deterministic: Whether to use deterministic action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            # Extract features
            features = self.feature_extractor(obs)
            
            # Get action distribution
            action_dist = self.policy_net(features)
            
            # Sample action
            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.sample()
            
            # Compute log probability
            log_prob = action_dist.log_prob(action).sum(-1)
            
            # Get value estimate
            value = self.value_net(features).squeeze(-1)
            
            return action, log_prob, value
    
    def learn(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using PPO.
        
        Args:
            rollout_data: Batch of rollout data
            
        Returns:
            Training statistics
        """
        # Extract data
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['old_log_probs']
        old_values = rollout_data['old_values']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        
        # Normalize advantages
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Extract features
        features = self.feature_extractor(observations)
        
        # Compute current policy
        action_dist = self.policy_net(features)
        current_log_probs = action_dist.log_prob(actions).sum(-1)
        
        # Compute current values
        current_values = self.value_net(features).squeeze(-1)
        
        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        if self.config.clip_range_vf is not None:
            # Clipped value loss
            value_pred_clipped = old_values + torch.clamp(
                current_values - old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf
            )
            value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
            value_loss_unclipped = F.mse_loss(current_values, returns)
            value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        else:
            value_loss = F.mse_loss(current_values, returns)
        
        # Compute entropy loss
        entropy_loss = -action_dist.entropy().mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.config.vf_coef * value_loss +
            self.config.ent_coef * entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Update counters
        self.n_updates += 1
        
        # Compute statistics
        with torch.no_grad():
            approx_kl = (old_log_probs - current_log_probs).mean()
            clip_fraction = ((ratio - 1).abs() > self.config.clip_range).float().mean()
            explained_variance = 1 - (returns - current_values).var() / returns.var()
        
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_fraction.item(),
            'explained_variance': explained_variance.item()
        }
        
        return stats
    
    def train_step(self, env) -> Dict[str, Any]:
        """Execute one training step (rollout + update).
        
        Args:
            env: Environment instance
            
        Returns:
            Training statistics
        """
        # Rollout phase
        rollout_stats = self._collect_rollout(env)
        
        # Training phase
        all_training_stats = []
        
        for epoch in range(self.config.n_epochs):
            epoch_stats = []
            
            for batch_data in self.buffer.get(self.config.batch_size):
                batch_stats = self.learn(batch_data)
                epoch_stats.append(batch_stats)
                
                # Early stopping based on KL divergence
                if (self.config.target_kl is not None and
                    batch_stats['approx_kl'] > self.config.target_kl):
                    logger.debug(f"Early stopping at epoch {epoch} due to KL divergence")
                    break
            
            if epoch_stats:
                # Average stats for this epoch
                avg_epoch_stats = {
                    key: np.mean([stats[key] for stats in epoch_stats])
                    for key in epoch_stats[0].keys()
                }
                all_training_stats.append(avg_epoch_stats)
        
        # Average training stats across epochs
        if all_training_stats:
            avg_training_stats = {
                key: np.mean([stats[key] for stats in all_training_stats])
                for key in all_training_stats[0].keys()
            }
        else:
            avg_training_stats = {}
        
        # Reset buffer for next rollout
        self.buffer.reset()
        
        # Combine rollout and training stats
        combined_stats = {**rollout_stats, **avg_training_stats}
        
        return combined_stats
    
    def _collect_rollout(self, env) -> Dict[str, Any]:
        """Collect rollout experience.
        
        Args:
            env: Environment instance
            
        Returns:
            Rollout statistics
        """
        episode_rewards = []
        episode_lengths = []
        
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(self.config.n_steps):
            # Convert observations to tensors
            obs_tensor = {
                key: torch.from_numpy(value).float().unsqueeze(0).to(self.device)
                for key, value in obs.items()
            }
            
            # Get action
            action, log_prob, value = self.get_action(obs_tensor)
            
            # Step environment
            next_obs, reward, done, info = env.step(action.cpu().numpy().squeeze(0))
            
            # Store experience
            self.buffer.add(
                obs=obs_tensor,
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                value=value.squeeze(0),
                reward=torch.tensor(reward, dtype=torch.float32, device=self.device),
                done=torch.tensor(done, dtype=torch.bool, device=self.device)
            )
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1
            
            # Handle episode termination
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
        
        # Get final value estimate for last observation
        obs_tensor = {
            key: torch.from_numpy(value).float().unsqueeze(0).to(self.device)
            for key, value in obs.items()
        }
        
        with torch.no_grad():
            features = self.feature_extractor(obs_tensor)
            last_values = self.value_net(features).squeeze()
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_values)
        
        # Rollout statistics
        stats = {
            'rollout/ep_rew_mean': np.mean(episode_rewards) if episode_rewards else 0.0,
            'rollout/ep_len_mean': np.mean(episode_lengths) if episode_lengths else 0.0,
            'rollout/n_episodes': len(episode_rewards),
            'rollout/total_timesteps': self.total_timesteps
        }
        
        return stats
    
    def save(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_updates': self.n_updates,
            'total_timesteps': self.total_timesteps,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.n_updates = checkpoint['n_updates']
        self.total_timesteps = checkpoint['total_timesteps']
        
        logger.info(f"Model loaded from {path}")


def create_ppo_agent(config_path: str, obs_space: Dict, action_space: Tuple[int]) -> PPOAgent:
    """Create PPO agent from configuration file.
    
    Args:
        config_path: Path to configuration file
        obs_space: Observation space specification
        action_space: Action space shape
        
    Returns:
        PPO agent instance
    """
    # Load configuration (implement config loading)
    config = PPOConfig()  # Use default for now
    
    # Create agent
    agent = PPOAgent(config, obs_space, action_space)
    
    return agent

"""
Enhanced Soft Actor-Critic (SAC) Implementation for CARLA

This module provides an implementation of the Soft Actor-Critic (SAC) algorithm
with support for multimodal observations from CARLA (images + vector state).
It includes several optimizations for autonomous driving tasks.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import feature extractors
try:
    from feature_extractors import CNNFeatureExtractor, MultimodalFeatureExtractor
    logger.info("Successfully imported feature extractors")
except ImportError as e:
    logger.error(f"Failed to import feature extractors: {e}")
    logger.error("Make sure feature_extractors.py is in the same directory")
    sys.exit(1)


@dataclass
class SACConfig:
    """Configuration for SAC algorithm."""
    # General
    random_seed: int = 42
    device: str = "auto"  # "auto", "cuda", or "cpu"
    checkpoint_dir: str = "./checkpoints/sac_carla"
    log_dir: str = "./logs/sac_carla"
    plot_dir: str = "./plots"

    # Observation and action
    image_observation: bool = True  # Whether observations include images
    vector_dim: int = 10  # Dimension of vector observations
    image_channels: int = 3  # Number of image channels
    image_height: int = 84  # Height of image observations
    image_width: int = 84  # Width of image observations
    action_dim: int = 2  # Dimension of action space

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    alpha: float = 0.2  # Temperature parameter for exploration
    auto_alpha_tuning: bool = True  # Whether to tune alpha automatically
    buffer_size: int = 100000  # Replay buffer size
    min_buffer_size: int = 1000  # Minimum replay buffer size before training
    reward_scale: float = 1.0  # Reward scaling factor

    # Neural Network
    feature_dim: int = 256  # Feature dimension
    hidden_dim: int = 256  # Hidden layer dimension
    log_std_min: float = -20  # Minimum log standard deviation
    log_std_max: float = 2  # Maximum log standard deviation

    # Evaluation and saving
    eval_episodes: int = 5  # Number of episodes for evaluation
    save_interval: int = 10  # Save checkpoint every n episodes

    # Preprocessing
    image_normalization: bool = True  # Whether to normalize images
    vector_normalization: bool = True  # Whether to normalize vector observations


class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self,
                 buffer_size: int,
                 image_observation: bool = True,
                 image_shape: Tuple[int, int, int] = (3, 84, 84),
                 vector_dim: int = 10,
                 action_dim: int = 2,
                 device: torch.device = torch.device("cpu")):
        """Initialize replay buffer.

        Args:
            buffer_size: Maximum size of buffer
            image_observation: Whether observations include images
            image_shape: Shape of image observations (C, H, W)
            vector_dim: Dimension of vector observations
            action_dim: Dimension of action space
            device: Device to store tensors
        """
        self.buffer_size = buffer_size
        self.image_observation = image_observation
        self.image_shape = image_shape
        self.vector_dim = vector_dim
        self.action_dim = action_dim
        self.device = device

        # Initialize buffer (different structure depending on observation type)
        if image_observation:
            self.images = np.zeros((buffer_size, *image_shape), dtype=np.uint8)
            self.next_images = np.zeros((buffer_size, *image_shape), dtype=np.uint8)
            self.vectors = np.zeros((buffer_size, vector_dim), dtype=np.float32)
            self.next_vectors = np.zeros((buffer_size, vector_dim), dtype=np.float32)
        else:
            self.observations = np.zeros((buffer_size, vector_dim), dtype=np.float32)
            self.next_observations = np.zeros((buffer_size, vector_dim), dtype=np.float32)

        # Common buffer elements
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

        # Buffer management
        self.ptr = 0
        self.size = 0

    def add(self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            action: np.ndarray,
            reward: float,
            next_observation: Union[np.ndarray, Dict[str, np.ndarray]],
            done: bool):
        """Add experience to buffer.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
        """
        if self.image_observation:
            # Handle dictionary observations with 'image' and 'vector' keys
            self.images[self.ptr] = observation['image']
            self.vectors[self.ptr] = observation['vector']
            self.next_images[self.ptr] = next_observation['image']
            self.next_vectors[self.ptr] = next_observation['vector']
        else:
            # Handle vector-only observations
            self.observations[self.ptr] = observation
            self.next_observations[self.ptr] = next_observation

        # Store action, reward, done
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def __len__(self) -> int:
        """Return the current size of the buffer.

        Returns:
            Current size of the buffer
        """
        return self.size

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        if self.image_observation:
            # Handle image observations
            observations = {
                'image': torch.as_tensor(self.images[idxs], device=self.device).float() / 255.0,
                'vector': torch.as_tensor(self.vectors[idxs], device=self.device)
            }

            next_observations = {
                'image': torch.as_tensor(self.next_images[idxs], device=self.device).float() / 255.0,
                'vector': torch.as_tensor(self.next_vectors[idxs], device=self.device)
            }
        else:
            # Handle vector-only observations
            observations = torch.as_tensor(self.observations[idxs], device=self.device)
            next_observations = torch.as_tensor(self.next_observations[idxs], device=self.device)

        # Convert other elements to tensors
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return observations, actions, rewards, next_observations, dones


class ImageObservationSACPolicy(nn.Module):
    """SAC policy network for image observations."""

    def __init__(self,
                 config: SACConfig,
                 device: torch.device):
        """Initialize policy network.

        Args:
            config: SAC configuration
            device: Device to run on
        """
        super().__init__()

        self.config = config
        self.device = device

        # Feature extractor for multimodal observations
        self.feature_extractor = MultimodalFeatureExtractor(
            image_channels=config.image_channels,
            image_height=config.image_height,
            image_width=config.image_width,
            vector_dim=config.vector_dim,
            image_feature_dim=config.feature_dim,
            combined_feature_dim=config.feature_dim
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )

        # Policy head for mean and log_std
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Initialize policy head
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observation: Dictionary with 'image' and 'vector' keys

        Returns:
            Tuple of (mean, log_std)
        """
        # Extract features from observation
        features = self.feature_extractor(observation)

        # Process features through policy network
        x = self.policy_net(features)

        # Get mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        # Constrain log_std
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)

        return mean, log_std

    def sample(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            observation: Dictionary with 'image' and 'vector' keys

        Returns:
            Tuple of (action, log_prob, tanh_mean)
        """
        mean, log_std = self.forward(observation)
        std = log_std.exp()

        # Sample from normal distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class VectorObservationSACPolicy(nn.Module):
    """SAC policy network for vector-only observations."""

    def __init__(self,
                 config: SACConfig,
                 device: torch.device):
        """Initialize policy network.

        Args:
            config: SAC configuration
            device: Device to run on
        """
        super().__init__()

        self.config = config
        self.device = device

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.vector_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )

        # Policy head for mean and log_std
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std_head = nn.Linear(config.hidden_dim, config.action_dim)

        # Initialize policy head
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observation: Vector observation

        Returns:
            Tuple of (mean, log_std)
        """
        # Process observation through policy network
        x = self.policy_net(observation)

        # Get mean and log_std
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        # Constrain log_std
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)

        return mean, log_std

    def sample(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            observation: Vector observation

        Returns:
            Tuple of (action, log_prob, tanh_mean)
        """
        mean, log_std = self.forward(observation)
        std = log_std.exp()

        # Sample from normal distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class ImageObservationSACCritic(nn.Module):
    """SAC critic network for image observations."""

    def __init__(self,
                 config: SACConfig,
                 device: torch.device):
        """Initialize critic network.

        Args:
            config: SAC configuration
            device: Device to run on
        """
        super().__init__()

        self.config = config
        self.device = device

        # Feature extractor for multimodal observations
        self.feature_extractor = MultimodalFeatureExtractor(
            image_channels=config.image_channels,
            image_height=config.image_height,
            image_width=config.image_width,
            vector_dim=config.vector_dim,
            image_feature_dim=config.feature_dim,
            combined_feature_dim=config.feature_dim
        )

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(config.feature_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(config.feature_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self,
                observation: Dict[str, torch.Tensor],
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observation: Dictionary with 'image' and 'vector' keys
            action: Action tensor

        Returns:
            Tuple of (q1_value, q2_value)
        """
        # Extract features from observation
        features = self.feature_extractor(observation)

        # Concatenate features with action
        x = torch.cat([features, action], dim=1)

        # Compute Q-values
        q1_value = self.q1(x)
        q2_value = self.q2(x)

        return q1_value, q2_value


class VectorObservationSACCritic(nn.Module):
    """SAC critic network for vector-only observations."""

    def __init__(self,
                 config: SACConfig,
                 device: torch.device):
        """Initialize critic network.

        Args:
            config: SAC configuration
            device: Device to run on
        """
        super().__init__()

        self.config = config
        self.device = device

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(config.vector_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(config.vector_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self,
                observation: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            observation: Vector observation
            action: Action tensor

        Returns:
            Tuple of (q1_value, q2_value)
        """
        # Concatenate observation with action
        x = torch.cat([observation, action], dim=1)

        # Compute Q-values
        q1_value = self.q1(x)
        q2_value = self.q2(x)

        return q1_value, q2_value


class EnhancedSAC:
    """Enhanced Soft Actor-Critic (SAC) algorithm."""

    def __init__(self, config: SACConfig):
        """Initialize SAC algorithm.

        Args:
            config: SAC configuration
        """
        self.config = config

        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Set random seed
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Initialize networks based on observation type
        if config.image_observation:
            self.actor = ImageObservationSACPolicy(config, self.device).to(self.device)
            self.critic = ImageObservationSACCritic(config, self.device).to(self.device)
            self.target_critic = ImageObservationSACCritic(config, self.device).to(self.device)
        else:
            self.actor = VectorObservationSACPolicy(config, self.device).to(self.device)
            self.critic = VectorObservationSACCritic(config, self.device).to(self.device)
            self.target_critic = VectorObservationSACCritic(config, self.device).to(self.device)

        # Hard copy initial weights to target network
        self.update_target_network(tau=1.0)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        # Initialize temperature parameter alpha
        self.log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)

        # Target entropy is negative of action dimension
        self.target_entropy = -config.action_dim

        # Initialize replay buffer
        buffer_image_shape = (config.image_channels, config.image_height, config.image_width)
        self.replay_buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            image_observation=config.image_observation,
            image_shape=buffer_image_shape if config.image_observation else None,
            vector_dim=config.vector_dim,
            action_dim=config.action_dim,
            device=self.device
        )

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.plot_dir, exist_ok=True)

        # Training metrics
        self.episode_rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []
        self.alphas = []
        self.eval_rewards = []

    def update_target_network(self, tau: float):
        """Update target network using polyak averaging.

        Args:
            tau: Polyak averaging coefficient (1 for hard update)
        """
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self,
                      observation: Union[Dict[str, np.ndarray], np.ndarray],
                      evaluate: bool = False) -> np.ndarray:
        """Select action based on observation.

        Args:
            observation: Observation from environment
            evaluate: Whether to evaluate (use mean) or explore

        Returns:
            Selected action
        """
        with torch.no_grad():
            if self.config.image_observation:
                # Process dictionary observation
                obs_dict = {
                    'image': torch.as_tensor(observation['image'], dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0,
                    'vector': torch.as_tensor(observation['vector'], dtype=torch.float32, device=self.device).unsqueeze(0)
                }

                if evaluate:
                    _, _, action = self.actor.sample(obs_dict)
                else:
                    action, _, _ = self.actor.sample(obs_dict)
            else:
                # Process vector observation
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                if evaluate:
                    _, _, action = self.actor.sample(obs_tensor)
                else:
                    action, _, _ = self.actor.sample(obs_tensor)

        return action.cpu().numpy()[0]

    def update_parameters(self, batch_size: int) -> Dict[str, float]:
        """Update model parameters using a batch of experiences.

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of loss metrics
        """
        # Sample a batch from replay buffer
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(batch_size)

        # Scale rewards
        rewards = rewards * self.config.reward_scale

        # Get current temperature parameter
        alpha = self.log_alpha.exp()

        # Update critic
        with torch.no_grad():
            # Sample next actions and compute their log probabilities
            next_actions, next_log_probs, _ = self.actor.sample(next_observations)

            # Compute target Q-values
            next_q1, next_q2 = self.target_critic(next_observations, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # Compute current Q-values
        current_q1, current_q2 = self.critic(observations, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pi, log_probs, _ = self.actor.sample(observations)
        q1_pi, q2_pi = self.critic(observations, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature parameter)
        if self.config.auto_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # Update target network
        self.update_target_network(self.config.tau)

        # Record metrics
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        self.alpha_losses.append(alpha_loss.item())
        self.alphas.append(alpha.item())

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }

    def save_checkpoint(self, episode: int):
        """Save model checkpoint.

        Args:
            episode: Current episode number
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"sac_episode_{episode}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save models
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, "critic.pt"))
        torch.save(self.target_critic.state_dict(), os.path.join(checkpoint_path, "target_critic.pt"))
        torch.save(self.log_alpha, os.path.join(checkpoint_path, "log_alpha.pt"))

        # Save optimizers
        torch.save(self.actor_optimizer.state_dict(), os.path.join(checkpoint_path, "actor_optimizer.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(checkpoint_path, "critic_optimizer.pt"))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(checkpoint_path, "alpha_optimizer.pt"))

        # Save training metrics
        np.save(os.path.join(checkpoint_path, "episode_rewards.npy"), np.array(self.episode_rewards))
        np.save(os.path.join(checkpoint_path, "critic_losses.npy"), np.array(self.critic_losses))
        np.save(os.path.join(checkpoint_path, "actor_losses.npy"), np.array(self.actor_losses))
        np.save(os.path.join(checkpoint_path, "alpha_losses.npy"), np.array(self.alpha_losses))
        np.save(os.path.join(checkpoint_path, "alphas.npy"), np.array(self.alphas))

        # Save configuration
        with open(os.path.join(checkpoint_path, "config.txt"), 'w') as f:
            for key, value in self.config.__dict__.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load models
        self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, "actor.pt"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, "critic.pt"), map_location=self.device))
        self.target_critic.load_state_dict(torch.load(os.path.join(checkpoint_path, "target_critic.pt"), map_location=self.device))
        self.log_alpha = torch.load(os.path.join(checkpoint_path, "log_alpha.pt"), map_location=self.device)

        # Load optimizers
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "actor_optimizer.pt"), map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "critic_optimizer.pt"), map_location=self.device))
        self.alpha_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "alpha_optimizer.pt"), map_location=self.device))

        # Load training metrics if available
        try:
            self.episode_rewards = np.load(os.path.join(checkpoint_path, "episode_rewards.npy")).tolist()
            self.critic_losses = np.load(os.path.join(checkpoint_path, "critic_losses.npy")).tolist()
            self.actor_losses = np.load(os.path.join(checkpoint_path, "actor_losses.npy")).tolist()
            self.alpha_losses = np.load(os.path.join(checkpoint_path, "alpha_losses.npy")).tolist()
            self.alphas = np.load(os.path.join(checkpoint_path, "alphas.npy")).tolist()
        except FileNotFoundError:
            logger.warning("Could not load all training metrics")

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def plot_training_metrics(self):
        """Plot training metrics."""
        if len(self.episode_rewards) == 0:
            logger.warning("No training metrics to plot")
            return

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # Plot critic and actor losses
        if len(self.critic_losses) > 0:
            axes[0, 1].plot(self.critic_losses, label='Critic Loss')
            axes[0, 1].set_title('Critic Loss')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)

        if len(self.actor_losses) > 0:
            axes[1, 0].plot(self.actor_losses, label='Actor Loss')
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)

        # Plot alpha
        if len(self.alphas) > 0:
            axes[1, 1].plot(self.alphas, label='Alpha')
            axes[1, 1].set_title('Temperature Parameter (Alpha)')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Alpha')
            axes[1, 1].grid(True)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'training_metrics.png'))
        plt.close()

        # Also save individual metrics as CSV for further analysis
        metrics_path = os.path.join(self.config.plot_dir, 'metrics')
        os.makedirs(metrics_path, exist_ok=True)

        np.savetxt(os.path.join(metrics_path, 'episode_rewards.csv'), np.array(self.episode_rewards), delimiter=',')
        np.savetxt(os.path.join(metrics_path, 'critic_losses.csv'), np.array(self.critic_losses), delimiter=',')
        np.savetxt(os.path.join(metrics_path, 'actor_losses.csv'), np.array(self.actor_losses), delimiter=',')
        np.savetxt(os.path.join(metrics_path, 'alpha_losses.csv'), np.array(self.alpha_losses), delimiter=',')
        np.savetxt(os.path.join(metrics_path, 'alphas.csv'), np.array(self.alphas), delimiter=',')

        logger.info(f"Saved training metrics to {self.config.plot_dir}")


# Test code
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Enhanced SAC")
    parser.add_argument('--image', action='store_true', help='Use image observations')
    parser.add_argument('--no-auto-alpha', action='store_false', dest='auto_alpha', help='Disable automatic alpha tuning')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Create SAC configuration
    config = SACConfig(
        image_observation=args.image,
        auto_alpha_tuning=args.auto_alpha,
        device=args.device,
        batch_size=args.batch_size
    )

    # Create SAC agent
    agent = EnhancedSAC(config)

    # Test observation processing
    if args.image:
        observation = {
            'image': np.random.randint(0, 256, (3, 84, 84), dtype=np.uint8),
            'vector': np.random.randn(10).astype(np.float32)
        }
    else:
        observation = np.random.randn(10).astype(np.float32)

    # Test action selection
    action = agent.select_action(observation)
    print(f"Selected action: {action}")

    # Test replay buffer
    if args.image:
        agent.replay_buffer.add(
            observation=observation,
            action=np.array([0.5, -0.5], dtype=np.float32),
            reward=1.0,
            next_observation={
                'image': np.random.randint(0, 256, (3, 84, 84), dtype=np.uint8),
                'vector': np.random.randn(10).astype(np.float32)
            },
            done=False
        )
    else:
        agent.replay_buffer.add(
            observation=observation,
            action=np.array([0.5, -0.5], dtype=np.float32),
            reward=1.0,
            next_observation=np.random.randn(10).astype(np.float32),
            done=False
        )

    print("Test complete!")

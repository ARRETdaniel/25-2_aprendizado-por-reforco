#!/usr/bin/env python
"""
Robust SAC implementation for CARLA.

This module provides a simplified and robust implementation of SAC for use
with the CARLA environment, focusing on stability and ease of use.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from collections import deque
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Add project root to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_environment.environment import CarlaEnvWrapper

class SimpleFeatureExtractor(nn.Module):
    """
    A simplified feature extractor for processing state inputs.

    This class processes state inputs into a unified feature representation,
    designed to be more robust to shape differences.
    """

    def __init__(self, state_dims: Dict[str, Tuple[int, ...]], feature_dim: int = 128):
        """
        Initialize the feature extractor.

        Args:
            state_dims: Dictionary of state space dimensions
            feature_dim: Dimension of output feature vector
        """
        super(SimpleFeatureExtractor, self).__init__()

        # Set up for processing image
        if 'image' in state_dims:
            self.has_image = True
            img_h, img_w, img_c = state_dims['image']
            # Simple CNN for processing images
            self.image_net = nn.Sequential(
                nn.Conv2d(img_c, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )

            # Calculate CNN output size
            test_input = torch.zeros(1, img_c, img_h, img_w)
            with torch.no_grad():
                test_output = self.image_net(test_input)
                cnn_output_size = test_output.shape[1]
        else:
            self.has_image = False
            cnn_output_size = 0

        # Process other state components
        other_dims = 0
        for key, dims in state_dims.items():
            if key != 'image':
                other_dims += np.prod(dims).item()

        # Combined network to merge all features
        self.combined_net = nn.Sequential(
            nn.Linear(cnn_output_size + other_dims, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process state dictionary into a feature vector.

        Args:
            state: Dictionary of state tensors:
                - 'image': Image tensor [B, H, W, C] (if present)
                - Other state components as tensors

        Returns:
            Feature vector [B, feature_dim]
        """
        features = []

        # Process image if present
        if self.has_image and 'image' in state:
            # Handle batch dimension properly
            if state['image'].dim() == 4:  # [B, H, W, C]
                x_img = state['image'].permute(0, 3, 1, 2)  # -> [B, C, H, W]
            else:  # [H, W, C]
                x_img = state['image'].permute(2, 0, 1).unsqueeze(0)  # -> [1, C, H, W]

            x_img = self.image_net(x_img)
            features.append(x_img)

        # Process all other state components
        for key, value in state.items():
            if key != 'image':
                # Handle different dimensions
                if value.dim() == 1:  # [D]
                    x = value.unsqueeze(0)  # -> [1, D]
                elif value.dim() == 2:  # [B, D]
                    x = value
                else:
                    # Flatten any higher dimensional tensors
                    x = value.reshape(value.size(0), -1) if value.dim() > 2 else value

                features.append(x)

        # Combine all features
        try:
            combined = torch.cat(features, dim=1)
        except RuntimeError as e:
            logger.error(f"Error combining features: {e}")
            logger.error(f"Feature shapes: {[f.shape for f in features]}")
            raise

        return self.combined_net(combined)

class SimpleQNetwork(nn.Module):
    """
    A simplified Q-network for SAC.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super(SimpleQNetwork, self).__init__()

        self.q_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for given state-action pair.

        Args:
            feature: Feature vector [B, feature_dim]
            action: Action tensor [B, action_dim]

        Returns:
            Q-value [B, 1]
        """
        x = torch.cat([feature, action], dim=1)
        return self.q_net(x)

class SimplePolicy(nn.Module):
    """
    A simplified Gaussian policy network for SAC.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 128,
                action_low: float = -1.0, action_high: float = 1.0):
        """
        Initialize the policy network.

        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            action_low: Lower bound of action range
            action_high: Upper bound of action range
        """
        super(SimplePolicy, self).__init__()

        self.action_dim = action_dim
        self.action_scale = torch.tensor((action_high - action_low) / 2.0).to(device)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0).to(device)

        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_net = nn.Linear(hidden_dim, action_dim)
        self.log_std_net = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and log_std of action distribution.

        Args:
            feature: Feature vector [B, feature_dim]

        Returns:
            Tuple of (mean, log_std) of action distribution
        """
        x = self.policy_net(feature)

        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.

        Args:
            feature: Feature vector [B, feature_dim]

        Returns:
            Tuple of (action, log_prob, tanh_mean)
        """
        mean, log_std = self.forward(feature)
        std = log_std.exp()

        # Sample from normal distribution
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick

        # Compute log probability
        # Apply tanh squashing to constrain actions
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias

        # Compute log probability with change of variable formula
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # Also compute the tanh of the mean for evaluation
        tanh_mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, tanh_mean

class ReplayBuffer:
    """
    Replay buffer for SAC.
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum capacity of the buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: Dict[str, np.ndarray], action: np.ndarray,
            reward: float, next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                              torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        # Initialize dictionaries for state and next_state
        states = {}
        next_states = {}

        # First, determine state structure from the first sample
        first_state, _, _, first_next_state, _ = batch[0]

        # Initialize state and next_state dictionaries with lists
        for key in first_state.keys():
            states[key] = []
            next_states[key] = []

        # Collect data from all samples
        actions = []
        rewards = []
        dones = []

        for state, action, reward, next_state, done in batch:
            # Collect states and next_states
            for key in state.keys():
                states[key].append(state[key])
                next_states[key].append(next_state[key])

            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # Convert lists to tensors
        for key in states.keys():
            states[key] = torch.FloatTensor(np.array(states[key])).to(device)
            next_states[key] = torch.FloatTensor(np.array(next_states[key])).to(device)

        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)

    def can_sample(self, batch_size: int) -> bool:
        """Check if the buffer has enough experiences for sampling."""
        return len(self) >= batch_size

class SimpleSAC:
    """
    Simplified Soft Actor-Critic implementation.

    This class implements a simplified version of the SAC algorithm,
    designed to be more robust to edge cases and easier to debug.
    """

    def __init__(self,
                state_dims: Dict[str, Tuple[int, ...]],
                action_dim: int,
                action_low: float = -1.0,
                action_high: float = 1.0,
                lr: float = 3e-4,
                gamma: float = 0.99,
                tau: float = 0.005,
                alpha: float = 0.2,
                auto_entropy: bool = True,
                buffer_size: int = 100000,
                batch_size: int = 128,
                feature_dim: int = 128,
                hidden_dim: int = 128,
                update_freq: int = 2,
                seed: Optional[int] = None):
        """
        Initialize the SAC agent.

        Args:
            state_dims: Dictionary of state space dimensions
            action_dim: Dimension of action space
            action_low: Lower bound of action range
            action_high: Upper bound of action range
            lr: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter for entropy
            auto_entropy: Whether to auto-tune entropy parameter
            buffer_size: Size of replay buffer
            batch_size: Training batch size
            feature_dim: Dimension of feature vector
            hidden_dim: Dimension of hidden layers
            update_freq: Number of steps between network updates
            seed: Random seed for reproducibility
        """
        # Set random seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Store parameters
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.step_count = 0

        # Initialize networks
        self.feature_extractor = SimpleFeatureExtractor(state_dims, feature_dim).to(device)
        self.policy = SimplePolicy(feature_dim, action_dim, hidden_dim, action_low, action_high).to(device)
        self.q1 = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)
        self.q2 = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)
        self.q1_target = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)
        self.q2_target = SimpleQNetwork(feature_dim, action_dim, hidden_dim).to(device)

        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Freeze target networks
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False

        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Initialize automatic entropy tuning
        if auto_entropy:
            self.target_entropy = -action_dim  # -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []

        logger.info("SimpleSAC initialized")

    def select_action(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        """
        Select an action based on the current policy.

        Args:
            state: Current state dictionary
            evaluate: Whether to use deterministic action for evaluation

        Returns:
            Selected action as numpy array
        """
        # Ensure state values are float32
        state_tensors = {}
        for key, value in state.items():
            # Convert to float32 if needed
            if value.dtype != np.float32:
                value = value.astype(np.float32)
            # Add batch dimension if needed
            if value.ndim == len(np.array(value.shape)) - 1:
                value = np.expand_dims(value, 0)
            state_tensors[key] = torch.FloatTensor(value).to(device)

        try:
            with torch.no_grad():
                features = self.feature_extractor(state_tensors)

                if evaluate:
                    # Use mean for evaluation (deterministic)
                    mean, _ = self.policy(features)
                    action = torch.tanh(mean) * self.policy.action_scale + self.policy.action_bias
                else:
                    # Sample action for training (stochastic)
                    action, _, _ = self.policy.sample(features)

            return action.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            # Return a safe fallback action (zeros)
            return np.zeros(self.action_dim)

    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update the SAC networks.

        Args:
            batch_size: Batch size for update, defaults to self.batch_size

        Returns:
            Dictionary of loss values
        """
        if batch_size is None:
            batch_size = self.batch_size

        if not self.memory.can_sample(batch_size):
            return {'actor_loss': 0, 'critic_loss': 0, 'alpha_loss': 0}

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Extract features - make sure to detach to avoid backward issues
        features = self.feature_extractor(states)
        next_features = self.feature_extractor(next_states).detach()

        # Get current alpha value
        alpha = self.log_alpha.exp().item() if self.auto_entropy else self.alpha

        # Update critic
        with torch.no_grad():
            # Sample actions from policy
            next_actions, next_log_probs, _ = self.policy.sample(next_features)

            # Compute target Q values
            q1_next = self.q1_target(next_features, next_actions)
            q2_next = self.q2_target(next_features, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Compute current Q values
        q1 = self.q1(features, actions)
        q2 = self.q2(features, actions)

        # Compute critic loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss

        # Update critics
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Record critic loss
        self.critic_losses.append(critic_loss.item())

        # Get fresh features for actor update to avoid backward issue
        with torch.no_grad():
            features_actor = self.feature_extractor(states)

        # Update actor
        actions_pi, log_probs, _ = self.policy.sample(features_actor)
        q1_pi = self.q1(features_actor, actions_pi)
        q2_pi = self.q2(features_actor, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss is expectation of Q - entropy
        actor_loss = (alpha * log_probs - q_pi).mean()

        # Update actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # Record actor loss
        self.actor_losses.append(actor_loss.item())

        # Update alpha if using automatic entropy tuning
        alpha_loss = 0
        if self.auto_entropy:
            # Get fresh log_probs for alpha update to avoid backward issue
            with torch.no_grad():
                features_alpha = self.feature_extractor(states)

            _, log_probs_alpha, _ = self.policy.sample(features_alpha)
            alpha_loss = -(self.log_alpha * (log_probs_alpha + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Record alpha loss
            self.alpha_losses.append(alpha_loss.item())

        # Soft update target networks
        self._soft_update_targets()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item()
        }

    def _soft_update_targets(self) -> None:
        """Soft update of target networks."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def step(self, state: Dict[str, np.ndarray], action: np.ndarray,
            reward: float, next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        Process a step and update the agent if needed.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        # Store transition in replay buffer
        self.memory.push(state, action, reward, next_state, done)

        # Increment step counter
        self.step_count += 1

        # Update networks if it's time
        if self.step_count % self.update_freq == 0 and self.memory.can_sample(self.batch_size):
            self.update()

    def save(self, directory: str) -> None:
        """
        Save agent models.

        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)

        torch.save(self.feature_extractor.state_dict(), f"{directory}/feature_extractor.pt")
        torch.save(self.policy.state_dict(), f"{directory}/policy.pt")
        torch.save(self.q1.state_dict(), f"{directory}/q1.pt")
        torch.save(self.q2.state_dict(), f"{directory}/q2.pt")

        if self.auto_entropy:
            torch.save(self.log_alpha, f"{directory}/log_alpha.pt")

        logger.info(f"Models saved to {directory}")

    def load(self, directory: str) -> None:
        """
        Load agent models.

        Args:
            directory: Directory to load models from
        """
        self.feature_extractor.load_state_dict(torch.load(f"{directory}/feature_extractor.pt"))
        self.policy.load_state_dict(torch.load(f"{directory}/policy.pt"))
        self.q1.load_state_dict(torch.load(f"{directory}/q1.pt"))
        self.q2.load_state_dict(torch.load(f"{directory}/q2.pt"))
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        if self.auto_entropy and os.path.exists(f"{directory}/log_alpha.pt"):
            self.log_alpha = torch.load(f"{directory}/log_alpha.pt")

        logger.info(f"Models loaded from {directory}")

def train_simple_sac(env: CarlaEnvWrapper,
                    agent: SimpleSAC,
                    n_episodes: int = 1000,
                    max_steps: int = 1000,
                    checkpoint_dir: str = './checkpoints',
                    checkpoint_freq: int = 100,
                    eval_freq: int = 100,
                    eval_episodes: int = 5,
                    early_stop_reward: Optional[float] = None) -> Dict[str, List[float]]:
    """
    Train the SimpleSAC agent.

    Args:
        env: CARLA environment wrapper
        agent: SimpleSAC agent
        n_episodes: Maximum number of episodes
        max_steps: Maximum steps per episode
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency of checkpoints in episodes
        eval_freq: Frequency of evaluation in episodes
        eval_episodes: Number of episodes for evaluation
        early_stop_reward: Early stopping reward threshold

    Returns:
        Dictionary of training statistics
    """
    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'alpha_losses': []
    }

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    total_steps = 0

    logger.info(f"Starting training for {n_episodes} episodes, max {max_steps} steps each")

    for episode in range(1, n_episodes + 1):
        episode_reward = 0
        episode_steps = 0

        # Reset environment
        try:
            state = env.reset()
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            logger.info("Attempting to continue...")
            continue

        # Episode loop
        for step in range(1, max_steps + 1):
            # Select action
            action = agent.select_action(state)

            # Take step in environment
            try:
                next_state, reward, done, info = env.step(action)

                # Process step in agent
                agent.step(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if done:
                    break

            except Exception as e:
                logger.error(f"Error during episode step: {e}")
                break

        # Record episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_steps)

        # Append recent losses
        if agent.actor_losses:
            stats['actor_losses'].extend(agent.actor_losses)
            agent.actor_losses = []

        if agent.critic_losses:
            stats['critic_losses'].extend(agent.critic_losses)
            agent.critic_losses = []

        if agent.alpha_losses:
            stats['alpha_losses'].extend(agent.alpha_losses)
            agent.alpha_losses = []

        # Log episode results
        logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

        # Save checkpoint
        if episode % checkpoint_freq == 0:
            agent.save(f"{checkpoint_dir}/episode_{episode}")
            logger.info(f"Saved checkpoint at episode {episode}")

        # Evaluate agent
        if episode % eval_freq == 0:
            eval_reward = evaluate_agent(env, agent, eval_episodes)
            stats['eval_rewards'].append(eval_reward)
            logger.info(f"Evaluation at episode {episode}: Average Reward = {eval_reward:.2f}")

        # Check for early stopping
        if early_stop_reward is not None and episode_reward >= early_stop_reward:
            logger.info(f"Early stopping at episode {episode}: Reward {episode_reward} >= {early_stop_reward}")
            agent.save(f"{checkpoint_dir}/early_stop_episode_{episode}")
            break

    # Final save
    agent.save(f"{checkpoint_dir}/final")

    return stats

def evaluate_agent(env: CarlaEnvWrapper,
                 agent: SimpleSAC,
                 n_episodes: int = 5,
                 max_steps: int = 1000) -> float:
    """
    Evaluate the agent without exploration.

    Args:
        env: CARLA environment wrapper
        agent: SAC agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Average episode reward
    """
    total_reward = 0

    for episode in range(1, n_episodes + 1):
        episode_reward = 0

        try:
            state = env.reset()

            for step in range(1, max_steps + 1):
                # Select action without exploration
                action = agent.select_action(state, evaluate=True)

                # Take step in environment
                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward

        except Exception as e:
            logger.error(f"Error during evaluation episode {episode}: {e}")

    # Calculate average reward
    avg_reward = total_reward / n_episodes

    return avg_reward

def quick_test_sac(n_episodes=3, max_steps=100):
    """
    Run a quick test of the SimpleSAC implementation.

    Args:
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        True if the test was successful, False otherwise
    """
    try:
        # Initialize environment with minimal settings
        logger.info("Creating environment with minimal settings...")
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            city_name='Town01',
            image_size=(84, 84),
            frame_skip=1,
            max_episode_steps=max_steps,
            weather_id=0,
            quality_level='Low'
        )

        # Get state dimensions
        state = env.reset()
        state_dims = {key: value.shape for key, value in state.items()}
        action_dim = env.action_space.shape[0]

        logger.info(f"State dimensions: {state_dims}")
        logger.info(f"Action dimension: {action_dim}")

        # Initialize agent
        logger.info("Initializing SimpleSAC agent...")
        agent = SimpleSAC(
            state_dims=state_dims,
            action_dim=action_dim,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            buffer_size=10000,  # Smaller for testing
            batch_size=32,      # Smaller for testing
            feature_dim=64,     # Smaller for testing
            hidden_dim=64,      # Smaller for testing
            update_freq=10,
            seed=42
        )

        # Train for a few episodes
        logger.info(f"Starting test training for {n_episodes} episodes...")
        stats = train_simple_sac(
            env=env,
            agent=agent,
            n_episodes=n_episodes,
            max_steps=max_steps,
            checkpoint_dir='./test_checkpoints',
            checkpoint_freq=n_episodes,  # Only save at the end
            eval_freq=n_episodes,        # Only evaluate at the end
            eval_episodes=1
        )

        # Check if training produced meaningful results
        has_rewards = len(stats['episode_rewards']) > 0
        has_losses = len(stats['critic_losses']) > 0

        # Clean up
        env.close()

        if has_rewards and has_losses:
            logger.info("Test completed successfully!")
            return True
        else:
            logger.warning("Test completed but may have issues")
            return False

    except Exception as e:
        logger.error(f"Error during test: {e}")
        if 'env' in locals():
            env.close()
        return False

if __name__ == "__main__":
    # Create checkpoint directory
    os.makedirs('./test_checkpoints', exist_ok=True)

    # Run quick test
    quick_test_sac()

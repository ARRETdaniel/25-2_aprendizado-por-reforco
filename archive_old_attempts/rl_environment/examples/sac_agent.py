"""
Soft Actor-Critic (SAC) implementation for CARLA.

This module implements a Soft Actor-Critic agent for learning autonomous driving
behaviors in the CARLA simulator environment with continuous actions.

SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
reinforcement learning framework. It is particularly suitable for continuous
action spaces and complex environments like autonomous driving.
"""

import os
import sys
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Union, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SACAgent")

# Add parent directory to path to import rl_environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError as e:
    logger.error(f"Failed to import CarlaEnvWrapper: {e}")
    sys.exit(1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define Experience replay tuple structure
Experience = namedtuple('Experience', field_names=[
    'state', 'action', 'reward', 'next_state', 'done'
])

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling training transitions.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state: Dict[str, np.ndarray], action: np.ndarray,
            reward: float, next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        Add an experience to the buffer.

        Args:
            state: Current state dictionary
            action: Action taken
            reward: Reward received
            next_state: Next state dictionary
            done: Whether the episode terminated
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor],
                                           torch.Tensor,
                                           torch.Tensor,
                                           Dict[str, torch.Tensor],
                                           torch.Tensor]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Sample experiences
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Convert to batch of experiences
        # For state dictionaries, handle each key separately
        states = {}
        next_states = {}

        # Initialize state dictionaries with empty lists for each key
        for key in experiences[0].state.keys():
            states[key] = []
            next_states[key] = []

        # Extract values
        for exp in experiences:
            for key in exp.state.keys():
                states[key].append(exp.state[key])
                next_states[key].append(exp.next_state[key])

        # Convert lists to tensors for each key
        for key in states.keys():
            states[key] = torch.FloatTensor(np.array(states[key])).to(device)
            next_states[key] = torch.FloatTensor(np.array(next_states[key])).to(device)

        # Extract and convert other fields
        actions = torch.FloatTensor(np.array([exp.action for exp in experiences])).to(device)
        rewards = torch.FloatTensor(np.array([exp.reward for exp in experiences])).to(device)
        dones = torch.FloatTensor(np.array([exp.done for exp in experiences]).astype(np.uint8)).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough experiences for sampling."""
        return len(self) >= batch_size


class FeatureExtractor(nn.Module):
    """
    Feature extractor for processing complex state inputs into a unified representation.

    This module processes images using CNNs and other state components using
    fully connected layers, then combines them into a single feature vector.
    """

    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 vehicle_state_size: int = 9,
                 navigation_size: int = 3,
                 detections_size: int = 10,
                 feature_dim: int = 256):
        """
        Initialize the feature extractor.

        Args:
            image_shape: Shape of input images (H, W, C)
            vehicle_state_size: Size of vehicle state vector
            navigation_size: Size of navigation vector
            detections_size: Size of detections vector
            feature_dim: Dimension of output feature vector
        """
        super(FeatureExtractor, self).__init__()

        # Image processing
        self.conv1 = nn.Conv2d(image_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size after convolutions
        def conv_output_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        # Calculate output size after all convolutions
        h, w = image_shape[0], image_shape[1]
        h = conv_output_size(h, 8, 4)  # After conv1
        h = conv_output_size(h, 4, 2)  # After conv2
        h = conv_output_size(h, 3, 1)  # After conv3

        w = conv_output_size(w, 8, 4)  # After conv1
        w = conv_output_size(w, 4, 2)  # After conv2
        w = conv_output_size(w, 3, 1)  # After conv3

        conv_output_features = h * w * 64

        # Vector data processing
        self.fc_vehicle = nn.Linear(vehicle_state_size, 64)
        self.fc_nav = nn.Linear(navigation_size, 32)
        self.fc_detect = nn.Linear(detections_size, 32)

        # Combined processing
        combined_size = 64 + 64 + 32 + 32  # Conv features + vehicle + nav + detect
        self.fc_combined = nn.Linear(combined_size, feature_dim)

        # Save dimensions for forward pass
        self.conv_output_features = conv_output_features
        self.feature_dim = feature_dim

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process state inputs into a feature vector.

        Args:
            state: Dictionary containing:
                - 'image': Image tensor [B, H, W, C]
                - 'vehicle_state': Vehicle state tensor [B, vehicle_state_size]
                - 'navigation': Navigation tensor [B, navigation_size]
                - 'detections': Detections tensor [B, detections_size]

        Returns:
            Feature vector [B, feature_dim]
        """
        # Process image
        # Convert from [B, H, W, C] to [B, C, H, W]
        x_img = state['image'].permute(0, 3, 1, 2)
        x_img = F.relu(self.conv1(x_img))
        x_img = F.relu(self.conv2(x_img))
        x_img = F.relu(self.conv3(x_img))
        x_img = x_img.reshape(x_img.size(0), -1)  # Flatten

        # Process vector inputs
        x_vehicle = F.relu(self.fc_vehicle(state['vehicle_state']))
        x_nav = F.relu(self.fc_nav(state['navigation']))
        x_detect = F.relu(self.fc_detect(state['detections']))

        # Combine all inputs
        combined = torch.cat([x_img, x_vehicle, x_nav, x_detect], dim=1)

        # Final processing
        features = F.relu(self.fc_combined(combined))

        return features


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC.

    This network outputs a Gaussian distribution for each action dimension,
    parameterized by mean and log standard deviation.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256,
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
        super(GaussianPolicy, self).__init__()

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Action range for squashing
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.mean]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Initialize log_std to be slightly negative for small initial std
        nn.init.xavier_uniform_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -1.0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.

        Args:
            features: Feature vector [B, feature_dim]

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            features: Feature vector [B, feature_dim]

        Returns:
            Tuple of (action, log_prob, tanh_mean)
            - action: Sampled action after squashing
            - log_prob: Log probability of the sampled action
            - tanh_mean: Mean action after squashing
        """
        mean, log_std = self.forward(features)
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

        # Account for the squashing transform using the formula:
        # log_prob -= sum(log(1 - tanh(x)^2 + eps))
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # Also compute the tanh of the mean for evaluation
        tanh_mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, tanh_mean


class QNetwork(nn.Module):
    """
    Q-Network for SAC, estimating state-action values.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the Q-Network.

        Args:
            feature_dim: Dimension of input features
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.q]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Estimate Q-value for a given state-action pair.

        Args:
            features: Feature vector [B, feature_dim]
            action: Action vector [B, action_dim]

        Returns:
            Q-value estimate [B, 1]
        """
        x = torch.cat([features, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)

        return q_value


class SAC:
    """
    Soft Actor-Critic agent for autonomous driving.

    This agent implements the SAC algorithm for learning continuous control
    policies in the CARLA environment.
    """

    def __init__(self,
                state_space: Dict[str, Tuple],
                action_dim: int,
                action_low: float = -1.0,
                action_high: float = 1.0,
                lr_actor: float = 3e-4,
                lr_critic: float = 3e-4,
                gamma: float = 0.99,
                tau: float = 0.005,
                alpha: float = 0.2,
                auto_entropy_tuning: bool = True,
                buffer_size: int = 100000,
                batch_size: int = 128,
                feature_dim: int = 256,
                hidden_dim: int = 256,
                update_frequency: int = 2,
                seed: Optional[int] = None):
        """
        Initialize the SAC agent.

        Args:
            state_space: Dictionary of state space shapes
            action_dim: Dimension of action space
            action_low: Lower bound of action range
            action_high: Upper bound of action range
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic networks
            gamma: Discount factor
            tau: Soft update coefficient for target network
            alpha: Temperature parameter for entropy
            auto_entropy_tuning: Whether to auto-tune entropy parameter
            buffer_size: Size of replay buffer
            batch_size: Training batch size
            feature_dim: Dimension of feature vector
            hidden_dim: Dimension of hidden layers
            update_frequency: Number of steps between network updates
            seed: Random seed for reproducibility
        """
        # Set random seeds if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Store parameters
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.step_count = 0

        # Extract image shape from state space
        image_shape = state_space['image']

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            image_shape=image_shape,
            vehicle_state_size=state_space['vehicle_state'][0],
            navigation_size=state_space['navigation'][0],
            detections_size=state_space['detections'][0],
            feature_dim=feature_dim
        ).to(device)

        # Initialize actor (policy) network
        self.actor = GaussianPolicy(
            feature_dim=feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_low=action_low,
            action_high=action_high
        ).to(device)

        # Initialize critic networks (two Q-networks for clipped double Q-learning)
        self.critic1 = QNetwork(feature_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(feature_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = QNetwork(feature_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = QNetwork(feature_dim, action_dim, hidden_dim).to(device)

        # Initialize target networks with same weights
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Freeze target networks for optimization stability
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # Initialize automatic entropy tuning if enabled
        if auto_entropy_tuning:
            # Target entropy is -dim(A)
            self.target_entropy = -action_dim
            # Log alpha is the trainable temperature parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Initialize training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []

        logger.info("SAC Agent initialized")
        logger.info(f"Feature extractor: {self.feature_extractor}")
        logger.info(f"Actor network: {self.actor}")
        logger.info(f"Critic network: {self.critic1}")

    def select_action(self, state: Dict[str, np.ndarray], evaluate: bool = False) -> np.ndarray:
        """
        Select an action based on current policy.

        Args:
            state: Current state dictionary
            evaluate: Whether to use deterministic action (mean) for evaluation

        Returns:
            Selected action as numpy array
        """
        # Convert state dict to tensors and add batch dimension
        state_tensors = {}
        for key, value in state.items():
            state_tensors[key] = torch.FloatTensor(value).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract features
            features = self.feature_extractor(state_tensors)

            if evaluate:
                # Deterministic action (mean) for evaluation
                mean, _ = self.actor(features)
                action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
            else:
                # Stochastic action for training
                action, _, _ = self.actor.sample(features)

        return action.cpu().numpy()[0]

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
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Update networks periodically
        self.step_count += 1
        if self.step_count % self.update_frequency == 0 and self.memory.is_ready(self.batch_size):
            for _ in range(self.update_frequency):  # Multiple updates per step for stability
                self._update_networks()

    def _update_networks(self) -> None:
        """
        Update critic and actor networks using SAC algorithm.
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Extract features for current and next states
        features = self.feature_extractor(states)
        next_features = self.feature_extractor(next_states)

        # Update critics
        self._update_critics(features, actions, rewards, next_features, dones)

        # Update actor and alpha
        self._update_actor_and_alpha(features)

        # Soft update target networks
        self._soft_update_targets()

    def _update_critics(self, features: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_features: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Update critic networks.

        Args:
            features: Current state features
            actions: Actions taken
            rewards: Rewards received
            next_features: Next state features
            dones: Episode termination flags
        """
        with torch.no_grad():
            # Sample actions from next state using current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_features)

            # Calculate target Q-values from target critics
            q1_next = self.critic1_target(next_features, next_actions)
            q2_next = self.critic2_target(next_features, next_actions)

            # Use minimum Q-value for stability
            q_next = torch.min(q1_next, q2_next)

            # Subtract entropy term for soft Q-learning
            if self.auto_entropy_tuning:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = self.alpha

            # Calculate target value with entropy term
            q_target = rewards.unsqueeze(-1) + \
                      (1.0 - dones.unsqueeze(-1)) * self.gamma * \
                      (q_next - alpha * next_log_probs)

        # Calculate current Q-values
        q1 = self.critic1(features, actions)
        q2 = self.critic2(features, actions)

        # Calculate critic losses (MSE)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        # Update first critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        # Update second critic
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # Record losses
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)

    def _update_actor_and_alpha(self, features: torch.Tensor) -> None:
        """
        Update actor network and entropy coefficient.

        Args:
            features: Current state features
        """
        # Sample actions from current policy
        actions, log_probs, _ = self.actor.sample(features)

        # Calculate Q-values for current policy
        q1 = self.critic1(features, actions)
        q2 = self.critic2(features, actions)
        q = torch.min(q1, q2)

        # Get current alpha value
        if self.auto_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Calculate actor loss: maximize Q-value and entropy
        actor_loss = (alpha * log_probs - q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Record actor loss
        self.actor_losses.append(actor_loss.item())

        # Update entropy coefficient if using automatic tuning
        if self.auto_entropy_tuning:
            # Calculate alpha loss: minimize alpha * (log_prob + target_entropy)
            alpha_loss = -(alpha * (log_probs + self.target_entropy).detach()).mean()

            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Record alpha loss
            self.alpha_losses.append(alpha_loss.item())

    def _soft_update_targets(self) -> None:
        """
        Soft update of target critic networks.

        Uses the formula:
            target_param = tau * param + (1 - tau) * target_param
        """
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: str) -> None:
        """
        Save the model weights.

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'alpha_losses': self.alpha_losses,
            'step_count': self.step_count
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str, eval_mode: bool = False) -> None:
        """
        Load model weights.

        Args:
            filepath: Path to load the model from
            eval_mode: Whether to set the model to evaluation mode
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file {filepath} not found")
            return

        checkpoint = torch.load(filepath, map_location=device)

        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.alpha_losses = checkpoint['alpha_losses']
        self.step_count = checkpoint['step_count']

        if eval_mode:
            self.feature_extractor.eval()
            self.actor.eval()
            self.critic1.eval()
            self.critic2.eval()
            self.critic1_target.eval()
            self.critic2_target.eval()

        logger.info(f"Model loaded from {filepath}")


def train_sac(env: CarlaEnvWrapper,
             agent: SAC,
             n_episodes: int = 1000,
             max_steps: int = 1000,
             checkpoint_dir: str = './checkpoints',
             checkpoint_freq: int = 100,
             eval_freq: int = 100,
             eval_episodes: int = 5) -> Dict[str, List[float]]:
    """
    Train the SAC agent.

    Args:
        env: CARLA environment wrapper
        agent: SAC agent
        n_episodes: Maximum number of episodes
        max_steps: Maximum steps per episode
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency of checkpoints in episodes
        eval_freq: Frequency of evaluation in episodes
        eval_episodes: Number of episodes for evaluation

    Returns:
        Dictionary of training statistics
    """
    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_steps': [],
        'episode_durations': [],
        'eval_rewards': [],
        'eval_steps': []
    }

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for episode in range(1, n_episodes + 1):
        start_time = time.time()
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        # Training episode
        for t in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update agent
            agent.step(state, action, reward, next_state, done)

            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # Record episode statistics
        duration = time.time() - start_time
        stats['episode_rewards'].append(episode_reward)
        stats['episode_steps'].append(episode_steps)
        stats['episode_durations'].append(duration)

        # Log progress
        logger.info(f"Episode {episode}/{n_episodes} | "
                   f"Reward: {episode_reward:.2f} | "
                   f"Steps: {episode_steps} | "
                   f"Duration: {duration:.2f}s")

        # Periodically evaluate
        if episode % eval_freq == 0:
            eval_rewards, eval_steps = evaluate_agent(env, agent, eval_episodes, max_steps)
            avg_eval_reward = np.mean(eval_rewards)
            avg_eval_steps = np.mean(eval_steps)

            stats['eval_rewards'].append(avg_eval_reward)
            stats['eval_steps'].append(avg_eval_steps)

            logger.info(f"Evaluation | "
                       f"Avg Reward: {avg_eval_reward:.2f} | "
                       f"Avg Steps: {avg_eval_steps:.2f}")

        # Save checkpoint
        if episode % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'sac_checkpoint_{episode}.pth')
            agent.save(checkpoint_path)

            # Also save latest checkpoint
            latest_path = os.path.join(checkpoint_dir, 'sac_latest.pth')
            agent.save(latest_path)

            # Log average score
            avg_reward = np.mean(stats['episode_rewards'][-checkpoint_freq:])
            logger.info(f"Average Reward (last {checkpoint_freq} episodes): {avg_reward:.2f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, 'sac_final.pth')
    agent.save(final_path)

    return stats


def evaluate_agent(env: CarlaEnvWrapper, agent: SAC, n_episodes: int = 10,
                  max_steps: int = 1000) -> Tuple[List[float], List[int]]:
    """
    Evaluate the agent's performance.

    Args:
        env: CARLA environment wrapper
        agent: SAC agent
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (rewards, steps) lists for each episode
    """
    rewards = []
    steps = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        for t in range(max_steps):
            # Select action deterministically
            action = agent.select_action(state, evaluate=True)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # Record episode statistics
        rewards.append(episode_reward)
        steps.append(episode_steps)

        # Log progress
        logger.info(f"Eval Episode {episode}/{n_episodes} | "
                   f"Reward: {episode_reward:.2f} | "
                   f"Steps: {episode_steps}")

    return rewards, steps


if __name__ == "__main__":
    try:
        # Initialize environment
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            city_name='Town01',
            image_size=(84, 84),
            frame_skip=2,
            max_episode_steps=1000,
            weather_id=0,
            quality_level='Low',
            random_start=True
        )

        # Define state and action spaces
        state_space = {
            'image': (84, 84, 3),
            'vehicle_state': (9,),
            'navigation': (3,),
            'detections': (10,)
        }
        action_dim = 3  # [throttle, brake, steer]

        # Initialize agent
        agent = SAC(
            state_space=state_space,
            action_dim=action_dim,
            action_low=np.array([0, 0, -1]),  # throttle, brake, steer
            action_high=np.array([1, 1, 1]),  # throttle, brake, steer
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_entropy_tuning=True,
            buffer_size=100000,
            batch_size=128,
            feature_dim=256,
            hidden_dim=256,
            update_frequency=2,
            seed=42
        )

        # Train agent
        stats = train_sac(
            env=env,
            agent=agent,
            n_episodes=500,
            max_steps=1000,
            checkpoint_dir='./checkpoints',
            checkpoint_freq=50,
            eval_freq=50,
            eval_episodes=5
        )

        # Close environment
        env.close()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'env' in locals():
            env.close()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if 'env' in locals():
            env.close()

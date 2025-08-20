"""
Neural Network Architectures for CARLA PPO Agent

This module provides optimized neural network architectures for multimodal
autonomous driving. Includes CNN feature extraction for camera images,
MLP processing for vector observations, and policy/value networks.

Architectures are based on successful implementations from research papers
and optimized for CARLA's observation space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """Multimodal feature extractor for CARLA observations.
    
    Processes camera images with CNN and vector observations with MLP,
    then fuses features for policy and value networks.
    """
    
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 vector_dim: int,
                 cnn_features: int = 256,
                 mlp_features: int = 128,
                 output_features: int = 512):
        """Initialize feature extractor.
        
        Args:
            image_shape: Shape of input images (C, H, W)
            vector_dim: Dimension of vector observations
            cnn_features: CNN output features
            mlp_features: MLP hidden features
            output_features: Final output features
        """
        super().__init__()
        
        self.image_shape = image_shape
        self.vector_dim = vector_dim
        self.output_features = output_features
        
        # CNN for image processing
        self.cnn = self._build_cnn(image_shape, cnn_features)
        
        # MLP for vector observations
        self.mlp = self._build_mlp(vector_dim, mlp_features)
        
        # Feature fusion
        fusion_input_dim = cnn_features + mlp_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_features),
            nn.ReLU(),
            nn.Linear(output_features, output_features),
            nn.ReLU()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"FeatureExtractor initialized - "
                   f"Image: {image_shape}, Vector: {vector_dim}, "
                   f"Output: {output_features}")
    
    def _build_cnn(self, image_shape: Tuple[int, int, int], output_features: int) -> nn.Module:
        """Build CNN for image feature extraction.
        
        Args:
            image_shape: Input image shape (C, H, W)
            output_features: Number of output features
            
        Returns:
            CNN module
        """
        channels, height, width = image_shape
        
        # Define CNN architecture
        layers = []
        
        # Conv block 1
        layers.extend([
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ])
        
        # Adaptive pooling to ensure consistent output size
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        # Flatten and fully connected
        layers.extend([
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, output_features),
            nn.ReLU()
        ])
        
        return nn.Sequential(*layers)
    
    def _build_mlp(self, input_dim: int, output_features: int) -> nn.Module:
        """Build MLP for vector feature extraction.
        
        Args:
            input_dim: Input vector dimension
            output_features: Number of output features
            
        Returns:
            MLP module
        """
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_features),
            nn.ReLU()
        )
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through feature extractor.
        
        Args:
            observations: Dictionary containing 'image' and 'vector' observations
            
        Returns:
            Extracted features tensor
        """
        # Process image observations
        image_features = self.cnn(observations['image'])
        
        # Process vector observations
        vector_features = self.mlp(observations['vector'])
        
        # Fuse features
        combined_features = torch.cat([image_features, vector_features], dim=1)
        output_features = self.fusion(combined_features)
        
        return output_features


class PolicyNetwork(nn.Module):
    """Policy network for continuous action spaces.
    
    Outputs mean and log standard deviation for action distribution.
    Uses separate networks for mean and std for better stability.
    """
    
    def __init__(self,
                 feature_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 log_std_init: float = 0.0,
                 log_std_min: float = -20.0,
                 log_std_max: float = 2.0):
        """Initialize policy network.
        
        Args:
            feature_dim: Input feature dimension
            action_dim: Output action dimension
            hidden_dim: Hidden layer dimension
            log_std_init: Initial log standard deviation
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are normalized to [-1, 1]
        )
        
        # Log std network (independent of state)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"PolicyNetwork initialized - "
                   f"Feature dim: {feature_dim}, Action dim: {action_dim}")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Normal:
        """Forward pass through policy network.
        
        Args:
            features: Input features
            
        Returns:
            Action distribution
        """
        # Compute action mean
        action_mean = self.mean_net(features)
        
        # Compute action standard deviation
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)
        
        # Create distribution
        action_dist = Normal(action_mean, action_std)
        
        return action_dist
    
    def get_action_log_prob(self,
                           features: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action log probabilities and entropy.
        
        Args:
            features: Input features
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        action_dist = self.forward(features)
        log_probs = action_dist.log_prob(actions).sum(-1)
        entropy = action_dist.entropy().sum(-1)
        
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Value network for state value estimation.
    
    Estimates V(s) for the current policy.
    """
    
    def __init__(self,
                 feature_dim: int,
                 hidden_dim: int = 256):
        """Initialize value network.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"ValueNetwork initialized - Feature dim: {feature_dim}")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network.
        
        Args:
            features: Input features
            
        Returns:
            State values
        """
        return self.network(features)


class CriticsEnsemble(nn.Module):
    """Ensemble of critic networks for improved stability.
    
    Uses multiple value networks and takes their average or minimum
    for more robust value estimation.
    """
    
    def __init__(self,
                 feature_dim: int,
                 n_critics: int = 2,
                 hidden_dim: int = 256,
                 use_min: bool = False):
        """Initialize critics ensemble.
        
        Args:
            feature_dim: Input feature dimension
            n_critics: Number of critic networks
            hidden_dim: Hidden layer dimension
            use_min: Whether to use minimum instead of mean
        """
        super().__init__()
        
        self.n_critics = n_critics
        self.use_min = use_min
        
        # Create critic networks
        self.critics = nn.ModuleList([
            ValueNetwork(feature_dim, hidden_dim)
            for _ in range(n_critics)
        ])
        
        logger.info(f"CriticsEnsemble initialized - "
                   f"N critics: {n_critics}, Use min: {use_min}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through critics ensemble.
        
        Args:
            features: Input features
            
        Returns:
            Ensemble value estimate
        """
        # Get values from all critics
        values = torch.stack([
            critic(features).squeeze(-1)
            for critic in self.critics
        ], dim=0)
        
        # Combine values
        if self.use_min:
            ensemble_value = torch.min(values, dim=0)[0]
        else:
            ensemble_value = torch.mean(values, dim=0)
        
        return ensemble_value.unsqueeze(-1)
    
    def get_all_values(self, features: torch.Tensor) -> torch.Tensor:
        """Get values from all critics separately.
        
        Args:
            features: Input features
            
        Returns:
            Values from all critics (n_critics, batch_size, 1)
        """
        return torch.stack([
            critic(features)
            for critic in self.critics
        ], dim=0)


class RecurrentFeatureExtractor(nn.Module):
    """Recurrent feature extractor for temporal dependencies.
    
    Uses LSTM to process sequential observations for better
    temporal understanding in autonomous driving.
    """
    
    def __init__(self,
                 image_shape: Tuple[int, int, int],
                 vector_dim: int,
                 hidden_dim: int = 256,
                 lstm_layers: int = 1,
                 output_features: int = 512):
        """Initialize recurrent feature extractor.
        
        Args:
            image_shape: Shape of input images (C, H, W)
            vector_dim: Dimension of vector observations
            hidden_dim: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            output_features: Final output features
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # Base feature extractor
        self.base_extractor = FeatureExtractor(
            image_shape=image_shape,
            vector_dim=vector_dim,
            output_features=hidden_dim
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_features)
        
        # Hidden state initialization
        self.hidden_state = None
        
        logger.info(f"RecurrentFeatureExtractor initialized - "
                   f"Hidden dim: {hidden_dim}, LSTM layers: {lstm_layers}")
    
    def forward(self,
                observations: Dict[str, torch.Tensor],
                reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through recurrent feature extractor.
        
        Args:
            observations: Dictionary containing observations
            reset_mask: Mask indicating episode resets
            
        Returns:
            Extracted features with temporal information
        """
        batch_size = observations['image'].size(0)
        
        # Extract base features
        base_features = self.base_extractor(observations)
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            self.hidden_state = (
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim,
                           device=base_features.device),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim,
                           device=base_features.device)
            )
        
        # Reset hidden state for terminated episodes
        if reset_mask is not None:
            reset_mask = reset_mask.view(1, -1, 1).expand_as(self.hidden_state[0])
            self.hidden_state = (
                self.hidden_state[0] * (1 - reset_mask),
                self.hidden_state[1] * (1 - reset_mask)
            )
        
        # Process through LSTM
        lstm_input = base_features.unsqueeze(1)  # Add sequence dimension
        lstm_output, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        
        # Project to output features
        output_features = self.output_proj(lstm_output.squeeze(1))
        
        return output_features
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state."""
        self.hidden_state = None


def create_networks(config: Dict, obs_space: Dict, action_dim: int) -> Dict[str, nn.Module]:
    """Create all required networks based on configuration.
    
    Args:
        config: Network configuration
        obs_space: Observation space specification
        action_dim: Action dimension
        
    Returns:
        Dictionary of networks
    """
    # Determine network types
    use_recurrent = config.get('use_recurrent', False)
    use_ensemble = config.get('use_ensemble', False)
    
    # Create feature extractor
    if use_recurrent:
        feature_extractor = RecurrentFeatureExtractor(
            image_shape=obs_space['image'],
            vector_dim=obs_space['vector'][0],
            hidden_dim=config.get('hidden_dim', 256),
            lstm_layers=config.get('lstm_layers', 1),
            output_features=config.get('feature_dim', 512)
        )
    else:
        feature_extractor = FeatureExtractor(
            image_shape=obs_space['image'],
            vector_dim=obs_space['vector'][0],
            output_features=config.get('feature_dim', 512)
        )
    
    # Create policy network
    policy_net = PolicyNetwork(
        feature_dim=config.get('feature_dim', 512),
        action_dim=action_dim,
        hidden_dim=config.get('policy_hidden_dim', 256)
    )
    
    # Create value network
    if use_ensemble:
        value_net = CriticsEnsemble(
            feature_dim=config.get('feature_dim', 512),
            n_critics=config.get('n_critics', 2),
            hidden_dim=config.get('value_hidden_dim', 256),
            use_min=config.get('use_min_critic', False)
        )
    else:
        value_net = ValueNetwork(
            feature_dim=config.get('feature_dim', 512),
            hidden_dim=config.get('value_hidden_dim', 256)
        )
    
    networks = {
        'feature_extractor': feature_extractor,
        'policy_net': policy_net,
        'value_net': value_net
    }
    
    logger.info("Networks created successfully")
    return networks

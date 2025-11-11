"""
CNN Feature Extractor for End-to-End Autonomous Vehicle Navigation.

Implements NatureCNN architecture for visual feature extraction from stacked camera frames,
following the Deep Q-Network (DQN) approach for reinforcement learning in continuous control.

This module is part of a Twin Delayed Deep Deterministic Policy Gradient (TD3) system for
autonomous vehicle navigation in CARLA simulator, combining visual perception with kinematic
state for robust decision-making in complex driving scenarios.

References:
    - Mnih et al., "Human-level control through deep reinforcement learning," Nature (2015)
    - Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods," ICML (2018)
    - Perot et al., "End-to-End Race Driving with Deep Reinforcement Learning," ICRA (2017)
    - Ben Elallid et al., "Deep Reinforcement Learning for Autonomous Vehicle Intersection Navigation" (2023)

Author: Daniel Terra
Date: 2025-01-16
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple


class NatureCNN(nn.Module):
    """
    NatureCNN visual feature extractor for end-to-end deep reinforcement learning.

    This convolutional neural network extracts spatial and temporal features from stacked
    camera frames for autonomous vehicle control. The architecture follows the Nature DQN
    paper (Mnih et al., 2015) with modifications for continuous control in TD3 framework.

    Key Design Choices:
        1. Frame Stacking: 4 consecutive frames provide temporal context for velocity estimation
        2. Grayscale: Reduces dimensionality while preserving structural information
        3. Leaky ReLU: Preserves negative values from zero-centered normalization [-1,1]
        4. Compact Architecture: 512-dim output balances expressiveness and efficiency

    Architecture:
        Input:   (batch, 4, 84, 84) - 4 stacked grayscale frames, normalized to [-1, 1]
        Conv1:   (batch, 32, 20, 20) - 32 filters, 8×8 kernel, stride 4
        Conv2:   (batch, 64, 9, 9)   - 64 filters, 4×4 kernel, stride 2
        Conv3:   (batch, 64, 7, 7)   - 64 filters, 3×3 kernel, stride 1
        Flatten: (batch, 3136)       - 64 × 7 × 7 = 3136 features
        FC:      (batch, 512)        - Fully connected layer
        Output:  512-dimensional feature vector for actor/critic networks

    Normalization Strategy:
        Input frames are zero-centered [-1, 1] following modern deep learning best practices.
        Leaky ReLU (negative_slope=0.01) is used instead of standard ReLU to preserve
        negative information and prevent "dying ReLU" problem, ensuring stable gradient flow.

    References:
        - Mnih et al. (2015): "Human-level control through deep reinforcement learning," Nature
        - Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
        - Perot et al. (2017): "End-to-End Race Driving with Deep Reinforcement Learning"
        - Sallab et al. (2017): "End-to-End Deep Reinforcement Learning for Lane Keeping Assist"

    Args:
        input_channels (int): Number of input channels (default: 4 for frame stacking)
        num_frames (int): Number of frames in stack for validation (default: 4)
        feature_dim (int): Output feature dimension (default: 512)

    Example:
        >>> cnn = NatureCNN(input_channels=4, feature_dim=512)
        >>> frames = torch.randn(32, 4, 84, 84)  # Batch of 32 stacked frames
        >>> features = cnn(frames)  # Shape: (32, 512)
    """

    def __init__(
        self,
        input_channels: int = 4,
        num_frames: int = 4,
        feature_dim: int = 512,
    ):
        """
        Initialize NatureCNN feature extractor.

        Args:
            input_channels: Number of input channels (4 for stacked frames)
            num_frames: Number of frames in stack (for validation)
            feature_dim: Output feature dimension (512)
        """
        super(NatureCNN, self).__init__()

        self.input_channels = input_channels
        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # Logger for debug output
        self.logger = logging.getLogger(__name__)

        # OPTIMIZATION: Step counter for throttled debug logging (every 100 calls)
        self.forward_step_counter = 0
        self.log_frequency = 100

        # Convolutional layers following Nature DQN architecture
        # Layer 1: Extract low-level features (edges, textures) with large receptive field
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0,
        )

        # Layer 2: Combine low-level features into mid-level representations
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0,
        )

        # Layer 3: Extract high-level semantic features (lanes, vehicles, road structure)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # Activation: Leaky ReLU preserves negative values from zero-centered normalization
        # Standard ReLU would kill ~50% of pixels, severely limiting feature capacity
        # Reference: Maas et al. (2013), "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Compute flattened size dynamically
        self._compute_flat_size()

        # Fully connected layer: Project spatial features to fixed-size representation
        self.fc = nn.Linear(self.flat_size, feature_dim)

        # Initialize weights for stable training
        self._initialize_weights()

    def _compute_flat_size(self):
        """
        Compute flattened feature size after convolutional layers.

        Uses a dummy forward pass to determine output dimensions, avoiding
        manual calculation errors. This is standard practice in modern CNNs.

        Output:
            Sets self.flat_size = 64 × 7 × 7 = 3136 for 84×84 input
        """
        with torch.no_grad():
            # Create dummy input (batch_size=1)
            dummy_input = torch.zeros(1, self.input_channels, 84, 84)

            # Forward through conv layers
            out = self.activation(self.conv1(dummy_input))  # (1, 32, 20, 20)
            out = self.activation(self.conv2(out))          # (1, 64, 9, 9)
            out = self.activation(self.conv3(out))          # (1, 64, 7, 7)

            # Compute flat size: 64 × 7 × 7 = 3136
            self.flat_size = int(np.prod(out.shape[1:]))

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization.

        Kaiming (He) initialization is optimal for ReLU-like activations (including Leaky ReLU),
        maintaining proper variance throughout the network for stable gradient flow.

        Reference:
            He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level
            Performance on ImageNet Classification"
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN for visual feature extraction.

        Processes stacked camera frames through convolutional layers to extract
        hierarchical features for autonomous driving: edges → textures → semantics.

        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)
               Stacked grayscale frames normalized to [-1, 1] (zero-centered)

        Returns:
            Feature vector of shape (batch_size, 512)
            Ready for concatenation with kinematic state in actor/critic networks

        Raises:
            ValueError: If input shape does not match expected (batch, 4, 84, 84)

        Note:
            Zero-centered normalization [-1, 1] is used (not [0, 1]) following modern
            best practices for deep learning. Leaky ReLU activation preserves negative
            information, preventing the "dying ReLU" problem where neurons become inactive.

        Example:
            >>> cnn = NatureCNN()
            >>> frames = torch.randn(16, 4, 84, 84)  # 16 samples
            >>> features = cnn(frames)
            >>> print(features.shape)  # torch.Size([16, 512])
        """
        # OPTIMIZATION: Increment step counter for throttled logging
        self.forward_step_counter += 1
        should_log = (self.forward_step_counter % self.log_frequency == 0)

        # Validate input shape
        if x.shape[1:] != (self.input_channels, 84, 84):
            raise ValueError(
                f"Expected input shape (batch, {self.input_channels}, 84, 84), "
                f"got {x.shape}"
            )

        # DEBUG: Log input statistics every 100 forward passes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every forward pass)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN FORWARD PASS #{self.forward_step_counter} - INPUT:\n"
                f"   Shape: {x.shape}\n"
                f"   Dtype: {x.dtype}\n"
                f"   Device: {x.device}\n"
                f"   Range: [{x.min().item():.3f}, {x.max().item():.3f}]\n"
                f"   Mean: {x.mean().item():.3f}, Std: {x.std().item():.3f}\n"
                f"   Has NaN: {torch.isnan(x).any().item()}\n"
                f"   Has Inf: {torch.isinf(x).any().item()}"
            )

        # Convolutional layers with Leaky ReLU activations
        # Preserves negative values from zero-centered normalization
        out = self.activation(self.conv1(x))   # (batch, 32, 20, 20)

        # DEBUG: Log after conv1 every 100 forward passes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every forward pass)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN LAYER 1 (Conv 32×8×8, stride=4):\n"
                f"   Output shape: {out.shape}\n"
                f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]\n"
                f"   Mean: {out.mean().item():.3f}, Std: {out.std().item():.3f}\n"
                f"   Active neurons: {(out > 0).float().mean().item()*100:.1f}%"
            )

        out = self.activation(self.conv2(out))  # (batch, 64, 9, 9)

        # DEBUG: Log after conv2 every 100 forward passes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every forward pass)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN LAYER 2 (Conv 64×4×4, stride=2):\n"
                f"   Output shape: {out.shape}\n"
                f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]\n"
                f"   Mean: {out.mean().item():.3f}, Std: {out.std().item():.3f}\n"
                f"   Active neurons: {(out > 0).float().mean().item()*100:.1f}%"
            )

        out = self.activation(self.conv3(out))  # (batch, 64, 7, 7)

        # DEBUG: Log after conv3 every 100 forward passes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every forward pass)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN LAYER 3 (Conv 64×3×3, stride=1):\n"
                f"   Output shape: {out.shape}\n"
                f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]\n"
                f"   Mean: {out.mean().item():.3f}, Std: {out.std().item():.3f}\n"
                f"   Active neurons: {(out > 0).float().mean().item()*100:.1f}%"
            )

        # Flatten spatial dimensions
        out = out.view(out.size(0), -1)  # (batch, 3136)

        # Fully connected projection to feature space
        features = self.fc(out)  # (batch, 512)

        # DEBUG: Log output features every 100 forward passes
        # OPTIMIZATION: Throttled to reduce logging overhead (was every forward pass)
        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   CNN FORWARD PASS - OUTPUT:\n"
                f"   Feature shape: {features.shape}\n"
                f"   Range: [{features.min().item():.3f}, {features.max().item():.3f}]\n"
                f"   Mean: {features.mean().item():.3f}, Std: {features.std().item():.3f}\n"
                f"   L2 norm: {features.norm(dim=1).mean().item():.3f}\n"
                f"   Has NaN: {torch.isnan(features).any().item()}\n"
                f"   Has Inf: {torch.isinf(features).any().item()}\n"
                f"   Feature quality: {'GOOD' if not (torch.isnan(features).any() or torch.isinf(features).any()) else 'BAD'}"
            )

        return features

    def get_feature_dim(self) -> int:
        """
        Get output feature dimension.

        Returns:
            int: Feature dimension (512)
        """
        return self.feature_dim

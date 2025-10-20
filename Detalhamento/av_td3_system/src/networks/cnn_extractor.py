"""
Convolutional Neural Network for Visual Feature Extraction

Implements NatureCNN architecture from DQN paper (Mnih et al., 2015):
- 3 convolutional layers with ReLU activations
- Flattening layer
- Fully connected layer (512 units) producing feature vector

Input: 4×84×84 stacked grayscale frames
Output: 512-dimensional feature vector for concatenation with kinematic features

Paper Reference: "Human-level control through deep reinforcement learning" (DQN)
Architecture: Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,1) → FC(512)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class NatureCNN(nn.Module):
    """
    NatureCNN visual feature extractor.

    Architecture:
    - Input: (batch, 4, 84, 84) - 4 stacked 84×84 grayscale frames
    - Conv1: 32 filters, 8×8 kernel, stride 4 → (batch, 32, 20, 20)
    - Conv2: 64 filters, 4×4 kernel, stride 2 → (batch, 64, 9, 9)
    - Conv3: 64 filters, 3×3 kernel, stride 1 → (batch, 64, 7, 7)
    - Flatten → (batch, 64*7*7=3136)
    - FC: 512 units → (batch, 512)
    - Output: 512-dimensional feature vector

    The output vector captures temporal dynamics and spatial features
    from the 4-frame stack, ready for concatenation with kinematic state.
    """

    def __init__(
        self,
        input_channels: int = 4,
        num_frames: int = 4,
        feature_dim: int = 512,
    ):
        """
        Initialize NatureCNN.

        Args:
            input_channels: Number of input channels (4 for stacked frames)
            num_frames: Number of frames in stack (for validation)
            feature_dim: Output feature dimension (512)
        """
        super(NatureCNN, self).__init__()

        self.input_channels = input_channels
        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # Convolutional layers
        # Layer 1: 32 filters, 8×8 kernel, stride 4
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0,
        )

        # Layer 2: 64 filters, 4×4 kernel, stride 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0,
        )

        # Layer 3: 64 filters, 3×3 kernel, stride 1
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # Compute flattened size
        self._compute_flat_size()

        # Fully connected layer
        self.fc = nn.Linear(self.flat_size, feature_dim)

        # Activation function
        self.relu = nn.ReLU()

    def _compute_flat_size(self):
        """
        Compute flattened feature size after conv layers.

        Uses dummy forward pass to determine output size.
        """
        with torch.no_grad():
            # Create dummy input (batch_size=1)
            dummy_input = torch.zeros(1, self.input_channels, 84, 84)

            # Forward through conv layers
            out = self.relu(self.conv1(dummy_input))  # (1, 32, 20, 20)
            out = self.relu(self.conv2(out))  # (1, 64, 9, 9)
            out = self.relu(self.conv3(out))  # (1, 64, 7, 7)

            # Compute flat size
            self.flat_size = int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.

        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)
               Values should be normalized to [0, 1]

        Returns:
            Feature vector of shape (batch_size, 512)
            Normalized by layer and ready for concatenation
        """
        # Validate input shape
        if x.shape[1:] != (self.input_channels, 84, 84):
            raise ValueError(
                f"Expected input shape (batch, {self.input_channels}, 84, 84), "
                f"got {x.shape}"
            )

        # Convolutional layers with ReLU activations
        out = self.relu(self.conv1(x))  # (batch, 32, 20, 20)
        out = self.relu(self.conv2(out))  # (batch, 64, 9, 9)
        out = self.relu(self.conv3(out))  # (batch, 64, 7, 7)

        # Flatten
        out = out.view(out.size(0), -1)  # (batch, flat_size)

        # Fully connected layer
        features = self.fc(out)  # (batch, 512)

        return features

    def get_feature_dim(self) -> int:
        """
        Get output feature dimension.

        Returns:
            512
        """
        return self.feature_dim


class StateEncoder(nn.Module):
    """
    Encodes full state for TD3/DDPG.

    Combines:
    - Visual features from CNN (512 dims)
    - Kinematic state (3 dims: velocity, lateral_dev, heading_err)
    - Navigation waypoints (20 dims: 10 waypoints × 2)
    - Total: 535 dims

    Can optionally normalize/process features before concatenation.
    """

    def __init__(
        self,
        cnn_feature_dim: int = 512,
        kinematic_dim: int = 3,
        waypoint_dim: int = 20,
        normalize: bool = True,
    ):
        """
        Initialize state encoder.

        Args:
            cnn_feature_dim: Output dimension of CNN (512)
            kinematic_dim: Dimension of kinematic state (3)
            waypoint_dim: Dimension of waypoint features (20)
            normalize: Whether to apply layer normalization to visual features
        """
        super(StateEncoder, self).__init__()

        self.cnn_feature_dim = cnn_feature_dim
        self.kinematic_dim = kinematic_dim
        self.waypoint_dim = waypoint_dim
        self.total_dim = cnn_feature_dim + kinematic_dim + waypoint_dim

        # Optional layer normalization for visual features
        self.normalize = normalize
        if normalize:
            self.layer_norm = nn.LayerNorm(cnn_feature_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        kinematic_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode full state by concatenating all components.

        Args:
            image_features: CNN features (batch_size, 512)
            kinematic_state: Kinematic + waypoint state (batch_size, 23)

        Returns:
            Full encoded state (batch_size, 535)
        """
        # Validate shapes
        if image_features.shape[1] != self.cnn_feature_dim:
            raise ValueError(
                f"Expected image features dim {self.cnn_feature_dim}, "
                f"got {image_features.shape[1]}"
            )

        expected_kinematic_dim = self.kinematic_dim + self.waypoint_dim
        if kinematic_state.shape[1] != expected_kinematic_dim:
            raise ValueError(
                f"Expected kinematic state dim {expected_kinematic_dim}, "
                f"got {kinematic_state.shape[1]}"
            )

        # Normalize visual features if enabled
        if self.normalize:
            image_features = self.layer_norm(image_features)

        # Concatenate all features
        full_state = torch.cat([image_features, kinematic_state], dim=1)

        return full_state

    def get_state_dim(self) -> int:
        """
        Get total state dimension.

        Returns:
            535 (512 + 3 + 20)
        """
        return self.total_dim


# Helper function for quick feature dimension calculation
def compute_nature_cnn_output_size(
    input_height: int = 84,
    input_width: int = 84,
) -> Tuple[int, int, int]:
    """
    Compute output dimensions after NatureCNN conv layers.

    Args:
        input_height: Input height (default 84)
        input_width: Input width (default 84)

    Returns:
        Tuple of (height, width, channels) after conv layers
        For 84×84 input: (7, 7, 64) → 3136 features after flatten

    Note: Calculated as:
    - After Conv1 (k=8, s=4): (84-8)/4+1 = 20
    - After Conv2 (k=4, s=2): (20-4)/2+1 = 9
    - After Conv3 (k=3, s=1): (9-3)/1+1 = 7
    """
    height = input_height
    width = input_width

    # Conv1: k=8, s=4
    height = (height - 8) // 4 + 1  # 20
    width = (width - 8) // 4 + 1  # 20

    # Conv2: k=4, s=2
    height = (height - 4) // 2 + 1  # 9
    width = (width - 4) // 2 + 1  # 9

    # Conv3: k=3, s=1
    height = (height - 3) // 1 + 1  # 7
    width = (width - 3) // 1 + 1  # 7

    channels = 64  # Output channels

    return (height, width, channels)


if __name__ == "__main__":
    """Quick test of CNN architecture."""
    import torch

    # Test CNN
    print("Testing NatureCNN architecture...")
    cnn = NatureCNN(input_channels=4, feature_dim=512)

    # Dummy input (batch_size=2, 4 stacked frames, 84×84)
    dummy_frames = torch.randn(2, 4, 84, 84)

    # Forward pass
    features = cnn(dummy_frames)

    print(f"Input shape: {dummy_frames.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Expected: (2, 512)")

    # Test state encoder
    print("\nTesting StateEncoder...")
    encoder = StateEncoder()

    dummy_kinematic = torch.randn(2, 23)
    full_state = encoder(features, dummy_kinematic)

    print(f"Image features: {features.shape}")
    print(f"Kinematic state: {dummy_kinematic.shape}")
    print(f"Full state: {full_state.shape}")
    print(f"Expected: (2, 535)")

    # Test output dimensions
    print("\nOutput dimensions after conv layers:")
    h, w, c = compute_nature_cnn_output_size(84, 84)
    print(f"Height: {h}, Width: {w}, Channels: {c}")
    print(f"Flattened size: {h * w * c}")

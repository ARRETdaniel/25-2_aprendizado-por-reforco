"""
CNN Feature Extractor for Visual Navigation.

Extracts features from stacked camera frames for TD3 agent.
Implements transfer learning using pretrained MobileNetV3 for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class NatureCNN(nn.Module):
    """
    Nature CNN architecture for feature extraction.

    Input: 4-channel stacked grayscale images (4, 84, 84)
    Output: 512-dimensional feature vector

    Architecture:
        - Conv1: 4x84x84 -> 32x20x20 (8x8 kernel, stride 4)
        - Conv2: 32x20x20 -> 64x9x9 (4x4 kernel, stride 2)
        - Conv3: 64x9x9 -> 64x7x7 (3x3 kernel, stride 1)
        - Flatten: 64x7x7 -> 3136
        - FC1: 3136 -> 512

    Reference:
        Mnih et al., "Human-level control through deep reinforcement learning"
        Nature 518.7540 (2015): 529-533.
    """

    def __init__(self, input_channels: int = 4, output_dim: int = 512):
        """
        Initialize Nature CNN.

        Args:
            input_channels: Number of input channels (default: 4 for stacked frames)
            output_dim: Dimension of output feature vector (default: 512)
        """
        super(NatureCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size after convolutions
        # Input: 84x84
        # After conv1: (84 - 8) // 4 + 1 = 20
        # After conv2: (20 - 4) // 2 + 1 = 9
        # After conv3: (9 - 3) // 1 + 1 = 7
        # Flattened: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)

        Returns:
            Feature vector of shape (batch_size, 512)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return x


class MobileNetV3FeatureExtractor(nn.Module):
    """
    MobileNetV3-based feature extractor with transfer learning.

    Uses pretrained MobileNetV3-Small as backbone for efficient feature extraction.
    Adapted for 4-channel grayscale input (stacked frames) and 512-dim output.

    This provides better feature learning than training from scratch while
    maintaining real-time performance suitable for autonomous navigation.

    Input: 4-channel stacked grayscale images (4, 84, 84)
    Output: 512-dimensional feature vector

    Architecture:
        - Input adaptation: 4-channel grayscale -> 3-channel RGB conversion
        - MobileNetV3-Small backbone (pretrained on ImageNet)
        - Custom classifier head: backbone_features -> 512

    Reference:
        Howard et al., "Searching for MobileNetV3" (2019)
        https://arxiv.org/abs/1905.02244
    """

    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize MobileNetV3 feature extractor.

        Args:
            input_channels: Number of input channels (default: 4 for stacked frames)
            output_dim: Dimension of output feature vector (default: 512)
            pretrained: Whether to use ImageNet pretrained weights (default: True)
            freeze_backbone: Whether to freeze backbone weights (default: False)
        """
        super(MobileNetV3FeatureExtractor, self).__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)

        # Adapt first conv layer for 4-channel input
        # Original: Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        # We'll use a projection layer to convert 4 channels -> 3 channels
        self.input_projection = nn.Conv2d(
            input_channels, 3,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        nn.init.kaiming_normal_(self.input_projection.weight, mode='fan_out', nonlinearity='relu')

        # Freeze backbone if requested (for faster training initially)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier with custom head
        # MobileNetV3-Small last conv outputs 576 features
        backbone_output_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier

        # Custom classifier head for 512-dim output
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_output_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)

        Returns:
            Feature vector of shape (batch_size, 512)
        """
        # Project 4 channels to 3 channels for backbone
        x = self.input_projection(x)

        # Extract features using MobileNetV3 backbone
        x = self.backbone.features(x)

        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Custom feature head
        x = self.feature_head(x)

        return x

    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet18-based feature extractor with transfer learning.

    Uses pretrained ResNet18 as backbone for robust feature extraction.
    Adapted for 4-channel grayscale input (stacked frames) and 512-dim output.

    This provides strong feature representations at the cost of slightly
    more computation compared to MobileNetV3.

    Input: 4-channel stacked grayscale images (4, 84, 84)
    Output: 512-dimensional feature vector

    Reference:
        He et al., "Deep Residual Learning for Image Recognition" (2016)
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize ResNet18 feature extractor.

        Args:
            input_channels: Number of input channels (default: 4 for stacked frames)
            output_dim: Dimension of output feature vector (default: 512)
            pretrained: Whether to use ImageNet pretrained weights (default: True)
            freeze_backbone: Whether to freeze backbone weights (default: False)
        """
        super(ResNet18FeatureExtractor, self).__init__()

        self.input_channels = input_channels
        self.output_dim = output_dim

        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Adapt first conv layer for 4-channel input
        self.input_projection = nn.Conv2d(
            input_channels, 3,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        nn.init.kaiming_normal_(self.input_projection.weight, mode='fan_out', nonlinearity='relu')

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove final FC layer and replace with custom head
        backbone_output_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Identity()

        # Custom classifier head
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_output_features, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)

        Returns:
            Feature vector of shape (batch_size, 512)
        """
        # Project 4 channels to 3 channels for backbone
        x = self.input_projection(x)

        # Extract features using ResNet18 backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Custom feature head
        x = self.feature_head(x)

        return x

    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_cnn_extractor(
    architecture: str = "mobilenet",
    input_channels: int = 4,
    output_dim: int = 512,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to get CNN feature extractor.

    Args:
        architecture: CNN architecture to use ('nature', 'mobilenet', 'resnet18')
        input_channels: Number of input channels (default: 4)
        output_dim: Output feature dimension (default: 512)
        pretrained: Use pretrained weights (default: True)
        freeze_backbone: Freeze backbone for transfer learning (default: False)

    Returns:
        CNN feature extractor module

    Raises:
        ValueError: If architecture is not recognized
    """
    if architecture.lower() == "nature":
        return NatureCNN(input_channels=input_channels, output_dim=output_dim)
    elif architecture.lower() == "mobilenet":
        return MobileNetV3FeatureExtractor(
            input_channels=input_channels,
            output_dim=output_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif architecture.lower() == "resnet18":
        return ResNet18FeatureExtractor(
            input_channels=input_channels,
            output_dim=output_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: 'nature', 'mobilenet', 'resnet18'"
        )

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

        # Activation function (must be defined before _compute_flat_size)
        self.relu = nn.ReLU()

        # Compute flattened size
        self._compute_flat_size()

        # Fully connected layer
        self.fc = nn.Linear(self.flat_size, feature_dim)

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

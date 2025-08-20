"""
Feature Extractors for DRL with CARLA

This module provides neural network feature extractors for processing
observations from CARLA, including image processing with CNNs and
feature combination for multimodal inputs (images + vector state).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union, List, Optional


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for processing image observations.

    This model follows a similar architecture to the one used in the
    "Deep Reinforcement Learning for Autonomous Driving" paper,
    but simplified for computational efficiency.
    """

    def __init__(self,
                 input_channels: int = 3,
                 input_height: int = 84,
                 input_width: int = 84,
                 output_dim: int = 256):
        """Initialize CNN feature extractor.

        Args:
            input_channels: Number of input channels (3 for RGB)
            input_height: Height of input images
            input_width: Width of input images
            output_dim: Output feature dimension
        """
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.output_dim = output_dim

        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size of CNN
        conv_output_size = self._get_conv_output_size((input_channels, input_height, input_width))

        # FC layers
        self.fc1 = nn.Linear(conv_output_size, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _get_conv_output_size(self, shape: Tuple[int, ...]) -> int:
        """Calculate the output size of the CNN layers.

        Args:
            shape: Input shape (channels, height, width)

        Returns:
            Size of flattened CNN output
        """
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return int(np.prod(x.shape))

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        # Ensure input is float and normalized to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Apply FC layer
        x = F.relu(self.fc1(x))

        return x


class MultimodalFeatureExtractor(nn.Module):
    """
    Multimodal feature extractor that combines image and vector features.

    This model processes both image observations (using a CNN) and vector
    state observations, combining them into a single feature vector.
    """

    def __init__(self,
                 image_channels: int = 3,
                 image_height: int = 84,
                 image_width: int = 84,
                 vector_dim: int = 10,
                 image_feature_dim: int = 256,
                 combined_feature_dim: int = 256):
        """Initialize multimodal feature extractor.

        Args:
            image_channels: Number of input channels for images
            image_height: Height of input images
            image_width: Width of input images
            vector_dim: Dimension of vector observations
            image_feature_dim: Dimension of image features
            combined_feature_dim: Dimension of combined features
        """
        super().__init__()

        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width
        self.vector_dim = vector_dim
        self.image_feature_dim = image_feature_dim
        self.combined_feature_dim = combined_feature_dim

        # Image feature extractor
        self.image_extractor = CNNFeatureExtractor(
            input_channels=image_channels,
            input_height=image_height,
            input_width=image_width,
            output_dim=image_feature_dim
        )

        # Vector feature extractor
        self.vector_extractor = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Combined feature extractor
        self.combined_extractor = nn.Sequential(
            nn.Linear(image_feature_dim + 64, combined_feature_dim),
            nn.ReLU()
        )

        # Initialize weights for vector and combined extractors
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for name, module in self.named_children():
            if name != 'image_extractor':  # Image extractor already initialized
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)

    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            observation: Dictionary with 'image' and 'vector' keys
                - 'image': Tensor of shape (batch_size, channels, height, width)
                - 'vector': Tensor of shape (batch_size, vector_dim)

        Returns:
            Feature tensor of shape (batch_size, combined_feature_dim)
        """
        # Extract image and vector from observation
        image = observation['image']
        vector = observation['vector']

        # Process image
        image_features = self.image_extractor(image)

        # Process vector
        vector_features = self.vector_extractor(vector)

        # Combine features
        combined = torch.cat([image_features, vector_features], dim=1)

        # Extract combined features
        features = self.combined_extractor(combined)

        return features


# Test code
if __name__ == "__main__":
    # Test CNN feature extractor
    cnn = CNNFeatureExtractor()
    test_input = torch.randn(2, 3, 84, 84)
    test_output = cnn(test_input)
    print(f"CNN output shape: {test_output.shape}")

    # Test multimodal feature extractor
    multimodal = MultimodalFeatureExtractor()
    test_observation = {
        'image': torch.randn(2, 3, 84, 84),
        'vector': torch.randn(2, 10)
    }
    test_output = multimodal(test_observation)
    print(f"Multimodal output shape: {test_output.shape}")

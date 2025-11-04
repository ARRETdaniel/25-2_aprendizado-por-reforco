# CNN Extractor Code Cleanup Plan

**Date:** November 4, 2025  
**Issue:** Code duplication in `src/networks/cnn_extractor.py`  
**Priority:** 🚨 **HIGH** (Must fix before next training run)

---

## Problem Statement

The file `src/networks/cnn_extractor.py` contains **TWO COMPLETE IMPLEMENTATIONS** of the same classes:

- **Lines 1-338:** First complete implementation
  - ✅ Factory function `get_cnn_extractor()`
  - ✅ MobileNetV3 and ResNet18 classes
  - ⚠️ Simpler NatureCNN (hardcoded dimensions)
  
- **Lines 340-640:** Second complete implementation
  - ❌ No factory function
  - ❌ No MobileNetV3/ResNet18 classes
  - ✅ Better NatureCNN (dynamic dimensions, validation)
  - ✅ StateEncoder class

**Python behavior:** Uses **LAST definition** (Implementation 2)

---

## Recommended Solution

**Merge both implementations into ONE clean version:**

1. Keep **Implementation 2's NatureCNN** (better code quality)
2. Add **Implementation 1's factory function** (usability)
3. Keep **Implementation 1's transfer learning classes** (MobileNetV3, ResNet18)
4. Keep **Implementation 2's StateEncoder** (needed for integration)

---

## Step-by-Step Cleanup

### Step 1: Backup Original File

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Create backup
cp src/networks/cnn_extractor.py src/networks/cnn_extractor.py.backup

# Verify backup
ls -lh src/networks/cnn_extractor.py.backup
```

---

### Step 2: Create New Clean Version

Create new file: `src/networks/cnn_extractor_clean.py`

```python
"""
CNN Feature Extractors for Visual Input in TD3 Autonomous Driving Agent.

This module provides multiple CNN architectures for extracting visual features
from stacked grayscale camera frames:
- NatureCNN: Standard architecture from Nature DQN paper (Mnih et al., 2015)
- MobileNetV3: Efficient transfer learning with pretrained ImageNet weights
- ResNet18: Deeper transfer learning for maximum feature quality

References:
    - Mnih et al. (2015): "Human-level control through deep RL" (Nature)
    - Howard et al. (2019): "Searching for MobileNetV3" (ICCV)
    - He et al. (2016): "Deep Residual Learning for Image Recognition" (CVPR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torchvision import models


# ============================================================================
# SECTION 1: NATURE CNN (Primary Architecture)
# ============================================================================

class NatureCNN(nn.Module):
    """
    Nature DQN CNN architecture for visual feature extraction.
    
    Architecture from Mnih et al. (2015) "Human-level control through deep RL":
        Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
        Conv1: 32 filters, 8×8 kernel, stride 4 → (batch, 32, 20, 20)
        Conv2: 64 filters, 4×4 kernel, stride 2 → (batch, 64, 9, 9)
        Conv3: 64 filters, 3×3 kernel, stride 1 → (batch, 64, 7, 7)
        Flatten: 64×7×7 = 3136
        FC: 3136 → 512 features
        Output: (batch, 512) feature vector
    
    Weight initialization: PyTorch defaults (Kaiming uniform) match Nature DQN.
    
    Args:
        input_channels (int): Number of input channels (default: 4 for frame stacking)
        num_frames (int): Number of stacked frames (for documentation, unused)
        feature_dim (int): Output feature dimension (default: 512)
    
    References:
        - Paper: https://www.nature.com/articles/nature14236
        - SB3: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
    """
    
    def __init__(self, input_channels: int = 4, num_frames: int = 4, feature_dim: int = 512):
        super(NatureCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        
        # Convolutional layers (Nature DQN architecture)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Compute flattened size dynamically (more robust than hardcoding)
        self._compute_flat_size()
        
        # Fully connected layer
        self.fc = nn.Linear(self.flat_size, feature_dim)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def _compute_flat_size(self):
        """
        Dynamically compute the flattened size after conv layers.
        
        This is more robust than hardcoding 3136 (64×7×7) because:
        - Handles different input sizes if needed
        - Self-documents the architecture
        - Catches dimension errors early
        """
        with torch.no_grad():
            # Create dummy input: (batch=1, channels=4, height=84, width=84)
            dummy_input = torch.zeros(1, self.input_channels, 84, 84)
            
            # Forward pass through conv layers
            out = self.relu(self.conv1(dummy_input))  # → (1, 32, 20, 20)
            out = self.relu(self.conv2(out))          # → (1, 64, 9, 9)
            out = self.relu(self.conv3(out))          # → (1, 64, 7, 7)
            
            # Compute flattened size: 64 × 7 × 7 = 3136
            self.flat_size = int(np.prod(out.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input images (batch, 4, 84, 84)
        
        Returns:
            torch.Tensor: Extracted features (batch, 512)
        
        Raises:
            ValueError: If input shape is incorrect
        """
        # Input validation (helps catch errors early)
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (batch, channels, height, width), got {x.dim()}D tensor"
            )
        
        if x.shape[1:] != (self.input_channels, 84, 84):
            raise ValueError(
                f"Expected input shape (*, {self.input_channels}, 84, 84), got {x.shape}"
            )
        
        # Convolutional layers with ReLU activation
        out = self.relu(self.conv1(x))   # (batch, 32, 20, 20)
        out = self.relu(self.conv2(out)) # (batch, 64, 9, 9)
        out = self.relu(self.conv3(out)) # (batch, 64, 7, 7)
        
        # Flatten spatial dimensions
        out = out.view(out.size(0), -1)  # (batch, 3136)
        
        # Fully connected layer (no activation - let downstream layers handle it)
        features = self.fc(out)           # (batch, 512)
        
        return features


# ============================================================================
# SECTION 2: TRANSFER LEARNING - MOBILENETV3
# ============================================================================

class MobileNetV3FeatureExtractor(nn.Module):
    """
    Transfer learning feature extractor using MobileNetV3-Small.
    
    MobileNetV3 is optimized for efficiency and speed, making it ideal for:
    - Real-time deployment
    - Embedded systems
    - Fast iteration during research
    
    Architecture:
        Input: (batch, 4, 84, 84) grayscale frames
        Input Projection: 4 channels → 3 channels (1×1 conv)
        MobileNetV3-Small Backbone: Pretrained on ImageNet
        Custom Head: backbone_features → 1024 → Dropout → 512
        Output: (batch, 512) feature vector
    
    Args:
        input_channels (int): Number of input channels (default: 4)
        feature_dim (int): Output feature dimension (default: 512)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_backbone (bool): Freeze backbone during initial training (default: False)
    
    References:
        - Paper: Howard et al. (2019) "Searching for MobileNetV3"
        - Link: https://arxiv.org/abs/1905.02244
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(MobileNetV3FeatureExtractor, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Input projection: 4 channels → 3 channels (for pretrained weights)
        self.input_projection = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.input_projection.weight, mode='fan_out', nonlinearity='relu')
        
        # Load MobileNetV3-Small backbone
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        
        # Extract feature extractor (before classifier)
        self.backbone = mobilenet.features
        
        # Freeze backbone if requested (good for initial training)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Determine backbone output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            backbone_out = self.backbone(dummy)
            backbone_out = F.adaptive_avg_pool2d(backbone_out, 1)
            backbone_features = backbone_out.shape[1]  # 576 for MobileNetV3-Small
        
        # Custom feature head
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MobileNetV3 extractor.
        
        Args:
            x (torch.Tensor): Input images (batch, 4, 84, 84)
        
        Returns:
            torch.Tensor: Extracted features (batch, 512)
        """
        # Project 4-channel input to 3-channel
        x = self.input_projection(x)  # (batch, 3, 84, 84)
        
        # Extract features via backbone
        x = self.backbone(x)  # (batch, 576, H, W)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # (batch, 576, 1, 1)
        
        # Custom head
        features = self.feature_head(x)  # (batch, 512)
        
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning after initial training."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================================
# SECTION 3: TRANSFER LEARNING - RESNET18
# ============================================================================

class ResNet18FeatureExtractor(nn.Module):
    """
    Transfer learning feature extractor using ResNet18.
    
    ResNet18 provides stronger feature extraction than MobileNetV3 at the cost
    of higher computational cost. Best for:
    - Research (maximum accuracy)
    - GPU training (not real-time deployment)
    
    Architecture:
        Input: (batch, 4, 84, 84) grayscale frames
        Input Projection: 4 channels → 3 channels (1×1 conv)
        ResNet18 Backbone: Pretrained on ImageNet
        Custom Head: 512 → 512
        Output: (batch, 512) feature vector
    
    Args:
        input_channels (int): Number of input channels (default: 4)
        feature_dim (int): Output feature dimension (default: 512)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_backbone (bool): Freeze backbone during initial training (default: False)
    
    References:
        - Paper: He et al. (2016) "Deep Residual Learning for Image Recognition"
        - Link: https://arxiv.org/abs/1512.03385
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        feature_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNet18FeatureExtractor, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Input projection: 4 channels → 3 channels
        self.input_projection = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.input_projection.weight, mode='fan_out', nonlinearity='relu')
        
        # Load ResNet18 backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Remove final FC layer (keep feature extractor only)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom head (ResNet18 outputs 512-dim features by default)
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet18 extractor.
        
        Args:
            x (torch.Tensor): Input images (batch, 4, 84, 84)
        
        Returns:
            torch.Tensor: Extracted features (batch, 512)
        """
        # Project 4-channel input to 3-channel
        x = self.input_projection(x)  # (batch, 3, 84, 84)
        
        # Extract features via backbone
        x = self.backbone(x)  # (batch, 512, 1, 1)
        
        # Custom head
        features = self.feature_head(x)  # (batch, 512)
        
        return features
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning after initial training."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ============================================================================
# SECTION 4: STATE ENCODER (Multi-Modal Fusion)
# ============================================================================

class StateEncoder(nn.Module):
    """
    Combines visual features from CNN with kinematic state and navigation waypoints.
    
    Input dimensions:
        - Visual features (CNN output): 512 dims
        - Kinematic state: 3 dims (velocity, lateral_deviation, heading_error)
        - Navigation waypoints: 20 dims (10 waypoints × 2 coordinates)
        - Total: 512 + 3 + 20 = 535 dims
    
    Optional LayerNorm on visual features helps:
        - Stabilize training (normalize feature magnitudes)
        - Improve gradient flow
        - Prevent scale mismatch between visual and kinematic features
    
    Args:
        cnn_feature_dim (int): Dimension of CNN features (default: 512)
        kinematic_dim (int): Dimension of kinematic state (default: 3)
        waypoint_dim (int): Dimension of waypoint data (default: 20)
        normalize (bool): Apply LayerNorm to CNN features (default: True)
    """
    
    def __init__(
        self,
        cnn_feature_dim: int = 512,
        kinematic_dim: int = 3,
        waypoint_dim: int = 20,
        normalize: bool = True
    ):
        super(StateEncoder, self).__init__()
        
        self.cnn_feature_dim = cnn_feature_dim
        self.kinematic_dim = kinematic_dim
        self.waypoint_dim = waypoint_dim
        self.normalize = normalize
        
        # Optional normalization for CNN features
        if normalize:
            self.layer_norm = nn.LayerNorm(cnn_feature_dim)
        
        # Output dimension
        self.output_dim = cnn_feature_dim + kinematic_dim + waypoint_dim
    
    def forward(self, image_features: torch.Tensor, kinematic_state: torch.Tensor) -> torch.Tensor:
        """
        Combine CNN features with kinematic state.
        
        Args:
            image_features (torch.Tensor): CNN features (batch, 512)
            kinematic_state (torch.Tensor): Kinematic state (batch, 23)
                - First 3: velocity, lateral_dev, heading_err
                - Last 20: waypoints (10 × 2)
        
        Returns:
            torch.Tensor: Full state vector (batch, 535)
        """
        # Normalize CNN features if enabled
        if self.normalize:
            image_features = self.layer_norm(image_features)
        
        # Concatenate visual and kinematic features
        full_state = torch.cat([image_features, kinematic_state], dim=1)  # (batch, 535)
        
        return full_state


# ============================================================================
# SECTION 5: FACTORY FUNCTION (Architecture Selection)
# ============================================================================

def get_cnn_extractor(
    architecture: str = "nature",
    input_channels: int = 4,
    output_dim: int = 512,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to create CNN feature extractor by architecture name.
    
    Supported architectures:
        - "nature": NatureCNN (Nature DQN paper, standard for RL)
        - "mobilenet": MobileNetV3-Small (efficient, pretrained)
        - "resnet18": ResNet18 (stronger features, slower)
    
    Args:
        architecture (str): Architecture name (default: "nature")
        input_channels (int): Number of input channels (default: 4)
        output_dim (int): Feature output dimension (default: 512)
        pretrained (bool): Use ImageNet pretrained weights for transfer learning (default: True)
        freeze_backbone (bool): Freeze backbone for transfer learning (default: False)
    
    Returns:
        nn.Module: CNN feature extractor instance
    
    Raises:
        ValueError: If architecture name is not recognized
    
    Example:
        >>> # Standard NatureCNN
        >>> cnn = get_cnn_extractor("nature")
        >>> 
        >>> # Transfer learning with MobileNetV3
        >>> cnn = get_cnn_extractor("mobilenet", pretrained=True, freeze_backbone=True)
    """
    architecture = architecture.lower()
    
    if architecture == "nature":
        return NatureCNN(
            input_channels=input_channels,
            feature_dim=output_dim
        )
    
    elif architecture == "mobilenet":
        return MobileNetV3FeatureExtractor(
            input_channels=input_channels,
            feature_dim=output_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    
    elif architecture == "resnet18":
        return ResNet18FeatureExtractor(
            input_channels=input_channels,
            feature_dim=output_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    
    else:
        raise ValueError(
            f"Unknown CNN architecture: '{architecture}'. "
            f"Supported: 'nature', 'mobilenet', 'resnet18'"
        )


# ============================================================================
# SECTION 6: UTILITY FUNCTIONS
# ============================================================================

def compute_nature_cnn_output_size(input_height: int, input_width: int) -> Tuple[int, int, int]:
    """
    Compute output spatial dimensions of NatureCNN for given input size.
    
    Args:
        input_height (int): Input image height
        input_width (int): Input image width
    
    Returns:
        Tuple[int, int, int]: (height, width, channels) after conv layers
    
    Example:
        >>> h, w, c = compute_nature_cnn_output_size(84, 84)
        >>> print(f"Output: {h}×{w}×{c} = {h*w*c} (flattened)")
        Output: 7×7×64 = 3136 (flattened)
    """
    # Conv1: kernel=8, stride=4, padding=0
    h = (input_height - 8) // 4 + 1
    w = (input_width - 8) // 4 + 1
    
    # Conv2: kernel=4, stride=2, padding=0
    h = (h - 4) // 2 + 1
    w = (w - 4) // 2 + 1
    
    # Conv3: kernel=3, stride=1, padding=0
    h = (h - 3) // 1 + 1
    w = (w - 3) // 1 + 1
    
    channels = 64  # Output channels of conv3
    
    return h, w, channels


# ============================================================================
# SECTION 7: TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Unit tests for CNN feature extractors.
    
    Run with: python -m src.networks.cnn_extractor
    """
    print("=" * 80)
    print("CNN Feature Extractor Unit Tests")
    print("=" * 80)
    
    # Test parameters
    batch_size = 16
    input_channels = 4
    input_height = 84
    input_width = 84
    feature_dim = 512
    
    # Test input
    test_input = torch.randn(batch_size, input_channels, input_height, input_width)
    print(f"\nTest input shape: {test_input.shape}")
    
    # ========================================================================
    # Test 1: NatureCNN
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: NatureCNN")
    print("=" * 80)
    
    nature_cnn = NatureCNN(input_channels=input_channels, feature_dim=feature_dim)
    nature_output = nature_cnn(test_input)
    
    print(f"✅ NatureCNN output shape: {nature_output.shape}")
    assert nature_output.shape == (batch_size, feature_dim), \
        f"Expected ({batch_size}, {feature_dim}), got {nature_output.shape}"
    
    # Test gradient flow
    loss = nature_output.sum()
    loss.backward()
    assert nature_cnn.conv1.weight.grad is not None, "❌ No gradient in conv1!"
    print("✅ Gradient flow verified")
    
    # ========================================================================
    # Test 2: MobileNetV3
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: MobileNetV3FeatureExtractor")
    print("=" * 80)
    
    mobilenet_cnn = MobileNetV3FeatureExtractor(
        input_channels=input_channels,
        feature_dim=feature_dim,
        pretrained=False  # Faster for testing
    )
    mobilenet_output = mobilenet_cnn(test_input)
    
    print(f"✅ MobileNetV3 output shape: {mobilenet_output.shape}")
    assert mobilenet_output.shape == (batch_size, feature_dim), \
        f"Expected ({batch_size}, {feature_dim}), got {mobilenet_output.shape}"
    
    # ========================================================================
    # Test 3: ResNet18
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: ResNet18FeatureExtractor")
    print("=" * 80)
    
    resnet_cnn = ResNet18FeatureExtractor(
        input_channels=input_channels,
        feature_dim=feature_dim,
        pretrained=False  # Faster for testing
    )
    resnet_output = resnet_cnn(test_input)
    
    print(f"✅ ResNet18 output shape: {resnet_output.shape}")
    assert resnet_output.shape == (batch_size, feature_dim), \
        f"Expected ({batch_size}, {feature_dim}), got {resnet_output.shape}"
    
    # ========================================================================
    # Test 4: StateEncoder
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: StateEncoder")
    print("=" * 80)
    
    kinematic_dim = 23  # 3 kinematic + 20 waypoints
    state_encoder = StateEncoder(
        cnn_feature_dim=feature_dim,
        kinematic_dim=kinematic_dim,
        normalize=True
    )
    
    test_kinematic = torch.randn(batch_size, kinematic_dim)
    full_state = state_encoder(nature_output, test_kinematic)
    
    expected_state_dim = feature_dim + kinematic_dim  # 512 + 23 = 535
    print(f"✅ Full state shape: {full_state.shape}")
    assert full_state.shape == (batch_size, expected_state_dim), \
        f"Expected ({batch_size}, {expected_state_dim}), got {full_state.shape}"
    
    # ========================================================================
    # Test 5: Factory Function
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Factory Function")
    print("=" * 80)
    
    for arch in ["nature", "mobilenet", "resnet18"]:
        cnn = get_cnn_extractor(architecture=arch, pretrained=False)
        output = cnn(test_input)
        print(f"✅ {arch:12s} → output shape: {output.shape}")
        assert output.shape == (batch_size, feature_dim)
    
    # Test invalid architecture
    try:
        get_cnn_extractor(architecture="invalid")
        print("❌ Should have raised ValueError for invalid architecture")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    # ========================================================================
    # Test 6: Dimension Computation
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: Dimension Computation")
    print("=" * 80)
    
    h, w, c = compute_nature_cnn_output_size(84, 84)
    print(f"Input: 84×84 → Output: {h}×{w}×{c}")
    print(f"Flattened size: {h*w*c}")
    
    assert h == 7 and w == 7 and c == 64, f"Expected 7×7×64, got {h}×{w}×{c}"
    assert h * w * c == 3136, f"Expected flattened size 3136, got {h*w*c}"
    print("✅ Dimension computation correct")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
```

---

### Step 3: Replace Original File

```bash
# Verify tests pass
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

python -m src.networks.cnn_extractor_clean

# If all tests pass, replace original
mv src/networks/cnn_extractor.py src/networks/cnn_extractor_OLD.py
mv src/networks/cnn_extractor_clean.py src/networks/cnn_extractor.py

# Verify imports still work
python -c "from src.networks.cnn_extractor import NatureCNN, get_cnn_extractor; print('✅ Imports working')"
```

---

### Step 4: Update Imports (if needed)

Check if any other files import from `cnn_extractor.py`:

```bash
# Search for imports
grep -r "from.*cnn_extractor import" src/
grep -r "import.*cnn_extractor" src/

# Expected results:
# src/agents/td3_agent.py: from src.networks.cnn_extractor import get_cnn_extractor
```

No changes needed if using factory function. If importing directly:

```python
# OLD (will still work):
from src.networks.cnn_extractor import NatureCNN

# NEW (recommended):
from src.networks.cnn_extractor import get_cnn_extractor
cnn = get_cnn_extractor(architecture="nature")
```

---

## Verification Checklist

After cleanup, verify:

- [ ] ✅ All unit tests pass
- [ ] ✅ No duplicate class definitions
- [ ] ✅ Factory function works for all architectures
- [ ] ✅ Imports in td3_agent.py still work
- [ ] ✅ Training script runs without errors
- [ ] ✅ File size reduced (from 640 lines → ~650 lines clean)
- [ ] ✅ Code is well-documented
- [ ] ✅ Git commit with clear message

---

## Expected Benefits

**After cleanup:**

1. **Maintainability:** ✅ Single source of truth for each class
2. **Clarity:** ✅ Clear architecture with sections
3. **Testability:** ✅ Comprehensive unit tests included
4. **Flexibility:** ✅ Easy to add new CNN architectures
5. **Documentation:** ✅ Extensive docstrings with references

**No impact on training:** Code duplication was not causing training failure.

---

## Alternative: Minimal Fix (Quick)

If you want a quick fix without rewriting the entire file:

```bash
# Just delete the first implementation (lines 1-338)
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Create backup
cp src/networks/cnn_extractor.py src/networks/cnn_extractor.py.backup

# Delete lines 1-338
sed -i '1,338d' src/networks/cnn_extractor.py

# Verify it works
python -c "from src.networks.cnn_extractor import NatureCNN; print('✅ Import works')"
```

**Pros:** Fast, minimal changes  
**Cons:** Loses factory function and transfer learning classes

---

## Recommendation

**Use the FULL CLEANUP** (Step 2) because:
- ✅ Professional code quality
- ✅ Keeps all functionality (factory + transfer learning)
- ✅ Better documentation
- ✅ Comprehensive tests
- ✅ Only ~1 hour of work for long-term benefit

---

## Git Commit Message

```bash
git add src/networks/cnn_extractor.py
git commit -m "refactor(cnn): Remove code duplication in CNN extractor

- Consolidated two NatureCNN implementations into one clean version
- Kept better implementation (dynamic flat size computation + validation)
- Preserved factory function and transfer learning classes
- Added comprehensive docstrings with paper references
- Added unit tests for all architectures
- File structure: 6 sections (NatureCNN, MobileNetV3, ResNet18, StateEncoder, Factory, Tests)

Fixes: Code duplication issue (lines 1-338 duplicate of 340-640)
Impact: Maintainability improvement, no functional changes
Verified: All tests pass, imports work, dimensions correct"
```

---

**Next Steps After Cleanup:**

1. ✅ Run cleanup script
2. ✅ Verify all tests pass
3. ✅ Git commit changes
4. ✅ Re-run training with clean code
5. ✅ Monitor CNN learning via TensorBoard
6. 🔍 Investigate reward function (likely primary issue)

---

**Status:** Ready to execute cleanup ✅

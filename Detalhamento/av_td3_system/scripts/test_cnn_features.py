#!/usr/bin/env python3
"""
CNN Feature Extractor Validation Tests

This script validates the NatureCNN feature extractor to ensure:
1. CNN produces non-zero, informative features (not constant/dead)
2. Device placement is consistent between CNN and TD3 agent
3. Input normalization range is correct ([-1, 1] expected)
4. Feature statistics are reasonable (no NaN, no extreme values)
5. CNN architecture matches Nature DQN specification

Usage:
    python3 scripts/test_cnn_features.py --device cpu
    python3 scripts/test_cnn_features.py --device cuda --num-samples 100

Author: Daniel Terra
Date: 2025-01-28
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.networks.cnn_extractor import NatureCNN
from src.agents.td3_agent import TD3Agent


class CNNFeatureValidator:
    """
    Validates CNN feature extractor for potential issues.

    Tests:
    1. Architecture validation (matches Nature DQN)
    2. Weight initialization check
    3. Feature quality assessment (non-zero, non-constant)
    4. Device placement consistency
    5. Normalization range compatibility
    """

    def __init__(self, device='cpu'):
        """
        Initialize validator.

        Args:
            device: Device to test on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        print(f"\n{'='*70}")
        print("CNN FEATURE EXTRACTOR VALIDATION")
        print(f"{'='*70}")
        print(f"Device: {self.device}")

    def test_architecture(self):
        """Test 1: Validate CNN architecture matches Nature DQN spec."""
        print(f"\n[TEST 1] Architecture Validation")
        print("-" * 70)

        cnn = NatureCNN(input_channels=4, feature_dim=512).to(self.device)

        # Check layer structure
        expected_layers = {
            'conv1': (4, 32, 8, 4),    # (in_channels, out_channels, kernel, stride)
            'conv2': (32, 64, 4, 2),
            'conv3': (64, 64, 3, 1),
        }

        errors = []

        # Validate Conv1
        if cnn.conv1.in_channels != expected_layers['conv1'][0]:
            errors.append(f"Conv1 in_channels: expected {expected_layers['conv1'][0]}, got {cnn.conv1.in_channels}")
        if cnn.conv1.out_channels != expected_layers['conv1'][1]:
            errors.append(f"Conv1 out_channels: expected {expected_layers['conv1'][1]}, got {cnn.conv1.out_channels}")
        if cnn.conv1.kernel_size != (expected_layers['conv1'][2], expected_layers['conv1'][2]):
            errors.append(f"Conv1 kernel_size: expected {expected_layers['conv1'][2]}, got {cnn.conv1.kernel_size}")
        if cnn.conv1.stride != (expected_layers['conv1'][3], expected_layers['conv1'][3]):
            errors.append(f"Conv1 stride: expected {expected_layers['conv1'][3]}, got {cnn.conv1.stride}")

        # Validate Conv2
        if cnn.conv2.in_channels != expected_layers['conv2'][0]:
            errors.append(f"Conv2 in_channels: expected {expected_layers['conv2'][0]}, got {cnn.conv2.in_channels}")
        if cnn.conv2.out_channels != expected_layers['conv2'][1]:
            errors.append(f"Conv2 out_channels: expected {expected_layers['conv2'][1]}, got {cnn.conv2.out_channels}")

        # Validate Conv3
        if cnn.conv3.in_channels != expected_layers['conv3'][0]:
            errors.append(f"Conv3 in_channels: expected {expected_layers['conv3'][0]}, got {cnn.conv3.in_channels}")
        if cnn.conv3.out_channels != expected_layers['conv3'][1]:
            errors.append(f"Conv3 out_channels: expected {expected_layers['conv3'][1]}, got {cnn.conv3.out_channels}")

        # Validate FC layer
        if cnn.fc.out_features != 512:
            errors.append(f"FC out_features: expected 512, got {cnn.fc.out_features}")

        # Calculate expected flattened size
        # Input: 84×84
        # After Conv1 (k=8, s=4): (84-8)/4+1 = 20
        # After Conv2 (k=4, s=2): (20-4)/2+1 = 9
        # After Conv3 (k=3, s=1): (9-3)/1+1 = 7
        # Flattened: 64 × 7 × 7 = 3136
        expected_flat_size = 64 * 7 * 7
        if cnn.fc.in_features != expected_flat_size:
            errors.append(f"FC in_features: expected {expected_flat_size}, got {cnn.fc.in_features}")

        if errors:
            print("❌ ARCHITECTURE MISMATCH:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print("✅ Architecture matches Nature DQN specification")
            print(f"   - Conv1: {cnn.conv1.in_channels}→{cnn.conv1.out_channels} (kernel={cnn.conv1.kernel_size[0]}, stride={cnn.conv1.stride[0]})")
            print(f"   - Conv2: {cnn.conv2.in_channels}→{cnn.conv2.out_channels} (kernel={cnn.conv2.kernel_size[0]}, stride={cnn.conv2.stride[0]})")
            print(f"   - Conv3: {cnn.conv3.in_channels}→{cnn.conv3.out_channels} (kernel={cnn.conv3.kernel_size[0]}, stride={cnn.conv3.stride[0]})")
            print(f"   - FC: {cnn.fc.in_features}→{cnn.fc.out_features}")
            return True

    def test_weight_initialization(self):
        """Test 2: Check if weights are properly initialized (not zero)."""
        print(f"\n[TEST 2] Weight Initialization Check")
        print("-" * 70)

        cnn = NatureCNN(input_channels=4, feature_dim=512).to(self.device)

        # Check if weights are not all zeros or constant
        layers = [cnn.conv1, cnn.conv2, cnn.conv3, cnn.fc]
        layer_names = ['Conv1', 'Conv2', 'Conv3', 'FC']

        all_good = True
        for layer, name in zip(layers, layer_names):
            weight = layer.weight.data

            # Check if weights are zero
            if torch.all(weight == 0):
                print(f"❌ {name} weights are ALL ZERO!")
                all_good = False
                continue

            # Check if weights are constant (std dev near zero)
            if weight.std() < 1e-6:
                print(f"❌ {name} weights are CONSTANT (std={weight.std():.2e})")
                all_good = False
                continue

            print(f"✅ {name}: mean={weight.mean():.4f}, std={weight.std():.4f}, range=[{weight.min():.4f}, {weight.max():.4f}]")

        return all_good

    def test_feature_quality(self, num_samples=50):
        """Test 3: Validate features are informative (non-zero, non-constant)."""
        print(f"\n[TEST 3] Feature Quality Assessment ({num_samples} samples)")
        print("-" * 70)

        cnn = NatureCNN(input_channels=4, feature_dim=512).to(self.device)
        cnn.eval()

        # Generate random inputs simulating preprocessed images [-1, 1]
        # This mimics the normalized output from sensors.py
        features_list = []

        with torch.no_grad():
            for i in range(num_samples):
                # Random input (4, 84, 84) normalized to [-1, 1]
                dummy_input = torch.randn(1, 4, 84, 84).to(self.device)
                dummy_input = torch.clamp(dummy_input, -1, 1)  # Ensure [-1, 1] range

                features = cnn(dummy_input)  # (1, 512)
                features_list.append(features.cpu().numpy())

        # Stack all features
        all_features = np.vstack(features_list)  # (num_samples, 512)

        # Statistical analysis
        mean_feature = all_features.mean(axis=0)  # (512,)
        std_feature = all_features.std(axis=0)    # (512,)

        # Check for issues
        issues = []

        # 1. Check for NaN
        if np.isnan(all_features).any():
            issues.append(f"⚠️  Features contain NaN values!")

        # 2. Check for Inf
        if np.isinf(all_features).any():
            issues.append(f"⚠️  Features contain Inf values!")

        # 3. Check if features are all zero
        if np.all(all_features == 0):
            issues.append(f"❌ Features are ALL ZERO!")

        # 4. Check if features are constant across samples
        overall_std = all_features.std()
        if overall_std < 0.01:
            issues.append(f"❌ Features are nearly CONSTANT (std={overall_std:.4f})")

        # 5. Count dead neurons (neurons with zero activation across all samples)
        dead_neurons = (std_feature < 1e-6).sum()
        dead_percent = (dead_neurons / 512) * 100
        if dead_percent > 10:
            issues.append(f"⚠️  {dead_neurons} dead neurons ({dead_percent:.1f}%)")

        # 6. Check for extreme values (potential gradient explosion)
        if np.abs(all_features).max() > 1000:
            issues.append(f"⚠️  Extreme feature values detected (max={np.abs(all_features).max():.2f})")

        # Print results
        if issues:
            print("❌ FEATURE QUALITY ISSUES:")
            for issue in issues:
                print(f"   {issue}")
            print()
        else:
            print("✅ Feature quality is good")

        print(f"📊 Feature Statistics:")
        print(f"   - Mean: {all_features.mean():.4f}")
        print(f"   - Std Dev: {all_features.std():.4f}")
        print(f"   - Min: {all_features.min():.4f}")
        print(f"   - Max: {all_features.max():.4f}")
        print(f"   - Dead neurons: {dead_neurons}/512 ({dead_percent:.1f}%)")
        print(f"   - Active neurons: {512 - dead_neurons}/512 ({100-dead_percent:.1f}%)")

        return len(issues) == 0

    def test_device_consistency(self):
        """Test 4: Verify CNN and TD3 agent are on the same device."""
        print(f"\n[TEST 4] Device Placement Consistency")
        print("-" * 70)

        # Initialize TD3 agent
        agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config_path="config/td3_config.yaml",
            device=self.device.type
        )

        # Initialize CNN
        cnn = NatureCNN(input_channels=4, feature_dim=512).to(self.device)

        # Check device placement
        cnn_device = next(cnn.parameters()).device
        agent_device = agent.device

        if cnn_device == agent_device:
            print(f"✅ Device consistency: CNN and Agent both on {cnn_device}")
            return True
        else:
            print(f"❌ DEVICE MISMATCH:")
            print(f"   - CNN on: {cnn_device}")
            print(f"   - Agent on: {agent_device}")
            print(f"   - This will cause runtime errors during training!")
            return False

    def test_normalization_compatibility(self):
        """Test 5: Verify CNN works with [-1, 1] normalized inputs."""
        print(f"\n[TEST 5] Normalization Range Compatibility")
        print("-" * 70)

        cnn = NatureCNN(input_channels=4, feature_dim=512).to(self.device)
        cnn.eval()

        # Test with different input ranges
        test_ranges = [
            ("[-1, 1] (expected)", -1.0, 1.0),
            ("[0, 1] (wrong)", 0.0, 1.0),
            ("[0, 255] (raw)", 0.0, 255.0),
        ]

        results = []

        with torch.no_grad():
            for range_name, min_val, max_val in test_ranges:
                # Create dummy input in specified range
                dummy_input = torch.rand(1, 4, 84, 84).to(self.device)
                dummy_input = dummy_input * (max_val - min_val) + min_val

                try:
                    features = cnn(dummy_input)

                    # Check if features are reasonable
                    has_nan = torch.isnan(features).any().item()
                    has_inf = torch.isinf(features).any().item()
                    std_val = features.std().item()

                    if has_nan or has_inf:
                        results.append((range_name, "❌ NaN/Inf", None))
                    elif std_val < 0.01:
                        results.append((range_name, "⚠️  Near-constant", std_val))
                    else:
                        results.append((range_name, "✅ Valid", std_val))

                except Exception as e:
                    results.append((range_name, f"❌ Error: {str(e)}", None))

        # Print results
        for range_name, status, std_val in results:
            if std_val is not None:
                print(f"   {status} {range_name}: std={std_val:.4f}")
            else:
                print(f"   {status} {range_name}")

        # Check if [-1, 1] works correctly
        expected_status = results[0][1]
        return expected_status.startswith("✅")

    def run_all_tests(self, num_samples=50):
        """Run all validation tests."""
        print(f"\nRunning all validation tests...")

        results = {
            "Architecture": self.test_architecture(),
            "Initialization": self.test_weight_initialization(),
            "Feature Quality": self.test_feature_quality(num_samples),
            "Device Consistency": self.test_device_consistency(),
            "Normalization": self.test_normalization_compatibility(),
        }

        # Summary
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}")

        passed = sum(results.values())
        total = len(results)

        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{status} - {test_name}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("✅ ALL TESTS PASSED - CNN is ready for training!")
            return True
        else:
            print("❌ SOME TESTS FAILED - Review issues before training!")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate NatureCNN feature extractor"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to test on (default: cpu)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of samples for feature quality test (default: 50)'
    )

    args = parser.parse_args()

    # Run validation
    validator = CNNFeatureValidator(device=args.device)
    success = validator.run_all_tests(num_samples=args.num_samples)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

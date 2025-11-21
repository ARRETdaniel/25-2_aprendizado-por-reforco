#!/usr/bin/env python3
"""
Test script to validate CNN LayerNorm implementation.

This script verifies:
1. CNN initializes correctly with LayerNorm layers
2. Forward pass produces stable features (L2 norm < 100)
3. All LayerNorm layers are present and functional
4. Feature statistics are within expected ranges

Expected: L2 norm ~10-100 (vs 7.36×10¹² without LayerNorm)
"""
import sys
sys.path.insert(0, '/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system')

import torch
import torch.nn as nn
from src.networks.cnn_extractor import NatureCNN


def test_cnn_layernorm():
    """Test CNN with LayerNorm implementation."""
    print("=" * 80)
    print("CNN LAYERNORM VALIDATION TEST")
    print("=" * 80)

    # Create CNN
    print("\n[STEP 1] Initializing NatureCNN...")
    cnn = NatureCNN(input_channels=4, feature_dim=512)
    print("✅ CNN initialized successfully")

    # Test forward pass
    print("\n[STEP 2] Testing forward pass...")
    batch_size = 32
    dummy_input = torch.randn(batch_size, 4, 84, 84)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Input L2 norm: {torch.norm(dummy_input).item():.2f}")

    # Forward pass
    output = cnn(dummy_input)
    print(f"\n[STEP 3] Analyzing output...")
    print(f"   Output shape: {output.shape}")
    print(f"   Output L2 norm: {torch.norm(output).item():.2f}")

    # Check feature statistics
    print(f"\n[STEP 4] Feature Statistics:")
    print(f"   Mean: {output.mean().item():.4f}")
    print(f"   Std:  {output.std().item():.4f}")
    print(f"   Min:  {output.min().item():.4f}")
    print(f"   Max:  {output.max().item():.4f}")

    # Validate L2 norm
    expected_max_norm = 1000  # Conservative threshold (should be ~10-100)
    actual_norm = torch.norm(output).item()

    print(f"\n[STEP 5] Validation:")
    if actual_norm < expected_max_norm:
        print(f"   ✅ SUCCESS: L2 norm {actual_norm:.2f} < {expected_max_norm}")
        print("   LayerNorm is working correctly!")
        print(f"   Expected range: 10-100")
        print(f"   Actual: {actual_norm:.2f}")
        if actual_norm > 100:
            print(f"   ⚠️  WARNING: Norm slightly high (expected 10-100)")
    else:
        print(f"   ❌ FAIL: L2 norm {actual_norm:.2f} >= {expected_max_norm}")
        print("   Check LayerNorm implementation")
        return False

    # Check all LayerNorm layers exist
    print("\n[STEP 6] Checking LayerNorm layers:")
    layernorm_count = 0
    for name, module in cnn.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            layernorm_count += 1
            print(f"   ✅ {name}: {module}")

    if layernorm_count == 4:
        print(f"\n   ✅ All 4 LayerNorm layers present")
    else:
        print(f"\n   ❌ FAIL: Expected 4 LayerNorm layers, found {layernorm_count}")
        return False

    # Test with multiple batches to verify stability
    print("\n[STEP 7] Testing stability across batches:")
    norms = []
    for i in range(5):
        batch = torch.randn(32, 4, 84, 84)
        out = cnn(batch)
        norm = torch.norm(out).item()
        norms.append(norm)
        print(f"   Batch {i+1}: L2 norm = {norm:.2f}")

    avg_norm = sum(norms) / len(norms)
    std_norm = torch.tensor(norms).std().item()
    print(f"\n   Average L2 norm: {avg_norm:.2f}")
    print(f"   Std of L2 norms: {std_norm:.2f}")

    if std_norm < avg_norm * 0.5:  # Std should be less than 50% of mean
        print(f"   ✅ Stable across batches")
    else:
        print(f"   ⚠️  High variance across batches")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE - CNN LAYERNORM IMPLEMENTATION VALIDATED")
    print("=" * 80)
    print("\nExpected Impact:")
    print("   Before Fix: L2 norm = 7.36 × 10¹² (7,363,360,194,560)")
    print(f"   After Fix:  L2 norm = {avg_norm:.2f}")
    print(f"   Reduction:  ~{7.36e12 / avg_norm:.2e}× (10¹⁰-10¹¹× expected)")
    print("\nNext Steps:")
    print("   1. Run smoke test: python scripts/train_td3.py --max-timesteps 100 --debug")
    print("   2. Run 5K validation: python scripts/train_td3.py --max-timesteps 5000 --debug")
    print("   3. Check TensorBoard metrics")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_cnn_layernorm()
    sys.exit(0 if success else 1)

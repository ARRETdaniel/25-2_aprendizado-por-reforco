"""
Test CNN Diagnostics System

Simple test script to verify CNN diagnostics are working correctly.
Tests gradient capture, weight tracking, and feature statistics.

Usage:
    python tests/test_cnn_diagnostics.py

Author: Daniel Terra
Date: 2025-01-XX
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np

from src.utils.cnn_diagnostics import CNNDiagnostics, quick_check_cnn_learning


def create_simple_cnn():
    """Create a simple CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # 84x84 → 42x42
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 42x42 → 21x21
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 21 * 21, 512),
        nn.ReLU()
    )


def test_diagnostics_basic():
    """Test basic diagnostics functionality."""
    print("\n" + "="*70)
    print("TEST 1: Basic Diagnostics Functionality")
    print("="*70)

    # Create CNN
    cnn = create_simple_cnn()
    cnn.train()

    # Create diagnostics
    diagnostics = CNNDiagnostics(cnn)
    print("✅ CNNDiagnostics initialized")

    # Create dummy data
    images = torch.randn(8, 4, 84, 84)
    target = torch.randn(8, 512)

    # Forward pass
    features = cnn(images)
    diagnostics.capture_features(features)
    print("✅ Features captured")

    # Backward pass
    loss = ((features - target) ** 2).mean()
    loss.backward()

    # Capture gradients
    grad_norms = diagnostics.capture_gradients()
    print(f"✅ Gradients captured: {len(grad_norms)} layers")

    # Optimizer step
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    optimizer.step()

    # Capture weights
    weight_stats = diagnostics.capture_weights()
    print(f"✅ Weights captured: {len(weight_stats)} layers")

    # Get summary
    summary = diagnostics.get_summary()
    print(f"✅ Summary generated")
    print(f"   Gradient flow OK: {summary['gradient_flow_ok']}")
    print(f"   Weights updating: {summary['weights_updating']}")

    return summary['gradient_flow_ok'] and summary['weights_updating']


def test_diagnostics_multi_step():
    """Test diagnostics over multiple training steps."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Step Training Monitoring")
    print("="*70)

    # Create CNN
    cnn = create_simple_cnn()
    cnn.train()

    # Create diagnostics
    diagnostics = CNNDiagnostics(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Run multiple training steps
    n_steps = 10
    for step in range(n_steps):
        # Forward pass
        images = torch.randn(8, 4, 84, 84)
        target = torch.randn(8, 512)
        features = cnn(images)

        # Backward pass
        loss = ((features - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()

        # Capture diagnostics
        diagnostics.capture_gradients()
        diagnostics.capture_features(features)
        optimizer.step()
        diagnostics.capture_weights()

    print(f"✅ Completed {n_steps} training steps")

    # Get summary
    summary = diagnostics.get_summary(last_n=n_steps)
    print(f"   Total captures: {summary['n_captures']}")

    # Check that metrics are being tracked
    has_stats = len(summary['recent_stats']) > 0
    print(f"   Recent stats available: {has_stats}")

    return has_stats


def test_gradient_flow_detection():
    """Test gradient flow detection (with and without gradients)."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow Detection")
    print("="*70)

    # Create CNN
    cnn = create_simple_cnn()
    cnn.train()

    # Test WITH gradients
    print("\n[A] Training mode (gradients enabled):")
    images = torch.randn(8, 4, 84, 84)
    target = torch.randn(8, 512)
    features = cnn(images)
    loss = ((features - target) ** 2).mean()
    loss.backward()

    is_learning, msg = quick_check_cnn_learning(cnn)
    print(msg)
    if not is_learning:
        print("❌ FAILED: Should detect gradients in training mode")
        return False

    # Test WITHOUT gradients
    print("\n[B] Inference mode (gradients disabled):")
    cnn.zero_grad()
    with torch.no_grad():
        features = cnn(images)

    is_learning, msg = quick_check_cnn_learning(cnn)
    print(msg)
    # Note: This should show no gradients (expected behavior)

    return True


def test_weight_change_tracking():
    """Test weight change tracking over training."""
    print("\n" + "="*70)
    print("TEST 4: Weight Change Tracking")
    print("="*70)

    # Create CNN
    cnn = create_simple_cnn()
    cnn.train()

    # Create diagnostics
    diagnostics = CNNDiagnostics(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)  # Higher LR for visible changes

    # Initial weights
    initial_norm = sum(p.norm().item() for p in cnn.parameters())
    print(f"Initial weight norm: {initial_norm:.4f}")

    # Train for several steps
    for step in range(20):
        images = torch.randn(8, 4, 84, 84)
        target = torch.randn(8, 512)
        features = cnn(images)
        loss = ((features - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        diagnostics.capture_weights()

    # Final weights
    final_norm = sum(p.norm().item() for p in cnn.parameters())
    print(f"Final weight norm: {final_norm:.4f}")

    # Check weights changed
    weight_change = abs(final_norm - initial_norm)
    print(f"Total weight change: {weight_change:.4f}")

    changed = weight_change > 1e-4
    if changed:
        print("✅ Weights are updating")
    else:
        print("❌ Weights not updating (might be learning rate too low)")

    return changed


def test_print_summary():
    """Test summary printing functionality."""
    print("\n" + "="*70)
    print("TEST 5: Summary Printing")
    print("="*70)

    # Create CNN
    cnn = create_simple_cnn()
    cnn.train()

    # Create diagnostics
    diagnostics = CNNDiagnostics(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Run a few training steps
    for _ in range(5):
        images = torch.randn(8, 4, 84, 84)
        target = torch.randn(8, 512)
        features = cnn(images)
        loss = ((features - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        diagnostics.capture_gradients()
        diagnostics.capture_features(features)
        optimizer.step()
        diagnostics.capture_weights()

    # Print summary
    print("\n[Summary Output]:")
    diagnostics.print_summary(last_n=5)

    return True


def run_all_tests():
    """Run all diagnostic tests."""
    print("\n" + "="*70)
    print("CNN DIAGNOSTICS TEST SUITE")
    print("="*70)

    tests = [
        ("Basic Functionality", test_diagnostics_basic),
        ("Multi-Step Training", test_diagnostics_multi_step),
        ("Gradient Flow Detection", test_gradient_flow_detection),
        ("Weight Change Tracking", test_weight_change_tracking),
        ("Summary Printing", test_print_summary)
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {str(e)}")
            results.append((name, False))

    # Print final results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12s} {name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n✅ ALL TESTS PASSED - CNN diagnostics working correctly!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

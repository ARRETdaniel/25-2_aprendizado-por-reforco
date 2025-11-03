"""
Test suite for select_action() Dict observation support (Bug #14 fix)

This test validates that select_action() can properly handle:
1. Dict observations {'image': (4,84,84), 'vector': (23,)}
2. Flat numpy array observations (535,) for backward compatibility
3. Deterministic flag for evaluation mode
4. Noise parameter for exploration mode

Reference: SELECT_ACTION_ANALYSIS.md - Fix 1: Dict Observation Support
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.td3_agent import TD3Agent
from src.models.cnn import NatureCNN


def test_dict_observation_handling():
    """Test 1: Verify select_action handles Dict observations correctly."""
    print("\n" + "="*70)
    print("TEST 1: Dict Observation Handling")
    print("="*70)

    # Initialize agent with CNN
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,
        config_path="config/td3_config.yaml"
    )

    # Create Dict observation
    obs_dict = {
        'image': np.random.randn(4, 84, 84).astype(np.float32),
        'vector': np.random.randn(23).astype(np.float32)
    }

    # Test deterministic action selection
    action_det = agent.select_action(obs_dict, deterministic=True)

    assert action_det.shape == (2,), f"Expected action shape (2,), got {action_det.shape}"
    assert np.all(action_det >= -1.0) and np.all(action_det <= 1.0), "Action out of bounds"

    print("✅ Dict observation handled successfully")
    print(f"   Input: Dict with image {obs_dict['image'].shape} and vector {obs_dict['vector'].shape}")
    print(f"   Output: Action {action_det.shape} with values in [-1, 1]")
    print(f"   Action: [{action_det[0]:.3f}, {action_det[1]:.3f}]")

    return True


def test_flat_observation_backward_compatibility():
    """Test 2: Verify select_action still works with flat numpy arrays."""
    print("\n" + "="*70)
    print("TEST 2: Flat Observation Backward Compatibility")
    print("="*70)

    # Initialize agent (no CNN for this test)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        config_path="config/td3_config.yaml"
    )

    # Create flat observation
    state = np.random.randn(535).astype(np.float32)

    # Test deterministic action selection
    action_det = agent.select_action(state, deterministic=True)

    assert action_det.shape == (2,), f"Expected action shape (2,), got {action_det.shape}"
    assert np.all(action_det >= -1.0) and np.all(action_det <= 1.0), "Action out of bounds"

    print("✅ Flat observation handled successfully (backward compatibility)")
    print(f"   Input: Flat array {state.shape}")
    print(f"   Output: Action {action_det.shape}")
    print(f"   Action: [{action_det[0]:.3f}, {action_det[1]:.3f}]")

    return True


def test_deterministic_flag():
    """Test 3: Verify deterministic flag produces consistent actions."""
    print("\n" + "="*70)
    print("TEST 3: Deterministic Flag Behavior")
    print("="*70)

    # Initialize agent with CNN
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,
        config_path="config/td3_config.yaml"
    )

    # Create Dict observation
    obs_dict = {
        'image': np.random.randn(4, 84, 84).astype(np.float32),
        'vector': np.random.randn(23).astype(np.float32)
    }

    # Test deterministic mode (should be same every time)
    action1 = agent.select_action(obs_dict, deterministic=True)
    action2 = agent.select_action(obs_dict, deterministic=True)
    action3 = agent.select_action(obs_dict, deterministic=True)

    assert np.allclose(action1, action2), "Deterministic actions should be identical"
    assert np.allclose(action2, action3), "Deterministic actions should be identical"

    print("✅ Deterministic mode produces consistent actions")
    print(f"   Action 1: [{action1[0]:.6f}, {action1[1]:.6f}]")
    print(f"   Action 2: [{action2[0]:.6f}, {action2[1]:.6f}]")
    print(f"   Action 3: [{action3[0]:.6f}, {action3[1]:.6f}]")
    print(f"   Max diff: {np.max(np.abs(action1 - action2)):.10f}")

    return True


def test_exploration_noise():
    """Test 4: Verify noise parameter adds exploration."""
    print("\n" + "="*70)
    print("TEST 4: Exploration Noise Behavior")
    print("="*70)

    # Initialize agent with CNN
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,
        config_path="config/td3_config.yaml"
    )

    # Create Dict observation
    obs_dict = {
        'image': np.random.randn(4, 84, 84).astype(np.float32),
        'vector': np.random.randn(23).astype(np.float32)
    }

    # Test exploration mode (should vary with noise)
    actions = []
    for _ in range(10):
        action = agent.select_action(obs_dict, noise=0.2, deterministic=False)
        actions.append(action)

    actions = np.array(actions)

    # Check that actions vary (std > 0)
    std_steering = np.std(actions[:, 0])
    std_throttle = np.std(actions[:, 1])

    assert std_steering > 0.01, f"Steering should vary with noise, got std={std_steering:.6f}"
    assert std_throttle > 0.01, f"Throttle should vary with noise, got std={std_throttle:.6f}"

    # Check that all actions are within bounds
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0), "Actions out of bounds"

    print("✅ Exploration noise adds variation to actions")
    print(f"   10 actions sampled with noise=0.2")
    print(f"   Steering std: {std_steering:.3f}")
    print(f"   Throttle std: {std_throttle:.3f}")
    print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   First 3 actions:")
    for i in range(3):
        print(f"     [{actions[i,0]:.3f}, {actions[i,1]:.3f}]")

    return True


def test_noise_vs_deterministic():
    """Test 5: Verify deterministic flag overrides noise parameter."""
    print("\n" + "="*70)
    print("TEST 5: Deterministic Flag Overrides Noise")
    print("="*70)

    # Initialize agent with CNN
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,
        config_path="config/td3_config.yaml"
    )

    # Create Dict observation
    obs_dict = {
        'image': np.random.randn(4, 84, 84).astype(np.float32),
        'vector': np.random.randn(23).astype(np.float32)
    }

    # Test that deterministic=True ignores noise parameter
    action1 = agent.select_action(obs_dict, noise=0.5, deterministic=True)
    action2 = agent.select_action(obs_dict, noise=0.5, deterministic=True)

    assert np.allclose(action1, action2), "Deterministic=True should ignore noise"

    print("✅ Deterministic flag correctly overrides noise parameter")
    print(f"   With noise=0.5, deterministic=True:")
    print(f"   Action 1: [{action1[0]:.6f}, {action1[1]:.6f}]")
    print(f"   Action 2: [{action2[0]:.6f}, {action2[1]:.6f}]")
    print(f"   Identical: {np.allclose(action1, action2)}")

    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("SELECT_ACTION Dict Observation Support Tests")
    print("Bug #14 Fix Validation")
    print("="*70)

    tests = [
        ("Dict Observation Handling", test_dict_observation_handling),
        ("Flat Observation Compatibility", test_flat_observation_backward_compatibility),
        ("Deterministic Flag", test_deterministic_flag),
        ("Exploration Noise", test_exploration_noise),
        ("Deterministic Overrides Noise", test_noise_vs_deterministic)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Bug #14 fix validated successfully.")
        print("\nNext steps:")
        print("  1. Run gradient flow test: python tests/test_gradient_flow.py")
        print("  2. Run integration test: python scripts/train_td3.py --steps 1000")
        print("  3. Run full training: python scripts/train_td3.py --steps 30000")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

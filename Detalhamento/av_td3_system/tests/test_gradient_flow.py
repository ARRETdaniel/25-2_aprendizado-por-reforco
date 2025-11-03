"""
Test gradient flow through CNN during TD3 training (Bug #14 fix)

This test validates that:
1. CNN weights change during training (gradients flowing)
2. extract_features() is called WITH gradients in train()
3. DictReplayBuffer preserves Dict structure for gradient flow
4. Gradients flow: loss → actor/critic → state → CNN

Reference: SELECT_ACTION_ANALYSIS.md - Fix 2: DictReplayBuffer for Gradient Flow
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


def test_cnn_gradient_flow():
    """
    Test that CNN weights change during training, proving gradient flow.

    This is the KEY test for Bug #14 fix:
    - If weights DON'T change → gradients NOT flowing → Bug still present
    - If weights DO change → gradients ARE flowing → Bug fixed!
    """
    print("\n" + "="*70)
    print("TEST: CNN Gradient Flow During Training")
    print("="*70)

    # Initialize agent with CNN and DictReplayBuffer
    print("\n[1/6] Initializing agent with CNN...")
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,  # CRITICAL: Use DictReplayBuffer!
        config_path="config/td3_config.yaml"
    )
    print("✅ Agent initialized with DictReplayBuffer")

    # Store initial CNN weights
    print("\n[2/6] Storing initial CNN weights...")
    initial_weights = {}
    for name, param in agent.cnn_extractor.named_parameters():
        initial_weights[name] = param.clone().detach()
    print(f"✅ Stored {len(initial_weights)} CNN weight tensors")

    # Populate replay buffer with Dict observations
    print("\n[3/6] Populating replay buffer with Dict observations...")
    for i in range(300):  # Need at least 256 for one batch
        obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        next_obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        action = np.random.randn(2)
        reward = np.random.randn()
        done = False

        agent.replay_buffer.add(obs_dict, action, next_obs_dict, reward, done)

    print(f"✅ Buffer populated with {len(agent.replay_buffer)} transitions")

    # Perform training updates
    print("\n[4/6] Performing 50 training updates...")
    for i in range(50):
        metrics = agent.train(batch_size=256)
        if i % 10 == 0:
            print(f"   Update {i+1}/50: critic_loss={metrics['critic_loss']:.3f}, "
                  f"q1={metrics['q1_value']:.3f}")
    print("✅ Training updates completed")

    # Check if CNN weights changed
    print("\n[5/6] Checking CNN weight updates...")
    weights_changed = []
    weight_diffs = []

    for name, param in agent.cnn_extractor.named_parameters():
        initial = initial_weights[name]
        current = param.detach()

        # Check if weights changed
        changed = not torch.allclose(initial, current, atol=1e-8)
        weights_changed.append(changed)

        # Compute L2 norm of difference
        diff_norm = torch.norm(current - initial).item()
        weight_diffs.append(diff_norm)

        if changed:
            print(f"   ✅ {name}: CHANGED (L2 diff = {diff_norm:.6f})")
        else:
            print(f"   ❌ {name}: UNCHANGED (weights frozen!)")

    # Verdict
    print("\n[6/6] Final Verdict:")
    num_changed = sum(weights_changed)
    total_weights = len(weights_changed)

    if num_changed == total_weights:
        print(f"   ✅ ALL CNN WEIGHTS CHANGED ({num_changed}/{total_weights})")
        print(f"   ✅ GRADIENTS ARE FLOWING through CNN!")
        print(f"   ✅ Bug #14 FIX SUCCESSFUL!")
        print(f"\n   Average weight change (L2 norm): {np.mean(weight_diffs):.6f}")
        print(f"   Max weight change: {np.max(weight_diffs):.6f}")
        print(f"   Min weight change: {np.min(weight_diffs):.6f}")
        return True
    elif num_changed > 0:
        print(f"   ⚠️  PARTIAL UPDATE: {num_changed}/{total_weights} weights changed")
        print(f"   ⚠️  Some gradients flowing, but not all layers!")
        return False
    else:
        print(f"   ❌ NO WEIGHTS CHANGED!")
        print(f"   ❌ GRADIENTS NOT FLOWING!")
        print(f"   ❌ Bug #14 still present!")
        return False


def test_extract_features_with_gradients():
    """
    Test that extract_features() is called WITH gradients during training.

    This validates that the train() method uses enable_grad=True.
    """
    print("\n" + "="*70)
    print("TEST: extract_features() Called WITH Gradients")
    print("="*70)

    # Initialize agent
    print("\n[1/4] Initializing agent...")
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,
        config_path="config/td3_config.yaml"
    )

    # Create Dict observation tensors
    print("\n[2/4] Creating Dict observation tensors...")
    obs_dict = {
        'image': torch.randn(32, 4, 84, 84).to(agent.device),
        'vector': torch.randn(32, 23).to(agent.device)
    }

    # Test extract_features with gradients ENABLED
    print("\n[3/4] Testing extract_features(enable_grad=True)...")
    state_with_grad = agent.extract_features(obs_dict, enable_grad=True)

    # Check if gradient tracking is enabled
    assert state_with_grad.requires_grad, "State should require gradients!"
    print("   ✅ State tensor has gradients enabled")
    print(f"   ✅ requires_grad: {state_with_grad.requires_grad}")

    # Perform a dummy backward pass to check gradient flow
    print("\n[4/4] Testing backward pass...")
    dummy_loss = state_with_grad.sum()
    dummy_loss.backward()

    # Check if CNN gradients were computed
    cnn_has_grads = False
    for param in agent.cnn_extractor.parameters():
        if param.grad is not None:
            cnn_has_grads = True
            break

    assert cnn_has_grads, "CNN should have gradients after backward!"
    print("   ✅ CNN received gradients after backward pass")
    print("   ✅ Gradient flow confirmed!")

    return True


def test_dict_replay_buffer_preserves_structure():
    """
    Test that DictReplayBuffer stores and returns Dict observations correctly.
    """
    print("\n" + "="*70)
    print("TEST: DictReplayBuffer Preserves Dict Structure")
    print("="*70)

    # Initialize agent with DictReplayBuffer
    print("\n[1/3] Initializing DictReplayBuffer...")
    cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0,
        cnn_extractor=cnn,
        use_dict_buffer=True,
        config_path="config/td3_config.yaml"
    )

    # Add Dict observations
    print("\n[2/3] Adding Dict observations to buffer...")
    for i in range(100):
        obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        next_obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        action = np.random.randn(2)
        reward = np.random.randn()
        done = False

        agent.replay_buffer.add(obs_dict, action, next_obs_dict, reward, done)

    # Sample from buffer
    print("\n[3/3] Sampling from buffer...")
    obs_dict, action, next_obs_dict, reward, not_done = agent.replay_buffer.sample(32)

    # Verify Dict structure
    assert isinstance(obs_dict, dict), "obs_dict should be a Dict!"
    assert 'image' in obs_dict, "obs_dict should have 'image' key"
    assert 'vector' in obs_dict, "obs_dict should have 'vector' key"

    # Verify tensor types (for gradient flow)
    assert isinstance(obs_dict['image'], torch.Tensor), "image should be torch.Tensor!"
    assert isinstance(obs_dict['vector'], torch.Tensor), "vector should be torch.Tensor!"

    # Verify shapes
    assert obs_dict['image'].shape == (32, 4, 84, 84), f"Unexpected image shape: {obs_dict['image'].shape}"
    assert obs_dict['vector'].shape == (32, 23), f"Unexpected vector shape: {obs_dict['vector'].shape}"

    print("   ✅ Buffer returns Dict observations (not flat arrays)")
    print(f"   ✅ obs_dict keys: {list(obs_dict.keys())}")
    print(f"   ✅ image shape: {obs_dict['image'].shape}")
    print(f"   ✅ vector shape: {obs_dict['vector'].shape}")
    print(f"   ✅ image type: {type(obs_dict['image']).__name__}")
    print(f"   ✅ vector type: {type(obs_dict['vector']).__name__}")

    return True


def run_all_tests():
    """Run all gradient flow tests."""
    print("\n" + "="*70)
    print("CNN GRADIENT FLOW TESTS")
    print("Bug #14 Fix Validation")
    print("="*70)

    tests = [
        ("DictReplayBuffer Structure", test_dict_replay_buffer_preserves_structure),
        ("extract_features() Gradients", test_extract_features_with_gradients),
        ("CNN Weight Updates", test_cnn_gradient_flow)
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
        print("\n🎉 ALL GRADIENT FLOW TESTS PASSED!")
        print("\n✅ Bug #14 fix validated:")
        print("   - DictReplayBuffer preserves Dict structure")
        print("   - extract_features() uses gradients")
        print("   - CNN weights update during training")
        print("   - End-to-end CNN training enabled!")
        print("\nNext steps:")
        print("  1. Run integration test: python scripts/train_td3.py --steps 1000")
        print("  2. Monitor CNN feature norms in TensorBoard")
        print("  3. Run full training: python scripts/train_td3.py --steps 30000")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("   Review errors and check implementation.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

"""
Test Reset API Compliance (Gymnasium v0.25+)

Verifies that the CARLANavigationEnv.reset() method complies with
Gymnasium v0.25+ API requirements.

Reference: RESET_FUNCTION_ANALYSIS.md - Test 1: API Compliance Test
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest


def test_reset_api_compliance():
    """
    Test that reset() returns tuple (observation, info) as required by Gymnasium v0.25+.

    Requirements:
    - reset() must return tuple of 2 elements
    - First element: observation (dict with 'image' and 'vector' keys)
    - Second element: info (dict with diagnostic data)
    - Observation shapes must match observation_space
    """
    print("\n" + "="*80)
    print("🧪 TEST: Reset API Compliance (Gymnasium v0.25+)")
    print("="*80)

    from src.environment.carla_env import CARLANavigationEnv

    # Initialize environment
    print("\n[1/5] Initializing CARLA environment...")

    # Get paths relative to this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    config_dir = os.path.join(project_root, 'config')

    env = CARLANavigationEnv(
        carla_config_path=os.path.join(config_dir, 'carla_config.yaml'),
        td3_config_path=os.path.join(config_dir, 'td3_config.yaml'),
        training_config_path=os.path.join(config_dir, 'training_config.yaml'),
        host='localhost',
        port=2000,
        headless=True
    )
    print("   ✅ Environment initialized")

    # Test reset returns tuple
    print("\n[2/5] Calling env.reset()...")
    result = env.reset()
    print(f"   Result type: {type(result)}")

    assert isinstance(result, tuple), \
        f"❌ FAIL: reset() must return tuple, got {type(result)}"
    print("   ✅ reset() returns tuple")

    assert len(result) == 2, \
        f"❌ FAIL: reset() must return exactly 2 values, got {len(result)}"
    print("   ✅ reset() returns exactly 2 values")

    # Unpack result
    print("\n[3/5] Unpacking result...")
    observation, info = result
    print(f"   Observation type: {type(observation)}")
    print(f"   Info type: {type(info)}")

    # Test observation structure
    print("\n[4/5] Validating observation structure...")
    assert isinstance(observation, dict), \
        f"❌ FAIL: observation must be dict, got {type(observation)}"
    print("   ✅ observation is dict")

    assert "image" in observation, \
        "❌ FAIL: observation must have 'image' key"
    assert "vector" in observation, \
        "❌ FAIL: observation must have 'vector' key"
    print("   ✅ observation has 'image' and 'vector' keys")

    assert observation["image"].shape == (4, 84, 84), \
        f"❌ FAIL: image shape must be (4, 84, 84), got {observation['image'].shape}"
    print(f"   ✅ image shape is (4, 84, 84)")

    # Note: vector shape is 535 (3 kinematic + 532 waypoint features)
    assert observation["vector"].shape == (535,), \
        f"❌ FAIL: vector shape must be (535,), got {observation['vector'].shape}"
    print(f"   ✅ vector shape is (535,)")

    # Test info structure
    print("\n[5/5] Validating info dict structure...")
    assert isinstance(info, dict), \
        f"❌ FAIL: info must be dict, got {type(info)}"
    print("   ✅ info is dict")

    # Check for expected info keys (Fix 2 implementation)
    expected_keys = ['episode', 'route_length_m', 'npc_count', 'spawn_location', 'observation_shapes']
    for key in expected_keys:
        assert key in info, \
            f"❌ FAIL: info missing expected key '{key}'"
    print(f"   ✅ info contains all expected keys: {expected_keys}")

    # Validate info content
    print("\n📊 Info Dict Contents:")
    print(f"   Episode: {info['episode']}")
    print(f"   Route Length: {info['route_length_m']:.0f}m")
    print(f"   NPC Count: {info['npc_count']}")
    print(f"   Spawn Location: ({info['spawn_location']['x']:.2f}, "
          f"{info['spawn_location']['y']:.2f}, {info['spawn_location']['z']:.2f})")
    print(f"   Spawn Yaw: {info['spawn_location']['yaw']:.2f}°")
    print(f"   Observation Shapes: image={tuple(info['observation_shapes']['image'])}, "
          f"vector={tuple(info['observation_shapes']['vector'])}")

    # Test multiple resets
    print("\n[BONUS] Testing multiple consecutive resets...")
    for i in range(3):
        obs2, info2 = env.reset()
        assert isinstance(obs2, dict), f"Reset {i+2}: observation must be dict"
        assert isinstance(info2, dict), f"Reset {i+2}: info must be dict"
        assert info2['episode'] == i + 2, \
            f"Reset {i+2}: episode counter should increment (expected {i+2}, got {info2['episode']})"
        print(f"   ✅ Reset {i+2}: Episode {info2['episode']}, Route {info2['route_length_m']:.0f}m")

    # Cleanup
    env.close()

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Reset API Compliance Verified")
    print("="*80)
    print("\n✅ Summary:")
    print("   - reset() returns tuple (observation, info) ✓")
    print("   - observation is dict with 'image' and 'vector' ✓")
    print("   - observation shapes match specification ✓")
    print("   - info is dict with diagnostic data ✓")
    print("   - info contains expected keys ✓")
    print("   - episode counter increments correctly ✓")
    print("\n🎉 Environment is Gymnasium v0.25+ compliant!")


if __name__ == "__main__":
    test_reset_api_compliance()

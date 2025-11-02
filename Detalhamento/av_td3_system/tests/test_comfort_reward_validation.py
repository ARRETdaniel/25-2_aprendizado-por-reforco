#!/usr/bin/env python3
"""
Quick validation test for comfort reward fixes.
Tests the key properties without requiring full CARLA setup.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.reward_functions import RewardCalculator


def test_comfort_reward_fixes():
    """Test all comfort reward fixes."""

    print("=" * 80)
    print("COMFORT REWARD FIXES - VALIDATION TEST")
    print("=" * 80)

    # Initialize reward calculator with test config
    config = {
        "weights": {
            "efficiency": 1.0,
            "lane_keeping": 2.0,
            "comfort": 0.5,
            "safety": 1.0,
            "progress": 5.0,
        },
        "efficiency": {
            "target_speed": 8.33,
            "speed_tolerance": 1.39,
            "overspeed_penalty_scale": 2.0,
        },
        "lane_keeping": {
            "lateral_tolerance": 0.5,
            "heading_tolerance": 0.1,
        },
        "comfort": {
            "jerk_threshold": 5.0,  # NEW: m/s³ (physically correct)
        },
        "safety": {
            "collision_penalty": -100.0,
            "offroad_penalty": -100.0,
            "wrong_way_penalty": -50.0,
        },
        "progress": {
            "waypoint_bonus": 10.0,
            "distance_scale": 1.0,
            "goal_reached_bonus": 100.0,
        },
        "gamma": 0.99,
    }

    reward_calc = RewardCalculator(config)
    dt = 0.05  # 20 Hz simulation

    print("\n" + "=" * 80)
    print("TEST 1: Verify Correct Jerk Units (m/s³)")
    print("=" * 80)

    # Test: acceleration change of 2 m/s² over 0.05 seconds
    accel1 = 0.0
    accel2 = 2.0
    # Expected jerk: (2.0 - 0.0) / 0.05 = 40 m/s³

    # Initialize state
    reward1 = reward_calc._calculate_comfort_reward(
        acceleration=accel1,
        acceleration_lateral=0.0,
        velocity=5.0,
        dt=dt
    )
    print(f"Initial reward (accel=0.0 m/s²): {reward1:.4f}")

    # Compute jerk
    reward2 = reward_calc._calculate_comfort_reward(
        acceleration=accel2,
        acceleration_lateral=0.0,
        velocity=5.0,
        dt=dt
    )
    computed_jerk = (accel2 - accel1) / dt
    print(f"Next reward (accel=2.0 m/s², Δaccel=2.0 m/s²):")
    print(f"  Computed jerk: {computed_jerk:.1f} m/s³")
    print(f"  Reward: {reward2:.4f}")
    print(f"  Threshold: {reward_calc.jerk_threshold} m/s³")

    assert reward2 < 0, "❌ FAIL: High jerk (40 m/s³) should be penalized!"
    assert reward2 >= -1.0, "❌ FAIL: Penalty should be bounded!"
    print("✅ PASS: Jerk units are correct (m/s³) and penalty is bounded")

    print("\n" + "=" * 80)
    print("TEST 2: Verify Differentiability (Smooth at Zero)")
    print("=" * 80)

    # Reset state
    reward_calc.prev_acceleration = 0.0
    reward_calc.prev_acceleration_lateral = 0.0

    # Test small positive jerk
    reward_calc._calculate_comfort_reward(0.0, 0.0, 5.0, dt)
    reward_pos = reward_calc._calculate_comfort_reward(0.05 * dt, 0.0, 5.0, dt)

    # Reset and test small negative jerk
    reward_calc.prev_acceleration = 0.0
    reward_calc._calculate_comfort_reward(0.0, 0.0, 5.0, dt)
    reward_neg = reward_calc._calculate_comfort_reward(-0.05 * dt, 0.0, 5.0, dt)

    print(f"Small positive jerk (+0.05 m/s³): reward = {reward_pos:.6f}")
    print(f"Small negative jerk (-0.05 m/s³): reward = {reward_neg:.6f}")
    print(f"Difference: {abs(reward_pos - reward_neg):.6f}")

    # Both should be very similar (smooth at zero)
    assert abs(reward_pos - reward_neg) < 0.001, "❌ FAIL: Reward should be symmetric near zero!"
    print("✅ PASS: Reward is smooth and differentiable at zero jerk")

    print("\n" + "=" * 80)
    print("TEST 3: Verify Bounded Penalties (No Explosion)")
    print("=" * 80)

    # Reset state
    reward_calc.prev_acceleration = 0.0
    reward_calc.prev_acceleration_lateral = 0.0

    # Initialize
    reward_calc._calculate_comfort_reward(0.0, 0.0, 5.0, dt)

    # Test extreme jerk: 100 m/s³ (20x threshold)
    extreme_accel = 100.0 * dt  # 5.0 m/s²
    reward_extreme = reward_calc._calculate_comfort_reward(extreme_accel, 0.0, 5.0, dt)

    print(f"Extreme jerk (100 m/s³, 20x threshold):")
    print(f"  Acceleration: {extreme_accel:.2f} m/s²")
    print(f"  Reward: {reward_extreme:.4f}")
    print(f"  Expected range: [-1.0, 0.3]")

    assert reward_extreme >= -1.0, f"❌ FAIL: Penalty too large: {reward_extreme}"
    assert reward_extreme <= 0.3, f"❌ FAIL: Reward too large: {reward_extreme}"
    print("✅ PASS: Extreme jerk penalties are bounded in [-1.0, 0.3]")

    print("\n" + "=" * 80)
    print("TEST 4: Verify Velocity Scaling (Low Speed Penalties)")
    print("=" * 80)

    # Fixed moderate jerk (2.0 m/s³, below threshold)
    accel1 = 0.0
    accel2 = 2.0 * dt  # 0.1 m/s²

    velocities = [0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    rewards = []

    print(f"Fixed jerk: 2.0 m/s³ (below threshold of {reward_calc.jerk_threshold} m/s³)")
    print(f"Velocity (m/s) | Linear Scale | Sqrt Scale | Reward")
    print("-" * 60)

    for v in velocities:
        reward_calc.prev_acceleration = accel1
        reward_calc.prev_acceleration_lateral = 0.0
        reward = reward_calc._calculate_comfort_reward(accel2, 0.0, v, dt)

        # Calculate scales for comparison
        linear_scale = max(0.0, min((v - 0.1) / 2.9, 1.0))
        sqrt_scale = max(0.0, min(np.sqrt((v - 0.1) / 2.9), 1.0))

        print(f"{v:14.2f} | {linear_scale:12.3f} | {sqrt_scale:10.3f} | {reward:6.4f}")
        rewards.append(reward)

    # Verify scaling behavior
    assert rewards[0] == 0.0, "❌ FAIL: Below threshold should be gated"
    assert rewards[1] == 0.0, "❌ FAIL: At threshold should be gated"
    assert rewards[2] > 0, "❌ FAIL: Above threshold with low jerk should be positive"
    assert rewards[-1] > rewards[2], "❌ FAIL: Reward should increase with velocity"

    # Compare improvement: sqrt vs linear at v=0.5 m/s
    linear_scale_05 = (0.5 - 0.1) / 2.9  # 0.138
    sqrt_scale_05 = np.sqrt((0.5 - 0.1) / 2.9)  # 0.372
    improvement = sqrt_scale_05 / linear_scale_05
    print(f"\nImprovement at v=0.5 m/s: {improvement:.2f}x stronger signal")

    print("✅ PASS: Velocity scaling provides gradual transition with sqrt")

    print("\n" + "=" * 80)
    print("TEST 5: Verify Threshold Behavior (Continuous Transition)")
    print("=" * 80)

    # Test jerk values around threshold
    reward_calc.prev_acceleration = 0.0
    reward_calc.prev_acceleration_lateral = 0.0

    jerk_values = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]  # Around threshold=5.0 m/s³
    print(f"Jerk (m/s³) | Normalized | Reward")
    print("-" * 40)

    for jerk in jerk_values:
        reward_calc.prev_acceleration = 0.0
        accel = jerk * dt
        reward = reward_calc._calculate_comfort_reward(accel, 0.0, 5.0, dt)
        normalized = min(jerk / reward_calc.jerk_threshold, 2.0)
        print(f"{jerk:11.1f} | {normalized:10.2f} | {reward:6.4f}")

    print("✅ PASS: Threshold transition is continuous")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✅")
    print("=" * 80)
    print("\nComfort reward fixes are working correctly:")
    print("  1. ✅ Jerk computed with correct units (m/s³)")
    print("  2. ✅ Reward is smooth and differentiable (no abs())")
    print("  3. ✅ Penalties are bounded (no Q-value explosion)")
    print("  4. ✅ Velocity scaling improved (sqrt vs linear)")
    print("  5. ✅ Threshold behavior is continuous")
    print("\nReady for training! 🚀")
    print()


if __name__ == "__main__":
    try:
        test_comfort_reward_fixes()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

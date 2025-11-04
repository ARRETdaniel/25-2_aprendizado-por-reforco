"""
Unit tests for PBRS dense safety rewards.

Tests validate that proximity-based penalties provide continuous
gradients for collision avoidance learning.

Reference: PBRS_IMPLEMENTATION_GUIDE.md (Step 4: Testing & Validation)
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.reward_functions import RewardCalculator


@pytest.fixture
def reward_calculator():
    """Create a reward calculator with test configuration."""
    config = {
        "weights": {
            "efficiency": 1.0,
            "lane_keeping": 2.0,
            "comfort": 0.5,
            "safety": 1.0,
            "progress": 5.0
        },
        "efficiency": {
            "target_speed": 8.33,
            "speed_tolerance": 1.39,
            "overspeed_penalty_scale": 2.0
        },
        "lane_keeping": {
            "lateral_tolerance": 0.5,
            "heading_tolerance": 0.1
        },
        "comfort": {
            "jerk_threshold": 5.0
        },
        "safety": {
            "collision_penalty": -10.0,
            "offroad_penalty": -10.0,
            "wrong_way_penalty": -5.0
        },
        "progress": {
            "waypoint_bonus": 10.0,
            "distance_scale": 50.0,
            "goal_reached_bonus": 100.0
        },
        "gamma": 0.99,
    }

    return RewardCalculator(config)


def test_pbrs_proximity_gradient(reward_calculator):
    """
    Test that proximity penalty increases as obstacle approaches.

    This validates the core PBRS functionality: continuous gradient
    that provides learning signal BEFORE collisions occur.
    """
    # Test distances from 10m to 0.5m (approaching obstacle)
    distances = [10.0, 5.0, 3.0, 1.0, 0.5]
    expected_penalties_approx = [-0.1, -0.2, -0.33, -1.0, -2.0]  # Approximate values

    penalties = []
    for distance in distances:
        reward_dict = reward_calculator.calculate(
            velocity=5.0,  # Moving at 5 m/s
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=50.0,
            waypoint_reached=False,
            goal_reached=False,
            lane_half_width=1.75,
            dt=0.05,
            # PBRS parameters
            distance_to_nearest_obstacle=distance,
            time_to_collision=None,
            collision_impulse=None,
        )
        penalties.append(reward_dict["breakdown"]["safety"])

    # Validate gradient: penalty should INCREASE (more negative) as distance DECREASES
    for i in range(len(penalties) - 1):
        assert penalties[i] > penalties[i+1], \
            f"Penalty should increase as obstacle approaches: " \
            f"{distances[i]}m={penalties[i]:.3f} vs {distances[i+1]}m={penalties[i+1]:.3f}"

    # Validate approximate magnitudes (within 20% tolerance)
    for i, (penalty, expected) in enumerate(zip(penalties, expected_penalties_approx)):
        tolerance = abs(expected) * 0.2  # 20% tolerance
        assert abs(penalty - expected) <= tolerance, \
            f"Penalty at {distances[i]}m: expected ~{expected:.3f}, got {penalty:.3f}"

    print(f"✅ PBRS Proximity Gradient Test PASSED")
    print(f"   Distances: {distances}")
    print(f"   Penalties: {[f'{p:.3f}' for p in penalties]}")
    print(f"   Expected:  {[f'{e:.3f}' for e in expected_penalties_approx]}")


def test_ttc_penalty(reward_calculator):
    """
    Test that TTC penalty increases as collision approaches.

    Validates the secondary safety signal for imminent collisions.
    """
    # Test TTC from 3s to 0.1s (imminent collision)
    ttc_values = [3.0, 2.0, 1.0, 0.5, 0.1]

    penalties = []
    for ttc in ttc_values:
        reward_dict = reward_calculator.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=50.0,
            waypoint_reached=False,
            goal_reached=False,
            lane_half_width=1.75,
            dt=0.05,
            # PBRS parameters
            distance_to_nearest_obstacle=5.0,  # Fixed distance
            time_to_collision=ttc,
            collision_impulse=None,
        )
        penalties.append(reward_dict["breakdown"]["safety"])

    # Validate gradient: penalty should INCREASE (more negative) as TTC DECREASES
    for i in range(len(penalties) - 1):
        assert penalties[i] > penalties[i+1], \
            f"Penalty should increase as TTC decreases: " \
            f"{ttc_values[i]}s={penalties[i]:.3f} vs {ttc_values[i+1]}s={penalties[i+1]:.3f}"

    print(f"✅ TTC Penalty Test PASSED")
    print(f"   TTC: {ttc_values}")
    print(f"   Penalties: {[f'{p:.3f}' for p in penalties]}")


def test_graduated_collision_penalty(reward_calculator):
    """
    Test that collision penalty scales with impact severity.

    Validates Priority 3 fix: graduated penalties based on collision impulse.
    """
    # Test collision impulses: soft, moderate, severe
    impulses = [10.0, 100.0, 500.0, 1000.0, 2000.0]  # Newtons
    expected_penalties = [-0.1, -1.0, -5.0, -10.0, -10.0]  # Capped at -10.0

    penalties = []
    for impulse in impulses:
        reward_dict = reward_calculator.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=True,  # Collision occurred
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=50.0,
            waypoint_reached=False,
            goal_reached=False,
            lane_half_width=1.75,
            dt=0.05,
            # PBRS parameters
            distance_to_nearest_obstacle=None,
            time_to_collision=None,
            collision_impulse=impulse,
        )
        penalties.append(reward_dict["breakdown"]["safety"])

    # Validate: higher impulse = higher penalty (more negative), with capping
    for i in range(len(penalties) - 1):
        assert penalties[i] >= penalties[i+1], \
            f"Penalty should increase with impact severity: " \
            f"{impulses[i]}N={penalties[i]:.3f} vs {impulses[i+1]}N={penalties[i+1]:.3f}"

    # Validate capping at -10.0
    assert penalties[-1] >= -10.0, f"Penalty should be capped at -10.0, got {penalties[-1]:.3f}"

    print(f"✅ Graduated Collision Penalty Test PASSED")
    print(f"   Impulses: {impulses}")
    print(f"   Penalties: {[f'{p:.3f}' for p in penalties]}")
    print(f"   Expected:  {[f'{e:.3f}' for e in expected_penalties]}")


def test_no_obstacle_no_penalty(reward_calculator):
    """
    Test that no proximity penalty is applied when no obstacle is detected.

    Validates that PBRS only activates when obstacles are present.
    """
    reward_dict = reward_calculator.calculate(
        velocity=8.33,  # Target speed
        lateral_deviation=0.0,
        heading_error=0.0,
        acceleration=0.0,
        acceleration_lateral=0.0,
        collision_detected=False,
        offroad_detected=False,
        wrong_way=False,
        distance_to_goal=50.0,
        waypoint_reached=False,
        goal_reached=False,
        lane_half_width=1.75,
        dt=0.05,
        # No obstacle detected
        distance_to_nearest_obstacle=None,
        time_to_collision=None,
        collision_impulse=None,
    )

    # Safety reward should be 0 or very close to 0 (only stopping penalty possible)
    safety_reward = reward_dict["breakdown"]["safety"]
    assert safety_reward >= -0.1, \
        f"Safety reward should be minimal when no obstacle: got {safety_reward:.3f}"

    print(f"✅ No Obstacle No Penalty Test PASSED")
    print(f"   Safety reward: {safety_reward:.3f}")


def test_distant_obstacle_minimal_penalty(reward_calculator):
    """
    Test that obstacles beyond 10m range produce no proximity penalty.

    Validates the 10m detection range threshold.
    """
    reward_dict = reward_calculator.calculate(
        velocity=8.33,
        lateral_deviation=0.0,
        heading_error=0.0,
        acceleration=0.0,
        acceleration_lateral=0.0,
        collision_detected=False,
        offroad_detected=False,
        wrong_way=False,
        distance_to_goal=50.0,
        waypoint_reached=False,
        goal_reached=False,
        lane_half_width=1.75,
        dt=0.05,
        # Obstacle beyond range
        distance_to_nearest_obstacle=15.0,  # > 10m threshold
        time_to_collision=None,
        collision_impulse=None,
    )

    # Safety reward should be minimal (no proximity penalty applied)
    safety_reward = reward_dict["breakdown"]["safety"]
    assert safety_reward >= -0.1, \
        f"Safety reward should be minimal for distant obstacle: got {safety_reward:.3f}"

    print(f"✅ Distant Obstacle Minimal Penalty Test PASSED")
    print(f"   Safety reward with obstacle @ 15m: {safety_reward:.3f}")


def test_combined_pbrs_and_collision(reward_calculator):
    """
    Test combined PBRS proximity and collision penalty.

    Validates that both penalties can coexist and sum correctly.
    """
    # Scenario: Close obstacle AND collision
    reward_dict = reward_calculator.calculate(
        velocity=5.0,
        lateral_deviation=0.0,
        heading_error=0.0,
        acceleration=0.0,
        acceleration_lateral=0.0,
        collision_detected=True,
        offroad_detected=False,
        wrong_way=False,
        distance_to_goal=50.0,
        waypoint_reached=False,
        goal_reached=False,
        lane_half_width=1.75,
        dt=0.05,
        # Close obstacle + collision
        distance_to_nearest_obstacle=1.0,  # Very close
        time_to_collision=0.2,  # Imminent
        collision_impulse=500.0,  # Moderate impact
    )

    safety_reward = reward_dict["breakdown"]["safety"]

    # Expected components:
    # - Proximity penalty @ 1m: ~-1.0
    # - TTC penalty @ 0.2s: ~-2.5
    # - Collision penalty @ 500N: -5.0
    # Total: ~-8.5

    expected_range = (-10.0, -6.0)  # Allow some variance
    assert expected_range[0] <= safety_reward <= expected_range[1], \
        f"Combined penalty should be in range {expected_range}, got {safety_reward:.3f}"

    print(f"✅ Combined PBRS and Collision Test PASSED")
    print(f"   Total safety penalty: {safety_reward:.3f}")
    print(f"   Expected range: {expected_range}")


def test_stopping_penalty_progression(reward_calculator):
    """
    Test progressive stopping penalty based on distance to goal.

    Validates that stopping near goal is allowed, but penalized far from goal.
    """
    # Test stopping at different distances from goal
    distances = [50.0, 15.0, 7.0, 3.0]  # Far to near goal

    penalties = []
    for dist in distances:
        reward_dict = reward_calculator.calculate(
            velocity=0.2,  # Nearly stopped
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=dist,
            waypoint_reached=False,
            goal_reached=False,
            lane_half_width=1.75,
            dt=0.05,
            distance_to_nearest_obstacle=None,
            time_to_collision=None,
            collision_impulse=None,
        )
        penalties.append(reward_dict["breakdown"]["safety"])

    # Validate: stopping penalty decreases as approaching goal
    assert penalties[0] < penalties[-1], \
        f"Stopping far from goal should be more penalized than near goal"

    print(f"✅ Stopping Penalty Progression Test PASSED")
    print(f"   Distances to goal: {distances}")
    print(f"   Stopping penalties: {[f'{p:.3f}' for p in penalties]}")


if __name__ == "__main__":
    # Run tests manually (for debugging)
    print("\n" + "="*70)
    print("PBRS Safety Reward Function Tests")
    print("="*70 + "\n")

    calc = reward_calculator()

    try:
        test_pbrs_proximity_gradient(calc)
        test_ttc_penalty(calc)
        test_graduated_collision_penalty(calc)
        test_no_obstacle_no_penalty(calc)
        test_distant_obstacle_minimal_penalty(calc)
        test_combined_pbrs_and_collision(calc)
        test_stopping_penalty_progression(calc)

        print("\n" + "="*70)
        print("✅ ALL PBRS TESTS PASSED")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise

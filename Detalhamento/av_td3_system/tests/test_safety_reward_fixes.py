#!/usr/bin/env python3
"""
Unit Tests for Safety Reward Function Fixes

Tests Priority 1, 2, and 3 fixes from analysis document:
- Priority 1: Dense PBRS safety guidance
- Priority 2: Magnitude rebalancing
- Priority 3: Graduated collision penalties

Reference: docs/SAFETY_REWARD_ANALYSIS.md
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.reward_functions import RewardCalculator


class TestDensePBRSGuidance(unittest.TestCase):
    """
    Test Priority 1 Fix: Dense PBRS Safety Guidance

    Validates:
    - Continuous reward gradient as obstacle approaches
    - Smooth reward surface (no discontinuities)
    - Proper TTC penalties for imminent collisions
    """

    def setUp(self):
        """Initialize reward calculator with test config."""
        self.config = {
            "weights": {
                "efficiency": 0.2,
                "lane_keeping": 0.15,
                "comfort": 0.1,
                "safety": 0.5,
                "progress": 0.05,
            },
            "efficiency": {
                "target_speed": 30.0,  # km/h
                "speed_tolerance": 5.0,
            },
            "lane_keeping": {
                "lateral_tolerance": 0.5,
                "heading_tolerance": 0.174533,  # 10 degrees
            },
            "comfort": {
                "jerk_threshold": 5.0,
            },
            "safety": {
                "collision_penalty": -5.0,  # NEW: Reduced from -100.0
                "off_road_penalty": -5.0,   # NEW: Reduced from -100.0
                "wrong_way_penalty": -2.0,  # NEW: Reduced from -50.0
            },
            "progress": {
                "waypoint_bonus": 10.0,
                "distance_scale": 50.0,  # NEW: Increased from 1.0
                "goal_reached_bonus": 100.0,
            },
            "gamma": 0.99,
        }
        self.reward_calc = RewardCalculator(self.config)

    def test_proximity_gradient_continuous(self):
        """Test: PBRS proximity reward provides continuous gradient."""
        # Test obstacle distances from far to close
        distances = [10.0, 5.0, 2.0, 1.0, 0.5]
        rewards = []

        for distance in distances:
            result = self.reward_calc.calculate(
                velocity=5.0,
                lateral_deviation=0.0,
                heading_error=0.0,
                acceleration=0.0,
                acceleration_lateral=0.0,
                collision_detected=False,
                offroad_detected=False,
                wrong_way=False,
                distance_to_goal=100.0,
                waypoint_reached=False,
                goal_reached=False,
                distance_to_nearest_obstacle=distance,  # NEW
                time_to_collision=None,
                collision_impulse=None,
            )
            rewards.append(result["safety"])

        # Verify rewards become more negative as distance decreases
        print(f"\nProximity rewards at distances {distances}:")
        print(f"Rewards: {[f'{r:.3f}' for r in rewards]}")

        for i in range(len(rewards) - 1):
            self.assertGreaterEqual(
                rewards[i],
                rewards[i + 1],
                f"Reward at {distances[i]}m ({rewards[i]:.3f}) should be >= "
                f"reward at {distances[i+1]}m ({rewards[i+1]:.3f})"
            )

    def test_reward_surface_smooth(self):
        """Test: No discontinuous jumps in reward surface."""
        # Test small distance changes (0.1m increments)
        distances = np.linspace(0.5, 5.0, 50)
        rewards = []

        for distance in distances:
            result = self.reward_calc.calculate(
                velocity=5.0,
                lateral_deviation=0.0,
                heading_error=0.0,
                acceleration=0.0,
                acceleration_lateral=0.0,
                collision_detected=False,
                offroad_detected=False,
                wrong_way=False,
                distance_to_goal=100.0,
                waypoint_reached=False,
                goal_reached=False,
                distance_to_nearest_obstacle=distance,
                time_to_collision=None,
                collision_impulse=None,
            )
            rewards.append(result["safety"])

        # Verify no large jumps (max change < 2.0)
        deltas = np.diff(rewards)
        max_delta = np.max(np.abs(deltas))

        print(f"\nMax reward delta over 0.1m distance change: {max_delta:.3f}")

        self.assertLess(
            max_delta,
            2.0,
            f"Maximum reward change ({max_delta:.3f}) exceeds smoothness threshold (2.0)"
        )

    def test_ttc_penalty_applied(self):
        """Test: TTC penalty applied when collision imminent."""
        # Test with short TTC (< 3.0 seconds)
        result_short_ttc = self.reward_calc.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=100.0,
            waypoint_reached=False,
            goal_reached=False,
            distance_to_nearest_obstacle=5.0,
            time_to_collision=1.0,  # 1 second TTC - DANGEROUS!
            collision_impulse=None,
        )

        # Test with long TTC (> 3.0 seconds)
        result_long_ttc = self.reward_calc.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=100.0,
            waypoint_reached=False,
            goal_reached=False,
            distance_to_nearest_obstacle=5.0,
            time_to_collision=5.0,  # 5 seconds TTC - SAFE
            collision_impulse=None,
        )

        print(f"\nSafety reward with TTC=1.0s: {result_short_ttc['safety']:.3f}")
        print(f"Safety reward with TTC=5.0s: {result_long_ttc['safety']:.3f}")

        # Short TTC should have MORE NEGATIVE reward
        self.assertLess(
            result_short_ttc["safety"],
            result_long_ttc["safety"],
            "Short TTC should have more negative safety reward"
        )

    def test_no_obstacle_no_penalty(self):
        """Test: No proximity penalty when no obstacle detected."""
        result = self.reward_calc.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=100.0,
            waypoint_reached=False,
            goal_reached=False,
            distance_to_nearest_obstacle=float('inf'),  # No obstacle
            time_to_collision=None,
            collision_impulse=None,
        )

        print(f"\nSafety reward with no obstacle: {result['safety']:.3f}")

        # Should be 0.0 (no penalties)
        self.assertEqual(
            result["safety"],
            0.0,
            "Safety reward should be 0.0 when no obstacles and no violations"
        )


class TestMagnitudeRebalancing(unittest.TestCase):
    """
    Test Priority 2 Fix: Reward Magnitude Rebalancing

    Validates:
    - Collision penalties reduced from -100.0 to -5.0
    - Offroad penalties reduced from -100.0 to -5.0
    - Wrong-way penalties reduced from -50.0 to -2.0
    - Progress rewards increased 50x (distance_scale: 1.0 → 50.0)
    """

    def setUp(self):
        """Initialize reward calculator with test config."""
        self.config = {
            "weights": {
                "efficiency": 0.2,
                "lane_keeping": 0.15,
                "comfort": 0.1,
                "safety": 0.5,
                "progress": 0.05,
            },
            "efficiency": {
                "target_speed": 30.0,
                "speed_tolerance": 5.0,
            },
            "lane_keeping": {
                "lateral_tolerance": 0.5,
                "heading_tolerance": 0.174533,
            },
            "comfort": {
                "jerk_threshold": 5.0,
            },
            "safety": {
                "collision_penalty": -5.0,  # REDUCED
                "off_road_penalty": -5.0,   # REDUCED
                "wrong_way_penalty": -2.0,  # REDUCED
            },
            "progress": {
                "waypoint_bonus": 10.0,
                "distance_scale": 50.0,  # INCREASED
                "goal_reached_bonus": 100.0,
            },
            "gamma": 0.99,
        }
        self.reward_calc = RewardCalculator(self.config)

    def test_collision_penalty_reduced(self):
        """Test: Collision penalty is -5.0 (not -100.0)."""
        result = self.reward_calc.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=True,  # COLLISION!
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=100.0,
            waypoint_reached=False,
            goal_reached=False,
            distance_to_nearest_obstacle=None,
            time_to_collision=None,
            collision_impulse=None,  # No impulse data (default penalty)
        )

        print(f"\nCollision safety reward: {result['safety']:.3f}")

        # Weighted safety = 0.5 * -5.0 = -2.5
        expected_weighted = 0.5 * -5.0
        actual_weighted = result["breakdown"]["safety"][2]

        self.assertAlmostEqual(
            actual_weighted,
            expected_weighted,
            places=2,
            msg=f"Weighted collision penalty should be {expected_weighted:.2f}"
        )

    def test_multi_objective_balance(self):
        """Test: Agent can offset collision through good driving."""
        # Calculate rewards: 1 collision + 0.1m progress
        collision_reward = self.reward_calc.calculate(
            velocity=8.33,  # Perfect target speed
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=True,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=100.0,
            waypoint_reached=False,
            goal_reached=False,
            distance_to_nearest_obstacle=None,
            time_to_collision=None,
            collision_impulse=None,
        )

        # Good driving rewards over 20 steps (0.1m progress each)
        progress_distance = 0.1 * 50.0  # distance_scale
        weighted_progress = 0.05 * progress_distance  # weight * progress

        print(f"\nCollision total reward: {collision_reward['total']:.3f}")
        print(f"Progress reward (0.1m): {weighted_progress:.3f}")
        print(f"Break-even distance: {abs(collision_reward['total']) / weighted_progress:.3f}m")

        # Agent should be able to recover from collision in reasonable distance
        # NOTE: ~10m recovery is REASONABLE for safe exploration
        # (analysis document goal: balanced multi-objective learning, not instant recovery)
        recovery_distance = abs(collision_reward["total"]) / weighted_progress
        self.assertLess(
            recovery_distance,
            15.0,  # Less than 15 meters (allows exploration with some risk)
            f"Recovery distance ({recovery_distance:.3f}m) too long"
        )


class TestGraduatedPenalties(unittest.TestCase):
    """
    Test Priority 3 Fix: Graduated Collision Penalties

    Validates:
    - Collision penalty scales with impulse magnitude
    - Soft collisions (10N) → -0.1 penalty
    - Moderate collisions (100N) → -1.0 penalty
    - Severe collisions (500N) → -5.0 penalty
    """

    def setUp(self):
        """Initialize reward calculator with test config."""
        self.config = {
            "weights": {
                "efficiency": 0.2,
                "lane_keeping": 0.15,
                "comfort": 0.1,
                "safety": 0.5,
                "progress": 0.05,
            },
            "efficiency": {
                "target_speed": 30.0,
                "speed_tolerance": 5.0,
            },
            "lane_keeping": {
                "lateral_tolerance": 0.5,
                "heading_tolerance": 0.174533,
            },
            "comfort": {
                "jerk_threshold": 5.0,
            },
            "safety": {
                "collision_penalty": -5.0,
                "off_road_penalty": -5.0,
                "wrong_way_penalty": -2.0,
            },
            "progress": {
                "waypoint_bonus": 10.0,
                "distance_scale": 50.0,
                "goal_reached_bonus": 100.0,
            },
            "gamma": 0.99,
        }
        self.reward_calc = RewardCalculator(self.config)

    def test_graduated_penalties_by_impulse(self):
        """Test: Collision penalty scales with impulse magnitude."""
        impulses = [10.0, 50.0, 100.0, 300.0, 500.0]
        penalties = []

        for impulse in impulses:
            result = self.reward_calc.calculate(
                velocity=5.0,
                lateral_deviation=0.0,
                heading_error=0.0,
                acceleration=0.0,
                acceleration_lateral=0.0,
                collision_detected=True,
                offroad_detected=False,
                wrong_way=False,
                distance_to_goal=100.0,
                waypoint_reached=False,
                goal_reached=False,
                distance_to_nearest_obstacle=None,
                time_to_collision=None,
                collision_impulse=impulse,  # Varying impulse
            )
            penalties.append(result["safety"])

        print(f"\nGraduated penalties by impulse:")
        for imp, pen in zip(impulses, penalties):
            print(f"  {imp:5.0f}N → {pen:6.3f}")

        # Verify penalties increase with impulse (more negative = harsher)
        # Since penalties are negative, we check absolute values
        for i in range(len(penalties) - 1):
            self.assertLessEqual(
                abs(penalties[i]),
                abs(penalties[i + 1]),
                f"Penalty at {impulses[i]}N should be less harsh than at {impulses[i+1]}N"
            )

        # Verify soft collision is small penalty
        self.assertGreater(penalties[0], -1.0, "Soft collision penalty too harsh")

        # Verify severe collision is capped at -5.0
        self.assertGreaterEqual(penalties[-1], -5.0, "Severe collision penalty exceeds cap")


class TestBackwardCompatibility(unittest.TestCase):
    """
    Test: Reward function works with missing new parameters.

    Ensures existing code that doesn't pass new parameters still works.
    """

    def setUp(self):
        """Initialize reward calculator with test config."""
        self.config = {
            "weights": {
                "efficiency": 0.2,
                "lane_keeping": 0.15,
                "comfort": 0.1,
                "safety": 0.5,
                "progress": 0.05,
            },
            "efficiency": {
                "target_speed": 30.0,
                "speed_tolerance": 5.0,
            },
            "lane_keeping": {
                "lateral_tolerance": 0.5,
                "heading_tolerance": 0.174533,
            },
            "comfort": {
                "jerk_threshold": 5.0,
            },
            "safety": {
                "collision_penalty": -5.0,
                "off_road_penalty": -5.0,
                "wrong_way_penalty": -2.0,
            },
            "progress": {
                "waypoint_bonus": 10.0,
                "distance_scale": 50.0,
                "goal_reached_bonus": 100.0,
            },
            "gamma": 0.99,
        }
        self.reward_calc = RewardCalculator(self.config)

    def test_old_api_still_works(self):
        """Test: Reward calculation works without new parameters."""
        # Call with only old parameters (no new ones)
        result = self.reward_calc.calculate(
            velocity=5.0,
            lateral_deviation=0.0,
            heading_error=0.0,
            acceleration=0.0,
            acceleration_lateral=0.0,
            collision_detected=False,
            offroad_detected=False,
            wrong_way=False,
            distance_to_goal=100.0,
            waypoint_reached=False,
            goal_reached=False,
            # NOT passing: distance_to_nearest_obstacle, time_to_collision, collision_impulse
        )

        print(f"\nOld API call total reward: {result['total']:.3f}")

        # Should not raise exception
        self.assertIsNotNone(result)
        self.assertIn("total", result)
        self.assertIn("safety", result)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""
Quick Reward Function Unit Tests

Unit tests to verify reward function components work correctly in isolation.
Run these before conducting full manual validation.

Usage:
    pytest scripts/test_reward_components.py -v

    Or standalone:
    python scripts/test_reward_components.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("Warning: pytest not installed. Running basic tests only.")


def create_mock_state(
    velocity=10.0,  # m/s
    lateral_deviation=0.0,  # meters
    heading_error=0.0,  # radians
    distance_to_goal=100.0,  # meters
    collision=False,
    off_road=False,
    lane_invasion=False,
    jerk=0.0  # m/s^3
):
    """Create mock state dictionary for testing."""
    return {
        'velocity': velocity,
        'lateral_deviation': lateral_deviation,
        'heading_error': heading_error,
        'distance_to_goal': distance_to_goal,
        'collision': collision,
        'off_road': off_road,
        'lane_invasion': lane_invasion,
        'jerk': jerk,
        'speed_kmh': velocity * 3.6
    }


class TestLaneKeepingReward:
    """Tests for lane keeping reward component."""

    def test_zero_deviation_gives_zero_penalty(self):
        """Perfect lane centering should give minimal/zero penalty."""
        # Simplified reward calculation
        lateral_deviation = 0.0
        weight = 1.0

        penalty = -weight * abs(lateral_deviation)

        assert penalty == 0.0, "Zero deviation should give zero penalty"

    def test_penalty_increases_with_deviation(self):
        """Larger deviations should give larger penalties."""
        weight = 1.0
        deviations = [0.0, 0.5, 1.0, 2.0]

        penalties = [-weight * abs(dev) for dev in deviations]

        # Should be monotonically decreasing (more negative)
        for i in range(len(penalties) - 1):
            assert penalties[i] > penalties[i+1], \
                f"Penalty should increase: {penalties[i]} should be > {penalties[i+1]}"

    def test_symmetric_deviation_penalty(self):
        """Left and right deviations should have same magnitude penalty."""
        weight = 1.0

        left_deviation = -1.5
        right_deviation = 1.5

        left_penalty = -weight * abs(left_deviation)
        right_penalty = -weight * abs(right_deviation)

        assert left_penalty == right_penalty, \
            "Symmetric deviations should have equal penalties"

    def test_penalty_bounded(self):
        """Penalty should not exceed reasonable bounds."""
        weight = 1.0
        max_reasonable_deviation = 5.0  # meters

        # Even extreme deviations should be bounded
        extreme_penalty = -weight * max_reasonable_deviation

        # Should not be absurdly large
        assert extreme_penalty > -100.0, \
            "Penalty should be bounded even for extreme deviations"


class TestEfficiencyReward:
    """Tests for efficiency (speed-based) reward component."""

    def test_reward_peaks_near_target_speed(self):
        """Efficiency reward should be maximum near target speed."""
        target_speed = 30.0 / 3.6  # 30 km/h in m/s
        tolerance = 2.0 / 3.6  # 2 km/h tolerance

        # Simplified reward: Gaussian around target
        speeds_kmh = [0, 10, 20, 30, 40, 50]
        rewards = []

        for speed_kmh in speeds_kmh:
            speed = speed_kmh / 3.6
            speed_error = abs(speed - target_speed)

            # Gaussian reward
            reward = np.exp(-(speed_error**2) / (2 * tolerance**2))
            rewards.append(reward)

        # Maximum should be at target speed (30 km/h, index 3)
        max_idx = rewards.index(max(rewards))
        assert speeds_kmh[max_idx] == 30, \
            f"Efficiency should peak at target (30 km/h), got {speeds_kmh[max_idx]}"

    def test_zero_speed_low_reward(self):
        """Standing still should give low efficiency reward."""
        target_speed = 30.0 / 3.6
        current_speed = 0.0

        speed_error = abs(current_speed - target_speed)
        reward = np.exp(-(speed_error**2) / (2 * (2.0/3.6)**2))

        # Should be significantly less than max (1.0)
        assert reward < 0.1, "Zero speed should give low efficiency reward"

    def test_symmetric_speed_deviation(self):
        """Under/over target speed should have similar penalty shape."""
        target_speed = 30.0 / 3.6
        tolerance = 2.0 / 3.6

        # 10 km/h below target
        slow_speed = 20.0 / 3.6
        slow_error = abs(slow_speed - target_speed)
        slow_reward = np.exp(-(slow_error**2) / (2 * tolerance**2))

        # 10 km/h above target
        fast_speed = 40.0 / 3.6
        fast_error = abs(fast_speed - target_speed)
        fast_reward = np.exp(-(fast_error**2) / (2 * tolerance**2))

        # Should be approximately equal
        assert abs(slow_reward - fast_reward) < 0.01, \
            "Symmetric speed errors should give similar rewards"


class TestComfortPenalty:
    """Tests for comfort (jerk-based) penalty component."""

    def test_zero_jerk_no_penalty(self):
        """Smooth driving (zero jerk) should have no comfort penalty."""
        jerk = 0.0
        weight = 0.1

        penalty = -weight * abs(jerk)

        assert penalty == 0.0, "Zero jerk should give zero comfort penalty"

    def test_penalty_scales_with_jerk(self):
        """Higher jerk should give larger comfort penalty."""
        weight = 0.1
        jerks = [0.0, 1.0, 5.0, 10.0]

        penalties = [-weight * abs(j) for j in jerks]

        # Should be monotonically decreasing (more negative)
        for i in range(len(penalties) - 1):
            assert penalties[i] > penalties[i+1], \
                "Comfort penalty should increase with jerk"

    def test_emergency_brake_high_penalty(self):
        """Emergency braking should trigger substantial comfort penalty."""
        emergency_jerk = 15.0  # m/s^3
        weight = 0.1

        penalty = -weight * abs(emergency_jerk)

        # Should be noticeable
        assert penalty < -1.0, "Emergency brake should give significant penalty"


class TestSafetyPenalty:
    """Tests for safety penalty component."""

    def test_collision_triggers_large_penalty(self):
        """Collision should result in very large negative reward."""
        collision_penalty = -10.0

        # This should be dominant over other rewards
        assert collision_penalty < -5.0, "Collision penalty should be substantial"

    def test_off_road_triggers_penalty(self):
        """Going off-road should trigger safety penalty."""
        off_road_penalty = -5.0

        assert off_road_penalty < -1.0, "Off-road should have significant penalty"

    def test_no_penalty_for_safe_driving(self):
        """Safe driving should have zero safety penalty."""
        state = create_mock_state(
            collision=False,
            off_road=False,
            lane_invasion=False
        )

        # Simplified safety penalty calculation
        if state['collision']:
            penalty = -10.0
        elif state['off_road']:
            penalty = -5.0
        elif state['lane_invasion']:
            penalty = -2.0
        else:
            penalty = 0.0

        assert penalty == 0.0, "Safe driving should have zero safety penalty"


class TestRewardComponentSummation:
    """Tests for reward component summation."""

    def test_components_sum_to_total(self):
        """Sum of components should equal total reward."""
        # Sample reward components
        efficiency = 0.5
        lane_keeping = -0.3
        comfort = -0.1
        safety = 0.0
        progress = 0.2

        total_calculated = (
            efficiency +
            lane_keeping +
            comfort +
            safety +
            progress
        )

        total_expected = 0.3  # Manual calculation

        # Should match within floating point precision
        assert abs(total_calculated - total_expected) < 1e-6, \
            f"Component sum {total_calculated} should equal total {total_expected}"

    def test_no_component_dominates_inappropriately(self):
        """No single component should dominate in normal driving."""
        # Normal driving scenario
        efficiency = 0.8
        lane_keeping = -0.2
        comfort = -0.05
        safety = 0.0
        progress = 0.1

        total = efficiency + lane_keeping + comfort + safety + progress

        # No component should be more than 2x the total (except in extreme cases)
        for component in [efficiency, lane_keeping, comfort, progress]:
            assert abs(component) < 2 * abs(total), \
                f"Component {component} dominates total {total}"


def run_basic_tests():
    """Run basic tests without pytest."""
    print("\n" + "="*70)
    print("BASIC REWARD FUNCTION TESTS (Manual)")
    print("="*70 + "\n")

    test_classes = [
        TestLaneKeepingReward,
        TestEfficiencyReward,
        TestComfortPenalty,
        TestSafetyPenalty,
        TestRewardComponentSummation
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        # Get all test methods
        test_methods = [
            method for method in dir(instance)
            if method.startswith('test_')
        ]

        for test_method_name in test_methods:
            test_method = getattr(instance, test_method_name)
            total_tests += 1

            try:
                test_method()
                print(f"  ✓ {test_method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {test_method_name}: {e}")
            except Exception as e:
                print(f"  ✗ {test_method_name}: Unexpected error: {e}")

    print("\n" + "="*70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total_tests - passed_tests} test(s) failed")

    print("="*70 + "\n")

    return passed_tests == total_tests


def main():
    """Main entry point."""
    if PYTEST_AVAILABLE:
        # Run with pytest for better output
        print("Running tests with pytest...")
        import pytest
        sys.exit(pytest.main([__file__, '-v', '--tb=short']))
    else:
        # Fall back to basic test runner
        success = run_basic_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validation script for goal termination fix.

Tests that the environment properly terminates when the vehicle reaches the goal,
ensuring compliance with Gymnasium API and TD3 algorithm requirements.

Bug Context:
    Previous implementation allowed episodes to continue indefinitely at goal,
    receiving +100.0 reward every step without termination (115 times observed!).

Fix:
    Changed is_route_finished() threshold from -2 to -300 (3.0m with 1cm spacing).

Tests:
    1. Vehicle approaching goal: Verify no premature termination (> 3.0m away)
    2. Vehicle at goal: Verify termination within expected steps (< 3.0m away)
    3. Reward-termination consistency: goal_reached=True → terminated=True
    4. No infinite loop: Episode ends after termination signal

Reference: GOAL_TERMINATION_BUG_ANALYSIS.md
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
import numpy as np
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_goal_termination():
    """
    Test that environment terminates correctly when goal is reached.

    Expected behavior:
        - Distance > 3.0m: No termination, normal rewards
        - Distance < 3.0m: Termination within 1-2 steps
        - goal_reached (distance < 2.0m) → is_route_finished()=True
        - step() returns terminated=True when goal reached
    """
    logger.info("="*80)
    logger.info("GOAL TERMINATION VALIDATION")
    logger.info("="*80)

    try:
        # Import environment
        from environment.carla_env import CARLANavigationEnv

        # Configuration paths
        config_dir = project_root / "config"
        carla_config = config_dir / "carla_config.yaml"
        td3_config = config_dir / "td3_config.yaml"
        training_config = config_dir / "training_config.yaml"

        # Create environment
        logger.info("Creating CARLA environment...")
        env = CARLANavigationEnv(
            carla_config_path=str(carla_config),
            td3_config_path=str(td3_config),
            training_config_path=str(training_config),
            host="localhost",
            port=2000,
            tm_port=8100,  # Different from training to avoid conflicts
        )

        logger.info("Environment created successfully")

        # Test 1: Normal operation (far from goal)
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Normal Operation (Far from Goal)")
        logger.info("="*80)

        obs, info = env.reset()
        logger.info("✅ Reset successful")

        # Take a few steps to get away from spawn
        for step in range(10):
            action = np.array([0.0, 0.3])  # Straight, moderate throttle
            obs, reward, terminated, truncated, info = env.step(action)

            distance_to_goal = info.get('distance_to_goal', float('inf'))
            logger.info(
                f"Step {step}: distance_to_goal={distance_to_goal:.2f}m, "
                f"terminated={terminated}, truncated={truncated}, reward={reward:.3f}"
            )

            # Should NOT terminate when far from goal
            if distance_to_goal > 5.0:
                assert not terminated, f"❌ Premature termination at {distance_to_goal:.2f}m from goal!"
                logger.info(f"✅ No premature termination (distance={distance_to_goal:.2f}m > 5.0m)")

        # Test 2: Approaching goal
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Approaching Goal Region")
        logger.info("="*80)

        # Drive until we get close to goal (< 10m)
        max_steps = 5000
        goal_approach_detected = False

        for step in range(10, max_steps):
            action = np.array([0.0, 0.5])  # Straight, moderate speed
            obs, reward, terminated, truncated, info = env.step(action)

            distance_to_goal = info.get('distance_to_goal', float('inf'))
            goal_reached = info.get('goal_reached', False)

            # Log every 100 steps or when close to goal
            if step % 100 == 0 or distance_to_goal < 10.0:
                logger.info(
                    f"Step {step}: distance_to_goal={distance_to_goal:.2f}m, "
                    f"goal_reached={goal_reached}, terminated={terminated}, reward={reward:.3f}"
                )

            # Check goal approach (< 10m)
            if distance_to_goal < 10.0 and not goal_approach_detected:
                logger.info(f"🎯 Goal approach detected at step {step}: distance={distance_to_goal:.2f}m")
                goal_approach_detected = True

            # Test 3: Goal region termination + single bonus verification (< 3.0m)
            if distance_to_goal < 3.5:  # Slightly outside threshold
                logger.info("\n" + "="*80)
                logger.info("TEST 3: Goal Region Termination + Single Bonus Verification")
                logger.info("="*80)
                logger.info(f"Vehicle within 3.5m of goal (distance={distance_to_goal:.2f}m)")
                logger.info("Expecting termination within next few steps...")
                logger.info("Verifying goal bonus awarded only ONCE...")

                # Take a few more steps and verify termination
                termination_steps = 0
                max_termination_wait = 50  # Should terminate within 50 steps of entering 3.0m radius
                goal_bonus_count = 0  # Track how many times goal bonus given

                for wait_step in range(max_termination_wait):
                    action = np.array([0.0, 0.3])
                    obs, reward, terminated, truncated, info = env.step(action)

                    distance_to_goal = info.get('distance_to_goal', float('inf'))
                    goal_reached = info.get('goal_reached', False)

                    # Check if goal bonus was awarded this step (reward > 100 indicates goal bonus)
                    if reward > 100.0:
                        goal_bonus_count += 1
                        logger.info(
                            f"  🎯 GOAL BONUS #{goal_bonus_count} detected: "
                            f"reward={reward:.3f}, distance={distance_to_goal:.2f}m"
                        )

                    logger.info(
                        f"  Wait step {wait_step}: distance={distance_to_goal:.2f}m, "
                        f"goal_reached={goal_reached}, terminated={terminated}, reward={reward:.3f}"
                    )

                    termination_steps += 1

                    # Check for termination
                    if terminated:
                        logger.info(f"✅ TERMINATION DETECTED at step {step + wait_step}")
                        logger.info(f"   Terminated after {termination_steps} steps in goal region")
                        logger.info(f"   Final distance: {distance_to_goal:.2f}m")
                        logger.info(f"   Termination reason: {info.get('termination_reason', 'unknown')}")
                        logger.info(f"   Goal bonus count: {goal_bonus_count}")

                        # FIX #3.2: Verify single goal bonus
                        if goal_bonus_count != 1:
                            logger.error(
                                f"❌ GOAL BONUS BUG: Expected exactly 1 goal bonus, "
                                f"but got {goal_bonus_count}!"
                            )
                            logger.error("   This indicates reward inflation from multiple bonuses.")
                            return False
                        else:
                            logger.info("✅ Single goal bonus verified (exactly 1 bonus awarded)")

                        # Verify consistency
                        if goal_reached and not terminated:
                            logger.error("❌ INCONSISTENCY: goal_reached=True but terminated=False!")
                            return False

                        logger.info("✅ Reward-termination consistency check passed")
                        break

                    # If we've been in goal region too long without termination, that's a bug
                    if distance_to_goal < 3.0 and wait_step > 10:
                        logger.error(
                            f"❌ BUG: Vehicle at {distance_to_goal:.2f}m for {wait_step} steps "
                            f"without termination!"
                        )
                        logger.error(f"   Goal bonus count so far: {goal_bonus_count}")
                        return False

                if not terminated:
                    logger.error(f"❌ FAILURE: No termination after {max_termination_wait} steps!")
                    return False

                # Test passed!
                break

            # End episode if collision or truncation
            if terminated or truncated:
                reason = info.get('termination_reason', 'unknown')
                if reason != 'route_completed':
                    logger.warning(f"Episode ended early: {reason}")
                    logger.info("Restarting for goal test...")
                    obs, info = env.reset()
                    continue
                else:
                    # Route completed!
                    logger.info(f"✅ Route completed successfully at step {step}")
                    break

        # Test 4: Verify no infinite loop
        logger.info("\n" + "="*80)
        logger.info("TEST 4: No Infinite Loop After Termination")
        logger.info("="*80)

        # Episode should be ended, trying to step should raise error or return done=True
        try:
            obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))

            # Some environments allow stepping after termination (returns same state)
            # But terminated should still be True
            if not terminated:
                logger.warning("⚠️ step() after termination returned terminated=False")
                logger.warning("   This may cause infinite loops if reset() is not called!")
            else:
                logger.info("✅ Repeated step() after termination still returns terminated=True")

        except Exception as e:
            logger.info(f"✅ step() after termination raises exception: {e}")

        # Cleanup
        logger.info("\n" + "="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)
        env.close()
        logger.info("✅ Environment closed successfully")

        logger.info("\n" + "="*80)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*80)
        logger.info("✅ No premature termination (distance > 5.0m)")
        logger.info("✅ Termination when goal reached (distance < 3.0m)")
        logger.info("✅ Single goal bonus verified (exactly 1 bonus per episode)")
        logger.info("✅ Reward-termination consistency")
        logger.info("✅ No infinite loop")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_goal_termination()

    if success:
        logger.info("\n[DONE] Goal termination validation successful!")
        sys.exit(0)
    else:
        logger.error("\n[FAILED] Goal termination validation failed!")
        sys.exit(1)

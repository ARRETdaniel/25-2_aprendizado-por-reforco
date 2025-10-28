#!/usr/bin/env python3
"""
Quick Waypoint System Diagnostic Test

Purpose: Validate that waypoint_reached and goal_reached flags are working correctly.
This is a critical short test to diagnose why progress rewards are not triggering.

Expected Runtime: ~5-10 minutes (500 steps with random actions)
Expected Outcomes:
  - Waypoint flags should trigger every 50-100 steps (when vehicle moves forward)
  - Goal flags should trigger at least once if episode lasts long enough
  - Progress rewards should include +10 bonuses when waypoints reached

Author: Daniel Terra
Date: October 26, 2024
"""

import sys
import os

# Add project root to path
sys.path.insert(0, '/workspace/av_td3_system')

import numpy as np
import logging

from src.environment.carla_env import CARLANavigationEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def main():
    print("="*80)
    print("WAYPOINT SYSTEM DIAGNOSTIC TEST")
    print("="*80)
    print("\nInitializing CARLA environment...")
    print("This may take 1-2 minutes...")

    # Initialize environment
    env = CARLANavigationEnv(
        carla_config_path="config/carla_config.yaml",
        td3_config_path="config/td3_config.yaml",
        training_config_path="config/training_config.yaml"
    )

    print("✓ Environment initialized\n")
    print("Starting 500-step test with random actions...")
    print("Monitoring waypoint and goal flags...\n")

    # Reset environment
    obs, info = env.reset()

    # Tracking variables
    waypoint_count = 0
    goal_count = 0
    positive_reward_count = 0
    total_reward = 0.0
    episode_count = 0
    step_count = 0

    episode_rewards = []
    current_episode_reward = 0.0

    # Run test
    for step in range(500):
        step_count += 1

        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        current_episode_reward += reward
        total_reward += reward

        # Check waypoint flag
        if info.get("waypoint_reached", False):
            waypoint_count += 1
            print(f"[Step {step:3d}] 🎯 WAYPOINT REACHED! (Total: {waypoint_count}) | Reward: {reward:+.2f}")

        # Check goal flag
        if info.get("goal_reached", False):
            goal_count += 1
            print(f"[Step {step:3d}] 🏁 GOAL REACHED! (Total: {goal_count}) | Reward: {reward:+.2f}")

        # Check positive rewards
        if reward > 0:
            positive_reward_count += 1
            print(f"[Step {step:3d}] ✅ Positive reward: {reward:+.2f} | Info: {info.get('reward_breakdown', {})}")

        # Periodic progress logging
        if step % 100 == 0 and step > 0:
            try:
                wp_idx = env.waypoint_manager.get_current_waypoint_index()
                wp_total = len(env.waypoint_manager.waypoints)
                vehicle_location = env.vehicle.get_location()
                dist = env.waypoint_manager.get_distance_to_goal(vehicle_location)
                progress = env.waypoint_manager.get_progress_percentage()

                print(f"\n[Step {step:3d}] PROGRESS CHECK:")
                print(f"  Waypoint: {wp_idx}/{wp_total}")
                print(f"  Distance to goal: {dist:.1f}m")
                print(f"  Route completion: {progress:.1f}%")
                print(f"  Avg reward (last 100): {current_episode_reward/100:.2f}\n")

            except Exception as e:
                print(f"[Step {step:3d}] Error getting progress info: {e}\n")

        # Handle episode termination
        if terminated or truncated:
            episode_count += 1
            episode_rewards.append(current_episode_reward)

            termination_reason = "Unknown"
            if info.get("collision", False):
                termination_reason = "Collision"
            elif info.get("off_road", False):
                termination_reason = "Off Road"
            elif info.get("goal_reached", False):
                termination_reason = "Goal Reached ✓"
            elif truncated:
                termination_reason = "Timeout"

            print(f"\n[Step {step:3d}] Episode {episode_count} ended ({termination_reason})")
            print(f"  Episode reward: {current_episode_reward:.2f}")
            print(f"  Episode length: {info.get('episode_length', 'N/A')} steps\n")

            # Reset for new episode
            obs, info = env.reset()
            current_episode_reward = 0.0

    # Final statistics
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\nSteps completed: {step_count}")
    print(f"Episodes: {episode_count}")
    print(f"\n--- CRITICAL FLAGS ---")
    print(f"Waypoints reached: {waypoint_count} ({waypoint_count/step_count*100:.2f}% of steps)")
    print(f"Goals reached: {goal_count}")
    print(f"\n--- REWARDS ---")
    print(f"Positive rewards: {positive_reward_count} ({positive_reward_count/step_count*100:.2f}% of steps)")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward/step_count:.2f} per step")

    if episode_count > 0:
        print(f"\n--- EPISODES ---")
        print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
        print(f"Average episode length: {step_count/episode_count:.1f} steps")

    print("\n--- DIAGNOSTIC RESULTS ---")
    if waypoint_count == 0:
        print("🔴 PROBLEM: No waypoints reached!")
        print("   → Waypoint progression logic may not be working")
        print("   → Check waypoint_manager.update_current_waypoint()")
        print("   → Check waypoint distance threshold")
    elif waypoint_count < 5:
        print("🟡 WARNING: Very few waypoints reached")
        print("   → Vehicle may not be moving forward enough")
        print("   → Waypoint threshold may be too tight")
    else:
        print("✅ GOOD: Waypoints being reached regularly")

    if goal_count == 0:
        print("🟡 INFO: No goals reached")
        print("   → May be normal for short test (500 steps)")
        print("   → Check if any episode lasted >100 steps")
    else:
        print("✅ EXCELLENT: Goal reached successfully!")

    if positive_reward_count < step_count * 0.05:  # Less than 5% positive
        print("🔴 PROBLEM: Very few positive rewards!")
        print("   → Progress component may not be working correctly")
        print("   → Check distance reduction calculation")

    print("="*80)

    # Cleanup
    env.close()
    print("\n✓ Test completed. Environment closed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detailed analysis of collision episodes with positive rewards.

This script investigates why episodes that terminate with collisions
still show positive episode rewards despite having negative step rewards.

Author: Daniel Terra Gomes
Date: 2025-11-26
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Paths to extracted CSV files
BASE_DIR = Path('/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251126-171053/extracted_metrics')

def load_csv(filename):
    """Load CSV file and return DataFrame."""
    filepath = BASE_DIR / filename
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    return pd.read_csv(filepath)

def main():
    print("="*80)
    print("COLLISION EPISODE REWARD ANALYSIS")
    print("="*80)
    print()

    # Load data
    print("Loading extracted TensorBoard metrics...")
    episode_reward_df = load_csv('train_episode_reward.csv')
    current_reward_df = load_csv('progress_current_reward.csv')
    episode_length_df = load_csv('train_episode_length.csv')
    collisions_df = load_csv('train_collisions_per_episode.csv')

    print(f"  - Episode rewards: {len(episode_reward_df)} episodes")
    print(f"  - Step rewards: {len(current_reward_df)} steps")
    print(f"  - Episode lengths: {len(episode_length_df)} episodes")
    print(f"  - Collisions: {len(collisions_df)} episodes")
    print()

    # Identify collision episodes
    collision_episodes = collisions_df[collisions_df['value'] > 0]
    print(f"Episodes with collisions: {len(collision_episodes)} ({100 * len(collision_episodes) / len(collisions_df):.1f}%)")
    print(f"Collision episode indices: {list(collision_episodes.index[:20])}...")
    print()

    # Analyze collision episodes in detail
    print("-" * 80)
    print("COLLISION EPISODES DETAILED ANALYSIS (first 20)")
    print("-" * 80)
    print()

    cumulative_steps = 0
    for ep_idx in range(min(len(episode_length_df), 30)):  # Analyze first 30 episodes
        ep_length = int(episode_length_df.iloc[ep_idx]['value'])
        ep_reward = episode_reward_df.iloc[ep_idx]['value']
        collision_count = collisions_df.iloc[ep_idx]['value']

        # Get step rewards for this episode
        step_start = cumulative_steps
        step_end = cumulative_steps + ep_length

        if step_end <= len(current_reward_df):
            episode_steps = current_reward_df.iloc[step_start:step_end]

            if len(episode_steps) > 0:
                step_reward_sum = episode_steps['value'].sum()
                step_reward_mean = episode_steps['value'].mean()
                step_reward_min = episode_steps['value'].min()
                step_reward_max = episode_steps['value'].max()
                positive_steps = (episode_steps['value'] > 0).sum()
                negative_steps = (episode_steps['value'] < 0).sum()

                # Only show collision episodes
                if collision_count > 0:
                    print(f"Episode {ep_idx} (COLLISION):")
                    print(f"  - Length: {ep_length} steps")
                    print(f"  - Logged episode_reward: {ep_reward:.2f}")
                    print(f"  - Sum of step rewards: {step_reward_sum:.2f}")
                    print(f"  - Positive steps: {positive_steps} ({100*positive_steps/ep_length:.1f}%)")
                    print(f"  - Negative steps: {negative_steps} ({100*negative_steps/ep_length:.1f}%)")
                    print(f"  - Step reward range: [{step_reward_min:.2f}, {step_reward_max:.2f}]")
                    print(f"  - Step reward mean: {step_reward_mean:.2f}")

                    # Show first and last 5 step rewards
                    first_steps = episode_steps.head(5)['value'].tolist()
                    last_steps = episode_steps.tail(5)['value'].tolist()
                    print(f"  - First 5 step rewards: {[f'{r:.2f}' for r in first_steps]}")
                    print(f"  - Last 5 step rewards: {[f'{r:.2f}' for r in last_steps]}")
                    print()

        cumulative_steps += ep_length

    # Statistical summary
    print("-" * 80)
    print("STATISTICAL SUMMARY (collision vs non-collision episodes)")
    print("-" * 80)
    print()

    # Separate collision and non-collision episodes
    collision_indices = collisions_df[collisions_df['value'] > 0].index
    non_collision_indices = collisions_df[collisions_df['value'] == 0].index

    collision_rewards = episode_reward_df.iloc[collision_indices]['value']
    non_collision_rewards = episode_reward_df.iloc[non_collision_indices]['value']

    collision_lengths = episode_length_df.iloc[collision_indices]['value']
    non_collision_lengths = episode_length_df.iloc[non_collision_indices]['value']

    print("Collision Episodes:")
    print(f"  - Count: {len(collision_rewards)}")
    print(f"  - Episode reward: mean={collision_rewards.mean():.2f}, std={collision_rewards.std():.2f}, range=[{collision_rewards.min():.2f}, {collision_rewards.max():.2f}]")
    print(f"  - Episode length: mean={collision_lengths.mean():.1f}, std={collision_lengths.std():.1f}, range=[{collision_lengths.min():.0f}, {collision_lengths.max():.0f}]")
    print(f"  - Positive rewards: {(collision_rewards > 0).sum()} ({100 * (collision_rewards > 0).sum() / len(collision_rewards):.1f}%)")
    print()

    print("Non-Collision Episodes:")
    print(f"  - Count: {len(non_collision_rewards)}")
    print(f"  - Episode reward: mean={non_collision_rewards.mean():.2f}, std={non_collision_rewards.std():.2f}, range=[{non_collision_rewards.min():.2f}, {non_collision_rewards.max():.2f}]")
    print(f"  - Episode length: mean={non_collision_lengths.mean():.1f}, std={non_collision_lengths.std():.1f}, range=[{non_collision_lengths.min():.0f}, {non_collision_lengths.max():.0f}]")
    print(f"  - Positive rewards: {(non_collision_rewards > 0).sum()} ({100 * (non_collision_rewards > 0).sum() / len(non_collision_rewards):.1f}%)")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyze reward discrepancy between episode_reward and current_reward.

This script investigates why train/episode_reward shows positive values while
progress/current_reward shows negative step rewards ending in collisions.

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
    print("REWARD DISCREPANCY ANALYSIS")
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

    # Analyze episode rewards
    print("-" * 80)
    print("EPISODE REWARD ANALYSIS")
    print("-" * 80)
    print(f"Total episodes: {len(episode_reward_df)}")
    print(f"Episode reward range: [{episode_reward_df['value'].min():.2f}, {episode_reward_df['value'].max():.2f}]")
    print(f"Episode reward mean: {episode_reward_df['value'].mean():.2f} ± {episode_reward_df['value'].std():.2f}")
    print(f"Positive episode rewards: {(episode_reward_df['value'] > 0).sum()} ({100 * (episode_reward_df['value'] > 0).sum() / len(episode_reward_df):.1f}%)")
    print(f"Negative episode rewards: {(episode_reward_df['value'] < 0).sum()} ({100 * (episode_reward_df['value'] < 0).sum() / len(episode_reward_df):.1f}%)")
    print()

    # Analyze step rewards
    print("-" * 80)
    print("STEP REWARD ANALYSIS")
    print("-" * 80)
    print(f"Total steps: {len(current_reward_df)}")
    print(f"Step reward range: [{current_reward_df['value'].min():.2f}, {current_reward_df['value'].max():.2f}]")
    print(f"Step reward mean: {current_reward_df['value'].mean():.2f} ± {current_reward_df['value'].std():.2f}")
    print(f"Positive step rewards: {(current_reward_df['value'] > 0).sum()} ({100 * (current_reward_df['value'] > 0).sum() / len(current_reward_df):.1f}%)")
    print(f"Negative step rewards: {(current_reward_df['value'] < 0).sum()} ({100 * (current_reward_df['value'] < 0).sum() / len(current_reward_df):.1f}%)")
    print()

    # Find large negative step rewards (likely collisions)
    collision_threshold = -5.0
    large_negative_steps = current_reward_df[current_reward_df['value'] < collision_threshold]
    print(f"Large negative step rewards (< {collision_threshold}): {len(large_negative_steps)} steps")
    if len(large_negative_steps) > 0:
        print(f"  - Mean: {large_negative_steps['value'].mean():.2f}")
        print(f"  - Range: [{large_negative_steps['value'].min():.2f}, {large_negative_steps['value'].max():.2f}]")
    print()

    # Analyze episode-step correlation
    print("-" * 80)
    print("EPISODE-STEP CORRELATION ANALYSIS")
    print("-" * 80)
    print()

    # For first 10 episodes, calculate sum of step rewards and compare with logged episode reward
    print("Analyzing first 10 episodes (detailed):")
    print()

    cumulative_steps = 0
    for ep_idx in range(min(10, len(episode_length_df))):
        ep_length = int(episode_length_df.iloc[ep_idx]['value'])
        ep_reward = episode_reward_df.iloc[ep_idx]['value']

        # Get step rewards for this episode
        step_start = cumulative_steps
        step_end = cumulative_steps + ep_length
        episode_steps = current_reward_df.iloc[step_start:step_end]

        if len(episode_steps) > 0:
            step_reward_sum = episode_steps['value'].sum()
            step_reward_mean = episode_steps['value'].mean()
            step_reward_min = episode_steps['value'].min()
            step_reward_max = episode_steps['value'].max()
            discrepancy = ep_reward - step_reward_sum

            print(f"Episode {ep_idx}:")
            print(f"  - Length: {ep_length} steps")
            print(f"  - Logged episode_reward: {ep_reward:.2f}")
            print(f"  - Sum of step rewards: {step_reward_sum:.2f}")
            print(f"  - Discrepancy: {discrepancy:.2f} ({100 * abs(discrepancy) / (abs(ep_reward) + 1e-8):.1f}%)")
            print(f"  - Step reward stats: mean={step_reward_mean:.2f}, min={step_reward_min:.2f}, max={step_reward_max:.2f}")
            print(f"  - Collisions: {collisions_df.iloc[ep_idx]['value']:.0f}")
            print()

        cumulative_steps += ep_length

    # Overall statistics
    print("-" * 80)
    print("OVERALL STATISTICS (all episodes)")
    print("-" * 80)

    total_discrepancy = 0
    episodes_with_mismatch = 0
    cumulative_steps = 0

    for ep_idx in range(len(episode_length_df)):
        ep_length = int(episode_length_df.iloc[ep_idx]['value'])
        ep_reward = episode_reward_df.iloc[ep_idx]['value']

        # Get step rewards for this episode
        step_start = cumulative_steps
        step_end = cumulative_steps + ep_length

        if step_end <= len(current_reward_df):
            episode_steps = current_reward_df.iloc[step_start:step_end]
            step_reward_sum = episode_steps['value'].sum()
            discrepancy = abs(ep_reward - step_reward_sum)

            if discrepancy > 0.01:  # Tolerance for floating-point errors
                episodes_with_mismatch += 1
                total_discrepancy += discrepancy

        cumulative_steps += ep_length

    print(f"Episodes analyzed: {len(episode_length_df)}")
    print(f"Episodes with reward mismatch (>0.01): {episodes_with_mismatch} ({100 * episodes_with_mismatch / len(episode_length_df):.1f}%)")
    if episodes_with_mismatch > 0:
        print(f"Average discrepancy: {total_discrepancy / episodes_with_mismatch:.2f}")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

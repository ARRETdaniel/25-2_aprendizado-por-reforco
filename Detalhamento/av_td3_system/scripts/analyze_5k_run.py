#!/usr/bin/env python3
"""
Analyze 5K Step Training Run - TD3 Metrics Validation
Based on:
- TD3 paper (Fujimoto et al., 2018)
- Stable-Baselines3 TD3 benchmarks
- Related CARLA papers

This script extracts TensorBoard metrics and validates against expected behavior.
"""

import os
import sys
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("NumPy not found, using pure Python fallback")
    np = None

try:
    import pandas as pd
except ImportError:
    print("Pandas not found, will skip CSV export")
    pd = None

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ TensorBoard not installed. Please install: pip install tensorboard")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not found, skipping visualization")
    plt = None

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_tensorboard_data(log_dir):
    """
    Load TensorBoard event data from the specified directory.

    Args:
        log_dir: Path to TensorBoard logs directory

    Returns:
        dict: Dictionary of metric_name -> list of (step, value) tuples
    """
    print(f"\n📊 Loading TensorBoard data from: {log_dir}")

    # Find the events file
    events_file = None
    for file in os.listdir(log_dir):
        if file.startswith('events.out.tfevents'):
            events_file = os.path.join(log_dir, file)
            break

    if not events_file:
        raise FileNotFoundError(f"No TensorBoard events file found in {log_dir}")

    print(f"✅ Found events file: {os.path.basename(events_file)}")

    # Load the event accumulator
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()

    # Extract all scalar tags
    tags = ea.Tags()['scalars']
    print(f"\n📈 Found {len(tags)} metric categories:")

    # Organize by category
    categories = {}
    for tag in tags:
        category = tag.split('/')[0] if '/' in tag else 'other'
        if category not in categories:
            categories[category] = []
        categories[category].append(tag)

    for category, tags_list in sorted(categories.items()):
        print(f"  - {category}: {len(tags_list)} metrics")

    # Extract all metrics
    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics, categories


def analyze_q_values(metrics):
    """
    Analyze Q-value metrics and validate against TD3 paper expectations.

    TD3 Paper (Hopper-v1, 1M steps):
    - 0-50K steps: Q-values 0 → ~500
    - 50K-200K steps: Q-values 500 → ~2000
    - 200K-1M steps: Q-values 2000 → ~4000

    Our 5K steps = 0.5% of 1M steps
    Expected Q-values: 0 → ~25-50 (linear interpolation)
    """
    print("\n" + "="*80)
    print("🔍 Q-VALUE ANALYSIS")
    print("="*80)

    # Extract Q-value metrics
    q_metrics = {
        'actor_q_mean': 'debug/actor_q_mean',
        'actor_q_std': 'debug/actor_q_std',
        'q1_value': 'debug/q1_value',
        'q2_value': 'debug/q2_value',
        'target_q': 'debug/target_q',
    }

    results = {}
    for name, tag in q_metrics.items():
        if tag in metrics:
            data = metrics[tag]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]

            results[name] = {
                'steps': steps,
                'values': values,
                'initial': values[0] if values else None,
                'final': values[-1] if values else None,
                'mean': np.mean(values) if values else None,
                'std': np.std(values) if values else None,
                'min': np.min(values) if values else None,
                'max': np.max(values) if values else None,
            }

            print(f"\n📊 {name}:")
            print(f"   Initial: {results[name]['initial']:.2f}")
            print(f"   Final: {results[name]['final']:.2f}")
            print(f"   Mean: {results[name]['mean']:.2f} ± {results[name]['std']:.2f}")
            print(f"   Range: [{results[name]['min']:.2f}, {results[name]['max']:.2f}]")

    # Validate against TD3 paper expectations
    print("\n🎯 VALIDATION AGAINST TD3 PAPER:")
    print(f"   Paper (Hopper-v1, 0-50K steps): Q-values 0 → 500")
    print(f"   Expected at 5K steps (10% of 50K): Q-values ~0 → 50")

    if 'actor_q_mean' in results:
        final_actor_q = results['actor_q_mean']['final']
        if final_actor_q < 100:
            print(f"   ✅ Actor Q-mean = {final_actor_q:.2f} - WITHIN expected range")
        elif final_actor_q < 500:
            print(f"   ⚠️  Actor Q-mean = {final_actor_q:.2f} - HIGHER than expected but acceptable")
        else:
            print(f"   ❌ Actor Q-mean = {final_actor_q:.2f} - MUCH HIGHER than expected!")

    # Check actor-critic divergence
    if 'actor_q_mean' in results and 'q1_value' in results:
        actor_q = results['actor_q_mean']['final']
        critic_q = results['q1_value']['final']
        divergence = actor_q / critic_q if critic_q != 0 else float('inf')

        print(f"\n🔬 ACTOR-CRITIC DIVERGENCE:")
        print(f"   Actor Q: {actor_q:.2f}")
        print(f"   Critic Q1: {critic_q:.2f}")
        print(f"   Ratio: {divergence:.2f}×")

        if divergence < 3:
            print(f"   ✅ Divergence <3× - GOOD (minimal overestimation)")
        elif divergence < 10:
            print(f"   ⚠️  Divergence {divergence:.2f}× - ACCEPTABLE (moderate overestimation)")
        else:
            print(f"   ❌ Divergence {divergence:.2f}× - CRITICAL (severe overestimation!)")

    return results


def analyze_rewards(metrics):
    """
    Analyze reward metrics and validate against paper expectations.
    """
    print("\n" + "="*80)
    print("🎁 REWARD ANALYSIS")
    print("="*80)

    reward_metrics = {
        'episode_reward': 'train/episode_reward',
        'episode_length': 'train/episode_length',
    }

    results = {}
    for name, tag in reward_metrics.items():
        if tag in metrics:
            data = metrics[tag]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]

            results[name] = {
                'steps': steps,
                'values': values,
                'initial': values[0] if values else None,
                'final': values[-1] if values else None,
                'mean': np.mean(values) if values else None,
                'std': np.std(values) if values else None,
                'trend': np.polyfit(steps, values, 1)[0] if len(values) > 1 else 0,
            }

            print(f"\n📊 {name}:")
            print(f"   Initial: {results[name]['initial']:.2f}")
            print(f"   Final: {results[name]['final']:.2f}")
            print(f"   Mean: {results[name]['mean']:.2f} ± {results[name]['std']:.2f}")
            print(f"   Trend: {results[name]['trend']:.4f} (slope per step)")

    # Validate learning signal
    if 'episode_reward' in results:
        trend = results['episode_reward']['trend']
        print(f"\n🎯 LEARNING SIGNAL:")
        if trend > 0.001:
            print(f"   ✅ Positive trend ({trend:.4f}) - Agent is learning!")
        elif trend > -0.001:
            print(f"   ⚠️  Near-zero trend ({trend:.4f}) - No clear learning signal yet")
        else:
            print(f"   ❌ Negative trend ({trend:.4f}) - Performance degrading!")

    return results


def analyze_losses(metrics):
    """
    Analyze loss metrics (critic loss, actor loss).
    """
    print("\n" + "="*80)
    print("📉 LOSS ANALYSIS")
    print("="*80)

    loss_metrics = {
        'critic_loss': 'losses/critic_loss',
        'actor_loss': 'losses/actor_loss',
    }

    results = {}
    for name, tag in loss_metrics.items():
        if tag in metrics:
            data = metrics[tag]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]

            results[name] = {
                'steps': steps,
                'values': values,
                'initial': values[0] if values else None,
                'final': values[-1] if values else None,
                'mean': np.mean(values) if values else None,
                'std': np.std(values) if values else None,
            }

            print(f"\n📊 {name}:")
            print(f"   Initial: {results[name]['initial']:.2f}")
            print(f"   Final: {results[name]['final']:.2f}")
            print(f"   Mean: {results[name]['mean']:.2f} ± {results[name]['std']:.2f}")

    # Validate actor loss
    if 'actor_loss' in results:
        final_actor_loss = results['actor_loss']['final']
        print(f"\n🎯 ACTOR LOSS VALIDATION:")
        print(f"   Actor loss = -Q(s,π(s)) by design")
        print(f"   Expected: NEGATIVE and GROWING (more negative) as policy improves")
        print(f"   Final actor loss: {final_actor_loss:.2f}")

        if final_actor_loss < 0:
            print(f"   ✅ Negative actor loss - CORRECT")
        else:
            print(f"   ⚠️  Positive actor loss - UNEXPECTED (check implementation)")

    return results


def analyze_gradients(metrics):
    """
    Analyze gradient norms to validate clipping is working.
    """
    print("\n" + "="*80)
    print("📏 GRADIENT NORM ANALYSIS")
    print("="*80)

    grad_metrics = {
        'actor_mlp_norm': 'gradients/actor_mlp_norm',
        'actor_cnn_norm': 'gradients/actor_cnn_norm',
        'critic_mlp_norm': 'gradients/critic_mlp_norm',
        'critic_cnn_norm': 'gradients/critic_cnn_norm',
    }

    results = {}
    for name, tag in grad_metrics.items():
        if tag in metrics:
            data = metrics[tag]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]

            results[name] = {
                'steps': steps,
                'values': values,
                'mean': np.mean(values) if values else None,
                'max': np.max(values) if values else None,
            }

            print(f"\n📊 {name}:")
            print(f"   Mean: {results[name]['mean']:.6f}")
            print(f"   Max: {results[name]['max']:.6f}")

            # Validate clipping
            if 'actor' in name:
                if results[name]['max'] <= 1.0:
                    print(f"   ✅ Gradient norm ≤ 1.0 - Clipping working correctly")
                else:
                    print(f"   ❌ Gradient norm > 1.0 - Clipping may be failing!")
            elif 'critic' in name:
                if results[name]['max'] <= 10.0:
                    print(f"   ✅ Gradient norm ≤ 10.0 - Clipping working correctly")
                else:
                    print(f"   ❌ Gradient norm > 10.0 - Clipping may be failing!")

    return results


def generate_summary_report(metrics, q_results, reward_results, loss_results, grad_results):
    """
    Generate a comprehensive summary report with pass/fail criteria.
    """
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE SUMMARY REPORT - 5K STEP VALIDATION")
    print("="*80)

    checks = []

    # Check 1: Q-value explosion
    if 'actor_q_mean' in q_results:
        final_q = q_results['actor_q_mean']['final']
        if final_q < 100:
            checks.append(("Q-Value Range", "✅ PASS", f"Actor Q = {final_q:.2f} < 100"))
        elif final_q < 500:
            checks.append(("Q-Value Range", "⚠️  WARN", f"Actor Q = {final_q:.2f} (acceptable but high)"))
        else:
            checks.append(("Q-Value Range", "❌ FAIL", f"Actor Q = {final_q:.2f} > 500 (explosion!)"))

    # Check 2: Actor-critic divergence
    if 'actor_q_mean' in q_results and 'q1_value' in q_results:
        actor_q = q_results['actor_q_mean']['final']
        critic_q = q_results['q1_value']['final']
        divergence = actor_q / critic_q if critic_q != 0 else float('inf')

        if divergence < 3:
            checks.append(("Actor-Critic Divergence", "✅ PASS", f"{divergence:.2f}× < 3×"))
        elif divergence < 10:
            checks.append(("Actor-Critic Divergence", "⚠️  WARN", f"{divergence:.2f}× (moderate overestimation)"))
        else:
            checks.append(("Actor-Critic Divergence", "❌ FAIL", f"{divergence:.2f}× > 10× (severe!)"))

    # Check 3: Learning signal
    if 'episode_reward' in reward_results:
        trend = reward_results['episode_reward']['trend']
        if trend > 0.001:
            checks.append(("Learning Signal", "✅ PASS", f"Positive trend ({trend:.4f})"))
        elif trend > -0.001:
            checks.append(("Learning Signal", "⚠️  WARN", f"Near-zero trend ({trend:.4f})"))
        else:
            checks.append(("Learning Signal", "❌ FAIL", f"Negative trend ({trend:.4f})"))

    # Check 4: Gradient clipping
    if 'actor_mlp_norm' in grad_results:
        max_actor_norm = grad_results['actor_mlp_norm']['max']
        if max_actor_norm <= 1.0:
            checks.append(("Actor Gradient Clipping", "✅ PASS", f"Max norm = {max_actor_norm:.6f} ≤ 1.0"))
        else:
            checks.append(("Actor Gradient Clipping", "❌ FAIL", f"Max norm = {max_actor_norm:.6f} > 1.0"))

    if 'critic_mlp_norm' in grad_results:
        max_critic_norm = grad_results['critic_mlp_norm']['max']
        if max_critic_norm <= 10.0:
            checks.append(("Critic Gradient Clipping", "✅ PASS", f"Max norm = {max_critic_norm:.6f} ≤ 10.0"))
        else:
            checks.append(("Critic Gradient Clipping", "❌ FAIL", f"Max norm = {max_critic_norm:.6f} > 10.0"))

    # Print summary table
    print(f"\n{'Check':<30} {'Status':<15} {'Details':<50}")
    print("-" * 95)
    for check_name, status, details in checks:
        print(f"{check_name:<30} {status:<15} {details:<50}")

    # Overall verdict
    print("\n" + "="*80)
    print("🎯 OVERALL VERDICT:")
    print("="*80)

    passes = sum(1 for _, status, _ in checks if "✅" in status)
    warns = sum(1 for _, status, _ in checks if "⚠️" in status)
    fails = sum(1 for _, status, _ in checks if "❌" in status)

    print(f"✅ PASS: {passes}/{len(checks)}")
    print(f"⚠️  WARN: {warns}/{len(checks)}")
    print(f"❌ FAIL: {fails}/{len(checks)}")

    if fails == 0 and warns == 0:
        print("\n🟢 RECOMMENDATION: System is performing as expected. PROCEED to 50K-100K training.")
    elif fails == 0:
        print("\n🟡 RECOMMENDATION: System is mostly healthy with minor issues. Monitor closely during 50K training.")
    else:
        print("\n🔴 RECOMMENDATION: Critical issues detected. Address before proceeding to longer training.")
        print("   See TD3_PAPER_COMPLETE_ANALYSIS.md for hyperparameter fixes.")

    return checks


def create_visualization(q_results, reward_results, loss_results, output_dir):
    """
    Create visualization plots for key metrics.
    """
    if not plt:
        print("\n⚠️  Matplotlib not available, skipping visualization")
        return

    print("\n📊 Creating visualization plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TD3 Training Metrics - 5K Steps', fontsize=16, fontweight='bold')

    # Plot 1: Q-values over time
    ax = axes[0, 0]
    if 'actor_q_mean' in q_results:
        ax.plot(q_results['actor_q_mean']['steps'], q_results['actor_q_mean']['values'],
                label='Actor Q-mean', linewidth=2, color='blue')
    if 'q1_value' in q_results:
        ax.plot(q_results['q1_value']['steps'], q_results['q1_value']['values'],
                label='Critic Q1', linewidth=2, color='green')
    if 'q2_value' in q_results:
        ax.plot(q_results['q2_value']['steps'], q_results['q2_value']['values'],
                label='Critic Q2', linewidth=2, color='orange')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Expected max (5K)')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Q-Value', fontsize=12)
    ax.set_title('Q-Value Trajectory', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode rewards
    ax = axes[0, 1]
    if 'episode_reward' in reward_results:
        steps = reward_results['episode_reward']['steps']
        values = reward_results['episode_reward']['values']
        ax.plot(steps, values, linewidth=2, color='purple', alpha=0.7)
        # Add trend line
        if len(steps) > 1:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "r--", linewidth=2, label=f'Trend: {z[0]:.4f}')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Losses
    ax = axes[1, 0]
    if 'critic_loss' in loss_results:
        ax.plot(loss_results['critic_loss']['steps'], loss_results['critic_loss']['values'],
                label='Critic Loss', linewidth=2, color='green')
    if 'actor_loss' in loss_results:
        ax.plot(loss_results['actor_loss']['steps'], loss_results['actor_loss']['values'],
                label='Actor Loss', linewidth=2, color='blue')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Episode length
    ax = axes[1, 1]
    if 'episode_length' in reward_results:
        steps = reward_results['episode_length']['steps']
        values = reward_results['episode_length']['values']
        ax.plot(steps, values, linewidth=2, color='orange', alpha=0.7)
        ax.axhline(y=np.mean(values), color='red', linestyle='--', linewidth=1,
                   alpha=0.5, label=f'Mean: {np.mean(values):.1f}')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Length (steps)', fontsize=12)
    ax.set_title('Episode Length Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, '5k_training_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")

    plt.close()


def main():
    """Main analysis function."""
    # Path to TensorBoard logs
    log_dir = PROJECT_ROOT / "data" / "logs" / "TD3_scenario_0_npcs_20_20251120-133459"

    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        print("\nSearching for alternative log directories...")
        logs_root = PROJECT_ROOT / "data" / "logs"
        if logs_root.exists():
            log_dirs = sorted([d for d in logs_root.iterdir() if d.is_dir()],
                            key=lambda x: x.stat().st_mtime, reverse=True)
            if log_dirs:
                print(f"\nFound {len(log_dirs)} log directories. Using most recent:")
                log_dir = log_dirs[0]
                print(f"✅ Using: {log_dir}")
            else:
                print("❌ No log directories found!")
                return
        else:
            print("❌ Logs root directory doesn't exist!")
            return

    print("="*80)
    print("🚀 TD3 5K STEP TRAINING ANALYSIS")
    print("="*80)
    print(f"Log Directory: {log_dir}")

    from datetime import datetime
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load TensorBoard data
    metrics, categories = load_tensorboard_data(log_dir)

    # Analyze different metric categories
    q_results = analyze_q_values(metrics)
    reward_results = analyze_rewards(metrics)
    loss_results = analyze_losses(metrics)
    grad_results = analyze_gradients(metrics)

    # Generate summary report
    checks = generate_summary_report(metrics, q_results, reward_results, loss_results, grad_results)

    # Create visualizations
    output_dir = PROJECT_ROOT / "docs" / "day-20"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_visualization(q_results, reward_results, loss_results, output_dir)

    # Save detailed CSV reports
    if pd:
        print("\n💾 Saving detailed CSV reports...")

        # Q-values CSV
        if q_results:
            q_df = pd.DataFrame({
                'step': q_results['actor_q_mean']['steps'],
                'actor_q_mean': q_results['actor_q_mean']['values'],
                'actor_q_std': q_results['actor_q_std']['values'] if 'actor_q_std' in q_results else None,
                'q1_value': q_results['q1_value']['values'] if 'q1_value' in q_results else None,
                'q2_value': q_results['q2_value']['values'] if 'q2_value' in q_results else None,
            })
            q_df.to_csv(output_dir / '5k_q_values.csv', index=False)
            print(f"✅ Saved Q-values to: {output_dir / '5k_q_values.csv'}")

        # Rewards CSV
        if reward_results:
            reward_df = pd.DataFrame({
                'step': reward_results['episode_reward']['steps'] if 'episode_reward' in reward_results else [],
                'episode_reward': reward_results['episode_reward']['values'] if 'episode_reward' in reward_results else [],
                'episode_length': reward_results['episode_length']['values'] if 'episode_length' in reward_results else [],
            })
            reward_df.to_csv(output_dir / '5k_rewards.csv', index=False)
            print(f"✅ Saved rewards to: {output_dir / '5k_rewards.csv'}")
    else:
        print("\n⚠️  Pandas not available, skipping CSV export")

    print("\n✅ Analysis complete!")
    print(f"\n📁 All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()

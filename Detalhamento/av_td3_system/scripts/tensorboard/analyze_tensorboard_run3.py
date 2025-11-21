#!/usr/bin/env python3
"""
Systematic TensorBoard Log Analysis for Run 3 (Post-Critical-Fixes)

Validates if the critical fixes from CRITICAL_FIXES_IMPLEMENTATION_SUMMARY.md worked:
1. Gradient clipping fix (CNN merged into main optimizers)
2. Hyperparameter fix (gamma=0.99, tau=0.005, lr=1e-3)
3. Q-value explosion prevention

Expected Metrics (from TD3 paper + fixes):
- Actor grad norm AFTER clip ≤ 1.0
- Critic grad norm AFTER clip ≤ 10.0
- Actor CNN grad norm AFTER clip ≤ 1.0
- Critic CNN grad norm AFTER clip ≤ 10.0
- Q-values: 0-50 initially (not millions)
- Episode lengths: Stable (not collapsing)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard not installed. Install with: pip install tensorboard")
    sys.exit(1)


class TensorBoardAnalyzer:
    """Extract and analyze metrics from TensorBoard event files"""
    
    def __init__(self, event_file: str):
        self.event_file = event_file
        self.ea = event_accumulator.EventAccumulator(event_file)
        self.ea.Reload()
        
        print(f"[LOAD] Loaded TensorBoard event file: {Path(event_file).name}")
        print(f"[LOAD] Available tags: {len(self.ea.Tags()['scalars'])} scalar metrics")
        
    def get_metric(self, tag: str) -> pd.DataFrame:
        """Extract single metric as DataFrame with [step, value, wall_time]"""
        try:
            events = self.ea.Scalars(tag)
            df = pd.DataFrame([
                {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
                for e in events
            ])
            return df
        except KeyError:
            return pd.DataFrame(columns=['step', 'value', 'wall_time'])
    
    def extract_all_metrics(self) -> Dict[str, pd.DataFrame]:
        """Extract all available metrics"""
        tags = self.ea.Tags()['scalars']
        metrics = {}
        
        print(f"\n[EXTRACT] Extracting {len(tags)} metrics...")
        for tag in tags:
            metrics[tag] = self.get_metric(tag)
            if len(metrics[tag]) > 0:
                print(f"  ✓ {tag}: {len(metrics[tag])} data points")
        
        return metrics


def analyze_gradient_clipping(metrics: Dict[str, pd.DataFrame]) -> Dict:
    """Analyze gradient clipping effectiveness (Fix #2 validation)"""
    
    print("\n" + "="*80)
    print("GRADIENT CLIPPING ANALYSIS (Fix #2: Merge CNN into Main Optimizers)")
    print("="*80)
    
    results = {
        'actor_grad_before': None,
        'actor_grad_after': None,
        'critic_grad_before': None,
        'critic_grad_after': None,
        'actor_cnn_grad_after': None,
        'critic_cnn_grad_after': None,
        'clipping_worked': False,
        'issues': []
    }
    
    # Expected limits (from td3_agent.py)
    ACTOR_LIMIT = 1.0
    CRITIC_LIMIT = 10.0
    
    # Extract gradient metrics
    grad_metrics = {
        'actor_before': 'debug/actor_grad_norm_BEFORE_clip',
        'actor_after': 'debug/actor_grad_norm_AFTER_clip',
        'critic_before': 'debug/critic_grad_norm_BEFORE_clip',
        'critic_after': 'debug/critic_grad_norm_AFTER_clip',
        'actor_cnn_after': 'debug/actor_cnn_grad_norm_AFTER_clip',
        'critic_cnn_after': 'debug/critic_cnn_grad_norm_AFTER_clip',
    }
    
    for key, tag in grad_metrics.items():
        if tag in metrics and len(metrics[tag]) > 0:
            df = metrics[tag]
            results[key.replace('_', '_grad_')] = df
            
            mean_val = df['value'].mean()
            max_val = df['value'].max()
            min_val = df['value'].min()
            std_val = df['value'].std()
            
            print(f"\n[{key.upper()}]")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Max:  {max_val:.4f}")
            print(f"  Min:  {min_val:.4f}")
            print(f"  Std:  {std_val:.4f}")
            
            # Check if clipping worked
            if 'after' in key:
                if 'actor' in key and 'cnn' not in key:
                    limit = ACTOR_LIMIT
                    metric_name = "Actor MLP"
                elif 'critic' in key and 'cnn' not in key:
                    limit = CRITIC_LIMIT
                    metric_name = "Critic MLP"
                elif 'actor_cnn' in key:
                    limit = ACTOR_LIMIT
                    metric_name = "Actor CNN"
                elif 'critic_cnn' in key:
                    limit = CRITIC_LIMIT
                    metric_name = "Critic CNN"
                else:
                    continue
                
                violations = (df['value'] > limit).sum()
                violation_pct = violations / len(df) * 100
                
                if violations > 0:
                    print(f"  ⚠️  LIMIT VIOLATIONS: {violations}/{len(df)} ({violation_pct:.1f}%) exceed {limit}")
                    results['issues'].append(f"{metric_name} exceeded {limit} in {violation_pct:.1f}% of updates")
                else:
                    print(f"  ✅ All gradients ≤ {limit}")
        else:
            print(f"\n[{key.upper()}] ❌ NOT FOUND in logs")
            results['issues'].append(f"{key} metric missing")
    
    # Overall assessment
    print("\n" + "-"*80)
    if len(results['issues']) == 0:
        print("✅ GRADIENT CLIPPING FIX: SUCCESS")
        print("   All gradient norms within expected limits")
        results['clipping_worked'] = True
    else:
        print("❌ GRADIENT CLIPPING FIX: FAILED")
        print("   Issues detected:")
        for issue in results['issues']:
            print(f"   - {issue}")
    
    return results


def analyze_hyperparameters(metrics: Dict[str, pd.DataFrame]) -> Dict:
    """Analyze if hyperparameter fixes improved stability (Fix #1 validation)"""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER FIX VALIDATION (Fix #1: gamma=0.99, tau=0.005, lr=1e-3)")
    print("="*80)
    
    results = {
        'q_values_stable': False,
        'episode_lengths_stable': False,
        'rewards_improving': False,
        'issues': []
    }
    
    # Q-value analysis
    q_tags = ['train/q1_value', 'train/q2_value', 'train/actor_q_value']
    print("\n[Q-VALUE STABILITY]")
    for tag in q_tags:
        if tag in metrics and len(metrics[tag]) > 0:
            df = metrics[tag]
            mean_val = df['value'].mean()
            max_val = df['value'].max()
            min_val = df['value'].min()
            
            print(f"  {tag}:")
            print(f"    Mean: {mean_val:.2f}, Range: [{min_val:.2f}, {max_val:.2f}]")
            
            # Check for explosion (previous issue: Q-values reached 1,796,760)
            if max_val > 1000:
                results['issues'].append(f"{tag} exploded to {max_val:.0f}")
                print(f"    ❌ EXPLOSION DETECTED: {max_val:.0f}")
            elif max_val > 100:
                results['issues'].append(f"{tag} potentially unstable (max={max_val:.0f})")
                print(f"    ⚠️  HIGH VALUES: {max_val:.0f}")
            else:
                print(f"    ✅ Stable range")
        else:
            print(f"  {tag}: ❌ NOT FOUND")
    
    # Episode length analysis
    if 'train/episode_length' in metrics:
        df = metrics['train/episode_length']
        mean_len = df['value'].mean()
        std_len = df['value'].std()
        
        print(f"\n[EPISODE LENGTH]")
        print(f"  Mean: {mean_len:.1f} steps")
        print(f"  Std:  {std_len:.1f} steps")
        
        # Check for collapse (previous issue: 50 → 2 steps)
        if mean_len < 10:
            results['issues'].append(f"Episode collapse (mean={mean_len:.1f} steps)")
            print(f"  ❌ COLLAPSE DETECTED")
        else:
            results['episode_lengths_stable'] = True
            print(f"  ✅ No collapse detected")
    
    # Reward analysis
    if 'train/episode_reward' in metrics:
        df = metrics['train/episode_reward']
        
        # Check if rewards are improving over time
        if len(df) > 10:
            early_mean = df.iloc[:len(df)//3]['value'].mean()
            late_mean = df.iloc[2*len(df)//3:]['value'].mean()
            
            print(f"\n[REWARD TREND]")
            print(f"  Early third: {early_mean:.2f}")
            print(f"  Late third:  {late_mean:.2f}")
            print(f"  Change:      {late_mean - early_mean:+.2f}")
            
            if late_mean > early_mean:
                results['rewards_improving'] = True
                print(f"  ✅ Rewards improving")
            else:
                print(f"  ⚠️  Rewards not improving")
    
    # Overall assessment
    print("\n" + "-"*80)
    if len(results['issues']) == 0 and results['episode_lengths_stable']:
        print("✅ HYPERPARAMETER FIX: SUCCESS")
        print("   Q-values stable, no episode collapse")
    else:
        print("❌ HYPERPARAMETER FIX: ISSUES DETECTED")
        for issue in results['issues']:
            print(f"   - {issue}")
    
    return results


def analyze_training_progress(metrics: Dict[str, pd.DataFrame]) -> Dict:
    """Analyze overall training progress and convergence"""
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS ANALYSIS")
    print("="*80)
    
    results = {
        'total_steps': 0,
        'total_episodes': 0,
        'avg_episode_length': 0,
        'avg_episode_reward': 0,
        'actor_loss_trend': None,
        'critic_loss_trend': None
    }
    
    # Extract training duration
    if 'train/episode_reward' in metrics:
        df = metrics['train/episode_reward']
        results['total_episodes'] = len(df)
        results['avg_episode_reward'] = df['value'].mean()
        
        # Estimate total steps from last step value
        if len(df) > 0:
            results['total_steps'] = int(df.iloc[-1]['step'])
    
    if 'train/episode_length' in metrics:
        df = metrics['train/episode_length']
        results['avg_episode_length'] = df['value'].mean()
    
    print(f"\n[TRAINING DURATION]")
    print(f"  Total steps: {results['total_steps']:,}")
    print(f"  Total episodes: {results['total_episodes']}")
    print(f"  Avg episode length: {results['avg_episode_length']:.1f} steps")
    print(f"  Avg episode reward: {results['avg_episode_reward']:.2f}")
    
    # Loss trends
    if 'train/actor_loss' in metrics:
        df = metrics['train/actor_loss']
        results['actor_loss_trend'] = df['value'].mean()
        print(f"\n[ACTOR LOSS]")
        print(f"  Mean: {df['value'].mean():.4f}")
        print(f"  Note: Should be negative (maximizing Q-values)")
    
    if 'train/critic_loss' in metrics:
        df = metrics['train/critic_loss']
        results['critic_loss_trend'] = df['value'].mean()
        print(f"\n[CRITIC LOSS]")
        print(f"  Mean: {df['value'].mean():.4f}")
    
    return results


def create_visualization(metrics: Dict[str, pd.DataFrame], output_dir: Path):
    """Create comprehensive visualization plots"""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Gradient Clipping Validation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gradient Clipping Analysis (Post-Fix Validation)', fontsize=16, fontweight='bold')
    
    grad_pairs = [
        ('debug/actor_grad_norm_BEFORE_clip', 'debug/actor_grad_norm_AFTER_clip', 'Actor MLP', 1.0),
        ('debug/critic_grad_norm_BEFORE_clip', 'debug/critic_grad_norm_AFTER_clip', 'Critic MLP', 10.0),
        ('debug/actor_cnn_grad_norm_AFTER_clip', None, 'Actor CNN', 1.0),
        ('debug/critic_cnn_grad_norm_AFTER_clip', None, 'Critic CNN', 10.0),
    ]
    
    for idx, (before_tag, after_tag, title, limit) in enumerate(grad_pairs):
        ax = axes[idx // 2, idx % 2]
        
        if before_tag in metrics and len(metrics[before_tag]) > 0:
            df_before = metrics[before_tag]
            ax.plot(df_before['step'], df_before['value'], label='Before Clip', alpha=0.7, linewidth=1)
        
        if after_tag and after_tag in metrics and len(metrics[after_tag]) > 0:
            df_after = metrics[after_tag]
            ax.plot(df_after['step'], df_after['value'], label='After Clip', alpha=0.9, linewidth=2)
        elif not after_tag and before_tag in metrics:
            # For CNN metrics (only AFTER exists)
            pass
        
        ax.axhline(y=limit, color='r', linestyle='--', label=f'Limit ({limit})', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_clipping_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: gradient_clipping_analysis.png")
    plt.close()
    
    # 2. Q-Value Stability
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Value Stability Analysis', fontsize=16, fontweight='bold')
    
    q_tags = [
        ('train/q1_value', 'Q1 Value'),
        ('train/q2_value', 'Q2 Value'),
        ('train/actor_q_value', 'Actor Q Value'),
        ('train/episode_reward', 'Episode Reward'),
    ]
    
    for idx, (tag, title) in enumerate(q_tags):
        ax = axes[idx // 2, idx % 2]
        
        if tag in metrics and len(metrics[tag]) > 0:
            df = metrics[tag]
            ax.plot(df['step'], df['value'], alpha=0.7, linewidth=1)
            
            # Add smoothed trend
            if len(df) > 10:
                window = min(50, len(df) // 10)
                smoothed = df['value'].rolling(window=window, center=True).mean()
                ax.plot(df['step'], smoothed, color='red', linewidth=2, label='Trend')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'q_value_stability.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: q_value_stability.png")
    plt.close()
    
    # 3. Training Metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
    
    metric_tags = [
        ('train/episode_length', 'Episode Length'),
        ('train/actor_loss', 'Actor Loss'),
        ('train/critic_loss', 'Critic Loss'),
        ('train/episode_reward', 'Episode Reward'),
    ]
    
    for idx, (tag, title) in enumerate(metric_tags):
        ax = axes[idx // 2, idx % 2]
        
        if tag in metrics and len(metrics[tag]) > 0:
            df = metrics[tag]
            ax.plot(df['step'], df['value'], alpha=0.6, linewidth=1)
            
            # Add smoothed trend
            if len(df) > 10:
                window = min(50, len(df) // 10)
                smoothed = df['value'].rolling(window=window, center=True).mean()
                ax.plot(df['step'], smoothed, color='red', linewidth=2, label='Trend')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Value')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: training_metrics.png")
    plt.close()


def main():
    """Main analysis pipeline"""
    
    print("="*80)
    print("TENSORBOARD LOG SYSTEMATIC ANALYSIS - RUN 3 (POST-CRITICAL-FIXES)")
    print("="*80)
    print()
    print("Context: Validating fixes from CRITICAL_FIXES_IMPLEMENTATION_SUMMARY.md")
    print("- Fix #1: Hyperparameters (gamma=0.99, tau=0.005, lr=1e-3)")
    print("- Fix #2: Gradient clipping (CNN merged into main optimizers)")
    print("- Fix #3: Q-value explosion prevention")
    print()
    
    # Locate event file
    event_file = Path("docs/day-20/run3/TD3_scenario_0_npcs_20_20251120-190526/events.out.tfevents.1763665526.danielterra.1.0")
    
    if not event_file.exists():
        print(f"ERROR: Event file not found: {event_file}")
        return 1
    
    # Initialize analyzer
    analyzer = TensorBoardAnalyzer(str(event_file))
    
    # Extract all metrics
    metrics = analyzer.extract_all_metrics()
    
    # Run analyses
    gradient_results = analyze_gradient_clipping(metrics)
    hyperparam_results = analyze_hyperparameters(metrics)
    progress_results = analyze_training_progress(metrics)
    
    # Create visualizations
    output_dir = Path("docs/day-20/run3/analysis")
    create_visualization(metrics, output_dir)
    
    # Generate summary report
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    print("\n[GRADIENT CLIPPING FIX]")
    if gradient_results['clipping_worked']:
        print("  ✅ SUCCESS: All gradients within limits")
    else:
        print("  ❌ FAILED: Issues detected")
        for issue in gradient_results['issues']:
            print(f"     - {issue}")
    
    print("\n[HYPERPARAMETER FIX]")
    if len(hyperparam_results['issues']) == 0:
        print("  ✅ SUCCESS: Q-values stable, no collapse")
    else:
        print("  ⚠️  PARTIAL: Some issues remain")
        for issue in hyperparam_results['issues']:
            print(f"     - {issue}")
    
    print("\n[TRAINING PROGRESS]")
    print(f"  Total steps: {progress_results['total_steps']:,}")
    print(f"  Total episodes: {progress_results['total_episodes']}")
    print(f"  Avg reward: {progress_results['avg_episode_reward']:.2f}")
    
    print("\n[OUTPUT FILES]")
    print(f"  Analysis directory: {output_dir}")
    print(f"  Visualizations: 3 PNG files generated")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

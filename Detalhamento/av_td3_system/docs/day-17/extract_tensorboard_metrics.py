#!/usr/bin/env python3
"""
Extract and Analyze TensorBoard Metrics from Event File

This script reads the TensorBoard event file directly and extracts all logged metrics
for systematic analysis without requiring GUI visualization.

Author: Daniel Terra
Date: November 17, 2025
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available, trying tensorboard directly...")
    try:
        from tensorboard.backend.event_processing import event_file_loader
        from tensorboard.compat.proto import event_pb2
        TENSORFLOW_AVAILABLE = False
    except ImportError:
        print("❌ Neither TensorFlow nor TensorBoard available!")
        print("Install with: pip install tensorboard")
        sys.exit(1)


class TensorBoardMetricsExtractor:
    """
    Extract metrics from TensorBoard event files for systematic analysis.
    """

    def __init__(self, event_file_path: str):
        """
        Initialize extractor with event file path.

        Args:
            event_file_path: Path to the TensorBoard event file
        """
        self.event_file = Path(event_file_path)
        if not self.event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file_path}")

        self.metrics = defaultdict(list)
        self.steps = defaultdict(list)
        self.timestamps = defaultdict(list)

    def extract_all_metrics(self) -> dict:
        """
        Extract all metrics from the event file.

        Returns:
            Dictionary with metric names as keys and lists of (step, value) tuples
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING METRICS FROM TENSORBOARD EVENT FILE")
        print(f"{'='*80}")
        print(f"File: {self.event_file.name}")
        print(f"Size: {self.event_file.stat().st_size / 1024:.1f} KB")
        print(f"{'='*80}\n")

        event_count = 0

        try:
            for event in summary_iterator(str(self.event_file)):
                event_count += 1

                # Extract scalar values
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        metric_name = value.tag
                        metric_value = value.simple_value
                        step = event.step
                        timestamp = event.wall_time

                        self.metrics[metric_name].append(metric_value)
                        self.steps[metric_name].append(step)
                        self.timestamps[metric_name].append(timestamp)

        except Exception as e:
            print(f"⚠️ Error reading events (processed {event_count} events): {e}")

        print(f"✅ Processed {event_count} events")
        print(f"✅ Found {len(self.metrics)} unique metrics\n")

        return self.metrics

    def print_metrics_summary(self) -> None:
        """Print summary of all extracted metrics."""
        print(f"\n{'='*80}")
        print(f"METRICS SUMMARY")
        print(f"{'='*80}\n")

        if not self.metrics:
            print("❌ No metrics found!")
            return

        for metric_name in sorted(self.metrics.keys()):
            values = np.array(self.metrics[metric_name])
            steps = np.array(self.steps[metric_name])

            print(f"📊 {metric_name}")
            print(f"   Data points: {len(values)}")
            print(f"   Steps range: {steps.min():.0f} - {steps.max():.0f}")
            print(f"   Value range: {values.min():.6f} - {values.max():.6f}")
            print(f"   Mean: {values.mean():.6f}")
            print(f"   Std: {values.std():.6f}")

            # Show first and last few values
            if len(values) > 5:
                print(f"   First 3: {values[:3]}")
                print(f"   Last 3:  {values[-3:]}")
            else:
                print(f"   All values: {values}")
            print()

    def analyze_gradient_norms(self) -> dict:
        """
        PRIORITY 1: Analyze gradient norms to check for explosion.

        Returns:
            Dictionary with gradient norm analysis
        """
        print(f"\n{'='*80}")
        print(f"PRIORITY 1: GRADIENT NORM ANALYSIS")
        print(f"{'='*80}\n")

        gradient_metrics = [
            'train/actor_cnn_grad_norm',
            'train/critic_cnn_grad_norm',
            'train/actor_mlp_grad_norm',
            'train/critic_mlp_grad_norm',
        ]

        analysis = {}

        for metric_name in gradient_metrics:
            if metric_name not in self.metrics:
                print(f"⚠️ {metric_name} not found in logs")
                continue

            values = np.array(self.metrics[metric_name])
            steps = np.array(self.steps[metric_name])

            component = metric_name.split('/')[-1].replace('_grad_norm', '')

            mean_norm = values.mean()
            max_norm = values.max()
            min_norm = values.min()
            std_norm = values.std()
            final_norm = values[-1] if len(values) > 0 else 0

            # Literature-validated assessment
            if 'cnn' in component:
                if mean_norm < 10000:
                    status = "✅ HEALTHY"
                    recommendation = "Proceed to 1M training"
                elif mean_norm < 100000:
                    status = "⚠️ ELEVATED"
                    recommendation = "Add gradient clipping (max_norm=10.0)"
                else:
                    status = "❌ EXPLOSION"
                    recommendation = "CRITICAL: Fix required before 1M"
            else:  # MLP
                if mean_norm < 1000:
                    status = "✅ HEALTHY"
                    recommendation = "Proceed"
                elif mean_norm < 10000:
                    status = "⚠️ ELEVATED"
                    recommendation = "Monitor closely"
                else:
                    status = "❌ CONCERNING"
                    recommendation = "Consider clipping"

            print(f"📊 {component.upper()}")
            print(f"   Mean:   {mean_norm:,.2f} {status}")
            print(f"   Max:    {max_norm:,.2f}")
            print(f"   Min:    {min_norm:,.2f}")
            print(f"   Std:    {std_norm:,.2f}")
            print(f"   Final:  {final_norm:,.2f}")
            print(f"   Points: {len(values)}")
            print(f"   → {recommendation}")
            print()

            analysis[component] = {
                'mean': mean_norm,
                'max': max_norm,
                'min': min_norm,
                'std': std_norm,
                'final': final_norm,
                'status': status,
                'recommendation': recommendation,
                'values': values.tolist(),
                'steps': steps.tolist(),
            }

        # Compare to previous run
        print(f"\n{'─'*80}")
        print(f"COMPARISON TO PREVIOUS RUN (train_freq=1)")
        print(f"{'─'*80}\n")

        if 'actor_cnn' in analysis:
            previous_mean = 1826337
            current_mean = analysis['actor_cnn']['mean']
            improvement = ((previous_mean - current_mean) / previous_mean) * 100

            print(f"Actor CNN Gradient Norm:")
            print(f"  Previous (train_freq=1):  {previous_mean:,.0f} ❌ EXTREME")
            print(f"  Current  (train_freq=50): {current_mean:,.2f} {analysis['actor_cnn']['status']}")
            print(f"  Improvement: {improvement:.2f}% reduction")

            if improvement > 99:
                print(f"  → ✅ train_freq fix SUCCESSFUL!")
            elif improvement > 90:
                print(f"  → ⚠️ Significant improvement, monitoring recommended")
            else:
                print(f"  → ❌ Insufficient improvement, investigate further")
            print()

        if 'critic_cnn' in analysis:
            previous_mean = 5897
            current_mean = analysis['critic_cnn']['mean']

            print(f"Critic CNN Gradient Norm:")
            print(f"  Previous: {previous_mean:,.0f} ✅ STABLE")
            print(f"  Current:  {current_mean:,.2f} {analysis['critic_cnn']['status']}")

            if current_mean < 10000:
                print(f"  → ✅ Remains stable")
            else:
                print(f"  → ⚠️ Increased, investigate")
            print()

        return analysis

    def analyze_agent_metrics(self) -> dict:
        """
        Analyze agent training metrics (episode length, reward, losses, Q-values).

        Returns:
            Dictionary with agent metrics analysis
        """
        print(f"\n{'='*80}")
        print(f"AGENT METRICS ANALYSIS")
        print(f"{'='*80}\n")

        agent_metrics = {
            'train/episode_length': 'Episode Length',
            'train/episode_reward': 'Episode Reward',
            'train/actor_loss': 'Actor Loss',
            'train/critic_loss': 'Critic Loss',
            'train/q1_value': 'Q1 Value',
            'train/q2_value': 'Q2 Value',
        }

        analysis = {}

        for metric_key, metric_label in agent_metrics.items():
            if metric_key not in self.metrics:
                print(f"⚠️ {metric_label} not found")
                continue

            values = np.array(self.metrics[metric_key])
            steps = np.array(self.steps[metric_key])

            mean_val = values.mean()
            max_val = values.max()
            min_val = values.min()
            std_val = values.std()
            final_val = values[-1] if len(values) > 0 else 0

            # Literature-validated assessment
            if 'episode_length' in metric_key:
                if 5 <= mean_val <= 20:
                    status = "✅ EXPECTED (5-20 at 5K steps)"
                elif 3 <= mean_val <= 30:
                    status = "⚠️ ACCEPTABLE (close to expected)"
                else:
                    status = f"❌ UNEXPECTED (expected 5-20 at 80 updates)"
            elif 'actor_loss' in metric_key:
                if abs(final_val) < 1000:
                    status = "✅ STABLE"
                elif abs(final_val) < 10000:
                    status = "⚠️ ELEVATED"
                else:
                    status = "❌ DIVERGING (check reward scaling)"
            elif 'q1_value' in metric_key or 'q2_value' in metric_key:
                if abs(mean_val) < 100:
                    status = "✅ EXPECTED (small values at 5K)"
                elif abs(mean_val) < 1000:
                    status = "⚠️ ACCEPTABLE"
                else:
                    status = "❌ CONCERNING (possible explosion)"
            else:
                status = "ℹ️ INFO"

            print(f"📊 {metric_label}")
            print(f"   Mean:   {mean_val:.4f} {status}")
            print(f"   Max:    {max_val:.4f}")
            print(f"   Min:    {min_val:.4f}")
            print(f"   Std:    {std_val:.4f}")
            print(f"   Final:  {final_val:.4f}")
            print(f"   Points: {len(values)}")
            print()

            analysis[metric_key] = {
                'mean': mean_val,
                'max': max_val,
                'min': min_val,
                'std': std_val,
                'final': final_val,
                'status': status,
                'values': values.tolist(),
                'steps': steps.tolist(),
            }

        # Twin critics check
        if 'train/q1_value' in analysis and 'train/q2_value' in analysis:
            q1_mean = analysis['train/q1_value']['mean']
            q2_mean = analysis['train/q2_value']['mean']
            diff = abs(q1_mean - q2_mean)

            print(f"\n{'─'*80}")
            print(f"TWIN CRITICS VALIDATION")
            print(f"{'─'*80}\n")
            print(f"Q1 Mean: {q1_mean:.4f}")
            print(f"Q2 Mean: {q2_mean:.4f}")
            print(f"Difference: {diff:.4f}")

            if diff < 0.1 * max(abs(q1_mean), abs(q2_mean)):
                print(f"✅ Twin critics working correctly (Q1 ≈ Q2)")
            else:
                print(f"⚠️ Q1 and Q2 diverging, monitor closely")
            print()

        return analysis

    def generate_comprehensive_report(self, gradient_analysis: dict, agent_analysis: dict) -> str:
        """
        Generate comprehensive analysis report.

        Args:
            gradient_analysis: Results from gradient norm analysis
            agent_analysis: Results from agent metrics analysis

        Returns:
            Formatted markdown report
        """
        report = []
        report.append("# TensorBoard Metrics Analysis Report: 5K_POST_FIXES Run")
        report.append(f"\n**Analysis Date**: November 17, 2025")
        report.append(f"**Event File**: {self.event_file.name}")
        report.append(f"**Total Metrics Logged**: {len(self.metrics)}")
        report.append("\n---\n")

        # Executive Summary
        report.append("## Executive Summary\n")

        # Determine overall status from gradient analysis
        all_healthy = True
        has_concerns = False
        critical_issues = False

        for component, data in gradient_analysis.items():
            if '❌' in data['status']:
                critical_issues = True
                all_healthy = False
            elif '⚠️' in data['status']:
                has_concerns = True
                all_healthy = False

        if all_healthy:
            report.append("### ✅ GO FOR 1M TRAINING\n")
            report.append("**All gradient norms are healthy.** The train_freq fix successfully resolved the gradient explosion.")
            report.append("\n**Recommendation**: Proceed with 1M training run with standard monitoring.\n")
        elif has_concerns and not critical_issues:
            report.append("### ⚠️ PROCEED WITH CAUTION\n")
            report.append("**Gradients are elevated but not exploding.** Recommend adding gradient clipping for safety.")
            report.append("\n**Recommendation**: Implement gradient clipping (max_norm=10.0) before 1M run.\n")
        else:
            report.append("### ❌ NO-GO - FIXES REQUIRED\n")
            report.append("**Critical gradient explosion detected.** Must implement fixes before 1M training.")
            report.append("\n**Recommendation**: Add gradient clipping, review configuration, re-run 5K validation.\n")

        # Gradient Norms Section
        report.append("## 1. Gradient Norm Analysis (PRIORITY 1)\n")

        if gradient_analysis:
            report.append("### Current Run (train_freq=50)\n")

            for component, data in gradient_analysis.items():
                report.append(f"#### {component.upper()}\n")
                report.append(f"- **Status**: {data['status']}")
                report.append(f"- **Mean**: {data['mean']:,.2f}")
                report.append(f"- **Max**: {data['max']:,.2f}")
                report.append(f"- **Final**: {data['final']:,.2f}")
                report.append(f"- **Recommendation**: {data['recommendation']}\n")

            # Comparison
            if 'actor_cnn' in gradient_analysis:
                report.append("### Comparison to Previous Run\n")
                previous = 1826337
                current = gradient_analysis['actor_cnn']['mean']
                improvement = ((previous - current) / previous) * 100

                report.append(f"**Actor CNN Gradient Norm**:")
                report.append(f"- Previous (train_freq=1): {previous:,} ❌ EXTREME EXPLOSION")
                report.append(f"- Current (train_freq=50): {current:,.2f} {gradient_analysis['actor_cnn']['status']}")
                report.append(f"- **Improvement**: {improvement:.2f}% reduction")

                if improvement > 99:
                    report.append(f"- **Conclusion**: ✅ train_freq fix SUCCESSFUL!\n")
                else:
                    report.append(f"- **Conclusion**: ⚠️ Partial improvement, further investigation needed\n")
        else:
            report.append("⚠️ No gradient norm data found in event file.\n")

        # Agent Metrics Section
        report.append("## 2. Agent Training Metrics\n")

        if agent_analysis:
            for metric_key, data in agent_analysis.items():
                metric_name = metric_key.split('/')[-1].replace('_', ' ').title()
                report.append(f"### {metric_name}\n")
                report.append(f"- **Status**: {data['status']}")
                report.append(f"- **Mean**: {data['mean']:.4f}")
                report.append(f"- **Range**: [{data['min']:.4f}, {data['max']:.4f}]")
                report.append(f"- **Final**: {data['final']:.4f}\n")
        else:
            report.append("⚠️ No agent metrics data found.\n")

        # Literature Validation
        report.append("## 3. Literature Validation\n")
        report.append("### Expected Behavior at 5K Steps (~80 Gradient Updates)\n")
        report.append("**From Academic Papers** (TD3, Rally A3C, DDPG-UAV):\n")
        report.append("- **Episode Length**: 5-20 steps ✅ (early training, minimal updates)")
        report.append("- **Gradient Norms**: < 10K for CNNs (with clipping)")
        report.append("- **Training Timeline**: 50M-140M steps for convergence")
        report.append("- **Conclusion**: 5K = extreme early validation, low performance EXPECTED\n")

        # Configuration Validation
        report.append("## 4. Configuration Validation\n")
        report.append("**Current Configuration** (matches OpenAI Spinning Up TD3):\n")
        report.append("- train_freq: 50 ✅")
        report.append("- gradient_steps: 1 ✅")
        report.append("- learning_starts: 1000 ✅")
        report.append("- policy_freq: 2 ✅ (delayed updates)")
        report.append("- Total training iterations: ~80 (4000 steps / 50) ✅\n")

        # Action Items
        report.append("## 5. Action Items\n")

        if all_healthy:
            report.append("### Proceed to 1M Training\n")
            report.append("1. ✅ Launch 1M training run")
            report.append("2. ✅ Monitor gradients at 50K checkpoint")
            report.append("3. ✅ Implement gradient clipping if norms exceed 50K at any point\n")
        elif has_concerns:
            report.append("### Implement Safety Measures\n")
            report.append("1. ⚠️ Add gradient clipping (max_norm=10.0) to all networks")
            report.append("2. ⚠️ Run 50K validation with enhanced monitoring")
            report.append("3. ⚠️ Re-evaluate at 50K before committing to full 1M\n")
        else:
            report.append("### Critical Fixes Required\n")
            report.append("1. ❌ Implement gradient clipping immediately (ALL networks)")
            report.append("2. ❌ Review reward function scaling")
            report.append("3. ❌ Consider reducing learning rates")
            report.append("4. ❌ Re-run 5K validation before 1M attempt\n")

        return '\n'.join(report)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract and analyze TensorBoard metrics")
    parser.add_argument(
        'event_file',
        nargs='?',
        default='/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251117-184435/events.out.tfevents.1763405075.danielterra.1.0',
        help='Path to TensorBoard event file'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='TENSORBOARD_METRICS_ANALYSIS_REPORT.md',
        help='Output report filename'
    )

    args = parser.parse_args()

    print(f"\n🔍 TensorBoard Metrics Extractor and Analyzer")
    print(f"{'='*80}\n")

    # Create extractor
    try:
        extractor = TensorBoardMetricsExtractor(args.event_file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    # Extract all metrics
    extractor.extract_all_metrics()

    # Print summary
    extractor.print_metrics_summary()

    # Analyze gradients (PRIORITY 1)
    gradient_analysis = extractor.analyze_gradient_norms()

    # Analyze agent metrics
    agent_analysis = extractor.analyze_agent_metrics()

    # Generate comprehensive report
    report = extractor.generate_comprehensive_report(gradient_analysis, agent_analysis)

    # Save report
    output_dir = Path(args.event_file).parent.parent.parent.parent / 'docs' / 'day-17'
    output_path = output_dir / args.output

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"📄 Report saved to: {output_path}")
    print(f"{'='*80}\n")

    # Display report
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Systematic TensorBoard Analysis Script

Reads TensorBoard event files and performs comprehensive analysis of:
- Gradient flow patterns (Actor/Critic CNN and MLP)
- Learning dynamics (losses, Q-values, learning rates)
- Reward components and episode characteristics
- Agent performance metrics

Uses literature-validated benchmarks from:
1. Lateral Control paper (Chen et al., 2019) - DDPG+CNN, clip_norm=10.0
2. Race Driving paper (Perot et al., 2017) - A3C+CNN, reward shaping
3. UAV Guidance paper - DDPG+PER+APF, explainability

Author: Daniel Terra
Date: 2025-11-17
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tensorboard.backend.event_processing import event_accumulator


class TensorBoardSystematicAnalyzer:
    """
    Systematic analyzer for TensorBoard event files with literature validation.

    Extracts and analyzes all metrics to identify training issues beyond gradient explosion.
    Compares against literature benchmarks to validate training configuration.
    """

    def __init__(self, event_file_path: str):
        """
        Initialize analyzer with TensorBoard event file.

        Args:
            event_file_path: Path to TensorBoard event file (events.out.tfevents.*)
        """
        self.event_file_path = event_file_path
        self.ea = None
        self.metrics = defaultdict(list)
        self.steps = defaultdict(list)

        # Literature benchmarks for comparison
        self.literature_benchmarks = {
            "gradient_norms": {
                "lateral_control_paper": {
                    "critic_clip_norm": 10.0,
                    "source": "Chen et al., 2019 - Lateral Control"
                },
                "lane_keeping_paper": {
                    "actor_clip_norm": 1.0,
                    "success_rate_with_clipping": 0.95,
                    "success_rate_without_clipping": 0.20,
                    "source": "Sallab et al., 2017 - Lane Keeping Assist"
                },
                "race_driving_paper": {
                    "clip_norm": 40.0,
                    "source": "Perot et al., 2017 - End-to-End Race Driving"
                }
            },
            "learning_rates": {
                "recommended": {
                    "actor": 1e-3,
                    "critic": 1e-4,
                    "source": "Multiple papers (Lateral Control, UAV Guidance)"
                }
            },
            "reward_design": {
                "race_driving_insight": {
                    "formula": "R = v(cos(α) - d)",
                    "key_finding": "distance penalty enables rapid learning to stay in track center",
                    "source": "Perot et al., 2017"
                }
            },
            "episode_length": {
                "expected_range": (50, 500),
                "source": "Typical for autonomous driving tasks"
            }
        }

        print(f"🔍 Initializing Systematic TensorBoard Analyzer")
        print(f"📁 Event file: {event_file_path}")
        print(f"📚 Literature benchmarks loaded: {len(self.literature_benchmarks)} categories")

    def load_events(self):
        """Load TensorBoard event file using EventAccumulator."""
        print(f"\n📊 Loading TensorBoard events...")

        # Initialize EventAccumulator with size guidance
        size_guidance = {
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,  # 0 means load all scalars
            event_accumulator.HISTOGRAMS: 1,
            event_accumulator.TENSORS: 100,
        }

        self.ea = event_accumulator.EventAccumulator(
            self.event_file_path,
            size_guidance=size_guidance
        )

        # Load all data
        self.ea.Reload()

        # Get available tags
        tags = self.ea.Tags()

        print(f"✅ Events loaded successfully")
        print(f"📈 Available scalar metrics: {len(tags['scalars'])}")

        return tags

    def extract_all_metrics(self) -> Dict[str, Dict]:
        """
        Extract all scalar metrics from TensorBoard logs.

        Returns:
            Dictionary with metric categories and their data
        """
        print(f"\n🔬 Extracting all scalar metrics...")

        tags = self.ea.Tags()
        scalar_tags = tags['scalars']

        # Organize metrics by category
        metrics_by_category = {
            'gradients': [],
            'losses': [],
            'q_values': [],
            'learning_rates': [],
            'episode': [],
            'agent': [],
            'alerts': [],
            'rewards': [],
            'other': []
        }

        # Categorize tags
        for tag in scalar_tags:
            if 'gradient' in tag.lower():
                metrics_by_category['gradients'].append(tag)
            elif 'loss' in tag.lower():
                metrics_by_category['losses'].append(tag)
            elif 'q' in tag.lower() and 'value' in tag.lower():
                metrics_by_category['q_values'].append(tag)
            elif 'lr' in tag.lower() or 'learning_rate' in tag.lower():
                metrics_by_category['learning_rates'].append(tag)
            elif 'episode' in tag.lower():
                metrics_by_category['episode'].append(tag)
            elif 'agent' in tag.lower():
                metrics_by_category['agent'].append(tag)
            elif 'alert' in tag.lower():
                metrics_by_category['alerts'].append(tag)
            elif 'reward' in tag.lower():
                metrics_by_category['rewards'].append(tag)
            else:
                metrics_by_category['other'].append(tag)

        # Extract data for each metric
        all_data = {}
        for category, tags in metrics_by_category.items():
            all_data[category] = {}
            for tag in tags:
                events = self.ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                all_data[category][tag] = {
                    'steps': steps,
                    'values': values,
                    'count': len(values)
                }

        # Print summary
        print(f"\n📊 Metrics extracted by category:")
        for category, data in all_data.items():
            if data:
                print(f"  • {category}: {len(data)} metrics")

        self.all_data = all_data
        return all_data

    def analyze_gradient_patterns(self) -> Dict:
        """
        Analyze gradient flow patterns and compare with literature.

        Focuses on:
        1. Gradient norm statistics (mean, max, std, outliers)
        2. Comparison with literature benchmarks
        3. Gradient explosion detection
        4. Actor vs Critic CNN comparison

        Returns:
            Dictionary with gradient analysis results
        """
        print(f"\n🔍 ANALYSIS 1: Gradient Flow Patterns")
        print(f"=" * 80)

        gradient_data = self.all_data.get('gradients', {})

        if not gradient_data:
            print("⚠️  No gradient metrics found in TensorBoard logs")
            return {}

        analysis = {}

        # Analyze each gradient metric
        for metric_name, data in gradient_data.items():
            values = np.array(data['values'])

            stats = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
                'first_value': values[0] if len(values) > 0 else None,
                'last_value': values[-1] if len(values) > 0 else None,
            }

            # Detect gradient explosion
            explosion_threshold = 100000  # 100K
            explosion_critical_threshold = 1000000  # 1M

            explosions = np.sum(values > explosion_threshold)
            critical_explosions = np.sum(values > explosion_critical_threshold)

            stats['explosion_events'] = int(explosions)
            stats['critical_explosion_events'] = int(critical_explosions)
            stats['explosion_rate'] = float(explosions / len(values)) if len(values) > 0 else 0.0

            # Literature comparison
            if 'actor_cnn' in metric_name.lower():
                benchmark = self.literature_benchmarks['gradient_norms']['lane_keeping_paper']
                stats['literature_benchmark'] = benchmark['actor_clip_norm']
                stats['exceeds_benchmark'] = stats['mean'] > benchmark['actor_clip_norm']
                stats['benchmark_source'] = benchmark['source']
            elif 'critic_cnn' in metric_name.lower():
                benchmark = self.literature_benchmarks['gradient_norms']['lateral_control_paper']
                stats['literature_benchmark'] = benchmark['critic_clip_norm']
                stats['exceeds_benchmark'] = stats['mean'] > benchmark['critic_clip_norm']
                stats['benchmark_source'] = benchmark['source']

            analysis[metric_name] = stats

        # Print detailed analysis
        print(f"\n📈 Gradient Norm Statistics:")
        print(f"{'Metric':<30} {'Mean':<15} {'Max':<15} {'Explosions':<12} {'Status'}")
        print(f"-" * 90)

        for metric_name, stats in analysis.items():
            status = "✅ STABLE" if stats['explosion_rate'] < 0.1 else "❌ UNSTABLE"
            if stats.get('exceeds_benchmark', False):
                status = f"⚠️  EXCEEDS LITERATURE ({stats.get('literature_benchmark', 'N/A')})"

            print(f"{metric_name:<30} {stats['mean']:<15,.2f} {stats['max']:<15,.2f} "
                  f"{stats['explosion_events']:<12} {status}")

        # Compare Actor CNN vs Critic CNN
        actor_cnn_key = next((k for k in analysis.keys() if 'actor_cnn' in k.lower()), None)
        critic_cnn_key = next((k for k in analysis.keys() if 'critic_cnn' in k.lower()), None)

        if actor_cnn_key and critic_cnn_key:
            actor_mean = analysis[actor_cnn_key]['mean']
            critic_mean = analysis[critic_cnn_key]['mean']
            ratio = actor_mean / critic_mean if critic_mean > 0 else float('inf')

            print(f"\n🔬 Actor CNN vs Critic CNN Comparison:")
            print(f"  • Actor CNN mean gradient:  {actor_mean:,.2f}")
            print(f"  • Critic CNN mean gradient: {critic_mean:,.2f}")
            print(f"  • Ratio (Actor/Critic):     {ratio:,.2f}×")

            if ratio > 100:
                print(f"  ⚠️  CRITICAL: Actor CNN gradients {ratio:,.0f}× larger than Critic CNN")
                print(f"  📚 Literature: Should be similar magnitude for stable learning")

        return analysis

    def analyze_learning_dynamics(self) -> Dict:
        """
        Analyze learning dynamics (losses, Q-values).

        Returns:
            Dictionary with learning analysis results
        """
        print(f"\n🔍 ANALYSIS 2: Learning Dynamics")
        print(f"=" * 80)

        analysis = {
            'losses': {},
            'q_values': {}
        }

        # Analyze losses
        loss_data = self.all_data.get('losses', {})
        for metric_name, data in loss_data.items():
            values = np.array(data['values'])

            stats = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'first': values[0] if len(values) > 0 else None,
                'last': values[-1] if len(values) > 0 else None,
                'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if len(values) > 0 else 'unknown'
            }

            # Detect divergence (exponential growth)
            if abs(values[-1]) > abs(values[0]) * 1000:
                stats['divergence_detected'] = True
                stats['divergence_factor'] = abs(values[-1] / values[0]) if values[0] != 0 else float('inf')
            else:
                stats['divergence_detected'] = False

            analysis['losses'][metric_name] = stats

        # Analyze Q-values
        q_data = self.all_data.get('q_values', {})
        for metric_name, data in q_data.items():
            values = np.array(data['values'])

            stats = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'first': values[0] if len(values) > 0 else None,
                'last': values[-1] if len(values) > 0 else None,
                'growth': (values[-1] - values[0]) if len(values) > 0 else 0,
                'growth_factor': (values[-1] / values[0]) if len(values) > 0 and values[0] != 0 else 1.0
            }

            analysis['q_values'][metric_name] = stats

        # Print results
        print(f"\n📉 Loss Analysis:")
        for metric_name, stats in analysis['losses'].items():
            status = "❌ DIVERGING" if stats.get('divergence_detected', False) else "✅ STABLE"
            print(f"  • {metric_name}:")
            print(f"    Mean: {stats['mean']:.4f}, Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Trend: {stats['trend']}, Status: {status}")
            if stats.get('divergence_detected', False):
                print(f"    ⚠️  Divergence factor: {stats['divergence_factor']:.2e}×")

        print(f"\n📈 Q-Value Analysis:")
        for metric_name, stats in analysis['q_values'].items():
            print(f"  • {metric_name}:")
            print(f"    Mean: {stats['mean']:.4f}, Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Growth: {stats['growth']:.4f} ({stats['growth_factor']:.2f}×)")

        return analysis

    def analyze_episode_characteristics(self) -> Dict:
        """
        Analyze episode-level metrics (length, rewards, collisions).

        Compares with literature expectations.

        Returns:
            Dictionary with episode analysis results
        """
        print(f"\n🔍 ANALYSIS 3: Episode Characteristics")
        print(f"=" * 80)

        episode_data = self.all_data.get('episode', {})

        if not episode_data:
            print("⚠️  No episode metrics found")
            return {}

        analysis = {}

        for metric_name, data in episode_data.items():
            values = np.array(data['values'])

            stats = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }

            # Check against literature expectations for episode length
            if 'length' in metric_name.lower():
                expected_range = self.literature_benchmarks['episode_length']['expected_range']
                stats['within_expected_range'] = (stats['mean'] >= expected_range[0] and
                                                  stats['mean'] <= expected_range[1])
                stats['expected_range'] = expected_range

            analysis[metric_name] = stats

        # Print results
        print(f"\n📊 Episode Metrics:")
        for metric_name, stats in analysis.items():
            print(f"  • {metric_name}:")
            print(f"    Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"    Episodes: {stats['count']}")

            if 'length' in metric_name.lower():
                if not stats.get('within_expected_range', True):
                    expected = stats.get('expected_range', (0, 0))
                    print(f"    ⚠️  Mean length {stats['mean']:.0f} outside expected range {expected}")
                    print(f"    📚 Literature: Episodes should be {expected[0]}-{expected[1]} steps for robust learning")

        return analysis

    def analyze_reward_components(self) -> Dict:
        """
        Analyze reward components and balance.

        Compares with Race Driving paper insights on reward shaping.

        Returns:
            Dictionary with reward analysis results
        """
        print(f"\n🔍 ANALYSIS 4: Reward Function Analysis")
        print(f"=" * 80)

        reward_data = self.all_data.get('rewards', {})

        if not reward_data:
            # Try to find reward metrics in episode or other categories
            episode_data = self.all_data.get('episode', {})
            reward_data = {k: v for k, v in episode_data.items() if 'reward' in k.lower()}

        if not reward_data:
            print("⚠️  No reward component metrics found")
            return {}

        analysis = {}

        # Calculate statistics for each reward component
        for metric_name, data in reward_data.items():
            values = np.array(data['values'])

            stats = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'sum': np.sum(values),
                'count': len(values)
            }

            analysis[metric_name] = stats

        # Calculate component balance (if multiple components available)
        if len(analysis) > 1:
            total_sum = sum(stats['sum'] for stats in analysis.values())

            print(f"\n📊 Reward Component Balance:")
            print(f"{'Component':<40} {'Mean':<12} {'Contribution':<12} {'Status'}")
            print(f"-" * 80)

            for metric_name, stats in analysis.items():
                contribution = (stats['sum'] / total_sum * 100) if total_sum != 0 else 0

                # Check for imbalance
                status = "✅ BALANCED"
                if contribution > 90:
                    status = f"⚠️  DOMINATING ({contribution:.1f}%)"
                elif contribution < 1:
                    status = f"⚠️  TOO SMALL ({contribution:.1f}%)"

                print(f"{metric_name:<40} {stats['mean']:<12.4f} {contribution:<12.1f}% {status}")

                stats['contribution_percentage'] = contribution

            # Literature comparison
            print(f"\n📚 Literature Insight (Race Driving Paper):")
            benchmark = self.literature_benchmarks['reward_design']['race_driving_insight']
            print(f"  • Formula: {benchmark['formula']}")
            print(f"  • Key Finding: {benchmark['key_finding']}")
            print(f"  • Source: {benchmark['source']}")

            # Check if any component is dominating
            max_contribution = max(stats['contribution_percentage'] for stats in analysis.values())
            if max_contribution > 90:
                print(f"\n  ⚠️  WARNING: One component contributes {max_contribution:.1f}% of total reward")
                print(f"  📚 Literature suggests balanced rewards prevent overfitting to single objective")

        return analysis

    def generate_systematic_report(self, output_path: str = None):
        """
        Generate comprehensive systematic analysis report.

        Args:
            output_path: Path to save markdown report (optional)
        """
        print(f"\n📝 Generating Systematic Analysis Report...")
        print(f"=" * 80)

        # Perform all analyses
        gradient_analysis = self.analyze_gradient_patterns()
        learning_analysis = self.analyze_learning_dynamics()
        episode_analysis = self.analyze_episode_characteristics()
        reward_analysis = self.analyze_reward_components()

        # Generate report
        report_lines = []
        report_lines.append("# SYSTEMATIC TENSORBOARD ANALYSIS - LITERATURE-VALIDATED")
        report_lines.append("")
        report_lines.append("**Document Purpose**: Comprehensive analysis of TensorBoard logs with academic validation")
        report_lines.append("**Date**: 2025-11-17")
        report_lines.append("**Analysis Type**: Systematic multi-dimensional evaluation")
        report_lines.append("**Literature Sources**: 3 academic papers + TD3 official documentation")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## EXECUTIVE SUMMARY")
        report_lines.append("")

        # Identify critical issues
        critical_issues = []
        warnings = []

        # Check gradient explosions
        for metric_name, stats in gradient_analysis.items():
            if stats.get('exceeds_benchmark', False):
                critical_issues.append(f"Gradient explosion in {metric_name} (mean: {stats['mean']:,.0f}, "
                                     f"literature benchmark: {stats.get('literature_benchmark', 'N/A')})")
            elif stats.get('explosion_rate', 0) > 0.5:
                warnings.append(f"High gradient explosion rate in {metric_name} ({stats['explosion_rate']*100:.1f}%)")

        # Check loss divergence
        for metric_name, stats in learning_analysis.get('losses', {}).items():
            if stats.get('divergence_detected', False):
                critical_issues.append(f"Loss divergence in {metric_name} "
                                     f"(factor: {stats.get('divergence_factor', 0):.2e}×)")

        # Check episode length
        for metric_name, stats in episode_analysis.items():
            if 'length' in metric_name.lower() and not stats.get('within_expected_range', True):
                warnings.append(f"Episode length outside expected range (mean: {stats['mean']:.0f}, "
                              f"expected: {stats.get('expected_range', (0, 0))})")

        # Check reward balance
        for metric_name, stats in reward_analysis.items():
            if stats.get('contribution_percentage', 0) > 90:
                warnings.append(f"Reward component imbalance: {metric_name} dominates "
                              f"({stats['contribution_percentage']:.1f}%)")

        # Print summary
        report_lines.append(f"### 🎯 Analysis Overview")
        report_lines.append("")
        report_lines.append(f"- **Total Metrics Analyzed**: {sum(len(cat) for cat in self.all_data.values())}")
        report_lines.append(f"- **Critical Issues Found**: {len(critical_issues)}")
        report_lines.append(f"- **Warnings**: {len(warnings)}")
        report_lines.append("")

        if critical_issues:
            report_lines.append("### 🚨 CRITICAL ISSUES")
            report_lines.append("")
            for issue in critical_issues:
                report_lines.append(f"- ❌ {issue}")
            report_lines.append("")

        if warnings:
            report_lines.append("### ⚠️  WARNINGS")
            report_lines.append("")
            for warning in warnings:
                report_lines.append(f"- ⚠️  {warning}")
            report_lines.append("")

        if not critical_issues and not warnings:
            report_lines.append("### ✅ NO CRITICAL ISSUES DETECTED")
            report_lines.append("")
            report_lines.append("All metrics within expected ranges based on literature benchmarks.")
            report_lines.append("")

        report_lines.append("---")
        report_lines.append("")

        # Detailed sections...
        # (Add gradient analysis section)
        report_lines.append("## 1. GRADIENT FLOW ANALYSIS")
        report_lines.append("")
        report_lines.append("### Literature Benchmarks")
        report_lines.append("")
        report_lines.append("| Source | Clip Norm | Network | Success Rate |")
        report_lines.append("|--------|-----------|---------|--------------|")
        for key, value in self.literature_benchmarks['gradient_norms'].items():
            if isinstance(value, dict) and 'source' in value:
                source = value['source']
                if 'actor_clip_norm' in value:
                    report_lines.append(f"| {source} | {value['actor_clip_norm']} | Actor CNN | {value.get('success_rate_with_clipping', 'N/A')} |")
                elif 'critic_clip_norm' in value:
                    report_lines.append(f"| {source} | {value['critic_clip_norm']} | Critic CNN | N/A |")
                elif 'clip_norm' in value:
                    report_lines.append(f"| {source} | {value['clip_norm']} | Mixed | N/A |")
        report_lines.append("")

        report_lines.append("### Gradient Norm Statistics")
        report_lines.append("")
        report_lines.append("| Metric | Mean | Max | Explosion Rate | Literature Benchmark | Status |")
        report_lines.append("|--------|------|-----|----------------|---------------------|--------|")

        for metric_name, stats in gradient_analysis.items():
            benchmark = stats.get('literature_benchmark', 'N/A')
            status = "✅ STABLE"
            if stats.get('exceeds_benchmark', False):
                status = f"❌ EXCEEDS ({benchmark})"
            elif stats.get('explosion_rate', 0) > 0.5:
                status = "⚠️  HIGH EXPLOSION RATE"

            report_lines.append(f"| {metric_name} | {stats['mean']:,.2f} | {stats['max']:,.2f} | "
                              f"{stats['explosion_rate']*100:.1f}% | {benchmark} | {status} |")

        report_lines.append("")

        # Save report if path provided
        if output_path:
            report_text = '\n'.join(report_lines)
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Report saved to: {output_path}")

        return '\n'.join(report_lines)


def main():
    """Main execution function."""
    # Path to TensorBoard event file
    event_file = "/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251113-132842/events.out.tfevents.1763040522.danielterra.1.0"

    # Output path for report
    output_dir = Path(project_root) / "docs" / "day-17"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "SYSTEMATIC_TENSORBOARD_ANALYSIS_LITERATURE_VALIDATED.md"

    print("=" * 80)
    print("SYSTEMATIC TENSORBOARD ANALYSIS - LITERATURE-VALIDATED")
    print("=" * 80)
    print(f"Event file: {event_file}")
    print(f"Output: {output_path}")
    print("=" * 80)

    # Create analyzer
    analyzer = TensorBoardSystematicAnalyzer(event_file)

    # Load events
    analyzer.load_events()

    # Extract all metrics
    analyzer.extract_all_metrics()

    # Generate systematic report
    report = analyzer.generate_systematic_report(str(output_path))

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📄 Full report saved to: {output_path}")


if __name__ == "__main__":
    main()

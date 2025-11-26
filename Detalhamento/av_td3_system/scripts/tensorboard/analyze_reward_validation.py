#!/usr/bin/env python3
"""
Reward Validation Analysis Script

Analyzes reward validation logs from manual control sessions to:
1. Identify anomalies in reward calculations
2. Validate reward components correlate correctly with metrics
3. Generate visualization plots
4. Produce validation report for paper

Usage:
    python scripts/analyze_reward_validation.py --log validation_logs/reward_validation_20240115_143022.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class ValidationIssue:
    """Represents a detected issue in reward validation."""
    severity: str  # 'critical', 'warning', 'info'
    component: str
    description: str
    affected_steps: List[int]
    recommendation: str


class RewardAnalyzer:
    """Analyzes reward validation data."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.data = self._load_data()
        self.df = self._create_dataframe()
        self.issues: List[ValidationIssue] = []

    def _load_data(self) -> Dict:
        """Load validation log JSON."""
        with open(self.log_path, 'r') as f:
            return json.load(f)

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to pandas DataFrame."""
        snapshots = self.data.get('snapshots', [])
        if not snapshots:
            raise ValueError("No snapshots found in validation log")

        df = pd.DataFrame(snapshots)

        # Calculate derived metrics
        df['speed_kmh'] = df['velocity'] * 3.6
        df['abs_lateral_deviation'] = df['lateral_deviation'].abs()
        df['heading_error_deg'] = np.rad2deg(df['heading_error'].abs())

        return df

    def validate_lane_keeping_correlation(self) -> ValidationIssue:
        """
        Validate that lane keeping reward correlates negatively with lateral deviation.

        Expected: Higher lateral deviation should give more negative reward.
        """
        # Calculate correlation
        corr, p_value = stats.pearsonr(
            self.df['abs_lateral_deviation'],
            self.df['lane_keeping_reward']
        )

        severity = 'info'
        description = f"Lateral deviation vs lane keeping reward correlation: {corr:.3f} (p={p_value:.4f})"
        recommendation = "No issues detected"
        affected_steps = []

        # Expected: strong negative correlation
        if corr > -0.5:
            severity = 'critical'
            recommendation = "Lane keeping reward should correlate more strongly (negatively) with lateral deviation"
            affected_steps = self.df.index.tolist()
        elif corr > -0.7:
            severity = 'warning'
            recommendation = "Consider strengthening lane keeping penalty weight"

        issue = ValidationIssue(
            severity=severity,
            component='lane_keeping',
            description=description,
            affected_steps=affected_steps,
            recommendation=recommendation
        )

        self.issues.append(issue)
        return issue

    def validate_efficiency_reward(self) -> ValidationIssue:
        """
        Validate efficiency reward behavior with respect to target speed.

        Expected: Reward should be highest near target speed, decrease away from it.
        """
        # Assuming target speed is around 30 km/h (adjust based on your config)
        target_speed = 30.0

        df_subset = self.df[self.df['speed_kmh'] > 0].copy()
        df_subset['speed_error'] = (df_subset['speed_kmh'] - target_speed).abs()

        # Efficiency reward should decrease as speed error increases
        corr, p_value = stats.pearsonr(df_subset['speed_error'], df_subset['efficiency_reward'])

        severity = 'info'
        description = f"Speed error vs efficiency reward correlation: {corr:.3f} (p={p_value:.4f})"
        recommendation = "No issues detected"
        affected_steps = []

        # Expected: negative correlation (higher error = lower reward)
        if corr > 0.2:
            severity = 'critical'
            recommendation = "Efficiency reward increasing with speed error - check reward function logic"
            affected_steps = df_subset.index.tolist()
        elif corr > -0.3:
            severity = 'warning'
            recommendation = "Weak correlation between speed error and efficiency reward"

        issue = ValidationIssue(
            severity=severity,
            component='efficiency',
            description=description,
            affected_steps=affected_steps,
            recommendation=recommendation
        )

        self.issues.append(issue)
        return issue

    def validate_comfort_penalty(self) -> ValidationIssue:
        """
        Validate comfort penalty is activated appropriately.

        Expected: Comfort penalty should be triggered during abrupt maneuvers.
        """
        # Calculate acceleration changes (jerk proxy)
        df_subset = self.df.copy()
        df_subset['velocity_change'] = df_subset['velocity'].diff().abs()

        # High velocity changes should correlate with comfort penalties
        high_jerk_steps = df_subset[df_subset['velocity_change'] > 0.5]

        if len(high_jerk_steps) > 0:
            avg_comfort_penalty = high_jerk_steps['comfort_penalty'].mean()

            severity = 'info'
            description = f"Average comfort penalty during high-jerk maneuvers: {avg_comfort_penalty:.4f}"
            recommendation = "No issues detected"
            affected_steps = []

            # Expect negative penalties during high jerk
            if avg_comfort_penalty > -0.01:
                severity = 'warning'
                recommendation = "Comfort penalty not activating during abrupt maneuvers"
                affected_steps = high_jerk_steps.index.tolist()
        else:
            severity = 'info'
            description = "No high-jerk maneuvers detected in session"
            recommendation = "Consider testing emergency brake scenarios"
            affected_steps = []

        issue = ValidationIssue(
            severity=severity,
            component='comfort',
            description=description,
            affected_steps=affected_steps,
            recommendation=recommendation
        )

        self.issues.append(issue)
        return issue

    def validate_safety_penalty(self) -> ValidationIssue:
        """
        Validate safety penalty triggers correctly.

        Expected: Large negative rewards for dangerous situations.
        """
        # Find steps with safety penalties
        safety_penalty_steps = self.df[self.df['safety_penalty'] < -0.01]

        severity = 'info'
        if len(safety_penalty_steps) > 0:
            avg_safety_penalty = safety_penalty_steps['safety_penalty'].mean()
            min_safety_penalty = safety_penalty_steps['safety_penalty'].min()

            description = (
                f"Safety penalties triggered in {len(safety_penalty_steps)} steps. "
                f"Avg: {avg_safety_penalty:.4f}, Min: {min_safety_penalty:.4f}"
            )

            # Safety penalties should be substantial
            if min_safety_penalty > -1.0:
                severity = 'warning'
                recommendation = "Safety penalty may be too weak - consider increasing magnitude"
            else:
                recommendation = "Safety penalties appear appropriately strong"

            affected_steps = safety_penalty_steps.index.tolist()
        else:
            description = "No safety penalties triggered in session"
            recommendation = "Test collision and off-road scenarios to validate"
            affected_steps = []

        issue = ValidationIssue(
            severity=severity,
            component='safety',
            description=description,
            affected_steps=affected_steps,
            recommendation=recommendation
        )

        self.issues.append(issue)
        return issue

    def validate_reward_components_sum(self) -> ValidationIssue:
        """
        Validate that reward components sum to total reward.

        Expected: total_reward ≈ sum of all components
        """
        df_subset = self.df.copy()
        df_subset['calculated_total'] = (
            df_subset['efficiency_reward'] +
            df_subset['lane_keeping_reward'] +
            df_subset['comfort_penalty'] +
            df_subset['safety_penalty'] +
            df_subset['progress_reward']
        )

        df_subset['reward_residual'] = (
            df_subset['total_reward'] - df_subset['calculated_total']
        ).abs()

        max_residual = df_subset['reward_residual'].max()
        mean_residual = df_subset['reward_residual'].mean()

        severity = 'info'
        description = f"Reward component sum residual - Mean: {mean_residual:.6f}, Max: {max_residual:.6f}"
        recommendation = "No issues detected"
        affected_steps = []

        # Residual should be near zero (within floating point tolerance)
        if max_residual > 0.001:
            severity = 'critical'
            recommendation = "Reward components do not sum to total - check reward function implementation"
            affected_steps = df_subset[df_subset['reward_residual'] > 0.001].index.tolist()
        elif mean_residual > 0.0001:
            severity = 'warning'
            recommendation = "Small discrepancies in reward component summation"

        issue = ValidationIssue(
            severity=severity,
            component='total_reward',
            description=description,
            affected_steps=affected_steps,
            recommendation=recommendation
        )

        self.issues.append(issue)
        return issue

    def detect_reward_anomalies(self) -> List[ValidationIssue]:
        """Detect statistical anomalies in rewards."""
        anomaly_issues = []

        # Check for extreme outliers (beyond 3 sigma)
        for component in ['total_reward', 'efficiency_reward', 'lane_keeping_reward',
                         'comfort_penalty', 'safety_penalty', 'progress_reward']:

            mean = self.df[component].mean()
            std = self.df[component].std()

            outliers = self.df[
                (self.df[component] < mean - 3*std) |
                (self.df[component] > mean + 3*std)
            ]

            if len(outliers) > 0:
                issue = ValidationIssue(
                    severity='warning',
                    component=component,
                    description=f"Found {len(outliers)} outliers beyond 3σ (mean={mean:.4f}, std={std:.4f})",
                    affected_steps=outliers.index.tolist(),
                    recommendation="Investigate these steps for unusual conditions"
                )
                anomaly_issues.append(issue)
                self.issues.append(issue)

        return anomaly_issues

    def generate_plots(self, output_dir: Path):
        """Generate visualization plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Reward components over time
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Reward Components Over Time", fontsize=16)

        components = [
            ('total_reward', 'Total Reward'),
            ('efficiency_reward', 'Efficiency Reward'),
            ('lane_keeping_reward', 'Lane Keeping Reward'),
            ('comfort_penalty', 'Comfort Penalty'),
            ('safety_penalty', 'Safety Penalty'),
            ('progress_reward', 'Progress Reward')
        ]

        for idx, (comp, title) in enumerate(components):
            ax = axes[idx // 2, idx % 2]
            ax.plot(self.df['step'], self.df[comp], linewidth=0.8)
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'reward_components_timeline.png', dpi=150)
        plt.close()

        # 2. Lateral deviation vs lane keeping reward
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['abs_lateral_deviation'], self.df['lane_keeping_reward'],
                   alpha=0.5, s=10)
        plt.xlabel('Absolute Lateral Deviation (m)')
        plt.ylabel('Lane Keeping Reward')
        plt.title('Lateral Deviation vs Lane Keeping Reward')

        # Add trend line
        z = np.polyfit(self.df['abs_lateral_deviation'], self.df['lane_keeping_reward'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df['abs_lateral_deviation'].min(),
                             self.df['abs_lateral_deviation'].max(), 100)
        plt.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'lateral_deviation_correlation.png', dpi=150)
        plt.close()

        # 3. Speed vs efficiency reward
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['speed_kmh'], self.df['efficiency_reward'],
                   alpha=0.5, s=10, c=self.df['step'], cmap='viridis')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Efficiency Reward')
        plt.title('Speed vs Efficiency Reward')
        plt.colorbar(label='Step')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'speed_efficiency_correlation.png', dpi=150)
        plt.close()

        # 4. Reward distribution by scenario type
        if 'scenario_type' in self.df.columns:
            scenario_counts = self.df['scenario_type'].value_counts()
            if len(scenario_counts) > 1:
                plt.figure(figsize=(12, 6))

                scenarios = self.df['scenario_type'].unique()
                for scenario in scenarios:
                    subset = self.df[self.df['scenario_type'] == scenario]
                    plt.hist(subset['total_reward'], bins=30, alpha=0.5, label=scenario)

                plt.xlabel('Total Reward')
                plt.ylabel('Frequency')
                plt.title('Reward Distribution by Scenario Type')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'reward_distribution_by_scenario.png', dpi=150)
                plt.close()

        # 5. Correlation heatmap
        plt.figure(figsize=(10, 8))

        correlation_vars = [
            'velocity', 'abs_lateral_deviation', 'heading_error',
            'total_reward', 'efficiency_reward', 'lane_keeping_reward',
            'comfort_penalty', 'safety_penalty', 'progress_reward'
        ]

        corr_matrix = self.df[correlation_vars].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix: Metrics vs Reward Components')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
        plt.close()

        print(f"\n[PLOTS] Generated visualization plots in {output_dir}")

    def generate_report(self, output_path: Path):
        """Generate markdown validation report."""
        report_lines = [
            "# Reward Function Validation Report",
            "",
            f"**Validation Log**: `{self.log_path.name}`",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Session Summary",
            ""
        ]

        # Session info
        session_info = self.data.get('session_info', {})
        report_lines.extend([
            f"- **Total Steps**: {session_info.get('total_steps', 'N/A')}",
            f"- **Duration**: {session_info.get('duration_seconds', 0):.2f} seconds",
            f"- **Start Time**: {session_info.get('start_time', 'N/A')}",
            ""
        ])

        # Descriptive statistics
        report_lines.extend([
            "## Reward Statistics",
            "",
            "### Total Reward",
            ""
        ])

        total_reward_stats = self.df['total_reward'].describe()
        for stat, value in total_reward_stats.items():
            report_lines.append(f"- **{stat}**: {value:.4f}")

        report_lines.extend(["", "### Component Breakdown", ""])

        components = ['efficiency_reward', 'lane_keeping_reward', 'comfort_penalty',
                     'safety_penalty', 'progress_reward']

        for comp in components:
            stats_data = self.df[comp].describe()
            report_lines.append(f"#### {comp.replace('_', ' ').title()}")
            report_lines.append("")
            report_lines.append(f"- Mean: {stats_data['mean']:.4f}")
            report_lines.append(f"- Std: {stats_data['std']:.4f}")
            report_lines.append(f"- Min: {stats_data['min']:.4f}")
            report_lines.append(f"- Max: {stats_data['max']:.4f}")
            report_lines.append("")

        # Validation issues
        report_lines.extend([
            "---",
            "",
            "## Validation Results",
            ""
        ])

        # Group issues by severity
        critical = [i for i in self.issues if i.severity == 'critical']
        warnings = [i for i in self.issues if i.severity == 'warning']
        info = [i for i in self.issues if i.severity == 'info']

        report_lines.append(f"**Summary**: {len(critical)} critical, {len(warnings)} warnings, {len(info)} info")
        report_lines.append("")

        if critical:
            report_lines.extend(["### 🔴 Critical Issues", ""])
            for issue in critical:
                report_lines.extend([
                    f"#### {issue.component}",
                    "",
                    f"**Description**: {issue.description}",
                    "",
                    f"**Recommendation**: {issue.recommendation}",
                    "",
                    f"**Affected Steps**: {len(issue.affected_steps)} steps",
                    ""
                ])

        if warnings:
            report_lines.extend(["### ⚠️ Warnings", ""])
            for issue in warnings:
                report_lines.extend([
                    f"#### {issue.component}",
                    "",
                    f"**Description**: {issue.description}",
                    "",
                    f"**Recommendation**: {issue.recommendation}",
                    ""
                ])

        if info:
            report_lines.extend(["### ℹ️ Information", ""])
            for issue in info:
                report_lines.extend([
                    f"- **{issue.component}**: {issue.description}",
                    ""
                ])

        # Scenario analysis
        if 'scenario_type' in self.df.columns:
            scenario_stats = self.df.groupby('scenario_type')['total_reward'].agg(['count', 'mean', 'std'])

            report_lines.extend([
                "---",
                "",
                "## Scenario Analysis",
                "",
                "| Scenario | Steps | Mean Reward | Std Reward |",
                "|----------|-------|-------------|------------|"
            ])

            for scenario, row in scenario_stats.iterrows():
                report_lines.append(
                    f"| {scenario} | {row['count']:.0f} | {row['mean']:.4f} | {row['std']:.4f} |"
                )

            report_lines.append("")

        # Recommendations
        report_lines.extend([
            "---",
            "",
            "## Recommendations for Paper",
            "",
            "1. **Reward Function Documentation**:"
        ])

        if critical:
            report_lines.append("   - ⚠️ Address critical issues before reporting reward function in paper")
        else:
            report_lines.append("   - ✅ Reward function appears correctly implemented")

        report_lines.extend([
            "",
            "2. **Edge Cases**:",
            "   - Test additional scenarios (intersections, lane changes, emergency stops)",
            "   - Validate behavior matches design specifications",
            "",
            "3. **Hyperparameter Tuning**:",
            "   - Consider adjusting component weights based on observed correlations",
            "   - Ensure safety penalties are sufficient to prevent dangerous behavior",
            "",
            "4. **Reproducibility**:",
            "   - Document exact reward weights used in training",
            "   - Include validation plots in supplementary materials",
            ""
        ])

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\n[REPORT] Generated validation report: {output_path}")

    def run_full_analysis(self, output_dir: Path):
        """Run complete validation analysis."""
        print("\n" + "="*70)
        print("REWARD VALIDATION ANALYSIS")
        print("="*70)
        print(f"\nAnalyzing: {self.log_path}")
        print(f"Total snapshots: {len(self.df)}")
        print("")

        # Run validations
        print("[1/7] Validating lane keeping correlation...")
        self.validate_lane_keeping_correlation()

        print("[2/7] Validating efficiency reward...")
        self.validate_efficiency_reward()

        print("[3/7] Validating comfort penalty...")
        self.validate_comfort_penalty()

        print("[4/7] Validating safety penalty...")
        self.validate_safety_penalty()

        print("[5/7] Validating reward component summation...")
        self.validate_reward_components_sum()

        print("[6/7] Detecting anomalies...")
        self.detect_reward_anomalies()

        # Generate outputs
        print("[7/7] Generating plots and report...")
        self.generate_plots(output_dir)

        report_path = output_dir / f"validation_report_{self.log_path.stem}.md"
        self.generate_report(report_path)

        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        critical = [i for i in self.issues if i.severity == 'critical']
        warnings = [i for i in self.issues if i.severity == 'warning']

        if critical:
            print(f"\n🔴 {len(critical)} CRITICAL ISSUES FOUND:")
            for issue in critical:
                print(f"   - {issue.component}: {issue.description}")
                print(f"     → {issue.recommendation}")

        if warnings:
            print(f"\n⚠️ {len(warnings)} WARNINGS:")
            for issue in warnings:
                print(f"   - {issue.component}: {issue.description}")

        if not critical and not warnings:
            print("\n✅ No critical issues or warnings detected!")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze reward validation logs")
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="Path to reward validation JSON log file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots and reports (default: same as log directory)"
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_path.parent / f"analysis_{log_path.stem}"

    # Run analysis
    try:
        analyzer = RewardAnalyzer(log_path)
        analyzer.run_full_analysis(output_dir)

        print(f"\n✅ Analysis complete! Results saved to: {output_dir}\n")
        return 0

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

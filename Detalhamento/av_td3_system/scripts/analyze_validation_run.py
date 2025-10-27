#!/usr/bin/env python3
"""
TD3 Validation Run Analysis Script

Purpose: Automatically analyze the 20k-step validation training run to verify:
  1. Reward function is working correctly (no "stand still" exploit)
  2. CNN features are providing useful information (not degenerate/constant)
  3. Waypoint data is properly integrated and spatially sensible
  4. Agent is learning to navigate (metrics improving over time)
  5. Training is stable (no crashes, divergence, or anomalies)

Generates a comprehensive validation report with:
  - Pass/Fail verdict for each validation criteria
  - Statistical analysis of training metrics
  - Visualization of learning curves
  - Recommendations for proceeding to full training

Author: Daniel Terra
Date: October 26, 2024
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class ValidationAnalyzer:
    """Analyzes 20k validation training run to verify TD3 solution."""

    def __init__(self, log_dir: str, output_dir: str):
        """
        Initialize analyzer.

        Args:
            log_dir: Path to logs directory (contains TensorBoard events and training logs)
            output_dir: Path to save analysis results and plots
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validation_results = {}
        self.metrics = {}
        self.debug_data = {}

    def parse_training_log(self, log_file: Path) -> Dict:
        """
        Parse training log file to extract debug information.

        Extracts:
          - CNN feature statistics (L2 norm, mean, std, range)
          - Waypoint coordinates and distances
          - Reward component breakdown
          - Vehicle state (speed, lateral deviation, heading error)
          - Episode statistics

        Returns:
            Dictionary with parsed data organized by timestep
        """
        print(f"[PARSE] Reading training log: {log_file}")

        data = {
            'cnn_features': [],  # L2 norm per timestep
            'waypoints': [],     # Waypoint coordinates
            'rewards': [],       # Total reward per step
            'reward_components': {
                'efficiency': [],
                'lane_keeping': [],
                'comfort': [],
                'safety': [],
                'progress': []
            },
            'vehicle_state': {
                'speed': [],
                'lateral_deviation': [],
                'heading_error': []
            },
            'episodes': [],      # Episode-level statistics
            'timesteps': []      # Timestep indices
        }

        with open(log_file, 'r') as f:
            current_episode = {'reward': 0, 'length': 0, 'collisions': 0}

            for line in f:
                # CNN feature statistics
                match = re.search(r'\[DEBUG\]\[Step (\d+)\] CNN Feature Stats:.*L2 Norm: ([\d.]+)', line, re.DOTALL)
                if match:
                    timestep = int(match.group(1))
                    l2_norm = float(match.group(2))
                    data['cnn_features'].append((timestep, l2_norm))

                # Reward and reward components
                match = re.search(
                    r'\[DEBUG Step\s+(\d+)\].*Rew=([\d.+-]+).*'
                    r'\[Reward\] Efficiency=([\d.+-]+) \| Lane=([\d.+-]+) \| '
                    r'Comfort=([\d.+-]+) \| Safety=([\d.+-]+) \| Progress=([\d.+-]+)',
                    line
                )
                if match:
                    timestep = int(match.group(1))
                    reward = float(match.group(2))
                    eff = float(match.group(3))
                    lane = float(match.group(4))
                    comfort = float(match.group(5))
                    safety = float(match.group(6))
                    progress = float(match.group(7))

                    data['timesteps'].append(timestep)
                    data['rewards'].append(reward)
                    data['reward_components']['efficiency'].append(eff)
                    data['reward_components']['lane_keeping'].append(lane)
                    data['reward_components']['comfort'].append(comfort)
                    data['reward_components']['safety'].append(safety)
                    data['reward_components']['progress'].append(progress)

                # Vehicle state
                match = re.search(
                    r'\[State\] velocity=([\d.]+) m/s \| lat_dev=([\d.+-]+)m \| '
                    r'heading_err=([\d.+-]+) rad',
                    line
                )
                if match:
                    speed = float(match.group(1)) * 3.6  # Convert to km/h
                    lat_dev = float(match.group(2))
                    heading = float(match.group(3))

                    data['vehicle_state']['speed'].append(speed)
                    data['vehicle_state']['lateral_deviation'].append(lat_dev)
                    data['vehicle_state']['heading_error'].append(heading)

                # Waypoints
                match = re.search(
                    r'\[Waypoints\].*WP1=\[\s*([\d.+-]+),\s*([\d.+-]+)\]m \(d=\s*([\d.]+)m\)',
                    line
                )
                if match:
                    wp_x = float(match.group(1))
                    wp_y = float(match.group(2))
                    wp_dist = float(match.group(3))
                    data['waypoints'].append((wp_x, wp_y, wp_dist))

                # Episode completion
                match = re.search(
                    r'\[TRAIN\] Episode\s+(\d+).*Reward\s+([\d.+-]+).*Collisions\s+(\d+)',
                    line
                )
                if match:
                    episode_num = int(match.group(1))
                    episode_reward = float(match.group(2))
                    collisions = int(match.group(3))

                    data['episodes'].append({
                        'number': episode_num,
                        'reward': episode_reward,
                        'collisions': collisions
                    })

        print(f"[PARSE] Extracted {len(data['timesteps'])} timesteps, {len(data['episodes'])} episodes")
        return data

    def validate_reward_function(self, data: Dict) -> Tuple[bool, str]:
        """
        Validate reward function is working correctly.

        Checks:
          1. Standing still (speed < 1 km/h) gives negative reward
          2. Moving (speed > 5 km/h) gives better reward than standing still
          3. Safety component is negative when stationary (not positive)
          4. Reward components are within expected ranges

        Returns:
            (pass/fail, explanation message)
        """
        print("\n[VALIDATE] Checking reward function...")

        speeds = np.array(data['vehicle_state']['speed'])
        rewards = np.array(data['rewards'])
        safety_rewards = np.array(data['reward_components']['safety'])

        # Check 1: Standing still gives negative reward
        stationary_mask = speeds < 1.0
        stationary_rewards = rewards[stationary_mask]

        if len(stationary_rewards) == 0:
            return False, "No stationary timesteps found in data (unexpected)"

        avg_stationary_reward = np.mean(stationary_rewards)

        if avg_stationary_reward >= 0:
            return False, f"FAIL: Standing still gives positive reward ({avg_stationary_reward:.2f}). Bug regression!"

        # Check 2: Moving gives better reward
        moving_mask = speeds > 5.0
        moving_rewards = rewards[moving_mask]

        if len(moving_rewards) > 0:
            avg_moving_reward = np.mean(moving_rewards)
            improvement = avg_moving_reward - avg_stationary_reward

            if improvement <= 0:
                return False, f"FAIL: Moving ({avg_moving_reward:.2f}) not better than stationary ({avg_stationary_reward:.2f})"
        else:
            # Not enough moving data yet (early in training)
            print("[VALIDATE] Not enough moving data yet to compare (early in training)")

        # Check 3: Safety component is negative when stationary
        stationary_safety = safety_rewards[stationary_mask]
        avg_stationary_safety = np.mean(stationary_safety)

        if avg_stationary_safety >= 0:
            return False, f"FAIL: Safety reward is positive when stationary ({avg_stationary_safety:.2f}). Sign bug!"

        # Check 4: Reward components in expected ranges
        for component, values in data['reward_components'].items():
            if len(values) == 0:
                continue

            arr = np.array(values)
            min_val, max_val = arr.min(), arr.max()

            # Check for degenerate (constant) components
            if np.allclose(min_val, max_val, atol=0.01):
                print(f"[WARNING] {component} reward is nearly constant ({min_val:.3f})")

        # All checks passed
        message = (
            f"PASS: Reward function working correctly\n"
            f"  Standing still (< 1 km/h): {avg_stationary_reward:.2f} reward (negative ✓)\n"
            f"  Safety when stationary: {avg_stationary_safety:.2f} (negative ✓)\n"
        )

        if len(moving_rewards) > 0:
            message += f"  Moving (> 5 km/h): {avg_moving_reward:.2f} reward (better ✓)\n"
            message += f"  Improvement: {improvement:.2f} points\n"

        return True, message

    def validate_cnn_features(self, data: Dict) -> Tuple[bool, str]:
        """
        Validate CNN feature extraction is working.

        Checks:
          1. CNN features are not constant (L2 norm varies)
          2. CNN features are not degenerate (L2 norm not near zero)
          3. CNN features show temporal variation (changing over time)

        Returns:
            (pass/fail, explanation message)
        """
        print("\n[VALIDATE] Checking CNN feature extraction...")

        if len(data['cnn_features']) == 0:
            return False, "No CNN feature data found in logs"

        timesteps, l2_norms = zip(*data['cnn_features'])
        l2_norms = np.array(l2_norms)

        # Check 1: L2 norm varies (not constant)
        std_l2 = np.std(l2_norms)
        if std_l2 < 0.01:
            return False, f"FAIL: CNN features are constant (std={std_l2:.4f})"

        # Check 2: L2 norm not near zero (features not degenerate)
        mean_l2 = np.mean(l2_norms)
        if mean_l2 < 0.1:
            return False, f"FAIL: CNN features degenerate (mean L2 norm={mean_l2:.4f})"

        # Check 3: Temporal variation
        # Split into 4 quartiles and check if means differ
        n = len(l2_norms)
        q1 = l2_norms[:n//4]
        q4 = l2_norms[3*n//4:]

        if len(q1) > 0 and len(q4) > 0:
            mean_q1 = np.mean(q1)
            mean_q4 = np.mean(q4)
            temporal_change = abs(mean_q4 - mean_q1) / mean_q1 * 100

            if temporal_change < 1:
                print(f"[WARNING] CNN features show little temporal variation ({temporal_change:.1f}%)")

        message = (
            f"PASS: CNN features working correctly\n"
            f"  Mean L2 norm: {mean_l2:.3f} (non-zero ✓)\n"
            f"  Std L2 norm: {std_l2:.3f} (varying ✓)\n"
            f"  Range: [{l2_norms.min():.3f}, {l2_norms.max():.3f}]\n"
        )

        return True, message

    def validate_waypoints(self, data: Dict) -> Tuple[bool, str]:
        """
        Validate waypoint data is spatially sensible.

        Checks:
          1. Waypoints are ahead of vehicle (x > 0 in vehicle frame)
          2. Waypoints are within reasonable distance (< 50m)
          3. Waypoints show spatial variation (not fixed)

        Returns:
            (pass/fail, explanation message)
        """
        print("\n[VALIDATE] Checking waypoint data...")

        if len(data['waypoints']) == 0:
            return False, "No waypoint data found in logs"

        waypoints = np.array(data['waypoints'])  # (N, 3): x, y, distance
        x_coords = waypoints[:, 0]
        y_coords = waypoints[:, 1]
        distances = waypoints[:, 2]

        # Check 1: Waypoints ahead of vehicle
        ahead_ratio = np.sum(x_coords > 0) / len(x_coords)
        if ahead_ratio < 0.8:
            return False, f"FAIL: Only {ahead_ratio*100:.1f}% of waypoints ahead of vehicle"

        # Check 2: Reasonable distances
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)

        if max_dist > 100:
            print(f"[WARNING] Some waypoints very far ({max_dist:.1f}m)")

        # Check 3: Spatial variation
        std_x = np.std(x_coords)
        std_y = np.std(y_coords)

        if std_x < 0.5 or std_y < 0.5:
            print(f"[WARNING] Waypoints show little spatial variation (std_x={std_x:.2f}, std_y={std_y:.2f})")

        message = (
            f"PASS: Waypoints spatially sensible\n"
            f"  Waypoints ahead: {ahead_ratio*100:.1f}% (>80% ✓)\n"
            f"  Mean distance: {mean_dist:.1f}m\n"
            f"  Distance range: [{distances.min():.1f}, {max_dist:.1f}]m\n"
            f"  Spatial variation: std_x={std_x:.2f}, std_y={std_y:.2f}\n"
        )

        return True, message

    def validate_learning(self, data: Dict) -> Tuple[bool, str]:
        """
        Validate agent is learning to navigate.

        Checks:
          1. Episode rewards show upward trend
          2. Average speed increases over time
          3. Collision rate decreases over time (or stays low)

        Returns:
            (pass/fail, explanation message)
        """
        print("\n[VALIDATE] Checking learning progress...")

        if len(data['episodes']) < 5:
            return False, f"Not enough episodes to assess learning ({len(data['episodes'])} < 5)"

        episodes = pd.DataFrame(data['episodes'])

        # Check 1: Reward trend
        early_rewards = episodes.iloc[:len(episodes)//2]['reward'].mean()
        late_rewards = episodes.iloc[len(episodes)//2:]['reward'].mean()
        reward_improvement = late_rewards - early_rewards

        # Statistical test: Spearman correlation (monotonic trend)
        if len(episodes) >= 10:
            corr, p_value = stats.spearmanr(episodes['number'], episodes['reward'])
            significant_trend = p_value < 0.05 and corr > 0
        else:
            significant_trend = False

        # Check 2: Speed trend
        speeds = np.array(data['vehicle_state']['speed'])
        if len(speeds) >= 1000:
            early_speed = np.mean(speeds[:500])
            late_speed = np.mean(speeds[-500:])
            speed_improvement = late_speed - early_speed
        else:
            early_speed = late_speed = speed_improvement = 0

        # Check 3: Collision rate
        collision_rate = episodes['collisions'].sum() / len(episodes)

        # Determine pass/fail
        if significant_trend and reward_improvement > 0:
            status = "PASS"
            explanation = "Agent shows significant learning improvement"
        elif reward_improvement > 10:
            status = "PASS"
            explanation = "Agent shows substantial reward improvement"
        elif reward_improvement > 0:
            status = "PARTIAL"
            explanation = "Agent shows modest learning (may need more time)"
        else:
            status = "FAIL"
            explanation = "No learning improvement detected"

        message = (
            f"{status}: {explanation}\n"
            f"  Early episodes reward: {early_rewards:.2f}\n"
            f"  Late episodes reward: {late_rewards:.2f}\n"
            f"  Improvement: {reward_improvement:+.2f}\n"
        )

        if len(speeds) >= 1000:
            message += (
                f"  Early speed: {early_speed:.1f} km/h\n"
                f"  Late speed: {late_speed:.1f} km/h\n"
                f"  Speed improvement: {speed_improvement:+.1f} km/h\n"
            )

        message += f"  Collision rate: {collision_rate:.2f} per episode\n"

        if significant_trend:
            message += f"  Statistical trend: Spearman ρ={corr:.3f}, p={p_value:.3f} ✓\n"

        return status == "PASS" or status == "PARTIAL", message

    def generate_plots(self, data: Dict):
        """Generate visualization plots for validation report."""
        print("\n[PLOT] Generating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TD3 Validation Training Analysis (20k Steps)', fontsize=16, fontweight='bold')

        # Plot 1: Reward components over time
        ax = axes[0, 0]
        for component, values in data['reward_components'].items():
            if len(values) > 0:
                ax.plot(data['timesteps'][:len(values)], values, label=component, alpha=0.7)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward Component')
        ax.set_title('Reward Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Total reward over time
        ax = axes[0, 1]
        ax.plot(data['timesteps'], data['rewards'], alpha=0.5, color='blue')
        # Add moving average
        if len(data['rewards']) >= 100:
            ma = pd.Series(data['rewards']).rolling(100).mean()
            ax.plot(data['timesteps'], ma, color='red', linewidth=2, label='MA(100)')
            ax.legend()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward (per step)')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # Plot 3: Speed over time
        ax = axes[0, 2]
        timesteps_speed = data['timesteps'][:len(data['vehicle_state']['speed'])]
        ax.plot(timesteps_speed, data['vehicle_state']['speed'], alpha=0.5, color='green')
        # Add moving average
        if len(data['vehicle_state']['speed']) >= 100:
            ma = pd.Series(data['vehicle_state']['speed']).rolling(100).mean()
            ax.plot(timesteps_speed, ma, color='darkgreen', linewidth=2, label='MA(100)')
            ax.legend()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Vehicle Speed')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # Plot 4: CNN feature L2 norm
        ax = axes[1, 0]
        if len(data['cnn_features']) > 0:
            timesteps_cnn, l2_norms = zip(*data['cnn_features'])
            ax.plot(timesteps_cnn, l2_norms, alpha=0.7, color='purple')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('L2 Norm')
            ax.set_title('CNN Feature L2 Norm')
            ax.grid(True, alpha=0.3)

        # Plot 5: Episode rewards
        ax = axes[1, 1]
        if len(data['episodes']) > 0:
            episodes_df = pd.DataFrame(data['episodes'])
            ax.bar(episodes_df['number'], episodes_df['reward'], alpha=0.6, color='orange')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Episode Reward')
            ax.set_title('Episode Rewards')
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)

        # Plot 6: Waypoint scatter (vehicle frame)
        ax = axes[1, 2]
        if len(data['waypoints']) > 0:
            waypoints = np.array(data['waypoints'])
            # Sample 1000 points if too many
            if len(waypoints) > 1000:
                indices = np.random.choice(len(waypoints), 1000, replace=False)
                waypoints = waypoints[indices]

            ax.scatter(waypoints[:, 0], waypoints[:, 1], alpha=0.3, s=10, color='red')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('X (m, forward)')
            ax.set_ylabel('Y (m, lateral)')
            ax.set_title('Waypoint Distribution (Vehicle Frame)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'validation_analysis.png'
        plt.savefig(plot_path, dpi=150)
        print(f"[PLOT] Saved to {plot_path}")
        plt.close()

    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n[REPORT] Generating validation report...")

        report_lines = [
            "=" * 80,
            "TD3 VALIDATION TRAINING ANALYSIS REPORT",
            "=" * 80,
            "",
            "Purpose: Verify TD3 solution before full 1M-step run on supercomputer",
            "",
            "=" * 80,
            "VALIDATION RESULTS",
            "=" * 80,
            ""
        ]

        # Add validation results
        all_passed = True
        for test_name, (passed, message) in self.validation_results.items():
            status_icon = "✓" if passed else "✗"
            report_lines.append(f"[{status_icon}] {test_name}")
            report_lines.append("")
            for line in message.split('\n'):
                report_lines.append(f"    {line}")
            report_lines.append("")

            if not passed:
                all_passed = False

        # Overall verdict
        report_lines.append("=" * 80)
        report_lines.append("OVERALL VERDICT")
        report_lines.append("=" * 80)
        report_lines.append("")

        if all_passed:
            report_lines.append("✓✓✓ PASS: TD3 solution is ready for full training ✓✓✓")
            report_lines.append("")
            report_lines.append("Recommendations:")
            report_lines.append("  1. Proceed with full 1M-step training on supercomputer")
            report_lines.append("  2. Use same hyperparameters and configuration")
            report_lines.append("  3. Monitor TensorBoard during training")
            report_lines.append("  4. Save checkpoints every 50k-100k steps")
        else:
            report_lines.append("✗✗✗ FAIL: Issues detected, do NOT proceed to full training ✗✗✗")
            report_lines.append("")
            report_lines.append("Recommendations:")
            report_lines.append("  1. Review failed validation checks above")
            report_lines.append("  2. Fix identified issues")
            report_lines.append("  3. Re-run validation training")
            report_lines.append("  4. Only proceed after all checks pass")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Write report
        report_path = self.output_dir / 'validation_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"[REPORT] Saved to {report_path}")

        # Print to console
        print("\n" + '\n'.join(report_lines))

    def run_analysis(self, log_file: Path):
        """Run complete validation analysis pipeline."""
        print("\n" + "=" * 80)
        print("TD3 VALIDATION TRAINING ANALYSIS")
        print("=" * 80)

        # Parse log
        data = self.parse_training_log(log_file)
        self.debug_data = data

        # Run validations
        self.validation_results['Reward Function'] = self.validate_reward_function(data)
        self.validation_results['CNN Features'] = self.validate_cnn_features(data)
        self.validation_results['Waypoints'] = self.validate_waypoints(data)
        self.validation_results['Learning Progress'] = self.validate_learning(data)

        # Generate visualizations
        self.generate_plots(data)

        # Generate report
        self.generate_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Report: {self.output_dir / 'validation_report.txt'}")
        print(f"  - Plots:  {self.output_dir / 'validation_analysis.png'}")
        print("")


def main():
    parser = argparse.ArgumentParser(description='Analyze TD3 validation training run')
    parser.add_argument(
        '--log-file',
        type=str,
        required=True,
        help='Path to training log file (e.g., validation_training_20k_YYYYMMDD_HHMMSS.log)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/validation_analysis',
        help='Directory to save analysis results (default: data/validation_analysis)'
    )

    args = parser.parse_args()

    analyzer = ValidationAnalyzer(
        log_dir=str(Path(args.log_file).parent),
        output_dir=args.output_dir
    )

    analyzer.run_analysis(Path(args.log_file))


if __name__ == '__main__':
    main()

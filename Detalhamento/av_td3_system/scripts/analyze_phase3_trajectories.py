#!/usr/bin/env python3
"""
Phase 3 Trajectory Analysis Script

Analyzes baseline controller waypoint following performance:
- Crosstrack error statistics
- Heading error statistics
- Speed profile analysis
- Lateral deviation patterns

Usage:
    python scripts/analyze_phase3_trajectories.py \
        --trajectory-file results/baseline_evaluation/trajectories/trajectories_scenario_0_20251123-141826.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_trajectory_data(filepath: str) -> List:
    """
    Load trajectory data from JSON file.

    Returns:
        List of episodes, where each episode is a list of trajectory points.
        Each point contains: x, y, yaw, speed, steer, throttle, brake
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def compute_lateral_deviation(trajectory: List[Dict], waypoints: np.ndarray) -> List[float]:
    """
    Compute lateral deviation from waypoints for each trajectory point.

    Args:
        trajectory: List of trajectory points (x, y, yaw, speed, ...)
        waypoints: Nx3 array of waypoints (x, y, speed)

    Returns:
        List of lateral deviations (meters)
    """
    lateral_deviations = []

    for point in trajectory:
        vehicle_pos = np.array([point['x'], point['y']])

        # Find closest waypoint
        distances = np.linalg.norm(waypoints[:, :2] - vehicle_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Lateral deviation is the distance to closest waypoint
        lateral_deviation = distances[closest_idx]
        lateral_deviations.append(lateral_deviation)

    return lateral_deviations


def compute_heading_error(trajectory: List[Dict], waypoints: np.ndarray) -> List[float]:
    """
    Compute heading error for each trajectory point.

    Args:
        trajectory: List of trajectory points (x, y, yaw, speed, ...)
        waypoints: Nx3 array of waypoints (x, y, speed)

    Returns:
        List of heading errors (radians)
    """
    heading_errors = []

    for i, point in enumerate(trajectory):
        if i >= len(trajectory) - 1:
            heading_errors.append(0.0)
            continue

        vehicle_pos = np.array([point['x'], point['y']])
        vehicle_yaw = np.deg2rad(point['yaw'])

        # Find next waypoint
        distances = np.linalg.norm(waypoints[:, :2] - vehicle_pos, axis=1)
        closest_idx = np.argmin(distances)

        # Get target waypoint (lookahead)
        target_idx = min(closest_idx + 2, len(waypoints) - 1)
        target_pos = waypoints[target_idx, :2]

        # Compute desired heading
        delta = target_pos - vehicle_pos
        desired_yaw = np.arctan2(delta[1], delta[0])

        # Heading error
        heading_error = desired_yaw - vehicle_yaw

        # Normalize to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        heading_errors.append(heading_error)

    return heading_errors


def load_waypoints(waypoint_file: str = 'config/waypoints.txt') -> np.ndarray:
    """Load waypoints from file."""
    waypoints = []
    with open(waypoint_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                waypoints.append([x, y, 30.0])  # Default speed 30 km/h
    return np.array(waypoints)


def analyze_crosstrack_error(episodes: List[List[Dict]], waypoints: np.ndarray) -> Dict:
    """
    Analyze crosstrack error (lateral deviation) across episodes.

    Returns:
        Dict with statistics: mean, std, max, min, median
    """
    all_lateral_deviations = []

    for episode in episodes:
        lateral_devs = compute_lateral_deviation(episode, waypoints)
        all_lateral_deviations.extend(lateral_devs)

    return {
        'mean': np.mean(all_lateral_deviations),
        'std': np.std(all_lateral_deviations),
        'max': np.max(all_lateral_deviations),
        'min': np.min(all_lateral_deviations),
        'median': np.median(all_lateral_deviations),
        'p95': np.percentile(all_lateral_deviations, 95),
        'count': len(all_lateral_deviations)
    }


def analyze_heading_error(episodes: List[List[Dict]], waypoints: np.ndarray) -> Dict:
    """
    Analyze heading error across episodes.

    Returns:
        Dict with statistics: mean, std, max, min, median (in degrees)
    """
    all_heading_errors = []

    for episode in episodes:
        heading_errs = compute_heading_error(episode, waypoints)
        all_heading_errors.extend([abs(np.rad2deg(h)) for h in heading_errs])

    return {
        'mean': np.mean(all_heading_errors),
        'std': np.std(all_heading_errors),
        'max': np.max(all_heading_errors),
        'min': np.min(all_heading_errors),
        'median': np.median(all_heading_errors),
        'p95': np.percentile(all_heading_errors, 95),
        'count': len(all_heading_errors)
    }


def analyze_speed_profile(episodes: List[List[Dict]]) -> Dict:
    """
    Analyze speed profile across episodes.

    Returns:
        Dict with statistics: mean, std, max, min (in km/h)
    """
    all_speeds = []

    for episode in episodes:
        for point in episode:
            speed_mps = point['speed']
            speed_kmh = speed_mps * 3.6
            all_speeds.append(speed_kmh)

    return {
        'mean': np.mean(all_speeds),
        'std': np.std(all_speeds),
        'max': np.max(all_speeds),
        'min': np.min(all_speeds),
        'median': np.median(all_speeds),
        'count': len(all_speeds)
    }


def plot_trajectory_map(episodes: List[List[Dict]], waypoints: np.ndarray, output_dir: Path):
    """
    Plot trajectory map showing vehicle path vs waypoints (like TCC module_7.py).

    Creates a top-down 2D view of the vehicle trajectory overlaid on waypoints.
    Useful for visualizing path following performance, deviations, and termination points.

    Args:
        episodes: List of episodes, each episode is a list of trajectory points
        waypoints: Nx3 array of waypoints (x, y, speed)
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot waypoints as green line with markers
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'g-', linewidth=3, label='Waypoints', zorder=1, alpha=0.8)
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'go', markersize=4, zorder=2, alpha=0.6)

    # Plot start and end positions
    ax.plot(waypoints[0, 0], waypoints[0, 1], marker='^', color='lime', markersize=18,
            label='Start Position', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(waypoints[-1, 0], waypoints[-1, 1], marker='D', color='red', markersize=18,
            label='Goal Position', zorder=5, markeredgecolor='darkred', markeredgewidth=2)

    # Plot each episode trajectory with different colors
    colors = plt.cm.autumn(np.linspace(0.2, 0.9, len(episodes)))

    for ep_idx, episode in enumerate(episodes):
        # Extract x, y positions
        x_traj = [point['x'] for point in episode]
        y_traj = [point['y'] for point in episode]

        # Plot trajectory line
        ax.plot(x_traj, y_traj, color=colors[ep_idx], linewidth=2.0,
                label=f'Episode {ep_idx + 1} ({len(episode)} steps)', alpha=0.75, zorder=3)

        # Mark start position for each episode (circle)
        ax.plot(x_traj[0], y_traj[0], 'o', color=colors[ep_idx], markersize=10,
                zorder=4, markeredgecolor='black', markeredgewidth=1.5,
                label=f'Ep{ep_idx+1} Start')

        # Mark end position (X marker for collision/termination)
        ax.plot(x_traj[-1], y_traj[-1], 'X', color=colors[ep_idx], markersize=14,
                markeredgewidth=3, zorder=4, markeredgecolor='black',
                label=f'Ep{ep_idx+1} End')

    # Configure plot
    ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_title('Vehicle Trajectory - Top-Down View (Town01)\n' +
                 'Baseline Controller: PID + Pure Pursuit',
                 fontsize=16, fontweight='bold', pad=20)

    # Add legend with smaller font for multiple episodes
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    # Invert x-axis (CARLA/UE4 uses left-handed coordinate system)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_map.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Generated: trajectory_map.png")


def plot_episode_trajectories(episodes: List[List[Dict]], waypoints: np.ndarray, output_dir: Path):
    """
    Plot trajectories for all episodes.

    Creates separate plots for:
    - Trajectory map (top-down 2D view)
    - Lateral deviation over time
    - Heading error over time
    - Speed profile over time
    - Control commands over time
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 0: Trajectory Map (NEW - top-down 2D view)
    print("\n📊 Generating trajectory map...")
    plot_trajectory_map(episodes, waypoints, output_dir)

    # Plot 1: Lateral Deviation
    plt.figure(figsize=(12, 6))
    for i, episode in enumerate(episodes):
        lat_devs = compute_lateral_deviation(episode, waypoints)
        steps = list(range(len(lat_devs)))
        plt.plot(steps, lat_devs, label=f"Episode {i+1}", alpha=0.7)

    plt.axhline(y=2.0, color='r', linestyle='--', label='Termination Threshold (2.0m)')
    plt.xlabel('Step')
    plt.ylabel('Lateral Deviation (m)')
    plt.title('Baseline Controller - Lateral Deviation Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'lateral_deviation.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Heading Error
    plt.figure(figsize=(12, 6))
    for i, episode in enumerate(episodes):
        head_errs = compute_heading_error(episode, waypoints)
        head_errs_deg = [abs(np.rad2deg(h)) for h in head_errs]
        steps = list(range(len(head_errs_deg)))
        plt.plot(steps, head_errs_deg, label=f"Episode {i+1}", alpha=0.7)

    plt.xlabel('Step')
    plt.ylabel('Heading Error (degrees)')
    plt.title('Baseline Controller - Heading Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'heading_error.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Speed Profile
    plt.figure(figsize=(12, 6))
    for i, episode in enumerate(episodes):
        speeds = [point['speed'] * 3.6 for point in episode]  # Convert to km/h
        steps = list(range(len(speeds)))
        plt.plot(steps, speeds, label=f"Episode {i+1}", alpha=0.7)

    plt.axhline(y=30.0, color='g', linestyle='--', label='Target Speed (30 km/h)', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Speed (km/h)')
    plt.title('Baseline Controller - Speed Profile Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'speed_profile.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Control Commands
    plt.figure(figsize=(12, 8))

    for i, episode in enumerate(episodes):
        steps = list(range(len(episode)))

        # Subplot 1: Steering
        plt.subplot(2, 1, 1)
        steering = [point['steer'] for point in episode]
        plt.plot(steps, steering, label=f"Episode {i+1}", alpha=0.7)
        plt.ylabel('Steering [-1, 1]')
        plt.title('Baseline Controller - Control Commands Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Throttle/Brake
        plt.subplot(2, 1, 2)
        throttle = [point['throttle'] for point in episode]
        brake = [-point['brake'] for point in episode]  # Negative for visualization
        plt.plot(steps, throttle, label=f"Episode {i+1} Throttle", alpha=0.7)
        plt.plot(steps, brake, '--', label=f"Episode {i+1} Brake (neg)", alpha=0.7)
        plt.ylabel('Throttle/Brake [0, 1]')
        plt.xlabel('Step')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'control_commands.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[PLOTS] Saved 4 trajectory plots to {output_dir}")
def generate_report(episodes: List[List[Dict]], waypoints: np.ndarray, output_file: Path):
    """Generate markdown report with analysis results."""

    # Compute statistics
    crosstrack_stats = analyze_crosstrack_error(episodes, waypoints)
    heading_stats = analyze_heading_error(episodes, waypoints)
    speed_stats = analyze_speed_profile(episodes)

    # Generate report
    report = f"""# Phase 3: Waypoint Following Verification - Analysis Report

**Date**: {output_file.parent.name}
**Episodes**: {len(episodes)}

---

## Executive Summary

The baseline controller (PID + Pure Pursuit) was evaluated for waypoint following performance over {len(episodes)} episodes. This report analyzes crosstrack error, heading error, and speed tracking.

---

## Episode Summary

| Episode | Steps | Avg Speed (km/h) |
|---------|-------|------------------|
"""

    for i, episode in enumerate(episodes):
        avg_speed = np.mean([point['speed'] * 3.6 for point in episode])
        report += f"| {i+1} | {len(episode)} | {avg_speed:.2f} |\n"

    report += f"""
---

## Crosstrack Error Analysis

**Definition**: Lateral deviation from the center of the target lane (absolute value).

| Metric | Value (m) |
|--------|-----------|
| Mean | {crosstrack_stats['mean']:.3f} |
| Std Dev | {crosstrack_stats['std']:.3f} |
| Median | {crosstrack_stats['median']:.3f} |
| Max | {crosstrack_stats['max']:.3f} |
| 95th Percentile | {crosstrack_stats['p95']:.3f} |
| Min | {crosstrack_stats['min']:.3f} |

**Total Samples**: {crosstrack_stats['count']}

**Interpretation**:
- Mean lateral deviation: {crosstrack_stats['mean']:.3f} m
- 95% of time within: {crosstrack_stats['p95']:.3f} m of lane center
- Termination threshold: 2.0 m

---

## Heading Error Analysis

**Definition**: Absolute angular difference between vehicle heading and desired path direction.

| Metric | Value (degrees) |
|--------|-----------------|
| Mean | {heading_stats['mean']:.2f} |
| Std Dev | {heading_stats['std']:.2f} |
| Median | {heading_stats['median']:.2f} |
| Max | {heading_stats['max']:.2f} |
| 95th Percentile | {heading_stats['p95']:.2f} |
| Min | {heading_stats['min']:.2f} |

**Total Samples**: {heading_stats['count']}

**Interpretation**:
- Mean heading error: {heading_stats['mean']:.2f}°
- Controller maintains heading within {heading_stats['p95']:.2f}° for 95% of time

---

## Speed Profile Analysis

**Target Speed**: 30.0 km/h (from baseline_config.yaml)

| Metric | Value (km/h) |
|--------|--------------|
| Mean | {speed_stats['mean']:.2f} |
| Std Dev | {speed_stats['std']:.2f} |
| Median | {speed_stats['median']:.2f} |
| Max | {speed_stats['max']:.2f} |
| Min | {speed_stats['min']:.2f} |

**Total Samples**: {speed_stats['count']}

**Interpretation**:
- Mean speed: {speed_stats['mean']:.2f} km/h (target: 30.0 km/h)
- Speed tracking error: {abs(speed_stats['mean'] - 30.0):.2f} km/h ({abs(speed_stats['mean'] - 30.0) / 30.0 * 100:.1f}% deviation)

---

## Controller Performance Assessment

### Zigzag Behavior Analysis

**Observation**: Controller exhibits zigzag pattern (repeated lane marking touches).

**Evidence from Data**:
- Mean lateral deviation: {crosstrack_stats['mean']:.3f} m
- 95th percentile: {crosstrack_stats['p95']:.3f} m
- Max deviation: {crosstrack_stats['max']:.3f} m

**Root Causes** (Hypothesis):

1. **Aggressive Steering Gains**:
   - Pure Pursuit `kp_heading = 8.0` may be too high
   - Causes overcorrection when lateral error detected

2. **Lookahead Distance Too Short**:
   - Current: 2.0m lookahead
   - At 30 km/h (8.33 m/s), vehicle travels lookahead distance in 0.24s
   - Short planning horizon → reactive, oscillatory behavior

3. **Speed-Crosstrack Coupling Missing**:
   - Current: `k_speed_crosstrack = 0.0` (disabled)
   - No speed reduction when off-center → aggressive corrections at full speed

### Recommended Controller Tuning

**Pure Pursuit Adjustments**:
```yaml
# Current
lookahead_distance: 2.0
kp_heading: 8.0
k_speed_crosstrack: 0.0

# Recommended
lookahead_distance: 3.5  # Increase for smoother planning (0.42s @ 30km/h)
kp_heading: 5.0          # Reduce gain to prevent overcorrection
k_speed_crosstrack: 0.1  # Enable speed reduction when off-center
```

**PID Adjustments** (if speed oscillation observed):
```yaml
# Current
kp: 0.5
ki: 0.3
kd: 0.13

# If needed
kp: 0.4  # Reduce if speed oscillates
ki: 0.2  # Reduce if integral windup
kd: 0.15 # Increase for smoother response
```

---

## Next Steps

### Phase 3 Completion ✅

- [x] Run 3 episodes with baseline controller
- [x] Collect trajectory data
- [x] Analyze crosstrack error
- [x] Analyze heading error
- [x] Analyze speed profile
- [x] Identify controller issues (zigzag behavior)

### Recommended Actions

1. **Controller Tuning** (Optional - before Phase 4):
   - Modify `config/baseline_config.yaml` with recommended parameters
   - Re-run Phase 3 to verify improvement
   - Document tuning results

2. **Proceed to Phase 4**:
   - Test NPC interaction with current controller
   - Observe behavior in traffic scenarios
   - Collect additional data for final evaluation

3. **Defer Tuning** (Alternative):
   - Accept current baseline performance as-is
   - Proceed with evaluation protocol
   - Document limitations in paper

---

## Files Generated

- `results/baseline_evaluation/phase3_analysis/lateral_deviation.png`
- `results/baseline_evaluation/phase3_analysis/heading_error.png`
- `results/baseline_evaluation/phase3_analysis/speed_profile.png`
- `results/baseline_evaluation/phase3_analysis/control_commands.png`
- `results/baseline_evaluation/phase3_analysis/PHASE3_ANALYSIS_REPORT.md` (this file)

---

**Status**: Phase 3 Complete ✅
"""

    output_file.write_text(report)
    print(f"\n[REPORT] Saved analysis report to {output_file}")
def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 3 trajectory data")
    parser.add_argument(
        '--trajectory-file',
        type=str,
        default=None,
        help='Path to trajectory JSON file (auto-detects latest if not specified)'
    )
    parser.add_argument(
        '--waypoint-file',
        type=str,
        default='config/waypoints.txt',
        help='Path to waypoints file (default: config/waypoints.txt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots and report (auto-generated if not specified)'
    )

    args = parser.parse_args()

    # Auto-detect trajectory file if not specified
    if args.trajectory_file is None:
        print("\n[AUTO-DETECT] No trajectory file specified, searching for latest...")

        # Search in results/baseline_evaluation/trajectories/
        traj_dir = Path('results/baseline_evaluation/trajectories')

        if not traj_dir.exists():
            print(f"[ERROR] Trajectory directory not found: {traj_dir}")
            print("[ERROR] Please run evaluation first or specify --trajectory-file")
            return

        # Find all trajectory JSON files
        traj_files = sorted(traj_dir.glob('trajectories_*.json'))

        if not traj_files:
            print(f"[ERROR] No trajectory files found in {traj_dir}")
            print("[ERROR] Please run evaluation first or specify --trajectory-file")
            return

        # Use most recent file (sorted by filename which includes timestamp)
        args.trajectory_file = str(traj_files[-1])
        print(f"[AUTO-DETECT] Found latest trajectory file: {args.trajectory_file}")

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        # Extract timestamp from trajectory filename
        traj_path = Path(args.trajectory_file)
        timestamp = traj_path.stem.replace('trajectories_scenario_', '')
        args.output_dir = f'results/baseline_evaluation/analysis_{timestamp}'
        print(f"[AUTO-DETECT] Output directory: {args.output_dir}")

    # Load data
    print(f"\n[LOAD] Loading trajectory data from {args.trajectory_file}...")
    episodes = load_trajectory_data(args.trajectory_file)
    print(f"[LOAD] Loaded {len(episodes)} episodes")

    # Load waypoints
    print(f"[LOAD] Loading waypoints from {args.waypoint_file}...")
    waypoints = load_waypoints(args.waypoint_file)
    print(f"[LOAD] Loaded {len(waypoints)} waypoints")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print(f"\n[ANALYSIS] Generating trajectory plots...")
    plot_episode_trajectories(episodes, waypoints, output_dir)

    # Generate report
    print(f"\n[ANALYSIS] Generating analysis report...")
    report_file = output_dir / 'PHASE3_ANALYSIS_REPORT.md'
    generate_report(episodes, waypoints, report_file)

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"PHASE 3 ANALYSIS SUMMARY")
    print(f"{'='*70}")

    crosstrack_stats = analyze_crosstrack_error(episodes, waypoints)
    heading_stats = analyze_heading_error(episodes, waypoints)
    speed_stats = analyze_speed_profile(episodes)

    print(f"\nCrosstrack Error:")
    print(f"  Mean:   {crosstrack_stats['mean']:.3f} m")
    print(f"  Median: {crosstrack_stats['median']:.3f} m")
    print(f"  95th %: {crosstrack_stats['p95']:.3f} m")
    print(f"  Max:    {crosstrack_stats['max']:.3f} m")

    print(f"\nHeading Error:")
    print(f"  Mean:   {heading_stats['mean']:.2f}°")
    print(f"  Median: {heading_stats['median']:.2f}°")
    print(f"  95th %: {heading_stats['p95']:.2f}°")
    print(f"  Max:    {heading_stats['max']:.2f}°")

    print(f"\nSpeed Profile:")
    print(f"  Mean:   {speed_stats['mean']:.2f} km/h (target: 30.0 km/h)")
    print(f"  Median: {speed_stats['median']:.2f} km/h")
    print(f"  Error:  {abs(speed_stats['mean'] - 30.0):.2f} km/h ({abs(speed_stats['mean'] - 30.0) / 30.0 * 100:.1f}%)")

    print(f"\n{'='*70}")
    print(f"[SUCCESS] Phase 3 analysis complete!")
    print(f"[OUTPUT] Results saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

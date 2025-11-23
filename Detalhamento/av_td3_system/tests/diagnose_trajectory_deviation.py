#!/usr/bin/env python3
"""
Trajectory Deviation Diagnostic Tool

Analyzes sudden lateral deviations in trajectory data to identify root causes.
Focuses on step ~130 where vehicle starts moving left unexpectedly.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_trajectory(filepath):
    """Load trajectory JSON data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_waypoints(filepath):
    """Load waypoints from TXT file."""
    waypoints = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                waypoints.append((x, y))
    return np.array(waypoints)


def calculate_crosstrack_error(x, y, waypoints):
    """Calculate minimum crosstrack error to waypoint path."""
    point = np.array([x, y])
    distances = np.linalg.norm(waypoints - point, axis=1)
    min_idx = np.argmin(distances)
    return distances[min_idx], min_idx


def analyze_episode(episode_data, waypoints, episode_num=0):
    """Analyze single episode for deviations."""
    
    if not episode_data:
        print(f"Episode {episode_num}: No data")
        return None
    
    print(f"\n{'='*80}")
    print(f"EPISODE {episode_num} ANALYSIS")
    print(f"{'='*80}")
    print(f"Total steps: {len(episode_data)}")
    
    # Extract data
    steps = []
    x_coords = []
    y_coords = []
    yaw_angles = []
    speeds = []
    steers = []
    crosstrack_errors = []
    wp_indices = []
    
    for i, step in enumerate(episode_data):
        steps.append(i)
        x_coords.append(step['x'])
        y_coords.append(step['y'])
        yaw_angles.append(step['yaw'])
        speeds.append(step['speed'])
        steers.append(step['steer'])
        
        # Calculate crosstrack error
        ct_error, wp_idx = calculate_crosstrack_error(step['x'], step['y'], waypoints)
        crosstrack_errors.append(ct_error)
        wp_indices.append(wp_idx)
    
    # Convert to numpy arrays
    steps = np.array(steps)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    yaw_angles = np.array(yaw_angles)
    speeds = np.array(speeds)
    steers = np.array(steers)
    crosstrack_errors = np.array(crosstrack_errors)
    wp_indices = np.array(wp_indices)
    
    # Calculate derivatives
    dy_dx = np.gradient(y_coords, x_coords)
    d_yaw = np.gradient(yaw_angles)
    d_steer = np.gradient(steers)
    d_ct_error = np.gradient(crosstrack_errors)
    
    # Find problem area (step ~130)
    problem_start = max(0, 130 - 20)
    problem_end = min(len(episode_data), 130 + 50)
    
    print(f"\nFOCUS AREA: Steps {problem_start}-{problem_end}")
    print(f"{'-'*80}")
    
    # Analyze Y-coordinate changes
    if problem_end <= len(y_coords):
        y_before = y_coords[problem_start]
        y_problem = y_coords[problem_start:problem_end]
        y_change = y_problem - y_before
        
        print(f"\nY-Coordinate Analysis:")
        print(f"  Y at step {problem_start}: {y_before:.4f}")
        print(f"  Y at step 130: {y_coords[130]:.4f}")
        print(f"  Y change (130-{problem_start}): {y_coords[130] - y_before:.4f}")
        print(f"  Max Y in range: {np.max(y_problem):.4f}")
        print(f"  Min Y in range: {np.min(y_problem):.4f}")
        print(f"  Y drift: {np.max(y_problem) - np.min(y_problem):.4f}")
    
    # Analyze steering
    if problem_end <= len(steers):
        steer_before = steers[problem_start]
        steer_problem = steers[problem_start:problem_end]
        
        print(f"\nSteering Analysis:")
        print(f"  Steer at step {problem_start}: {steer_before:.6f} rad ({np.degrees(steer_before):.3f}°)")
        print(f"  Steer at step 130: {steers[130]:.6f} rad ({np.degrees(steers[130]):.3f}°)")
        print(f"  Mean steer in range: {np.mean(steer_problem):.6f} rad ({np.degrees(np.mean(steer_problem)):.3f}°)")
        print(f"  Max steer in range: {np.max(steer_problem):.6f} rad ({np.degrees(np.max(steer_problem)):.3f}°)")
        print(f"  Min steer in range: {np.min(steer_problem):.6f} rad ({np.degrees(np.min(steer_problem)):.3f}°)")
        print(f"  Steer std dev: {np.std(steer_problem):.6f} rad ({np.degrees(np.std(steer_problem)):.3f}°)")
    
    # Analyze crosstrack error
    if problem_end <= len(crosstrack_errors):
        ct_before = crosstrack_errors[problem_start]
        ct_problem = crosstrack_errors[problem_start:problem_end]
        
        print(f"\nCrosstrack Error Analysis:")
        print(f"  CT error at step {problem_start}: {ct_before:.4f} m")
        print(f"  CT error at step 130: {crosstrack_errors[130]:.4f} m")
        print(f"  Mean CT error in range: {np.mean(ct_problem):.4f} m")
        print(f"  Max CT error in range: {np.max(ct_problem):.4f} m")
        print(f"  CT error change rate: {np.mean(d_ct_error[problem_start:problem_end]):.6f} m/step")
    
    # Analyze waypoint tracking
    if problem_end <= len(wp_indices):
        wp_before = wp_indices[problem_start]
        wp_130 = wp_indices[130]
        wp_problem = wp_indices[problem_start:problem_end]
        
        print(f"\nWaypoint Tracking:")
        print(f"  Waypoint index at step {problem_start}: {wp_before}")
        print(f"  Waypoint index at step 130: {wp_130}")
        print(f"  Waypoint progress: {wp_130 - wp_before} waypoints")
        print(f"  Waypoint indices in range: {np.min(wp_problem)} to {np.max(wp_problem)}")
    
    # Identify critical moments
    print(f"\nCRITICAL MOMENTS:")
    print(f"{'-'*80}")
    
    # Find steps with largest Y-coordinate changes
    if len(dy_dx) > problem_end:
        dy_dx_problem = np.abs(dy_dx[problem_start:problem_end])
        critical_dy_idx = np.argsort(dy_dx_problem)[-5:]  # Top 5
        print(f"\nTop 5 Y-direction changes (dy/dx):")
        for idx in critical_dy_idx[::-1]:
            actual_step = problem_start + idx
            print(f"  Step {actual_step}: dy/dx = {dy_dx[actual_step]:.6f}, " +
                  f"Y = {y_coords[actual_step]:.4f}, " +
                  f"Steer = {steers[actual_step]:.6f} rad ({np.degrees(steers[actual_step]):.3f}°)")
    
    # Find steps with largest steering changes
    if len(d_steer) > problem_end:
        d_steer_problem = np.abs(d_steer[problem_start:problem_end])
        critical_steer_idx = np.argsort(d_steer_problem)[-5:]  # Top 5
        print(f"\nTop 5 steering rate changes (d_steer/dt):")
        for idx in critical_steer_idx[::-1]:
            actual_step = problem_start + idx
            print(f"  Step {actual_step}: d_steer = {d_steer[actual_step]:.6f} rad/step, " +
                  f"Steer = {steers[actual_step]:.6f} rad ({np.degrees(steers[actual_step]):.3f}°)")
    
    # Detailed step-by-step for critical range
    print(f"\nDETAILED STEP-BY-STEP (Steps 125-140):")
    print(f"{'-'*80}")
    print(f"{'Step':<6} {'X':<10} {'Y':<10} {'Yaw (°)':<10} {'Speed':<8} {'Steer (°)':<12} {'CT Err':<8} {'WP Idx':<8}")
    print(f"{'-'*80}")
    
    for step_idx in range(125, min(141, len(episode_data))):
        print(f"{step_idx:<6} " +
              f"{x_coords[step_idx]:<10.3f} " +
              f"{y_coords[step_idx]:<10.4f} " +
              f"{yaw_angles[step_idx]:<10.3f} " +
              f"{speeds[step_idx]:<8.3f} " +
              f"{np.degrees(steers[step_idx]):<12.6f} " +
              f"{crosstrack_errors[step_idx]:<8.4f} " +
              f"{wp_indices[step_idx]:<8}")
    
    # Generate diagnostic plots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: Trajectory with waypoints
    axes[0].plot(waypoints[:, 0], waypoints[:, 1], 'g--', alpha=0.5, label='Waypoints', linewidth=2)
    axes[0].plot(x_coords, y_coords, 'b-', alpha=0.7, label='Vehicle Path', linewidth=1)
    axes[0].scatter(x_coords[problem_start:problem_end], y_coords[problem_start:problem_end], 
                   c='red', s=20, alpha=0.5, label=f'Problem Area ({problem_start}-{problem_end})')
    axes[0].scatter(x_coords[130], y_coords[130], c='orange', s=100, marker='*', 
                   label=f'Step 130', zorder=5)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'Episode {episode_num}: Trajectory vs Waypoints')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot 2: Y-coordinate over steps
    axes[1].plot(steps, y_coords, 'b-', linewidth=1)
    axes[1].axhline(y=129.49, color='g', linestyle='--', alpha=0.5, label='Target Y (waypoints)')
    axes[1].axvspan(problem_start, problem_end, alpha=0.2, color='red', label='Problem Area')
    axes[1].axvline(x=130, color='orange', linestyle=':', label='Step 130')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Y Coordinate (m)')
    axes[1].set_title('Y-Coordinate Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Steering angle over steps
    axes[2].plot(steps, np.degrees(steers), 'r-', linewidth=1)
    axes[2].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Zero Steer')
    axes[2].axvspan(problem_start, problem_end, alpha=0.2, color='red', label='Problem Area')
    axes[2].axvline(x=130, color='orange', linestyle=':', label='Step 130')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Steering Angle (degrees)')
    axes[2].set_title('Steering Command Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Crosstrack error over steps
    axes[3].plot(steps, crosstrack_errors, 'purple', linewidth=1)
    axes[3].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Zero Error')
    axes[3].axvspan(problem_start, problem_end, alpha=0.2, color='red', label='Problem Area')
    axes[3].axvline(x=130, color='orange', linestyle=':', label='Step 130')
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Crosstrack Error (m)')
    axes[3].set_title('Crosstrack Error Over Time')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, {
        'steps': steps,
        'x': x_coords,
        'y': y_coords,
        'yaw': yaw_angles,
        'speed': speeds,
        'steer': steers,
        'crosstrack_error': crosstrack_errors,
        'wp_indices': wp_indices
    }


def main():
    """Main diagnostic function."""
    # Paths
    workspace = Path('/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system')
    traj_file = workspace / 'results/baseline_evaluation/trajectories/trajectories_scenario_0_20251123-161931.json'
    waypoints_file = workspace / 'config/waypoints.txt'
    output_dir = workspace / 'docs/day-23/baseline/diagnosis'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRAJECTORY DEVIATION DIAGNOSTIC")
    print("="*80)
    print(f"Trajectory file: {traj_file}")
    print(f"Waypoints file: {waypoints_file}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    trajectory_data = load_trajectory(traj_file)
    waypoints = load_waypoints(waypoints_file)
    
    print(f"  Loaded {len(trajectory_data)} episodes")
    print(f"  Loaded {len(waypoints)} waypoints")
    
    # Analyze each episode
    for ep_num, episode in enumerate(trajectory_data):
        if not episode:
            continue
            
        result = analyze_episode(episode, waypoints, ep_num)
        
        if result:
            fig, data = result
            
            # Save plot
            plot_file = output_dir / f'episode_{ep_num}_deviation_diagnosis.png'
            fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"\nSaved diagnostic plot: {plot_file}")
            plt.close(fig)
    
    print(f"\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

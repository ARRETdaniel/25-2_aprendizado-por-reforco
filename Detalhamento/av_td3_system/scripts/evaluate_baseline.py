#!/usr/bin/env python3
"""
Baseline Controller (PID + Pure Pursuit) Evaluation Script for CARLA 0.9.16

This script evaluates the classical PID + Pure Pursuit controller for comparison
with the TD3 deep reinforcement learning agent. It follows the same evaluation
protocol as train_td3.py to ensure fair comparison.

Evaluation Metrics (from research paper):
- Safety: Success Rate (%), Avg. Collisions/km, TTC analysis
- Efficiency: Avg. Speed (km/h), Route Completion Time (s)
- Comfort: Avg. Longitudinal Jerk (m/s³), Avg. Lateral Acceleration (m/s²)

Configuration:
- Load from config/baseline_config.yaml
- Uses same CARLA environment as TD3 (carla_env.py)
- Uses same waypoints as TD3 (config/waypoints.txt)

Author: Daniel Terra
Date: 2025
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
import cv2

from src.baselines.baseline_controller import BaselineController
from src.environment.carla_env import CARLANavigationEnv


class BaselineEvaluationPipeline:
    """
    Evaluation pipeline for PID + Pure Pursuit baseline controller.

    Follows the same structure as TD3TrainingPipeline.evaluate() for fair comparison.
    """

    def __init__(
        self,
        scenario: int = 0,
        seed: int = 42,
        num_episodes: int = 20,
        carla_config_path: str = "config/carla_config.yaml",
        agent_config_path: str = "config/td3_config.yaml",  # For env compatibility
        training_config_path: str = "config/training_config.yaml",
        baseline_config_path: str = "config/baseline_config.yaml",
        output_dir: str = "results/baseline_evaluation",
        save_trajectory: bool = True,
        debug: bool = False,
        use_ros_bridge: bool = False
    ):
        """
        Initialize baseline evaluation pipeline.

        Args:
            scenario: Traffic density scenario (0=20, 1=50, 2=100 NPCs)
            seed: Random seed for reproducibility
            num_episodes: Number of evaluation episodes (default: 20)
            carla_config_path: Path to CARLA config
            agent_config_path: Path to TD3 config (for env initialization)
            training_config_path: Path to training config (with scenarios)
            baseline_config_path: Path to baseline controller config
            output_dir: Directory for evaluation results
            save_trajectory: Whether to save trajectory data
            debug: Enable debug logging
            use_ros_bridge: Use ROS 2 Bridge for vehicle control (Phase 5)
        """
        # Set random seeds
        np.random.seed(seed)

        self.scenario = scenario
        self.seed = seed
        self.num_episodes = num_episodes
        self.save_trajectory = save_trajectory
        self.debug = debug
        self.use_ros_bridge = use_ros_bridge

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='[%(levelname)s] %(message)s'
        )

        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configurations
        print("\n" + "="*70)
        print("BASELINE CONTROLLER EVALUATION - PID + PURE PURSUIT")
        print("="*70)
        print(f"\n[CONFIG] Loading configurations...")

        with open(carla_config_path, 'r') as f:
            self.carla_config = yaml.safe_load(f)

        with open(agent_config_path, 'r') as f:
            self.agent_config = yaml.safe_load(f)

        with open(baseline_config_path, 'r') as f:
            self.baseline_config = yaml.safe_load(f)

        print(f"[CONFIG] CARLA config: {carla_config_path}")
        print(f"[CONFIG] Baseline config: {baseline_config_path}")
        print(f"[CONFIG] Scenario: {scenario} (0=20, 1=50, 2=100 NPCs)")

        # Update NPC density based on scenario
        npc_densities = [20, 50, 100]
        os.environ['CARLA_SCENARIO_INDEX'] = str(scenario)
        print(f"[CONFIG] Set CARLA_SCENARIO_INDEX={scenario}")

        if scenario < len(npc_densities):
            print(f"[CONFIG] Expected NPC count: {npc_densities[scenario]}")
        else:
            print(f"[WARNING] Invalid scenario index, will use default")

        # Traffic Manager port configuration
        # Must match TD3 training for fair comparison
        self.tm_port = 8000
        print(f"[CONFIG] Traffic Manager port: {self.tm_port}")

        # Initialize environment
        print(f"\n[ENVIRONMENT] Initializing CARLA environment...")
        if self.use_ros_bridge:
            print(f"[ENVIRONMENT] ROS 2 Bridge control ENABLED (Phase 5)")
            print(f"[ENVIRONMENT] Vehicle control via /carla/ego_vehicle/vehicle_control_cmd")
        else:
            print(f"[ENVIRONMENT] Using direct CARLA API control")

        self.env = CARLANavigationEnv(
            carla_config_path,
            agent_config_path,
            training_config_path,
            tm_port=self.tm_port,
            use_ros_bridge=self.use_ros_bridge
        )
        print(f"[ENVIRONMENT] Map: {self.carla_config.get('world', {}).get('map', 'Town01')}")
        print(f"[ENVIRONMENT] Max episode steps: {self.agent_config.get('training', {}).get('max_episode_steps', 2000)}")

        # Load waypoints
        waypoints_file = self.carla_config.get("route", {}).get("waypoints_file", "config/waypoints.txt")
        self.waypoints = self._load_waypoints(waypoints_file)
        print(f"[WAYPOINTS] Loaded {len(self.waypoints)} waypoints from {waypoints_file}")

        # Initialize baseline controller
        print(f"\n[CONTROLLER] Initializing PID + Pure Pursuit baseline...")
        controller_config = self.baseline_config['baseline_controller']

        # Initialize with TRUE Pure Pursuit parameters (fixed 2025-01-23)
        self.controller = BaselineController(
            pid_kp=controller_config['pid']['kp'],
            pid_ki=controller_config['pid']['ki'],
            pid_kd=controller_config['pid']['kd'],
            kp_lookahead=controller_config['pure_pursuit']['kp_lookahead'],
            min_lookahead=controller_config['pure_pursuit']['min_lookahead'],
            wheelbase=controller_config['pure_pursuit']['wheelbase'],
            target_speed=controller_config['general']['target_speed_kmh']
        )
        print(f"[CONTROLLER] PID gains: kp={controller_config['pid']['kp']}, ki={controller_config['pid']['ki']}, kd={controller_config['pid']['kd']}")
        print(f"[CONTROLLER] Pure Pursuit: kp_lookahead={controller_config['pure_pursuit']['kp_lookahead']}, min_lookahead={controller_config['pure_pursuit']['min_lookahead']}m, wheelbase={controller_config['pure_pursuit']['wheelbase']}m")
        print(f"[CONTROLLER] Target speed: {controller_config['general']['target_speed_kmh']} km/h")

        # Get simulation timestep
        self.dt = self.carla_config.get("simulation", {}).get("fixed_delta_seconds", 0.05)
        print(f"[CONTROLLER] Simulation timestep: {self.dt}s")

        # Statistics tracking
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_lane_invasions = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_jerks = []
        self.episode_lateral_accels = []
        self.episode_trajectories = []

        # NEW: Statistics for paper-ready metrics
        self.episode_jerks_list = []           # Avg jerk per episode (m/s³)
        self.episode_lateral_accels_list = []  # Avg lateral accel per episode (m/s²)
        self.episode_ttc_list = []             # Avg TTC per episode (seconds)
        self.episode_collisions_per_km = []    # Collisions/km per episode
        self.episode_completion_times = []     # Episode duration (seconds)
        self.episode_route_distances_km = []   # Route length traveled (km)

        # Setup debug visualization if enabled
        if self.debug:
            print(f"\n[DEBUG MODE ENABLED]")
            print(f"[DEBUG] Visual feedback enabled (OpenCV display)")
            print(f"[DEBUG] Press 'q' to quit, 'p' to pause/unpause")

            # Setup OpenCV window
            self.window_name = "Baseline Evaluation - Debug View"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 600)  # 800px camera + 400px info
            self.paused = False

        print(f"\n[INIT] Baseline evaluation pipeline ready!")
        print(f"[INIT] Number of episodes: {num_episodes}")
        print(f"[INIT] Seed: {seed}")
        print(f"[INIT] Debug mode: {debug}")
        print("="*70 + "\n")

    def _load_waypoints(self, waypoints_file: str) -> List[Tuple[float, float, float]]:
        """
        Load waypoints from text file.

        Format: Each line contains "x, y, z" (comma-separated)

        Args:
            waypoints_file: Path to waypoints file

        Returns:
            List of (x, y, z) tuples
        """
        waypoints = []

        with open(waypoints_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        z = float(parts[2].strip())
                        waypoints.append((x, y, z))

        return waypoints

    def _display_debug_frame(self, obs_dict, control, info, step, episode_reward):
        """
        Display debug visualization for baseline evaluation.

        Shows:
        - Camera view (800x600) from front-facing camera
        - Info panel (400x600) with control/state information

        Args:
            obs_dict: Observation dictionary with 'image' key (4-frame stack, shape: (4, 84, 84))
            control: CARLA VehicleControl object with steer, throttle, brake
            info: Info dictionary from environment
            step: Current step number
            episode_reward: Cumulative episode reward
        """
        if not self.debug:
            return

        try:
            # ===== CAMERA FRAME PROCESSING =====
            # Extract latest frame from stack (shape: 84x84 grayscale)
            latest_frame = obs_dict['image'][-1]  # Last frame of 4-frame stack

            # CRITICAL: Denormalize from [-1, 1] to [0, 1]
            # (Environment normalizes images to [-1, 1] range)
            latest_frame_denorm = (latest_frame + 1.0) / 2.0

            # Convert to uint8 [0, 255]
            frame_uint8 = (latest_frame_denorm * 255).astype(np.uint8)

            # Resize for display (84x84 -> 800x600)
            frame_resized = cv2.resize(frame_uint8, (800, 600), interpolation=cv2.INTER_LINEAR)

            # Convert grayscale to BGR for color overlay
            display_frame = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

            # ===== INFO PANEL CREATION =====
            info_panel = np.zeros((600, 400, 3), dtype=np.uint8)

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (255, 255, 255)  # White text
            y_offset = 30
            line_spacing = 25

            # Title
            cv2.putText(info_panel, "BASELINE EVALUATION - DEBUG", (10, y_offset),
                       font, 0.6, (0, 255, 255), 2)  # Yellow title
            y_offset += line_spacing + 10

            # Separator
            cv2.line(info_panel, (10, y_offset), (390, y_offset), (100, 100, 100), 1)
            y_offset += line_spacing

            # Episode/Step info
            cv2.putText(info_panel, f"Step: {step}", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            # Control commands
            cv2.putText(info_panel, "CONTROL COMMANDS:", (10, y_offset),
                       font, font_scale, (0, 255, 0), thickness)  # Green
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Steering:  {control.steer:+.3f}", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Throttle:  {control.throttle:.3f}", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Brake:     {control.brake:.3f}", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing + 10

            # Vehicle state
            speed_kmh = info.get('speed', 0.0)
            lateral_dev = info.get('lateral_deviation', 0.0)
            heading_error = info.get('heading_error', 0.0)

            cv2.putText(info_panel, "VEHICLE STATE:", (10, y_offset),
                       font, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Speed:     {speed_kmh:.2f} km/h", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Lat Dev:   {lateral_dev:.3f} m", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Head Err:  {np.rad2deg(heading_error):.2f} deg", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing + 10

            # Reward information
            cv2.putText(info_panel, "REWARD:", (10, y_offset),
                       font, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Episode:   {episode_reward:.2f}", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing + 10

            # Progress information
            distance_to_goal = info.get('distance_to_goal', 0.0)
            waypoint_idx = info.get('waypoint_index', 0)

            cv2.putText(info_panel, "PROGRESS:", (10, y_offset),
                       font, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Dist Goal: {distance_to_goal:.2f} m", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Waypoint:  {waypoint_idx}", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing + 10

            # Safety information
            collision_count = info.get('collision_count', 0)
            lane_invasion_count = info.get('lane_invasion_count', 0)

            cv2.putText(info_panel, "SAFETY:", (10, y_offset),
                       font, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Collisions:    {collision_count}", (10, y_offset),
                       font, font_scale, (0, 0, 255) if collision_count > 0 else color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, f"  Lane Inv:      {lane_invasion_count}", (10, y_offset),
                       font, font_scale, (255, 0, 0) if lane_invasion_count > 0 else color, thickness)
            y_offset += line_spacing + 20

            # Instructions
            cv2.putText(info_panel, "CONTROLS:", (10, y_offset),
                       font, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, "  'q' - Quit", (10, y_offset),
                       font, font_scale, color, thickness)
            y_offset += line_spacing

            cv2.putText(info_panel, "  'p' - Pause/Unpause", (10, y_offset),
                       font, font_scale, color, thickness)

            # ===== COMBINE AND DISPLAY =====
            combined_frame = np.hstack([display_frame, info_panel])
            cv2.imshow(self.window_name, combined_frame)

            # ===== KEYBOARD HANDLING =====
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print(f"\n[DEBUG] User requested quit (pressed 'q')")
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord('p'):
                self.paused = not self.paused
                if self.paused:
                    print(f"\n[DEBUG] PAUSED - Press 'p' to resume")
                    while self.paused:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('p'):
                            self.paused = False
                            print(f"[DEBUG] RESUMED")
                        elif key == ord('q'):
                            print(f"\n[DEBUG] User requested quit (pressed 'q')")
                            cv2.destroyAllWindows()
                            sys.exit(0)

        except Exception as e:
            print(f"[DEBUG] Visualization error: {e}")
            import traceback
            traceback.print_exc()

    def evaluate(self):
        """
        Run evaluation episodes and collect metrics.

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n[EVAL] Starting evaluation...")

        max_episode_steps = self.agent_config.get("training", {}).get("max_episode_steps", 1000)

        for episode in range(self.num_episodes):
            print(f"\n[EVAL] Episode {episode + 1}/{self.num_episodes}")

            # Reset environment
            obs_dict, reset_info = self.env.reset()

            # Reset controller
            self.controller.reset()

            # Episode tracking
            episode_reward = 0.0
            episode_length = 0
            episode_speeds = []
            episode_jerks = []
            episode_lateral_accels = []
            episode_ttc_values = []  # Time-to-Collision measurements
            trajectory = []

            # Track velocity and acceleration history for jerk calculation
            velocity_history = []  # m/s
            acceleration_history = []  # m/s²
            prev_velocity = 0.0
            prev_acceleration = 0.0

            # Track route length traveled for collisions/km metric
            route_distance_traveled = 0.0
            prev_vehicle_location = None

            # Track episode start time for completion time metric
            episode_start_time = time.time()

            done = False
            truncated = False

            while not (done or truncated) and episode_length < max_episode_steps:
                # Get vehicle reference from environment
                vehicle = self.env.vehicle

                # Compute control command using baseline controller
                control = self.controller.compute_control(
                    vehicle=vehicle,
                    waypoints=self.waypoints,
                    dt=self.dt
                )

                # Step environment
                # NOTE: carla_env.py expects action as numpy array [steer, throttle/brake]
                # We need to convert VehicleControl to action format
                action = np.array([control.steer, control.throttle - control.brake])

                next_obs_dict, reward, done, truncated, info = self.env.step(action)

                # Display debug visualization if enabled
                if self.debug:
                    self._display_debug_frame(next_obs_dict, control, info, episode_length, episode_reward)

                # Collect metrics
                episode_reward += reward
                episode_length += 1

                # Get vehicle state for metrics
                debug_info = self.controller.get_debug_info(vehicle)
                current_speed = debug_info['speed_m_s']
                episode_speeds.append(current_speed)

                # ===============================================================
                # SAFETY METRIC: Time-to-Collision (TTC)
                # ===============================================================
                # Get distance to nearest obstacle from info dict
                distance_to_obstacle = info.get('distance_to_nearest_obstacle', float('inf'))
                if distance_to_obstacle < float('inf') and current_speed > 0.1:
                    ttc = distance_to_obstacle / current_speed
                    episode_ttc_values.append(ttc)

                # ===============================================================
                # COMFORT METRIC: Longitudinal Jerk (rate of change of acceleration)
                # ===============================================================
                # Calculate current acceleration from velocity change
                current_acceleration = (current_speed - prev_velocity) / self.dt if episode_length > 0 else 0.0

                # Calculate jerk from acceleration change
                if episode_length > 1:
                    jerk = (current_acceleration - prev_acceleration) / self.dt
                    episode_jerks.append(abs(jerk))  # Store magnitude

                # Update history
                prev_velocity = current_speed
                prev_acceleration = current_acceleration
                velocity_history.append(current_speed)
                acceleration_history.append(current_acceleration)

                # ===============================================================
                # COMFORT METRIC: Lateral Acceleration
                # ===============================================================
                # Get lateral acceleration from vehicle state
                # lateral_accel = v² * curvature = v² / radius
                # For small dt, we can approximate from steering angle and speed
                # lateral_accel ≈ v² * tan(steering_angle) / wheelbase
                wheelbase = 2.89  # meters (typical for CARLA vehicles)
                steering_angle = control.steer * 0.7  # Convert normalized [-1,1] to radians (max ~40°)
                if abs(steering_angle) > 0.01:  # Avoid division by small values
                    lateral_accel = abs((current_speed ** 2) * np.tan(steering_angle) / wheelbase)
                    episode_lateral_accels.append(lateral_accel)

                # ===============================================================
                # EFFICIENCY METRIC: Route Distance Traveled (for collisions/km)
                # ===============================================================
                current_location = np.array([debug_info['position']['x'],
                                            debug_info['position']['y']])
                if prev_vehicle_location is not None:
                    distance_step = np.linalg.norm(current_location - prev_vehicle_location)
                    route_distance_traveled += distance_step
                prev_vehicle_location = current_location

                # Store trajectory
                if self.save_trajectory:
                    trajectory.append({
                        'x': debug_info['position']['x'],
                        'y': debug_info['position']['y'],
                        'yaw': debug_info['rotation']['yaw'],
                        'speed': debug_info['speed_m_s'],
                        'steer': control.steer,
                        'throttle': control.throttle,
                        'brake': control.brake,
                        'acceleration': current_acceleration,
                        'jerk': abs(jerk) if episode_length > 1 else 0.0,
                        'lateral_accel': lateral_accel if abs(steering_angle) > 0.01 else 0.0
                    })

                obs_dict = next_obs_dict

            # Episode finished - LOG WHY IT ENDED
            termination_reason = info.get('termination_reason', 'unknown')
            success = info.get('success', 0)
            collision_count = info.get('collision_count', 0)
            lane_invasion_count = info.get('lane_invasion_count', 0)

            # DIAGNOSTIC: Log termination details
            print(f"\n[TERMINATION] Episode ended after {episode_length} steps:")
            print(f"  Reason: {termination_reason}")
            print(f"  Done: {done} (natural MDP termination)")
            print(f"  Truncated: {truncated} (time limit)")
            print(f"  Max Steps: {max_episode_steps}")
            print(f"  Success: {bool(success)}")
            print(f"  Collisions: {collision_count}")
            print(f"  Lane Invasions: {lane_invasion_count}")

            # Log position at termination
            final_pos = info.get('vehicle_state', {})
            if final_pos and 'location' in final_pos:
                x = final_pos['location'].get('x', None)
                y = final_pos['location'].get('y', None)
                if x is not None and y is not None:
                    print(f"  Final Position: ({x:.2f}, {y:.2f})")
                else:
                    print(f"  Final Position: N/A")
            else:
                print(f"  Final Position: N/A")

            # Log waypoint progress
            waypoint_idx = info.get('current_waypoint_idx', 0)
            progress_pct = info.get('progress_percentage', 0.0)
            distance_to_goal = info.get('distance_to_goal', 0.0)
            print(f"  Waypoint Index: {waypoint_idx}/{len(self.waypoints)-1}")
            print(f"  Progress: {progress_pct:.1f}%")
            print(f"  Distance to Goal: {distance_to_goal:.2f} m")

            # Calculate episode metrics
            avg_speed_ms = np.mean(episode_speeds) if episode_speeds else 0.0
            avg_speed_kmh = avg_speed_ms * 3.6

            # ===============================================================
            # Calculate per-episode aggregated metrics
            # ===============================================================

            # EFFICIENCY: Episode completion time
            episode_completion_time = time.time() - episode_start_time

            # SAFETY: Collisions per kilometer
            route_distance_km = route_distance_traveled / 1000.0 if route_distance_traveled > 0 else 0.0
            collisions_per_km = collision_count / route_distance_km if route_distance_km > 0 else 0.0

            # SAFETY: Average Time-to-Collision (TTC)
            avg_ttc = np.mean(episode_ttc_values) if episode_ttc_values else float('inf')

            # COMFORT: Average longitudinal jerk
            avg_jerk = np.mean(episode_jerks) if episode_jerks else 0.0

            # COMFORT: Average lateral acceleration
            avg_lateral_accel = np.mean(episode_lateral_accels) if episode_lateral_accels else 0.0

            # Store episode statistics (existing)
            self.episode_rewards.append(episode_reward)
            self.episode_successes.append(success)
            self.episode_collisions.append(collision_count)
            self.episode_lane_invasions.append(lane_invasion_count)
            self.episode_lengths.append(episode_length)
            self.episode_speeds.append(avg_speed_kmh)

            # Store NEW episode statistics (comfort, safety, efficiency)
            self.episode_jerks_list.append(avg_jerk)
            self.episode_lateral_accels_list.append(avg_lateral_accel)
            self.episode_ttc_list.append(avg_ttc)
            self.episode_collisions_per_km.append(collisions_per_km)
            self.episode_completion_times.append(episode_completion_time)
            self.episode_route_distances_km.append(route_distance_km)

            if self.save_trajectory:
                self.episode_trajectories.append(trajectory)

            # Print episode summary (enhanced)
            print(f"[EVAL] Episode {episode + 1} complete:")
            print(f"       Reward: {episode_reward:.2f}")
            print(f"       Success: {bool(success)}")
            print(f"       Collisions: {collision_count} ({collisions_per_km:.3f} per km)")
            print(f"       Lane Invasions: {lane_invasion_count}")
            print(f"       Length: {episode_length} steps ({episode_completion_time:.1f}s)")
            print(f"       Distance: {route_distance_km:.3f} km")
            print(f"       Avg Speed: {avg_speed_kmh:.2f} km/h")
            print(f"       Avg Jerk: {avg_jerk:.3f} m/s³")
            print(f"       Avg Lateral Accel: {avg_lateral_accel:.3f} m/s²")
            if avg_ttc < float('inf'):
                print(f"       Avg TTC: {avg_ttc:.2f} s")

        # Calculate aggregate metrics
        metrics = self._calculate_metrics()

        # Print summary
        self._print_summary(metrics)

        # Save results
        self._save_results(metrics)

        # Cleanup debug window if enabled
        if self.debug:
            cv2.destroyAllWindows()
            print(f"\n[DEBUG] Closed debug window")

        return metrics

    def _calculate_metrics(self) -> Dict:
        """
        Calculate aggregate evaluation metrics.

        Metrics align with paper requirements:
        - Safety: Success Rate, Collisions/km, TTC
        - Efficiency: Avg Speed, Route Completion Time
        - Comfort: Longitudinal Jerk, Lateral Acceleration

        Returns:
            Dictionary with mean and std for each metric
        """
        # Filter valid TTC values (exclude infinity)
        valid_ttc_values = [ttc for ttc in self.episode_ttc_list if ttc < float('inf')]

        metrics = {
            # ===============================================================
            # SAFETY METRICS
            # ===============================================================
            'success_rate': np.mean(self.episode_successes),
            'avg_collisions': np.mean(self.episode_collisions),
            'std_collisions': np.std(self.episode_collisions),
            'avg_collisions_per_km': np.mean(self.episode_collisions_per_km),
            'std_collisions_per_km': np.std(self.episode_collisions_per_km),
            'avg_lane_invasions': np.mean(self.episode_lane_invasions),
            'std_lane_invasions': np.std(self.episode_lane_invasions),
            'avg_ttc_seconds': np.mean(valid_ttc_values) if valid_ttc_values else float('inf'),
            'std_ttc_seconds': np.std(valid_ttc_values) if valid_ttc_values else 0.0,
            'min_ttc_seconds': np.min(valid_ttc_values) if valid_ttc_values else float('inf'),

            # ===============================================================
            # EFFICIENCY METRICS
            # ===============================================================
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_speed_kmh': np.mean(self.episode_speeds),
            'std_speed_kmh': np.std(self.episode_speeds),
            'avg_episode_length': np.mean(self.episode_lengths),
            'std_episode_length': np.std(self.episode_lengths),
            'avg_completion_time_s': np.mean(self.episode_completion_times),
            'std_completion_time_s': np.std(self.episode_completion_times),
            'avg_route_distance_km': np.mean(self.episode_route_distances_km),
            'std_route_distance_km': np.std(self.episode_route_distances_km),

            # ===============================================================
            # COMFORT METRICS
            # ===============================================================
            'avg_jerk_m_s3': np.mean(self.episode_jerks_list),
            'std_jerk_m_s3': np.std(self.episode_jerks_list),
            'max_jerk_m_s3': np.max(self.episode_jerks_list) if self.episode_jerks_list else 0.0,
            'avg_lateral_accel_m_s2': np.mean(self.episode_lateral_accels_list),
            'std_lateral_accel_m_s2': np.std(self.episode_lateral_accels_list),
            'max_lateral_accel_m_s2': np.max(self.episode_lateral_accels_list) if self.episode_lateral_accels_list else 0.0,

            # Raw data for further analysis
            'num_episodes': self.num_episodes,
            'scenario': self.scenario,
            'seed': self.seed
        }

        return metrics

    def _print_summary(self, metrics: Dict):
        """Print evaluation summary with all paper-ready metrics."""
        print("\n" + "="*70)
        print("BASELINE EVALUATION SUMMARY - PID + PURE PURSUIT")
        print("="*70)

        print(f"\n{'='*70}")
        print("SAFETY METRICS")
        print(f"{'='*70}")
        print(f"  Success Rate:          {metrics['success_rate']*100:.1f}%")
        print(f"  Avg Collisions:        {metrics['avg_collisions']:.2f} ± {metrics['std_collisions']:.2f}")
        print(f"  Avg Collisions/km:     {metrics['avg_collisions_per_km']:.3f} ± {metrics['std_collisions_per_km']:.3f}")
        print(f"  Avg Lane Invasions:    {metrics['avg_lane_invasions']:.2f} ± {metrics['std_lane_invasions']:.2f}")
        if metrics['avg_ttc_seconds'] < float('inf'):
            print(f"  Avg TTC:               {metrics['avg_ttc_seconds']:.2f} ± {metrics['std_ttc_seconds']:.2f} s")
            print(f"  Min TTC:               {metrics['min_ttc_seconds']:.2f} s")
        else:
            print(f"  Avg TTC:               N/A (no obstacles detected)")

        print(f"\n{'='*70}")
        print("EFFICIENCY METRICS")
        print(f"{'='*70}")
        print(f"  Mean Reward:           {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Avg Speed:             {metrics['avg_speed_kmh']:.2f} ± {metrics['std_speed_kmh']:.2f} km/h")
        print(f"  Avg Completion Time:   {metrics['avg_completion_time_s']:.1f} ± {metrics['std_completion_time_s']:.1f} s")
        print(f"  Avg Route Distance:    {metrics['avg_route_distance_km']:.3f} ± {metrics['std_route_distance_km']:.3f} km")
        print(f"  Avg Episode Length:    {metrics['avg_episode_length']:.1f} ± {metrics['std_episode_length']:.1f} steps")

        print(f"\n{'='*70}")
        print("COMFORT METRICS")
        print(f"{'='*70}")
        print(f"  Avg Jerk:              {metrics['avg_jerk_m_s3']:.3f} ± {metrics['std_jerk_m_s3']:.3f} m/s³")
        print(f"  Max Jerk:              {metrics['max_jerk_m_s3']:.3f} m/s³")
        print(f"  Avg Lateral Accel:     {metrics['avg_lateral_accel_m_s2']:.3f} ± {metrics['std_lateral_accel_m_s2']:.3f} m/s²")
        print(f"  Max Lateral Accel:     {metrics['max_lateral_accel_m_s2']:.3f} m/s²")

        print(f"\n{'='*70}")
        print("CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Scenario:              {self.scenario} (NPC density)")
        print(f"  Episodes:              {metrics['num_episodes']}")
        print(f"  Seed:                  {metrics['seed']}")
        print("="*70 + "\n")

    def _save_results(self, metrics: Dict):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scenario_name = f"scenario_{self.scenario}"
        filename = f"baseline_{scenario_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Prepare results dictionary
        results = {
            'config': {
                'scenario': self.scenario,
                'seed': self.seed,
                'num_episodes': self.num_episodes,
                'controller_params': self.baseline_config['baseline_controller']
            },
            'metrics': metrics,
            'episodes': {
                'rewards': [float(r) for r in self.episode_rewards],
                'successes': [int(s) for s in self.episode_successes],
                'collisions': [int(c) for c in self.episode_collisions],
                'lane_invasions': [int(l) for l in self.episode_lane_invasions],
                'lengths': [int(l) for l in self.episode_lengths],
                'speeds_kmh': [float(s) for s in self.episode_speeds]
            }
        }

        # Save trajectories if enabled
        if self.save_trajectory:
            trajectory_dir = self.output_dir / "trajectories"
            trajectory_dir.mkdir(exist_ok=True)
            trajectory_file = trajectory_dir / f"trajectories_{scenario_name}_{timestamp}.json"

            with open(trajectory_file, 'w') as f:
                json.dump(self.episode_trajectories, f, indent=2)

            print(f"[SAVE] Trajectories saved to {trajectory_file}")

        # Save main results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[SAVE] Results saved to {filepath}")

        # Generate LaTeX table
        latex_table = self._generate_latex_table(metrics)
        latex_file = self.output_dir / f"latex_table_{scenario_name}_{timestamp}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        print(f"[SAVE] LaTeX table saved to {latex_file}")

    def _generate_latex_table(self, metrics: Dict) -> str:
        """
        Generate LaTeX table for paper with baseline performance metrics.

        Aligns with paper requirements (ourPaper.tex):
        - Safety: Success Rate, Collisions/km, TTC
        - Efficiency: Avg Speed, Completion Time
        - Comfort: Jerk, Lateral Acceleration

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            LaTeX table code as string
        """
        latex = "% Baseline Controller Performance - PID + Pure Pursuit\n"
        latex += "% Generated by evaluate_baseline.py\n"
        latex += f"% Scenario: {self.scenario}, Episodes: {self.num_episodes}, Seed: {self.seed}\n\n"

        latex += "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Baseline Controller Performance (PID + Pure Pursuit)}\n"
        latex += "\\label{tab:baseline_performance}\n"
        latex += "\\begin{tabular}{|l|c|c|}\n"
        latex += "\\hline\n"
        latex += "\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std Dev} \\\\\n"
        latex += "\\hline\n"
        latex += "\\hline\n"

        # Safety Metrics Section
        latex += "\\multicolumn{3}{|l|}{\\textbf{Safety Metrics}} \\\\\n"
        latex += "\\hline\n"
        latex += f"Success Rate (\\%) & {metrics['success_rate']*100:.1f} & -- \\\\\n"
        latex += f"Avg Collisions & {metrics['avg_collisions']:.2f} & {metrics['std_collisions']:.2f} \\\\\n"
        latex += f"Collisions/km & {metrics['avg_collisions_per_km']:.3f} & {metrics['std_collisions_per_km']:.3f} \\\\\n"
        latex += f"Lane Invasions & {metrics['avg_lane_invasions']:.2f} & {metrics['std_lane_invasions']:.2f} \\\\\n"

        if metrics['avg_ttc_seconds'] < float('inf'):
            latex += f"Avg TTC (s) & {metrics['avg_ttc_seconds']:.2f} & {metrics['std_ttc_seconds']:.2f} \\\\\n"
            latex += f"Min TTC (s) & {metrics['min_ttc_seconds']:.2f} & -- \\\\\n"
        else:
            latex += f"Avg TTC (s) & N/A & -- \\\\\n"

        latex += "\\hline\n"

        # Efficiency Metrics Section
        latex += "\\multicolumn{3}{|l|}{\\textbf{Efficiency Metrics}} \\\\\n"
        latex += "\\hline\n"
        latex += f"Avg Speed (km/h) & {metrics['avg_speed_kmh']:.2f} & {metrics['std_speed_kmh']:.2f} \\\\\n"
        latex += f"Completion Time (s) & {metrics['avg_completion_time_s']:.1f} & {metrics['std_completion_time_s']:.1f} \\\\\n"
        latex += f"Route Distance (km) & {metrics['avg_route_distance_km']:.3f} & {metrics['std_route_distance_km']:.3f} \\\\\n"
        latex += f"Episode Length (steps) & {metrics['avg_episode_length']:.1f} & {metrics['std_episode_length']:.1f} \\\\\n"
        latex += f"Mean Reward & {metrics['mean_reward']:.2f} & {metrics['std_reward']:.2f} \\\\\n"
        latex += "\\hline\n"

        # Comfort Metrics Section
        latex += "\\multicolumn{3}{|l|}{\\textbf{Comfort Metrics}} \\\\\n"
        latex += "\\hline\n"
        latex += f"Avg Jerk (m/s³) & {metrics['avg_jerk_m_s3']:.3f} & {metrics['std_jerk_m_s3']:.3f} \\\\\n"
        latex += f"Max Jerk (m/s³) & {metrics['max_jerk_m_s3']:.3f} & -- \\\\\n"
        latex += f"Avg Lateral Accel (m/s²) & {metrics['avg_lateral_accel_m_s2']:.3f} & {metrics['std_lateral_accel_m_s2']:.3f} \\\\\n"
        latex += f"Max Lateral Accel (m/s²) & {metrics['max_lateral_accel_m_s2']:.3f} & -- \\\\\n"
        latex += "\\hline\n"

        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    def close(self):
        """Clean up environment."""
        print("\n[CLEANUP] Closing environment...")
        self.env.close()
        print("[CLEANUP] Done!")


def main():
    """Main entry point for baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate PID + Pure Pursuit baseline controller in CARLA"
    )

    parser.add_argument(
        '--scenario',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Traffic density scenario: 0=20 NPCs, 1=50 NPCs, 2=100 NPCs'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--num-episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes'
    )

    parser.add_argument(
        '--baseline-config',
        type=str,
        default='config/baseline_config.yaml',
        help='Path to baseline controller config'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/baseline_evaluation',
        help='Output directory for results'
    )

    parser.add_argument(
        '--no-trajectory',
        action='store_true',
        help='Disable trajectory saving'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--use-ros-bridge',
        action='store_true',
        help='Use ROS 2 Bridge for vehicle control (Phase 5 integration)'
    )

    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip automatic trajectory analysis after evaluation'
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = BaselineEvaluationPipeline(
        scenario=args.scenario,
        seed=args.seed,
        num_episodes=args.num_episodes,
        baseline_config_path=args.baseline_config,
        output_dir=args.output_dir,
        save_trajectory=not args.no_trajectory,
        debug=args.debug,
        use_ros_bridge=args.use_ros_bridge
    )

    trajectory_file = None  # Will store the trajectory file path for analysis

    try:
        # Run evaluation
        metrics = pipeline.evaluate()

        # Get trajectory file path for analysis
        if not args.no_trajectory:
            # Find the most recent trajectory file
            traj_dir = Path(args.output_dir) / 'trajectories'
            if traj_dir.exists():
                traj_files = sorted(traj_dir.glob('trajectories_*.json'))
                if traj_files:
                    trajectory_file = traj_files[-1]  # Most recent file

        # Print final summary
        print("\n[SUCCESS] Baseline evaluation complete!")
        print(f"[SUCCESS] Results saved to {args.output_dir}")

        # Automatically run trajectory analysis
        if not args.skip_analysis and not args.no_trajectory:
            print("\n" + "="*80)
            print("AUTOMATIC TRAJECTORY ANALYSIS")
            print("="*80)

            # Import and run analysis script
            import subprocess

            # Let the analysis script auto-detect the trajectory file
            cmd = ['python3', 'scripts/analyze_phase3_trajectories.py']

            print(f"\nRunning: {' '.join(cmd)}\n")
            print("   (auto-detecting latest trajectory file...)\n")

            result = subprocess.run(cmd, cwd=project_root)

            if result.returncode == 0:
                print("\nTrajectory analysis completed successfully!")
                print("Analysis results saved")
                print("   Plots generated:")
                print("   - trajectory_map.png (NEW: top-down 2D view)")
                print("   - lateral_deviation.png")
                print("   - heading_error.png")
                print("   - speed_profile.png")
                print("   - control_commands.png")
                print("   - PHASE3_ANALYSIS_REPORT.md")
            else:
                print(f"\nTrajectory analysis failed with exit code {result.returncode}")
                print("You can run analysis manually with:")
                print("  python scripts/analyze_phase3_trajectories.py")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Evaluation interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        pipeline.close()


if __name__ == "__main__":
    main()

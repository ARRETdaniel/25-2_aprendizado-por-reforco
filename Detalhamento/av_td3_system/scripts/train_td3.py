#!/usr/bin/env python3
"""
TD3 Agent Training Script for Autonomous Vehicle Navigation in CARLA

This script orchestrates the complete training pipeline:
1. Initialize CARLA environment (Gym interface)
2. Initialize TD3 agent with networks and replay buffer
3. Training loop with:
   - Exploration phase (random actions for buffer population)
   - Policy learning phase (select_action + train steps)
   - Periodic evaluation
   - Checkpoint saving
4. Logging and monitoring (TensorBoard)

Configuration:
- Load from config/carla_config.yaml, config/td3_config.yaml
- Command-line arguments for scenario selection, seed, etc.
- TensorBoard logging to logs/ directory

Author: Daniel Terra
Date: 2024
"""

import sys
import os
sys.path.insert(0, '/workspace/av_td3_system')

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.td3_agent import TD3Agent
from src.environment.carla_env import CARLANavigationEnv


class TD3TrainingPipeline:
    """
    Main training pipeline for TD3 agent in CARLA environment.

    Responsibilities:
    - Environment management and episode resets
    - Training loop orchestration
    - Agent state updates (select_action, train)
    - Evaluation and logging
    - Checkpoint management
    """

    def __init__(
        self,
        scenario: int = 0,
        seed: int = 42,
        max_timesteps: int = int(1e6),
        eval_freq: int = 5000,
        checkpoint_freq: int = 10000,
        num_eval_episodes: int = 10,
        carla_config_path: str = "config/carla_config.yaml",
        agent_config_path: str = "config/td3_config.yaml",
        log_dir: str = "data/logs",
        checkpoint_dir: str = "data/checkpoints",
        debug: bool = False
    ):
        """
        Initialize training pipeline.

        Args:
            scenario: Traffic density scenario (0=20, 1=50, 2=100 NPCs)
            seed: Random seed for reproducibility
            max_timesteps: Maximum training timesteps (default: 1M)
            eval_freq: Evaluation frequency (steps)
            checkpoint_freq: Checkpoint saving frequency (steps)
            num_eval_episodes: Number of episodes per evaluation
            carla_config_path: Path to CARLA config
            agent_config_path: Path to TD3 config
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for checkpoints
            debug: Enable visual feedback with OpenCV (for short runs)
        """
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.scenario = scenario
        self.seed = seed
        self.max_timesteps = max_timesteps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.num_eval_episodes = num_eval_episodes
        self.debug = debug

        # Create directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load configurations
        print("\n" + "="*70)
        print("TD3 TRAINING PIPELINE - AUTONOMOUS VEHICLE NAVIGATION")
        print("="*70)
        print(f"\n[CONFIG] Loading configurations...")

        with open(carla_config_path, 'r') as f:
            self.carla_config = yaml.safe_load(f)

        with open(agent_config_path, 'r') as f:
            self.agent_config = yaml.safe_load(f)

        print(f"[CONFIG] CARLA config: {carla_config_path}")
        print(f"[CONFIG] Agent config: {agent_config_path}")
        print(f"[CONFIG] Scenario: {scenario} (0=20, 1=50, 2=100 NPCs)")

        # Update NPC density based on scenario
        npc_densities = [20, 50, 100]
        if scenario < len(npc_densities):
            if 'traffic' in self.carla_config:
                self.carla_config['traffic']['num_vehicles'] = npc_densities[scenario]
                print(f"[CONFIG] NPC count set to: {npc_densities[scenario]}")
            else:
                print(f"[WARNING] 'traffic' section not found in config, using default NPC count")

        # Initialize environment
        print(f"\n[ENVIRONMENT] Initializing CARLA environment...")
        self.env = CARLANavigationEnv(
            carla_config_path,
            agent_config_path,
            agent_config_path  # Use same config for training_config_path
        )
        print(f"[ENVIRONMENT] State space: {self.env.observation_space}")
        print(f"[ENVIRONMENT] Action space: {self.env.action_space}")

        # Initialize agent
        print(f"\n[AGENT] Initializing TD3 agent...")
        # Use CPU for agent to save GPU memory for CARLA (6GB RTX 2060 constraint)
        agent_device = 'cpu' if self.debug else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[AGENT] Using device: {agent_device}")
        self.agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config=self.agent_config,
            device=agent_device
        )

        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scenario_name = f"scenario_{scenario}_npcs_{npc_densities[scenario]}"
        self.log_name = f"TD3_{scenario_name}_{timestamp}"
        self.log_path = self.log_dir / self.log_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_path))

        print(f"[LOGGING] TensorBoard logs: {self.log_path}")

        # Debug mode setup
        if self.debug:
            print(f"\n[DEBUG] Visual feedback enabled (OpenCV display)")
            print(f"[DEBUG] Press 'q' to quit, 'p' to pause/unpause")
            self.window_name = "TD3 Training - Debug View"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 600)
            self.paused = False

        # Training state
        self.episode_num = 0
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_collision_count = 0
        self.episode_steps_since_collision = 0

        # Statistics
        self.training_rewards = []
        self.eval_rewards = []
        self.eval_success_rates = []
        self.eval_collisions = []

        print(f"\n[INIT] Training pipeline ready!")
        print(f"[INIT] Max timesteps: {max_timesteps:,}")
        print(f"[INIT] Seed: {seed}")
        print("="*70 + "\n")

    def flatten_dict_obs(self, obs_dict):
        """
        Flatten Dict observation to 1D array for TD3 agent.

        Args:
            obs_dict: Dictionary with 'image' (4, 84, 84) and 'vector' (23,) keys

        Returns:
            np.ndarray: Flattened state vector of shape (535,)
                        - First 512 elements: Image features (averaged across frames)
                        - Last 23 elements: Vector state (velocity, waypoints, etc.)
        """
        # Extract image and flatten across frames
        image = obs_dict['image']  # Shape: (4, 84, 84)
        image_flat = image.reshape(4, -1).mean(axis=0)  # Average across frames: (7056,)
        image_features = image_flat[:512]  # Take first 512 features

        # Extract vector state
        vector = obs_dict['vector']  # Shape: (23,)

        # Concatenate to final state
        flat_state = np.concatenate([image_features, vector]).astype(np.float32)

        return flat_state  # Shape: (535,)

    def _visualize_debug(self, obs_dict, action, reward, info, t):
        """
        Display debug visualization using OpenCV.

        Shows:
        - Front camera view
        - Action values (steering, throttle/brake)
        - Reward breakdown
        - Vehicle state
        - Episode info

        Args:
            obs_dict: Current observation dictionary with 'image' and 'vector' keys
            action: Action taken [steering, throttle/brake]
            reward: Reward received
            info: Environment info dict
            t: Current timestep
        """
        try:
            # Extract camera image from observation dict
            # obs_dict['image'] has shape (4, 84, 84) - 4 stacked grayscale frames
            if 'image' not in obs_dict:
                return

            # Get the latest frame from the stack
            latest_frame = obs_dict['image'][-1]  # Shape: (84, 84)

            # Convert from [0, 1] float to [0, 255] uint8
            frame_uint8 = (latest_frame * 255).astype(np.uint8)

            # Resize to larger display size
            frame_resized = cv2.resize(frame_uint8, (800, 600), interpolation=cv2.INTER_LINEAR)

            # Convert grayscale to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

            # Add info overlay
            vehicle_state = info.get('vehicle_state', {})
            reward_breakdown = info.get('reward_breakdown', {})

            # Create info panel
            info_panel = np.zeros((600, 400, 3), dtype=np.uint8)

            # Title
            cv2.putText(info_panel, "TD3 TRAINING - DEBUG", (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

            # Timestep info
            y_offset = 70
            cv2.putText(info_panel, f"Timestep: {t}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"Episode: {self.episode_num}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"Episode Step: {self.episode_timesteps}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Action
            y_offset += 40
            cv2.putText(info_panel, "ACTION:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Steering: {action[0]:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Throttle/Brake: {action[1]:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Vehicle state
            y_offset += 40
            cv2.putText(info_panel, "VEHICLE STATE:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            speed_kmh = vehicle_state.get('velocity', 0) * 3.6
            cv2.putText(info_panel, f"  Speed: {speed_kmh:.1f} km/h", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Lat Dev: {vehicle_state.get('lateral_deviation', 0):.2f} m",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Head Err: {vehicle_state.get('heading_error', 0):.2f} rad",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Reward
            y_offset += 40
            cv2.putText(info_panel, "REWARD:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            reward_color = (0, 255, 0) if reward > 0 else (0, 0, 255)
            cv2.putText(info_panel, f"  Total: {reward:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, reward_color, 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Episode: {self.episode_reward:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Reward breakdown
            y_offset += 30
            cv2.putText(info_panel, "Breakdown:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            for component, (weight, value, weighted) in reward_breakdown.items():
                y_offset += 20
                text = f"  {component[:8]}: {weighted:+.2f}"
                cv2.putText(info_panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Collision info
            y_offset += 30
            collision_color = (0, 0, 255) if self.episode_collision_count > 0 else (0, 255, 0)
            cv2.putText(info_panel, f"Collisions: {self.episode_collision_count}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, collision_color, 1)

            # Instructions
            y_offset = 580
            cv2.putText(info_panel, "q:quit  p:pause", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Combine camera and info panel
            combined_frame = np.hstack([display_frame, info_panel])

            # Display
            cv2.imshow(self.window_name, combined_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n[DEBUG] User requested quit (q pressed)")
                cv2.destroyAllWindows()
                self.env.close()
                import sys
                sys.exit(0)
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"[DEBUG] Paused: {self.paused}")

            # Pause handling
            while self.paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    self.paused = False
                    print("[DEBUG] Resumed")
                elif key == ord('q'):
                    print("\n[DEBUG] User requested quit (q pressed)")
                    cv2.destroyAllWindows()
                    self.env.close()
                    import sys
                    sys.exit(0)

        except Exception as e:
            print(f"[DEBUG] Visualization error: {e}")
            # Continue training even if visualization fails

    def train(self):
        """
        Main training loop.

        Implements the complete TD3 training workflow:
        1. Initialize state from environment reset
        2. For each timestep until convergence:
           - Exploration phase (random actions)
           - Policy learning phase (agent.select_action + agent.train)
           - Episode management and metrics tracking
           - Periodic evaluation
           - Checkpoint saving
        """
        print(f"[TRAINING] Starting training loop...")

        # Get initial observation (Dict) and flatten for agent
        obs_dict = self.env.reset()
        state = self.flatten_dict_obs(obs_dict)
        done = False

        start_timesteps = self.agent_config.get('algorithm', {}).get('learning_starts', 25000)
        batch_size = self.agent_config.get('algorithm', {}).get('batch_size', 256)

        for t in range(1, int(self.max_timesteps) + 1):
            self.episode_timesteps += 1

            # Select action based on training phase
            if t < start_timesteps:
                # Exploration phase: random actions to populate replay buffer
                action = self.env.action_space.sample()
            else:
                # Learning phase: use policy with exploration noise
                action = self.agent.select_action(
                    state,
                    noise=self.agent.expl_noise
                )

            # Step environment (get Dict observation)
            next_obs_dict, reward, done, truncated, info = self.env.step(action)

            # Flatten next observation for agent
            next_state = self.flatten_dict_obs(next_obs_dict)

            # Debug visualization (use original Dict observation)
            if self.debug:
                self._visualize_debug(obs_dict, action, reward, info, t)

                # 🔍 DEBUG: Print detailed step info to terminal every 10 steps
                if t % 10 == 0:
                    vehicle_state = info.get('vehicle_state', {})
                    reward_breakdown = info.get('reward_breakdown', {})

                    # Extract observation data for debugging
                    vector_obs = next_obs_dict.get('vector', np.array([]))
                    image_obs = next_obs_dict.get('image', np.array([]))

                    # Parse vector observation (velocity, lat_dev, heading_error, waypoints[20])
                    velocity = vector_obs[0] if len(vector_obs) > 0 else 0.0
                    lat_dev = vector_obs[1] if len(vector_obs) > 1 else 0.0
                    heading_err = vector_obs[2] if len(vector_obs) > 2 else 0.0

                    # Waypoints are elements [3:23] (10 waypoints × 2 coords = 20 values)
                    waypoints_flat = vector_obs[3:23] if len(vector_obs) >= 23 else []

                    # Image statistics (4 stacked frames, 84×84)
                    if len(image_obs.shape) == 3:  # (4, 84, 84)
                        img_mean = np.mean(image_obs)
                        img_std = np.std(image_obs)
                        img_min = np.min(image_obs)
                        img_max = np.max(image_obs)
                    else:
                        img_mean = img_std = img_min = img_max = 0.0

                    # Reward breakdown (format: reward_breakdown is already the "breakdown" dict)
                    # Each component is a tuple: (weight, raw_value, weighted_value)
                    eff_tuple = reward_breakdown.get('efficiency', (0, 0, 0))
                    lane_tuple = reward_breakdown.get('lane_keeping', (0, 0, 0))
                    comfort_tuple = reward_breakdown.get('comfort', (0, 0, 0))
                    safety_tuple = reward_breakdown.get('safety', (0, 0, 0))

                    # Extract weighted values (index 2)
                    eff_reward = eff_tuple[2] if isinstance(eff_tuple, tuple) else 0.0
                    lane_reward = lane_tuple[2] if isinstance(lane_tuple, tuple) else 0.0
                    comfort_reward = comfort_tuple[2] if isinstance(comfort_tuple, tuple) else 0.0
                    safety_reward = safety_tuple[2] if isinstance(safety_tuple, tuple) else 0.0

                    # Print main debug line
                    print(
                        f"\n🔍 [DEBUG Step {t:4d}] "
                        f"Act=[steer:{action[0]:+.3f}, thr/brk:{action[1]:+.3f}] | "
                        f"Rew={reward:+7.2f} | "
                        f"Speed={vehicle_state.get('velocity', 0)*3.6:5.1f} km/h | "
                        f"LatDev={vehicle_state.get('lateral_deviation', 0):+.2f}m | "
                        f"Collisions={self.episode_collision_count}"
                    )

                    # Print reward breakdown
                    print(
                        f"   💰 Reward: Efficiency={eff_reward:+.2f} | "
                        f"Lane={lane_reward:+.2f} | "
                        f"Comfort={comfort_reward:+.2f} | "
                        f"Safety={safety_reward:+.2f}"
                    )

                    # Print first 3 waypoints (most relevant)
                    if len(waypoints_flat) >= 6:
                        wp1_x, wp1_y = waypoints_flat[0], waypoints_flat[1]
                        wp2_x, wp2_y = waypoints_flat[2], waypoints_flat[3]
                        wp3_x, wp3_y = waypoints_flat[4], waypoints_flat[5]

                        # Calculate distances
                        dist1 = np.sqrt(wp1_x**2 + wp1_y**2)
                        dist2 = np.sqrt(wp2_x**2 + wp2_y**2)
                        dist3 = np.sqrt(wp3_x**2 + wp3_y**2)

                        print(
                            f"   📍 Waypoints (vehicle frame): "
                            f"WP1=[{wp1_x:+6.1f}, {wp1_y:+6.1f}]m (d={dist1:5.1f}m) | "
                            f"WP2=[{wp2_x:+6.1f}, {wp2_y:+6.1f}]m (d={dist2:5.1f}m) | "
                            f"WP3=[{wp3_x:+6.1f}, {wp3_y:+6.1f}]m (d={dist3:5.1f}m)"
                        )

                    # Print image statistics
                    print(
                        f"   🖼️  Image: shape={image_obs.shape} | "
                        f"mean={img_mean:.3f} | std={img_std:.3f} | "
                        f"range=[{img_min:.3f}, {img_max:.3f}]"
                    )

                    # Print state vector info
                    print(
                        f"   📊 State: velocity={velocity:.2f} m/s | "
                        f"lat_dev={lat_dev:+.3f}m | "
                        f"heading_err={heading_err:+.3f} rad ({np.degrees(heading_err):+.1f}°) | "
                        f"vector_dim={len(vector_obs)}"
                    )

            # Track episode metrics
            self.episode_reward += reward
            self.episode_collision_count += info.get('collision_count', 0)
            self.episode_steps_since_collision = info.get('steps_since_collision', 0)

            # Store transition in replay buffer (use flat states)
            done_bool = float(done or truncated) if self.episode_timesteps < 300 else True  # 300s timeout
            self.agent.replay_buffer.add(
                state,
                action,
                next_state,
                reward,
                done_bool
            )

            # Update state for next iteration (both representations)
            state = next_state
            obs_dict = next_obs_dict

            # Train agent (only after exploration phase)
            if t > start_timesteps:
                metrics = self.agent.train(batch_size=batch_size)

                # Log training metrics every 100 steps
                if t % 100 == 0:
                    self.writer.add_scalar('train/critic_loss', metrics['critic_loss'], t)
                    self.writer.add_scalar('train/q1_value', metrics['q1_value'], t)
                    self.writer.add_scalar('train/q2_value', metrics['q2_value'], t)

                    if 'actor_loss' in metrics:  # Actor updated only on delayed steps
                        self.writer.add_scalar('train/actor_loss', metrics['actor_loss'], t)

            # Episode termination
            if done or truncated:
                # Log episode metrics
                self.training_rewards.append(self.episode_reward)
                self.writer.add_scalar('train/episode_reward', self.episode_reward, self.episode_num)
                self.writer.add_scalar('train/episode_length', self.episode_timesteps, self.episode_num)
                self.writer.add_scalar(
                    'train/collisions_per_episode',
                    self.episode_collision_count,
                    self.episode_num
                )

                # Console logging every 10 episodes
                if self.episode_num % 10 == 0:
                    avg_reward = np.mean(self.training_rewards[-10:]) if len(self.training_rewards) >= 10 else np.mean(self.training_rewards)
                    print(
                        f"[TRAIN] Episode {self.episode_num:4d} | "
                        f"Timestep {t:7d} | "
                        f"Reward {self.episode_reward:8.2f} | "
                        f"Avg Reward (10ep) {avg_reward:8.2f} | "
                        f"Collisions {self.episode_collision_count:2d}"
                    )

                # Reset episode (get Dict and flatten)
                obs_dict = self.env.reset()
                state = self.flatten_dict_obs(obs_dict)
                self.episode_num += 1
                self.episode_reward = 0
                self.episode_timesteps = 0
                self.episode_collision_count = 0

            # Periodic evaluation
            if t % self.eval_freq == 0:
                print(f"\n[EVAL] Evaluation at timestep {t:,}...")
                eval_metrics = self.evaluate()

                self.writer.add_scalar('eval/mean_reward', eval_metrics['mean_reward'], t)
                self.writer.add_scalar('eval/success_rate', eval_metrics['success_rate'], t)
                self.writer.add_scalar('eval/avg_collisions', eval_metrics['avg_collisions'], t)
                self.writer.add_scalar('eval/avg_episode_length', eval_metrics['avg_episode_length'], t)

                self.eval_rewards.append(eval_metrics['mean_reward'])
                self.eval_success_rates.append(eval_metrics['success_rate'])
                self.eval_collisions.append(eval_metrics['avg_collisions'])

                print(
                    f"[EVAL] Mean Reward: {eval_metrics['mean_reward']:.2f} | "
                    f"Success Rate: {eval_metrics['success_rate']*100:.1f}% | "
                    f"Avg Collisions: {eval_metrics['avg_collisions']:.2f} | "
                    f"Avg Length: {eval_metrics['avg_episode_length']:.0f}"
                )
                print()

            # Checkpoint saving
            if t % self.checkpoint_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"td3_scenario_{self.scenario}_step_{t}.pth"
                self.agent.save_checkpoint(str(checkpoint_path))
                print(f"[CHECKPOINT] Saved to {checkpoint_path}")

        print(f"\n[TRAINING] Training complete!")
        self.close()

    def evaluate(self) -> dict:
        """
        Evaluate agent on multiple episodes without exploration noise.

        Returns:
            Dictionary with evaluation metrics:
            - mean_reward: Average episode reward
            - std_reward: Std dev of episode rewards
            - success_rate: Fraction of successful episodes
            - avg_collisions: Average collisions per episode
            - avg_episode_length: Average episode length
        """
        eval_rewards = []
        eval_successes = []
        eval_collisions = []
        eval_lengths = []

        for episode in range(self.num_eval_episodes):
            obs_dict = self.env.reset()  # Use main env, not eval_env
            state = self.flatten_dict_obs(obs_dict)  # Flatten Dict → flat array
            episode_reward = 0
            episode_length = 0
            done = False
            max_eval_steps = 1000  # Safety limit: max 1000 steps per eval episode

            while not done and episode_length < max_eval_steps:
                # Deterministic action (no noise)
                action = self.agent.select_action(state, noise=0.0)
                next_obs_dict, reward, done, truncated, info = self.env.step(action)  # Use main env
                next_state = self.flatten_dict_obs(next_obs_dict)  # Flatten next obs

                episode_reward += reward
                episode_length += 1
                state = next_state

                if truncated:
                    done = True

            # Log if episode hit the safety limit
            if episode_length >= max_eval_steps:
                print(f"[EVAL] Warning: Episode {episode+1} reached max eval steps ({max_eval_steps})")

            eval_rewards.append(episode_reward)
            eval_successes.append(info.get('success', 0))
            eval_collisions.append(info.get('collision_count', 0))
            eval_lengths.append(episode_length)

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'success_rate': np.mean(eval_successes),
            'avg_collisions': np.mean(eval_collisions),
            'avg_episode_length': np.mean(eval_lengths)
        }

    def save_final_results(self):
        """Save final training results to JSON."""
        results = {
            'scenario': self.scenario,
            'seed': self.seed,
            'total_timesteps': self.max_timesteps,
            'total_episodes': self.episode_num,
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': [float(x) for x in self.eval_success_rates],
            'eval_collisions': [float(x) for x in self.eval_collisions],
            'final_eval_mean_reward': float(np.mean(self.eval_rewards[-5:])) if len(self.eval_rewards) > 0 else 0,
            'final_eval_success_rate': float(np.mean(self.eval_success_rates[-5:])) if len(self.eval_success_rates) > 0 else 0
        }

        results_path = self.log_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[RESULTS] Saved to {results_path}")

    def close(self):
        """Clean up resources."""
        if self.debug:
            cv2.destroyAllWindows()
            print(f"[DEBUG] OpenCV windows closed")
        self.save_final_results()
        self.env.close()
        self.writer.close()
        print(f"[CLEANUP] Environment closed, logging finalized")


def main():
    """
    Main entry point for TD3 training script.

    Command-line arguments:
    - --scenario: Traffic scenario (0=20, 1=50, 2=100 NPCs)
    - --seed: Random seed
    - --max-timesteps: Maximum training timesteps
    - --eval-freq: Evaluation frequency
    - --checkpoint-freq: Checkpoint saving frequency
    """
    parser = argparse.ArgumentParser(
        description="Train TD3 agent for autonomous vehicle navigation in CARLA"
    )
    parser.add_argument(
        '--scenario',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Traffic scenario (0=20, 1=50, 2=100 NPCs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-timesteps',
        type=int,
        default=int(1e6),
        help='Maximum training timesteps'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5000,
        help='Evaluation frequency (timesteps)'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10000,
        help='Checkpoint saving frequency (timesteps)'
    )
    parser.add_argument(
        '--num-eval-episodes',
        type=int,
        default=10,
        help='Number of episodes per evaluation'
    )
    parser.add_argument(
        '--carla-config',
        type=str,
        default='config/carla_config.yaml',
        help='Path to CARLA config file'
    )
    parser.add_argument(
        '--agent-config',
        type=str,
        default='config/td3_config.yaml',
        help='Path to TD3 agent config file'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='data/logs',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='data/checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable visual debug mode with OpenCV display (recommended for short runs, e.g., 1000 steps)'
    )

    args = parser.parse_args()

    # Initialize and run training
    trainer = TD3TrainingPipeline(
        scenario=args.scenario,
        seed=args.seed,
        max_timesteps=args.max_timesteps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        num_eval_episodes=args.num_eval_episodes,
        carla_config_path=args.carla_config,
        agent_config_path=args.agent_config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        debug=args.debug
    )

    trainer.train()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Autonomous Vehicle Agents in CARLA

This script evaluates trained agents (TD3, DDPG, IDM+MOBIL) on multiple scenarios
and collects comprehensive metrics for paper analysis.

Metrics Collected:
1. Safety:
   - Success rate (%)
   - Average collisions per episode
   - Average collisions per kilometer
   - TTC (Time-To-Collision) violations count
   - Minimum TTC recorded

2. Efficiency:
   - Average speed (km/h)
   - Route completion time (seconds)
   - Distance traveled (meters)

3. Comfort:
   - Average longitudinal jerk (m/s³)
   - Average lateral acceleration (m/s²)
   - Maximum jerk recorded

Output Formats:
- CSV: Per-episode metrics for statistical analysis
- JSON: Aggregated summary statistics
- Console: Real-time progress and final results

Author: Daniel Terra
Date: 2024
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from src.agents.ddpg_agent import DDPGAgent
from src.agents.td3_agent import TD3Agent
from src.baselines.idm_mobil import IDMMOBILBaseline
from src.environment.carla_env import CARLANavigationEnv


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for autonomous driving agents.

    Manages:
    - Agent loading (TD3, DDPG, IDM+MOBIL)
    - Multi-scenario evaluation
    - Comprehensive metrics collection
    - Results export (CSV, JSON)
    """

    def __init__(
        self,
        agent_type: str,
        checkpoint_path: str = None,
        scenario: int = 0,
        num_episodes: int = 20,
        carla_config_path: str = "config/carla_config.yaml",
        agent_config_path: str = None,
        results_dir: str = "data/evaluation_results",
        seed: int = 42
    ):
        """
        Initialize evaluation pipeline.

        Args:
            agent_type: Type of agent ('td3', 'ddpg', 'idm')
            checkpoint_path: Path to trained agent checkpoint (not needed for IDM)
            scenario: Traffic density scenario (0=20, 1=50, 2=100 NPCs)
            num_episodes: Number of evaluation episodes
            carla_config_path: Path to CARLA config
            agent_config_path: Path to agent config (td3_config.yaml or ddpg_config.yaml)
            results_dir: Directory for evaluation results
            seed: Random seed for reproducibility
        """
        self.agent_type = agent_type.lower()
        self.checkpoint_path = checkpoint_path
        self.scenario = scenario
        self.num_episodes = num_episodes
        self.seed = seed

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load configurations
        print("\n" + "="*70)
        print(f"EVALUATION PIPELINE - {self.agent_type.upper()} AGENT")
        print("="*70)
        print(f"\n[CONFIG] Loading configurations...")

        with open(carla_config_path, 'r') as f:
            self.carla_config = yaml.safe_load(f)

        # Update NPC density based on scenario
        npc_densities = [20, 50, 100]
        if scenario < len(npc_densities):
            self.carla_config['simulation']['npc_count'] = npc_densities[scenario]
            print(f"[CONFIG] Scenario: {scenario} (NPCs: {npc_densities[scenario]})")

        # Initialize environment
        print(f"\n[ENVIRONMENT] Initializing CARLA environment...")
        self.env = CARLANavigationEnv(self.carla_config)
        print(f"[ENVIRONMENT] State space: {self.env.observation_space}")
        print(f"[ENVIRONMENT] Action space: {self.env.action_space}")

        # Initialize agent
        print(f"\n[AGENT] Initializing {self.agent_type.upper()} agent...")
        self.agent = self._load_agent(agent_config_path)

        print(f"\n[INIT] Evaluation pipeline ready!")
        print(f"[INIT] Num episodes: {num_episodes}")
        print(f"[INIT] Seed: {seed}")
        print("="*70 + "\n")

    def _load_agent(self, agent_config_path: str = None):
        """
        Load agent based on type.

        Args:
            agent_config_path: Path to agent config file

        Returns:
            Initialized agent (TD3Agent, DDPGAgent, or IDMMOBILBaseline)
        """
        if self.agent_type == 'idm':
            # IDM+MOBIL doesn't need checkpoint or config
            agent = IDMMOBILBaseline()
            print(f"[AGENT] IDM+MOBIL baseline initialized")
            return agent

        # DRL agents need config and checkpoint
        if agent_config_path is None:
            agent_config_path = f"config/{self.agent_type}_config.yaml"

        with open(agent_config_path, 'r') as f:
            agent_config = yaml.safe_load(f)

        if self.agent_type == 'td3':
            agent = TD3Agent(
                state_dim=535,
                action_dim=2,
                max_action=1.0,
                config=agent_config
            )
        elif self.agent_type == 'ddpg':
            agent = DDPGAgent(
                state_dim=535,
                action_dim=2,
                max_action=1.0,
                config=agent_config
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Load checkpoint
        if self.checkpoint_path is not None:
            agent.load_checkpoint(self.checkpoint_path)
            print(f"[AGENT] Loaded checkpoint: {self.checkpoint_path}")
        else:
            print(f"[WARNING] No checkpoint provided, using untrained agent!")

        return agent

    def evaluate(self) -> Dict:
        """
        Run evaluation episodes and collect comprehensive metrics.

        Returns:
            Dictionary with aggregated results and per-episode data
        """
        print(f"[EVALUATION] Starting evaluation for {self.num_episodes} episodes...")

        # Per-episode metrics storage
        episode_metrics = []

        for ep in range(1, self.num_episodes + 1):
            print(f"\n[EPISODE {ep}/{self.num_episodes}] Starting...")

            # Reset environment
            state = self.env.reset()
            done = False
            episode_data = {
                'episode': ep,
                'timesteps': 0,
                'total_reward': 0.0,
                'distance_traveled': 0.0,
                'collision_count': 0,
                'success': 0,
                'completion_time': 0.0,
                'speeds': [],  # For calculating average speed
                'jerks': [],  # For calculating average jerk
                'lateral_accels': [],  # For calculating average lateral acceleration
                'ttc_values': [],  # For TTC analysis
            }

            # Episode loop
            while not done:
                # Select action (deterministic, no exploration noise)
                if self.agent_type == 'idm':
                    # IDM+MOBIL uses environment state directly
                    action = self.agent.compute_action(self.env)
                else:
                    # DRL agents use state vector
                    action = self.agent.select_action(state, noise=0.0)

                # Step environment
                next_state, reward, done, info = self.env.step(action)

                # Accumulate metrics
                episode_data['total_reward'] += reward
                episode_data['timesteps'] += 1
                episode_data['collision_count'] = info.get('collision_count', 0)
                episode_data['success'] = info.get('success', 0)

                # Speed (convert m/s to km/h)
                current_speed = info.get('speed', 0.0) * 3.6
                episode_data['speeds'].append(current_speed)

                # Distance (accumulate)
                episode_data['distance_traveled'] += info.get('speed', 0.0) * 0.05  # dt=0.05s

                # Jerk (longitudinal acceleration derivative)
                if 'jerk' in info:
                    episode_data['jerks'].append(abs(info['jerk']))

                # Lateral acceleration
                if 'lateral_accel' in info:
                    episode_data['lateral_accels'].append(abs(info['lateral_accel']))

                # TTC (Time-To-Collision)
                if 'ttc' in info and info['ttc'] is not None:
                    episode_data['ttc_values'].append(info['ttc'])

                state = next_state

            # Calculate episode-level statistics
            episode_data['completion_time'] = episode_data['timesteps'] * 0.05  # dt=0.05s
            episode_data['avg_speed_kmh'] = np.mean(episode_data['speeds']) if episode_data['speeds'] else 0.0
            episode_data['avg_jerk_ms3'] = np.mean(episode_data['jerks']) if episode_data['jerks'] else 0.0
            episode_data['max_jerk_ms3'] = np.max(episode_data['jerks']) if episode_data['jerks'] else 0.0
            episode_data['avg_lateral_accel_ms2'] = np.mean(episode_data['lateral_accels']) if episode_data['lateral_accels'] else 0.0
            episode_data['collisions_per_km'] = (episode_data['collision_count'] / (episode_data['distance_traveled'] / 1000.0)
                                                   if episode_data['distance_traveled'] > 0 else 0.0)
            episode_data['ttc_violations'] = sum(1 for ttc in episode_data['ttc_values'] if ttc < 1.0)
            episode_data['min_ttc'] = np.min(episode_data['ttc_values']) if episode_data['ttc_values'] else None

            # Remove raw time-series data (keep only statistics)
            episode_data.pop('speeds')
            episode_data.pop('jerks')
            episode_data.pop('lateral_accels')
            episode_data.pop('ttc_values')

            episode_metrics.append(episode_data)

            # Console logging
            print(
                f"[EPISODE {ep}] "
                f"Reward: {episode_data['total_reward']:.2f} | "
                f"Success: {episode_data['success']} | "
                f"Collisions: {episode_data['collision_count']} | "
                f"Avg Speed: {episode_data['avg_speed_kmh']:.1f} km/h | "
                f"Time: {episode_data['completion_time']:.1f}s"
            )

        # Aggregate statistics across all episodes
        aggregated_results = self._aggregate_results(episode_metrics)

        return {
            'agent_type': self.agent_type,
            'scenario': self.scenario,
            'num_episodes': self.num_episodes,
            'seed': self.seed,
            'checkpoint_path': self.checkpoint_path,
            'aggregated_metrics': aggregated_results,
            'per_episode_metrics': episode_metrics
        }

    def _aggregate_results(self, episode_metrics: List[Dict]) -> Dict:
        """
        Aggregate per-episode metrics into summary statistics.

        Args:
            episode_metrics: List of per-episode metric dictionaries

        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        # Extract arrays for each metric
        rewards = [ep['total_reward'] for ep in episode_metrics]
        successes = [ep['success'] for ep in episode_metrics]
        collisions = [ep['collision_count'] for ep in episode_metrics]
        collisions_per_km = [ep['collisions_per_km'] for ep in episode_metrics]
        speeds = [ep['avg_speed_kmh'] for ep in episode_metrics]
        times = [ep['completion_time'] for ep in episode_metrics]
        jerks = [ep['avg_jerk_ms3'] for ep in episode_metrics]
        max_jerks = [ep['max_jerk_ms3'] for ep in episode_metrics]
        lateral_accels = [ep['avg_lateral_accel_ms2'] for ep in episode_metrics]
        ttc_violations = [ep['ttc_violations'] for ep in episode_metrics]
        min_ttcs = [ep['min_ttc'] for ep in episode_metrics if ep['min_ttc'] is not None]

        aggregated = {
            # Safety
            'success_rate_pct': np.mean(successes) * 100.0,
            'avg_collisions_per_episode': np.mean(collisions),
            'std_collisions_per_episode': np.std(collisions),
            'avg_collisions_per_km': np.mean(collisions_per_km),
            'total_ttc_violations': np.sum(ttc_violations),
            'avg_ttc_violations_per_episode': np.mean(ttc_violations),
            'min_ttc_recorded': np.min(min_ttcs) if min_ttcs else None,

            # Efficiency
            'avg_speed_kmh': np.mean(speeds),
            'std_speed_kmh': np.std(speeds),
            'avg_completion_time_s': np.mean(times),
            'std_completion_time_s': np.std(times),

            # Comfort
            'avg_jerk_ms3': np.mean(jerks),
            'std_jerk_ms3': np.std(jerks),
            'max_jerk_recorded_ms3': np.max(max_jerks) if max_jerks else 0.0,
            'avg_lateral_accel_ms2': np.mean(lateral_accels),
            'std_lateral_accel_ms2': np.std(lateral_accels),

            # Rewards
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
        }

        return aggregated

    def save_results(self, results: Dict):
        """
        Save evaluation results to CSV and JSON formats.

        Args:
            results: Dictionary from evaluate() method
        """
        # Create filename prefix
        filename_prefix = f"{self.agent_type}_scenario_{self.scenario}_eval"

        # Save per-episode metrics to CSV
        csv_path = self.results_dir / f"{filename_prefix}_episodes.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = results['per_episode_metrics'][0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['per_episode_metrics'])

        print(f"\n[RESULTS] Per-episode CSV saved: {csv_path}")

        # Save aggregated results to JSON
        json_path = self.results_dir / f"{filename_prefix}_summary.json"
        summary = {
            'agent_type': results['agent_type'],
            'scenario': results['scenario'],
            'num_episodes': results['num_episodes'],
            'seed': results['seed'],
            'checkpoint_path': results['checkpoint_path'],
            'metrics': results['aggregated_metrics']
        }

        with open(json_path, 'w') as jsonfile:
            json.dump(summary, jsonfile, indent=2)

        print(f"[RESULTS] Summary JSON saved: {json_path}")

    def print_summary(self, results: Dict):
        """
        Print formatted summary of evaluation results.

        Args:
            results: Dictionary from evaluate() method
        """
        metrics = results['aggregated_metrics']

        print("\n" + "="*70)
        print(f"EVALUATION SUMMARY - {self.agent_type.upper()} (Scenario {self.scenario})")
        print("="*70)

        print("\n[SAFETY METRICS]")
        print(f"  Success Rate:              {metrics['success_rate_pct']:.1f}%")
        print(f"  Avg Collisions/Episode:    {metrics['avg_collisions_per_episode']:.2f} ± {metrics['std_collisions_per_episode']:.2f}")
        print(f"  Avg Collisions/km:         {metrics['avg_collisions_per_km']:.2f}")
        print(f"  Total TTC Violations:      {metrics['total_ttc_violations']:.0f}")
        print(f"  Avg TTC Violations/Ep:     {metrics['avg_ttc_violations_per_episode']:.2f}")
        if metrics['min_ttc_recorded'] is not None:
            print(f"  Min TTC Recorded:          {metrics['min_ttc_recorded']:.2f}s")

        print("\n[EFFICIENCY METRICS]")
        print(f"  Avg Speed:                 {metrics['avg_speed_kmh']:.1f} ± {metrics['std_speed_kmh']:.1f} km/h")
        print(f"  Avg Completion Time:       {metrics['avg_completion_time_s']:.1f} ± {metrics['std_completion_time_s']:.1f}s")

        print("\n[COMFORT METRICS]")
        print(f"  Avg Longitudinal Jerk:     {metrics['avg_jerk_ms3']:.2f} ± {metrics['std_jerk_ms3']:.2f} m/s³")
        print(f"  Max Jerk Recorded:         {metrics['max_jerk_recorded_ms3']:.2f} m/s³")
        print(f"  Avg Lateral Acceleration:  {metrics['avg_lateral_accel_ms2']:.2f} ± {metrics['std_lateral_accel_ms2']:.2f} m/s²")

        print("\n[REWARD METRICS]")
        print(f"  Avg Reward:                {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Min/Max Reward:            {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")

        print("="*70 + "\n")

    def close(self):
        """Clean up resources."""
        self.env.close()
        print("[CLEANUP] Environment closed")


def main():
    """
    Main entry point for evaluation script.

    Command-line arguments:
    - --agent: Agent type ('td3', 'ddpg', 'idm')
    - --checkpoint: Path to trained agent checkpoint
    - --scenario: Traffic scenario (0=20, 1=50, 2=100 NPCs)
    - --num-episodes: Number of evaluation episodes
    - --seed: Random seed
    """
    parser = argparse.ArgumentParser(
        description="Evaluate autonomous vehicle agents in CARLA"
    )
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        choices=['td3', 'ddpg', 'idm'],
        help='Agent type to evaluate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to trained agent checkpoint (not needed for IDM)'
    )
    parser.add_argument(
        '--scenario',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Traffic scenario (0=20, 1=50, 2=100 NPCs)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes'
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
        default=None,
        help='Path to agent config file (default: config/{agent}_config.yaml)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/evaluation_results',
        help='Directory for evaluation results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Initialize evaluation pipeline
    evaluator = EvaluationPipeline(
        agent_type=args.agent,
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        carla_config_path=args.carla_config,
        agent_config_path=args.agent_config,
        results_dir=args.results_dir,
        seed=args.seed
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Save and print results
    evaluator.save_results(results)
    evaluator.print_summary(results)

    # Cleanup
    evaluator.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Trained Agents

Evaluates trained TD3 and DDPG agents on autonomous driving tasks.
Supports batch evaluation across multiple models, scenarios, and seeds.
Collects comprehensive metrics: safety, efficiency, comfort.

Outputs:
  - JSON results with per-episode metrics
  - CSV summary with agent comparisons
  - TensorBoard event files (optional)
  
Author: Paper DRL
Date: 2025-10
"""

import os
import sys
import json
import argparse
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# PyTorch imports
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.td3_agent import TD3Agent
from src.agents.ddpg_agent import DDPGAgent
from src.environments.carla_env import CARLAEnvironment
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Comprehensive agent evaluator supporting multiple agents and scenarios.
    
    Evaluates trained agents and collects comprehensive metrics:
    - Safety: success rate, collisions, collision counts, mean TTC
    - Efficiency: average speed, episode length, completion time
    - Comfort: longitudinal jerk, lateral acceleration
    """
    
    def __init__(self, config_carla: dict, config_agent: dict,
                 agent_type: str = 'TD3', device: str = 'cuda'):
        """
        Initialize evaluator with agent and environment.
        
        Args:
            config_carla (dict): CARLA environment configuration
            config_agent (dict): Agent configuration (TD3 or DDPG)
            agent_type (str): 'TD3' or 'DDPG'
            device (str): 'cuda' or 'cpu'
        """
        self.config_carla = config_carla
        self.config_agent = config_agent
        self.agent_type = agent_type
        self.device = device
        
        logger.info(f"Initializing {agent_type} evaluator on device: {device}")
        
        # Initialize environment
        self.env = CARLAEnvironment(config=config_carla)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        
        # Initialize agent (will load checkpoint later)
        if agent_type == 'TD3':
            self.agent = TD3Agent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=1.0,
                device=device
            )
        elif agent_type == 'DDPG':
            self.agent = DDPGAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=1.0,
                device=device
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        logger.info(f"{agent_type} agent initialized (state_dim={self.state_dim}, action_dim={self.action_dim})")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load agent checkpoint from file.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            self.agent.load_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def evaluate_episode(self) -> Dict:
        """
        Run single evaluation episode and collect metrics.
        
        Returns:
            dict: Metrics for this episode including:
                - reward: cumulative episode reward
                - length: number of steps
                - collisions: collision count
                - success: whether episode completed without collision
                - avg_speed: average vehicle speed (m/s)
                - avg_accel_x: average longitudinal acceleration (m/s²)
                - avg_accel_y: average lateral acceleration (m/s²)
                - max_accel_x: maximum longitudinal acceleration
                - max_accel_y: maximum lateral acceleration
                - min_ttc: minimum time-to-collision observed
        """
        state, _ = self.env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        collisions_start = self.env.collision_count
        
        # Trajectory tracking for comfort metrics
        velocities = []
        accel_x_values = []
        accel_y_values = []
        ttc_values = []
        
        done = False
        max_steps = self.config_carla.get('max_episode_steps', 500)
        
        while not done and episode_length < max_steps:
            # Deterministic action (no exploration noise) for evaluation
            action = self.agent.select_action(state, noise=0.0)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Get vehicle dynamics
            vehicle = self.env.vehicle
            velocity = vehicle.get_velocity()
            speed = velocity.length()
            velocities.append(speed)
            
            # Get acceleration (estimate from velocity change)
            if episode_length > 1:
                # Get previous velocity (approximate from state if available)
                # For now, we'll use CARLA's acceleration if available
                accel = vehicle.get_acceleration()
                accel_x = accel.x
                accel_y = accel.y
                accel_x_values.append(accel_x)
                accel_y_values.append(accel_y)
            
            # Collect TTC information from environment (if available)
            if hasattr(self.env, 'min_ttc'):
                ttc_values.append(self.env.min_ttc)
            
            # Move to next state
            state = next_state
        
        # Compute summary statistics
        collisions_this_episode = self.env.collision_count - collisions_start
        success = collisions_this_episode == 0
        
        # Safety metrics
        avg_speed = np.mean(velocities) if velocities else 0.0
        avg_accel_x = np.mean(accel_x_values) if accel_x_values else 0.0
        avg_accel_y = np.mean(accel_y_values) if accel_y_values else 0.0
        max_accel_x = np.max(np.abs(accel_x_values)) if accel_x_values else 0.0
        max_accel_y = np.max(np.abs(accel_y_values)) if accel_y_values else 0.0
        min_ttc = np.min(ttc_values) if ttc_values else float('inf')
        
        # Comfort metrics: jerk is second derivative of position
        # Approximate as change in acceleration
        jerk_x_values = []
        jerk_y_values = []
        if len(accel_x_values) > 1:
            jerk_x_values = np.diff(accel_x_values)
            jerk_y_values = np.diff(accel_y_values)
        
        avg_jerk_x = np.mean(np.abs(jerk_x_values)) if jerk_x_values is not None else 0.0
        avg_jerk_y = np.mean(np.abs(jerk_y_values)) if jerk_y_values is not None else 0.0
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'collisions': collisions_this_episode,
            'success': success,
            'avg_speed': avg_speed,
            'avg_accel_x': avg_accel_x,
            'avg_accel_y': avg_accel_y,
            'max_accel_x': max_accel_x,
            'max_accel_y': max_accel_y,
            'min_ttc': min_ttc,
            'avg_jerk_x': avg_jerk_x,
            'avg_jerk_y': avg_jerk_y,
        }
    
    def evaluate(self, num_episodes: int = 20) -> Dict:
        """
        Run multiple evaluation episodes and aggregate metrics.
        
        Args:
            num_episodes (int): Number of evaluation episodes
            
        Returns:
            dict: Aggregated metrics with mean, std, min, max for each metric
        """
        logger.info(f"Running {num_episodes} evaluation episodes...")
        
        episodes = []
        
        for ep in range(num_episodes):
            episode_metrics = self.evaluate_episode()
            episodes.append(episode_metrics)
            
            logger.debug(
                f"  Episode {ep+1}/{num_episodes}: "
                f"reward={episode_metrics['reward']:.2f}, "
                f"length={episode_metrics['length']}, "
                f"success={episode_metrics['success']}, "
                f"collisions={episode_metrics['collisions']}"
            )
        
        # Aggregate metrics across episodes
        aggregated = {}
        
        for key in ['reward', 'length', 'collisions', 'avg_speed', 'avg_accel_x', 
                   'avg_accel_y', 'max_accel_x', 'max_accel_y', 'min_ttc', 
                   'avg_jerk_x', 'avg_jerk_y']:
            values = [ep[key] for ep in episodes]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Aggregate boolean metrics
        success_values = [ep['success'] for ep in episodes]
        aggregated['success_rate'] = {
            'mean': np.mean(success_values),
            'count': np.sum(success_values),
            'total': len(success_values)
        }
        
        # Safety metric: collisions per episode
        total_collisions = sum(ep['collisions'] for ep in episodes)
        aggregated['collisions_per_episode'] = {
            'mean': np.mean([ep['collisions'] for ep in episodes]),
            'total': total_collisions
        }
        
        logger.info(f"Evaluation completed: {num_episodes} episodes")
        
        return {
            'episodes': episodes,
            'aggregated': aggregated,
            'num_episodes': num_episodes
        }


def evaluate_batch(agent_configs: List[Tuple[str, str, str]], 
                   scenario: int = 0,
                   num_episodes: int = 20,
                   output_dir: str = 'data/eval_results') -> Dict:
    """
    Evaluate multiple trained agents in batch mode.
    
    Args:
        agent_configs (list): List of (agent_type, checkpoint_path, run_name) tuples
        scenario (int): CARLA scenario to evaluate on
        num_episodes (int): Episodes per agent
        output_dir (str): Directory to save results
        
    Returns:
        dict: Results from all agents
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    config_carla = load_config('config/carla_config.yaml')
    config_carla['scenario'] = scenario
    config_td3 = load_config('config/td3_config.yaml')
    config_ddpg = load_config('config/ddpg_config.yaml')
    
    results_all = {}
    
    for agent_type, checkpoint_path, run_name in agent_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {agent_type}: {run_name}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"{'='*60}")
        
        # Select config
        config_agent = config_td3 if agent_type == 'TD3' else config_ddpg
        
        # Create evaluator
        evaluator = AgentEvaluator(
            config_carla=config_carla,
            config_agent=config_agent,
            agent_type=agent_type,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load checkpoint
        if not evaluator.load_checkpoint(checkpoint_path):
            logger.error(f"Failed to load checkpoint for {run_name}, skipping...")
            continue
        
        # Run evaluation
        eval_results = evaluator.evaluate(num_episodes=num_episodes)
        results_all[run_name] = {
            'agent_type': agent_type,
            'checkpoint': checkpoint_path,
            'results': eval_results
        }
        
        # Log summary
        agg = eval_results['aggregated']
        logger.info(f"\nResults for {run_name}:")
        logger.info(f"  Success Rate: {agg['success_rate']['mean']:.1%} ({agg['success_rate']['count']}/{agg['success_rate']['total']})")
        logger.info(f"  Avg Reward: {agg['reward']['mean']:.2f} ± {agg['reward']['std']:.2f}")
        logger.info(f"  Avg Speed: {agg['avg_speed']['mean']:.2f} ± {agg['avg_speed']['std']:.2f} m/s")
        logger.info(f"  Collisions/Episode: {agg['collisions_per_episode']['mean']:.2f}")
        logger.info(f"  Avg Lateral Accel: {agg['avg_accel_y']['mean']:.3f} ± {agg['avg_accel_y']['std']:.3f} m/s²")
        logger.info(f"  Avg Jerk (Lateral): {agg['avg_jerk_y']['mean']:.3f} ± {agg['avg_jerk_y']['std']:.3f} m/s³")
    
    # Save aggregated results
    results_file = output_dir / f"evaluation_scenario_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} not JSON serializable")
        
        json.dump(results_all, f, indent=2, default=convert)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return results_all


def generate_comparison_csv(results_all: Dict, scenario: int, output_dir: str = 'data/eval_results'):
    """
    Generate comparison CSV from evaluation results.
    
    Args:
        results_all (dict): Results dictionary from evaluate_batch
        scenario (int): Scenario number
        output_dir (str): Directory to save CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / f"comparison_scenario_{scenario}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['Agent', 'Agent_Type', 'Success_Rate', 'Avg_Reward', 'Std_Reward',
                     'Avg_Speed_mps', 'Collisions_Per_Episode', 'Avg_Lateral_Accel',
                     'Avg_Jerk_Lateral', 'Avg_Episode_Length']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for run_name, run_results in results_all.items():
            agg = run_results['results']['aggregated']
            
            writer.writerow({
                'Agent': run_name,
                'Agent_Type': run_results['agent_type'],
                'Success_Rate': f"{agg['success_rate']['mean']:.3f}",
                'Avg_Reward': f"{agg['reward']['mean']:.2f}",
                'Std_Reward': f"{agg['reward']['std']:.2f}",
                'Avg_Speed_mps': f"{agg['avg_speed']['mean']:.2f}",
                'Collisions_Per_Episode': f"{agg['collisions_per_episode']['mean']:.2f}",
                'Avg_Lateral_Accel': f"{agg['avg_accel_y']['mean']:.4f}",
                'Avg_Jerk_Lateral': f"{agg['avg_jerk_y']['mean']:.4f}",
                'Avg_Episode_Length': f"{agg['length']['mean']:.0f}",
            })
    
    logger.info(f"Comparison CSV saved to {csv_file}")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Agent Evaluation Script'
    )
    
    # Evaluation arguments
    parser.add_argument('--scenario', type=int, default=0, choices=[0, 1, 2],
                        help='Traffic scenario: 0=20 NPCs, 1=50 NPCs, 2=100 NPCs')
    parser.add_argument('--num-episodes', type=int, default=20,
                        help='Number of evaluation episodes per agent')
    parser.add_argument('--output-dir', type=str, default='data/eval_results',
                        help='Output directory for results')
    
    # Agent checkpoint arguments
    parser.add_argument('--td3-checkpoint', type=str, default=None,
                        help='Path to trained TD3 checkpoint')
    parser.add_argument('--ddpg-checkpoint', type=str, default=None,
                        help='Path to trained DDPG checkpoint')
    
    # Config arguments
    parser.add_argument('--carla-config', type=str, default='config/carla_config.yaml',
                        help='Path to CARLA config YAML')
    parser.add_argument('--td3-config', type=str, default='config/td3_config.yaml',
                        help='Path to TD3 config YAML')
    parser.add_argument('--ddpg-config', type=str, default='config/ddpg_config.yaml',
                        help='Path to DDPG config YAML')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Agent Evaluation Script")
    logger.info(f"{'='*80}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Episodes per agent: {args.num_episodes}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*80}\n")
    
    # Build agent configs
    agent_configs = []
    
    if args.td3_checkpoint:
        agent_configs.append(('TD3', args.td3_checkpoint, 'TD3'))
    
    if args.ddpg_checkpoint:
        agent_configs.append(('DDPG', args.ddpg_checkpoint, 'DDPG'))
    
    if not agent_configs:
        logger.error("No checkpoints specified. Use --td3-checkpoint and/or --ddpg-checkpoint")
        sys.exit(1)
    
    # Run batch evaluation
    results_all = evaluate_batch(
        agent_configs=agent_configs,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )
    
    # Generate comparison CSV
    generate_comparison_csv(results_all, args.scenario, args.output_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()


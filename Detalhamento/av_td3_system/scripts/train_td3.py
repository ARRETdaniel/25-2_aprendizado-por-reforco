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

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

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
        checkpoint_dir: str = "data/checkpoints"
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
            self.carla_config['simulation']['npc_count'] = npc_densities[scenario]
            print(f"[CONFIG] NPC count set to: {npc_densities[scenario]}")

        # Initialize environment
        print(f"\n[ENVIRONMENT] Initializing CARLA environment...")
        self.env = CARLANavigationEnv(self.carla_config)
        print(f"[ENVIRONMENT] State space: {self.env.observation_space}")
        print(f"[ENVIRONMENT] Action space: {self.env.action_space}")

        # Initialize agent
        print(f"\n[AGENT] Initializing TD3 agent...")
        self.agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config=self.agent_config
        )

        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scenario_name = f"scenario_{scenario}_npcs_{npc_densities[scenario]}"
        self.log_name = f"TD3_{scenario_name}_{timestamp}"
        self.log_path = self.log_dir / self.log_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_path))
        
        print(f"[LOGGING] TensorBoard logs: {self.log_path}")

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
        
        state = self.env.reset()
        done = False
        
        start_timesteps = self.agent_config['training']['start_timesteps']
        
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

            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Track episode metrics
            self.episode_reward += reward
            self.episode_collision_count += info.get('collision_count', 0)
            self.episode_steps_since_collision = info.get('steps_since_collision', 0)

            # Store transition in replay buffer
            done_bool = float(done) if self.episode_timesteps < 300 else True  # 300s timeout
            self.agent.replay_buffer.add(
                state,
                action,
                next_state,
                reward,
                done_bool
            )

            state = next_state

            # Train agent (only after exploration phase)
            if t > start_timesteps:
                metrics = self.agent.train(batch_size=self.agent_config['training']['batch_size'])
                
                # Log training metrics every 100 steps
                if t % 100 == 0:
                    self.writer.add_scalar('train/critic_loss', metrics['critic_loss'], t)
                    self.writer.add_scalar('train/q1_value', metrics['q1_value'], t)
                    self.writer.add_scalar('train/q2_value', metrics['q2_value'], t)
                    
                    if 'actor_loss' in metrics:  # Actor updated only on delayed steps
                        self.writer.add_scalar('train/actor_loss', metrics['actor_loss'], t)

            # Episode termination
            if done:
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

                # Reset episode metrics
                state = self.env.reset()
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

        for ep in range(self.num_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Deterministic action (no noise)
                action = self.agent.select_action(state, noise=0.0)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                state = next_state

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
        checkpoint_dir=args.checkpoint_dir
    )

    trainer.train()


if __name__ == "__main__":
    main()

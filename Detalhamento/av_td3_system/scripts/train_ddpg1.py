#!/usr/bin/env python3
"""
DDPG Training Script for Autonomous Vehicle Navigation in CARLA

Implements complete training pipeline for DDPG baseline agent on autonomous driving tasks.
CRITICAL: Uses IDENTICAL hyperparameters to TD3 for fair algorithmic comparison.
Only algorithmic differences: single Critic, immediate updates, no target smoothing.

Structure mirrors train_td3.py for easy A/B testing and comparison.

Author: Paper DRL
Date: 2025-10
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
import numpy as np

# PyTorch imports
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ddpg_agent import DDPGAgent
from src.environments.carla_env import CARLAEnvironment
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DDPGTrainingPipeline:
    """
    DDPG training pipeline orchestrator.

    Manages complete training lifecycle: initialization, exploration phase,
    learning phase, evaluation, checkpointing, and logging.

    IMPORTANT: Uses identical hyperparameters and architecture as TD3Agent
    to ensure fair algorithmic comparison. Only difference: DDPG algorithm.
    """

    def __init__(self, config_carla: dict, config_ddpg: dict, config_training: dict,
                 scenario: int = 0, seed: int = 42, log_dir: str = "data/logs",
                 checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize DDPG training pipeline.

        Args:
            config_carla (dict): CARLA environment configuration
            config_ddpg (dict): DDPG agent configuration (identical hyperparameters to TD3)
            config_training (dict): Training procedure configuration
            scenario (int): Traffic scenario (0=20 NPCs, 1=50 NPCs, 2=100 NPCs)
            seed (int): Random seed for reproducibility
            log_dir (str): TensorBoard log directory
            checkpoint_dir (str): Checkpoint save directory
        """
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.scenario = scenario
        self.seed = seed
        self.config_carla = config_carla
        self.config_ddpg = config_ddpg
        self.config_training = config_training

        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npcs = [20, 50, 100][scenario]
        self.run_dir = self.log_dir / f"DDPG_scenario_{scenario}_npcs_{npcs}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DDPG Training Pipeline initialized (scenario={scenario}, seed={seed})")
        logger.info(f"Run directory: {self.run_dir}")

        # Initialize environment
        logger.info("Initializing CARLA environment...")
        self.env = CARLAEnvironment(config=config_carla)
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        logger.info(f"State dimension: {self.state_dim}, Action dimension: {self.action_dim}")

        # Initialize DDPG agent with config
        logger.info("Initializing DDPG agent...")
        # Create agent with identical hyperparameters as TD3 for fair comparison
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=1.0,  # Actions normalized to [-1, 1]
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"DDPG Agent initialized on device: {self.agent.device}")
        logger.info("CRITICAL: Using IDENTICAL hyperparameters as TD3 for fair comparison")
        logger.info(f"  - Learning rate (Actor): {self.config_ddpg.get('actor_lr', 3e-4)}")
        logger.info(f"  - Learning rate (Critic): {self.config_ddpg.get('critic_lr', 3e-4)}")
        logger.info(f"  - Soft update tau: {self.config_ddpg.get('tau', 0.005)}")
        logger.info(f"  - Discount gamma: {self.config_ddpg.get('gamma', 0.99)}")

        # Training metrics
        self.episode_num = 0
        self.total_timesteps = 0
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_timesteps = 0

        # TensorBoard logging (using numpy arrays for metrics)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))
            logger.info(f"TensorBoard logging enabled: {self.run_dir}/tensorboard")
        except ImportError:
            logger.warning("TensorBoard not available, skipping TensorBoard logging")
            self.writer = None

        # Results tracking
        self.training_rewards = []
        self.eval_rewards = []
        self.eval_success_rates = []
        self.eval_collisions = []

        logger.info("DDPG Training Pipeline ready for training")

    def train(self, max_timesteps: int = int(1e6),
              start_timesteps: int = int(1e4),
              eval_freq: int = int(5e3),
              checkpoint_freq: int = int(1e4),
              num_eval_episodes: int = 10):
        """
        Main training loop.

        Implements standard RL training procedure:
        1. Exploration phase: random actions to populate replay buffer
        2. Learning phase: collect transitions and train agent
        3. Periodic evaluation: test agent performance
        4. Checkpoint saving: save trained models

        Args:
            max_timesteps (int): Maximum training timesteps (default 1M)
            start_timesteps (int): Exploration phase duration in steps (default 10K)
            eval_freq (int): Evaluation frequency in timesteps (default 5K)
            checkpoint_freq (int): Checkpoint save frequency in timesteps (default 10K)
            num_eval_episodes (int): Episodes per evaluation (default 10)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting DDPG Training")
        logger.info(f"{'='*80}")
        logger.info(f"Max timesteps: {max_timesteps:,.0f}")
        logger.info(f"Exploration phase: {start_timesteps:,.0f} steps")
        logger.info(f"Evaluation frequency: {eval_freq:,.0f} steps")
        logger.info(f"Checkpoint frequency: {checkpoint_freq:,.0f} steps")
        logger.info(f"Num eval episodes: {num_eval_episodes}")
        logger.info(f"{'='*80}\n")

        # Get exploration noise from config (DDPG uses 0.1 like TD3)
        expl_noise = self.config_training.get('expl_noise', 0.1)

        # Reset environment for episode start
        state, _ = self.env.reset()
        self.episode_reward = 0.0
        self.episode_collisions = self.env.collision_count
        self.episode_timesteps = 0

        # Training loop
        for t in range(int(max_timesteps)):
            self.total_timesteps = t

            # EXPLORATION PHASE: First start_timesteps use random actions
            if t < start_timesteps:
                # Random action for exploration
                action = np.random.uniform(-1, 1, size=(self.action_dim,))
                logger.debug(f"Exploration step {t+1}/{start_timesteps}: random action")
            else:
                # LEARNING PHASE: Use agent's policy with exploration noise
                action = self.agent.select_action(state, noise=expl_noise)

                # Train agent on replay buffer batch
                if len(self.agent.replay_buffer) >= self.config_training.get('batch_size', 256):
                    self.agent.train(
                        batch_size=self.config_training.get('batch_size', 256)
                    )

            # Step environment with action
            next_state, reward, done, info = self.env.step(action)

            # Update metrics
            self.episode_reward += reward
            self.episode_timesteps += 1

            # Store transition in replay buffer
            self.agent.replay_buffer.add(state, action, next_state, reward, done)

            # Prepare for next step
            state = next_state

            # Episode termination handling
            if done or self.episode_timesteps >= self.config_training.get('max_episode_steps', 500):
                # Reset environment for next episode
                logger.info(
                    f"Episode {self.episode_num+1} | Timestep {t+1:,} | "
                    f"Reward: {self.episode_reward:.1f} | "
                    f"Collisions: {self.env.collision_count - self.episode_collisions} | "
                    f"Steps: {self.episode_timesteps}"
                )

                # Log to TensorBoard
                if self.writer is not None and self.episode_num % 10 == 0:
                    self.writer.add_scalar('training/episode_reward',
                                          self.episode_reward,
                                          self.episode_num)
                    self.writer.add_scalar('training/episode_length',
                                          self.episode_timesteps,
                                          self.episode_num)

                # Store metrics
                self.training_rewards.append(self.episode_reward)

                # Reset for next episode
                state, _ = self.env.reset()
                self.episode_num += 1
                self.episode_reward = 0.0
                self.episode_collisions = self.env.collision_count
                self.episode_timesteps = 0

            # PERIODIC EVALUATION
            if (t + 1) % eval_freq == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluation at timestep {t+1:,}")
                logger.info(f"{'='*60}")

                eval_results = self.evaluate(num_episodes=num_eval_episodes)

                if self.writer is not None:
                    self.writer.add_scalar('eval/avg_reward',
                                          eval_results['avg_reward'],
                                          t)
                    self.writer.add_scalar('eval/success_rate',
                                          eval_results['success_rate'],
                                          t)
                    self.writer.add_scalar('eval/collisions_per_km',
                                          eval_results['collisions_per_km'],
                                          t)

                self.eval_rewards.append(eval_results['avg_reward'])
                self.eval_success_rates.append(eval_results['success_rate'])
                self.eval_collisions.append(eval_results['collisions_per_km'])

                logger.info(f"Average Reward: {eval_results['avg_reward']:.2f}")
                logger.info(f"Success Rate: {eval_results['success_rate']:.1%}")
                logger.info(f"Collisions/km: {eval_results['collisions_per_km']:.2f}")
                logger.info(f"Average Episode Length: {eval_results['avg_episode_length']:.0f} steps\n")

            # CHECKPOINT SAVING
            if (t + 1) % checkpoint_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"ddpg_scenario_{self.scenario}_step_{t+1}.pth"
                self.agent.save_checkpoint(str(checkpoint_path))
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Training completed!")
        logger.info(f"Total timesteps: {self.total_timesteps:,}")
        logger.info(f"Total episodes: {self.episode_num}")
        logger.info(f"{'='*80}\n")

        # Save final results
        self.save_final_results()

        return self.training_rewards

    def evaluate(self, num_episodes: int = 10) -> dict:
        """
        Evaluate agent on deterministic episodes (no exploration noise).

        Args:
            num_episodes (int): Number of evaluation episodes

        Returns:
            dict: Metrics including avg_reward, success_rate, collisions_per_km, etc.
        """
        logger.info(f"Running {num_episodes} evaluation episodes...")

        episode_rewards = []
        episode_lengths = []
        collisions_list = []
        success_count = 0
        distances_traveled = []

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            collisions_start = self.env.collision_count
            distance_start = self.env.vehicle.get_velocity().length()  # Approximate

            done = False
            while not done and episode_length < self.config_training.get('max_episode_steps', 500):
                # Deterministic action (no noise) for evaluation
                action = self.agent.select_action(state, noise=0.0)
                state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            collisions_this_episode = self.env.collision_count - collisions_start
            collisions_list.append(collisions_this_episode)

            # Success = reached destination without collision
            if collisions_this_episode == 0:
                success_count += 1

            distances_traveled.append(distance_start * episode_length)

            logger.debug(f"  Eval episode {ep+1}/{num_episodes}: "
                        f"reward={episode_reward:.2f}, length={episode_length}, "
                        f"collisions={collisions_this_episode}")

        # Compute summary statistics
        avg_reward = np.mean(episode_rewards)
        success_rate = success_count / num_episodes
        total_distance_km = np.sum(distances_traveled) / 1000.0
        collisions_per_km = np.sum(collisions_list) / max(total_distance_km, 1.0)
        avg_episode_length = np.mean(episode_lengths)

        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'collisions_per_km': collisions_per_km,
            'avg_episode_length': avg_episode_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }

    def save_final_results(self):
        """Save training results to JSON file for analysis."""
        results = {
            'scenario': self.scenario,
            'seed': self.seed,
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.episode_num,
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': self.eval_success_rates,
            'eval_collisions_per_km': self.eval_collisions,
            'config': {
                'carla': self.config_carla,
                'ddpg': self.config_ddpg,
                'training': self.config_training,
            }
        }

        results_file = self.run_dir / "results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {type(obj)} not JSON serializable")

            json.dump(results, f, indent=2, default=convert)

        logger.info(f"Results saved to {results_file}")

        # Also save training config for reference
        config_file = self.run_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'carla': self.config_carla,
                'ddpg': self.config_ddpg,
                'training': self.config_training,
            }, f, indent=2, default=convert)

        logger.info(f"Configuration saved to {config_file}")


def main():
    """Main entry point for DDPG training script."""
    parser = argparse.ArgumentParser(
        description='DDPG Training Script for Autonomous Vehicle Navigation'
    )

    # Scenario and seed arguments
    parser.add_argument('--scenario', type=int, default=0, choices=[0, 1, 2],
                        help='Traffic scenario: 0=20 NPCs, 1=50 NPCs, 2=100 NPCs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Training arguments
    parser.add_argument('--max-timesteps', type=float, default=1e6,
                        help='Maximum training timesteps (default 1M)')
    parser.add_argument('--eval-freq', type=float, default=5e3,
                        help='Evaluation frequency in timesteps (default 5K)')
    parser.add_argument('--checkpoint-freq', type=float, default=1e4,
                        help='Checkpoint save frequency in timesteps (default 10K)')
    parser.add_argument('--num-eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes (default 10)')

    # Path arguments
    parser.add_argument('--carla-config', type=str,
                        default='config/carla_config.yaml',
                        help='Path to CARLA config YAML')
    parser.add_argument('--ddpg-config', type=str,
                        default='config/ddpg_config.yaml',
                        help='Path to DDPG config YAML')
    parser.add_argument('--training-config', type=str,
                        default='config/training_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--log-dir', type=str, default='data/logs',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints',
                        help='Checkpoint save directory')

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info(f"DDPG Training Script")
    logger.info(f"{'='*80}")
    logger.info(f"Arguments:")
    logger.info(f"  Scenario: {args.scenario} ({'20 NPCs' if args.scenario==0 else '50 NPCs' if args.scenario==1 else '100 NPCs'})")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Max timesteps: {args.max_timesteps:,.0f}")
    logger.info(f"  Eval frequency: {args.eval_freq:,.0f}")
    logger.info(f"  Checkpoint frequency: {args.checkpoint_freq:,.0f}")
    logger.info(f"{'='*80}\n")

    # Load configurations
    logger.info("Loading configurations...")
    config_carla = load_config(args.carla_config)
    config_ddpg = load_config(args.ddpg_config)
    config_training = load_config(args.training_config)

    # Adjust scenario in CARLA config
    config_carla['scenario'] = args.scenario

    logger.info("Configurations loaded successfully")

    # Create and run training pipeline
    pipeline = DDPGTrainingPipeline(
        config_carla=config_carla,
        config_ddpg=config_ddpg,
        config_training=config_training,
        scenario=args.scenario,
        seed=args.seed,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )

    # Run training
    training_rewards = pipeline.train(
        max_timesteps=int(args.max_timesteps),
        eval_freq=int(args.eval_freq),
        checkpoint_freq=int(args.checkpoint_freq),
        num_eval_episodes=args.num_eval_episodes
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"DDPG Training Completed Successfully!")
    logger.info(f"Results saved to: {pipeline.run_dir}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()

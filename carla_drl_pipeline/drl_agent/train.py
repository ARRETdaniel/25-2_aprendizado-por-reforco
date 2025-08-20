"""
Main Training Script for CARLA DRL Pipeline

This script orchestrates the complete training pipeline, integrating:
- CARLA simulation environment
- ROS 2 communication bridge 
- PPO reinforcement learning algorithm
- Monitoring and evaluation systems
- Model checkpointing and deployment

Usage:
    python train.py --config configs/train.yaml --sim-config configs/sim.yaml
"""

import os
import sys
import time
import argparse
import logging
import signal
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from ppo_algorithm import PPOAgent, PPOConfig
from environment_wrapper import CarlaROS2Environment
from networks import create_networks
from utils import setup_logging, save_config, load_config, create_directories

# Configure logging
logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages the complete DRL training pipeline."""
    
    def __init__(self, train_config_path: str, sim_config_path: str):
        """Initialize training manager.
        
        Args:
            train_config_path: Path to training configuration
            sim_config_path: Path to simulation configuration
        """
        # Load configurations
        self.train_config = load_config(train_config_path)
        self.sim_config = load_config(sim_config_path)
        
        # Setup directories
        self.experiment_dir = self._setup_experiment_dir()
        create_directories(self.experiment_dir)
        
        # Setup logging
        setup_logging(
            log_file=self.experiment_dir / "training.log",
            level=self.train_config.get('log_level', 'INFO')
        )
        
        # Save configurations
        save_config(self.train_config, self.experiment_dir / "train_config.yaml")
        save_config(self.sim_config, self.experiment_dir / "sim_config.yaml")
        
        # Initialize components
        self.environment = None
        self.agent = None
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'loss_values': [],
            'eval_rewards': [],
            'eval_steps': []
        }
        
        # Training state
        self.current_episode = 0
        self.total_timesteps = 0
        self.best_eval_reward = -np.inf
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False
        
        logger.info(f"TrainingManager initialized - Experiment: {self.experiment_dir}")
    
    def _setup_experiment_dir(self) -> Path:
        """Setup experiment directory with timestamp.
        
        Returns:
            Path to experiment directory
        """
        base_dir = Path(self.train_config.get('experiment_dir', './experiments'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.train_config.get('experiment_name', 'carla_drl')
        
        experiment_dir = base_dir / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_dir
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum} - initiating graceful shutdown")
        self.shutdown_requested = True
    
    def setup_environment(self):
        """Setup CARLA environment."""
        logger.info("Setting up CARLA environment...")
        
        # Create environment configuration file
        env_config_path = self.experiment_dir / "env_config.yaml"
        env_config = {
            **self.sim_config,
            'observation': self.train_config['environment']['observation'],
            'reward': self.train_config['environment']['reward'],
            'communication': self.train_config['environment']['communication'],
            'max_episode_steps': self.train_config['environment']['max_episode_steps']
        }
        save_config(env_config, env_config_path)
        
        # Create environment
        self.environment = CarlaROS2Environment(str(env_config_path))
        
        logger.info("CARLA environment setup complete")
    
    def setup_agent(self):
        """Setup PPO agent."""
        logger.info("Setting up PPO agent...")
        
        # Create PPO configuration
        ppo_config = PPOConfig(**self.train_config['algorithm'])
        ppo_config.tensorboard_log = str(self.experiment_dir / "tensorboard")
        ppo_config.checkpoint_dir = str(self.experiment_dir / "checkpoints")
        ppo_config.model_save_path = str(self.experiment_dir / "models")
        
        # Get environment spaces
        obs_space = {
            'image': self.environment.observation_space['image'].shape,
            'vector': self.environment.observation_space['vector'].shape
        }
        action_space = self.environment.action_space.shape
        
        # Create agent
        self.agent = PPOAgent(
            config=ppo_config,
            obs_space=obs_space,
            action_space=action_space
        )
        
        # Load checkpoint if specified
        checkpoint_path = self.train_config.get('checkpoint_path')
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.agent.load(checkpoint_path)
            
            # Update training state
            self.current_episode = self.agent.n_updates
            self.total_timesteps = self.agent.total_timesteps
        
        logger.info("PPO agent setup complete")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        try:
            # Setup components
            self.setup_environment()
            self.setup_agent()
            
            # Training parameters
            max_episodes = self.train_config.get('max_episodes', 1000)
            eval_frequency = self.train_config.get('eval_frequency', 50)
            save_frequency = self.train_config.get('save_frequency', 100)
            
            # Training loop
            while (self.current_episode < max_episodes and 
                   not self.shutdown_requested):
                
                # Training step
                episode_stats = self._train_episode()
                
                # Update statistics
                self._update_training_stats(episode_stats)
                
                # Logging
                if self.current_episode % self.train_config.get('log_frequency', 10) == 0:
                    self._log_training_progress()
                
                # Evaluation
                if self.current_episode % eval_frequency == 0:
                    eval_stats = self._evaluate_agent()
                    self._update_eval_stats(eval_stats)
                    
                    # Save best model
                    if eval_stats['mean_reward'] > self.best_eval_reward:
                        self.best_eval_reward = eval_stats['mean_reward']
                        best_model_path = self.experiment_dir / "models" / "best_model.pt"
                        self.agent.save(str(best_model_path))
                        logger.info(f"New best model saved: {eval_stats['mean_reward']:.2f}")
                
                # Save checkpoint
                if self.current_episode % save_frequency == 0:
                    checkpoint_path = (
                        self.experiment_dir / "checkpoints" / 
                        f"checkpoint_episode_{self.current_episode}.pt"
                    )
                    self.agent.save(str(checkpoint_path))
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Plot progress
                if self.current_episode % (eval_frequency * 2) == 0:
                    self._plot_training_progress()
                
                self.current_episode += 1
            
            # Final save
            final_model_path = self.experiment_dir / "models" / "final_model.pt"
            self.agent.save(str(final_model_path))
            
            # Final evaluation
            final_eval = self._evaluate_agent(n_episodes=10)
            logger.info(f"Final evaluation: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
            
            # Save final statistics
            self._save_training_stats()
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self._cleanup()
    
    def _train_episode(self) -> Dict[str, Any]:
        """Execute one training episode.
        
        Returns:
            Episode statistics
        """
        try:
            # Execute training step
            stats = self.agent.train_step(self.environment)
            
            # Update total timesteps
            self.total_timesteps = self.agent.total_timesteps
            
            return stats
            
        except Exception as e:
            logger.error(f"Training episode failed: {e}")
            return {}
    
    def _evaluate_agent(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate agent performance.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        logger.info(f"Evaluating agent for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Convert observations to tensors
                obs_tensor = {
                    key: torch.from_numpy(value).float().unsqueeze(0).to(self.agent.device)
                    for key, value in obs.items()
                }
                
                # Get deterministic action
                action, _, _ = self.agent.get_action(obs_tensor, deterministic=True)
                action_np = action.cpu().numpy().squeeze(0)
                
                # Step environment
                obs, reward, done, info = self.environment.step(action_np)
                
                episode_reward += reward
                episode_length += 1
                
                # Safety check
                if episode_length > self.train_config['environment']['max_episode_steps']:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            logger.debug(f"Eval episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")
        
        # Calculate statistics
        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        logger.info(f"Evaluation complete: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        
        return stats
    
    def _update_training_stats(self, episode_stats: Dict[str, Any]):
        """Update training statistics.
        
        Args:
            episode_stats: Episode statistics
        """
        # Extract relevant statistics
        if 'rollout/ep_rew_mean' in episode_stats:
            self.training_stats['episode_rewards'].append(episode_stats['rollout/ep_rew_mean'])
        
        if 'rollout/ep_len_mean' in episode_stats:
            self.training_stats['episode_lengths'].append(episode_stats['rollout/ep_len_mean'])
        
        if 'total_loss' in episode_stats:
            self.training_stats['loss_values'].append(episode_stats['total_loss'])
    
    def _update_eval_stats(self, eval_stats: Dict[str, float]):
        """Update evaluation statistics.
        
        Args:
            eval_stats: Evaluation statistics
        """
        self.training_stats['eval_rewards'].append(eval_stats['mean_reward'])
        self.training_stats['eval_steps'].append(self.current_episode)
    
    def _log_training_progress(self):
        """Log current training progress."""
        recent_rewards = self.training_stats['episode_rewards'][-10:]
        recent_lengths = self.training_stats['episode_lengths'][-10:]
        recent_losses = self.training_stats['loss_values'][-10:]
        
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0.0
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        logger.info(
            f"Episode {self.current_episode:4d} | "
            f"Timesteps: {self.total_timesteps:7d} | "
            f"Reward: {avg_reward:7.2f} | "
            f"Length: {avg_length:6.1f} | "
            f"Loss: {avg_loss:8.4f}"
        )
    
    def _plot_training_progress(self):
        """Plot and save training progress."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Episode rewards
            if self.training_stats['episode_rewards']:
                axes[0, 0].plot(self.training_stats['episode_rewards'])
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].grid(True)
            
            # Episode lengths
            if self.training_stats['episode_lengths']:
                axes[0, 1].plot(self.training_stats['episode_lengths'])
                axes[0, 1].set_title('Episode Lengths')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Steps')
                axes[0, 1].grid(True)
            
            # Loss values
            if self.training_stats['loss_values']:
                axes[1, 0].plot(self.training_stats['loss_values'])
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Update')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True)
            
            # Evaluation rewards
            if self.training_stats['eval_rewards'] and self.training_stats['eval_steps']:
                axes[1, 1].plot(
                    self.training_stats['eval_steps'],
                    self.training_stats['eval_rewards'],
                    'ro-'
                )
                axes[1, 1].set_title('Evaluation Rewards')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = self.experiment_dir / "plots" / f"progress_episode_{self.current_episode}.png"
            plot_path.parent.mkdir(exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.debug(f"Progress plot saved: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to plot progress: {e}")
    
    def _save_training_stats(self):
        """Save training statistics to file."""
        stats_path = self.experiment_dir / "training_stats.yaml"
        
        # Convert numpy arrays to lists for YAML serialization
        stats_dict = {}
        for key, values in self.training_stats.items():
            if values:
                stats_dict[key] = [float(v) for v in values]
            else:
                stats_dict[key] = []
        
        # Add metadata
        stats_dict['metadata'] = {
            'total_episodes': self.current_episode,
            'total_timesteps': self.total_timesteps,
            'best_eval_reward': float(self.best_eval_reward),
            'experiment_dir': str(self.experiment_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        save_config(stats_dict, stats_path)
        logger.info(f"Training statistics saved: {stats_path}")
    
    def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.environment:
                self.environment.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train CARLA DRL Agent")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--sim-config',
        type=str, 
        default='configs/sim.yaml',
        help='Path to simulation configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Validate configuration files
    if not Path(args.config).exists():
        print(f"Error: Training config file not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.sim_config).exists():
        print(f"Error: Simulation config file not found: {args.sim_config}")
        sys.exit(1)
    
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and run training manager
        trainer = TrainingManager(args.config, args.sim_config)
        trainer.train()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

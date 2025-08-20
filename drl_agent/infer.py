"""
DRL Agent Inference Module for CARLA Autonomous Driving

This module loads a trained PPO model and runs inference for autonomous
driving evaluation in the CARLA simulator.
"""
import os
import sys
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import yaml
import argparse

# ROS 2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# DRL imports
import torch
from stable_baselines3 import PPO
import gymnasium as gym

# Local imports
sys.path.append(str(Path(__file__).parent.parent / "configs"))
from config_models import AlgorithmConfig, load_train_config
from train import CarlaROS2Environment, CNNFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarlaInferenceAgent:
    """
    Inference agent for running trained DRL policies in CARLA.
    
    Loads a trained PPO model and executes autonomous driving in the
    CARLA simulator via ROS 2 communication.
    """
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize inference agent.
        
        Args:
            model_path: Path to trained PPO model
            config_path: Path to training configuration file
        """
        self.model_path = Path(model_path)
        self.config = load_train_config(config_path)
        
        # Load trained model
        self.model = self._load_model()
        
        # Create environment for inference
        self.env = CarlaROS2Environment(self.config)
        
        # Statistics tracking
        self.episode_stats = []
        self.current_episode_stats = {
            'episode_length': 0,
            'episode_return': 0.0,
            'collisions': 0,
            'lane_invasions': 0,
            'max_speed': 0.0,
            'avg_speed': 0.0,
            'speed_samples': []
        }
        
        logger.info(f"Inference agent initialized with model: {model_path}")
        
    def _load_model(self) -> PPO:
        """Load trained PPO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        try:
            # Load model with custom feature extractor
            model = PPO.load(
                str(self.model_path),
                custom_objects={
                    'policy_class': 'MultiInputPolicy',
                    'features_extractor_class': CNNFeatureExtractor
                }
            )
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def _reset_episode_stats(self) -> None:
        """Reset statistics for new episode."""
        self.current_episode_stats = {
            'episode_length': 0,
            'episode_return': 0.0,
            'collisions': 0,
            'lane_invasions': 0,
            'max_speed': 0.0,
            'avg_speed': 0.0,
            'speed_samples': []
        }
        
    def _update_episode_stats(self, obs: Dict, reward: float, info: Dict) -> None:
        """Update episode statistics."""
        self.current_episode_stats['episode_length'] += 1
        self.current_episode_stats['episode_return'] += reward
        
        # Track collisions and lane invasions
        if info.get('collision', False):
            self.current_episode_stats['collisions'] += 1
            
        if info.get('lane_invasion', False):
            self.current_episode_stats['lane_invasions'] += 1
            
        # Track speed statistics
        speed = np.linalg.norm(obs['velocity'])
        self.current_episode_stats['speed_samples'].append(speed)
        self.current_episode_stats['max_speed'] = max(
            self.current_episode_stats['max_speed'], speed
        )
        
    def _finalize_episode_stats(self) -> Dict[str, float]:
        """Calculate final episode statistics."""
        stats = self.current_episode_stats.copy()
        
        # Calculate average speed
        if stats['speed_samples']:
            stats['avg_speed'] = np.mean(stats['speed_samples'])
        else:
            stats['avg_speed'] = 0.0
            
        # Remove speed samples for cleaner output
        del stats['speed_samples']
        
        return stats
        
    def run_episode(self, deterministic: bool = True, render: bool = False) -> Dict[str, float]:
        """
        Run a single episode with the trained agent.
        
        Args:
            deterministic: Whether to use deterministic policy
            render: Whether to render the environment (not implemented)
            
        Returns:
            Dictionary with episode statistics
        """
        logger.info("Starting new episode...")
        self._reset_episode_stats()
        
        # Reset environment
        obs, info = self.env.reset()
        done = False
        
        try:
            while not done:
                # Get action from trained model
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Execute action
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Update statistics
                self._update_episode_stats(obs, reward, info)
                
                # Log progress periodically
                if self.current_episode_stats['episode_length'] % 100 == 0:
                    logger.info(f"Episode step: {self.current_episode_stats['episode_length']}, "
                               f"Return: {self.current_episode_stats['episode_return']:.2f}")
                               
        except KeyboardInterrupt:
            logger.info("Episode interrupted by user")
            done = True
            
        except Exception as e:
            logger.error(f"Error during episode execution: {e}")
            done = True
            
        # Finalize statistics
        episode_stats = self._finalize_episode_stats()
        self.episode_stats.append(episode_stats)
        
        # Log episode summary
        logger.info(f"Episode completed:")
        logger.info(f"  Length: {episode_stats['episode_length']} steps")
        logger.info(f"  Return: {episode_stats['episode_return']:.2f}")
        logger.info(f"  Collisions: {episode_stats['collisions']}")
        logger.info(f"  Lane invasions: {episode_stats['lane_invasions']}")
        logger.info(f"  Max speed: {episode_stats['max_speed']:.2f} m/s")
        logger.info(f"  Avg speed: {episode_stats['avg_speed']:.2f} m/s")
        
        return episode_stats
        
    def run_evaluation(self, n_episodes: int = 5, deterministic: bool = True) -> Dict[str, float]:
        """
        Run multi-episode evaluation.
        
        Args:
            n_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with aggregated statistics
        """
        logger.info(f"Starting evaluation with {n_episodes} episodes...")
        
        self.episode_stats = []
        
        try:
            for episode in range(n_episodes):
                logger.info(f"Episode {episode + 1}/{n_episodes}")
                self.run_episode(deterministic=deterministic)
                
                # Brief pause between episodes
                time.sleep(2.0)
                
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            
        # Calculate aggregated statistics
        if self.episode_stats:
            aggregated_stats = self._calculate_aggregated_stats()
            self._log_evaluation_summary(aggregated_stats)
            return aggregated_stats
        else:
            logger.warning("No episodes completed")
            return {}
            
    def _calculate_aggregated_stats(self) -> Dict[str, float]:
        """Calculate aggregated statistics across all episodes."""
        if not self.episode_stats:
            return {}
            
        stats_arrays = {
            'episode_length': [ep['episode_length'] for ep in self.episode_stats],
            'episode_return': [ep['episode_return'] for ep in self.episode_stats],
            'collisions': [ep['collisions'] for ep in self.episode_stats],
            'lane_invasions': [ep['lane_invasions'] for ep in self.episode_stats],
            'max_speed': [ep['max_speed'] for ep in self.episode_stats],
            'avg_speed': [ep['avg_speed'] for ep in self.episode_stats]
        }
        
        aggregated = {}
        for key, values in stats_arrays.items():
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
            
        # Calculate success rate (episodes without collisions)
        collision_free_episodes = sum(1 for ep in self.episode_stats if ep['collisions'] == 0)
        aggregated['success_rate'] = collision_free_episodes / len(self.episode_stats)
        
        # Calculate safety rate (episodes without collisions or lane invasions)
        safe_episodes = sum(1 for ep in self.episode_stats 
                           if ep['collisions'] == 0 and ep['lane_invasions'] == 0)
        aggregated['safety_rate'] = safe_episodes / len(self.episode_stats)
        
        return aggregated
        
    def _log_evaluation_summary(self, stats: Dict[str, float]) -> None:
        """Log evaluation summary."""
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Episodes completed: {len(self.episode_stats)}")
        logger.info(f"Success rate (no collisions): {stats['success_rate']:.2%}")
        logger.info(f"Safety rate (no violations): {stats['safety_rate']:.2%}")
        logger.info(f"Average episode return: {stats['episode_return_mean']:.2f} ± {stats['episode_return_std']:.2f}")
        logger.info(f"Average episode length: {stats['episode_length_mean']:.1f} ± {stats['episode_length_std']:.1f}")
        logger.info(f"Average speed: {stats['avg_speed_mean']:.2f} ± {stats['avg_speed_std']:.2f} m/s")
        logger.info(f"Collisions per episode: {stats['collisions_mean']:.2f} ± {stats['collisions_std']:.2f}")
        logger.info(f"Lane invasions per episode: {stats['lane_invasions_mean']:.2f} ± {stats['lane_invasions_std']:.2f}")
        logger.info("=" * 50)
        
    def close(self) -> None:
        """Close environment and cleanup resources."""
        if hasattr(self, 'env'):
            self.env.close()
        logger.info("Inference agent closed")


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description='Run inference with trained CARLA DRL agent')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained PPO model (.zip file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/train.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic policy (default: True for inference)'
    )
    parser.add_argument(
        '--single-episode',
        action='store_true',
        help='Run only a single episode'
    )
    
    args = parser.parse_args()
    
    try:
        # Create inference agent
        agent = CarlaInferenceAgent(args.model, args.config)
        
        if args.single_episode:
            # Run single episode
            episode_stats = agent.run_episode(deterministic=True)
            
        else:
            # Run evaluation
            evaluation_stats = agent.run_evaluation(
                n_episodes=args.episodes,
                deterministic=True
            )
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
        
    finally:
        if 'agent' in locals():
            agent.close()
            
    return 0


if __name__ == "__main__":
    sys.exit(main())

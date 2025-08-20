#!/usr/bin/env python3
"""
Enhanced PPO Training Pipeline for CARLA DRL.
Builds upon existing SAC implementation with production-grade PPO, curriculum learning,
and comprehensive monitoring. Integrates with existing module_7.py and rl_environment.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import pickle
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_ppo_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import existing components
try:
    from carla_env import CarlaEnv, CarlaEnvConfig
    from feature_extractors import CNNFeatureExtractor, MultimodalFeatureExtractor
    from ros_bridge import DRLBridge
    logger.info("Successfully imported existing RL environment components")
except ImportError as e:
    logger.error(f"Failed to import RL environment components: {e}")
    logger.error("Make sure all required modules are in the Python path")

# Import advanced configuration
try:
    from configs.advanced_config_models import AdvancedPipelineConfig, load_config
    logger.info("Successfully imported advanced configuration models")
except ImportError as e:
    logger.warning(f"Failed to import advanced config models: {e}")
    logger.warning("Using fallback configuration loading")

@dataclass
class TrainingMetrics:
    """Container for training metrics and statistics."""
    episode: int = 0
    total_timesteps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    success_rate: float = 0.0
    collision_rate: float = 0.0
    lane_invasion_rate: float = 0.0
    average_speed: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    explained_variance: float = 0.0
    learning_rate: float = 0.0
    curriculum_stage: str = "unknown"
    stage_progress: float = 0.0

class CurriculumManager:
    """Manages curriculum learning progression and stage transitions."""
    
    def __init__(self, curriculum_config: Dict[str, Any]):
        """Initialize curriculum manager with configuration."""
        self.config = curriculum_config
        self.stages = curriculum_config.get('stages', [])
        self.current_stage_idx = 0
        self.current_stage = self.stages[0] if self.stages else {}
        self.stage_episodes = 0
        self.stage_successes = 0
        self.stage_metrics = deque(maxlen=100)
        
        # Progression settings
        self.auto_progression = curriculum_config.get('auto_progression', True)
        self.progression_threshold = curriculum_config.get('progression_threshold', 0.8)
        self.regression_threshold = curriculum_config.get('regression_threshold', 0.3)
        self.evaluation_episodes = curriculum_config.get('stage_evaluation_episodes', 50)
        
        # Dynamic difficulty
        self.dynamic_difficulty = curriculum_config.get('dynamic_difficulty', True)
        self.difficulty_window = curriculum_config.get('difficulty_window', 100)
        self.adaptation_rate = curriculum_config.get('difficulty_adaptation_rate', 0.1)
        
        logger.info(f"Initialized curriculum with {len(self.stages)} stages")
        logger.info(f"Starting with stage: {self.current_stage.get('name', 'unknown')}")
    
    def update_stage_metrics(self, success: bool, reward: float, metrics: Dict[str, float]) -> None:
        """Update metrics for current curriculum stage."""
        self.stage_episodes += 1
        if success:
            self.stage_successes += 1
        
        # Store detailed metrics
        stage_metric = {
            'success': success,
            'reward': reward,
            'collision': metrics.get('collision', False),
            'lane_invasion': metrics.get('lane_invasion', False),
            'speed': metrics.get('average_speed', 0.0)
        }
        self.stage_metrics.append(stage_metric)
        
        # Log stage progress
        if self.stage_episodes % 10 == 0:
            success_rate = self.stage_successes / self.stage_episodes
            logger.info(f"Stage '{self.current_stage.get('name')}' progress: "
                       f"{self.stage_episodes}/{self.current_stage.get('episodes')} episodes, "
                       f"success rate: {success_rate:.2%}")
    
    def should_progress_stage(self) -> bool:
        """Check if conditions are met to progress to next stage."""
        if not self.auto_progression or self.current_stage_idx >= len(self.stages) - 1:
            return False
        
        # Check minimum episodes completed
        min_episodes = max(self.evaluation_episodes, 
                          self.current_stage.get('episodes', 1000) // 10)
        if self.stage_episodes < min_episodes:
            return False
        
        # Calculate success rate over evaluation window
        recent_metrics = list(self.stage_metrics)[-self.evaluation_episodes:]
        if len(recent_metrics) < self.evaluation_episodes:
            return False
        
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        
        # Check progression threshold
        threshold = self.current_stage.get('success_threshold', self.progression_threshold)
        return success_rate >= threshold
    
    def should_regress_stage(self) -> bool:
        """Check if conditions are met to regress to previous stage."""
        if self.current_stage_idx == 0:
            return False
        
        # Only check after sufficient episodes
        if self.stage_episodes < self.evaluation_episodes:
            return False
        
        recent_metrics = list(self.stage_metrics)[-self.evaluation_episodes:]
        if len(recent_metrics) < self.evaluation_episodes:
            return False
        
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        return success_rate < self.regression_threshold
    
    def progress_to_next_stage(self) -> bool:
        """Progress to the next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.current_stage = self.stages[self.current_stage_idx]
            self.stage_episodes = 0
            self.stage_successes = 0
            self.stage_metrics.clear()
            
            logger.info(f"Progressed to stage {self.current_stage_idx + 1}: "
                       f"'{self.current_stage.get('name')}'")
            return True
        return False
    
    def regress_to_previous_stage(self) -> bool:
        """Regress to the previous curriculum stage."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            self.current_stage = self.stages[self.current_stage_idx]
            self.stage_episodes = 0
            self.stage_successes = 0
            self.stage_metrics.clear()
            
            logger.warning(f"Regressed to stage {self.current_stage_idx + 1}: "
                          f"'{self.current_stage.get('name')}'")
            return True
        return False
    
    def get_current_stage_config(self) -> Dict[str, Any]:
        """Get configuration for current curriculum stage."""
        return self.current_stage.copy()
    
    def get_stage_progress(self) -> float:
        """Get progress through current stage (0.0 to 1.0)."""
        total_episodes = self.current_stage.get('episodes', 1000)
        return min(1.0, self.stage_episodes / total_episodes)

class EnhancedPPOPolicy(nn.Module):
    """Enhanced PPO policy network with attention and multi-modal input support."""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 features_extractor_config: Dict[str, Any],
                 net_arch: List[int] = [256, 256],
                 activation_fn: str = "relu",
                 ortho_init: bool = True):
        """Initialize enhanced PPO policy network."""
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.net_arch = net_arch
        
        # Activation function
        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation_fn == "elu":
            self.activation_fn = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        
        # Feature extractor
        self.features_extractor = self._create_features_extractor(features_extractor_config)
        features_dim = self.features_extractor.features_dim
        
        # Policy network (actor)
        policy_layers = []
        input_dim = features_dim
        
        for hidden_dim in net_arch:
            policy_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn,
                nn.Dropout(0.1) if features_extractor_config.get('dropout', 0.0) > 0 else nn.Identity()
            ])
            input_dim = hidden_dim
        
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Action head
        if hasattr(action_space, 'n'):  # Discrete action space
            self.action_head = nn.Linear(input_dim, action_space.n)
            self.action_type = "discrete"
        else:  # Continuous action space
            self.action_mean = nn.Linear(input_dim, action_space.shape[0])
            self.action_log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
            self.action_type = "continuous"
        
        # Value network (critic)
        value_layers = []
        input_dim = features_dim
        
        for hidden_dim in net_arch:
            value_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn,
                nn.Dropout(0.1) if features_extractor_config.get('dropout', 0.0) > 0 else nn.Identity()
            ])
            input_dim = hidden_dim
        
        self.value_net = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        if ortho_init:
            self._init_weights()
    
    def _create_features_extractor(self, config: Dict[str, Any]):
        """Create feature extractor based on configuration."""
        extractor_type = config.get('type', 'multimodal_cnn')
        
        if extractor_type == "multimodal_cnn":
            return MultimodalFeatureExtractor(
                observation_space=self.observation_space,
                features_dim=config.get('cnn_features_dim', 512),
                normalize_images=config.get('normalize_images', True),
                attention_mechanism=config.get('attention_mechanism', True),
                temporal_context=config.get('temporal_context', 4)
            )
        elif extractor_type == "cnn":
            return CNNFeatureExtractor(
                observation_space=self.observation_space,
                features_dim=config.get('cnn_features_dim', 512)
            )
        else:
            raise ValueError(f"Unsupported feature extractor type: {extractor_type}")
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations):
        """Forward pass through the network."""
        features = self.features_extractor(observations)
        
        # Policy forward
        policy_features = self.policy_net(features)
        
        # Value forward
        value_features = self.value_net(features)
        value = self.value_head(value_features)
        
        if self.action_type == "discrete":
            action_logits = self.action_head(policy_features)
            return action_logits, value
        else:
            action_mean = self.action_mean(policy_features)
            action_std = torch.exp(self.action_log_std)
            return action_mean, action_std, value
    
    def get_action_distribution(self, observations):
        """Get action distribution for given observations."""
        if self.action_type == "discrete":
            logits, _ = self.forward(observations)
            return Categorical(logits=logits)
        else:
            mean, std, _ = self.forward(observations)
            return Normal(mean, std)
    
    def evaluate_actions(self, observations, actions):
        """Evaluate actions for given observations."""
        dist = self.get_action_distribution(observations)
        
        if self.action_type == "discrete":
            _, values = self.forward(observations)
        else:
            _, _, values = self.forward(observations)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_probs, entropy

class EnhancedPPOTrainer:
    """Enhanced PPO trainer with curriculum learning and comprehensive monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced PPO trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        seed = config.get('pipeline', {}).get('random_seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize curriculum manager
        curriculum_config = config.get('curriculum', {})
        self.curriculum_manager = CurriculumManager(curriculum_config) if curriculum_config.get('enabled', False) else None
        
        # Initialize policy
        self.policy = self._create_policy()
        self.optimizer = self._create_optimizer()
        
        # Training parameters
        training_config = config.get('training', {})
        ppo_config = training_config.get('ppo', {})
        
        self.n_steps = ppo_config.get('n_steps', 2048)
        self.n_epochs = ppo_config.get('n_epochs', 10)
        self.batch_size = training_config.get('batch_size', 64)
        self.gamma = ppo_config.get('gamma', 0.99)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.clip_range = ppo_config.get('clip_range', 0.2)
        self.ent_coef = ppo_config.get('ent_coef', 0.01)
        self.vf_coef = ppo_config.get('vf_coef', 0.5)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        
        # Logging and monitoring
        self.tensorboard_writer = self._setup_tensorboard()
        self.metrics = TrainingMetrics()
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Model saving
        self.save_interval = training_config.get('checkpointing', {}).get('checkpoint_frequency', 10000)
        self.model_dir = Path("models/enhanced_ppo")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Enhanced PPO trainer initialized successfully")
    
    def _create_environment(self) -> CarlaEnv:
        """Create and configure CARLA environment."""
        carla_config = self.config.get('carla', {})
        
        # Create environment configuration
        env_config = CarlaEnvConfig(
            random_seed=self.config.get('pipeline', {}).get('random_seed', 42),
            timeout=carla_config.get('timeout', 10.0),
            use_image_observations=True,
            render=carla_config.get('monitoring', {}).get('enable_metrics', True),
            max_episode_steps=1000
        )
        
        return CarlaEnv(config=env_config)
    
    def _create_policy(self):
        """Create and initialize PPO policy."""
        training_config = self.config.get('training', {})
        
        policy = EnhancedPPOPolicy(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_extractor_config=training_config.get('features_extractor', {}),
            net_arch=training_config.get('policy_network', {}).get('net_arch', [256, 256]),
            activation_fn=training_config.get('policy_network', {}).get('activation_fn', "relu"),
            ortho_init=training_config.get('policy_network', {}).get('ortho_init', True)
        ).to(self.device)
        
        logger.info(f"Created PPO policy with {sum(p.numel() for p in policy.parameters())} parameters")
        return policy
    
    def _create_optimizer(self):
        """Create optimizer for policy training."""
        lr = self.config.get('training', {}).get('learning_rate', 3e-4)
        weight_decay = self.config.get('training', {}).get('regularization', {}).get('weight_decay', 1e-5)
        
        return optim.Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        log_dir = Path("logs/enhanced_ppo") / f"run_{int(time.time())}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return SummaryWriter(log_dir=str(log_dir))
    
    def collect_trajectories(self) -> Dict[str, torch.Tensor]:
        """Collect trajectories for PPO training."""
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs = self.env.reset()
        
        for step in range(self.n_steps):
            # Convert observation to tensor
            obs_tensor = self._obs_to_tensor(obs)
            
            with torch.no_grad():
                # Get action distribution and value
                dist = self.policy.get_action_distribution(obs_tensor)
                if self.policy.action_type == "discrete":
                    _, value = self.policy(obs_tensor)
                else:
                    _, _, value = self.policy(obs_tensor)
                
                # Sample action
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            
            # Store transition
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            # Update curriculum if enabled
            if self.curriculum_manager and done:
                success = info.get('success', False)
                self.curriculum_manager.update_stage_metrics(success, reward, info)
                
                # Check for stage transitions
                if self.curriculum_manager.should_progress_stage():
                    self.curriculum_manager.progress_to_next_stage()
                    self._update_environment_for_stage()
                elif self.curriculum_manager.should_regress_stage():
                    self.curriculum_manager.regress_to_previous_stage()
                    self._update_environment_for_stage()
            
            obs = next_obs if not done else self.env.reset()
        
        # Convert to tensors
        trajectories = {
            'observations': torch.stack(observations),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards, dtype=torch.float32, device=self.device),
            'values': torch.stack(values).squeeze(),
            'log_probs': torch.stack(log_probs),
            'dones': torch.tensor(dones, dtype=torch.bool, device=self.device)
        }
        
        return trajectories
    
    def compute_advantages(self, trajectories: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        rewards = trajectories['rewards']
        values = trajectories['values']
        dones = trajectories['dones']
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Assuming episode ends
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, trajectories: Dict[str, torch.Tensor], advantages: torch.Tensor, returns: torch.Tensor):
        """Update policy using PPO algorithm."""
        observations = trajectories['observations']
        actions = trajectories['actions']
        old_log_probs = trajectories['log_probs']
        old_values = trajectories['values']
        
        # Create dataset
        dataset_size = observations.shape[0]
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Evaluate current policy
                values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                values = values.squeeze()
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute policy loss (PPO clipped objective)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Update metrics
        self.metrics.policy_loss = np.mean(policy_losses)
        self.metrics.value_loss = np.mean(value_losses)
        self.metrics.entropy_loss = np.mean(entropy_losses)
    
    def train(self, total_timesteps: int):
        """Main training loop."""
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        timesteps = 0
        episode = 0
        
        while timesteps < total_timesteps:
            # Collect trajectories
            trajectories = self.collect_trajectories()
            timesteps += self.n_steps
            
            # Compute advantages
            advantages, returns = self.compute_advantages(trajectories)
            
            # Update policy
            self.update_policy(trajectories, advantages, returns)
            
            # Update metrics
            self.metrics.total_timesteps = timesteps
            self.metrics.episode = episode
            
            # Log to TensorBoard
            self._log_metrics()
            
            # Save model checkpoint
            if timesteps % self.save_interval == 0:
                self.save_model(timesteps)
            
            episode += 1
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}, Timesteps: {timesteps}, "
                           f"Avg Reward: {np.mean(self.episode_rewards):.2f}")
        
        logger.info("Training completed successfully")
        self.save_model(timesteps, is_final=True)
    
    def _obs_to_tensor(self, obs):
        """Convert observation to tensor."""
        if isinstance(obs, dict):
            tensor_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor_obs[key] = torch.from_numpy(value).float().to(self.device)
                else:
                    tensor_obs[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
            return tensor_obs
        else:
            return torch.from_numpy(obs).float().to(self.device)
    
    def _update_environment_for_stage(self):
        """Update environment configuration for current curriculum stage."""
        if not self.curriculum_manager:
            return
        
        stage_config = self.curriculum_manager.get_current_stage_config()
        
        # Update environment parameters based on stage
        # This would integrate with your existing CARLA configuration
        logger.info(f"Updated environment for stage: {stage_config.get('name')}")
    
    def _log_metrics(self):
        """Log metrics to TensorBoard."""
        if not self.tensorboard_writer:
            return
        
        timesteps = self.metrics.total_timesteps
        
        # Training metrics
        self.tensorboard_writer.add_scalar('Loss/Policy', self.metrics.policy_loss, timesteps)
        self.tensorboard_writer.add_scalar('Loss/Value', self.metrics.value_loss, timesteps)
        self.tensorboard_writer.add_scalar('Loss/Entropy', self.metrics.entropy_loss, timesteps)
        
        # Performance metrics
        if self.episode_rewards:
            self.tensorboard_writer.add_scalar('Performance/Episode_Reward', 
                                             np.mean(self.episode_rewards), timesteps)
            self.tensorboard_writer.add_scalar('Performance/Episode_Length',
                                             np.mean(self.episode_lengths), timesteps)
        
        # Curriculum metrics
        if self.curriculum_manager:
            self.tensorboard_writer.add_scalar('Curriculum/Stage', 
                                             self.curriculum_manager.current_stage_idx, timesteps)
            self.tensorboard_writer.add_scalar('Curriculum/Stage_Progress',
                                             self.curriculum_manager.get_stage_progress(), timesteps)
    
    def save_model(self, timesteps: int, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timesteps': timesteps,
            'config': self.config,
            'metrics': self.metrics
        }
        
        if is_final:
            save_path = self.model_dir / "final_model.pth"
        else:
            save_path = self.model_dir / f"checkpoint_{timesteps}.pth"
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved model checkpoint to {save_path}")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced PPO Training for CARLA DRL")
    parser.add_argument("--config", type=str, 
                       default="configs/advanced_pipeline_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--timesteps", type=int, default=2000000,
                       help="Total training timesteps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Create trainer
    trainer = EnhancedPPOTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        # Implement checkpoint loading
        logger.info(f"Resuming training from {args.resume}")
    
    # Start training
    trainer.train(args.timesteps)

if __name__ == "__main__":
    main()

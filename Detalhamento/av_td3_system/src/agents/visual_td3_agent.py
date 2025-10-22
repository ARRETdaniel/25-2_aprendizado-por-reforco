"""
Visual TD3 Agent with CNN Feature Extraction

Extension of TD3Agent to handle Dict observations with visual input.
Integrates CNN feature extractor for end-to-end visual navigation.

Architecture:
    Dict observation {'image': (4,84,84), 'vector': (23,)}
        ↓
    CNN Feature Extractor: image (4,84,84) → features (512,)
        ↓
    Concatenate: features (512,) + vector (23,) → state (535,)
        ↓
    TD3 Actor/Critic: state (535,) → action (2,)
"""

import torch
import torch.nn as nn
from typing import Dict, Union, Tuple
import numpy as np

from src.agents.td3_agent import TD3Agent
from src.networks.cnn_extractor import get_cnn_extractor


class VisualTD3Agent(TD3Agent):
    """
    TD3 Agent with integrated CNN for visual navigation.
    
    Extends TD3Agent to process Dict observations containing:
    - 'image': (4, 84, 84) stacked grayscale frames
    - 'vector': (23,) kinematic + waypoint features
    
    The CNN extracts 512-dim features from images, which are concatenated
    with the 23-dim vector to form the 535-dim state for TD3.
    """
    
    def __init__(
        self,
        state_dim: int = 535,
        action_dim: int = 2,
        max_action: float = 1.0,
        config=None,
        config_path=None,
        cnn_architecture: str = "mobilenet",
        pretrained_cnn: bool = True,
        freeze_cnn: bool = False
    ):
        """
        Initialize Visual TD3 agent.
        
        Args:
            state_dim: Total state dimension (default: 535)
            action_dim: Action dimension (default: 2)
            max_action: Maximum action value (default: 1.0)
            config: Config dict (if None, loads from config_path)
            config_path: Path to config file
            cnn_architecture: CNN type ('nature', 'mobilenet', 'resnet18')
            pretrained_cnn: Use pretrained weights for transfer learning
            freeze_cnn: Freeze CNN backbone initially
        """
        # Initialize base TD3 agent
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            config=config,
            config_path=config_path
        )
        
        # Initialize CNN feature extractor
        self.cnn = get_cnn_extractor(
            architecture=cnn_architecture,
            input_channels=4,  # 4 stacked grayscale frames
            output_dim=512,  # Output feature dimension
            pretrained=pretrained_cnn,
            freeze_backbone=freeze_cnn
        ).to(self.device)
        
        # CNN optimizer
        self.cnn_optimizer = torch.optim.Adam(
            self.cnn.parameters(),
            lr=self.actor_lr  # Use same LR as actor
        )
        
        self.cnn_architecture = cnn_architecture
        self.freeze_cnn = freeze_cnn
        
        print(f"\n📹 Visual TD3 Agent Initialized:")
        print(f"  CNN Architecture: {cnn_architecture}")
        print(f"  Pretrained: {pretrained_cnn}")
        print(f"  Frozen: {freeze_cnn}")
        print(f"  CNN Parameters: {sum(p.numel() for p in self.cnn.parameters()):,}")
        print(f"  Total Parameters: {self.get_total_parameters():,}")
    
    def process_observation(
        self,
        obs: Union[Dict, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process observation into image tensor and vector tensor.
        
        Args:
            obs: Either Dict with 'image' and 'vector' keys, or flat numpy array
            
        Returns:
            Tuple of (image_tensor, vector_tensor)
        """
        if isinstance(obs, dict):
            # Dict observation from environment
            image = obs['image']  # Shape: (4, 84, 84)
            vector = obs['vector']  # Shape: (23,)
            
            # Convert to tensors
            image_tensor = torch.FloatTensor(image).unsqueeze(0).to(self.device)  # (1, 4, 84, 84)
            vector_tensor = torch.FloatTensor(vector).unsqueeze(0).to(self.device)  # (1, 23)
            
        else:
            # Flat numpy array (for compatibility with existing code)
            # Assume first 512 dims are image features, last 23 are vector
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Split back into image features and vector
            # Note: This assumes the flat state was created by flattening CNN output
            # For true end-to-end, use Dict observations
            image_features = obs_tensor[:, :512]
            vector_tensor = obs_tensor[:, 512:]
            
            # Create dummy image tensor (won't be used by CNN)
            image_tensor = None
        
        return image_tensor, vector_tensor
    
    def extract_features(
        self,
        obs: Union[Dict, np.ndarray],
        batch: bool = False
    ) -> torch.Tensor:
        """
        Extract state features from observation using CNN.
        
        Args:
            obs: Observation (Dict or flat array)
            batch: Whether obs is a batch
            
        Returns:
            State tensor of shape (batch_size, 535) or (535,)
        """
        if isinstance(obs, dict):
            # Dict observation - use CNN
            if batch:
                # Batch of observations
                images = torch.FloatTensor(np.stack([o['image'] for o in obs])).to(self.device)
                vectors = torch.FloatTensor(np.stack([o['vector'] for o in obs])).to(self.device)
            else:
                # Single observation
                images = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
                vectors = torch.FloatTensor(obs['vector']).unsqueeze(0).to(self.device)
            
            # Extract CNN features
            with torch.no_grad():
                cnn_features = self.cnn(images)  # (batch, 512)
            
            # Concatenate with vector features
            state = torch.cat([cnn_features, vectors], dim=1)  # (batch, 535)
            
        else:
            # Flat array - already preprocessed
            if batch:
                state = torch.FloatTensor(obs).to(self.device)
            else:
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        return state
    
    def select_action(
        self,
        obs: Union[Dict, np.ndarray],
        noise: float = 0.0
    ) -> np.ndarray:
        """
        Select action using actor network with optional exploration noise.
        
        Args:
            obs: Observation (Dict or flat array)
            noise: Exploration noise std (0.0 = deterministic)
            
        Returns:
            Action array of shape (action_dim,)
        """
        # Extract state features
        state = self.extract_features(obs, batch=False)
        
        # Get action from actor
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add exploration noise if specified
        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train(self, replay_buffer) -> Dict[str, float]:
        """
        Perform one training step on a batch from replay buffer.
        
        Extends base TD3 train() to include CNN feature extraction.
        
        Args:
            replay_buffer: Experience replay buffer
            
        Returns:
            Dict with training metrics
        """
        self.total_it += 1
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample(self.batch_size)
        state_obs = batch['state']
        action = batch['action']
        next_state_obs = batch['next_state']
        reward = batch['reward']
        not_done = batch['not_done']
        
        # Extract features for current and next states
        # Check if observations are Dict type
        if isinstance(state_obs[0], dict):
            # Use CNN for feature extraction
            state_images = torch.FloatTensor(
                np.stack([s['image'] for s in state_obs])
            ).to(self.device)
            state_vectors = torch.FloatTensor(
                np.stack([s['vector'] for s in state_obs])
            ).to(self.device)
            
            next_state_images = torch.FloatTensor(
                np.stack([s['image'] for s in next_state_obs])
            ).to(self.device)
            next_state_vectors = torch.FloatTensor(
                np.stack([s['vector'] for s in next_state_obs])
            ).to(self.device)
            
            # Extract CNN features (with gradients for training)
            state_cnn_features = self.cnn(state_images)
            state = torch.cat([state_cnn_features, state_vectors], dim=1)
            
            with torch.no_grad():
                next_state_cnn_features = self.cnn(next_state_images)
                next_state = torch.cat([next_state_cnn_features, next_state_vectors], dim=1)
        else:
            # Already flat arrays
            state = torch.FloatTensor(state_obs).to(self.device)
            next_state = torch.FloatTensor(next_state_obs).to(self.device)
        
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-value (minimum of twin critics)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss (MSE)
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + \
                      torch.nn.functional.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        if isinstance(state_obs[0], dict):
            # Also zero CNN optimizer if training end-to-end
            self.cnn_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if isinstance(state_obs[0], dict) and not self.freeze_cnn:
            self.cnn_optimizer.step()
        
        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        # Return training metrics
        metrics = {
            'critic_loss': critic_loss.item(),
            'q1_value': current_Q1.mean().item(),
            'q2_value': current_Q2.mean().item()
        }
        
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss.item()
        
        return metrics
    
    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning."""
        if hasattr(self.cnn, 'unfreeze_backbone'):
            self.cnn.unfreeze_backbone()
        self.freeze_cnn = False
        print("🔓 CNN backbone unfrozen for fine-tuning")
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters including CNN."""
        total = super().get_total_parameters()
        total += sum(p.numel() for p in self.cnn.parameters())
        return total
    
    def save_checkpoint(self, filepath: str):
        """
        Save agent checkpoint including CNN.
        
        Args:
            filepath: Path to save checkpoint
        """
        # Call parent save
        super().save_checkpoint(filepath)
        
        # Save CNN separately
        cnn_path = filepath.replace('.pth', '_cnn.pth')
        torch.save({
            'cnn_state_dict': self.cnn.state_dict(),
            'cnn_optimizer_state_dict': self.cnn_optimizer.state_dict(),
            'cnn_architecture': self.cnn_architecture,
            'freeze_cnn': self.freeze_cnn
        }, cnn_path)
        
        print(f"💾 CNN checkpoint saved: {cnn_path}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load agent checkpoint including CNN.
        
        Args:
            filepath: Path to checkpoint
        """
        # Load parent checkpoint
        super().load_checkpoint(filepath)
        
        # Load CNN checkpoint
        cnn_path = filepath.replace('.pth', '_cnn.pth')
        checkpoint = torch.load(cnn_path, map_location=self.device)
        
        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.cnn_optimizer.load_state_dict(checkpoint['cnn_optimizer_state_dict'])
        self.cnn_architecture = checkpoint['cnn_architecture']
        self.freeze_cnn = checkpoint['freeze_cnn']
        
        print(f"📂 CNN checkpoint loaded: {cnn_path}")

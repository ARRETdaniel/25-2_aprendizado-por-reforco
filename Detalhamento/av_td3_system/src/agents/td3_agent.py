"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

Implements the TD3 algorithm for continuous control in autonomous driving.
TD3 improves upon DDPG with three key mechanisms:
1. Twin Critics: Uses two Q-networks and takes minimum for target value
2. Delayed Policy Updates: Updates actor less frequently than critics
3. Target Policy Smoothing: Adds noise to target actions for regularization

Reference: "Addressing Function Approximation Error in Actor-Critic Methods"
           (Fujimoto et al., ICML 2018)

Author: Daniel Terra
Date: 2024
"""

import copy
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.networks.actor import Actor
from src.networks.critic import TwinCritic
from src.utils.replay_buffer import ReplayBuffer
from src.utils.dict_replay_buffer import DictReplayBuffer


class TD3Agent:
    """
    TD3 agent for autonomous vehicle control.

    The agent learns a deterministic policy that maps states (visual features +
    kinematic data + waypoints) to continuous actions (steering + throttle/brake).

    Key Attributes:
        actor: Policy network μ_φ(s)
        actor_target: Target policy for stable learning
        critic: Twin Q-networks Q_θ1(s,a) and Q_θ2(s,a)
        critic_target: Target Q-networks for stable learning
        replay_buffer: Experience replay buffer
        total_it: Total training iterations (for delayed policy updates)
    """

    def __init__(
        self,
        state_dim: int = 535,
        action_dim: int = 2,
        max_action: float = 1.0,
        cnn_extractor: Optional[torch.nn.Module] = None,  # DEPRECATED: Use actor_cnn/critic_cnn instead
        actor_cnn: Optional[torch.nn.Module] = None,  # 🔧 FIX: Separate CNN for actor
        critic_cnn: Optional[torch.nn.Module] = None,  # 🔧 FIX: Separate CNN for critic
        use_dict_buffer: bool = True,  # Use DictReplayBuffer for gradient-enabled training
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize TD3 agent with networks and hyperparameters.

        Args:
            state_dim: Dimension of state space (default: 535)
                      = 512 (CNN features) + 3 (kinematic) + 20 (waypoints)
            action_dim: Dimension of action space (default: 2)
                       = [steering, throttle/brake]
            max_action: Maximum absolute value of actions (default: 1.0)
            cnn_extractor: DEPRECATED - Use actor_cnn/critic_cnn for separate CNNs
            actor_cnn: CNN feature extractor for actor network (end-to-end training)
                      🔧 FIX: Separate CNN prevents gradient interference
            critic_cnn: CNN feature extractor for critic network (end-to-end training)
                       🔧 FIX: Actor and critic optimize their own CNNs independently
            use_dict_buffer: If True, use DictReplayBuffer for gradient flow
                            If False, use standard ReplayBuffer (no CNN training)
            config: Dictionary with TD3 hyperparameters (if None, loads from file)
            config_path: Path to YAML config file (default: config/td3_config.yaml)
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.

        Note:
            🔧 CRITICAL FIX: TD3 now uses SEPARATE CNNs for actor and critic.
            This prevents gradient interference that was causing training failure.
            Reference: Stable-Baselines3 TD3 uses share_features_extractor=False
        """
        # Load configuration
        if config is None:
            if config_path is None:
                config_path = "config/td3_config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        # Store dimensions and config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.config = config

        # Extract hyperparameters from config
        algo_config = config['algorithm']
        self.discount = algo_config.get('gamma', algo_config.get('discount', 0.99))  # γ discount factor
        self.tau = algo_config['tau']  # Soft update rate for target networks
        self.policy_noise = algo_config['policy_noise']  # Noise for target smoothing
        self.noise_clip = algo_config['noise_clip']  # Clip range for target noise
        self.policy_freq = algo_config['policy_freq']  # Delayed policy update frequency
        self.actor_lr = algo_config.get('actor_lr', algo_config.get('learning_rate', 0.0003))
        self.critic_lr = algo_config.get('critic_lr', algo_config.get('learning_rate', 0.0003))

        # Training config
        training_config = config.get('training', config.get('algorithm', {}))
        self.batch_size = training_config.get('batch_size', 256)
        self.buffer_size = training_config.get('buffer_size', 1000000)
        self.start_timesteps = training_config.get('start_timesteps', training_config.get('learning_starts', 25000))

        # Exploration config (handle both nested and flat structures)
        # NOTE: expl_noise stored for reference but not used in select_action
        # Noise is passed explicitly by training loop with exponential decay schedule
        exploration_config = config.get('exploration', {})
        self.expl_noise = exploration_config.get('expl_noise', algo_config.get('exploration_noise', 0.1))

        # Set device
        if device is not None:
            self.device = torch.device(device)
            print(f"TD3Agent initialized on device: {self.device} (manually specified)")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"TD3Agent initialized on device: {self.device}")

        # Initialize actor networks
        network_config = config.get('networks', {}).get('actor', {})
        hidden_layers = network_config.get('hidden_sizes', network_config.get('hidden_layers', [256, 256]))
        # Extract hidden_size from list (Actor uses fixed 3-layer architecture with same width)
        hidden_size = hidden_layers[0] if isinstance(hidden_layers, list) else hidden_layers
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_size=hidden_size
        ).to(self.device)

        # Create target actor as deep copy
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr
        )

        # Initialize twin critic networks
        critic_config = config.get('networks', {}).get('critic', {})
        hidden_layers = critic_config.get('hidden_sizes', critic_config.get('hidden_layers', [256, 256]))
        # Extract hidden_size from list (Critic uses fixed 3-layer architecture with same width)
        hidden_size = hidden_layers[0] if isinstance(hidden_layers, list) else hidden_layers
        self.critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size
        ).to(self.device)

        # Create target critics as deep copy
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr
        )

        # Initialize CNN for end-to-end training (Bug #13 fix + CRITICAL gradient interference fix)
        # 🔧 FIX: Support backward compatibility with cnn_extractor parameter
        if cnn_extractor is not None and (actor_cnn is None or critic_cnn is None):
            print("  WARNING: cnn_extractor parameter is DEPRECATED!")
            print("  Using cnn_extractor for both actor and critic (NOT RECOMMENDED)")
            print("  This can cause gradient interference. Use actor_cnn/critic_cnn instead.")
            if actor_cnn is None:
                actor_cnn = cnn_extractor
            if critic_cnn is None:
                critic_cnn = cnn_extractor

        self.actor_cnn = actor_cnn
        self.critic_cnn = critic_cnn
        self.use_dict_buffer = use_dict_buffer

        # Initialize CNN optimizers (separate for actor and critic)
        if self.actor_cnn is not None:
            # Actor CNN should be in training mode
            self.actor_cnn.train()

            # Create actor CNN optimizer
            cnn_config = config.get('networks', {}).get('cnn', {})
            cnn_lr = cnn_config.get('learning_rate', 1e-4)  # Conservative 1e-4 for CNN
            self.actor_cnn_optimizer = torch.optim.Adam(
                self.actor_cnn.parameters(),
                lr=cnn_lr
            )
            print(f"  Actor CNN optimizer initialized with lr={cnn_lr}")
            print(f"  Actor CNN mode: training (gradients enabled)")
        else:
            self.actor_cnn_optimizer = None

        if self.critic_cnn is not None:
            # Critic CNN should be in training mode
            self.critic_cnn.train()

            # Create critic CNN optimizer (may share same CNN config)
            cnn_config = config.get('networks', {}).get('cnn', {})
            cnn_lr = cnn_config.get('learning_rate', 1e-4)
            self.critic_cnn_optimizer = torch.optim.Adam(
                self.critic_cnn.parameters(),
                lr=cnn_lr
            )
            print(f"  Critic CNN optimizer initialized with lr={cnn_lr}")
            print(f"  Critic CNN mode: training (gradients enabled)")
        else:
            self.critic_cnn_optimizer = None

        # Validation checks
        if use_dict_buffer and (self.actor_cnn is None or self.critic_cnn is None):
            print("  ⚠️  WARNING: DictReplayBuffer enabled but CNN(s) missing!")
            if self.actor_cnn is None:
                print("      actor_cnn is None - actor will use zero features")
            if self.critic_cnn is None:
                print("      critic_cnn is None - critic will use zero features")

        # Check if same CNN instance is shared (not recommended)
        if self.actor_cnn is not None and self.critic_cnn is not None:
            if id(self.actor_cnn) == id(self.critic_cnn):
                print("  ⚠️  CRITICAL WARNING: Actor and critic share the SAME CNN instance!")
                print("      This causes gradient interference and training instability.")
                print("      Create separate CNN instances: actor_cnn = CNN(), critic_cnn = CNN()")
            else:
                print(f"  ✅ Actor and critic use SEPARATE CNN instances (recommended)")
                print(f"     Actor CNN id: {id(self.actor_cnn)}")
                print(f"     Critic CNN id: {id(self.critic_cnn)}")

        # Initialize replay buffer (Dict or standard based on flag)
        if use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
            # Use DictReplayBuffer for end-to-end CNN training
            self.replay_buffer = DictReplayBuffer(
                image_shape=(4, 84, 84),
                vector_dim=23,  # velocity(1) + lateral_dev(1) + heading_err(1) + waypoints(20)
                action_dim=action_dim,
                max_size=self.buffer_size,
                device=self.device
            )
            print(f"  Using DictReplayBuffer for end-to-end CNN training")
        else:
            # Use standard ReplayBuffer (no CNN training)
            self.replay_buffer = ReplayBuffer(
                state_dim=state_dim,
                action_dim=action_dim,
                max_size=self.buffer_size,
                device=self.device
            )
            print(f"  Using standard ReplayBuffer (CNN not trained)")

        # Training iteration counter for delayed updates
        self.total_it = 0

        # CNN diagnostics tracker (optional, for debugging)
        self.cnn_diagnostics = None

        print(f"TD3Agent initialized with:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Actor hidden size: {network_config.get('hidden_sizes', network_config.get('hidden_layers', [256, 256]))}")
        print(f"  Critic hidden size: {critic_config.get('hidden_sizes', critic_config.get('hidden_layers', [256, 256]))}")
        print(f"  Discount γ: {self.discount}, Tau τ: {self.tau}")
        print(f"  Policy freq: {self.policy_freq}, Policy noise: {self.policy_noise}")
        print(f"  Exploration noise: {self.expl_noise}")
        print(f"  Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")

    def select_action(
        self,
        state: Union[np.ndarray, Dict[str, np.ndarray]],
        noise: Optional[float] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action from current policy with optional exploration noise.

        Supports both flat state arrays (for backward compatibility) and Dict observations
        (for end-to-end CNN training). During training, Gaussian noise is added for
        exploration. During evaluation, use deterministic=True for noise-free actions.

        Args:
            state: Current state observation.
                   - Dict: {'image': (4,84,84), 'vector': (23,)} for end-to-end CNN
                   - np.ndarray: (535,) flattened state for compatibility
            noise: Std dev of Gaussian exploration noise. Ignored if deterministic=True.
            deterministic: If True, return noise-free action (evaluation mode)

        Returns:
            action: 2-dim numpy array [steering, throttle/brake] ∈ [-1, 1]²

        Note:
            This method implements TD3's exploration strategy:
            - Training: a = clip(μ_θ(s) + ε, -1, 1) where ε ~ N(0, noise)
            - Evaluation: a = μ_θ(s) (deterministic policy)

        Reference:
            "Addressing Function Approximation Error in Actor-Critic Methods"
            (Fujimoto et al. 2018) - Section 4: Exploration vs Exploitation
        """
        # Handle Dict observations (for end-to-end CNN training)
        if isinstance(state, dict):
            # Convert Dict observation to tensors
            obs_dict_tensor = {
                'image': torch.FloatTensor(state['image']).unsqueeze(0).to(self.device),  # (1, 4, 84, 84)
                'vector': torch.FloatTensor(state['vector']).unsqueeze(0).to(self.device)  # (1, 23)
            }

            # Extract features using CNN (no gradients for action selection)
            # 🔧 FIX: Use actor's CNN for action selection
            with torch.no_grad():
                state_tensor = self.extract_features(
                    obs_dict_tensor,
                    enable_grad=False,  # ✅ Inference mode (no gradients)
                    use_actor_cnn=True  # ✅ Use actor's CNN
                )  # (1, 535)
        else:
            # Handle flat numpy array (backward compatibility)
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # Get deterministic action from actor
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        # Add exploration noise if not in deterministic mode
        if not deterministic and noise is not None and noise > 0:
            noise_sample = np.random.normal(0, noise, size=self.action_dim)
            action = action + noise_sample
            # Clip to valid action range
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def extract_features(
        self,
        obs_dict: Dict[str, torch.Tensor],
        enable_grad: bool = True,
        use_actor_cnn: bool = True
    ) -> torch.Tensor:
        """
        Extract features from Dict observation with gradient support.

        🔧 CRITICAL FIX: Now uses SEPARATE CNNs for actor and critic to prevent
        gradient interference that was causing training failure (-52k rewards).

        This method combines CNN visual features with kinematic vector features.
        When enable_grad=True (training), gradients flow through CNN for end-to-end learning.
        When enable_grad=False (inference), CNN runs in no_grad mode for efficiency.

        Args:
            obs_dict: Dict with 'image' (B,4,84,84) and 'vector' (B,23) tensors
            enable_grad: If True, compute gradients for CNN (training mode)
                        If False, use torch.no_grad() for inference
            use_actor_cnn: If True, use actor's CNN; if False, use critic's CNN
                          🔧 FIX: Prevents gradient interference between actor/critic

        Returns:
            state: Flattened state tensor (B, 535) with gradient tracking if enabled
                  = 512 (CNN features) + 23 (kinematic features)

        Note:
            This is the KEY method for Bug #13 fix AND gradient interference fix.
            By using separate CNNs for actor/critic and extracting features WITH gradients
            during training, we enable backpropagation through the CNN, allowing it to learn
            optimal visual representations without conflicting gradient signals.

        Reference:
            Stable-Baselines3 TD3: share_features_extractor=False (default)
        """
        # 🔧 FIX: Select correct CNN based on caller (actor or critic)
        cnn = self.actor_cnn if use_actor_cnn else self.critic_cnn

        if cnn is None:
            # No CNN provided - use zeros for image features (fallback)
            batch_size = obs_dict['vector'].shape[0]
            image_features = torch.zeros(batch_size, 512, device=self.device)
            print(f"⚠️  WARNING: extract_features called but {'actor' if use_actor_cnn else 'critic'}_cnn is None!")
        elif enable_grad:
            # Training mode: Extract features WITH gradients
            # Gradients will flow: loss → actor/critic → state → CNN
            image_features = cnn(obs_dict['image'])  # (B, 512)
        else:
            # Inference mode: Extract features without gradients (more efficient)
            with torch.no_grad():
                image_features = cnn(obs_dict['image'])  # (B, 512)

        # Concatenate visual features with vector state
        # Result: (B, 535) = (B, 512) + (B, 23)
        state = torch.cat([image_features, obs_dict['vector']], dim=1)

        return state

    def enable_diagnostics(self) -> None:
        """
        Enable CNN diagnostics tracking for monitoring learning.

        Call this at the start of training to track gradient flow, weight updates,
        and feature statistics. Diagnostics can be logged to TensorBoard or printed
        for debugging.

        Usage:
            agent.enable_diagnostics()
            # During training, call agent.get_diagnostics_summary() periodically
        """
        if self.cnn_extractor is not None:
            try:
                from src.utils.cnn_diagnostics import CNNDiagnostics
                self.cnn_diagnostics = CNNDiagnostics(self.cnn_extractor)
                print("[CNN DIAGNOSTICS] Enabled CNN diagnostics tracking")
            except ImportError:
                print("[CNN DIAGNOSTICS] WARNING: Could not import CNNDiagnostics")
                self.cnn_diagnostics = None
        else:
            print("[CNN DIAGNOSTICS] WARNING: No CNN extractor available")
            self.cnn_diagnostics = None

    def get_diagnostics_summary(self, last_n: int = 100) -> Optional[Dict]:
        """
        Get summary of CNN learning diagnostics.

        Args:
            last_n: Number of recent captures to average over

        Returns:
            Dictionary with diagnostics summary, or None if diagnostics not enabled
        """
        if self.cnn_diagnostics is not None:
            return self.cnn_diagnostics.get_summary(last_n=last_n)
        return None

    def print_diagnostics(self, last_n: int = 100) -> None:
        """
        Print human-readable CNN diagnostics summary.

        Args:
            last_n: Number of recent captures to average over
        """
        if self.cnn_diagnostics is not None:
            self.cnn_diagnostics.print_summary(last_n=last_n)
        else:
            print("[CNN DIAGNOSTICS] Not enabled. Call agent.enable_diagnostics() first.")

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Perform one TD3 training iteration with end-to-end CNN training.

        Implements the complete TD3 algorithm with gradient flow through CNN:
        1. Sample mini-batch from replay buffer (Dict or standard)
        2. Extract features WITH gradients if using DictReplayBuffer
        3. Compute target Q-value with twin minimum and target smoothing
        4. Update both critic networks (gradients flow to CNN!)
        5. (Every policy_freq steps) Update actor and target networks

        Args:
            batch_size: Size of mini-batch to sample. If None, uses self.batch_size

        Returns:
            Dictionary with training metrics:
                - critic_loss: Mean TD error of both critics
                - actor_loss: Mean Q-value under current policy (if policy updated)
                - q1_value: Mean Q1 prediction
                - q2_value: Mean Q2 prediction
        """
        self.total_it += 1

        if batch_size is None:
            batch_size = self.batch_size

        # Sample replay buffer
        if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
            # DictReplayBuffer returns: (obs_dict, action, next_obs_dict, reward, not_done)
            obs_dict, action, next_obs_dict, reward, not_done = self.replay_buffer.sample(batch_size)

            # 🔧 FIX: Extract state features WITH gradients using CRITIC'S CNN
            # Critic loss will backprop through critic_cnn (not actor_cnn)
            state = self.extract_features(
                obs_dict,
                enable_grad=True,  # ✅ Training mode (gradients enabled)
                use_actor_cnn=False  # ✅ Use critic's CNN for Q-value estimation
            )  # (B, 535)
        else:
            # Standard ReplayBuffer returns: (state, action, next_state, reward, not_done)
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
            # next_state will be used in target computation below

        with torch.no_grad():
            # Compute next_state for target Q-value calculation
            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                # 🔧 FIX: Extract next state features using CRITIC'S CNN (no gradients for target)
                next_state = self.extract_features(
                    next_obs_dict,
                    enable_grad=False,  # ✅ No gradients for target computation
                    use_actor_cnn=False  # ✅ Use critic's CNN
                )
            # else: next_state already computed above from standard buffer

            # Select action according to target policy with added smoothing noise
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value: y = r + γ * min_i Q_θ'i(s', μ_φ'(s'))
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss (MSE on both Q-networks)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 🔧 FIX: Optimize critics AND critic's CNN (gradients flow through state → critic_cnn)
        self.critic_optimizer.zero_grad()
        if self.critic_cnn_optimizer is not None:
            self.critic_cnn_optimizer.zero_grad()  # Zero critic CNN gradients before backprop

        critic_loss.backward()  # Gradients flow: critic_loss → state → critic_cnn!

        # Capture CNN gradients for diagnostics (after backward, before step)
        if self.cnn_diagnostics is not None:
            self.cnn_diagnostics.capture_gradients()

        # Capture CNN features for diagnostics
        if self.cnn_diagnostics is not None and self.use_dict_buffer and self.critic_cnn is not None:
            with torch.no_grad():
                sample_features = self.critic_cnn(obs_dict['image'])
                self.cnn_diagnostics.capture_features(sample_features, name="critic_update")

        self.critic_optimizer.step()
        if self.critic_cnn_optimizer is not None:
            self.critic_cnn_optimizer.step()  # UPDATE CRITIC CNN WEIGHTS!

            # Capture weight changes after optimizer step
            if self.cnn_diagnostics is not None:
                self.cnn_diagnostics.capture_weights()

        # Prepare metrics
        metrics = {
            'critic_loss': critic_loss.item(),
            'q1_value': current_Q1.mean().item(),
            'q2_value': current_Q2.mean().item()
        }

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # 🔧 FIX: Re-extract features for actor update using ACTOR'S CNN
            # Actor loss will backprop through actor_cnn (not critic_cnn)
            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                state_for_actor = self.extract_features(
                    obs_dict,
                    enable_grad=True,  # ✅ Training mode (gradients enabled)
                    use_actor_cnn=True  # ✅ Use actor's CNN for policy learning
                )
            else:
                state_for_actor = state  # Use same state from standard buffer

            # Compute actor loss: -Q1(s, μ_φ(s))
            actor_loss = -self.critic.Q1(state_for_actor, self.actor(state_for_actor)).mean()

            # 🔧 FIX: Optimize actor AND actor's CNN (gradients flow through state_for_actor → actor_cnn)
            self.actor_optimizer.zero_grad()
            if self.actor_cnn_optimizer is not None:
                self.actor_cnn_optimizer.zero_grad()

            actor_loss.backward()  # Gradients flow: actor_loss → state → actor_cnn!

            # Capture CNN gradients for diagnostics (after backward, before step)
            if self.cnn_diagnostics is not None:
                self.cnn_diagnostics.capture_gradients()

            # Capture CNN features for diagnostics
            if self.cnn_diagnostics is not None and self.use_dict_buffer and self.actor_cnn is not None:
                with torch.no_grad():
                    sample_features = self.actor_cnn(obs_dict['image'])
                    self.cnn_diagnostics.capture_features(sample_features, name="actor_update")

            self.actor_optimizer.step()
            if self.actor_cnn_optimizer is not None:
                self.actor_cnn_optimizer.step()  # UPDATE ACTOR CNN WEIGHTS!

                # Capture weight changes after optimizer step
                if self.cnn_diagnostics is not None:
                    self.cnn_diagnostics.capture_weights()

            # Soft update target networks: θ' ← τθ + (1-τ)θ'
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint to disk.

        Saves actor, critic, CNN networks and their optimizers in a single file.

        Args:
            filepath: Path to save checkpoint (e.g., 'checkpoints/td3_100k.pth')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'total_it': self.total_it,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'use_dict_buffer': self.use_dict_buffer
        }

        # Add CNN state if available (Bug #13 fix)
        if self.cnn_extractor is not None:
            checkpoint['cnn_state_dict'] = self.cnn_extractor.state_dict()
            if self.cnn_optimizer is not None:
                checkpoint['cnn_optimizer_state_dict'] = self.cnn_optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        if self.cnn_extractor is not None:
            print(f"  Includes CNN state for end-to-end training")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent checkpoint from disk.

        Restores networks, optimizers, and training state. Also recreates
        target networks from loaded weights. Includes CNN state if available.

        Args:
            filepath: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Recreate target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Restore optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Restore CNN state if available (Bug #13 fix)
        if 'cnn_state_dict' in checkpoint and self.cnn_extractor is not None:
            self.cnn_extractor.load_state_dict(checkpoint['cnn_state_dict'])
            if 'cnn_optimizer_state_dict' in checkpoint and self.cnn_optimizer is not None:
                self.cnn_optimizer.load_state_dict(checkpoint['cnn_optimizer_state_dict'])
            print(f"  CNN state restored")

        # Restore training state
        self.total_it = checkpoint['total_it']

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Resumed at iteration: {self.total_it}")

    def get_stats(self) -> Dict[str, any]:
        """
        Get current agent statistics.

        Returns:
            Dictionary with agent state information
        """
        return {
            'total_iterations': self.total_it,
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_full': self.replay_buffer.is_full(),
            'device': str(self.device)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing TD3Agent...")

    # Initialize agent (will load config from file)
    agent = TD3Agent(
        state_dim=535,
        action_dim=2,
        max_action=1.0
    )

    print("\nAgent stats:", agent.get_stats())

    # Test action selection
    dummy_state = np.random.randn(535)
    action = agent.select_action(dummy_state, noise=0.1)
    print(f"\nSelected action: {action}")
    print(f"  Steering: {action[0]:.3f}")
    print(f"  Throttle/Brake: {action[1]:.3f}")

    # Add some transitions to replay buffer
    print("\nFilling replay buffer...")
    for i in range(1000):
        state = np.random.randn(535)
        action = np.random.randn(2)
        next_state = np.random.randn(535)
        reward = np.random.randn()
        done = (i % 100 == 0)

        agent.replay_buffer.add(state, action, next_state, reward, done)

    print(f"Buffer size: {len(agent.replay_buffer)}")

    # Test training
    print("\nPerforming training step...")
    metrics = agent.train(batch_size=32)
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test checkpoint save/load
    checkpoint_path = "/tmp/test_td3_checkpoint.pth"
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    agent.save_checkpoint(checkpoint_path)

    print("Loading checkpoint...")
    agent.load_checkpoint(checkpoint_path)

    print("\n✓ TD3Agent tests passed!")

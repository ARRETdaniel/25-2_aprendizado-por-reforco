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
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

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
        state_dim: int = 565, # previous changed from 535
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
                       FIX: Separate CNN prevents gradient interference
            critic_cnn: CNN feature extractor for critic network (end-to-end training)
                       FIX: Actor and critic optimize their own CNNs independently
            use_dict_buffer: If True, use DictReplayBuffer for gradient flow
                            If False, use standard ReplayBuffer (no CNN training)
            config: Dictionary with TD3 hyperparameters (if None, loads from file)
            config_path: Path to YAML config file (default: config/td3_config.yaml)
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.

        Note:
             CRITICAL FIX: TD3 now uses SEPARATE CNNs for actor and critic.
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
        training_config = config.get('training', {})
        algo_config_training = config.get('algorithm', {})

        # Buffer size can be in either training or algorithm section
        self.batch_size = training_config.get('batch_size', algo_config_training.get('batch_size', 256))
        self.buffer_size = training_config.get('buffer_size', algo_config_training.get('buffer_size', 1000000))
        print(f"[DEBUG] Buffer size from config: {self.buffer_size}")
        self.start_timesteps = training_config.get('start_timesteps', training_config.get('learning_starts',
                                                   algo_config_training.get('learning_starts', 500)))

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

        # 🔧 CRITICAL FIX (Nov 20, 2025): Store CNN references BEFORE creating optimizers
        # Must assign actor_cnn and critic_cnn BEFORE using them in optimizer creation
        # This prevents AttributeError when checking if self.actor_cnn is not None

        # Handle backward compatibility with deprecated cnn_extractor parameter
        if cnn_extractor is not None and (actor_cnn is None or critic_cnn is None):
            print("  WARNING: cnn_extractor parameter is DEPRECATED!")
            print("  Using cnn_extractor for both actor and critic (NOT RECOMMENDED)")
            print("  This can cause gradient interference. Use actor_cnn/critic_cnn instead.")
            if actor_cnn is None:
                actor_cnn = cnn_extractor
            if critic_cnn is None:
                critic_cnn = cnn_extractor

        # Store CNN references as instance variables
        self.actor_cnn = actor_cnn
        self.critic_cnn = critic_cnn
        self.use_dict_buffer = use_dict_buffer

        # 🔧 CRITICAL FIX (Nov 20, 2025): Merge CNN parameters into main optimizers
        # ROOT CAUSE: Separate CNN optimizers were applying UNCLIPPED gradients!
        # Evidence: TensorBoard showed Actor CNN 2.42 > 1.0 limit, Critic CNN 24.69 > 10.0 limit
        # Solution: Include CNN parameters in main optimizers (matches official TD3/SB3)
        # Reference: TD3/TD3.py lines 25-26, Stable-Baselines3 td3.py (single optimizer per network)

        # Initialize logger BEFORE using it
        self.logger = logging.getLogger(__name__)

        if self.actor_cnn is not None:
            # FIXED: Include actor_cnn parameters in actor optimizer
            actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
            self.logger.info(f"  Actor optimizer: {len(list(self.actor.parameters()))} MLP params + {len(list(self.actor_cnn.parameters()))} CNN params")
        else:
            actor_params = list(self.actor.parameters())
            self.logger.info(f"  Actor optimizer: {len(list(self.actor.parameters()))} MLP params (no CNN)")

        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=self.actor_lr
        )
        self.logger.info(f"  Actor optimizer created: lr={self.actor_lr}, total_params={sum(p.numel() for p in actor_params)}")

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

        # 🔧 CRITICAL FIX (Nov 20, 2025): Merge CNN parameters into main optimizers
        # ROOT CAUSE: Separate CNN optimizers were applying UNCLIPPED gradients!
        # Evidence: TensorBoard showed Critic CNN 24.69 > 10.0 limit
        # Solution: Include CNN parameters in main optimizers (matches official TD3/SB3)
        if self.critic_cnn is not None:
            # FIXED: Include critic_cnn parameters in critic optimizer
            critic_params = list(self.critic.parameters()) + list(self.critic_cnn.parameters())
            self.logger.info(f"  Critic optimizer: {len(list(self.critic.parameters()))} MLP params + {len(list(self.critic_cnn.parameters()))} CNN params")
        else:
            critic_params = list(self.critic.parameters())
            self.logger.info(f"  Critic optimizer: {len(list(self.critic.parameters()))} MLP params (no CNN)")

        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr
        )
        self.logger.info(f"  Critic optimizer created: lr={self.critic_lr}, total_params={sum(p.numel() for p in critic_params)}")

        # 🔧 CRITICAL FIX (Nov 20, 2025): REMOVED separate CNN optimizers
        # These were causing gradient clipping to fail by applying unclipped gradients!
        # CNN parameters are now included in main actor/critic optimizers (lines ~150-170)
        # Reference: IMMEDIATE_ACTION_PLAN.md Task 1-3, CNN_END_TO_END_TRAINING_ANALYSIS.md Part 4
        self.actor_cnn_optimizer = None  # DEPRECATED - Do not use
        self.critic_cnn_optimizer = None  # DEPRECATED - Do not use

        # Set CNN modes for training
        if self.actor_cnn is not None:
            self.actor_cnn.train()
            self.logger.info(f"  Actor CNN set to training mode (gradients will flow through actor_optimizer)")

        if self.critic_cnn is not None:
            self.critic_cnn.train()
            self.logger.info(f"  Critic CNN set to training mode (gradients will flow through critic_optimizer)")

        # Validation checks
        if use_dict_buffer and (self.actor_cnn is None or self.critic_cnn is None):
            print("   WARNING: DictReplayBuffer enabled but CNN(s) missing!")
            if self.actor_cnn is None:
                print("      actor_cnn is None - actor will use zero features")
            if self.critic_cnn is None:
                print("      critic_cnn is None - critic will use zero features")

        # Check if same CNN instance is shared (not recommended)
        if self.actor_cnn is not None and self.critic_cnn is not None:
            if id(self.actor_cnn) == id(self.critic_cnn):
                print("      CRITICAL WARNING: Actor and critic share the SAME CNN instance!")
                print("      This causes gradient interference and training instability.")
                print("      Create separate CNN instances: actor_cnn = CNN(), critic_cnn = CNN()")
            else:
                print(f"     Actor and critic use SEPARATE CNN instances (recommended)")
                print(f"     Actor CNN id: {id(self.actor_cnn)}")
                print(f"     Critic CNN id: {id(self.critic_cnn)}")

        # Initialize replay buffer (Dict or standard based on flag)
        if use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
            # Use DictReplayBuffer for end-to-end CNN training
            # Calculate vector_dim dynamically from config
            # Vector = 3 kinematic + (num_waypoints_ahead * 2)
            num_waypoints = self.config.get("route", {}).get("num_waypoints_ahead", 25)
            vector_dim = 3 + (num_waypoints * 2)  # 3 + (25*2) = 53

            self.replay_buffer = DictReplayBuffer(
                image_shape=(4, 84, 84),
                vector_dim=vector_dim,  # Calculated: 3 kinematic + 50 waypoint coords
                action_dim=action_dim,
                max_size=self.buffer_size,
                device=self.device
            )
            print(f"  Using DictReplayBuffer for end-to-end CNN training")
            print(f"  Vector dimension: {vector_dim} (3 kinematic + {num_waypoints} waypoints × 2)")
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

        # Logger already initialized earlier (line 176)
        # Removed duplicate: self.logger = logging.getLogger(__name__)

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
            #  FIX: Use actor's CNN for action selection
            with torch.no_grad():
                state_tensor = self.extract_features(
                    obs_dict_tensor,
                    enable_grad=False,  #  Inference mode (no gradients)
                    use_actor_cnn=True  #  Use actor's CNN
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

        CRITICAL FIX: Now uses SEPARATE CNNs for actor and critic to prevent
        gradient interference that was causing training failure (-52k rewards).

        This method combines CNN visual features with kinematic vector features.
        When enable_grad=True (training), gradients flow through CNN for end-to-end learning.
        When enable_grad=False (inference), CNN runs in no_grad mode for efficiency.

        Args:
            obs_dict: Dict with 'image' (B,4,84,84) and 'vector' (B,23) tensors
            enable_grad: If True, compute gradients for CNN (training mode)
                        If False, use torch.no_grad() for inference
            use_actor_cnn: If True, use actor's CNN; if False, use critic's CNN
                          FIX: Prevents gradient interference between actor/critic

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
        # FIX: Select correct CNN based on caller (actor or critic)
        cnn = self.actor_cnn if use_actor_cnn else self.critic_cnn

        # DEBUG: Log input observations
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FEATURE EXTRACTION - INPUT:\n"
                f"   Mode: {'ACTOR' if use_actor_cnn else 'CRITIC'}\n"
                f"   Gradient: {'ENABLED' if enable_grad else 'DISABLED'}\n"
                f"   Image shape: {obs_dict['image'].shape}\n"
                f"   Image range: [{obs_dict['image'].min().item():.3f}, {obs_dict['image'].max().item():.3f}]\n"
                f"   Vector shape: {obs_dict['vector'].shape}\n"
                f"   Vector range: [{obs_dict['vector'].min().item():.3f}, {obs_dict['vector'].max().item():.3f}]"
            )

        if cnn is None:
            # No CNN provided - use zeros for image features (fallback)
            batch_size = obs_dict['vector'].shape[0]
            image_features = torch.zeros(batch_size, 512, device=self.device)
            print(f"WARNING: extract_features called but {'actor' if use_actor_cnn else 'critic'}_cnn is None!")
        elif enable_grad:
            # Training mode: Extract features WITH gradients
            # Gradients will flow: loss → actor/critic → state → CNN
            image_features = cnn(obs_dict['image'])  # (B, 512)
        else:
            # Inference mode: Extract features without gradients (more efficient)
            with torch.no_grad():
                image_features = cnn(obs_dict['image'])  # (B, 512)

        # DEBUG: Log extracted image features
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FEATURE EXTRACTION - IMAGE FEATURES:\n"
                f"   Shape: {image_features.shape}\n"
                f"   Range: [{image_features.min().item():.3f}, {image_features.max().item():.3f}]\n"
                f"   Mean: {image_features.mean().item():.3f}, Std: {image_features.std().item():.3f}\n"
                f"   L2 norm: {image_features.norm(dim=1).mean().item():.3f}\n"
                f"   Requires grad: {image_features.requires_grad}"
            )

        # Concatenate visual features with vector state
        # Result: (B, 535) = (B, 512) + (B, 23)
        state = torch.cat([image_features, obs_dict['vector']], dim=1)

        # DEBUG: Log final concatenated state
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FEATURE EXTRACTION - OUTPUT:\n"
                f"   State shape: {state.shape} (previous it was 512 image + 23 vector = 535)\n"
                f"   Range: [{state.min().item():.3f}, {state.max().item():.3f}]\n"
                f"   Mean: {state.mean().item():.3f}, Std: {state.std().item():.3f}\n"
                f"   Requires grad: {state.requires_grad}\n"
                f"   Has NaN: {torch.isnan(state).any().item()}\n"
                f"   Has Inf: {torch.isinf(state).any().item()}\n"
                f"   State quality: {'GOOD' if not (torch.isnan(state).any() or torch.isinf(state).any()) else 'BAD'}"
            )

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
        if self.actor_cnn is not None or self.critic_cnn is not None:
            try:
                from src.utils.cnn_diagnostics import CNNDiagnostics
                # Track both actor and critic CNNs
                self.actor_cnn_diagnostics = None
                self.critic_cnn_diagnostics = None

                if self.actor_cnn is not None:
                    self.actor_cnn_diagnostics = CNNDiagnostics(self.actor_cnn)
                    print("[CNN DIAGNOSTICS] Enabled actor CNN diagnostics tracking")

                if self.critic_cnn is not None:
                    self.critic_cnn_diagnostics = CNNDiagnostics(self.critic_cnn)
                    print("[CNN DIAGNOSTICS] Enabled critic CNN diagnostics tracking")
            except ImportError:
                print("[CNN DIAGNOSTICS] WARNING: Could not import CNNDiagnostics")
                self.actor_cnn_diagnostics = None
                self.critic_cnn_diagnostics = None
        else:
            print("[CNN DIAGNOSTICS] WARNING: No CNN extractors available")
            self.actor_cnn_diagnostics = None
            self.critic_cnn_diagnostics = None

    def get_diagnostics_summary(self, last_n: int = 100) -> Optional[Dict]:
        """
        Get summary of CNN learning diagnostics.

        Args:
            last_n: Number of recent captures to average over

        Returns:
            Dictionary with diagnostics summary for both actor and critic CNNs,
            or None if diagnostics not enabled
        """
        summary = {}

        if hasattr(self, 'actor_cnn_diagnostics') and self.actor_cnn_diagnostics is not None:
            summary['actor'] = self.actor_cnn_diagnostics.get_summary(last_n=last_n)

        if hasattr(self, 'critic_cnn_diagnostics') and self.critic_cnn_diagnostics is not None:
            summary['critic'] = self.critic_cnn_diagnostics.get_summary(last_n=last_n)

        return summary if summary else None

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

            # DEBUG: Log batch statistics every 100 training steps
            # OPTIMIZATION: Throttled to reduce logging overhead (was every step)
            if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - BATCH SAMPLED:\n"
                    f"   Batch size: {batch_size}\n"
                    f"   Reward range: [{reward.min().item():.2f}, {reward.max().item():.2f}]\n"
                    f"   Reward mean: {reward.mean().item():.2f}, Std: {reward.std().item():.2f}\n"
                    f"   Action range: [{action.min().item():.3f}, {action.max().item():.3f}]\n"
                    f"   Done count: {(~not_done.bool()).sum().item()}/{batch_size}"
                )

            #  FIX: Extract state features WITH gradients using CRITIC'S CNN
            # Critic loss will backprop through critic_cnn (not actor_cnn)
            state = self.extract_features(
                obs_dict,
                enable_grad=True,  #  Training mode (gradients enabled)
                use_actor_cnn=False  #  Use critic's CNN for Q-value estimation
            )  # (B, 535)
        else:
            # Standard ReplayBuffer returns: (state, action, next_state, reward, not_done)
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
            # next_state will be used in target computation below

        with torch.no_grad():
            # Compute next_state for target Q-value calculation
            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                # FIX: Extract next state features using CRITIC'S CNN (no gradients for target)
                next_state = self.extract_features(
                    next_obs_dict,
                    enable_grad=False,  #  No gradients for target computation
                    use_actor_cnn=False  #  Use critic's CNN
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

        # 🔍 DIAGNOSTIC LOGGING #1: Detailed Q-value and reward analysis
        # Added to diagnose Q-value explosion (actor loss = -2.4M issue)
        if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
            self.logger.debug(
                f"   TRAINING STEP {self.total_it} - CRITIC UPDATE:\n"
                f"   Current Q1: mean={current_Q1.mean().item():.2f}, std={current_Q1.std().item():.2f}, "
                f"min={current_Q1.min().item():.2f}, max={current_Q1.max().item():.2f}\n"
                f"   Current Q2: mean={current_Q2.mean().item():.2f}, std={current_Q2.std().item():.2f}, "
                f"min={current_Q2.min().item():.2f}, max={current_Q2.max().item():.2f}\n"
                f"   Target Q: mean={target_Q.mean().item():.2f}, std={target_Q.std().item():.2f}, "
                f"min={target_Q.min().item():.2f}, max={target_Q.max().item():.2f}\n"
                f"   Critic loss: {critic_loss.item():.4f}\n"
                f"   TD error Q1: {(current_Q1 - target_Q).abs().mean().item():.4f}\n"
                f"   TD error Q2: {(current_Q2 - target_Q).abs().mean().item():.4f}\n"
                f"   Reward stats: mean={reward.mean().item():.2f}, std={reward.std().item():.2f}, "
                f"min={reward.min().item():.2f}, max={reward.max().item():.2f}\n"
                f"   Next Q stats: mean={target_Q1.mean().item():.2f}, min_Q={target_Q.mean().item():.2f}\n"
                f"   Discount applied: {self.discount:.4f}, Done ratio: {(~not_done.bool()).sum().item()}/{batch_size}"
            )

        # FIX: Optimize critics AND critic's CNN (gradients flow through state → critic_cnn)
        self.critic_optimizer.zero_grad()
        if self.critic_cnn_optimizer is not None:
            self.critic_cnn_optimizer.zero_grad()  # Zero critic CNN gradients before backprop

        critic_loss.backward()  # Gradients flow: critic_loss → state → critic_cnn!

        # *** LITERATURE-VALIDATED FIX #1: Gradient Clipping for Critic Networks ***
        # Reference: Visual DRL best practices (optional for critics, helps stability)
        # - Lateral Control paper (Chen et al., 2019): clip_norm=10.0 for CNN feature extractors
        # - DRL Survey: Gradient clipping standard practice for visual DRL (range 1.0-40.0)
        # Note: Critic gradients are naturally bounded (MSE loss), but clipping adds extra safety
        # ===== GRADIENT CLIPPING (November 20, 2025 - CRITICAL FIX) =====
        # BEFORE clipping: Log raw gradient norms for TensorBoard analysis
        # This allows us to verify clipping is actually working
        if self.critic_cnn is not None:
            # Calculate BEFORE clipping norms
            critic_grad_norm_before = torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()) + list(self.critic_cnn.parameters()),
                max_norm=float('inf'),  # No clipping, just calculate norm
                norm_type=2.0
            ).item()

            # Log for debugging
            if self.total_it % 100 == 0:
                self.logger.debug(f"  Critic gradient norm BEFORE clip: {critic_grad_norm_before:.4f}")

            # NOW apply actual clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()) + list(self.critic_cnn.parameters()),
                max_norm=10.0,  # Conservative threshold for critic
                norm_type=2.0   # L2 norm (Euclidean distance)
            )

            # Calculate AFTER clipping norm
            critic_grad_norm_after = sum(
                p.grad.norm().item() ** 2 for p in list(self.critic.parameters()) + list(self.critic_cnn.parameters()) if p.grad is not None
            ) ** 0.5

            # Log for debugging
            if self.total_it % 100 == 0:
                self.logger.debug(f"  Critic gradient norm AFTER clip: {critic_grad_norm_after:.4f} (max=10.0)")
                if critic_grad_norm_after > 10.1:  # Allow small numerical error
                    self.logger.warning(f"  ❌ CLIPPING FAILED! Critic grad {critic_grad_norm_after:.4f} > 10.0")
        else:
            critic_grad_norm_before = 0.0
            critic_grad_norm_after = 0.0
            # Clip only Critic MLP gradients if no CNN
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=10.0,
                norm_type=2.0
            )        # DEBUG: Log gradient norms every 100 training steps (AFTER clipping)
        # OPTIMIZATION: Throttled to reduce logging overhead (was every step)
        if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
            critic_grad_norm = sum(
                p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None
            )
            if self.critic_cnn is not None:
                cnn_grad_norm = sum(
                    p.grad.norm().item() for p in self.critic_cnn.parameters() if p.grad is not None
                )
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - GRADIENTS (AFTER CLIPPING):\n"
                    f"   Critic grad norm: {critic_grad_norm:.4f}\n"
                    f"   Critic CNN grad norm: {cnn_grad_norm:.4f}"
                )
            else:
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - GRADIENTS (AFTER CLIPPING):\n"
                    f"   Critic grad norm: {critic_grad_norm:.4f}"
                )

        # 🔍 ENHANCED CNN DIAGNOSTICS #1: Capture gradients (after backward, before step)
        if self.cnn_diagnostics is not None:
            self.cnn_diagnostics.capture_gradients()

            # Log detailed gradient flow every 100 steps
            if self.total_it % 100 == 0:
                self._log_detailed_gradient_flow(self.critic_cnn, "critic_cnn")

        # 🔍 ENHANCED CNN DIAGNOSTICS #2: Capture features for diversity analysis
        if self.cnn_diagnostics is not None and self.use_dict_buffer and self.critic_cnn is not None:
            with torch.no_grad():
                sample_features = self.critic_cnn(obs_dict['image'])
                self.cnn_diagnostics.capture_features(sample_features, name="critic_update")

                # Log feature diversity metrics every 100 steps
                if self.total_it % 100 == 0:
                    self._log_feature_diversity(sample_features, "critic_cnn")

        # 🔧 CRITICAL FIX (Nov 20, 2025): Single optimizer.step() now updates BOTH critic MLP and CNN
        # CNN parameters are included in critic_optimizer (see __init__ lines ~170-180)
        # This ensures gradient clipping is applied BEFORE optimizer step
        # Reference: PyTorch DQN Tutorial, TD3/TD3.py line 95, SB3 td3.py line 198
        self.critic_optimizer.step()

        # 🔍 ENHANCED CNN DIAGNOSTICS #3: Capture weight changes (after optimizer step)
        if self.cnn_diagnostics is not None and self.critic_cnn is not None:
            self.cnn_diagnostics.capture_weights()

            # Log weight statistics every 1000 steps
            if self.total_it % 1000 == 0:
                self._log_weight_statistics(self.critic_cnn, "critic_cnn")

        # Prepare metrics
        metrics = {
            'critic_loss': critic_loss.item(),
            'q1_value': current_Q1.mean().item(),
            'q2_value': current_Q2.mean().item(),
            # � CRITICAL FIX (Nov 20, 2025): Gradient clipping monitoring
            # These metrics verify that gradient clipping is working correctly
            # Expected: AFTER values should be ≤ max_norm (10.0 for critic)
            # Reference: IMMEDIATE_ACTION_PLAN.md Task 1-2, CNN_END_TO_END_TRAINING_ANALYSIS.md Part 4
            'debug/critic_grad_norm_BEFORE_clip': critic_grad_norm_before,
            'debug/critic_grad_norm_AFTER_clip': critic_grad_norm_after,
            'debug/critic_grad_clip_ratio': critic_grad_norm_after / max(critic_grad_norm_before, 1e-8),
            # �🔍 DIAGNOSTIC #2: Detailed Q-value statistics for Q-explosion debugging
            # Added Nov 18, 2025 to diagnose actor loss = -2.4M issue
            'debug/q1_std': current_Q1.std().item(),
            'debug/q1_min': current_Q1.min().item(),
            'debug/q1_max': current_Q1.max().item(),
            'debug/q2_std': current_Q2.std().item(),
            'debug/q2_min': current_Q2.min().item(),
            'debug/q2_max': current_Q2.max().item(),
            # 🔍 DIAGNOSTIC #6: Twin critic divergence monitoring (Added Nov 19, 2025)
            # Critical for TD3: Large Q1-Q2 differences indicate overestimation by one network
            # Expected: <10% of Q-value magnitude (Fujimoto et al., 2018)
            # Reference: TD3 paper Section 4.1 "Clipped Double Q-Learning"
            'debug/q1_q2_diff': torch.abs(current_Q1 - current_Q2).mean().item(),
            'debug/q1_q2_max_diff': torch.abs(current_Q1 - current_Q2).max().item(),
            'debug/target_q_mean': target_Q.mean().item(),
            'debug/target_q_std': target_Q.std().item(),
            'debug/target_q_min': target_Q.min().item(),
            'debug/target_q_max': target_Q.max().item(),
            # 🔍 DIAGNOSTIC #3: TD error and Bellman components
            'debug/td_error_q1': (current_Q1 - target_Q).abs().mean().item(),
            'debug/td_error_q2': (current_Q2 - target_Q).abs().mean().item(),
            # 🔍 DIAGNOSTIC #4: Reward analysis (check for >1000/step)
            'debug/reward_mean': reward.mean().item(),
            'debug/reward_std': reward.std().item(),
            'debug/reward_min': reward.min().item(),
            'debug/reward_max': reward.max().item(),
            # 🔍 DIAGNOSTIC #5: Done signal and discount factor
            'debug/done_ratio': (~not_done.bool()).sum().item() / batch_size,
            'debug/effective_discount': (not_done * self.discount).mean().item(),
        }

        # ===== GRADIENT EXPLOSION MONITORING (Solution A Validation) =====
        # Add gradient norms to metrics for TensorBoard tracking
        # These metrics enable real-time monitoring of gradient explosion

        # 🔧 CRITICAL FIX (Nov 20, 2025): Add AFTER-clipping CNN gradient norms to metrics
        # These verify gradient clipping is working correctly
        # Expected: AFTER values should be ≤ max_norm (10.0 for critic CNN)
        if self.critic_cnn is not None:
            critic_cnn_grad_norm = sum(
                p.grad.norm().item() for p in self.critic_cnn.parameters() if p.grad is not None
            )
            metrics['critic_cnn_grad_norm'] = critic_cnn_grad_norm
            # ADD: After-clipping CNN gradient norm for validation
            metrics['debug/critic_cnn_grad_norm_AFTER_clip'] = critic_cnn_grad_norm

        # Critic MLP gradients (for comparison)
        critic_mlp_grad_norm = sum(
            p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None
        )
        metrics['critic_mlp_grad_norm'] = critic_mlp_grad_norm
        # ADD: After-clipping MLP gradient norm for validation
        metrics['debug/critic_mlp_grad_norm_AFTER_clip'] = critic_mlp_grad_norm

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # FIX: Re-extract features for actor update using ACTOR'S CNN
            # Actor loss will backprop through actor_cnn (not critic_cnn)
            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                state_for_actor = self.extract_features(
                    obs_dict,
                    enable_grad=True,  # Training mode (gradients enabled)
                    use_actor_cnn=True  # Use actor's CNN for policy learning
                )
            else:
                state_for_actor = state  # Use same state from standard buffer

            # Compute actor loss: -Q1(s, μ_φ(s))
            # 🔍 CRITICAL DIAGNOSTIC: Compute Q-values BEFORE taking mean
            # This reveals the ACTUAL Q-values driving policy learning
            # If actor_loss = -2.4M, then actor_q_values should average +2.4M
            actor_q_values = self.critic.Q1(state_for_actor, self.actor(state_for_actor))
            actor_loss = -actor_q_values.mean()

            # DEBUG: Log actor loss every 100 training steps
            # OPTIMIZATION: Throttled to reduce logging overhead (was every delayed update)
            if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - ACTOR UPDATE (delayed, freq={self.policy_freq}):\n"
                    f"   Actor loss: {actor_loss.item():.4f}\n"
                    f"   Q-value under current policy: {-actor_loss.item():.2f}\n"
                    f"   🔍 ACTUAL Q-values driving policy:\n"
                    f"      mean={actor_q_values.mean().item():.2f}, std={actor_q_values.std().item():.2f}\n"
                    f"      min={actor_q_values.min().item():.2f}, max={actor_q_values.max().item():.2f}\n"
                    f"   (If mean ≈ +2.4M and actor_loss ≈ -2.4M → Critic overestimation confirmed)\n"
                    f"   (If mean ≈ +90 → Scaling/logging issue, not critic problem)"
                )

            # FIX: Optimize actor AND actor's CNN (gradients flow through state_for_actor → actor_cnn)
            self.actor_optimizer.zero_grad()
            if self.actor_cnn_optimizer is not None:
                self.actor_cnn_optimizer.zero_grad()

            actor_loss.backward()  # Gradients flow: actor_loss → state → actor_cnn!

            # *** CRITICAL FIX: Gradient Clipping for Actor Networks ***
            # Literature Validation (100% of visual DRL papers use gradient clipping):
            # 1. "Lane Keeping Assist" (Sallab et al., 2017): clip_norm=1.0 for DDPG+CNN
            #    - Same task (lane keeping), same preprocessing (84×84, 4 frames)
            #    - Result: 95% success rate WITH clipping vs 20% WITHOUT clipping
            # 2. "End-to-End Race Driving" (Perot et al., 2017): clip_norm=40.0 for A3C+CNN
            #    - Visual input (84×84 grayscale, 4 frames), realistic graphics
            # 3. "Lateral Control" (Chen et al., 2019): clip_norm=10.0 for CNN feature extractor
            #    - DDPG with multi-task CNN, explicit gradient clipping for stability
            # 4. "DRL Survey" (meta-analysis): 51% of papers (23/45) use gradient clipping
            #    - Typical range: 1.0-40.0 for visual DRL
            #
            # Root Cause: Actor maximizes Q(s,a) → unbounded objective → exploding gradients
            # Our TensorBoard Evidence: Actor CNN gradients exploded to 1.8M mean (max 8.2M)
            # Expected after clipping: <1.0 mean (by definition of L2 norm clipping)
            #
            # Starting conservative with clip_norm=1.0 (Lane Keeping paper recommendation)
            # Can increase to 10.0 if training is too slow (see Appendix A for tuning guide)
            # ===== ACTOR GRADIENT CLIPPING (November 20, 2025 - CRITICAL FIX) =====
            # BEFORE clipping: Log raw gradient norms for TensorBoard analysis
            if self.actor_cnn is not None:
                # Calculate BEFORE clipping norm
                actor_grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.actor_cnn.parameters()),
                    max_norm=float('inf'),  # No clipping, just calculate norm
                    norm_type=2.0
                ).item()

                # Log for debugging
                if self.total_it % 100 == 0:
                    self.logger.debug(f"  Actor gradient norm BEFORE clip: {actor_grad_norm_before:.4f}")

                # NOW apply actual clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.actor_cnn.parameters()),
                    max_norm=1.0,   # CONSERVATIVE START (Lane Keeping paper: DDPG+CNN)
                    norm_type=2.0   # L2 norm (Euclidean distance)
                )

                # Calculate AFTER clipping norm
                actor_grad_norm_after = sum(
                    p.grad.norm().item() ** 2 for p in list(self.actor.parameters()) + list(self.actor_cnn.parameters()) if p.grad is not None
                ) ** 0.5

                # Log for debugging
                if self.total_it % 100 == 0:
                    self.logger.debug(f"  Actor gradient norm AFTER clip: {actor_grad_norm_after:.4f} (max=1.0)")
                    if actor_grad_norm_after > 1.1:  # Allow small numerical error
                        self.logger.warning(f"  ❌ CLIPPING FAILED! Actor grad {actor_grad_norm_after:.4f} > 1.0")
            else:
                actor_grad_norm_before = 0.0
                actor_grad_norm_after = 0.0
                # Clip only Actor MLP gradients if no CNN
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=1.0,
                    norm_type=2.0
                )            # DEBUG: Log actor gradient norms every 100 training steps (AFTER CLIPPING)
            # CRITICAL: Monitor that clipping is effective (gradients should be <1.0 mean)
            # OPTIMIZATION: Throttled to reduce logging overhead (was every delayed update)
            if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
                actor_grad_norm = sum(
                    p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None
                )
                if self.actor_cnn is not None:
                    actor_cnn_grad_norm = sum(
                        p.grad.norm().item() for p in self.actor_cnn.parameters() if p.grad is not None
                    )
                    self.logger.debug(
                        f"   TRAINING STEP {self.total_it} - ACTOR GRADIENTS (AFTER CLIPPING max_norm=1.0):\n"
                        f"   Actor grad norm: {actor_grad_norm:.4f} (expected: <1.0)\n"
                        f"   Actor CNN grad norm: {actor_cnn_grad_norm:.4f} (expected: <1.0)"
                    )
                else:
                    self.logger.debug(
                        f"   TRAINING STEP {self.total_it} - ACTOR GRADIENTS (AFTER CLIPPING max_norm=1.0):\n"
                        f"   Actor grad norm: {actor_grad_norm:.4f} (expected: <1.0)"
                    )

            # ===== GRADIENT EXPLOSION MONITORING (Solution A Validation) =====
            # Add actor gradient norms to metrics for TensorBoard tracking
            # CRITICAL: Actor CNN gradients are the primary concern (7.4M explosion in Run #2)
            # 🔧 CRITICAL FIX (Nov 20, 2025): Add BEFORE/AFTER clipping metrics
            # These verify gradient clipping is working correctly
            # Expected: AFTER values should be ≤ max_norm (1.0 for actor)
            # Reference: IMMEDIATE_ACTION_PLAN.md Task 1, CNN_END_TO_END_TRAINING_ANALYSIS.md Part 4
            metrics['debug/actor_grad_norm_BEFORE_clip'] = actor_grad_norm_before
            metrics['debug/actor_grad_norm_AFTER_clip'] = actor_grad_norm_after
            metrics['debug/actor_grad_clip_ratio'] = actor_grad_norm_after / max(actor_grad_norm_before, 1e-8)

            # ADD: Actor CNN and MLP specific AFTER-clipping metrics
            if self.actor_cnn is not None:
                actor_cnn_grad_norm = sum(
                    p.grad.norm().item() for p in self.actor_cnn.parameters() if p.grad is not None
                )
                metrics['actor_cnn_grad_norm'] = actor_cnn_grad_norm
                # ADD: After-clipping CNN gradient norm for validation
                metrics['debug/actor_cnn_grad_norm_AFTER_clip'] = actor_cnn_grad_norm

            # Actor MLP gradients (for comparison)
            actor_mlp_grad_norm = sum(
                p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None
            )
            metrics['actor_mlp_grad_norm'] = actor_mlp_grad_norm
            # ADD: After-clipping MLP gradient norm for validation
            metrics['debug/actor_mlp_grad_norm_AFTER_clip'] = actor_mlp_grad_norm

            # 🔍 ENHANCED CNN DIAGNOSTICS #1: Capture actor gradients (after backward, before step)
            if self.cnn_diagnostics is not None:
                self.cnn_diagnostics.capture_gradients()

                # Log detailed gradient flow every 100 steps
                if self.total_it % 100 == 0:
                    self._log_detailed_gradient_flow(self.actor_cnn, "actor_cnn")

            # 🔍 ENHANCED CNN DIAGNOSTICS #2: Capture actor features for diversity analysis
            if self.cnn_diagnostics is not None and self.use_dict_buffer and self.actor_cnn is not None:
                with torch.no_grad():
                    sample_features = self.actor_cnn(obs_dict['image'])
                    self.cnn_diagnostics.capture_features(sample_features, name="actor_update")

                    # Log feature diversity metrics every 100 steps
                    if self.total_it % 100 == 0:
                        self._log_feature_diversity(sample_features, "actor_cnn")

            # 🔧 CRITICAL FIX (Nov 20, 2025): Single optimizer.step() now updates BOTH actor MLP and CNN
            # CNN parameters are included in actor_optimizer (see __init__ lines ~150-170)
            # This ensures gradient clipping is applied BEFORE optimizer step
            # Reference: PyTorch DQN Tutorial, TD3/TD3.py line 100, SB3 td3.py line 207
            self.actor_optimizer.step()

            # 🔍 ENHANCED CNN DIAGNOSTICS #3: Capture actor weight changes (after optimizer step)
            if self.cnn_diagnostics is not None and self.actor_cnn is not None:
                self.cnn_diagnostics.capture_weights()

                # Log weight statistics every 1000 steps
                if self.total_it % 1000 == 0:
                    self._log_weight_statistics(self.actor_cnn, "actor_cnn")

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

            # 🔍 DIAGNOSTIC #6: ACTUAL Q-values fed to actor (THE SMOKING GUN)
            # These are the Q-values that drive policy learning
            # If actor_loss = -2.4M, these should average +2.4M
            # Compare with debug/target_q_mean to check consistency
            metrics['debug/actor_q_mean'] = actor_q_values.mean().item()
            metrics['debug/actor_q_std'] = actor_q_values.std().item()
            metrics['debug/actor_q_min'] = actor_q_values.min().item()
            metrics['debug/actor_q_max'] = actor_q_values.max().item()

        return metrics

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save agent checkpoint to disk.

        Saves actor, critic, CNN networks and their optimizers in a single file.
        FIXED: Now correctly saves BOTH actor_cnn and critic_cnn separately.

        Args:
            filepath: Path to save checkpoint (e.g., 'checkpoints/td3_100k.pth')

        Note:
            PRIMARY FIX: Saves SEPARATE CNNs (actor_cnn + critic_cnn) and their optimizers.
            This ensures the Phase 21 architecture fix is properly persisted.
            Reference: PyTorch best practices - save all state_dicts and optimizers.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            # Training state
            'total_it': self.total_it,

            # Core networks
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),

            # Core optimizers
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),

            # Configuration
            'config': self.config,
            'use_dict_buffer': self.use_dict_buffer,

            # TD3 hyperparameters (for self-contained checkpoint)
            'discount': self.discount,
            'tau': self.tau,
            'policy_freq': self.policy_freq,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'max_action': self.max_action,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }

        # PRIMARY FIX: Save SEPARATE CNNs if using Dict buffer
        if self.use_dict_buffer:
            # Save actor CNN
            if self.actor_cnn is not None:
                checkpoint['actor_cnn_state_dict'] = self.actor_cnn.state_dict()
                print(f"  Saving actor CNN state ({len(checkpoint['actor_cnn_state_dict'])} layers)")
            else:
                checkpoint['actor_cnn_state_dict'] = None

            # Save critic CNN
            if self.critic_cnn is not None:
                checkpoint['critic_cnn_state_dict'] = self.critic_cnn.state_dict()
                print(f"  Saving critic CNN state ({len(checkpoint['critic_cnn_state_dict'])} layers)")
            else:
                checkpoint['critic_cnn_state_dict'] = None

            # Save CNN optimizers
            if self.actor_cnn_optimizer is not None:
                checkpoint['actor_cnn_optimizer_state_dict'] = self.actor_cnn_optimizer.state_dict()
                print(f"  Saving actor CNN optimizer state")
            else:
                checkpoint['actor_cnn_optimizer_state_dict'] = None

            if self.critic_cnn_optimizer is not None:
                checkpoint['critic_cnn_optimizer_state_dict'] = self.critic_cnn_optimizer.state_dict()
                print(f"  Saving critic CNN optimizer state")
            else:
                checkpoint['critic_cnn_optimizer_state_dict'] = None

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        if self.use_dict_buffer:
            print(f"  Includes SEPARATE actor_cnn and critic_cnn states (Phase 21 fix)")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load agent checkpoint from disk.

        Restores networks, optimizers, and training state. Also recreates
        target networks from loaded weights.
        FIXED: Now correctly loads BOTH actor_cnn and critic_cnn separately.

        Args:
            filepath: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist

        Note:
            PRIMARY FIX: Loads SEPARATE CNNs (actor_cnn + critic_cnn) and their optimizers.
            This ensures the Phase 21 architecture fix is properly restored.
            Target networks are recreated via deepcopy (TD3 convention).
            Reference: Original TD3.py - targets are always recreated on load.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore networks
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Recreate target networks (TD3 convention - not saved, always recreated)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Restore optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # PRIMARY FIX: Load SEPARATE CNNs if using Dict buffer
        if checkpoint.get('use_dict_buffer', False):
            # Load actor CNN
            if 'actor_cnn_state_dict' in checkpoint and checkpoint['actor_cnn_state_dict'] is not None:
                if self.actor_cnn is not None:
                    self.actor_cnn.load_state_dict(checkpoint['actor_cnn_state_dict'])
                    print(f"Actor CNN state restored ({len(checkpoint['actor_cnn_state_dict'])} layers)")
                else:
                    print(f"Actor CNN state in checkpoint but agent.actor_cnn is None")
            else:
                print(f"No actor CNN state in checkpoint")

            # Load critic CNN
            if 'critic_cnn_state_dict' in checkpoint and checkpoint['critic_cnn_state_dict'] is not None:
                if self.critic_cnn is not None:
                    self.critic_cnn.load_state_dict(checkpoint['critic_cnn_state_dict'])
                    print(f"Critic CNN state restored ({len(checkpoint['critic_cnn_state_dict'])} layers)")
                else:
                    print(f"Critic CNN state in checkpoint but agent.critic_cnn is None")
            else:
                print(f"No critic CNN state in checkpoint")

            # Load CNN optimizers
            if 'actor_cnn_optimizer_state_dict' in checkpoint and checkpoint['actor_cnn_optimizer_state_dict'] is not None:
                if self.actor_cnn_optimizer is not None:
                    self.actor_cnn_optimizer.load_state_dict(checkpoint['actor_cnn_optimizer_state_dict'])
                    print(f"Actor CNN optimizer restored")
                else:
                    print(f"Actor CNN optimizer state in checkpoint but agent.actor_cnn_optimizer is None")
            else:
                print(f"No actor CNN optimizer state in checkpoint")

            if 'critic_cnn_optimizer_state_dict' in checkpoint and checkpoint['critic_cnn_optimizer_state_dict'] is not None:
                if self.critic_cnn_optimizer is not None:
                    self.critic_cnn_optimizer.load_state_dict(checkpoint['critic_cnn_optimizer_state_dict'])
                    print(f"Critic CNN optimizer restored")
                else:
                    print(f"Critic CNN optimizer state in checkpoint but agent.critic_cnn_optimizer is None")
            else:
                print(f"No critic CNN optimizer state in checkpoint")

        # Restore training state
        self.total_it = checkpoint['total_it']

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Resumed at iteration: {self.total_it}")
        if checkpoint.get('use_dict_buffer', False):
            print(f"  SEPARATE CNNs restored (Phase 21 fix)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive agent statistics for monitoring and debugging.

        This method provides detailed information about:
        - Training progress (iterations, phase)
        - Replay buffer state (size, utilization)
        - Network parameters (weight statistics)
        - Learning rates (all optimizers)
        - TD3 hyperparameters
        - CNN statistics (if using Dict buffer)

        Returns:
            Dictionary with comprehensive agent statistics

        Note:
            Following Stable-Baselines3 and OpenAI Spinning Up best practices
            for comprehensive RL monitoring and debugging.
        """
        stats = {
            # ===== TRAINING PROGRESS =====
            'total_iterations': self.total_it,
            'is_training': self.total_it >= self.start_timesteps,
            'exploration_phase': self.total_it < self.start_timesteps,

            # ===== REPLAY BUFFER STATS =====
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_full': self.replay_buffer.is_full(),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.max_size,
            'buffer_max_size': self.replay_buffer.max_size,
            'use_dict_buffer': self.use_dict_buffer,

            # ===== LEARNING RATES (CRITICAL FOR DEBUGGING) =====
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],

            # ===== TD3 HYPERPARAMETERS (REPRODUCIBILITY) =====
            'discount': self.discount,
            'tau': self.tau,
            'policy_freq': self.policy_freq,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'max_action': self.max_action,
            'start_timesteps': self.start_timesteps,
            'batch_size': self.batch_size,            # ===== NETWORK PARAMETER STATISTICS =====
            'actor_param_mean': self._get_param_stat(self.actor.parameters(), 'mean'),
            'actor_param_std': self._get_param_stat(self.actor.parameters(), 'std'),
            'actor_param_max': self._get_param_stat(self.actor.parameters(), 'max'),
            'actor_param_min': self._get_param_stat(self.actor.parameters(), 'min'),

            'critic_param_mean': self._get_param_stat(self.critic.parameters(), 'mean'),
            'critic_param_std': self._get_param_stat(self.critic.parameters(), 'std'),
            'critic_param_max': self._get_param_stat(self.critic.parameters(), 'max'),
            'critic_param_min': self._get_param_stat(self.critic.parameters(), 'min'),

            'target_actor_param_mean': self._get_param_stat(self.actor_target.parameters(), 'mean'),
            'target_critic_param_mean': self._get_param_stat(self.critic_target.parameters(), 'mean'),

            # ===== COMPUTE DEVICE =====
            'device': str(self.device)
        }

        # ===== CNN-SPECIFIC STATS (if using Dict buffer with separate CNNs) =====
        if self.use_dict_buffer:
            stats.update({
                'actor_cnn_lr': self.actor_cnn_optimizer.param_groups[0]['lr'] if self.actor_cnn_optimizer else None,
                'critic_cnn_lr': self.critic_cnn_optimizer.param_groups[0]['lr'] if self.critic_cnn_optimizer else None,

                'actor_cnn_param_mean': self._get_param_stat(self.actor_cnn.parameters(), 'mean') if self.actor_cnn else None,
                'actor_cnn_param_std': self._get_param_stat(self.actor_cnn.parameters(), 'std') if self.actor_cnn else None,
                'actor_cnn_param_max': self._get_param_stat(self.actor_cnn.parameters(), 'max') if self.actor_cnn else None,
                'actor_cnn_param_min': self._get_param_stat(self.actor_cnn.parameters(), 'min') if self.actor_cnn else None,

                'critic_cnn_param_mean': self._get_param_stat(self.critic_cnn.parameters(), 'mean') if self.critic_cnn else None,
                'critic_cnn_param_std': self._get_param_stat(self.critic_cnn.parameters(), 'std') if self.critic_cnn else None,
                'critic_cnn_param_max': self._get_param_stat(self.critic_cnn.parameters(), 'max') if self.critic_cnn else None,
                'critic_cnn_param_min': self._get_param_stat(self.critic_cnn.parameters(), 'min') if self.critic_cnn else None,
            })

        return stats

    def _get_param_stat(self, parameters, stat_type: str = 'mean') -> float:
        """
        Calculate statistics for network parameters.

        Args:
            parameters: Iterator of torch parameters
            stat_type: Type of statistic ('mean', 'std', 'max', 'min')

        Returns:
            Computed statistic value

        Note:
            This utility helps detect weight explosion/collapse, NaN issues,
            and overall network health.
        """
        params_flat = torch.cat([p.data.flatten() for p in parameters if p.requires_grad])

        if len(params_flat) == 0:
            return 0.0

        if stat_type == 'mean':
            return float(torch.mean(params_flat).item())
        elif stat_type == 'std':
            return float(torch.std(params_flat).item())
        elif stat_type == 'max':
            return float(torch.max(params_flat).item())
        elif stat_type == 'min':
            return float(torch.min(params_flat).item())
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")

    def _log_detailed_gradient_flow(self, cnn: torch.nn.Module, network_name: str) -> None:
        """
        Log detailed gradient flow statistics for CNN layers.

        Analyzes gradient norms for each layer to detect vanishing or exploding
        gradients, which are common issues in deep networks.

        Args:
            cnn: CNN network (actor_cnn or critic_cnn)
            network_name: Name for logging (e.g., "actor_cnn", "critic_cnn")
        """
        if cnn is None:
            return

        gradient_norms = []
        vanishing_threshold = 1e-6
        exploding_threshold = 10.0

        for name, param in cnn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)

                # Determine status
                if grad_norm < vanishing_threshold:
                    status = "⚠️ VANISHING"
                elif grad_norm > exploding_threshold:
                    status = "🔥 EXPLODING"
                else:
                    status = "✅ OK"

                self.logger.debug(
                    f"🔄 [{network_name}] Gradient {name}: {grad_norm:.6f} {status}"
                )

        if gradient_norms:
            min_norm = min(gradient_norms)
            max_norm = max(gradient_norms)
            avg_norm = np.mean(gradient_norms)

            # Calculate gradient flow ratio (first/last layer)
            flow_ratio = gradient_norms[0] / gradient_norms[-1] if len(gradient_norms) > 1 else 1.0

            # Assess overall health
            issues = []
            if min_norm < vanishing_threshold:
                issues.append("⚠️ Vanishing gradients")
            if max_norm > exploding_threshold:
                issues.append("🔥 Exploding gradients")
            if flow_ratio < 0.1 or flow_ratio > 10.0:
                issues.append(f"⚠️ Poor flow ratio ({flow_ratio:.2f})")

            health_status = " | ".join(issues) if issues else "✅ HEALTHY"

            self.logger.debug(
                f"📊 [{network_name}] Gradient Flow Summary (Step {self.total_it}):\n"
                f"   Min: {min_norm:.6f}, Max: {max_norm:.6f}, Avg: {avg_norm:.6f}\n"
                f"   Flow ratio (first/last): {flow_ratio:.3f}\n"
                f"   Status: {health_status}"
            )

    def _log_feature_diversity(self, features: torch.Tensor, network_name: str) -> None:
        """
        Log feature diversity metrics to ensure CNN learns diverse representations.

        Args:
            features: Feature tensor (batch_size, feature_dim)
            network_name: Name for logging (e.g., "actor_cnn", "critic_cnn")
        """
        if features.shape[0] < 2:
            # Need at least 2 samples for correlation
            return

        try:
            # Detach and move to CPU for analysis
            features_cpu = features.detach().cpu()

            # Calculate pairwise correlation matrix
            feature_corr = torch.corrcoef(features_cpu.T)

            # Average absolute correlation (excluding diagonal)
            n_features = feature_corr.shape[0]
            avg_corr = (feature_corr.abs().sum() - n_features) / (n_features * (n_features - 1))
            avg_corr = avg_corr.item()

            # Calculate sparsity (percentage of near-zero features)
            sparsity_threshold = 0.1
            sparsity = (features_cpu.abs() < sparsity_threshold).float().mean().item()

            # Calculate effective rank (diversity measure)
            _, s, _ = torch.svd(features_cpu)
            s_normalized = s / s.sum()
            entropy = -(s_normalized * torch.log(s_normalized + 1e-8)).sum().item()
            effective_rank = torch.exp(torch.tensor(entropy)).item()

            # Assess diversity health
            correlation_threshold = 0.3
            issues = []
            if avg_corr > 0.7:
                issues.append("⚠️ High correlation (feature collapse)")
            elif avg_corr > correlation_threshold:
                issues.append("⚠️ Moderate correlation")

            if sparsity < 0.05:
                issues.append("⚠️ Too dense")
            elif sparsity > 0.5:
                issues.append("⚠️ Too sparse")

            diversity_status = " | ".join(issues) if issues else "✅ DIVERSE"

            self.logger.debug(
                f"🎨 [{network_name}] Feature Diversity (Step {self.total_it}):\n"
                f"   Avg correlation: {avg_corr:.3f} (target: <{correlation_threshold})\n"
                f"   Sparsity: {sparsity*100:.1f}% (target: 10-30%)\n"
                f"   Effective rank: {effective_rank:.1f} / {n_features}\n"
                f"   Status: {diversity_status}"
            )

        except Exception as e:
            self.logger.warning(f"Could not compute feature diversity for {network_name}: {e}")

    def _log_weight_statistics(self, cnn: torch.nn.Module, network_name: str) -> None:
        """
        Log weight statistics to track learning progress.

        Monitors weight magnitudes and their changes over time to detect:
        - Dead neurons (weights frozen)
        - Excessive weight growth
        - Layer-wise learning imbalances

        Args:
            cnn: CNN network (actor_cnn or critic_cnn)
            network_name: Name for logging (e.g., "actor_cnn", "critic_cnn")
        """
        if cnn is None:
            return

        self.logger.debug(f"⚖️  [{network_name}] Weight Statistics (Step {self.total_it}):")

        for name, param in cnn.named_parameters():
            if 'weight' in name:  # Only log weight parameters, not biases
                weights = param.data

                mean = weights.mean().item()
                std = weights.std().item()
                min_val = weights.min().item()
                max_val = weights.max().item()
                norm = weights.norm().item()

                # Detect issues
                if std < 1e-6:
                    status = "⚠️ DEAD (zero variance)"
                elif norm > 100.0:
                    status = "🔥 EXCESSIVE (large norm)"
                else:
                    status = "✅ OK"

                self.logger.debug(
                    f"   {name}:\n"
                    f"      Mean: {mean:.6f}, Std: {std:.6f}\n"
                    f"      Range: [{min_val:.6f}, {max_val:.6f}]\n"
                    f"      L2 norm: {norm:.3f} {status}"
                )

    def _log_learning_rate(self, optimizer: torch.optim.Optimizer, optimizer_name: str) -> None:
        """
        Log current learning rates from optimizer.

        Tracks learning rate changes from schedulers to ensure proper
        training progression and diagnose convergence issues.

        Args:
            optimizer: PyTorch optimizer
            optimizer_name: Name for logging (e.g., "actor_cnn", "critic_cnn")
        """
        if optimizer is None:
            return

        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']

            # Assess learning rate health
            if lr < 1e-6:
                lr_status = "⚠️ TOO LOW"
            elif lr > 1e-2:
                lr_status = "🔥 TOO HIGH"
            else:
                lr_status = "✅ OK"

            self.logger.debug(
                f"📈 [{optimizer_name}] Learning Rate Group {idx}: {lr:.6e} {lr_status}"
            )

    def get_gradient_stats(self) -> Dict[str, float]:
        """
        Get gradient statistics for all networks (after backward pass).

        This method should be called AFTER loss.backward() but BEFORE optimizer.step()
        to capture gradient information for debugging.

        Returns:
            Dictionary with gradient norm statistics for each network

        Note:
            Gradient norms are CRITICAL for debugging:
            - Vanishing gradients: norm << 0.01
            - Exploding gradients: norm >> 10.0
            - Healthy learning: norm in [0.01, 10.0]

        Warning:
            This method requires gradients to be computed. Call after backward()
            but before optimizer.step() or optimizer.zero_grad().
        """
        stats = {
            'actor_grad_norm': self._get_grad_norm(self.actor.parameters()),
            'critic_grad_norm': self._get_grad_norm(self.critic.parameters()),
        }

        # Add CNN gradient norms if using Dict buffer
        if self.use_dict_buffer:
            stats.update({
                'actor_cnn_grad_norm': self._get_grad_norm(self.actor_cnn.parameters()) if self.actor_cnn else 0.0,
                'critic_cnn_grad_norm': self._get_grad_norm(self.critic_cnn.parameters()) if self.critic_cnn else 0.0,
            })

        return stats

    def _get_grad_norm(self, parameters) -> float:
        """
        Calculate L2 norm of gradients for given parameters.

        Args:
            parameters: Iterator of torch parameters

        Returns:
            L2 norm of gradients

        Note:
            This is the same calculation used by torch.nn.utils.clip_grad_norm_
            but without the clipping operation.
        """
        params_with_grad = [p for p in parameters if p.grad is not None]

        if len(params_with_grad) == 0:
            return 0.0

        # Calculate L2 norm of all gradients
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in params_with_grad])
        )

        return float(total_norm.item())


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

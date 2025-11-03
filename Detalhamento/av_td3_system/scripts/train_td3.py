#!/usr/bin/env python3
# ALWAYS fetch latest CARLA docs for better context, and Read Contextual paper and official TD3 docs.
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
Date: 2025
"""

import sys
import os
sys.path.insert(0, '/workspace/av_td3_system')

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn  # For CNN weight initialization
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.td3_agent import TD3Agent
from src.environment.carla_env import CARLANavigationEnv
from src.networks.cnn_extractor import NatureCNN


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
        training_config_path: str = "config/training_config.yaml",
        log_dir: str = "data/logs",
        checkpoint_dir: str = "data/checkpoints",
        debug: bool = False,
        device: str = 'cpu'
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
            training_config_path: Path to training config (with scenarios)
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for checkpoints
            debug: Enable visual feedback with OpenCV (for short runs)
            device: Device for TD3 agent ('cpu', 'cuda', or 'auto')
        """
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.scenario = scenario
        self.scenario = scenario
        self.seed = seed
        self.max_timesteps = max_timesteps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.num_eval_episodes = num_eval_episodes
        self.debug = debug

        # Store config paths for creating eval environment
        self.carla_config_path = carla_config_path
        self.agent_config_path = agent_config_path
        self.training_config_path = training_config_path

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

        # Set environment variable for CARLA to use correct scenario
        # (carla_env.py reads this to select from training_config.yaml scenarios list)
        os.environ['CARLA_SCENARIO_INDEX'] = str(scenario)
        print(f"[CONFIG] Set CARLA_SCENARIO_INDEX={scenario}")

        if scenario < len(npc_densities):
            print(f"[CONFIG] Expected NPC count: {npc_densities[scenario]}")
        else:
            print(f"[WARNING] Invalid scenario index, will use default")

        # Initialize environment
        print(f"\n[ENVIRONMENT] Initializing CARLA environment...")
        self.env = CARLANavigationEnv(
            carla_config_path,
            agent_config_path,
            training_config_path  # Fixed: use training_config_path for scenarios
        )
        print(f"[ENVIRONMENT] State space: {self.env.observation_space}")
        print(f"[ENVIRONMENT] Action space: {self.env.action_space}")

        # Initialize agent
        print(f"\n[AGENT] Initializing TD3 agent...")
        # Device selection strategy:
        # - 'cpu': Force CPU (use when CARLA needs GPU, e.g., 6GB RTX 2060)
        # - 'cuda': Force CUDA (use on supercomputers with dedicated training GPUs)
        # - 'auto': Automatically detect best available device
        if device == 'auto':
            agent_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[AGENT] Device set to 'auto', detected: {agent_device}")
        else:
            agent_device = device
            print(f"[AGENT] Device explicitly set to: {agent_device}")

        if agent_device == 'cpu':
            print(f"[AGENT] Running on CPU to reserve GPU memory for CARLA simulator")

        # 🔧 CRITICAL FIX: Initialize SEPARATE CNN instances for actor and critic
        # This prevents gradient interference that was causing training failure (-52k rewards)
        # Reference: Stable-Baselines3 TD3 uses share_features_extractor=False
        print(f"[AGENT] Initializing SEPARATE NatureCNN feature extractors for actor and critic...")

        self.actor_cnn = NatureCNN(
            input_channels=4,  # 4 stacked frames
            num_frames=4,
            feature_dim=512    # Output 512-dim features
        ).to(agent_device)

        self.critic_cnn = NatureCNN(
            input_channels=4,  # 4 stacked frames
            num_frames=4,
            feature_dim=512    # Output 512-dim features
        ).to(agent_device)

        # Initialize weights properly and set to TRAIN mode
        print(f"[AGENT] Initializing CNN weights (Kaiming for ReLU networks)...")
        self._initialize_cnn_weights()  # Kaiming init for both CNNs

        self.actor_cnn.train()   # Enable training mode for actor CNN
        self.critic_cnn.train()  # Enable training mode for critic CNN

        print(f"[AGENT] Actor CNN initialized on {agent_device} (id: {id(self.actor_cnn)})")
        print(f"[AGENT] Critic CNN initialized on {agent_device} (id: {id(self.critic_cnn)})")
        print(f"[AGENT] CNNs are SEPARATE instances: {id(self.actor_cnn) != id(self.critic_cnn)}")
        print(f"[AGENT] CNN architecture: 4×84×84 → Conv layers → 512 features")
        print(f"[AGENT] CNN training mode: ENABLED (weights will be updated during training)")

        # Initialize TD3Agent WITH SEPARATE CNNs for end-to-end training
        self.agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            actor_cnn=self.actor_cnn,   # ← Separate CNN for actor!
            critic_cnn=self.critic_cnn,  # ← Separate CNN for critic!
            use_dict_buffer=True,        # ← Enable DictReplayBuffer!
            config=self.agent_config,
            device=agent_device
        )

        #print(f"[AGENT] CNN passed to TD3Agent for end-to-end training")
        #print(f"[AGENT] DictReplayBuffer enabled for gradient flow")

        # NOTE: CNN optimizer is now managed by TD3Agent (not here)
        #print(f"[AGENT] Initializing NatureCNN feature extractor...")
        #self.cnn_extractor = NatureCNN(
        #    input_channels=4,  # 4 stacked frames
        #    num_frames=4,
        #    feature_dim=512    # Output 512-dim features
        #).to(agent_device)

        # BUG FIX (2025-01-28): CRITICAL - CNN must be trained, not frozen!
        # PREVIOUS BUG: self.cnn_extractor.eval() froze CNN in evaluation mode
        # This caused CNN weights to remain random throughout training, producing
        # meaningless features. TD3 agent trained on random noise, not learned visual representations.
        #
        # FIX: Initialize weights properly and set to TRAIN mode
        #self._initialize_cnn_weights()  # Kaiming init for ReLU networks
        #self.cnn_extractor.train()  # Enable training mode (NOT eval()!)

        print(f"[AGENT] CNN extractor initialized on {agent_device}")
        print(f"[AGENT] CNN passed to TD3Agent for end-to-end training")
        print(f"[AGENT] DictReplayBuffer enabled for gradient flow")

        # NOTE: CNN optimizer is now managed by TD3Agent (not here)

        # Initialize logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scenario_name = f"scenario_{scenario}_npcs_{npc_densities[scenario]}"
        self.log_name = f"TD3_{scenario_name}_{timestamp}"
        self.log_path = self.log_dir / self.log_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_path))

        print(f"[LOGGING] TensorBoard logs: {self.log_path}")

        # Debug mode setup
        if self.debug:
            print(f"\n{'='*70}")
            print(f"[DEBUG MODE ENABLED]")
            print(f"{'='*70}")
            print(f"[DEBUG] Visual feedback enabled (OpenCV display)")
            print(f"[DEBUG] Press 'q' to quit, 'p' to pause/unpause")
            print(f"\n[DEBUG] CNN diagnostics enabled for training monitoring")
            print(f"[DEBUG] Tracking: gradient flow, weight updates, feature statistics")
            print(f"[DEBUG] TensorBoard metrics: cnn_diagnostics/*")
            print(f"[DEBUG] Console output: Every 1000 steps")
            print(f"{'='*70}\n")

            # Enable CNN diagnostics
            self.agent.enable_diagnostics(self.writer)

            # Setup OpenCV window
            self.window_name = "TD3 Training - Debug View"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 600)
            self.paused = False

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

    def _initialize_cnn_weights(self):
        """
        Initialize CNN weights with Kaiming initialization for ReLU activations.

        🔧 UPDATED: Now initializes BOTH actor_cnn and critic_cnn separately.

        BUG FIX (2025-01-28): Explicit weight initialization ensures reproducibility
        and optimal gradient flow for ReLU-based networks. PyTorch defaults use
        Kaiming uniform, but explicit initialization documents our intent clearly.

        Reference:
        - He et al. (2015): "Delving Deep into Rectifiers"
        - PyTorch nn.init docs: https://pytorch.org/docs/stable/nn.init.html
        """
        # Initialize actor CNN
        print(f"[INIT] Initializing Actor CNN weights...")
        for module in self.actor_cnn.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming normal initialization for Conv2d with ReLU
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',  # Preserve variance in forward pass
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Kaiming normal initialization for Linear layers with ReLU
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize critic CNN
        print(f"[INIT] Initializing Critic CNN weights...")
        for module in self.critic_cnn.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming normal initialization for Conv2d with ReLU
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',  # Preserve variance in forward pass
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Kaiming normal initialization for Linear layers with ReLU
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        print("[AGENT] CNN weights initialized with Kaiming normal (optimized for ReLU)")

    def flatten_dict_obs(self, obs_dict, enable_grad=False):
        """
        Flatten Dict observation to 1D array for TD3 agent using CNN feature extraction.

        🔧 UPDATED: Uses actor_cnn for consistency with select_action behavior.

        Args:
            obs_dict: Dictionary with 'image' (4, 84, 84) and 'vector' (23,) keys

        Returns:
            np.ndarray: Flattened state vector of shape (535,)
                        - First 512 elements: CNN-extracted visual features
                        - Last 23 elements: Vector state (velocity, waypoints, etc.)
        """
        # Extract image and convert to PyTorch tensor
        image = obs_dict['image']  # Shape: (4, 84, 84)

        # Convert to tensor and add batch dimension
        # Expected shape: (1, 4, 84, 84)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.agent.device)

        # Extract features using actor's CNN (no gradient tracking needed)
        with torch.no_grad():
            image_features = self.actor_cnn(image_tensor)  # Shape: (1, 512)

        # Convert back to numpy and remove batch dimension
        image_features = image_features.cpu().numpy().squeeze()  # Shape: (512,)

        # Extract vector state
        vector = obs_dict['vector']  # Shape: (23,)

        # Concatenate to final state: [512 CNN features, 23 kinematic/waypoint]
        flat_state = np.concatenate([image_features, vector]).astype(np.float32)

        return flat_state  # Shape: (535,)

    def _visualize_debug(self, obs_dict, action, reward, info, t):
        """
        Display debug visualization using OpenCV.

        Shows:
        - Front camera view
        - Action values (steering, throttle/brake)
        - Reward breakdown
        - Vehicle state
        - Episode info

        Args:
            obs_dict: Current observation dictionary with 'image' and 'vector' keys
            action: Action taken [steering, throttle/brake]
            reward: Reward received
            info: Environment info dict
            t: Current timestep
        """
        try:
            # Extract camera image from observation dict
            # obs_dict['image'] has shape (4, 84, 84) - 4 stacked grayscale frames
            if 'image' not in obs_dict:
                return

            # Get the latest frame from the stack
            latest_frame = obs_dict['image'][-1]  # Shape: (84, 84)

            # Convert from [0, 1] float to [0, 255] uint8
            frame_uint8 = (latest_frame * 255).astype(np.uint8)

            # Resize to larger display size
            frame_resized = cv2.resize(frame_uint8, (800, 600), interpolation=cv2.INTER_LINEAR)

            # Convert grayscale to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

            # Add info overlay
            vehicle_state = info.get('vehicle_state', {})
            reward_breakdown = info.get('reward_breakdown', {})

            # Create info panel
            info_panel = np.zeros((600, 400, 3), dtype=np.uint8)

            # Title
            cv2.putText(info_panel, "TD3 TRAINING - DEBUG", (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

            # Timestep info
            y_offset = 70
            cv2.putText(info_panel, f"Timestep: {t}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"Episode: {self.episode_num}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"Episode Step: {self.episode_timesteps}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Action
            y_offset += 40
            cv2.putText(info_panel, "ACTION:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Steering: {action[0]:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Throttle/Brake: {action[1]:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Vehicle state
            y_offset += 40
            cv2.putText(info_panel, "VEHICLE STATE:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            speed_kmh = vehicle_state.get('velocity', 0) * 3.6
            cv2.putText(info_panel, f"  Speed: {speed_kmh:.1f} km/h", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Lat Dev: {vehicle_state.get('lateral_deviation', 0):.2f} m",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Head Err: {vehicle_state.get('heading_error', 0):.2f} rad",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Reward
            y_offset += 40
            cv2.putText(info_panel, "REWARD:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            reward_color = (0, 255, 0) if reward > 0 else (0, 0, 255)
            cv2.putText(info_panel, f"  Total: {reward:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, reward_color, 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Episode: {self.episode_reward:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Reward breakdown
            y_offset += 30
            cv2.putText(info_panel, "Breakdown:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            for component, (weight, value, weighted) in reward_breakdown.items():
                y_offset += 20
                text = f"  {component[:8]}: {weighted:+.2f}"
                cv2.putText(info_panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Progress info
            y_offset += 30
            cv2.putText(info_panel, "PROGRESS:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            distance_to_goal = info.get('distance_to_goal', 0)
            cv2.putText(info_panel, f"  To Goal: {distance_to_goal:.1f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            progress_pct = info.get('progress_percentage', 0)
            cv2.putText(info_panel, f"  Progress: {progress_pct:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            wp_idx = info.get('current_waypoint_idx', 0)
            cv2.putText(info_panel, f"  Waypoint: {wp_idx}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Collision info
            y_offset += 30
            collision_color = (0, 0, 255) if self.episode_collision_count > 0 else (0, 255, 0)
            cv2.putText(info_panel, f"Collisions: {self.episode_collision_count}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, collision_color, 1)

            # Instructions
            y_offset = 580
            cv2.putText(info_panel, "q:quit  p:pause", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Combine camera and info panel
            combined_frame = np.hstack([display_frame, info_panel])

            # Display
            cv2.imshow(self.window_name, combined_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n[DEBUG] User requested quit (q pressed)")
                cv2.destroyAllWindows()
                self.env.close()
                import sys
                sys.exit(0)
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"[DEBUG] Paused: {self.paused}")

            # Pause handling
            while self.paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    self.paused = False
                    print("[DEBUG] Resumed")
                elif key == ord('q'):
                    print("\n[DEBUG] User requested quit (q pressed)")
                    cv2.destroyAllWindows()
                    self.env.close()
                    import sys
                    sys.exit(0)

        except Exception as e:
            print(f"[DEBUG] Visualization error: {e}")
            # Continue training even if visualization fails

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
        print(f"[TRAINING] Initializing CARLA environment (spawning actors)...")
        print(f"[TRAINING] This may take 1-5 minutes on first reset. Please be patient...")
        print(f"[TRAINING] Connecting to CARLA server...")

        import time
        start_reset = time.time()

        # Get initial observation (Dict) from environment
        # Gymnasium v0.25+ compliance: reset() returns (observation, info) tuple
        obs_dict, reset_info = self.env.reset()

        reset_duration = time.time() - start_reset
        print(f"[TRAINING] Environment initialized successfully in {reset_duration:.1f} seconds!")
        print(f"[TRAINING] Episode {reset_info.get('episode', 1)}: Route {reset_info.get('route_length_m', 0):.0f}m, NPCs {reset_info.get('npc_count', 0)}")
        print(f"[TRAINING] Actors spawned, sensors ready")
        print(f"[TRAINING] Beginning training from timestep 1 to {self.max_timesteps:,}")

        # BUG FIX #14: No flattening! Keep Dict observations for gradient flow
        done = False

        # DEBUG: Verify CNN → TD3 data flow at initialization
        if self.debug:
            print(f"\n{'='*70}")
            print(f"[DEBUG] CNN -> TD3 DATA FLOW VERIFICATION (Initialization)")
            print(f"{'='*70}")
            print(f"[DEBUG] Camera Input:")
            print(f"   Shape: {obs_dict['image'].shape}")  # (4, 84, 84)
            print(f"   Range: [{obs_dict['image'].min():.3f}, {obs_dict['image'].max():.3f}]")
            print(f"   Mean: {obs_dict['image'].mean():.3f}, Std: {obs_dict['image'].std():.3f}")
            print(f"\n[DEBUG] Vector State (Kinematic + Waypoints):")
            print(f"   Shape: {obs_dict['vector'].shape}")  # (23,)
            print(f"   Velocity: {obs_dict['vector'][0]:.3f} m/s")
            print(f"   Lateral Deviation: {obs_dict['vector'][1]:.3f} m")
            print(f"   Heading Error: {obs_dict['vector'][2]:.3f} rad")
            print(f"   Waypoints: {obs_dict['vector'][3:23].shape} (10 waypoints × 2)")
            print(f"\n[DEBUG] Dict Observation Structure:")
            print(f"   Type: {type(obs_dict)}")
            print(f"   Keys: {list(obs_dict.keys())}")
            print(f"   Image shape: {obs_dict['image'].shape}")
            print(f"   Vector shape: {obs_dict['vector'].shape}")
            print(f"{'='*70}\n")

        start_timesteps = self.agent_config.get('algorithm', {}).get('learning_starts', 25000)
        batch_size = self.agent_config.get('algorithm', {}).get('batch_size', 256)

        # Phase logging
        print(f"\n[TRAINING PHASES]")
        print(f"  Phase 1 (Steps 1-{start_timesteps:,}): EXPLORATION (random actions, filling replay buffer)")
        print(f"  Phase 2 (Steps {start_timesteps+1:,}-{self.max_timesteps:,}): LEARNING (policy updates)")
        print(f"  Evaluation every {self.eval_freq:,} steps")
        print(f"  Checkpoints every {self.checkpoint_freq:,} steps")
        print(f"\n[PROGRESS] Training starting now - logging every 100 steps...\n")

        # Flag to track first training update
        first_training_logged = False

        for t in range(1, int(self.max_timesteps) + 1):
            self.episode_timesteps += 1

            # Log every 10 steps to show training is progressing
            if t % 100 == 0:
                phase = "EXPLORATION" if t <= start_timesteps else "LEARNING"
                print(f"[{phase}] Processing step {t:6d}/{self.max_timesteps:,}...", flush=True)

            # Select action based on training phase
            if t < start_timesteps:
                # Exploration phase: BIASED FORWARD exploration
                # BUG FIX (2025-01-28): Previously used env.action_space.sample() which samples
                # throttle/brake uniformly from [-1,1], resulting in E[net_force]=0 (vehicle stationary).
                # Mathematical proof: P(throttle)=0.5, P(brake)=0.5 → E[forward_force]=0.25-0.25=0 N
                #
                # NEW: Biased forward exploration to ensure vehicle moves during data collection:
                # - steering ∈ [-1, 1]: Full random steering (exploration)
                # - throttle ∈ [0, 1]: FORWARD ONLY (no brake during exploration)
                # This ensures vehicle accumulates driving experience instead of staying stationary.
                action = np.array([
                    np.random.uniform(-1, 1),   # Steering: random left/right
                    np.random.uniform(0, 1)      # Throttle: forward only (0=idle, 1=full throttle)
                ])
            else:
                # Learning phase: use policy with exploration noise
                # CURRICULUM LEARNING: Exponential decay of exploration noise
                # Start high (0.3) after exploration phase, decay to baseline (0.1) over 20k steps
                # Formula: noise = noise_min + (noise_max - noise_min) * exp(-decay_rate * steps_since_learning_start)
                noise_min = 0.1  # Baseline noise (original TD3 value)
                noise_max = 0.3  # Initial high exploration after random phase
                decay_steps = 20000  # Decay over 20k learning steps
                decay_rate = 5.0 / decay_steps  # ln(noise_max/noise_min) / decay_steps ≈ 0.00025

                steps_since_learning_start = t - start_timesteps
                current_noise = noise_min + (noise_max - noise_min) * np.exp(-decay_rate * steps_since_learning_start)

                # Log exploration noise to TensorBoard (every 100 steps)
                if t % 100 == 0:
                    self.writer.add_scalar('train/exploration_noise', current_noise, t)

                # BUG FIX #14: Pass Dict observation directly (no flattening!)
                # This enables gradient flow through CNN during training
                action = self.agent.select_action(
                    obs_dict,  # Dict observation {'image': (4,84,84), 'vector': (23,)}
                    noise=current_noise,
                    deterministic=False  # Exploration mode
                )

            # Step environment (get Dict observation)
            next_obs_dict, reward, done, truncated, info = self.env.step(action)

            # BUG FIX #14: No flattening! Store Dict observations directly
            # CNN features will be extracted WITH gradients during training

            # DEBUG: Log CNN features by extracting them temporarily (only for logging)
            if t % 100 == 0 and self.debug:
                # Extract CNN features just for debug logging (with no_grad)
                # Use actor_cnn for consistency
                with torch.no_grad():
                    image_tensor = torch.FloatTensor(next_obs_dict['image']).unsqueeze(0).to(self.agent.device)
                    cnn_features = self.actor_cnn(image_tensor).cpu().numpy().squeeze()

                print(f"\n[DEBUG][Step {t}] CNN Feature Stats:")
                print(f"  L2 Norm: {np.linalg.norm(cnn_features):.3f}")
                print(f"  Mean: {cnn_features.mean():.3f}, Std: {cnn_features.std():.3f}")
                print(f"  Range: [{cnn_features.min():.3f}, {cnn_features.max():.3f}]")
                print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}] (steering, throttle/brake)")

            # Debug visualization (use original Dict observation)
            if self.debug:
                self._visualize_debug(obs_dict, action, reward, info, t)

                # DEBUG: Print detailed step info to terminal every 10 steps
                if t % 10 == 0:
                    vehicle_state = info.get('vehicle_state', {})
                    reward_breakdown = info.get('reward_breakdown', {})

                    # Extract observation data for debugging
                    vector_obs = next_obs_dict.get('vector', np.array([]))
                    image_obs = next_obs_dict.get('image', np.array([]))

                    # Parse vector observation (velocity, lat_dev, heading_error, waypoints[20])
                    velocity = vector_obs[0] if len(vector_obs) > 0 else 0.0
                    lat_dev = vector_obs[1] if len(vector_obs) > 1 else 0.0
                    heading_err = vector_obs[2] if len(vector_obs) > 2 else 0.0

                    # Waypoints are elements [3:23] (10 waypoints × 2 coords = 20 values)
                    waypoints_flat = vector_obs[3:23] if len(vector_obs) >= 23 else []

                    # Image statistics (4 stacked frames, 84×84)
                    if len(image_obs.shape) == 3:  # (4, 84, 84)
                        img_mean = np.mean(image_obs)
                        img_std = np.std(image_obs)
                        img_min = np.min(image_obs)
                        img_max = np.max(image_obs)
                    else:
                        img_mean = img_std = img_min = img_max = 0.0

                    # Reward breakdown (format: reward_breakdown is already the "breakdown" dict)
                    # Each component is a tuple: (weight, raw_value, weighted_value)
                    eff_tuple = reward_breakdown.get('efficiency', (0, 0, 0))
                    lane_tuple = reward_breakdown.get('lane_keeping', (0, 0, 0))
                    comfort_tuple = reward_breakdown.get('comfort', (0, 0, 0))
                    safety_tuple = reward_breakdown.get('safety', (0, 0, 0))
                    progress_tuple = reward_breakdown.get('progress', (0, 0, 0))

                    # Extract weighted values (index 2)
                    eff_reward = eff_tuple[2] if isinstance(eff_tuple, tuple) else 0.0
                    lane_reward = lane_tuple[2] if isinstance(lane_tuple, tuple) else 0.0
                    comfort_reward = comfort_tuple[2] if isinstance(comfort_tuple, tuple) else 0.0
                    safety_reward = safety_tuple[2] if isinstance(safety_tuple, tuple) else 0.0
                    progress_reward = progress_tuple[2] if isinstance(progress_tuple, tuple) else 0.0

                    # Print main debug line
                    print(
                        f"\n[DEBUG Step {t:4d}] "
                        f"Act=[steer:{action[0]:+.3f}, thr/brk:{action[1]:+.3f}] | "
                        f"Rew={reward:+7.2f} | "
                        f"Speed={vehicle_state.get('velocity', 0)*3.6:5.1f} km/h | "
                        f"LatDev={vehicle_state.get('lateral_deviation', 0):+.2f}m | "
                        f"Collisions={self.episode_collision_count}"
                    )

                    # Print reward breakdown
                    print(
                        f"   [Reward] Efficiency={eff_reward:+.2f} | "
                        f"Lane={lane_reward:+.2f} | "
                        f"Comfort={comfort_reward:+.2f} | "
                        f"Safety={safety_reward:+.2f} | "
                        f"Progress={progress_reward:+.2f}"
                    )

                    # Print first 3 waypoints (most relevant)
                    if len(waypoints_flat) >= 6:
                        wp1_x, wp1_y = waypoints_flat[0], waypoints_flat[1]
                        wp2_x, wp2_y = waypoints_flat[2], waypoints_flat[3]
                        wp3_x, wp3_y = waypoints_flat[4], waypoints_flat[5]

                        # Calculate distances
                        dist1 = np.sqrt(wp1_x**2 + wp1_y**2)
                        dist2 = np.sqrt(wp2_x**2 + wp2_y**2)
                        dist3 = np.sqrt(wp3_x**2 + wp3_y**2)

                        print(
                            f"   [Waypoints] (vehicle frame): "
                            f"WP1=[{wp1_x:+6.1f}, {wp1_y:+6.1f}]m (d={dist1:5.1f}m) | "
                            f"WP2=[{wp2_x:+6.1f}, {wp2_y:+6.1f}]m (d={dist2:5.1f}m) | "
                            f"WP3=[{wp3_x:+6.1f}, {wp3_y:+6.1f}]m (d={dist3:5.1f}m)"
                        )

                    # Print image statistics
                    print(
                        f"   [Image] shape={image_obs.shape} | "
                        f"mean={img_mean:.3f} | std={img_std:.3f} | "
                        f"range=[{img_min:.3f}, {img_max:.3f}]"
                    )

                    # Print state vector info
                    print(
                        f"   [State] velocity={velocity:.2f} m/s | "
                        f"lat_dev={lat_dev:+.3f}m | "
                        f"heading_err={heading_err:+.3f} rad ({np.degrees(heading_err):+.1f}°) | "
                        f"vector_dim={len(vector_obs)}"
                    )

            # Track episode metrics
            self.episode_reward += reward
            self.episode_collision_count += info.get('collision_count', 0)
            self.episode_steps_since_collision = info.get('steps_since_collision', 0)

            #  FIX BUG #12: Use ONLY done (terminated) for TD3 bootstrapping
            # Per official TD3 implementation (main.py line 133):
            #   done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
            #
            # With Gymnasium API (v0.26+), the environment provides:
            #   - done: Natural MDP termination (collision, goal) → V(s')=0
            #   - truncated: Time limit → V(s')≠0
            #
            # WRONG (previous): done_bool = float(done or truncated) if self.episode_timesteps < 300 else True
            # CORRECT (now): done_bool = float(done)
            #
            # This ensures TD3 learns correct Q-values:
            #   - If done=True: target_Q = reward + 0 (no future value)
            #   - If truncated=True: target_Q = reward + gamma*V(next_state) (has future value)
            done_bool = float(done)

            # Store Dict observation directly in replay buffer (Bug #13 fix)
            # This enables gradient flow through CNN during training
            # CRITICAL: Store raw Dict observations (NOT flattened states!)
            self.agent.replay_buffer.add(
                obs_dict=obs_dict,        # Current Dict observation {'image': (4,84,84), 'vector': (23,)}
                action=action,
                next_obs_dict=next_obs_dict,  # Next Dict observation
                reward=reward,
                done=done_bool
            )

            # Update observation for next iteration (no state variable needed!)
            obs_dict = next_obs_dict

            # Train agent (only after exploration phase)
            if t > start_timesteps:
                # Log transition to learning phase (only once)
                if not first_training_logged:
                    print(f"\n{'='*70}")
                    print(f"[PHASE TRANSITION] Starting LEARNING phase at step {t:,}")
                    print(f"[PHASE TRANSITION] Replay buffer size: {len(self.agent.replay_buffer):,}")
                    print(f"[PHASE TRANSITION] Policy updates will now begin...")
                    print(f"{'='*70}\n")
                    first_training_logged = True

                metrics = self.agent.train(batch_size=batch_size)

                # Log training metrics every 100 steps
                if t % 100 == 0:
                    self.writer.add_scalar('train/critic_loss', metrics['critic_loss'], t)
                    self.writer.add_scalar('train/q1_value', metrics['q1_value'], t)
                    self.writer.add_scalar('train/q2_value', metrics['q2_value'], t)

                    if 'actor_loss' in metrics:  # Actor updated only on delayed steps
                        self.writer.add_scalar('train/actor_loss', metrics['actor_loss'], t)

                    # Log CNN diagnostics every 100 steps (if debug mode enabled)
                    if self.debug and self.agent.cnn_diagnostics is not None:
                        self.agent.cnn_diagnostics.log_to_tensorboard(t)

                        # Print detailed CNN diagnostics every 1000 steps
                        if t % 1000 == 0:
                            print(f"\n{'='*70}")
                            print(f"[CNN DIAGNOSTICS] Step {t:,}")
                            print(f"{'='*70}")
                            self.agent.print_diagnostics(max_history=1000)
                            print(f"{'='*70}\n")

                # ===== LOG AGENT STATISTICS EVERY 1000 STEPS (RECOMMENDATION 4) =====
                # Following Stable-Baselines3 and OpenAI Spinning Up best practices
                if t % 1000 == 0:
                    agent_stats = self.agent.get_stats()

                    # Training progress
                    self.writer.add_scalar('agent/total_iterations', agent_stats['total_iterations'], t)
                    self.writer.add_scalar('agent/is_training', int(agent_stats['is_training']), t)

                    # Replay buffer
                    self.writer.add_scalar('agent/buffer_utilization', agent_stats['buffer_utilization'], t)

                    # Learning rates (CRITICAL - Phase 22 finding would be visible!)
                    self.writer.add_scalar('agent/actor_lr', agent_stats['actor_lr'], t)
                    self.writer.add_scalar('agent/critic_lr', agent_stats['critic_lr'], t)

                    # CNN learning rates (if using Dict buffer)
                    if agent_stats.get('actor_cnn_lr') is not None:
                        self.writer.add_scalar('agent/actor_cnn_lr', agent_stats['actor_cnn_lr'], t)
                        self.writer.add_scalar('agent/critic_cnn_lr', agent_stats['critic_cnn_lr'], t)

                    # Network parameter statistics (detect weight explosion/collapse)
                    self.writer.add_scalar('agent/actor_param_mean', agent_stats['actor_param_mean'], t)
                    self.writer.add_scalar('agent/actor_param_std', agent_stats['actor_param_std'], t)
                    self.writer.add_scalar('agent/critic_param_mean', agent_stats['critic_param_mean'], t)
                    self.writer.add_scalar('agent/critic_param_std', agent_stats['critic_param_std'], t)

                    # CNN parameter statistics (if using Dict buffer)
                    if agent_stats.get('actor_cnn_param_mean') is not None:
                        self.writer.add_scalar('agent/actor_cnn_param_mean', agent_stats['actor_cnn_param_mean'], t)
                        self.writer.add_scalar('agent/actor_cnn_param_std', agent_stats['actor_cnn_param_std'], t)
                        self.writer.add_scalar('agent/critic_cnn_param_mean', agent_stats['critic_cnn_param_mean'], t)
                        self.writer.add_scalar('agent/critic_cnn_param_std', agent_stats['critic_cnn_param_std'], t)

                    # Print summary of key statistics
                    if t % 5000 == 0:  # Print every 5000 steps
                        print(f"\n{'='*70}")
                        print(f"[AGENT STATISTICS] Step {t:,}")
                        print(f"{'='*70}")
                        print(f"Training Phase: {'LEARNING' if agent_stats['is_training'] else 'EXPLORATION'}")
                        print(f"Buffer Utilization: {agent_stats['buffer_utilization']:.1%}")
                        print(f"Learning Rates:")
                        print(f"  Actor:  {agent_stats['actor_lr']:.6f}")
                        print(f"  Critic: {agent_stats['critic_lr']:.6f}")
                        if agent_stats.get('actor_cnn_lr') is not None:
                            print(f"  Actor CNN:  {agent_stats['actor_cnn_lr']:.6f}")
                            print(f"  Critic CNN: {agent_stats['critic_cnn_lr']:.6f}")
                        print(f"Network Stats:")
                        print(f"  Actor  - mean: {agent_stats['actor_param_mean']:+.6f}, std: {agent_stats['actor_param_std']:.6f}")
                        print(f"  Critic - mean: {agent_stats['critic_param_mean']:+.6f}, std: {agent_stats['critic_param_std']:.6f}")
                        print(f"{'='*70}\n")

            # ALWAYS log progress every 100 steps (not just debug mode)
            if t % 100 == 0:
                phase = "EXPLORATION" if t <= start_timesteps else "LEARNING"
                vehicle_state = info.get('vehicle_state', {})
                speed_kmh = vehicle_state.get('velocity', 0) * 3.6

                print(
                    f"[{phase}] Step {t:6d}/{self.max_timesteps:,} | "
                    f"Episode {self.episode_num:4d} | "
                    f"Ep Step {self.episode_timesteps:4d} | "
                    f"Reward={reward:+7.2f} | "
                    f"Speed={speed_kmh:5.1f} km/h | "
                    f"Buffer={len(self.agent.replay_buffer):7d}/{self.agent.replay_buffer.max_size}"
                )

                # Log step-based metrics every 100 steps (so TensorBoard has data even during exploration)
                self.writer.add_scalar('progress/buffer_size', len(self.agent.replay_buffer), t)
                self.writer.add_scalar('progress/episode_steps', self.episode_timesteps, t)
                self.writer.add_scalar('progress/current_reward', reward, t)
                self.writer.add_scalar('progress/speed_kmh', speed_kmh, t)

                # Flush TensorBoard writer every 100 steps to ensure data is written to disk
                self.writer.flush()            # Episode termination
            if done or truncated:
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

                # Reset episode (get Dict observation)
                # Gymnasium v0.25+ compliance: reset() returns (observation, info) tuple
                # BUG FIX #14: No flattening! Keep Dict for gradient flow
                obs_dict, _ = self.env.reset()
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

        FIXED: Creates a separate evaluation environment to avoid interfering
        with training environment state (RNG, CARLA actors, internal counters).

        Returns:
            Dictionary with evaluation metrics:
            - mean_reward: Average episode reward
            - std_reward: Std dev of episode rewards
            - success_rate: Fraction of successful episodes
            - avg_collisions: Average collisions per episode
            - avg_episode_length: Average episode length
        """
        # FIXED: Create separate eval environment (don't reuse self.env)
        print(f"[EVAL] Creating temporary evaluation environment...")
        eval_env = CARLANavigationEnv(
            self.carla_config_path,
            self.agent_config_path,
            self.training_config_path  # Fixed: use training_config_path for scenarios
        )

        eval_rewards = []
        eval_successes = []
        eval_collisions = []
        eval_lengths = []

        # FIXED: Use max_episode_steps from config, not max_timesteps (total training steps)
        max_eval_steps = self.agent_config.get("training", {}).get("max_episode_steps", 1000)

        for episode in range(self.num_eval_episodes):
            # Gymnasium v0.25+ compliance: reset() returns (observation, info) tuple
            obs_dict, _ = eval_env.reset()  # Use eval_env, not self.env
            # BUG FIX #14: No flattening! Pass Dict directly to select_action
            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < max_eval_steps:
                # Deterministic action (no noise, no exploration)
                action = self.agent.select_action(
                    obs_dict,  # Dict observation
                    deterministic=True  # Evaluation mode
                )
                next_obs_dict, reward, done, truncated, info = eval_env.step(action)  # Use eval_env

                episode_reward += reward
                episode_length += 1
                obs_dict = next_obs_dict

                if truncated:
                    done = True

            # Log if episode hit the safety limit
            if episode_length >= max_eval_steps:
                print(f"[EVAL] Warning: Episode {episode+1} reached max eval steps ({max_eval_steps})")

            eval_rewards.append(episode_reward)
            eval_successes.append(info.get('success', 0))
            eval_collisions.append(info.get('collision_count', 0))
            eval_lengths.append(episode_length)

        # FIXED: Clean up eval environment
        print(f"[EVAL] Closing evaluation environment...")
        eval_env.close()

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
        """Clean up resources with graceful error handling."""
        # Step 1: Close debug windows
        try:
            if self.debug:
                cv2.destroyAllWindows()
                print(f"[DEBUG] OpenCV windows closed")
        except Exception as e:
            print(f"[WARNING] OpenCV cleanup failed: {e}")

        # Step 2: Save final results
        try:
            self.save_final_results()
        except Exception as e:
            print(f"[WARNING] Result saving failed: {e}")
            # Training data still in TensorBoard, so acceptable failure

        # Step 3: Close CARLA environment
        try:
            self.env.close()
        except Exception as e:
            print(f"[WARNING] Environment cleanup failed: {e}")
            # Resource leak, but training is complete

        # Step 4: Close TensorBoard writer
        try:
            self.writer.close()
        except Exception as e:
            print(f"[WARNING] TensorBoard cleanup failed: {e}")
            # Recent events might be lost, but most data already flushed

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
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: OpenCV visualization + CNN diagnostics (gradient flow, weight updates, feature stats). Recommended for short runs to verify learning. TensorBoard: cnn_diagnostics/* metrics. Console: detailed output every 1000 steps.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'auto'],
        help='Device for TD3 agent (cpu/cuda/auto). Use "cpu" for systems with limited GPU memory (CARLA uses GPU), "cuda" for dedicated training GPUs, "auto" to automatically detect.'
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
        checkpoint_dir=args.checkpoint_dir,
        debug=args.debug,
        device=args.device
    )

    trainer.train()


if __name__ == "__main__":
    main()

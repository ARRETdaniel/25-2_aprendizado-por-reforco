"""
DRL Training with CARLA via ROS 2 Bridge

This module implements a Deep Reinforcement Learning (DRL) agent that interacts
with the CARLA simulator through a ROS 2 bridge. It's designed to be run in
a Python 3.12 environment (or any modern Python version).
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import OpenCV for visualization
try:
    import cv2
    HAS_CV2 = True
    logger.info("Successfully imported OpenCV")
except ImportError:
    logger.warning("OpenCV not found, visualization will be disabled")
    HAS_CV2 = False

# Try to import ROS bridge
try:
    from ros_bridge import DRLBridge
    logger.info("Successfully imported ROS bridge")
except ImportError as e:
    logger.error(f"Failed to import ROS bridge: {e}")
    logger.error("Make sure ros_bridge.py is in the same directory")
    sys.exit(1)


@dataclass
class DRLConfig:
    """Configuration for DRL training."""
    # General
    random_seed: int = 42
    checkpoint_dir: str = "./checkpoints/sac_carla"
    log_dir: str = "./logs/sac_carla"
    plot_dir: str = "./plots"

    # Environment
    state_dim: int = 17  # Matches the state dimensions from CARLA
    action_dim: int = 2  # Throttle, steering

    # Training
    num_episodes: int = 100
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    alpha: float = 0.2  # Temperature parameter for exploration
    buffer_size: int = 100000  # Replay buffer size
    min_buffer_size: int = 1000  # Minimum replay buffer size before training
    target_update_interval: int = 1  # Target network update interval

    # Neural Network
    hidden_dim: int = 256

    # Evaluation
    eval_interval: int = 5  # Evaluate every n episodes
    eval_episodes: int = 3  # Number of episodes for evaluation

    # Visualization
    render: bool = True
    render_interval: int = 1  # Render every n steps

    # Saving
    save_interval: int = 10  # Save checkpoint every n episodes


class ReplayBuffer:
    """Experience replay buffer for DRL."""

    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: torch.device):
        """Initialize replay buffer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            buffer_size: Maximum size of buffer
            device: Device to store tensors
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer.

        Args:
            state: State
            action: Action
            reward: Reward
            next_state: Next state
            done: Done flag
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.states[idx]).to(self.device)
        actions = torch.FloatTensor(self.actions[idx]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[idx]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[idx]).to(self.device)
        dones = torch.FloatTensor(self.dones[idx]).to(self.device)

        return states, actions, rewards, next_states, dones


class Actor(nn.Module):
    """Actor network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, log_std_min: float = -20, log_std_max: float = 2):
        """Initialize actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: State tensor

        Returns:
            Tuple of (mean, log_std)
        """
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            state: State tensor

        Returns:
            Tuple of (action, log_prob, tanh_mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)

        # Compute log probability
        log_prob = normal.log_prob(x_t)

        # Apply correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Critic network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Initialize critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (q1, q2)
        """
        sa = torch.cat([state, action], 1)

        q1 = self.q1(sa)
        q2 = self.q2(sa)

        return q1, q2


class SAC:
    """Soft Actor-Critic (SAC) agent."""

    def __init__(self, config: DRLConfig, device: torch.device):
        """Initialize SAC agent.

        Args:
            config: DRL configuration
            device: Device to run on
        """
        self.config = config
        self.device = device

        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Initialize actor network
        self.actor = Actor(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)

        # Initialize critic networks
        self.critic = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        # Initialize target critic network
        self.target_critic = Critic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.update_target_network(tau=1.0)  # Hard update

        # Initialize temperature parameter alpha
        self.log_alpha = torch.tensor(np.log(config.alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = -config.action_dim  # Heuristic

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.state_dim, config.action_dim, config.buffer_size, device)

        # Training metrics
        self.episode_rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []
        self.alphas = []

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.plot_dir, exist_ok=True)

        # Evaluation metrics
        self.eval_rewards = []

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy.

        Args:
            state: State array
            evaluate: Whether to evaluate (use mean) or explore

        Returns:
            Action array
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def update_target_network(self, tau: float):
        """Update target network using polyak averaging.

        Args:
            tau: Polyak averaging coefficient
        """
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_parameters(self, batch_size: int) -> Dict[str, float]:
        """Update the model parameters using a batch of experiences.

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of losses
        """
        # Sample experiences from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        alpha = self.log_alpha.exp()

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pi, log_probs, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update target network
        self.update_target_network(self.config.tau)

        # Record metrics
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())
        self.alpha_losses.append(alpha_loss.item())
        self.alphas.append(alpha.item())

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item()
        }

    def save_checkpoint(self, episode: int):
        """Save model checkpoint.

        Args:
            episode: Current episode
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"sac_episode_{episode}")
        os.makedirs(checkpoint_path, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(checkpoint_path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_path, "critic.pt"))
        torch.save(self.target_critic.state_dict(), os.path.join(checkpoint_path, "target_critic.pt"))
        torch.save(self.log_alpha, os.path.join(checkpoint_path, "log_alpha.pt"))

        # Save optimizer states
        torch.save(self.actor_optimizer.state_dict(), os.path.join(checkpoint_path, "actor_optimizer.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(checkpoint_path, "critic_optimizer.pt"))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(checkpoint_path, "alpha_optimizer.pt"))

        # Save training metrics
        np.save(os.path.join(checkpoint_path, "episode_rewards.npy"), np.array(self.episode_rewards))
        np.save(os.path.join(checkpoint_path, "critic_losses.npy"), np.array(self.critic_losses))
        np.save(os.path.join(checkpoint_path, "actor_losses.npy"), np.array(self.actor_losses))
        np.save(os.path.join(checkpoint_path, "alpha_losses.npy"), np.array(self.alpha_losses))
        np.save(os.path.join(checkpoint_path, "alphas.npy"), np.array(self.alphas))
        np.save(os.path.join(checkpoint_path, "eval_rewards.npy"), np.array(self.eval_rewards))

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.actor.load_state_dict(torch.load(os.path.join(checkpoint_path, "actor.pt")))
        self.critic.load_state_dict(torch.load(os.path.join(checkpoint_path, "critic.pt")))
        self.target_critic.load_state_dict(torch.load(os.path.join(checkpoint_path, "target_critic.pt")))
        self.log_alpha = torch.load(os.path.join(checkpoint_path, "log_alpha.pt"))

        # Load optimizer states
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "actor_optimizer.pt")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "critic_optimizer.pt")))
        self.alpha_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "alpha_optimizer.pt")))

        # Load training metrics
        self.episode_rewards = np.load(os.path.join(checkpoint_path, "episode_rewards.npy")).tolist()
        self.critic_losses = np.load(os.path.join(checkpoint_path, "critic_losses.npy")).tolist()
        self.actor_losses = np.load(os.path.join(checkpoint_path, "actor_losses.npy")).tolist()
        self.alpha_losses = np.load(os.path.join(checkpoint_path, "alpha_losses.npy")).tolist()
        self.alphas = np.load(os.path.join(checkpoint_path, "alphas.npy")).tolist()

        try:
            self.eval_rewards = np.load(os.path.join(checkpoint_path, "eval_rewards.npy")).tolist()
        except:
            self.eval_rewards = []

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def plot_rewards(self):
        """Plot training and evaluation rewards."""
        plt.figure(figsize=(12, 5))

        # Plot training rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.grid(True)

        # Plot evaluation rewards if available
        if len(self.eval_rewards) > 0:
            plt.subplot(1, 2, 2)
            plt.plot(list(range(0, len(self.episode_rewards), self.config.eval_interval)),
                    self.eval_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Evaluation Rewards')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'rewards.png'))
        plt.close()

    def plot_losses(self):
        """Plot training losses."""
        plt.figure(figsize=(15, 5))

        # Plot critic losses
        plt.subplot(1, 3, 1)
        plt.plot(self.critic_losses)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('Critic Losses')
        plt.grid(True)

        # Plot actor losses
        plt.subplot(1, 3, 2)
        plt.plot(self.actor_losses)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.title('Actor Losses')
        plt.grid(True)

        # Plot alpha values
        plt.subplot(1, 3, 3)
        plt.plot(self.alphas)
        plt.xlabel('Update Step')
        plt.ylabel('Alpha')
        plt.title('Temperature Parameter')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'losses.png'))
        plt.close()


class CARLADRLTrainer:
    """Trainer for DRL agent in CARLA simulator."""

    def __init__(self, config: DRLConfig, use_ros: bool = True):
        """Initialize DRL trainer.

        Args:
            config: DRL configuration
            use_ros: Whether to use ROS 2 or file-based communication
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize ROS bridge
        self.ros_bridge = DRLBridge(use_ros=use_ros)

        # Initialize SAC agent
        self.agent = SAC(config, self.device)

        # Initialize visualization
        self.fig = None
        self.ax = None
        if HAS_CV2 and config.render:
            cv2.namedWindow("CARLA Camera", cv2.WINDOW_NORMAL)

    def _reset_environment(self, seed: Optional[int] = None):
        """Reset the CARLA environment.

        Args:
            seed: Optional random seed
        """
        params = {}
        if seed is not None:
            params["seed"] = seed

        # Send reset command via ROS bridge
        self.ros_bridge.publish_control("reset", params)

        # Wait for the environment to reset
        time.sleep(0.5)

        # Get initial observation
        camera, state, _, _, _ = self.ros_bridge.get_latest_observation()

        return state

    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step the CARLA environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Send action via ROS bridge
        self.ros_bridge.publish_action(action)

        # Wait for the environment to step
        time.sleep(0.05)

        # Get observation
        camera, next_state, reward, done, info = self.ros_bridge.get_latest_observation()

        # Visualize camera if available and enabled
        if HAS_CV2 and self.config.render and camera is not None:
            cv2.imshow("CARLA Camera", cv2.cvtColor(camera, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return next_state, reward, done, info

    def train(self, checkpoint_path: Optional[str] = None):
        """Train the DRL agent.

        Args:
            checkpoint_path: Optional path to load checkpoint from
        """
        logger.info("Starting DRL training")

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.agent.load_checkpoint(checkpoint_path)
            start_episode = len(self.agent.episode_rewards)
            logger.info(f"Resuming training from episode {start_episode}")
        else:
            start_episode = 0

        try:
            # Main training loop
            for episode in range(start_episode, self.config.num_episodes):
                # Reset environment
                state = self._reset_environment(seed=episode)
                if state is None:
                    logger.warning("Failed to get initial state, skipping episode")
                    continue

                episode_reward = 0
                episode_steps = 0
                done = False

                # Episode loop
                while not done and episode_steps < self.config.max_steps_per_episode:
                    # Select action
                    action = self.agent.select_action(state)

                    # Take action and observe
                    next_state, reward, done, info = self._step_environment(action)
                    if next_state is None:
                        logger.warning("Failed to get next state, ending episode")
                        break

                    # Add experience to replay buffer
                    self.agent.replay_buffer.add(state, action, reward, next_state, done)

                    # Update state and counters
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1

                    # Update parameters if enough experiences
                    if self.agent.replay_buffer.size >= self.config.min_buffer_size:
                        self.agent.update_parameters(self.config.batch_size)

                    # Check if done
                    if done:
                        break

                # Record episode reward
                self.agent.episode_rewards.append(episode_reward)

                # Log episode results
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

                # Evaluate agent periodically
                if episode > 0 and episode % self.config.eval_interval == 0:
                    eval_reward = self.evaluate(num_episodes=self.config.eval_episodes)
                    self.agent.eval_rewards.append(eval_reward)
                    logger.info(f"Evaluation at episode {episode}: Average Reward = {eval_reward:.2f}")

                # Save checkpoint periodically
                if episode > 0 and episode % self.config.save_interval == 0:
                    self.agent.save_checkpoint(episode)
                    self.agent.plot_rewards()
                    self.agent.plot_losses()

            # Save final checkpoint
            self.agent.save_checkpoint(self.config.num_episodes)
            self.agent.plot_rewards()
            self.agent.plot_losses()

            logger.info("Training complete")

        except KeyboardInterrupt:
            logger.info("Training interrupted")
            # Save checkpoint on interrupt
            self.agent.save_checkpoint(episode)
            self.agent.plot_rewards()
            self.agent.plot_losses()

        finally:
            # Clean up
            self.cleanup()

    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the DRL agent.

        Args:
            num_episodes: Number of episodes for evaluation

        Returns:
            Average reward across episodes
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")

        eval_rewards = []

        for episode in range(num_episodes):
            # Reset environment
            state = self._reset_environment()
            if state is None:
                logger.warning("Failed to get initial state, skipping evaluation episode")
                continue

            episode_reward = 0
            episode_steps = 0
            done = False

            # Episode loop
            while not done and episode_steps < self.config.max_steps_per_episode:
                # Select action (deterministically)
                action = self.agent.select_action(state, evaluate=True)

                # Take action and observe
                next_state, reward, done, info = self._step_environment(action)
                if next_state is None:
                    logger.warning("Failed to get next state, ending evaluation episode")
                    break

                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_steps += 1

                # Check if done
                if done:
                    break

            # Record episode reward
            eval_rewards.append(episode_reward)
            logger.info(f"Evaluation Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

        # Calculate average reward
        avg_reward = np.mean(eval_rewards)
        logger.info(f"Evaluation complete: Average Reward = {avg_reward:.2f}")

        return avg_reward

    def run_trained_agent(self, checkpoint_path: str, num_episodes: int = 5):
        """Run trained agent.

        Args:
            checkpoint_path: Path to checkpoint directory
            num_episodes: Number of episodes to run
        """
        logger.info(f"Running trained agent from {checkpoint_path}")

        # Load checkpoint
        self.agent.load_checkpoint(checkpoint_path)

        try:
            for episode in range(num_episodes):
                # Reset environment
                state = self._reset_environment()
                if state is None:
                    logger.warning("Failed to get initial state, skipping episode")
                    continue

                episode_reward = 0
                episode_steps = 0
                done = False

                # Episode loop
                while not done and episode_steps < self.config.max_steps_per_episode:
                    # Select action (deterministically)
                    action = self.agent.select_action(state, evaluate=True)

                    # Take action and observe
                    next_state, reward, done, info = self._step_environment(action)
                    if next_state is None:
                        logger.warning("Failed to get next state, ending episode")
                        break

                    # Update state and counters
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1

                    # Check if done
                    if done:
                        break

                # Log episode results
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

        except KeyboardInterrupt:
            logger.info("Execution interrupted")

        finally:
            # Clean up
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up")

        # Close visualization windows
        if HAS_CV2 and self.config.render:
            cv2.destroyAllWindows()

        # Shutdown ROS bridge
        if self.ros_bridge is not None:
            self.ros_bridge.shutdown()

        # Close any open matplotlib figures
        plt.close('all')


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CARLA DRL Trainer")
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--no-ros', action='store_true', help='Disable ROS bridge (use file-based communication)')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agent instead of training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create DRL configuration
    config = DRLConfig(
        random_seed=args.seed,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        render=not args.no_render
    )

    # Create DRL trainer
    trainer = CARLADRLTrainer(config, use_ros=not args.no_ros)

    # Run trainer
    if args.evaluate and args.checkpoint is not None:
        trainer.run_trained_agent(args.checkpoint)
    else:
        trainer.train(checkpoint_path=args.checkpoint)

    return 0


if __name__ == '__main__':
    sys.exit(main())

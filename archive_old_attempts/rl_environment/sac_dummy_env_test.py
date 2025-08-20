#!/usr/bin/env python
"""
End-to-end test for SAC using a dummy environment.

This script tests the complete SAC algorithm with a dummy environment
that mimics the structure of the CARLA environment but doesn't require
an actual CARLA server.
"""

import os
import sys
import logging
import numpy as np
import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SimpleSAC after path setup
from rl_environment.simple_sac import SimpleSAC

class DummyEnv:
    """
    A simple environment that mimics the structure of the CARLA environment.
    """

    def __init__(self):
        """Initialize the dummy environment."""
        # Define action and observation spaces
        self.action_space = type('', (), {
            'shape': (3,),
            'low': -1.0,
            'high': 1.0
        })()

        # Define observation space structure
        self.observation_space = type('', (), {
            'spaces': {
                'image': type('', (), {'shape': (84, 84, 3)}),
                'vehicle_state': type('', (), {'shape': (9,)}),
                'navigation': type('', (), {'shape': (3,)}),
                'detections': type('', (), {'shape': (10,)})
            }
        })()

        # Internal state
        self.step_count = 0
        self.max_steps = 100
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.goal = np.random.uniform(-10, 10, size=2)
        self.obstacles = [np.random.uniform(-10, 10, size=2) for _ in range(3)]

    def reset(self):
        """Reset the environment."""
        self.step_count = 0
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.goal = np.random.uniform(-8, 8, size=2)
        self.obstacles = [np.random.uniform(-10, 10, size=2) for _ in range(3)]

        return self._get_observation()

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: NumPy array of shape (3,)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.step_count += 1

        # Extract steering, throttle, brake
        steering = action[0]
        throttle = max(0, action[1])
        brake = max(0, -action[1])

        # Simple physics
        # Update velocity
        acceleration = 2.0 * throttle - 1.0 * brake * np.sign(np.linalg.norm(self.velocity))
        self.velocity[0] += acceleration * np.cos(steering * np.pi)
        self.velocity[1] += acceleration * np.sin(steering * np.pi)

        # Apply drag
        self.velocity *= 0.9

        # Update position
        self.position += self.velocity * 0.1

        # Calculate reward
        distance_to_goal = np.linalg.norm(self.position - self.goal)
        goal_reward = -0.1 * distance_to_goal

        # Obstacle penalties
        obstacle_penalty = 0
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.position - obstacle)
            if distance < 1.0:
                obstacle_penalty -= 1.0

        # Speed reward/penalty
        speed = np.linalg.norm(self.velocity)
        speed_reward = 0.1 * speed if distance_to_goal > 1.0 else -0.1 * speed

        # Total reward
        reward = goal_reward + obstacle_penalty + speed_reward

        # Check if episode is done
        done = (self.step_count >= self.max_steps or
                distance_to_goal < 0.5 or
                obstacle_penalty < -0.5)

        # Additional info
        info = {
            'is_success': distance_to_goal < 0.5,
            'timeout': self.step_count >= self.max_steps,
            'crashed': obstacle_penalty < -0.5
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """Generate observation."""
        # Generate image (random for simplicity)
        image = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)

        # Create vehicle state
        vehicle_state = np.zeros(9, dtype=np.float32)
        vehicle_state[0:2] = self.position
        vehicle_state[2:4] = self.velocity
        vehicle_state[4:6] = self.goal

        # Navigation info (vector to goal)
        navigation = np.zeros(3, dtype=np.float32)
        direction = self.goal - self.position
        navigation[0:2] = direction / (np.linalg.norm(direction) + 1e-6)
        navigation[2] = np.linalg.norm(direction)

        # Detections (obstacle info)
        detections = np.zeros(10, dtype=np.float32)
        for i, obstacle in enumerate(self.obstacles[:3]):
            idx = i * 3
            direction = obstacle - self.position
            detections[idx:idx+2] = direction
            detections[idx+2] = np.linalg.norm(direction)
        detections[9] = len(self.obstacles)

        return {
            'image': image.astype(np.float32),
            'vehicle_state': vehicle_state,
            'navigation': navigation,
            'detections': detections
        }

    def close(self):
        """Close the environment."""
        pass

def plot_rewards(rewards, title="Training Rewards"):
    """Plot rewards over episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    # Create directory for plots
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{title.lower().replace(' ', '_')}.png")
    logger.info(f"Plot saved to ./plots/{title.lower().replace(' ', '_')}.png")

def train_sac(env, agent, n_episodes=100, eval_freq=10):
    """
    Train the SAC agent in the environment.

    Args:
        env: Environment
        agent: SAC agent
        n_episodes: Number of episodes
        eval_freq: Frequency of evaluation

    Returns:
        Dictionary of training statistics
    """
    # Training statistics
    stats = {
        'episode_rewards': [],
        'episode_steps': [],
        'eval_rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'alpha_losses': []
    }

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        done = False
        while not done:
            # Select action
            action = agent.select_action(state)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Process step in agent
            agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_steps += 1

            if episode_steps >= 100:  # Maximum steps per episode
                break

        # Record episode statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_steps'].append(episode_steps)

        # Log progress
        logger.info(f"Episode {episode}/{n_episodes}: " +
                  f"Reward = {episode_reward:.2f}, " +
                  f"Steps = {episode_steps}")

        # Evaluate agent
        if episode % eval_freq == 0:
            eval_reward = evaluate_agent(env, agent, n_episodes=5)
            stats['eval_rewards'].append(eval_reward)
            logger.info(f"Evaluation: Average Reward = {eval_reward:.2f}")

    # Create checkpoint directory and save agent
    os.makedirs("./checkpoints", exist_ok=True)
    agent.save("./checkpoints/sac_dummy_env")

    # Plot rewards
    plot_rewards(stats['episode_rewards'], "Training Rewards")
    plot_rewards(stats['eval_rewards'], "Evaluation Rewards")

    return stats

def evaluate_agent(env, agent, n_episodes=5):
    """
    Evaluate the agent without exploration.

    Args:
        env: Environment
        agent: SAC agent
        n_episodes: Number of evaluation episodes

    Returns:
        Average episode reward
    """
    total_reward = 0

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        done = False
        while not done:
            # Select action without exploration
            action = agent.select_action(state, evaluate=True)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward

            if info.get('is_success', False):
                logger.info("Goal reached!")
                break

            if info.get('crashed', False):
                logger.info("Crashed into obstacle.")
                break

            if info.get('timeout', False):
                logger.info("Episode timed out.")
                break

        total_reward += episode_reward

    return total_reward / n_episodes

def main():
    """Run the end-to-end test."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("Creating dummy environment...")
    env = DummyEnv()

    # Get state dimensions from a sample observation
    state = env.reset()
    state_dims = {key: value.shape for key, value in state.items()}
    action_dim = env.action_space.shape[0]

    logger.info(f"State dimensions: {state_dims}")
    logger.info(f"Action dimension: {action_dim}")

    logger.info("Initializing SAC agent...")
    agent = SimpleSAC(
        state_dims=state_dims,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=10000,
        batch_size=64,
        feature_dim=128,
        hidden_dim=128,
        update_freq=2,
        seed=42
    )

    logger.info("Starting training...")
    stats = train_sac(env, agent, n_episodes=50, eval_freq=10)

    logger.info("Training completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

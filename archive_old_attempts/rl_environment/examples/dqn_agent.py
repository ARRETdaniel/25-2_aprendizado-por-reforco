"""
Minimal Deep Q-Network (DQN) agent example for CARLA.
"""

import os
import sys
import numpy as np
import random
from collections import deque
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import rl_environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError:
    logger.error("Failed to import CarlaEnvWrapper. Make sure the environment is correctly installed.")
    sys.exit(1)


class DQN(nn.Module):
    """
    Deep Q-Network model.
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        # Feature extraction for image (CNN)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size
        self.cnn_output_dim = self._get_conv_output_dim((3, 84, 84))

        # Feature extraction for vehicle state and navigation
        self.fc_features = nn.Sequential(
            nn.Linear(state_dim['vehicle_state'] + state_dim['navigation'] + state_dim['detections'], 128),
            nn.ReLU()
        )

        # Combined features
        self.fc_combined = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 128, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def _get_conv_output_dim(self, shape):
        """Calculate output dimension of conv layers."""
        batch_size = 1
        input_data = torch.zeros(batch_size, *shape)
        output_feat = self.conv(input_data)
        return int(np.prod(output_feat.shape))

    def forward(self, image, vehicle_state, navigation, detections):
        """Forward pass through the network."""
        # Image features
        image_features = self.conv(image)

        # Concatenate other state components
        other_features = torch.cat([vehicle_state, navigation, detections], dim=1)
        other_features = self.fc_features(other_features)

        # Combine all features
        combined = torch.cat([image_features, other_features], dim=1)
        q_values = self.fc_combined(combined)

        return q_values


class DQNAgent:
    """
    Deep Q-Network agent for CARLA.
    """
    def __init__(self, state_dim, action_dim, device='cuda',
                 learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000,
                 batch_size=64, target_update=10):
        """
        Initialize the DQN agent.

        Args:
            state_dim: State dimensions dictionary
            action_dim: Action dimension
            device: Device to run the model on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting value of epsilon
            epsilon_end: Minimum value of epsilon
            epsilon_decay: Decay rate of epsilon
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update: How often to update target network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Initialize Q networks
        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.memory = deque(maxlen=memory_size)

        # Initialize step counter
        self.steps_done = 0

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            # Random action
            return np.array([
                random.uniform(0.0, 1.0),   # Throttle
                random.uniform(0.0, 0.3),   # Brake
                random.uniform(-0.3, 0.3)   # Steering
            ])
        else:
            # Greedy action
            with torch.no_grad():
                # Prepare state for network
                image = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
                vehicle_state = torch.FloatTensor(state['vehicle_state']).unsqueeze(0).to(self.device)
                navigation = torch.FloatTensor(state['navigation']).unsqueeze(0).to(self.device)
                detections = torch.FloatTensor(state['detections']).unsqueeze(0).to(self.device)

                # Get Q-values
                q_values = self.q_network(image, vehicle_state, navigation, detections)

                # Get action with highest Q-value
                action_idx = q_values.argmax(dim=1).item()

                # Convert discrete action to continuous
                # This is a simplified mapping - in practice, you'd define a more complete mapping
                actions = [
                    [0.0, 0.0, 0.0],  # Idle
                    [0.5, 0.0, 0.0],  # Accelerate
                    [1.0, 0.0, 0.0],  # Full acceleration
                    [0.0, 0.5, 0.0],  # Brake
                    [0.0, 1.0, 0.0],  # Full brake
                    [0.5, 0.0, -0.5],  # Accelerate and steer left
                    [0.5, 0.0, 0.5],   # Accelerate and steer right
                    [0.0, 0.0, -0.5],  # Steer left
                    [0.0, 0.0, 0.5]    # Steer right
                ]

                return np.array(actions[action_idx])

    def update_epsilon(self):
        """Update epsilon value with decay."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the agent using experiences from replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)

        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Prepare batch for training
        # (This is simplified - in practice, you'd need to handle the dictionary structure of states)
        # ...

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


def dqn_agent_demo():
    """
    Demo using a DQN agent with the CARLA environment wrapper.
    """
    logger.info("Starting DQN agent demo...")

    # Create the environment
    try:
        env = CarlaEnvWrapper(
            host='localhost',
            port=2000,
            town='Town01',
            fps=30,
            image_size=(84, 84),
            frame_skip=2,
            max_episode_steps=1000
        )
        logger.info("Environment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return

    # Get state and action dimensions
    state_dim = {
        'image': (84, 84, 3),
        'vehicle_state': 9,
        'navigation': 3,
        'detections': 10
    }
    action_dim = 9  # Number of discrete actions

    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device)

    try:
        # Run for a few episodes
        for episode in range(3):
            logger.info(f"Starting episode {episode+1}")

            # Reset the environment
            state = env.reset()
            done = False
            total_reward = 0.0
            step = 0

            # Run episode
            while not done:
                # Select action
                action = agent.select_action(state)

                # Take a step
                next_state, reward, done, info = env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                agent.train()

                total_reward += reward
                step += 1
                state = next_state

                if step % 10 == 0:
                    logger.info(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

            # Update epsilon
            agent.update_epsilon()

            logger.info(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    except Exception as e:
        logger.error(f"Error during demo: {e}")
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        logger.info("Demo ended.")


def main():
    """
    Main function.
    """
    dqn_agent_demo()


if __name__ == "__main__":
    main()

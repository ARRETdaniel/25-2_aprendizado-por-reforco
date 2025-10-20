"""
Experience Replay Buffer for TD3/DDPG

Stores transitions (s, a, s', r, done) and samples random mini-batches for training.
This implementation is based on the original TD3 paper implementation with
adaptations for our CARLA autonomous driving task.

Key Features:
- Circular buffer with fixed maximum capacity
- Random sampling for breaking temporal correlations
- Automatic device management (CPU/CUDA)
- Memory-efficient numpy storage with torch conversion on sampling

Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al. 2018)
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.

    Stores transitions as (state, action, next_state, reward, not_done) tuples
    and samples random mini-batches for network training. The 'not_done' flag
    is the inverse of 'done' for convenient use in Bellman updates.

    Attributes:
        max_size: Maximum number of transitions to store
        ptr: Current write pointer in circular buffer
        size: Current number of stored transitions
        device: PyTorch device for tensor conversion (cuda/cpu)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(1e6),
        device: str = None
    ):
        """
        Initialize replay buffer.

        Args:
            state_dim: Dimension of state space (535 for our setup)
            action_dim: Dimension of action space (2: steering + throttle/brake)
            max_size: Maximum buffer capacity (default: 1 million transitions)
            device: PyTorch device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Preallocate numpy arrays for efficient storage
        # Using float32 to save memory (vs float64)
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        # Set device for tensor conversion
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool
    ) -> None:
        """
        Add a transition to the replay buffer.

        The buffer operates as a circular queue: when full, old transitions
        are overwritten starting from the beginning.

        Args:
            state: Current state observation (535-dim)
            action: Action taken (2-dim: steering, throttle/brake)
            next_state: Next state after taking action (535-dim)
            reward: Reward received from environment
            done: Episode termination flag (True if episode ended)
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)  # Inverse for Bellman update

        # Move pointer and update size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch from the buffer.

        Sampling is uniform random to break temporal correlations in the data.
        All sampled data is converted to PyTorch tensors on the specified device.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, next_states, rewards, not_dones) as torch.Tensor
            - states: (batch_size, state_dim)
            - actions: (batch_size, action_dim)
            - next_states: (batch_size, state_dim)
            - rewards: (batch_size, 1)
            - not_dones: (batch_size, 1) - multiplicative mask for terminal states

        Raises:
            ValueError: If batch_size exceeds current buffer size
        """
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with {self.size} transitions"
            )

        # Sample random indices
        ind = np.random.randint(0, self.size, size=batch_size)

        # Convert to torch tensors and move to device
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer has reached maximum capacity."""
        return self.size >= self.max_size

    def clear(self) -> None:
        """Reset buffer to empty state."""
        self.ptr = 0
        self.size = 0


# Example usage and testing
if __name__ == "__main__":
    print("Testing ReplayBuffer...")

    # Initialize buffer
    buffer = ReplayBuffer(state_dim=535, action_dim=2, max_size=1000)

    # Add some dummy transitions
    for i in range(100):
        state = np.random.randn(535)
        action = np.random.randn(2)
        next_state = np.random.randn(535)
        reward = np.random.randn()
        done = (i % 10 == 0)  # Episode ends every 10 steps

        buffer.add(state, action, next_state, reward, done)

    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer full: {buffer.is_full()}")

    # Sample a batch
    batch = buffer.sample(batch_size=32)
    states, actions, next_states, rewards, not_dones = batch

    print(f"\nSampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Not dones: {not_dones.shape}")
    print(f"  Device: {states.device}")

    print("\n✓ ReplayBuffer tests passed!")

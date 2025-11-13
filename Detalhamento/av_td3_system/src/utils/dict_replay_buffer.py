"""
Dict Observation Replay Buffer for End-to-End CNN Training

This replay buffer stores Dict observations (images + vectors) instead of
pre-computed flattened states. This enables gradient backpropagation through
the CNN feature extractor during TD3 training, allowing end-to-end learning.

Key Differences from Standard ReplayBuffer:
- Stores raw image observations (4×84×84) instead of CNN features (512-dim)
- Stores vector observations (23-dim) separately
- Returns Dict observations as PyTorch tensors (not flattened numpy arrays)
- Enables gradient flow: CNN → Actor/Critic → Backprop → CNN weight updates

Reference: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al. 2018)
           + Multi-modal RL with Dict observation spaces (Stable-Baselines3)
"""

import numpy as np
import torch
from typing import Tuple, Dict


class DictReplayBuffer:
    """
    Experience replay buffer for Dict observations.

    Stores transitions as (obs_dict, action, next_obs_dict, reward, done) where
    obs_dict = {'image': (4,84,84), 'vector': (23,)}.

    This design enables CNN training during TD3 updates by preserving gradient
    flow from actor/critic losses back through the CNN feature extractor.

    Attributes:
        max_size: Maximum number of transitions to store
        ptr: Current write pointer in circular buffer
        size: Current number of stored transitions
        device: PyTorch device for tensor conversion (cuda/cpu)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (4, 84, 84),  # (channels, height, width)
        vector_dim: int = 53,  #  FIX: velocity(1) + lateral_dev(1) + heading_err(1) + waypoints(50 = 25*2)
        action_dim: int = 2,   # steering + throttle/brake
        max_size: int = int(1e6),
        device: str = None
    ):
        """
        Initialize Dict replay buffer.

        Args:
            image_shape: Shape of image observations (C, H, W)
            vector_dim: Dimension of vector state (kinematic + waypoints)
            action_dim: Dimension of action space
            max_size: Maximum buffer capacity (default: 1 million transitions)
            device: PyTorch device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Store dimensions
        self.image_shape = image_shape
        self.vector_dim = vector_dim
        self.action_dim = action_dim

        # Preallocate numpy arrays for efficient storage
        # Images: (max_size, C, H, W) - stored as float32 for memory efficiency
        self.images = np.zeros((max_size, *image_shape), dtype=np.float32)
        self.next_images = np.zeros((max_size, *image_shape), dtype=np.float32)

        # Vectors: (max_size, vector_dim)
        self.vectors = np.zeros((max_size, vector_dim), dtype=np.float32)
        self.next_vectors = np.zeros((max_size, vector_dim), dtype=np.float32)

        # Actions, rewards, dones
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.not_dones = np.zeros((max_size, 1), dtype=np.float32)

        # Set device for tensor conversion
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"DictReplayBuffer initialized:")
        print(f"  Max size: {max_size:,}")
        print(f"  Image shape: {image_shape}")
        print(f"  Vector dim: {vector_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Device: {self.device}")

        # Calculate memory usage
        image_bytes = max_size * np.prod(image_shape) * 4 * 2  # 2 for curr + next
        vector_bytes = max_size * vector_dim * 4 * 2
        other_bytes = max_size * (action_dim + 2) * 4  # actions, rewards, not_dones
        total_mb = (image_bytes + vector_bytes + other_bytes) / (1024 * 1024)
        print(f"  Estimated memory: {total_mb:.1f} MB")

    def add(
        self,
        obs_dict: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs_dict: Dict[str, np.ndarray],
        reward: float,
        done: bool
    ) -> None:
        """
        Add a transition to the replay buffer.

        Args:
            obs_dict: Current observation {'image': (4,84,84), 'vector': (23,)}
            action: Action taken (2-dim: steering, throttle/brake)
            next_obs_dict: Next observation after taking action
            reward: Reward received from environment
            done: Episode termination flag
        """
        # Store image observations
        self.images[self.ptr] = obs_dict['image']
        self.next_images[self.ptr] = next_obs_dict['image']

        # Store vector observations
        self.vectors[self.ptr] = obs_dict['vector']
        self.next_vectors[self.ptr] = next_obs_dict['vector']

        # Store action, reward, done
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1.0 - float(done)

        # Move pointer and update size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch from the buffer.

        Returns Dict observations as PyTorch tensors (NOT numpy arrays) to enable
        gradient flow through CNN during training.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (obs_dict, actions, next_obs_dict, rewards, not_dones):
            - obs_dict: {'image': (B,4,84,84), 'vector': (B,23)} as torch.Tensor
            - actions: (B, 2) as torch.Tensor
            - next_obs_dict: {'image': (B,4,84,84), 'vector': (B,23)} as torch.Tensor
            - rewards: (B, 1) as torch.Tensor
            - not_dones: (B, 1) as torch.Tensor

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
        # IMPORTANT: Keep tensors for gradient flow (not numpy arrays!)
        obs_dict = {
            'image': torch.FloatTensor(self.images[ind]).to(self.device),      # (B, 4, 84, 84)
            'vector': torch.FloatTensor(self.vectors[ind]).to(self.device)     # (B, 23)
        }

        next_obs_dict = {
            'image': torch.FloatTensor(self.next_images[ind]).to(self.device),  # (B, 4, 84, 84)
            'vector': torch.FloatTensor(self.next_vectors[ind]).to(self.device) # (B, 23)
        }

        actions = torch.FloatTensor(self.actions[ind]).to(self.device)         # (B, 2)
        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)         # (B, 1)
        not_dones = torch.FloatTensor(self.not_dones[ind]).to(self.device)     # (B, 1)

        return obs_dict, actions, next_obs_dict, rewards, not_dones

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
    print("Testing DictReplayBuffer...")

    # Initialize buffer
    buffer = DictReplayBuffer(
        image_shape=(4, 84, 84),
        vector_dim=23,
        action_dim=2,
        max_size=1000
    )

    # Add some dummy transitions
    for i in range(100):
        obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        action = np.random.randn(2)
        next_obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        reward = np.random.randn()
        done = (i % 10 == 0)

        buffer.add(obs_dict, action, next_obs_dict, reward, done)

    print(f"\nBuffer size: {len(buffer)}")
    print(f"Buffer full: {buffer.is_full()}")

    # Sample a batch
    obs_dict, actions, next_obs_dict, rewards, not_dones = buffer.sample(batch_size=32)

    print(f"\nSampled batch shapes:")
    print(f"  obs_dict['image']: {obs_dict['image'].shape}")
    print(f"  obs_dict['vector']: {obs_dict['vector'].shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  next_obs_dict['image']: {next_obs_dict['image'].shape}")
    print(f"  next_obs_dict['vector']: {next_obs_dict['vector'].shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Not dones: {not_dones.shape}")

    print(f"\nDevice check:")
    print(f"  obs_dict['image'].device: {obs_dict['image'].device}")
    print(f"  obs_dict['vector'].device: {obs_dict['vector'].device}")

    print(f"\nGradient tracking:")
    print(f"  obs_dict['image'].requires_grad: {obs_dict['image'].requires_grad}")
    print(f"  obs_dict['vector'].requires_grad: {obs_dict['vector'].requires_grad}")

    print("\n✓ DictReplayBuffer tests passed!")

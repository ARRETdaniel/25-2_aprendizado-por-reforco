#!/usr/bin/env python3
"""
Enhanced CARLA PPO Trainer - Optimized for Speed and Aggressive Driving
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import zmq
import msgpack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SpeedOptimizedCarlaEnv(gym.Env):
    """
    Speed-optimized CARLA Environment with aggressive driving rewards.
    """

    def __init__(self, zmq_port=5555, render_mode=None):
        super().__init__()

        self.zmq_port = zmq_port
        self.render_mode = render_mode

        # Initialize ZMQ bridge
        self.context = zmq.Context()

        # Subscriber to receive CARLA data
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.data_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        self.data_socket.connect(f"tcp://localhost:{zmq_port}")

        # Publisher to send actions
        self.action_socket = self.context.socket(zmq.PUB)
        self.action_socket.bind(f"tcp://*:{zmq_port + 1}")

        time.sleep(0.1)
        self.bridge_connected = True

        # Enhanced Action space: [steering, throttle/brake, aggressive_mode]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(
                low=0, high=255,
                shape=(64, 64, 3),
                dtype=np.uint8
            ),
            'vehicle_state': spaces.Box(
                low=np.array([-1000.0, -1000.0, -np.pi, -100.0, -50.0, -50.0]),
                high=np.array([1000.0, 1000.0, np.pi, 100.0, 50.0, 50.0]),
                dtype=np.float32
            )
        })

        # Environment state
        self.step_count = 0
        self.max_episode_steps = 500
        self.episode_reward = 0.0
        self.last_carla_data = None
        self.previous_velocity = 0.0
        self.speed_history = []
        self.stuck_counter = 0  # Initialize stuck counter
        self.has_collision = False  # Initialize collision tracking
        self.total_episodes = 0
        self.total_steps = 0

        print(f"🏎️ Speed-Optimized CARLA Environment initialized (ZMQ port: {zmq_port})")

    def step(self, action):
        """Execute environment step with speed optimization."""
        self.step_count += 1
        self.total_steps += 1

        # Parse enhanced action
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        aggressive_factor = float(np.clip(action[2], 0.0, 1.0))

        # Enhanced throttle processing with aggressive mode - IMPROVED
        if throttle_brake >= 0:
            throttle = throttle_brake

            # CRITICAL: More aggressive baseline throttle
            if throttle > 0:
                throttle = max(throttle, 0.2)  # Minimum baseline throttle

            if aggressive_factor > 0.5:  # Aggressive mode
                throttle = max(throttle, 0.6)  # Higher minimum for aggressive mode
                throttle = min(throttle * 1.3, 1.0)  # 30% boost in aggressive mode

            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
            # Reduce braking effectiveness to encourage speed
            brake = brake * 0.7  # Only 70% braking power

        # Send action to CARLA
        action_sent = False
        if self.bridge_connected:
            try:
                action_data = {
                    'steering': steering,
                    'throttle': throttle,
                    'brake': brake,
                    'aggressive': aggressive_factor > 0.5
                }
                packed_action = msgpack.packb(action_data)
                self.action_socket.send(packed_action)
                action_sent = True
            except Exception as e:
                print(f"⚠️ Action send error: {e}")

        # Get observation
        observation = self._get_observation()

        # Calculate enhanced reward
        reward = self._calculate_speed_reward(observation, action)
        self.episode_reward += reward

        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.step_count >= self.max_episode_steps

        info = {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'bridge_connected': self.bridge_connected,
            'action_sent': action_sent,
            'speed': observation.get('vehicle_state', [0]*6)[3],
            'aggressive_mode': aggressive_factor > 0.5
        }

        return observation, reward, terminated, truncated, info

    def _calculate_speed_reward(self, observation, action):
        """Enhanced reward function with collision penalties and proper speed optimization."""
        try:
            vehicle_state = observation['vehicle_state']
            velocity = vehicle_state[3]  # Forward speed
            acceleration = vehicle_state[4:6]  # acceleration components

            # Get CARLA measurements for collision detection
            measurements = getattr(self, 'last_carla_data', {}).get('measurements', {})

            # CRITICAL: Collision penalties
            collision_penalty = 0.0
            collision_vehicles = measurements.get('collision_vehicles', 0.0)
            collision_pedestrians = measurements.get('collision_pedestrians', 0.0)
            collision_other = measurements.get('collision_other', 0.0)
            intersection_otherlane = measurements.get('intersection_otherlane', 0.0)
            intersection_offroad = measurements.get('intersection_offroad', 0.0)

            # Heavy penalties for collisions
            if collision_vehicles > 0:
                collision_penalty -= 50.0  # Severe penalty for vehicle collision
            if collision_pedestrians > 0:
                collision_penalty -= 100.0  # Extreme penalty for pedestrian collision
            if collision_other > 0:
                collision_penalty -= 25.0  # Penalty for hitting objects
            if intersection_otherlane > 0.1:
                collision_penalty -= 10.0  # Penalty for wrong lane
            if intersection_offroad > 0.1:
                collision_penalty -= 15.0  # Penalty for going off-road

            # Track speed history
            self.speed_history.append(velocity)
            if len(self.speed_history) > 10:
                self.speed_history.pop(0)

            # 1. SPEED REWARDS (primary focus)
            speed_reward = 0.0
            if velocity > 0:
                # Exponential speed reward - INCREASED for better speed optimization
                speed_reward = min(velocity * 0.5, 8.0)  # Up to 8.0 for high speed (increased from 5.0)

                # Bonus for consistent high speed - ENHANCED
                if velocity > 5.0:
                    speed_reward += 1.0  # Lower threshold, early reward
                if velocity > 10.0:
                    speed_reward += 3.0  # High speed bonus (increased from 2.0)
                if velocity > 20.0:
                    speed_reward += 5.0  # Very high speed bonus (increased from 3.0)
                if velocity > 30.0:
                    speed_reward += 7.0  # Extreme speed bonus (NEW)

            # 2. ACCELERATION REWARDS - ENHANCED
            accel_reward = 0.0
            if len(self.speed_history) >= 2:
                speed_change = self.speed_history[-1] - self.speed_history[-2]
                if speed_change > 0:  # Accelerating
                    accel_reward = speed_change * 0.2  # Increased from 0.1

            # 3. AGGRESSIVE DRIVING BONUS - ENHANCED
            steering, throttle_brake, aggressive_factor = action[0], action[1], action[2]
            aggressive_bonus = 0.0
            if aggressive_factor > 0.5:
                aggressive_bonus = 1.0  # Increased from 0.5

                # Extra bonus for high throttle in aggressive mode
                if throttle_brake > 0.6:
                    aggressive_bonus += 0.5  # Increased from 0.3
                if throttle_brake > 0.8:
                    aggressive_bonus += 1.0  # NEW: Extra bonus for maximum throttle

            # 4. REDUCED PENALTIES - but keep steering smooth
            steering_penalty = -abs(steering) * 0.01  # Further reduced from 0.02

            # 5. VELOCITY CONSISTENCY REWARD - ENHANCED
            consistency_reward = 0.0
            if len(self.speed_history) >= 5:
                avg_speed = np.mean(self.speed_history[-5:])
                if avg_speed > 2.0:  # Lower threshold (reduced from 5.0)
                    consistency_reward = 0.3  # Increased from 0.2
                if avg_speed > 10.0:
                    consistency_reward = 0.8  # NEW: Better consistency reward

            # 6. SURVIVAL REWARD - only if no collisions
            survival_reward = 0.02 if collision_penalty == 0 else 0.0

            # 7. FORWARD MOVEMENT BONUS
            forward_bonus = max(0, velocity * 0.1)  # Bonus just for moving forward

            total_reward = (speed_reward + accel_reward + aggressive_bonus +
                          steering_penalty + survival_reward + consistency_reward +
                          forward_bonus + collision_penalty)

            # Store collision info for termination check
            self.has_collision = (collision_vehicles > 0 or collision_pedestrians > 0 or
                                collision_other > 0 or intersection_offroad > 0.2)

            self.previous_velocity = velocity
            return total_reward

        except Exception as e:
            print(f"❌ Reward calculation error: {e}")
            return 0.0

    def _get_observation(self):
        """Get observation from CARLA via ZMQ."""
        try:
            # Try to receive CARLA data
            if self.bridge_connected:
                try:
                    packed_data = self.data_socket.recv(zmq.NOBLOCK)
                    carla_data = msgpack.unpackb(packed_data, raw=False)
                    return self._process_carla_data(carla_data)
                except zmq.Again:
                    pass  # No data available
                except Exception as e:
                    print(f"⚠️ ZMQ receive error: {e}")

            # Use last data or dummy
            if self.last_carla_data:
                return self._process_carla_data(self.last_carla_data)
            else:
                return self._get_dummy_observation()

        except Exception as e:
            print(f"❌ Observation error: {e}")
            return self._get_dummy_observation()

    def _process_carla_data(self, carla_data):
        """Process CARLA data into observation format."""
        try:
            # Extract camera data
            sensors = carla_data.get('sensors', {})
            camera_data = None

            for sensor_name, sensor_value in sensors.items():
                if 'Camera' in sensor_name:
                    if isinstance(sensor_value, list):
                        camera_array = np.array(sensor_value, dtype=np.uint8)
                        if len(camera_array.shape) == 3 and camera_array.shape[2] == 3:
                            if camera_array.shape[:2] == (64, 64):
                                camera_data = camera_array
                                break
                        elif len(camera_array.shape) == 1:
                            try:
                                camera_data = camera_array.reshape(64, 64, 3)
                                break
                            except:
                                continue

            if camera_data is None:
                camera_data = np.ones((64, 64, 3), dtype=np.uint8) * 128

            # Extract vehicle state
            measurements = carla_data.get('measurements', {})
            position = measurements.get('position', [0.0, 0.0, 0.0])
            rotation = measurements.get('rotation', [0.0, 0.0, 0.0])
            velocity = measurements.get('velocity', 0.0)
            acceleration = measurements.get('acceleration', [0.0, 0.0, 0.0])

            vehicle_state = np.array([
                position[0],      # x position
                position[1],      # y position
                rotation[1],      # yaw
                velocity,         # forward speed (enhanced range)
                acceleration[0],  # acceleration x
                acceleration[1]   # acceleration y
            ], dtype=np.float32)

            self.last_carla_data = carla_data

            return {
                'camera': camera_data,
                'vehicle_state': vehicle_state
            }

        except Exception as e:
            print(f"❌ Error processing CARLA data: {e}")
            return self._get_dummy_observation()

    def _get_dummy_observation(self):
        """Get dummy observation when CARLA is not available."""
        return {
            'camera': np.ones((64, 64, 3), dtype=np.uint8) * 128,
            'vehicle_state': np.zeros(6, dtype=np.float32)
        }

    def _is_terminated(self, observation):
        """Check if episode is terminated - ENHANCED with collision detection."""
        try:
            vehicle_state = observation['vehicle_state']
            velocity = vehicle_state[3]

            # CRITICAL: Reset on collision
            if hasattr(self, 'has_collision') and self.has_collision:
                print("💥 Episode terminated due to collision!")
                return True

            # Initialize stuck counter if not exists
            if not hasattr(self, 'stuck_counter'):
                self.stuck_counter = 0

            # Give the car time to start moving (skip stuck check for first 10 steps)
            if self.step_count > 10:
                # Reset if stuck (moving very slowly for too long)
                if velocity < 0.5:  # Increased threshold from 0.1 to 0.5
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0

                # Terminate if stuck for 30 steps (reduced from 50)
                if self.stuck_counter > 30:
                    print("🐌 Episode terminated due to being stuck!")
                    return True

            # Terminate if episode is too long (encourage efficiency)
            if self.step_count > self.max_episode_steps:
                print("⏰ Episode terminated due to max steps!")
                return True

            return False

        except Exception as e:
            print(f"❌ Termination check error: {e}")
            return False

    def reset(self, **kwargs):
        """Reset environment - ENHANCED with collision tracking."""
        self.step_count = 0
        self.episode_reward = 0.0
        self.previous_velocity = 0.0
        self.speed_history = []
        self.stuck_counter = 0
        self.has_collision = False  # CRITICAL: Initialize collision tracking
        self.total_episodes += 1

        # Print episode statistics
        if self.total_episodes > 1:
            print(f"🏁 Episode {self.total_episodes-1} completed: {self.step_count} steps, reward: {self.episode_reward:.2f}")

        # Get initial observation
        initial_obs = self._get_observation()

        return initial_obs, {}

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'data_socket'):
            self.data_socket.close()
        if hasattr(self, 'action_socket'):
            self.action_socket.close()
        if hasattr(self, 'context'):
            self.context.term()


class SpeedProgressCallback(BaseCallback):
    """Callback to monitor speed optimization progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_speeds = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log speed information
        if 'speed' in self.locals.get('infos', [{}])[0]:
            speed = self.locals['infos'][0]['speed']
            if len(self.episode_speeds) == 0 or speed > 0:
                self.episode_speeds.append(speed)

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_speeds) > 0:
            avg_speed = np.mean(self.episode_speeds[-100:])  # Last 100 steps
            max_speed = np.max(self.episode_speeds[-100:])
            print(f"🏎️ Speed Stats - Avg: {avg_speed:.2f} km/h | Max: {max_speed:.2f} km/h")
            self.episode_speeds = []


def create_speed_optimized_env():
    """Create speed-optimized environment."""
    return SpeedOptimizedCarlaEnv(zmq_port=5555)


def train_speed_optimized_ppo():
    """Train PPO with speed optimization."""
    print("🏎️ Starting Speed-Optimized CARLA PPO Training")
    print("=" * 60)

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")

    # Create environment
    env = DummyVecEnv([create_speed_optimized_env])

    # Create feature extractor for multi-modal input
    class SpeedOptimizedExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=1030):  # 1024 + 6
            super().__init__(observation_space, features_dim)

            # CNN for camera input
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024, 1024),
                nn.ReLU()
            )

            # FC layer for combining features
            self.fc = nn.Sequential(
                nn.Linear(1030, 512),  # 1024 + 6
                nn.ReLU(),
                nn.Linear(512, features_dim),
                nn.ReLU()
            )

        def forward(self, observations):
            # Process camera
            camera = observations['camera']
            if isinstance(camera, np.ndarray):
                camera = torch.from_numpy(camera).float()
            else:
                camera = camera.float()

            camera = camera / 255.0

            if len(camera.shape) == 3:
                camera = camera.unsqueeze(0)

            if camera.shape[3] == 3:
                camera = camera.permute(0, 3, 1, 2)

            cnn_features = self.cnn(camera)

            # Process vehicle state
            vehicle_state = observations['vehicle_state']
            if isinstance(vehicle_state, np.ndarray):
                vehicle_state = torch.from_numpy(vehicle_state).float()
            else:
                vehicle_state = vehicle_state.float()

            if len(vehicle_state.shape) == 1:
                vehicle_state = vehicle_state.unsqueeze(0)

            # Combine features
            combined = torch.cat([cnn_features, vehicle_state], dim=1)
            features = self.fc(combined)

            return features

    # Optimized PPO policy
    policy_kwargs = dict(
        features_extractor_class=SpeedOptimizedExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[256, 256],  # Larger network
        activation_fn=nn.ReLU,
    )

    # Create PPO model with speed-optimized hyperparameters
    model = PPO(
        'MultiInputPolicy',
        env,
        learning_rate=5e-4,      # Higher learning rate for faster learning
        n_steps=1024,            # Longer rollouts
        batch_size=128,          # Larger batches
        n_epochs=20,             # More training epochs
        gamma=0.99,              # Standard discount
        gae_lambda=0.95,         # Standard GAE
        clip_range=0.3,          # Slightly larger clip range
        ent_coef=0.01,           # Higher entropy for exploration
        vf_coef=0.5,             # Standard value function coefficient
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log="./logs/speed_optimized_carla/"
    )

    print(f"🖥️ Model device: {model.device}")

    # Create callback
    callback = SpeedProgressCallback()

    # Training
    total_timesteps = 50000  # Extended training
    print(f"🎯 Training target: {total_timesteps:,} timesteps")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # Disable progress bar to avoid dependency issues
        )

        # Save model
        model_path = "models/speed_optimized_carla_ppo"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"💾 Model saved to {model_path}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        model_path = "models/speed_optimized_carla_ppo_interrupted"
        model.save(model_path)
        print(f"💾 Partial model saved to {model_path}")

    # Cleanup
    env.close()
    print("🏁 Speed-optimized training completed!")


if __name__ == "__main__":
    print("🏎️ Enhanced CARLA Speed Optimization Trainer")
    print("🔗 Ensure CARLA ZMQ client is running:")
    print("   py -3.6 carla_client_py36/carla_zmq_client.py")
    print()

    train_speed_optimized_ppo()

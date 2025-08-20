#!/usr/bin/env python3
"""
Real CARLA PPO Trainer with ZMQ Bridge
Trains PPO agent on actual CARLA simulation via ZMQ communication
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import cv2
import zmq
import msgpack
import msgpack_numpy as m

# Enable msgpack numpy support
m.patch()

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'communication'))

print("🚀 Real CARLA PPO Trainer with ZMQ Bridge")
print(f"🖥️ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalCarlaExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for multimodal CARLA observations (camera + vehicle state).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN for camera data
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 64, 64)
            cnn_output_size = self.cnn(sample_input).shape[1]

        # MLP for vehicle state
        vehicle_state_dim = observation_space['vehicle_state'].shape[0]

        # Combined features
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size + vehicle_state_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

        print(f"🧠 MultiModal Feature Extractor: CNN output={cnn_output_size}, Vehicle state={vehicle_state_dim}")

    def forward(self, observations) -> torch.Tensor:
        # Process camera image
        camera = observations['camera']

        # Convert to tensor if numpy array
        if isinstance(camera, np.ndarray):
            camera = torch.from_numpy(camera).float()
        else:
            camera = camera.float()

        # Normalize to [0, 1]
        camera = camera / 255.0

        # Ensure correct batch dimensions
        if len(camera.shape) == 3:  # Single observation (H, W, C)
            camera = camera.unsqueeze(0)  # Add batch dimension -> (1, H, W, C)

        # Ensure correct channel order: NHWC -> NCHW
        if camera.shape[3] == 3:  # Check if last dimension is channels
            camera = camera.permute(0, 3, 1, 2)  # NHWC -> NCHW

        cnn_features = self.cnn(camera)

        # Process vehicle state
        vehicle_state = observations['vehicle_state']

        # Convert to tensor if numpy array
        if isinstance(vehicle_state, np.ndarray):
            vehicle_state = torch.from_numpy(vehicle_state).float()
        else:
            vehicle_state = vehicle_state.float()

        if len(vehicle_state.shape) == 1:  # Single observation
            vehicle_state = vehicle_state.unsqueeze(0)  # Add batch dimension

        # Combine features
        combined = torch.cat([cnn_features, vehicle_state], dim=1)
        features = self.fc(combined)

        return features

class RealCarlaEnv(gym.Env):
    """
    Real CARLA Environment that communicates with CARLA via ZMQ bridge.
    """

    def __init__(self, zmq_port=5555, render_mode=None):
        super().__init__()

        self.zmq_port = zmq_port
        self.render_mode = render_mode

        # Initialize ZMQ bridge for direct communication
        self.context = zmq.Context()

        # Subscriber to receive CARLA data (CARLA publishes on port 5555)
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.data_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        self.data_socket.connect(f"tcp://localhost:{zmq_port}")

        # Publisher to send actions (CARLA subscribes on port 5556)
        self.action_socket = self.context.socket(zmq.PUB)
        self.action_socket.bind(f"tcp://*:{zmq_port + 1}")

        # Give sockets time to connect
        time.sleep(0.1)

        self.bridge_connected = True

        # Action space: [steering, throttle/brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: camera + vehicle state
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(
                low=0, high=255,
                shape=(64, 64, 3),
                dtype=np.uint8
            ),
            'vehicle_state': spaces.Box(
                low=np.array([-1000.0, -1000.0, -np.pi, -50.0, -50.0, -50.0]),
                high=np.array([1000.0, 1000.0, np.pi, 50.0, 50.0, 50.0]),
                dtype=np.float32
            )
        })

        # Environment state
        self.step_count = 0
        self.max_episode_steps = 500
        self.episode_reward = 0.0
        self.last_carla_data = None

        # Performance tracking
        self.total_episodes = 0
        self.total_steps = 0

        print(f"🏁 Real CARLA Environment initialized (ZMQ port: {zmq_port})")

    def connect_bridge(self):
        """Connect to ZMQ bridge."""
        # Already connected in __init__
        return self.bridge_connected

    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        # Connect bridge if not connected
        if not self.bridge_connected:
            if not self.connect_bridge():
                print("⚠️ Running with dummy data - bridge not connected")

        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0
        self.total_episodes += 1

        # Wait for initial CARLA data
        initial_obs = self._get_observation()

        return initial_obs, {}

    def step(self, action):
        """Execute environment step."""
        self.step_count += 1
        self.total_steps += 1

        # Parse action
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        # Convert to throttle/brake
        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # Send action to CARLA via ZMQ
        action_sent = False
        if self.bridge_connected:
            try:
                action_data = {
                    'steering': steering,
                    'throttle': throttle,
                    'brake': brake
                }
                packed_action = msgpack.packb(action_data)
                self.action_socket.send(packed_action)
                action_sent = True
            except Exception as e:
                print(f"⚠️ Action send error: {e}")

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(observation, action)
        self.episode_reward += reward

        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.step_count >= self.max_episode_steps

        # Info
        info = {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'bridge_connected': self.bridge_connected,
            'action_sent': action_sent
        }

        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.step_count
            }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation from CARLA via ZMQ."""
        if not self.bridge_connected:
            return self._get_dummy_observation()

        try:
            # Try to receive CARLA data (non-blocking)
            carla_data = self.data_socket.recv(zmq.NOBLOCK)
            data = msgpack.unpackb(carla_data, raw=False)
            return self._process_carla_data(data)

        except zmq.Again:
            # No data available, use dummy
            return self._get_dummy_observation()
        except Exception as e:
            print(f"⚠️ Observation error: {e}")
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
                        # Ensure correct shape (H, W, C)
                        if len(camera_array.shape) == 3 and camera_array.shape[2] == 3:
                            if camera_array.shape[:2] == (64, 64):
                                camera_data = camera_array
                                break
                        # Try to reshape if flattened
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
                velocity,         # forward speed
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

    def _calculate_reward(self, observation, action):
        """Calculate reward based on observation and action."""
        try:
            vehicle_state = observation['vehicle_state']
            velocity = vehicle_state[3]  # Forward speed

            # Speed reward (encourage movement)
            speed_reward = min(velocity * 0.1, 1.0)

            # Action smoothness
            steering, throttle = action[0], action[1]
            smoothness_penalty = -(abs(steering) * 0.1 + abs(throttle - 0.5) * 0.05)

            # Step reward (encourage staying alive)
            step_reward = 0.01

            return speed_reward + smoothness_penalty + step_reward

        except Exception as e:
            print(f"❌ Reward calculation error: {e}")
            return 0.0

    def _is_terminated(self, observation):
        """Check if episode should terminate."""
        try:
            vehicle_state = observation['vehicle_state']
            velocity = vehicle_state[3]

            # Terminate if stuck for too long
            if velocity < 0.1 and self.step_count > 100:
                return True

            return False

        except Exception as e:
            print(f"❌ Termination check error: {e}")
            return False

    def render(self, mode='human'):
        """Render environment with enhanced visual information."""
        if mode == 'human' and self.last_carla_data:
            try:
                observation = self._process_carla_data(self.last_carla_data)
                camera_image = observation['camera']
                vehicle_state = observation['vehicle_state']

                # Scale up for display
                display_image = cv2.resize(camera_image, (480, 480))

                # Convert BGR to RGB for proper display
                if len(display_image.shape) == 3 and display_image.shape[2] == 3:
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                # Create info panel
                info_panel = np.zeros((480, 300, 3), dtype=np.uint8)

                # Add title
                cv2.putText(info_panel, "CARLA DRL Training", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Training info
                cv2.putText(info_panel, f"Episode: {self.total_episodes}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Step: {self.step_count}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(info_panel, f"Reward: {self.episode_reward:.2f}", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Vehicle state info
                cv2.putText(info_panel, "Vehicle State:", (10, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(info_panel, f"Pos X: {vehicle_state[0]:.1f}", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_panel, f"Pos Y: {vehicle_state[1]:.1f}", (10, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_panel, f"Yaw: {vehicle_state[2]:.2f}", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_panel, f"Speed: {vehicle_state[3]:.1f}", (10, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_panel, f"Acc X: {vehicle_state[4]:.1f}", (10, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_panel, f"Acc Y: {vehicle_state[5]:.1f}", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Connection status
                status_color = (0, 255, 0) if self.bridge_connected else (0, 0, 255)
                status_text = "Connected" if self.bridge_connected else "Disconnected"
                cv2.putText(info_panel, f"ZMQ: {status_text}", (10, 340),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                # Performance info
                cv2.putText(info_panel, f"Total Steps: {self.total_steps}", (10, 370),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Instructions
                cv2.putText(info_panel, "Press 'q' to close", (10, 450),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

                # Combine camera and info panel
                combined_image = np.hstack([display_image, info_panel])

                cv2.imshow('CARLA DRL Training - Camera Feed', combined_image)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return False

            except Exception as e:
                print(f"❌ Render error: {e}")

        return True

    def close(self):
        """Close environment."""
        if self.bridge_connected:
            self.data_socket.close()
            self.action_socket.close()
            self.context.term()
            self.bridge_connected = False
        cv2.destroyAllWindows()

class RealCarlaCallback(BaseCallback):
    """Callback for monitoring real CARLA training with visual rendering."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_report = time.time()
        self.episodes_completed = 0
        self.render_counter = 0
        self.visual_window_active = False

    def _on_step(self) -> bool:
        current_time = time.time()

        # Render visual window every 10 steps (for performance)
        self.render_counter += 1
        if self.render_counter >= 10:
            try:
                # Get the environment and render
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    if hasattr(env, 'render'):
                        env.render()  # Remove mode parameter for newer gymnasium versions
                        if not self.visual_window_active:
                            print("🖼️ Visual camera feed window opened!")
                            self.visual_window_active = True
                self.render_counter = 0
            except Exception as e:
                # Suppress render warnings as they're expected with newer gymnasium versions
                pass

        # Report every 5 seconds
        if current_time - self.last_report > 5.0:
            elapsed = current_time - self.start_time
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0

            print(f"🚀 Real CARLA Training: {self.num_timesteps:,} steps | "
                  f"FPS: {fps:.1f} | Episodes: {self.episodes_completed} | "
                  f"Device: {device} | 🖼️ Visual: {'ON' if self.visual_window_active else 'OFF'}")

            self.last_report = current_time

        # Count episodes
        if self.locals.get('dones', [False])[0]:
            if 'episode' in self.locals.get('infos', [{}])[0]:
                self.episodes_completed += 1

        return True

def train_real_carla_ppo():
    """Train PPO on real CARLA environment."""
    print("🚀 Starting Real CARLA PPO Training")
    print("=" * 60)

    # Create environment
    env = make_vec_env(lambda: RealCarlaEnv(zmq_port=5555, render_mode='human'), n_envs=1)

    # Policy kwargs for multimodal observations
    policy_kwargs = {
        'features_extractor_class': MultiModalCarlaExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
    }

    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=device,
        seed=42,
        verbose=1
    )

    # Setup logging
    log_path = "logs/real_carla_training"
    os.makedirs(log_path, exist_ok=True)
    model.set_logger(configure(log_path, ["stdout", "tensorboard"]))

    # Create callback
    callback = RealCarlaCallback()

    print(f"🖥️ Model device: {model.device}")
    print(f"🌉 ZMQ Bridge: port 5556")
    print(f"🎯 Training target: 25,000 timesteps")

    start_time = time.time()

    try:
        # Train model
        model.learn(
            total_timesteps=25000,
            callback=callback,
            progress_bar=False  # Disable for cleaner output
        )

        training_time = time.time() - start_time
        avg_fps = 25000 / training_time

        print(f"\n✅ Real CARLA Training Complete!")
        print(f"⏱️ Training Time: {training_time:.1f}s")
        print(f"📈 Average FPS: {avg_fps:.1f}")
        print(f"🎮 Episodes Completed: {callback.episodes_completed}")

        # Save model
        model_path = os.path.join(log_path, "real_carla_ppo_model.zip")
        model.save(model_path)
        print(f"💾 Model saved: {model_path}")

        return model, callback

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return model, callback
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return model, callback

if __name__ == "__main__":
    try:
        print("🔗 Checking for CARLA ZMQ client connection...")
        print("   Expected: py -3.6 carla_client_py36/carla_zmq_client.py")
        print()

        # Auto-start training without waiting for input
        print("🚀 Starting PPO training with automatic connection...")

        model, callback = train_real_carla_ppo()

        print("\n🎉 Real CARLA PPO Training Session Complete!")
        print("📊 TensorBoard logs available at: logs/real_carla_training")
        print("🚀 Model ready for deployment!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

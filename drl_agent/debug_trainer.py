#!/usr/bin/env python3
"""
Debug version of CARLA DRL Trainer with verbose output
"""

import os
import sys
import numpy as np
import zmq
import msgpack
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import torch

class DebugCARLAEnv(gym.Env):
    def __init__(self, zmq_port=5555):
        super().__init__()
        
        # Action space: [steering, throttle_brake, aggressive_factor]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(
                low=0, high=255,
                shape=(84, 84, 3),
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
        self.stuck_counter = 0
        self.has_collision = False
        self.total_episodes = 0
        self.last_carla_data = None
        self.previous_velocity = 0.0
        
        # ZMQ setup
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect(f"tcp://localhost:{zmq_port}")
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.data_socket.setsockopt(zmq.RCVTIMEO, 100)
        
        self.action_socket = self.context.socket(zmq.PUSH)
        self.action_socket.connect(f"tcp://localhost:{zmq_port + 1}")
        
        self.bridge_connected = False
        time.sleep(1)
        print(f"🐛 Debug CARLA Environment initialized")

    def step(self, action):
        """Execute environment step with debug output."""
        self.step_count += 1
        
        print(f"🐛 Step {self.step_count}: action = {action}")
        
        # Send action to CARLA
        try:
            action_data = {
                'steering': float(action[0]),
                'throttle': max(0, float(action[1])),
                'brake': max(0, -float(action[1])),
                'hand_brake': False,
                'reverse': False
            }
            packed_action = msgpack.packb(action_data)
            self.action_socket.send(packed_action, zmq.NOBLOCK)
            print(f"🐛 Sent action: steering={action_data['steering']:.2f}, throttle={action_data['throttle']:.2f}")
        except Exception as e:
            print(f"🐛 Action send error: {e}")

        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation, action)
        
        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'stuck_counter': self.stuck_counter
        }
        
        print(f"🐛 Step {self.step_count}: reward={reward:.3f}, terminated={terminated}, stuck_counter={self.stuck_counter}")
        
        self.episode_reward += reward
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get observation with debug output."""
        try:
            # Try to receive CARLA data
            try:
                packed_data = self.data_socket.recv(zmq.NOBLOCK)
                carla_data = msgpack.unpackb(packed_data, raw=False)
                self.last_carla_data = carla_data
                self.bridge_connected = True
                print(f"🐛 Received CARLA data: {list(carla_data.keys())}")
                return self._process_carla_data(carla_data)
            except zmq.Again:
                print("🐛 No CARLA data available, using last data")
                if self.last_carla_data:
                    return self._process_carla_data(self.last_carla_data)
                else:
                    print("🐛 No data available, using dummy")
                    return self._get_dummy_observation()
        except Exception as e:
            print(f"🐛 Observation error: {e}")
            return self._get_dummy_observation()

    def _process_carla_data(self, carla_data):
        """Process CARLA data with debug output."""
        try:
            # Extract camera data
            sensors = carla_data.get('sensors', {})
            camera_data = None
            
            for key, sensor in sensors.items():
                if sensor.get('type') == 'camera':
                    camera_data = sensor.get('data', [])
                    break
            
            # Process camera image
            if camera_data and len(camera_data) > 0:
                camera_array = np.array(camera_data, dtype=np.uint8)
                if camera_array.size > 0:
                    try:
                        camera_image = camera_array.reshape((600, 800, 4))[:, :, :3]
                        # Resize to 84x84
                        from PIL import Image
                        pil_image = Image.fromarray(camera_image)
                        resized_image = pil_image.resize((84, 84))
                        camera_obs = np.array(resized_image)
                    except:
                        camera_obs = np.zeros((84, 84, 3), dtype=np.uint8)
                else:
                    camera_obs = np.zeros((84, 84, 3), dtype=np.uint8)
            else:
                camera_obs = np.zeros((84, 84, 3), dtype=np.uint8)
            
            # Extract vehicle state
            vehicle_data = carla_data.get('vehicle', {})
            location = vehicle_data.get('location', {'x': 0, 'y': 0})
            rotation = vehicle_data.get('rotation', {'yaw': 0})
            velocity = vehicle_data.get('velocity', {'x': 0, 'y': 0})
            
            x, y = location['x'], location['y']
            yaw = rotation['yaw'] * np.pi / 180.0
            vx, vy = velocity['x'], velocity['y']
            speed = np.sqrt(vx**2 + vy**2) * 3.6  # Convert to km/h
            
            print(f"🐛 Vehicle state: x={x:.1f}, y={y:.1f}, speed={speed:.1f} km/h")
            
            vehicle_state = np.array([x, y, yaw, speed, vx, vy], dtype=np.float32)
            
            return {
                'camera': camera_obs,
                'vehicle_state': vehicle_state
            }
            
        except Exception as e:
            print(f"🐛 Data processing error: {e}")
            return self._get_dummy_observation()

    def _get_dummy_observation(self):
        """Get dummy observation."""
        return {
            'camera': np.zeros((84, 84, 3), dtype=np.uint8),
            'vehicle_state': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }

    def _calculate_reward(self, observation, action):
        """Calculate reward with debug output."""
        try:
            vehicle_state = observation['vehicle_state']
            velocity = vehicle_state[3]  # Speed in km/h
            
            # Simple speed reward
            speed_reward = min(velocity * 0.1, 2.0)
            
            print(f"🐛 Reward calculation: speed={velocity:.1f} km/h, reward={speed_reward:.3f}")
            
            return speed_reward
        except Exception as e:
            print(f"🐛 Reward calculation error: {e}")
            return 0.0

    def _is_terminated(self, observation):
        """Check termination with debug output."""
        try:
            vehicle_state = observation['vehicle_state']
            velocity = vehicle_state[3]
            
            print(f"🐛 Termination check: step={self.step_count}, velocity={velocity:.2f}, stuck_counter={self.stuck_counter}")
            
            # Give the car time to start moving (skip stuck check for first 10 steps)
            if self.step_count > 10:
                if velocity < 0.5:
                    self.stuck_counter += 1
                    print(f"🐛 Car is slow, stuck_counter increased to {self.stuck_counter}")
                else:
                    self.stuck_counter = 0
                    print(f"🐛 Car is moving, stuck_counter reset")

                # Terminate if stuck for 30 steps
                if self.stuck_counter > 30:
                    print("🐛 Episode terminated due to being stuck!")
                    return True
            else:
                print(f"🐛 Skipping stuck check for first 10 steps")

            # Terminate if episode is too long
            if self.step_count > self.max_episode_steps:
                print("🐛 Episode terminated due to max steps!")
                return True

            return False

        except Exception as e:
            print(f"🐛 Termination check error: {e}")
            return False

    def reset(self, **kwargs):
        """Reset environment with debug output."""
        print(f"🐛 Resetting environment. Previous episode: {self.total_episodes}, steps: {self.step_count}, reward: {self.episode_reward:.2f}")
        
        self.step_count = 0
        self.episode_reward = 0.0
        self.stuck_counter = 0
        self.has_collision = False
        self.total_episodes += 1
        
        # Get initial observation
        initial_obs = self._get_observation()
        
        print(f"🐛 Environment reset complete. Starting episode {self.total_episodes}")
        
        return initial_obs, {}

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'data_socket'):
            self.data_socket.close()
        if hasattr(self, 'action_socket'):
            self.action_socket.close()
        if hasattr(self, 'context'):
            self.context.term()


def main():
    print("🐛 Debug CARLA DRL Trainer Starting...")
    
    # Create environment
    env = DebugCARLAEnv()
    
    # Test environment for a few steps
    print("🐛 Testing environment...")
    
    observation, info = env.reset()
    print(f"🐛 Initial observation keys: {observation.keys()}")
    print(f"🐛 Vehicle state: {observation['vehicle_state']}")
    
    for step in range(20):
        # Simple action: slight throttle, no steering
        action = np.array([0.0, 0.3, 0.5])  # [steering, throttle, aggressive]
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"🐛 Step {step + 1}: reward={reward:.3f}, terminated={terminated}")
        
        if terminated or truncated:
            print("🐛 Episode ended, resetting...")
            observation, info = env.reset()
            
        time.sleep(0.1)  # Small delay
    
    env.close()
    print("🐛 Debug test complete!")


if __name__ == "__main__":
    main()

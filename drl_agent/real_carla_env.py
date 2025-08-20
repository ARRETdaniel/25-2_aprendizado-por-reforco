#!/usr/bin/env python3
"""
Real CARLA Environment for DRL Training
Connects to actual CARLA simulation via ZMQ bridge
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import logging

# Add communication bridge
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'communication'))
try:
    from zmq_bridge import DRLBridgeServer
except ImportError:
    print("❌ ZMQ Bridge not found. Run: pip install pyzmq msgpack msgpack-numpy")
    raise

print(f"🚗 Real CARLA Environment - ZMQ Bridge Integration")
print(f"🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

class RealCarlaEnvironment(gym.Env):
    """
    Real CARLA Environment that connects to actual CARLA simulation.
    Receives sensor data and sends control commands via ZMQ bridge.
    """

    def __init__(self,
                 bridge_port: int = 5556,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 300,
                 target_speed: float = 8.0):
        """
        Initialize Real CARLA Environment.

        Args:
            bridge_port: Port for ZMQ communication bridge
            render_mode: Rendering mode ('human' or None)
            max_episode_steps: Maximum steps per episode
            target_speed: Target driving speed in m/s
        """
        super().__init__()

        self.bridge_port = bridge_port
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.target_speed = target_speed

        # Initialize bridge
        self.bridge = DRLBridgeServer(port=bridge_port)
        self.bridge_connected = False

        # Define action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space: camera image + vehicle state
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
        self.episode_reward = 0.0
        self.last_carla_data = None
        self.last_position = np.array([0.0, 0.0])

        # Performance tracking
        self.total_episodes = 0
        self.total_steps = 0
        self.collision_count = 0

        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        print("🏁 Real CARLA Environment initialized")

    def connect_to_carla(self) -> bool:
        """Connect to CARLA via ZMQ bridge."""
        if self.bridge_connected:
            return True

        try:
            if self.bridge.start():
                self.bridge_connected = True
                self.logger.info("✅ Connected to CARLA via ZMQ bridge")
                return True
            else:
                self.logger.error("❌ Failed to start bridge server")
                return False

        except Exception as e:
            self.logger.error(f"❌ Bridge connection error: {e}")
            return False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Connect to CARLA if not connected
        if not self.bridge_connected:
            if not self.connect_to_carla():
                raise RuntimeError("Cannot connect to CARLA bridge")

        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0
        self.total_episodes += 1

        # Wait for initial CARLA data
        initial_data = self._wait_for_carla_data(timeout=5.0)
        if initial_data is None:
            self.logger.warning("⚠️ No initial CARLA data received")
            # Return dummy observation
            return self._get_dummy_observation(), {}

        self.last_carla_data = initial_data
        observation = self._process_carla_data(initial_data)

        self.logger.info(f"🔄 Episode {self.total_episodes} reset")
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.step_count += 1
        self.total_steps += 1

        # Parse action
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        # Convert throttle/brake
        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # Send action to CARLA
        action_sent = self.bridge.send_action(steering, throttle, brake)
        if not action_sent:
            self.logger.warning("⚠️ Failed to send action to CARLA")

        # Receive new sensor data
        carla_data = self._wait_for_carla_data(timeout=1.0)
        if carla_data is None:
            self.logger.warning("⚠️ No CARLA data received")
            # Use last known data
            carla_data = self.last_carla_data

        self.last_carla_data = carla_data

        # Process observation
        observation = self._process_carla_data(carla_data)

        # Calculate reward
        reward = self._calculate_reward(carla_data, action)
        self.episode_reward += reward

        # Check termination
        terminated = self._is_terminated(carla_data)
        truncated = self.step_count >= self.max_episode_steps

        # Info dictionary
        info = {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'carla_connected': self.bridge_connected,
            'action_sent': action_sent
        }

        if carla_data:
            measurements = carla_data.get('measurements', {})
            info.update({
                'vehicle_speed': measurements.get('velocity', 0.0),
                'position': measurements.get('position', [0.0, 0.0, 0.0])
            })

        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.step_count
            }
            self.logger.info(f"🏁 Episode ended: reward={self.episode_reward:.3f}, steps={self.step_count}")

        return observation, reward, terminated, truncated, info

    def _wait_for_carla_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """Wait for CARLA data with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                data = self.bridge.receive_carla_data()
                if data is not None:
                    return data
                time.sleep(0.01)  # Small delay to prevent busy waiting
            except Exception as e:
                self.logger.warning(f"⚠️ Error receiving CARLA data: {e}")
                break

        return None

    def _process_carla_data(self, carla_data: Optional[Dict]) -> Dict:
        """Process CARLA data into observation format."""
        if carla_data is None:
            return self._get_dummy_observation()

        try:
            # Extract camera data
            sensors = carla_data.get('sensors', {})
            camera_data = None

            for sensor_name, sensor_value in sensors.items():
                if 'Camera' in sensor_name:
                    if isinstance(sensor_value, list):
                        # Convert from list back to numpy array
                        camera_data = np.array(sensor_value, dtype=np.uint8)
                    break

            if camera_data is None or camera_data.size == 0:
                # Create dummy camera image
                camera_data = np.ones((64, 64, 3), dtype=np.uint8) * 128
            else:
                # Ensure correct shape
                if camera_data.shape != (64, 64, 3):
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

            return {
                'camera': camera_data,
                'vehicle_state': vehicle_state
            }

        except Exception as e:
            self.logger.error(f"❌ Error processing CARLA data: {e}")
            return self._get_dummy_observation()

    def _get_dummy_observation(self) -> Dict:
        """Get dummy observation when CARLA data is unavailable."""
        return {
            'camera': np.ones((64, 64, 3), dtype=np.uint8) * 128,
            'vehicle_state': np.zeros(6, dtype=np.float32)
        }

    def _calculate_reward(self, carla_data: Optional[Dict], action: np.ndarray) -> float:
        """Calculate reward based on CARLA data and action."""
        if carla_data is None:
            return -1.0  # Penalty for no data

        try:
            measurements = carla_data.get('measurements', {})
            velocity = measurements.get('velocity', 0.0)
            position = measurements.get('position', [0.0, 0.0, 0.0])

            # Speed reward (encourage target speed)
            speed_error = abs(velocity - self.target_speed)
            speed_reward = max(0, 1.0 - speed_error / self.target_speed)

            # Progress reward (encourage forward movement)
            current_position = np.array(position[:2])
            if hasattr(self, 'last_position'):
                progress = np.linalg.norm(current_position - self.last_position)
                progress_reward = min(progress * 0.1, 0.5)  # Cap progress reward
            else:
                progress_reward = 0.0

            self.last_position = current_position

            # Action smoothness penalty
            steering, throttle = action[0], action[1]
            smoothness_penalty = -(abs(steering) * 0.1 + abs(throttle - 0.3) * 0.05)

            # Combine rewards
            total_reward = speed_reward + progress_reward + smoothness_penalty

            # Small step reward to encourage staying alive
            step_reward = 0.01

            return total_reward + step_reward

        except Exception as e:
            self.logger.error(f"❌ Error calculating reward: {e}")
            return -0.1

    def _is_terminated(self, carla_data: Optional[Dict]) -> bool:
        """Check if episode should terminate."""
        if carla_data is None:
            return False  # Don't terminate on missing data

        try:
            measurements = carla_data.get('measurements', {})
            velocity = measurements.get('velocity', 0.0)

            # Terminate if vehicle is stuck (too slow for too long)
            if velocity < 0.5 and self.step_count > 50:
                self.logger.info("🛑 Terminated: vehicle stuck")
                return True

            # Could add collision detection here if available in measurements

            return False

        except Exception as e:
            self.logger.error(f"❌ Error checking termination: {e}")
            return False

    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human' and self.last_carla_data:
            try:
                # Get camera data
                observation = self._process_carla_data(self.last_carla_data)
                camera_image = observation['camera']
                vehicle_state = observation['vehicle_state']

                # Scale up for display
                display_image = cv2.resize(camera_image, (320, 320))

                # Add info overlay
                cv2.putText(display_image, f"Step: {self.step_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Reward: {self.episode_reward:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"Speed: {vehicle_state[3]:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, "Real CARLA DRL", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Real CARLA DRL Training', display_image)
                cv2.waitKey(1)

            except Exception as e:
                self.logger.error(f"❌ Render error: {e}")

    def close(self):
        """Clean up environment."""
        if self.bridge_connected:
            self.bridge.stop()
            self.bridge_connected = False

        cv2.destroyAllWindows()
        self.logger.info("🧹 Real CARLA Environment closed")

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'bridge_connected': self.bridge_connected,
            'collision_count': self.collision_count
        }


if __name__ == "__main__":
    # Test the real CARLA environment
    print("🧪 Testing Real CARLA Environment...")

    env = RealCarlaEnvironment(render_mode='human')

    try:
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"📊 Observation shapes: camera={obs['camera'].shape}, state={obs['vehicle_state'].shape}")

        for step in range(10):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step}: reward={reward:.3f}, terminated={terminated}")
            env.render()

            if terminated or truncated:
                obs, info = env.reset()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("⚠️ Test interrupted")
    finally:
        env.close()
        print("✅ Real CARLA Environment test completed")
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: multimodal (image + vector)
        # Camera: 84x84x3 (standard for DRL)
        # Vector: [x, y, yaw, speed_x, speed_y, angular_vel] = 6D
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(84, 84, 3),
                dtype=np.uint8
            ),
            'vector': spaces.Box(
                low=np.array([-100.0, -100.0, -np.pi, -20.0, -20.0, -np.pi]),
                high=np.array([100.0, 100.0, np.pi, 20.0, 20.0, np.pi]),
                dtype=np.float32
            )
        })

        # Environment state
        self.client = None
        self.episode_active = False
        self.step_count = 0
        self.max_episode_steps = 1000
        self.episode_reward = 0.0

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        # Rendering
        self.render_window_name = "CARLA Real Environment"

        print(f"✅ RealCarlaEnv initialized")
        print(f"   Action space: {self.action_space}")
        print(f"   Observation space: {self.observation_space}")

    def _connect_to_carla(self) -> bool:
        """Connect to CARLA server."""
        try:
            # Import CARLA modules (requires Python 3.6 compatibility)
            try:
                from carla.client import make_carla_client, VehicleControl
                from carla.settings import CarlaSettings
                from carla.sensor import Camera
                from carla.image_converter import to_rgb_array
                print("✅ CARLA modules imported successfully")
            except ImportError as e:
                print(f"❌ Failed to import CARLA modules: {e}")
                print("⚠️ This environment requires CARLA Python API")
                return False

            # Store CARLA classes for later use
            self.make_carla_client = make_carla_client
            self.VehicleControl = VehicleControl
            self.CarlaSettings = CarlaSettings
            self.Camera = Camera
            self.to_rgb_array = to_rgb_array

            print(f"🔌 Connecting to CARLA server at {self.host}:{self.port}")
            self.client_context = self.make_carla_client(self.host, self.port, timeout=self.timeout)
            self.client = self.client_context.__enter__()

            print("✅ Connected to CARLA server")
            return True

        except Exception as e:
            print(f"❌ Failed to connect to CARLA: {e}")
            return False

    def _configure_episode(self) -> bool:
        """Configure CARLA episode for DRL training."""
        try:
            print("⚙️ Configuring CARLA episode...")

            # Create optimized settings for DRL (CARLA 0.8.4 API)
            settings = self.CarlaSettings()
            settings.set(
                SynchronousMode=True,  # Synchronous for deterministic training
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=10,   # Moderate traffic
                NumberOfPedestrians=15,
                WeatherId=1,          # Clear conditions (correct API)
                QualityLevel='Low'    # Performance optimized
            )            # RGB Camera for perception
            camera = self.Camera('CameraRGB')
            camera.set_image_size(640, 480)  # Will be resized to 84x84
            camera.set_position(2.0, 0.0, 1.4)
            camera.set_rotation(-15.0, 0.0, 0.0)
            settings.add_sensor(camera)

            # Load settings and start episode
            scene = self.client.load_settings(settings)
            player_start = np.random.randint(1, min(len(scene.player_start_spots), 10))
            self.client.start_episode(player_start)

            self.episode_active = True
            self.step_count = 0
            self.episode_reward = 0.0

            print(f"✅ Episode configured and started at position {player_start}")
            return True

        except Exception as e:
            print(f"❌ Failed to configure episode: {e}")
            return False

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current multimodal observation from CARLA."""
        try:
            # Read sensor data from CARLA
            measurements, sensor_data = self.client.read_data()

            # Process camera image
            if 'CameraRGB' in sensor_data:
                # Convert to RGB array
                image = self.to_rgb_array(sensor_data['CameraRGB'])

                # Resize to 84x84 for DRL (standard size)
                image_resized = cv2.resize(image, (84, 84))

                # Ensure uint8 format
                image_obs = image_resized.astype(np.uint8)
            else:
                # Fallback black image
                image_obs = np.zeros((84, 84, 3), dtype=np.uint8)

            # Get vehicle state
            player = measurements.player_measurements
            vector_obs = np.array([
                player.transform.location.x,     # x position
                player.transform.location.y,     # y position
                player.transform.rotation.yaw * np.pi / 180.0,  # yaw in radians
                player.velocity.x,               # velocity x
                player.velocity.y,               # velocity y
                0.0                             # angular velocity (approximation)
            ], dtype=np.float32)

            # Store measurements for reward calculation
            self.current_measurements = measurements

            return {
                'image': image_obs,
                'vector': vector_obs
            }

        except Exception as e:
            print(f"⚠️ Error getting observation: {e}")
            # Return safe fallback observation
            return {
                'image': np.zeros((84, 84, 3), dtype=np.uint8),
                'vector': np.zeros(6, dtype=np.float32)
            }

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on driving performance."""
        try:
            player = self.current_measurements.player_measurements

            # Speed reward (encourage forward movement)
            speed = np.sqrt(player.velocity.x**2 + player.velocity.y**2)
            target_speed = 10.0  # m/s target
            speed_reward = max(0, 1.0 - abs(speed - target_speed) / target_speed)

            # Collision penalty
            collision_penalty = 0.0
            if (player.collision_vehicles > 0 or
                player.collision_pedestrians > 0 or
                player.collision_other > 0):
                collision_penalty = -10.0

            # Lane keeping (penalize off-road)
            lane_penalty = 0.0
            if player.intersection_offroad > 0.5:
                lane_penalty = -1.0

            # Action smoothness (discourage erratic control)
            steering, throttle = action[0], action[1]
            smoothness_penalty = -(abs(steering) * 0.1 + abs(throttle - 0.3) * 0.1)

            # Progress reward (encourage forward movement)
            progress_reward = max(0, player.velocity.x) * 0.1

            # Combine rewards
            total_reward = (speed_reward +
                          collision_penalty +
                          lane_penalty +
                          smoothness_penalty +
                          progress_reward)

            return total_reward

        except Exception as e:
            print(f"⚠️ Error calculating reward: {e}")
            return 0.0

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        try:
            player = self.current_measurements.player_measurements

            # Terminate on collision
            if (player.collision_vehicles > 0 or
                player.collision_pedestrians > 0 or
                player.collision_other > 0):
                return True

            # Terminate if significantly off-road
            if player.intersection_offroad > 0.8:
                return True

            # Terminate if moving backwards too much
            if player.velocity.x < -2.0:
                return True

            return False

        except Exception as e:
            print(f"⚠️ Error checking termination: {e}")
            return True  # Safe termination on error

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Connect to CARLA if not connected
        if self.client is None:
            if not self._connect_to_carla():
                raise RuntimeError("Failed to connect to CARLA server")

        # Configure new episode
        if not self._configure_episode():
            raise RuntimeError("Failed to configure CARLA episode")

        # Get initial observation
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one environment step."""
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")

        self.step_count += 1
        self.frame_count += 1

        # Parse action
        steering = np.clip(action[0], -1.0, 1.0)
        throttle_brake = np.clip(action[1], -1.0, 1.0)

        # Convert to CARLA control
        control = self.VehicleControl()
        control.steer = float(steering)

        if throttle_brake >= 0:
            control.throttle = float(throttle_brake)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = float(-throttle_brake)

        # Send control to CARLA
        self.client.send_control(control)

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward

        # Check termination
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps

        # Info dict
        info = {
            'episode_reward': self.episode_reward,
            'step_count': self.step_count,
            'frame_count': self.frame_count
        }

        # Episode ending
        if terminated or truncated:
            self.episode_active = False
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.step_count
            }

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human' and self.episode_active:
            try:
                observation = self._get_observation()
                image = observation['image']

                # Scale up for better visibility
                display_image = cv2.resize(image, (420, 420))

                # Add info overlay
                cv2.putText(display_image, f"Step: {self.step_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Reward: {self.episode_reward:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(self.render_window_name, display_image)
                cv2.waitKey(1)

                return display_image

            except Exception as e:
                print(f"⚠️ Render error: {e}")

        return None

    def close(self):
        """Clean up environment."""
        print("🧹 Closing Real CARLA Environment...")

        self.episode_active = False

        if cv2.getWindowProperty(self.render_window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(self.render_window_name)

        # Note: We don't close CARLA client to allow reuse
        runtime = time.time() - self.start_time
        avg_fps = self.frame_count / runtime if runtime > 0 else 0

        print(f"📊 Session Statistics:")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("✅ Cleanup completed")

# Test function
def test_real_carla_env():
    """Test the Real CARLA Environment."""
    print("🧪 Testing Real CARLA Environment...")

    try:
        # Create environment
        env = RealCarlaEnv(render_mode='human')

        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful")
        print(f"   Image shape: {obs['image'].shape}")
        print(f"   Vector shape: {obs['vector'].shape}")

        # Test a few steps
        for step in range(10):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step + 1}: reward={reward:.3f}, terminated={terminated}")

            # Render
            env.render()

            if terminated or truncated:
                break

            time.sleep(0.1)  # Small delay for visualization

        env.close()
        print("🎉 Test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_real_carla_env()

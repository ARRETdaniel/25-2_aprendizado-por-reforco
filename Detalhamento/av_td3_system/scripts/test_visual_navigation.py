#!/usr/bin/env python3
"""
Visual Testing Script for TD3 Autonomous Navigation System

This script runs the TD3 agent for 100 steps with live camera visualization
using OpenCV, similar to module_7.py. This is for TESTING/DEBUGGING only.

For actual training, use train_td3.py which runs headless for maximum performance.

Usage:
    python3 scripts/test_visual_navigation.py

Requirements:
    - CARLA server running on port 2000
    - OpenCV installed (cv2)
    - TD3 agent with trained weights (optional, will use random if not found)
"""

import sys
import os
sys.path.insert(0, '/workspace')

import numpy as np
import cv2
import time
import argparse
from datetime import datetime
from pathlib import Path

# Import torch for GPU memory monitoring
import torch

# CARLA imports (must be available in environment)
try:
    import carla
except ImportError:
    print("ERROR: CARLA Python API not found!")
    print("Make sure you're running inside the Docker container or have CARLA installed.")
    sys.exit(1)

# Local imports
from src.environment.carla_env import CARLANavigationEnv
from src.agents.td3_agent import TD3Agent


def log_gpu_memory(prefix=""):
    """Log GPU memory usage for debugging OOM issues."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"🧠 {prefix} GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")
    else:
        print(f"⚠️  CUDA not available")


class VisualNavigationTester:
    """
    Visual testing wrapper for TD3 autonomous navigation.

    Displays live camera feed using OpenCV while agent drives.
    """

    def __init__(
        self,
        carla_config_path: str = '/workspace/config/carla_config.yaml',
        td3_config_path: str = '/workspace/config/td3_config.yaml',
        training_config_path: str = '/workspace/config/training_config.yaml',
        checkpoint_path: str = None,
        window_name: str = "TD3 Visual Navigation - Front Camera",
        max_steps: int = 500,  # Increased from 100 to 500 steps
        display_info: bool = True
    ):
        """
        Initialize visual navigation tester.

        Args:
            carla_config_path: Path to CARLA configuration file
            td3_config_path: Path to TD3 algorithm configuration file
            training_config_path: Path to training configuration file
            checkpoint_path: Optional path to trained agent checkpoint
            window_name: Name of OpenCV display window
            max_steps: Maximum number of steps to run
            display_info: Whether to overlay text information on display
        """
        self.window_name = window_name
        self.max_steps = max_steps
        self.display_info = display_info
        self.checkpoint_path = checkpoint_path

        print("="*80)
        print("🎥 TD3 VISUAL NAVIGATION TESTER")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Max steps: {max_steps}")
        print(f"Display info overlay: {display_info}")

        # Check CUDA memory optimization settings
        cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        print(f"\n🧠 PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")
        log_gpu_memory("Initial")

        # Initialize environment
        print("\n🌍 Initializing CARLA environment...")
        self.env = CARLANavigationEnv(
            carla_config_path=carla_config_path,
            td3_config_path=td3_config_path,
            training_config_path=training_config_path
        )
        print("✅ Environment ready")
        log_gpu_memory("After CARLA env")

        # Initialize agent
        print("\n🤖 Initializing TD3 agent...")
        print("⚠️  Using CPU device to save GPU memory for CARLA")
        self.agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config_path=td3_config_path,
            device='cpu'  # Use CPU to leave GPU for CARLA
        )
        log_gpu_memory("After TD3 agent")        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            self.agent.load_checkpoint(checkpoint_path)
            print("✅ Checkpoint loaded")
        else:
            print("⚠️  No checkpoint loaded - using untrained agent (random policy)")

        print(f"✅ Agent ready (device: {self.agent.device})")

        # Create OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # Metrics tracking
        self.episode_metrics = {
            'step': 0,
            'reward_sum': 0.0,
            'speed': 0.0,
            'steering': 0.0,
            'throttle': 0.0,
            'collisions': 0,
            'frames_processed': 0
        }

    def flatten_dict_obs(self, obs_dict):
        """
        Flatten Dict observation into single numpy array.

        Expected structure:
            {'image': (4, 84, 84), 'vector': (23,)}

        Returns:
            Flat array of shape (535,) = 512 (image features) + 23 (vector)
        """
        # Extract image and flatten
        image = obs_dict['image']  # Shape: (4, 84, 84)

        # Use average pooling to reduce to 512 features
        image_flat = image.reshape(4, -1).mean(axis=0)  # (7056,) per channel
        image_features = image_flat[:512]  # Take first 512 features

        # Pad if needed
        if len(image_features) < 512:
            image_features = np.pad(image_features, (0, 512 - len(image_features)))

        vector = obs_dict['vector']  # Shape: (23,)

        # Concatenate: 512 (image) + 23 (vector) = 535
        flat_state = np.concatenate([image_features, vector]).astype(np.float32)

        return flat_state

    def get_display_frame(self, obs_dict):
        """
        Extract and prepare camera frame for display.

        Args:
            obs_dict: Dictionary observation with 'image' key

        Returns:
            BGR image ready for cv2.imshow()
        """
        # Get the latest frame from stacked images (last of 4 frames)
        # obs_dict['image'] has shape (4, 84, 84) - 4 grayscale frames
        latest_frame = obs_dict['image'][-1]  # Shape: (84, 84)

        # Convert from [0, 1] float to [0, 255] uint8
        frame_uint8 = (latest_frame * 255).astype(np.uint8)

        # Resize to larger display size
        frame_resized = cv2.resize(frame_uint8, (800, 600), interpolation=cv2.INTER_LINEAR)

        # Convert grayscale to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

        return frame_bgr

    def overlay_info(self, frame, obs_dict, action, reward, info):
        """
        Overlay text information on the display frame.

        Args:
            frame: BGR image
            obs_dict: Current observation
            action: Current action [steering, throttle/brake]
            reward: Current reward
            info: Environment info dict
        """
        if not self.display_info:
            return frame

        # Extract metrics
        step = self.episode_metrics['step']
        speed = info.get('vehicle_state', {}).get('velocity', 0.0)  # Speed in m/s from vehicle_state
        lateral_dev = info.get('vehicle_state', {}).get('lateral_deviation', 0.0)
        heading_error = info.get('vehicle_state', {}).get('heading_error', 0.0)
        collision = self.sensors.is_collision_detected() if hasattr(self, 'sensors') else False

        # Update tracked metrics
        self.episode_metrics['speed'] = speed
        self.episode_metrics['steering'] = action[0]
        self.episode_metrics['throttle'] = action[1]
        self.episode_metrics['reward_sum'] += reward
        if collision:
            self.episode_metrics['collisions'] += 1

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color_white = (255, 255, 255)
        color_green = (0, 255, 0)
        color_red = (0, 0, 255)
        color_yellow = (0, 255, 255)

        # Text content
        texts = [
            f"Step: {step}/{self.max_steps}",
            f"Speed: {speed*3.6:.1f} km/h",
            f"Reward: {reward:.3f} (Total: {self.episode_metrics['reward_sum']:.2f})",
            f"Steering: {action[0]:+.3f}",
            f"Throttle/Brake: {action[1]:+.3f}",
            f"Lateral Dev: {lateral_dev:.2f} m",
            f"Heading Error: {np.degrees(heading_error):.1f}°",
            f"Collisions: {self.episode_metrics['collisions']}"
        ]

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw text
        y_offset = 35
        for i, text in enumerate(texts):
            y_pos = y_offset + i * 25

            # Choose color based on content
            if "Collision" in text and collision:
                color = color_red
            elif "Speed" in text:
                color = color_green
            elif "Reward" in text:
                color = color_yellow
            else:
                color = color_white

            cv2.putText(frame, text, (20, y_pos), font, font_scale, color, thickness)

        # Draw action indicators (steering wheel and throttle bar)
        self._draw_action_indicators(frame, action)

        return frame

    def _draw_action_indicators(self, frame, action):
        """Draw visual indicators for steering and throttle."""
        h, w = frame.shape[:2]

        # Steering wheel indicator (bottom left)
        center_x, center_y = 100, h - 80
        radius = 40
        cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 2)

        # Steering direction line
        angle = -action[0] * 90  # -90 to +90 degrees
        end_x = int(center_x + radius * np.sin(np.radians(angle)))
        end_y = int(center_y - radius * np.cos(np.radians(angle)))
        cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)

        # Throttle/Brake bar indicator (bottom right)
        bar_x = w - 60
        bar_y_bottom = h - 20
        bar_height = 100
        bar_width = 30

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y_bottom - bar_height),
                     (bar_x + bar_width, bar_y_bottom), (100, 100, 100), 2)

        # Fill bar based on throttle/brake
        if action[1] > 0:  # Throttle (green)
            fill_height = int(bar_height * action[1])
            cv2.rectangle(frame, (bar_x, bar_y_bottom - fill_height),
                         (bar_x + bar_width, bar_y_bottom), (0, 255, 0), -1)
        else:  # Brake (red)
            fill_height = int(bar_height * abs(action[1]))
            cv2.rectangle(frame, (bar_x, bar_y_bottom - fill_height),
                         (bar_x + bar_width, bar_y_bottom), (0, 0, 255), -1)

    def run(self):
        """
        Run visual navigation test.

        Returns:
            dict: Final episode metrics
        """
        print("\n" + "="*80)
        print("🏁 STARTING VISUAL NAVIGATION TEST")
        print("="*80)
        print("Press 'q' to quit, 'p' to pause, SPACE to step when paused")
        print()

        # Reset environment
        obs_dict = self.env.reset()
        state = self.flatten_dict_obs(obs_dict)

        done = False
        truncated = False
        paused = False
        step = 0

        try:
            while step < self.max_steps and not (done or truncated):
                step_start = time.time()

                # Select action (deterministic for testing)
                action = self.agent.select_action(state, noise=0.0)

                # 🔍 DIAGNOSTIC: Log action values for first 10 steps
                if step < 10:
                    print(f"\n🔍 DEBUG Step {step}:")
                    print(f"   Action: steering={action[0]:+.6f}, throttle/brake={action[1]:+.6f}")
                    if action[1] > 0:
                        print(f"   Control: throttle={action[1]:.6f}, brake=0.0")
                    else:
                        print(f"   Control: throttle=0.0, brake={-action[1]:.6f}")

                # Execute action (debug logging is inside env.step() for first 10 steps)
                result = self.env.step(action)

                # 🔍 DIAGNOSTIC: Check what we received from env.step()
                if step < 10:
                    print(f"   env.step() returned {len(result)} items: {type(result)}")
                    if len(result) == 5:
                        next_obs_dict, reward, done, truncated, info = result
                        print(f"   info type: {type(info)}, is None: {info is None}")
                        if info is not None:
                            print(f"   info keys: {list(info.keys()) if isinstance(info, dict) else 'not a dict'}")
                    else:
                        print(f"   ERROR: Expected 5 return values, got {len(result)}")
                        next_obs_dict, reward, done, truncated, info = result[0], result[1], result[2], result[3], None
                else:
                    next_obs_dict, reward, done, truncated, info = result

                # 🔍 DIAGNOSTIC: Log speed from info dict for first 10 steps
                if step < 10 and info is not None:
                    try:
                        vehicle_state = info.get('vehicle_state')
                        print(f"   vehicle_state type: {type(vehicle_state)}, is None: {vehicle_state is None}")
                        if vehicle_state is not None:
                            print(f"   vehicle_state keys: {list(vehicle_state.keys()) if isinstance(vehicle_state, dict) else 'not a dict'}")
                            speed_mps = vehicle_state.get('velocity', 0.0) if isinstance(vehicle_state, dict) else 0.0
                            speed_kmh = speed_mps * 3.6
                            collision_info = info.get('collision_info')
                            collision = collision_info.get('detected', False) if isinstance(collision_info, dict) else False
                            print(f"   Post-Step Speed: {speed_kmh:.2f} km/h ({speed_mps:.2f} m/s)")
                            print(f"   Collision: {collision}")
                            print(f"   Reward: {reward:.4f}")

                            # 🔍 NEW: Print detailed reward breakdown
                            reward_breakdown = info.get('reward_breakdown', {})
                            if reward_breakdown:
                                print(f"   Reward Breakdown:")
                                for component, (weight, value, weighted) in reward_breakdown.items():
                                    print(f"     {component:15s}: w={weight:+.2f} × v={value:+.4f} = {weighted:+.4f}")
                        else:
                            print(f"   ⚠️  vehicle_state is None!")
                    except Exception as e:
                        print(f"   Debug logging error: {e}")
                        import traceback
                        traceback.print_exc()
                next_state = self.flatten_dict_obs(next_obs_dict)

                # Update metrics
                self.episode_metrics['step'] = step
                self.episode_metrics['frames_processed'] += 1

                # Get display frame
                display_frame = self.get_display_frame(obs_dict)

                # Overlay information
                display_frame = self.overlay_info(display_frame, obs_dict, action, reward, info)

                # Show frame
                cv2.imshow(self.window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n⚠️  User requested quit")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"\n{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")

                # Handle pause
                while paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        paused = False
                        print("▶️  RESUMED")
                    elif key == ord(' '):
                        break  # Step forward one frame
                    elif key == ord('q'):
                        print("\n⚠️  User requested quit")
                        paused = False
                        done = True
                        break

                # Update state
                state = next_state
                obs_dict = next_obs_dict
                step += 1

                # Calculate FPS
                step_time = time.time() - step_start
                fps = 1.0 / step_time if step_time > 0 else 0

                # Print progress every 10 steps
                if step % 10 == 0 and info is not None:
                    vehicle_state = info.get('vehicle_state', {})
                    speed_mps = vehicle_state.get('velocity', 0.0)
                    speed_kmh = speed_mps * 3.6
                    print(f"Step {step:3d}/{self.max_steps} | "
                          f"Reward: {reward:7.3f} | "
                          f"Speed: {speed_kmh:5.1f} km/h | "
                          f"FPS: {fps:5.1f}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user (Ctrl+C)")

        finally:
            # Cleanup
            print("\n" + "="*80)
            print("🏁 TEST FINISHED")
            print("="*80)
            print(f"Total steps: {step}")
            print(f"Total reward: {self.episode_metrics['reward_sum']:.2f}")
            print(f"Collisions: {self.episode_metrics['collisions']}")
            if info is not None:
                print(f"Termination reason: {info.get('termination_reason', 'unknown')}")

            cv2.destroyAllWindows()

            # Don't destroy environment here - let it be handled by cleanup

        return self.episode_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visual testing for TD3 autonomous navigation"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to trained agent checkpoint'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Maximum number of steps to run (default: 100)'
    )
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Disable information overlay on display'
    )

    args = parser.parse_args()

    # Create tester
    tester = VisualNavigationTester(
        checkpoint_path=args.checkpoint,
        max_steps=args.max_steps,
        display_info=not args.no_overlay
    )

    # Run test
    metrics = tester.run()

    print("\n✅ Visual navigation test complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
PPO Training Script for CARLA DRL Pipeline
Enhanced PPO implementation with CARLA environment integration

Author: GitHub Copilot
Date: 2025-01-26
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import cv2
import matplotlib.pyplot as plt
from collections import deque

# ROS 2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, String
from cv_bridge import CvBridge

# DRL imports
from ppo_agent import PPOAgent
from network_architectures import CarlaActorCritic
from environment_wrapper import CarlaEnvironmentWrapper
from reward_functions import CarlaRewardCalculator
from visualization import TrainingVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drl_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CarlaROS2Node(Node):
    """ROS 2 node for CARLA DRL training"""
    
    def __init__(self, config: Dict):
        super().__init__('carla_drl_node')
        
        self.config = config
        self.cv_bridge = CvBridge()
        
        # Data storage
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_vehicle_state = None
        self.latest_reward = 0.0
        self.episode_done = False
        
        # Threading locks
        self.data_lock = threading.Lock()
        
        # Setup ROS 2 subscriptions and publishers
        self._setup_ros2_interface()
        
        logger.info("CARLA ROS 2 node initialized")
    
    def _setup_ros2_interface(self):
        """Setup ROS 2 publishers and subscribers"""
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/carla/ego_vehicle/camera/rgb/image_raw',
            self._rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/carla/ego_vehicle/camera/depth/image_raw',
            self._depth_callback,
            10
        )
        
        self.vehicle_state_sub = self.create_subscription(
            String,  # JSON string with vehicle state
            '/carla/ego_vehicle/vehicle_status',
            self._vehicle_state_callback,
            10
        )
        
        self.reward_sub = self.create_subscription(
            Float32,
            '/carla/training/reward',
            self._reward_callback,
            10
        )
        
        self.episode_info_sub = self.create_subscription(
            String,  # JSON string with episode info
            '/carla/training/episode_info',
            self._episode_info_callback,
            10
        )
        
        # Publishers
        self.control_pub = self.create_publisher(
            Twist,
            '/carla/ego_vehicle/vehicle_control_cmd',
            10
        )
        
        self.reset_pub = self.create_publisher(
            Bool,
            '/carla/training/episode_reset',
            10
        )
        
        logger.info("ROS 2 interface setup complete")
    
    def _rgb_callback(self, msg):
        """Handle RGB camera data"""
        try:
            with self.data_lock:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                self.latest_rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"RGB callback error: {e}")
    
    def _depth_callback(self, msg):
        """Handle depth camera data"""
        try:
            with self.data_lock:
                self.latest_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            logger.error(f"Depth callback error: {e}")
    
    def _vehicle_state_callback(self, msg):
        """Handle vehicle state data"""
        try:
            import json
            with self.data_lock:
                self.latest_vehicle_state = json.loads(msg.data)
        except Exception as e:
            logger.error(f"Vehicle state callback error: {e}")
    
    def _reward_callback(self, msg):
        """Handle reward signal"""
        with self.data_lock:
            self.latest_reward = msg.data
    
    def _episode_info_callback(self, msg):
        """Handle episode information"""
        try:
            import json
            episode_info = json.loads(msg.data)
            with self.data_lock:
                self.episode_done = episode_info.get('done', False)
        except Exception as e:
            logger.error(f"Episode info callback error: {e}")
    
    def get_observation(self) -> Optional[Dict]:
        """Get current observation from ROS topics"""
        with self.data_lock:
            if (self.latest_rgb_image is not None and 
                self.latest_vehicle_state is not None):
                
                # Preprocess RGB image
                rgb_resized = cv2.resize(self.latest_rgb_image, (84, 84))
                rgb_normalized = rgb_resized.astype(np.float32) / 255.0
                rgb_tensor = np.transpose(rgb_normalized, (2, 0, 1))  # CHW format
                
                # Preprocess depth image (if available)
                depth_tensor = None
                if self.latest_depth_image is not None:
                    depth_resized = cv2.resize(self.latest_depth_image, (84, 84))
                    depth_normalized = np.clip(depth_resized / 100.0, 0, 1)  # Normalize to [0,1]
                    depth_tensor = depth_normalized[np.newaxis, ...]  # Add channel dimension
                
                # Extract vehicle state vector
                state = self.latest_vehicle_state
                vehicle_vector = np.array([
                    state['velocity']['x'],
                    state['velocity']['y'], 
                    state['velocity']['speed_kmh'] / 100.0,  # Normalize speed
                    state['rotation']['yaw'] / 180.0,  # Normalize angle
                    state['acceleration']['x'],
                    state['acceleration']['y']
                ], dtype=np.float32)
                
                observation = {
                    'camera_rgb': rgb_tensor,
                    'camera_depth': depth_tensor,
                    'vehicle_state': vehicle_vector,
                    'reward': self.latest_reward,
                    'done': self.episode_done
                }
                
                return observation
        
        return None
    
    def send_control_command(self, action: np.ndarray):
        """Send control command to CARLA"""
        try:
            # Convert action to control message
            control_msg = Twist()
            
            # Map normalized actions [-1, 1] to control values
            throttle_brake = action[0]  # Combined throttle/brake
            steering = action[1]        # Steering angle
            
            if throttle_brake >= 0:
                control_msg.linear.x = float(throttle_brake)  # Throttle
            else:
                control_msg.linear.x = float(throttle_brake)  # Brake (negative)
            
            control_msg.angular.z = float(steering)
            
            self.control_pub.publish(control_msg)
            
        except Exception as e:
            logger.error(f"Control command error: {e}")
    
    def reset_episode(self):
        """Send episode reset signal"""
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_pub.publish(reset_msg)

class PPOTrainer:
    """PPO trainer for CARLA DRL"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize ROS 2
        rclpy.init()
        self.ros_node = CarlaROS2Node(config)
        
        # Training configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            observation_space=self._get_observation_space(),
            action_space=self._get_action_space(),
            config=config['drl']['ppo'],
            device=self.device
        )
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        
        # Monitoring
        self.tensorboard_writer = SummaryWriter(
            log_dir=f"../monitoring/tensorboard_logs/ppo_{int(time.time())}"
        )
        
        # Visualization
        self.visualizer = TrainingVisualizer(config['visualization'])
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        logger.info("PPO trainer initialized")
    
    def _get_observation_space(self) -> Dict:
        """Define observation space"""
        return {
            'camera_rgb': {'shape': (3, 84, 84), 'dtype': 'float32'},
            'camera_depth': {'shape': (1, 84, 84), 'dtype': 'float32'},
            'vehicle_state': {'shape': (6,), 'dtype': 'float32'}
        }
    
    def _get_action_space(self) -> Dict:
        """Define action space"""
        return {
            'shape': (2,),  # [throttle_brake, steering]
            'low': [-1.0, -1.0],
            'high': [1.0, 1.0],
            'dtype': 'float32'
        }
    
    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Wait for initial data from CARLA"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            
            observation = self.ros_node.get_observation()
            if observation is not None:
                logger.info("✅ Initial data received from CARLA")
                return True
        
        logger.error("❌ Timeout waiting for CARLA data")
        return False
    
    def train_episode(self) -> Dict:
        """Train one episode"""
        episode_start_time = time.time()
        episode_reward = 0.0
        episode_length = 0
        episode_losses = []
        
        # Reset episode
        self.ros_node.reset_episode()
        time.sleep(1.0)  # Wait for reset
        
        # Wait for initial observation
        if not self.wait_for_data():
            return {'reward': 0.0, 'length': 0, 'losses': []}
        
        logger.info(f"🚀 Starting episode {self.episode}")
        
        # Episode loop
        done = False
        step = 0
        max_steps = self.config['drl']['training']['max_steps_per_episode']
        
        while not done and step < max_steps:
            # Get observation
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)
            observation = self.ros_node.get_observation()
            
            if observation is None:
                time.sleep(0.01)
                continue
            
            # Select action
            action, action_log_prob, value = self.agent.select_action(observation)
            
            # Send action to CARLA
            self.ros_node.send_control_command(action)
            
            # Wait for next observation
            time.sleep(0.033)  # ~30 FPS
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)
            next_observation = self.ros_node.get_observation()
            
            if next_observation is not None:
                # Store transition
                reward = next_observation['reward']
                done = next_observation['done']
                
                self.agent.store_transition(
                    observation, action, reward, action_log_prob, value, done
                )
                
                episode_reward += reward
                episode_length += 1
                step += 1
                self.total_steps += 1
                
                # Update visualization
                if step % 10 == 0:  # Update every 10 steps
                    self.visualizer.update_realtime_plots({
                        'step_reward': reward,
                        'episode_reward': episode_reward,
                        'action': action,
                        'observation': observation
                    })
        
        # Train agent at end of episode
        if len(self.agent.memory) > 0:
            losses = self.agent.update()
            episode_losses.extend(losses)
        
        # Episode statistics
        episode_duration = time.time() - episode_start_time
        avg_reward = episode_reward / max(episode_length, 1)
        
        episode_stats = {
            'reward': episode_reward,
            'length': episode_length,
            'duration': episode_duration,
            'avg_reward': avg_reward,
            'losses': episode_losses
        }
        
        logger.info(f"✅ Episode {self.episode} complete: "
                   f"Reward={episode_reward:.2f}, "
                   f"Length={episode_length}, "
                   f"Duration={episode_duration:.1f}s")
        
        return episode_stats
    
    def evaluate_agent(self, num_episodes: int = 5) -> Dict:
        """Evaluate trained agent"""
        logger.info(f"🎯 Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_lengths = []
        
        # Set agent to evaluation mode
        self.agent.eval()
        
        for eval_ep in range(num_episodes):
            episode_reward = 0.0
            episode_length = 0
            
            # Reset episode
            self.ros_node.reset_episode()
            time.sleep(1.0)
            
            if not self.wait_for_data():
                continue
            
            done = False
            step = 0
            max_steps = 500  # Shorter evaluation episodes
            
            while not done and step < max_steps:
                rclpy.spin_once(self.ros_node, timeout_sec=0.01)
                observation = self.ros_node.get_observation()
                
                if observation is None:
                    continue
                
                # Select action deterministically
                with torch.no_grad():
                    action, _, _ = self.agent.select_action(observation, deterministic=True)
                
                self.ros_node.send_control_command(action)
                
                time.sleep(0.033)
                rclpy.spin_once(self.ros_node, timeout_sec=0.01)
                next_observation = self.ros_node.get_observation()
                
                if next_observation is not None:
                    reward = next_observation['reward']
                    done = next_observation['done']
                    
                    episode_reward += reward
                    episode_length += 1
                    step += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            logger.info(f"  Eval episode {eval_ep + 1}: "
                       f"Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Restore training mode
        self.agent.train()
        
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'rewards': eval_rewards,
            'lengths': eval_lengths
        }
        
        logger.info(f"✅ Evaluation complete: "
                   f"Mean reward={eval_stats['mean_reward']:.2f}±{eval_stats['std_reward']:.2f}")
        
        return eval_stats
    
    def save_checkpoint(self, episode: int, reward: float):
        """Save training checkpoint"""
        checkpoint_dir = Path("../monitoring/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'agent_state_dict': self.agent.state_dict(),
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pth")
        
        # Save best checkpoint
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(checkpoint, checkpoint_dir / "best_checkpoint.pth")
            logger.info(f"💾 New best checkpoint saved: reward={reward:.2f}")
        
        # Save periodic checkpoint
        if episode % 100 == 0:
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_episode_{episode}.pth")
    
    def log_metrics(self, episode_stats: Dict):
        """Log training metrics"""
        episode = self.episode
        
        # TensorBoard logging
        self.tensorboard_writer.add_scalar('Episode/Reward', episode_stats['reward'], episode)
        self.tensorboard_writer.add_scalar('Episode/Length', episode_stats['length'], episode)
        self.tensorboard_writer.add_scalar('Episode/Duration', episode_stats['duration'], episode)
        self.tensorboard_writer.add_scalar('Episode/AvgReward', episode_stats['avg_reward'], episode)
        
        if episode_stats['losses']:
            avg_loss = np.mean(episode_stats['losses'])
            self.tensorboard_writer.add_scalar('Training/Loss', avg_loss, episode)
        
        # Rolling averages
        self.episode_rewards.append(episode_stats['reward'])
        self.episode_lengths.append(episode_stats['length'])
        
        if len(self.episode_rewards) >= 10:
            avg_reward_10 = np.mean(list(self.episode_rewards)[-10:])
            avg_length_10 = np.mean(list(self.episode_lengths)[-10:])
            
            self.tensorboard_writer.add_scalar('RollingAvg/Reward_10', avg_reward_10, episode)
            self.tensorboard_writer.add_scalar('RollingAvg/Length_10', avg_length_10, episode)
        
        if len(self.episode_rewards) >= 100:
            avg_reward_100 = np.mean(self.episode_rewards)
            avg_length_100 = np.mean(self.episode_lengths)
            
            self.tensorboard_writer.add_scalar('RollingAvg/Reward_100', avg_reward_100, episode)
            self.tensorboard_writer.add_scalar('RollingAvg/Length_100', avg_length_100, episode)
    
    def train(self):
        """Main training loop"""
        logger.info("🎓 Starting PPO training...")
        
        max_episodes = self.config['drl']['training']['total_episodes']
        save_interval = self.config['drl']['training']['save_interval']
        eval_interval = self.config['drl']['training']['eval_interval']
        
        try:
            for episode in range(max_episodes):
                self.episode = episode
                
                # Train episode
                episode_stats = self.train_episode()
                
                # Log metrics
                self.log_metrics(episode_stats)
                
                # Save checkpoint
                if episode % save_interval == 0:
                    self.save_checkpoint(episode, episode_stats['reward'])
                
                # Evaluate agent
                if episode % eval_interval == 0 and episode > 0:
                    eval_stats = self.evaluate_agent()
                    self.tensorboard_writer.add_scalar('Evaluation/MeanReward', 
                                                     eval_stats['mean_reward'], episode)
                
                # Update visualization
                self.visualizer.update_episode_plots(episode_stats)
                
                # Check early stopping conditions
                if len(self.episode_rewards) >= 100:
                    recent_avg = np.mean(list(self.episode_rewards)[-100:])
                    if recent_avg > 1000:  # Success threshold
                        logger.info(f"🎉 Training goal achieved! Average reward: {recent_avg:.2f}")
                        break
        
        except KeyboardInterrupt:
            logger.info("⚠️ Training interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up training resources...")
        
        # Save final checkpoint
        self.save_checkpoint(self.episode, self.episode_rewards[-1] if self.episode_rewards else 0.0)
        
        # Close TensorBoard writer
        self.tensorboard_writer.close()
        
        # Cleanup visualizer
        self.visualizer.cleanup()
        
        # Shutdown ROS 2
        self.ros_node.destroy_node()
        rclpy.shutdown()
        
        logger.info("✅ Training cleanup complete")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CARLA PPO Training')
    parser.add_argument('--config', default='../configs/complete_system_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--checkpoint', help='Checkpoint file to load')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.episodes:
        config['drl']['training']['total_episodes'] = args.episodes
    
    if args.visualize:
        config['visualization']['plots']['enabled'] = True
        config['visualization']['camera_display']['enabled'] = True
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            trainer.agent.load_state_dict(checkpoint['agent_state_dict'])
            trainer.episode = checkpoint['episode']
            trainer.total_steps = checkpoint['total_steps']
            trainer.best_reward = checkpoint['best_reward']
            logger.info(f"✅ Checkpoint loaded: episode {trainer.episode}")
        else:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return 1
    
    try:
        if args.eval_only:
            # Evaluation mode
            eval_stats = trainer.evaluate_agent(num_episodes=10)
            print(f"\n🎯 Evaluation Results:")
            print(f"Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"Mean Length: {eval_stats['mean_length']:.1f}")
        else:
            # Training mode
            logger.info("🚀 Starting CARLA PPO training with real-time visualization!")
            trainer.train()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return 1
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    sys.exit(main())

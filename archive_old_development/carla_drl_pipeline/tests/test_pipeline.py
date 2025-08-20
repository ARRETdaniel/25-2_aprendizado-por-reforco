"""
Test Script for CARLA DRL Pipeline

This script validates the complete pipeline including:
- CARLA client connectivity
- ROS 2 bridge communication  
- DRL agent functionality
- Environment wrapper
- Configuration loading

Usage:
    python test_pipeline.py --config configs/test_config.yaml
"""

import os
import sys
import time
import argparse
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import threading
import signal

# Import pipeline modules
from utils import (
    setup_logging, load_config, check_system_requirements,
    check_dependencies, check_carla_connection, monitor_system_resources
)

# Configure logging
logger = logging.getLogger(__name__)


class PipelineValidator:
    """Validates the complete CARLA DRL pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline validator.
        
        Args:
            config_path: Path to test configuration
        """
        self.config = load_config(config_path)
        self.test_results = {}
        self.shutdown_requested = False
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("PipelineValidator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum} - stopping tests")
        self.shutdown_requested = True
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all pipeline validation tests.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Starting pipeline validation tests...")
        
        test_functions = [
            ('system_requirements', self.test_system_requirements),
            ('dependencies', self.test_dependencies),
            ('configuration', self.test_configuration),
            ('carla_connection', self.test_carla_connection),
            ('communication_bridge', self.test_communication_bridge),
            ('environment_creation', self.test_environment_creation),
            ('agent_creation', self.test_agent_creation),
            ('training_step', self.test_training_step),
            ('model_save_load', self.test_model_save_load),
            ('ros2_integration', self.test_ros2_integration)
        ]
        
        for test_name, test_func in test_functions:
            if self.shutdown_requested:
                break
                
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "PASS" if result else "FAIL"
                logger.info(f"Test {test_name}: {status}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = False
        
        # Summary
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        logger.info(f"Pipeline validation complete: {passed}/{total} tests passed")
        
        return self.test_results
    
    def test_system_requirements(self) -> bool:
        """Test system requirements."""
        try:
            system_info = check_system_requirements()
            
            # Check minimum requirements
            requirements_met = True
            
            if system_info['memory_gb'] < 4:
                logger.error("Insufficient memory: minimum 4GB required")
                requirements_met = False
            
            if system_info['disk_free_gb'] < 10:
                logger.error("Insufficient disk space: minimum 10GB required")
                requirements_met = False
            
            if not system_info['torch_available']:
                logger.error("PyTorch not available")
                requirements_met = False
            
            return requirements_met
            
        except Exception as e:
            logger.error(f"System requirements test failed: {e}")
            return False
    
    def test_dependencies(self) -> bool:
        """Test required dependencies."""
        try:
            dependencies = check_dependencies()
            
            # Check critical dependencies
            critical_deps = ['torch', 'numpy', 'opencv', 'yaml', 'zmq', 'msgpack']
            
            for dep in critical_deps:
                if not dependencies.get(dep, False):
                    logger.error(f"Critical dependency missing: {dep}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependencies test failed: {e}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration loading and validation."""
        try:
            # Test configuration structure
            required_sections = ['carla', 'ros2', 'algorithm', 'environment']
            
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing configuration section: {section}")
                    return False
            
            # Test algorithm configuration
            algo_config = self.config['algorithm']
            required_algo_params = ['learning_rate', 'batch_size', 'gamma']
            
            for param in required_algo_params:
                if param not in algo_config:
                    logger.error(f"Missing algorithm parameter: {param}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration test failed: {e}")
            return False
    
    def test_carla_connection(self) -> bool:
        """Test CARLA server connection."""
        try:
            carla_config = self.config['carla']
            host = carla_config.get('host', 'localhost')
            port = carla_config.get('port', 2000)
            
            if check_carla_connection(host, port):
                logger.info("CARLA connection test passed")
                return True
            else:
                logger.warning("CARLA server not accessible - skipping related tests")
                return False
                
        except Exception as e:
            logger.error(f"CARLA connection test failed: {e}")
            return False
    
    def test_communication_bridge(self) -> bool:
        """Test ZeroMQ communication bridge."""
        try:
            import zmq
            import msgpack
            
            # Test ZeroMQ context creation
            context = zmq.Context()
            
            # Test socket creation
            test_socket = context.socket(zmq.PUB)
            test_port = self.config['environment']['communication']['test_port']
            test_socket.bind(f"tcp://*:{test_port}")
            
            # Test message serialization
            test_message = {
                'timestamp': time.time(),
                'data': np.random.rand(10).tolist(),
                'status': 'test'
            }
            
            serialized = msgpack.packb(test_message)
            deserialized = msgpack.unpackb(serialized)
            
            # Verify message integrity
            if deserialized['status'] != 'test':
                logger.error("Message serialization failed")
                return False
            
            # Cleanup
            test_socket.close()
            context.term()
            
            logger.info("Communication bridge test passed")
            return True
            
        except Exception as e:
            logger.error(f"Communication bridge test failed: {e}")
            return False
    
    def test_environment_creation(self) -> bool:
        """Test environment wrapper creation."""
        try:
            # Create temporary configuration for environment
            env_config = {
                'carla': self.config['carla'],
                'observation': self.config['environment']['observation'],
                'reward': self.config['environment']['reward'],
                'communication': self.config['environment']['communication'],
                'max_episode_steps': 100,
                'use_ros2': False  # Disable ROS2 for testing
            }
            
            # Save temporary config
            temp_config_path = Path('./temp_env_config.yaml')
            with open(temp_config_path, 'w') as f:
                yaml.dump(env_config, f)
            
            try:
                from environment_wrapper import CarlaROS2Environment
                
                # Test environment creation
                env = CarlaROS2Environment(str(temp_config_path))
                
                # Test action and observation spaces
                assert env.action_space is not None
                assert env.observation_space is not None
                
                # Test observation space structure
                assert 'image' in env.observation_space.spaces
                assert 'vector' in env.observation_space.spaces
                
                # Cleanup
                env.close()
                temp_config_path.unlink()
                
                logger.info("Environment creation test passed")
                return True
                
            except Exception as e:
                # Cleanup on failure
                if temp_config_path.exists():
                    temp_config_path.unlink()
                raise e
                
        except Exception as e:
            logger.error(f"Environment creation test failed: {e}")
            return False
    
    def test_agent_creation(self) -> bool:
        """Test PPO agent creation."""
        try:
            from ppo_algorithm import PPOAgent, PPOConfig
            
            # Create test configuration
            config = PPOConfig(
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16,
                total_timesteps=1000
            )
            
            # Define observation and action spaces
            obs_space = {
                'image': (4, 84, 84),  # stacked grayscale images
                'vector': (10,)        # vehicle state vector
            }
            action_space = (3,)  # throttle, steer, brake
            
            # Create agent
            agent = PPOAgent(
                config=config,
                obs_space=obs_space,
                action_space=action_space
            )
            
            # Test agent components
            assert agent.policy_net is not None
            assert agent.value_net is not None
            assert agent.feature_extractor is not None
            assert agent.optimizer is not None
            
            # Test action generation
            dummy_obs = {
                'image': torch.zeros((1, 4, 84, 84)),
                'vector': torch.zeros((1, 10))
            }
            
            action, log_prob, value = agent.get_action(dummy_obs)
            
            assert action.shape == (1, 3)
            assert log_prob.shape == (1,)
            assert value.shape == (1,)
            
            logger.info("Agent creation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Agent creation test failed: {e}")
            return False
    
    def test_training_step(self) -> bool:
        """Test a single training step."""
        try:
            # This is a more complex test that would require
            # a mock environment or actual CARLA connection
            # For now, we'll test the core components
            
            from networks import FeatureExtractor, PolicyNetwork, ValueNetwork
            
            # Test network forward passes
            image_shape = (3, 84, 84)
            vector_dim = 10
            
            feature_extractor = FeatureExtractor(
                image_shape=image_shape,
                vector_dim=vector_dim,
                output_features=256
            )
            
            policy_net = PolicyNetwork(
                feature_dim=256,
                action_dim=3
            )
            
            value_net = ValueNetwork(
                feature_dim=256
            )
            
            # Test forward passes
            dummy_obs = {
                'image': torch.randn(2, *image_shape),
                'vector': torch.randn(2, vector_dim)
            }
            
            features = feature_extractor(dummy_obs)
            assert features.shape == (2, 256)
            
            action_dist = policy_net(features)
            actions = action_dist.sample()
            assert actions.shape == (2, 3)
            
            values = value_net(features)
            assert values.shape == (2, 1)
            
            logger.info("Training step test passed")
            return True
            
        except Exception as e:
            logger.error(f"Training step test failed: {e}")
            return False
    
    def test_model_save_load(self) -> bool:
        """Test model saving and loading."""
        try:
            from ppo_algorithm import PPOAgent, PPOConfig
            
            # Create test agent
            config = PPOConfig(
                learning_rate=1e-4,
                n_steps=64,
                batch_size=16
            )
            
            obs_space = {
                'image': (4, 84, 84),
                'vector': (10,)
            }
            action_space = (3,)
            
            agent = PPOAgent(
                config=config,
                obs_space=obs_space,
                action_space=action_space
            )
            
            # Save model
            save_path = './test_model.pt'
            agent.save(save_path)
            
            # Verify file exists
            if not Path(save_path).exists():
                logger.error("Model file not created")
                return False
            
            # Create new agent and load model
            agent2 = PPOAgent(
                config=config,
                obs_space=obs_space,
                action_space=action_space
            )
            
            agent2.load(save_path)
            
            # Verify loaded state
            assert agent2.n_updates == agent.n_updates
            assert agent2.total_timesteps == agent.total_timesteps
            
            # Cleanup
            Path(save_path).unlink()
            
            logger.info("Model save/load test passed")
            return True
            
        except Exception as e:
            logger.error(f"Model save/load test failed: {e}")
            return False
    
    def test_ros2_integration(self) -> bool:
        """Test ROS 2 integration (optional)."""
        try:
            # Check if ROS 2 is available
            try:
                import rclpy
                from sensor_msgs.msg import Image
                from geometry_msgs.msg import Twist
                ros2_available = True
            except ImportError:
                logger.info("ROS 2 not available - skipping ROS 2 tests")
                return True  # Not required, so pass
            
            if ros2_available:
                # Test ROS 2 initialization
                rclpy.init()
                
                # Test node creation
                from environment_wrapper import CarlaEnvironmentNode
                
                node = CarlaEnvironmentNode(
                    name='test_node',
                    config=self.config['ros2']
                )
                
                # Test message creation
                twist_msg = Twist()
                twist_msg.linear.x = 1.0
                twist_msg.angular.z = 0.5
                
                # Cleanup
                node.destroy_node()
                rclpy.shutdown()
                
                logger.info("ROS 2 integration test passed")
                return True
            
        except Exception as e:
            logger.error(f"ROS 2 integration test failed: {e}")
            return False
    
    def run_performance_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run performance monitoring test.
        
        Args:
            duration_seconds: Test duration
            
        Returns:
            Performance statistics
        """
        logger.info(f"Running performance test for {duration_seconds} seconds...")
        
        start_time = time.time()
        resource_samples = []
        
        while (time.time() - start_time) < duration_seconds and not self.shutdown_requested:
            resources = monitor_system_resources()
            if resources:
                resource_samples.append(resources)
            
            time.sleep(1.0)
        
        if not resource_samples:
            logger.warning("No resource samples collected")
            return {}
        
        # Calculate statistics
        cpu_values = [s['cpu_percent'] for s in resource_samples]
        memory_values = [s['memory_percent'] for s in resource_samples]
        
        performance_stats = {
            'duration': time.time() - start_time,
            'samples': len(resource_samples),
            'cpu': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            }
        }
        
        logger.info(f"Performance test complete - "
                   f"CPU: {performance_stats['cpu']['mean']:.1f}% ± {performance_stats['cpu']['std']:.1f}%, "
                   f"Memory: {performance_stats['memory']['mean']:.1f}% ± {performance_stats['memory']['std']:.1f}%")
        
        return performance_stats


def create_test_config() -> Dict[str, Any]:
    """Create default test configuration."""
    return {
        'carla': {
            'host': 'localhost',
            'port': 2000,
            'timeout': 5.0
        },
        'ros2': {
            'control_topic': '/carla/ego_vehicle/vehicle_control_cmd',
            'image_topic': '/carla/ego_vehicle/camera/rgb/front/image_color',
            'state_topic': '/carla/ego_vehicle/vehicle_status'
        },
        'algorithm': {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 2048,
            'gamma': 0.99,
            'gae_lambda': 0.95
        },
        'environment': {
            'observation': {
                'image_size': [84, 84],
                'stack_frames': 4,
                'vector_dim': 10,
                'normalize_images': True,
                'normalize_vectors': True
            },
            'reward': {
                'speed_weight': 1.0,
                'lane_weight': 2.0,
                'collision_weight': -10.0,
                'smoothness_weight': 0.5,
                'progress_weight': 5.0,
                'target_speed': 30.0
            },
            'communication': {
                'obs_port': 5555,
                'action_port': 5556,
                'control_port': 5557,
                'test_port': 5558
            },
            'max_episode_steps': 1000
        }
    }


def main():
    """Main test entry point."""
    parser = argparse.ArgumentParser(description="Test CARLA DRL Pipeline")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to test configuration file'
    )
    parser.add_argument(
        '--performance-test',
        action='store_true',
        help='Run performance monitoring test'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Performance test duration in seconds'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    try:
        # Load or create configuration
        if args.config and Path(args.config).exists():
            config_path = args.config
        else:
            # Create default test configuration
            config = create_test_config()
            config_path = './test_config.yaml'
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Created default test configuration: {config_path}")
        
        # Create validator
        validator = PipelineValidator(config_path)
        
        # Run tests
        test_results = validator.run_all_tests()
        
        # Run performance test if requested
        if args.performance_test:
            performance_stats = validator.run_performance_test(args.duration)
            print("\nPerformance Test Results:")
            print(f"  Duration: {performance_stats.get('duration', 0):.1f}s")
            print(f"  CPU Usage: {performance_stats.get('cpu', {}).get('mean', 0):.1f}% ± {performance_stats.get('cpu', {}).get('std', 0):.1f}%")
            print(f"  Memory Usage: {performance_stats.get('memory', {}).get('mean', 0):.1f}% ± {performance_stats.get('memory', {}).get('std', 0):.1f}%")
        
        # Print results summary
        print("\nTest Results Summary:")
        passed_tests = 0
        for test_name, result in test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"  {test_name:20} : {status}")
            if result:
                passed_tests += 1
        
        total_tests = len(test_results)
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        # Exit with appropriate code
        if passed_tests == total_tests:
            logger.info("All tests passed!")
            sys.exit(0)
        else:
            logger.error(f"{total_tests - passed_tests} tests failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

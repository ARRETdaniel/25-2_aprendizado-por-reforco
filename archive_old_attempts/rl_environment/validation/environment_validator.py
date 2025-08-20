"""
Comprehensive validation test suite for CARLA RL environment wrapper.

This script performs systematic validation of all environment components:
1. Environment initialization with different parameters
2. Reset functionality and observation space validation
3. Action space handling (continuous and discrete)
4. Step function and transition dynamics
5. Reward function calculations
6. Termination conditions
7. Environment cleanup and resource management

Usage:
    python -m rl_environment.validation.environment_validator

Author: Autonomous Driving Research Team
Date: August 15, 2025
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("EnvValidator")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import environment wrapper
try:
    from rl_environment import CarlaEnvWrapper
except ImportError as e:
    logger.error(f"Failed to import CarlaEnvWrapper: {e}")
    sys.exit(1)


class EnvironmentValidator:
    """
    Comprehensive validator for the CARLA RL environment wrapper.

    This class runs a series of tests to validate different aspects of the
    environment wrapper's functionality, capturing metrics and producing
    a detailed report.
    """

    def __init__(self,
                output_dir: str = './validation_results',
                host: str = 'localhost',
                port: int = 2000):
        """
        Initialize the environment validator.

        Args:
            output_dir: Directory to save validation results
            host: CARLA server host
            port: CARLA server port
        """
        self.output_dir = output_dir
        self.host = host
        self.port = port
        self.env = None
        self.results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize file handler for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"validation_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info(f"Environment validator initialized. Results will be saved to {output_dir}")

    def _create_environment(self, **kwargs) -> CarlaEnvWrapper:
        """
        Create a CARLA environment with the given parameters.

        Args:
            **kwargs: Parameters to pass to CarlaEnvWrapper constructor

        Returns:
            Initialized CARLA environment
        """
        default_params = {
            'host': self.host,
            'port': self.port,
            'city_name': 'Town01',
            'image_size': (84, 84),
            'frame_skip': 2,
            'max_episode_steps': 200,
            'weather_id': 0,
            'quality_level': 'Low',
            'random_start': True
        }

        # Update default parameters with provided kwargs
        params = {**default_params, **kwargs}

        try:
            env = CarlaEnvWrapper(**params)
            return env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}", exc_info=True)
            raise

    def _close_environment(self) -> None:
        """Close the environment if it exists."""
        if self.env is not None:
            try:
                self.env.close()
                self.env = None
            except Exception as e:
                logger.error(f"Error closing environment: {e}", exc_info=True)

    def _record_test_result(self, test_name: str, passed: bool, details: Dict[str, Any] = None) -> None:
        """
        Record the result of a test.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            details: Additional details about the test result
        """
        if details is None:
            details = {}

        self.results["tests"][test_name] = {
            "passed": passed,
            "details": details
        }

        self.results["summary"]["total_tests"] += 1
        if passed:
            self.results["summary"]["passed"] += 1
        else:
            self.results["summary"]["failed"] += 1

        if passed:
            logger.info(f"✅ Test '{test_name}' passed")
        else:
            logger.error(f"❌ Test '{test_name}' failed: {details.get('error', 'No error details')}")

    def validate_initialization(self) -> bool:
        """
        Test environment initialization with different parameters.

        Returns:
            True if all initialization tests passed
        """
        logger.info("Testing environment initialization...")

        # Test with default parameters
        try:
            self.env = self._create_environment()
            self._record_test_result("init_default", True,
                                   {"message": "Environment initialized with default parameters"})
        except Exception as e:
            self._record_test_result("init_default", False,
                                   {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Test with different image sizes
        try:
            self.env = self._create_environment(image_size=(64, 64))
            self._record_test_result("init_image_size", True,
                                   {"message": "Environment initialized with custom image size"})
        except Exception as e:
            self._record_test_result("init_image_size", False,
                                   {"error": str(e)})
        finally:
            self._close_environment()

        # Test with different frame skip
        try:
            self.env = self._create_environment(frame_skip=4)
            self._record_test_result("init_frame_skip", True,
                                   {"message": "Environment initialized with custom frame skip"})
        except Exception as e:
            self._record_test_result("init_frame_skip", False,
                                   {"error": str(e)})
        finally:
            self._close_environment()

        # Check if all initialization tests passed
        init_tests = [test for test in self.results["tests"] if test.startswith("init_")]
        return all(self.results["tests"][test]["passed"] for test in init_tests)

    def validate_reset(self) -> bool:
        """
        Test environment reset functionality.

        Returns:
            True if all reset tests passed
        """
        logger.info("Testing environment reset functionality...")

        try:
            self.env = self._create_environment()

            # Test reset
            observation = self.env.reset()

            # Check observation structure
            observation_keys = list(observation.keys())
            expected_keys = ['image', 'vehicle_state', 'navigation', 'detections']
            missing_keys = [key for key in expected_keys if key not in observation_keys]

            if missing_keys:
                self._record_test_result("reset_observation_structure", False,
                                      {"error": f"Missing observation keys: {missing_keys}",
                                       "actual_keys": observation_keys})
                return False

            # Check observation shapes
            shapes = {key: observation[key].shape for key in observation_keys}
            expected_shapes = {
                'image': (84, 84, 3),
                'vehicle_state': (9,),
                'navigation': (3,),
                'detections': (10,)
            }

            shape_errors = []
            for key, expected_shape in expected_shapes.items():
                if key in shapes and shapes[key] != expected_shape:
                    shape_errors.append(f"{key}: expected {expected_shape}, got {shapes[key]}")

            if shape_errors:
                self._record_test_result("reset_observation_shapes", False,
                                      {"error": f"Incorrect observation shapes: {shape_errors}",
                                       "actual_shapes": shapes})
            else:
                self._record_test_result("reset_observation_shapes", True,
                                      {"shapes": shapes})

            # Check observation values
            value_errors = []

            # Check image range (should be 0-255 for uint8 RGB)
            if observation['image'].min() < 0 or observation['image'].max() > 255:
                value_errors.append(f"Image values out of range [0, 255]: min={observation['image'].min()}, max={observation['image'].max()}")

            # Check vehicle state for NaN or inf
            if np.any(np.isnan(observation['vehicle_state'])) or np.any(np.isinf(observation['vehicle_state'])):
                value_errors.append("Vehicle state contains NaN or inf values")

            # Check navigation data for NaN or inf
            if np.any(np.isnan(observation['navigation'])) or np.any(np.isinf(observation['navigation'])):
                value_errors.append("Navigation data contains NaN or inf values")

            if value_errors:
                self._record_test_result("reset_observation_values", False,
                                      {"error": f"Observation value issues: {value_errors}"})
            else:
                self._record_test_result("reset_observation_values", True,
                                      {"message": "Observation values within expected ranges"})

            # Multiple resets test
            try:
                for i in range(3):
                    observation = self.env.reset()
                    time.sleep(0.5)  # Brief delay to allow environment to reset

                self._record_test_result("reset_multiple", True,
                                      {"message": "Multiple resets completed successfully"})
            except Exception as e:
                self._record_test_result("reset_multiple", False,
                                      {"error": f"Multiple resets failed: {str(e)}"})

        except Exception as e:
            self._record_test_result("reset", False,
                                  {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Check if all reset tests passed
        reset_tests = [test for test in self.results["tests"] if test.startswith("reset_")]
        return all(self.results["tests"][test]["passed"] for test in reset_tests)

    def validate_action_space(self) -> bool:
        """
        Test environment action space handling.

        Returns:
            True if all action space tests passed
        """
        logger.info("Testing action space handling...")

        # Test continuous action space
        try:
            self.env = self._create_environment()

            # Check action_space_type attribute
            if not hasattr(self.env, 'action_space_type'):
                self._record_test_result("action_space_type_attribute", False,
                                      {"error": "Environment missing action_space_type attribute"})
                return False

            # Get action space type
            action_space_type = self.env.action_space_type
            self._record_test_result("action_space_type", True,
                                  {"action_space_type": action_space_type})

            # Reset environment
            observation = self.env.reset()

            # Test continuous actions if applicable
            if action_space_type == 'continuous':
                # Test valid continuous actions
                action = np.array([0.5, 0.0, 0.0])  # Throttle, Brake, Steer
                try:
                    next_obs, reward, done, info = self.env.step(action)
                    self._record_test_result("continuous_action_valid", True,
                                          {"action": action.tolist(), "reward": float(reward)})
                except Exception as e:
                    self._record_test_result("continuous_action_valid", False,
                                          {"error": str(e), "action": action.tolist()})

                # Test out-of-bounds continuous actions
                action = np.array([1.5, -0.5, 2.0])  # Out of bounds values
                try:
                    next_obs, reward, done, info = self.env.step(action)
                    # Check if the environment clipped the values properly
                    if 'clipped_action' in info:
                        clipped_action = info['clipped_action']
                        if (clipped_action[0] <= 1.0 and clipped_action[0] >= 0.0 and
                            clipped_action[1] <= 1.0 and clipped_action[1] >= 0.0 and
                            clipped_action[2] <= 1.0 and clipped_action[2] >= -1.0):
                            self._record_test_result("continuous_action_clipping", True,
                                                  {"original": action.tolist(), "clipped": clipped_action.tolist()})
                        else:
                            self._record_test_result("continuous_action_clipping", False,
                                                  {"error": "Action not properly clipped",
                                                   "original": action.tolist(),
                                                   "clipped": clipped_action.tolist()})
                    else:
                        # Action handling passed but clipping not explicitly reported
                        self._record_test_result("continuous_action_handling", True,
                                              {"message": "Out-of-bounds action accepted without error"})
                except Exception as e:
                    self._record_test_result("continuous_action_handling", False,
                                          {"error": str(e), "action": action.tolist()})

            # Test discrete actions if applicable
            if action_space_type == 'discrete':
                # Test valid discrete action
                action = 0  # Idle action
                try:
                    next_obs, reward, done, info = self.env.step(action)
                    self._record_test_result("discrete_action_valid", True,
                                          {"action": action, "reward": float(reward)})
                except Exception as e:
                    self._record_test_result("discrete_action_valid", False,
                                          {"error": str(e), "action": action})

                # Test out-of-bounds discrete action
                action = 10  # Should be out of bounds for 9 discrete actions
                try:
                    next_obs, reward, done, info = self.env.step(action)
                    self._record_test_result("discrete_action_bounds", False,
                                          {"error": "Out-of-bounds discrete action accepted without error",
                                           "action": action})
                except Exception as e:
                    self._record_test_result("discrete_action_bounds", True,
                                          {"message": f"Out-of-bounds action properly rejected: {str(e)}",
                                           "action": action})
        except Exception as e:
            self._record_test_result("action_space", False,
                                  {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Check if all action space tests passed
        action_tests = [test for test in self.results["tests"]
                       if test.startswith(("action_", "continuous_", "discrete_"))]
        return all(self.results["tests"][test]["passed"] for test in action_tests)

    def validate_step_function(self) -> bool:
        """
        Test environment step function and transition dynamics.

        Returns:
            True if all step function tests passed
        """
        logger.info("Testing step function and transition dynamics...")

        try:
            self.env = self._create_environment(max_episode_steps=100)

            # Reset environment
            observation = self.env.reset()

            # Test basic step function
            if self.env.action_space_type == 'continuous':
                action = np.array([0.5, 0.0, 0.0])  # Throttle, no brake, no steering
            else:
                action = 1  # Accelerate action in discrete space

            try:
                next_obs, reward, done, info = self.env.step(action)

                # Check if step function returns all expected values
                if not isinstance(next_obs, dict) or not isinstance(reward, (int, float)) or not isinstance(done, bool) or not isinstance(info, dict):
                    self._record_test_result("step_return_types", False,
                                          {"error": "Step function returned incorrect types",
                                           "types": {
                                               "observation": type(next_obs).__name__,
                                               "reward": type(reward).__name__,
                                               "done": type(done).__name__,
                                               "info": type(info).__name__
                                           }})
                else:
                    self._record_test_result("step_return_types", True,
                                          {"types": {
                                              "observation": type(next_obs).__name__,
                                              "reward": type(reward).__name__,
                                              "done": type(done).__name__,
                                              "info": type(info).__name__
                                          }})

                # Check observation consistency
                if set(next_obs.keys()) != set(observation.keys()):
                    self._record_test_result("step_observation_consistency", False,
                                          {"error": "Observation keys changed after step",
                                           "initial_keys": list(observation.keys()),
                                           "next_keys": list(next_obs.keys())})
                else:
                    self._record_test_result("step_observation_consistency", True,
                                          {"keys": list(next_obs.keys())})

                # Check reward value
                if not np.isfinite(reward):
                    self._record_test_result("step_reward_value", False,
                                          {"error": f"Reward is not finite: {reward}"})
                else:
                    self._record_test_result("step_reward_value", True,
                                          {"reward": float(reward)})

                # Check info dictionary
                expected_info_keys = ['episode_step', 'total_reward']
                missing_info_keys = [key for key in expected_info_keys if key not in info]
                if missing_info_keys:
                    self._record_test_result("step_info_keys", False,
                                          {"error": f"Missing info keys: {missing_info_keys}",
                                           "actual_keys": list(info.keys())})
                else:
                    self._record_test_result("step_info_keys", True,
                                          {"info_keys": list(info.keys())})

            except Exception as e:
                self._record_test_result("step_basic", False,
                                      {"error": str(e)})

            # Test multiple steps
            try:
                # Reset environment
                observation = self.env.reset()

                # Run multiple steps
                steps_to_run = 20
                step_results = []

                for i in range(steps_to_run):
                    if self.env.action_space_type == 'continuous':
                        # Oscillating steering
                        action = np.array([0.7, 0.0, 0.5 * np.sin(i * 0.5)])
                    else:
                        # Cycle through different actions
                        action = i % 9

                    next_obs, reward, done, info = self.env.step(action)

                    # Record basic metrics
                    if 'vehicle_state' in next_obs:
                        speed = np.linalg.norm(next_obs['vehicle_state'][3:6])  # vel x,y,z
                    else:
                        speed = None

                    step_results.append({
                        "step": i,
                        "reward": float(reward),
                        "done": done,
                        "speed": float(speed) if speed is not None else None
                    })

                    if done:
                        break

                self._record_test_result("step_multiple", True,
                                      {"steps_completed": len(step_results),
                                       "step_results": step_results})
            except Exception as e:
                self._record_test_result("step_multiple", False,
                                      {"error": str(e)})

        except Exception as e:
            self._record_test_result("step_function", False,
                                  {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Check if all step function tests passed
        step_tests = [test for test in self.results["tests"] if test.startswith("step_")]
        return all(self.results["tests"][test]["passed"] for test in step_tests)

    def validate_reward_function(self) -> bool:
        """
        Test environment reward function calculations.

        Returns:
            True if all reward function tests passed
        """
        logger.info("Testing reward function calculations...")

        try:
            # Test with default reward config
            self.env = self._create_environment()

            # Reset environment
            observation = self.env.reset()

            # Run a few steps to observe rewards
            rewards = []
            reward_config = getattr(self.env, 'reward_config', {})

            for i in range(10):
                if self.env.action_space_type == 'continuous':
                    # Try different actions to see reward variations
                    if i < 3:
                        action = np.array([0.8, 0.0, 0.0])  # Strong throttle
                    elif i < 6:
                        action = np.array([0.3, 0.0, 0.2])  # Gentle throttle with steering
                    else:
                        action = np.array([0.0, 0.5, 0.0])  # Brake
                else:
                    # Cycle through discrete actions
                    action = i % 9

                next_obs, reward, done, info = self.env.step(action)
                rewards.append(float(reward))

                if done:
                    break

            self._record_test_result("reward_default_config", True,
                                  {"reward_config": reward_config,
                                   "rewards": rewards,
                                   "mean_reward": float(np.mean(rewards)),
                                   "min_reward": float(np.min(rewards)),
                                   "max_reward": float(np.max(rewards))})

            # Check for reward variance
            if len(rewards) > 1 and np.var(rewards) < 1e-6:
                self._record_test_result("reward_variance", False,
                                      {"error": "Reward has almost no variance",
                                       "variance": float(np.var(rewards))})
            else:
                self._record_test_result("reward_variance", True,
                                      {"variance": float(np.var(rewards))})

            # Close environment
            self._close_environment()

            # Test with custom reward config
            custom_reward_config = {
                'progress_weight': 2.0,
                'lane_deviation_weight': 1.0,
                'collision_penalty': 200.0,
                'speed_weight': 0.5,
                'action_smoothness_weight': 0.2
            }

            self.env = self._create_environment(reward_config=custom_reward_config)

            # Reset environment
            observation = self.env.reset()

            # Run a few steps to observe rewards
            custom_rewards = []

            for i in range(10):
                if self.env.action_space_type == 'continuous':
                    # Use the same action pattern as before for comparison
                    if i < 3:
                        action = np.array([0.8, 0.0, 0.0])  # Strong throttle
                    elif i < 6:
                        action = np.array([0.3, 0.0, 0.2])  # Gentle throttle with steering
                    else:
                        action = np.array([0.0, 0.5, 0.0])  # Brake
                else:
                    # Cycle through discrete actions
                    action = i % 9

                next_obs, reward, done, info = self.env.step(action)
                custom_rewards.append(float(reward))

                if done:
                    break

            self._record_test_result("reward_custom_config", True,
                                  {"reward_config": custom_reward_config,
                                   "rewards": custom_rewards,
                                   "mean_reward": float(np.mean(custom_rewards)),
                                   "min_reward": float(np.min(custom_rewards)),
                                   "max_reward": float(np.max(custom_rewards))})

            # Check if custom config resulted in different rewards
            if len(rewards) > 0 and len(custom_rewards) > 0:
                mean_diff = abs(np.mean(rewards) - np.mean(custom_rewards))
                if mean_diff < 0.01:
                    self._record_test_result("reward_config_effect", False,
                                          {"error": "Custom reward config had minimal impact on rewards",
                                           "mean_difference": float(mean_diff)})
                else:
                    self._record_test_result("reward_config_effect", True,
                                          {"mean_difference": float(mean_diff)})

        except Exception as e:
            self._record_test_result("reward_function", False,
                                  {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Check if all reward function tests passed
        reward_tests = [test for test in self.results["tests"] if test.startswith("reward_")]
        return all(self.results["tests"][test]["passed"] for test in reward_tests)

    def validate_termination_conditions(self) -> bool:
        """
        Test environment termination conditions.

        Returns:
            True if all termination conditions tests passed
        """
        logger.info("Testing termination conditions...")

        try:
            # Test max episode steps termination
            self.env = self._create_environment(max_episode_steps=30)

            # Reset environment
            observation = self.env.reset()

            # Run until termination
            steps = 0
            done = False
            termination_reason = None

            while not done and steps < 50:  # Safety limit
                if self.env.action_space_type == 'continuous':
                    action = np.array([0.5, 0.0, 0.0])  # Throttle forward
                else:
                    action = 1  # Accelerate in discrete space

                _, _, done, info = self.env.step(action)
                steps += 1

                if done and 'termination_reason' in info:
                    termination_reason = info['termination_reason']

            if not done:
                self._record_test_result("termination_max_steps", False,
                                      {"error": f"Environment did not terminate after {steps} steps",
                                       "max_episode_steps": 30})
            elif termination_reason and 'max' in termination_reason.lower():
                self._record_test_result("termination_max_steps", True,
                                      {"steps": steps,
                                       "termination_reason": termination_reason})
            else:
                self._record_test_result("termination_max_steps", False,
                                      {"error": f"Environment terminated but not due to max steps: {termination_reason}",
                                       "steps": steps})

            # Close environment
            self._close_environment()

            # TODO: Ideally, we would test other termination conditions like:
            # - Collision termination
            # - Off-road termination
            # - Goal reached termination
            # However, these are harder to test deterministically

            # Instead, we'll run a longer episode to see if other termination conditions occur
            self.env = self._create_environment(max_episode_steps=300)

            # Reset environment
            observation = self.env.reset()

            # Run with random actions to try to trigger terminations
            steps = 0
            done = False
            termination_reason = None

            while not done and steps < 300:
                if self.env.action_space_type == 'continuous':
                    # Generate more erratic actions to try to cause collisions
                    action = np.random.uniform(low=-1, high=1, size=3)
                    # Normalize throttle and brake to [0, 1]
                    action[0] = (action[0] + 1) / 2  # throttle
                    action[1] = (action[1] + 1) / 2  # brake
                else:
                    action = np.random.randint(0, 9)

                _, _, done, info = self.env.step(action)
                steps += 1

                if done and 'termination_reason' in info:
                    termination_reason = info['termination_reason']
                    break

            self._record_test_result("termination_other_conditions", True,
                                  {"steps": steps,
                                   "termination_reason": termination_reason,
                                   "message": "Ran episode with random actions to check termination"})

        except Exception as e:
            self._record_test_result("termination_conditions", False,
                                  {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Check if all termination tests passed
        termination_tests = [test for test in self.results["tests"] if test.startswith("termination_")]
        return all(self.results["tests"][test]["passed"] for test in termination_tests)

    def validate_cleanup(self) -> bool:
        """
        Test environment cleanup and resource management.

        Returns:
            True if all cleanup tests passed
        """
        logger.info("Testing environment cleanup...")

        try:
            # Create and immediately close environment
            self.env = self._create_environment()
            self.env.close()
            self.env = None
            self._record_test_result("cleanup_basic", True,
                                  {"message": "Environment closed successfully"})

            # Create, reset, and close
            self.env = self._create_environment()
            observation = self.env.reset()
            self.env.close()
            self.env = None
            self._record_test_result("cleanup_after_reset", True,
                                  {"message": "Environment closed successfully after reset"})

            # Create, reset, step a few times, and close
            self.env = self._create_environment()
            observation = self.env.reset()

            for i in range(5):
                if self.env.action_space_type == 'continuous':
                    action = np.array([0.5, 0.0, 0.0])
                else:
                    action = 1

                next_obs, reward, done, info = self.env.step(action)
                if done:
                    break

            self.env.close()
            self.env = None
            self._record_test_result("cleanup_after_steps", True,
                                  {"message": "Environment closed successfully after steps"})

            # Multiple create/close cycles
            for i in range(3):
                self.env = self._create_environment()
                observation = self.env.reset()
                self.env.close()
                self.env = None
                time.sleep(1)  # Brief delay between cycles

            self._record_test_result("cleanup_multiple_cycles", True,
                                  {"message": "Multiple environment cycles completed successfully"})

        except Exception as e:
            self._record_test_result("cleanup", False,
                                  {"error": str(e)})
            return False
        finally:
            self._close_environment()

        # Check if all cleanup tests passed
        cleanup_tests = [test for test in self.results["tests"] if test.startswith("cleanup_")]
        return all(self.results["tests"][test]["passed"] for test in cleanup_tests)

    def run_all_tests(self) -> bool:
        """
        Run all validation tests.

        Returns:
            True if all tests passed
        """
        logger.info("Running all environment validation tests...")

        start_time = time.time()

        # Run all validation tests
        init_result = self.validate_initialization()
        reset_result = self.validate_reset()
        action_space_result = self.validate_action_space()
        step_result = self.validate_step_function()
        reward_result = self.validate_reward_function()
        termination_result = self.validate_termination_conditions()
        cleanup_result = self.validate_cleanup()

        duration = time.time() - start_time

        # Update summary
        self.results["summary"]["duration"] = duration

        # Calculate overall result
        overall_result = (init_result and reset_result and action_space_result and
                        step_result and reward_result and termination_result and cleanup_result)

        self.results["summary"]["overall_passed"] = overall_result

        # Update counts
        self.results["summary"]["passed"] = sum(1 for test in self.results["tests"].values() if test["passed"])
        self.results["summary"]["failed"] = sum(1 for test in self.results["tests"].values() if not test["passed"])
        self.results["summary"]["total_tests"] = len(self.results["tests"])

        # Save results
        self._save_results()

        # Log summary
        logger.info(f"Validation complete in {duration:.2f}s")
        logger.info(f"Tests: {self.results['summary']['total_tests']}, " +
                   f"Passed: {self.results['summary']['passed']}, " +
                   f"Failed: {self.results['summary']['failed']}")
        logger.info(f"Overall result: {'PASSED' if overall_result else 'FAILED'}")

        return overall_result

    def _save_results(self) -> None:
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"results_{timestamp}.json")

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Also save a summary file
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.txt")

        with open(summary_file, 'w') as f:
            f.write("Environment Validation Summary\n")
            f.write("============================\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {self.results['summary']['duration']:.2f}s\n")
            f.write(f"Tests: {self.results['summary']['total_tests']}\n")
            f.write(f"Passed: {self.results['summary']['passed']}\n")
            f.write(f"Failed: {self.results['summary']['failed']}\n")
            f.write(f"Overall result: {'PASSED' if self.results['summary']['overall_passed'] else 'FAILED'}\n")
            f.write("\n")
            f.write("Failed Tests:\n")
            f.write("------------\n")

            # List failed tests
            failed_tests = [(name, test) for name, test in self.results["tests"].items() if not test["passed"]]
            if failed_tests:
                for name, test in failed_tests:
                    f.write(f"- {name}: {test['details'].get('error', 'Unknown error')}\n")
            else:
                f.write("None\n")

        logger.info(f"Summary saved to {summary_file}")


def main():
    """Run the environment validator."""
    parser = argparse.ArgumentParser(description="CARLA RL Environment Validator")
    parser.add_argument("--host", type=str, default="localhost",
                      help="CARLA server host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000,
                      help="CARLA server port (default: 2000)")
    parser.add_argument("--output-dir", type=str, default="./validation_results",
                      help="Output directory for validation results (default: ./validation_results)")

    args = parser.parse_args()

    # Create and run validator
    validator = EnvironmentValidator(
        output_dir=args.output_dir,
        host=args.host,
        port=args.port
    )

    try:
        success = validator.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

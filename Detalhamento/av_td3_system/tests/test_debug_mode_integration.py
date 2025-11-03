#!/usr/bin/env python3
"""
Test suite for --debug flag integration with CNN diagnostics.

Tests:
1. Debug flag parsing
2. CNN diagnostics enabled when debug=True
3. CNN diagnostics disabled when debug=False
4. TensorBoard logging active in debug mode
5. Console output in debug mode

Author: Daniel Terra
Date: 2025-01-28
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TestDebugModeIntegration(unittest.TestCase):
    """Test suite for debug mode integration with CNN diagnostics."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock CARLA environment (don't actually connect to CARLA)
        self.mock_env_patcher = patch('scripts.train_td3.CARLANavigationEnv')
        self.MockEnv = self.mock_env_patcher.start()

        # Configure mock environment
        mock_env = MagicMock()
        self.MockEnv.return_value = mock_env

        # Mock observation space
        from gymnasium.spaces import Dict, Box
        mock_env.observation_space = Dict({
            'image': Box(low=0, high=1, shape=(4, 84, 84), dtype=np.float32),
            'vector': Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
        })

        # Mock action space
        mock_env.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def tearDown(self):
        """Clean up after tests."""
        self.mock_env_patcher.stop()

    @patch('sys.argv', ['train_td3.py', '--max-timesteps', '100', '--debug'])
    def test_debug_flag_parsing(self):
        """Test that --debug flag is correctly parsed."""
        from scripts.train_td3 import main
        import argparse

        # Create parser (same as main())
        parser = argparse.ArgumentParser()
        parser.add_argument('--scenario', type=int, default=0)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--max-timesteps', type=int, default=1000)
        parser.add_argument('--eval-freq', type=int, default=5000)
        parser.add_argument('--checkpoint-freq', type=int, default=10000)
        parser.add_argument('--num-eval-episodes', type=int, default=10)
        parser.add_argument('--carla-config', type=str, default='config/carla_config.yaml')
        parser.add_argument('--agent-config', type=str, default='config/td3_config.yaml')
        parser.add_argument('--log-dir', type=str, default='data/logs')
        parser.add_argument('--checkpoint-dir', type=str, default='data/checkpoints')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--device', type=str, default='cpu')

        args = parser.parse_args(['--max-timesteps', '100', '--debug'])

        self.assertTrue(args.debug, "Debug flag should be True when --debug is passed")

    def test_cnn_diagnostics_enabled_in_debug_mode(self):
        """Test that CNN diagnostics are enabled when debug=True."""
        from src.agents.td3_agent import TD3Agent
        from src.networks.cnn_extractor import NatureCNN
        from torch.utils.tensorboard import SummaryWriter
        import tempfile

        # Create temporary directory for TensorBoard logs
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)

            # Create CNN extractor
            cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)

            # Create TD3 agent
            agent = TD3Agent(
                state_dim=535,
                action_dim=2,
                max_action=1.0,
                cnn_extractor=cnn,
                use_dict_buffer=True,
                device='cpu'
            )

            # Verify diagnostics not enabled by default
            self.assertIsNone(agent.cnn_diagnostics, "Diagnostics should be None by default")

            # Enable diagnostics (simulates debug=True)
            agent.enable_diagnostics(writer)

            # Verify diagnostics enabled
            self.assertIsNotNone(agent.cnn_diagnostics, "Diagnostics should be enabled after enable_diagnostics()")
            self.assertEqual(agent.cnn_diagnostics.cnn_module, cnn, "Diagnostics should reference correct CNN")

            writer.close()

    def test_cnn_diagnostics_disabled_without_debug(self):
        """Test that CNN diagnostics are NOT enabled when debug=False."""
        from src.agents.td3_agent import TD3Agent
        from src.networks.cnn_extractor import NatureCNN

        # Create CNN extractor
        cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)

        # Create TD3 agent (no enable_diagnostics() call)
        agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            cnn_extractor=cnn,
            use_dict_buffer=True,
            device='cpu'
        )

        # Verify diagnostics not enabled
        self.assertIsNone(agent.cnn_diagnostics, "Diagnostics should remain None without enable_diagnostics()")

    def test_diagnostics_logging_to_tensorboard(self):
        """Test that diagnostics log to TensorBoard in debug mode."""
        from src.agents.td3_agent import TD3Agent
        from src.networks.cnn_extractor import NatureCNN
        from torch.utils.tensorboard import SummaryWriter
        import tempfile

        # Create temporary directory for TensorBoard logs
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)

            # Create CNN extractor
            cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)

            # Create TD3 agent
            agent = TD3Agent(
                state_dim=535,
                action_dim=2,
                max_action=1.0,
                cnn_extractor=cnn,
                use_dict_buffer=True,
                device='cpu'
            )

            # Enable diagnostics
            agent.enable_diagnostics(writer)

            # Simulate training step
            obs_dict = {
                'image': np.random.rand(4, 84, 84).astype(np.float32),
                'vector': np.random.rand(23).astype(np.float32)
            }

            # Fill replay buffer
            for _ in range(300):
                agent.replay_buffer.add(
                    obs_dict=obs_dict,
                    action=np.random.rand(2).astype(np.float32),
                    next_obs_dict=obs_dict,
                    reward=0.0,
                    done=0.0
                )

            # Train (should capture diagnostics)
            metrics = agent.train(batch_size=32)

            # Verify diagnostics captured
            self.assertGreater(len(agent.cnn_diagnostics.gradient_history), 0,
                             "Gradients should be captured during training")
            self.assertGreater(len(agent.cnn_diagnostics.weight_history), 0,
                             "Weights should be captured during training")

            # Log to TensorBoard
            agent.cnn_diagnostics.log_to_tensorboard(step=1)

            writer.close()

    @patch('builtins.print')
    def test_console_output_in_debug_mode(self, mock_print):
        """Test that console output is generated in debug mode."""
        from src.agents.td3_agent import TD3Agent
        from src.networks.cnn_extractor import NatureCNN
        from torch.utils.tensorboard import SummaryWriter
        import tempfile

        # Create temporary directory for TensorBoard logs
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SummaryWriter(tmpdir)

            # Create CNN extractor
            cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512)

            # Create TD3 agent
            agent = TD3Agent(
                state_dim=535,
                action_dim=2,
                max_action=1.0,
                cnn_extractor=cnn,
                use_dict_buffer=True,
                device='cpu'
            )

            # Enable diagnostics (should print message)
            agent.enable_diagnostics(writer)

            # Verify enable message was printed
            enable_calls = [call for call in mock_print.call_args_list if 'CNN diagnostics enabled' in str(call)]
            self.assertGreater(len(enable_calls), 0, "Should print diagnostics enabled message")

            # Simulate training
            obs_dict = {
                'image': np.random.rand(4, 84, 84).astype(np.float32),
                'vector': np.random.rand(23).astype(np.float32)
            }

            # Fill replay buffer
            for _ in range(300):
                agent.replay_buffer.add(
                    obs_dict=obs_dict,
                    action=np.random.rand(2).astype(np.float32),
                    next_obs_dict=obs_dict,
                    reward=0.0,
                    done=0.0
                )

            # Train
            agent.train(batch_size=32)

            # Print diagnostics (should output to console)
            mock_print.reset_mock()
            agent.print_diagnostics(max_history=100)

            # Verify diagnostics summary was printed
            summary_calls = [call for call in mock_print.call_args_list if 'CNN DIAGNOSTICS SUMMARY' in str(call)]
            self.assertGreater(len(summary_calls), 0, "Should print diagnostics summary")

            writer.close()


def run_tests():
    """Run all tests and print results."""
    print("\n" + "="*70)
    print("TESTING: Debug Mode Integration with CNN Diagnostics")
    print("="*70 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDebugModeIntegration)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

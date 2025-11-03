"""
Unit tests for get_stats() and get_gradient_stats() methods

Tests the comprehensive statistics collection following Phase 25 improvements:
- Bug #16 fix: Expanded statistics from 4 to 25+ metrics
- Type hint fix: Dict[str, Any] instead of Dict[str, any]
- Integration with training loop
- Gradient statistics collection

Reference:
- ANALYSIS_GET_STATS.md (comprehensive analysis)
- GET_STATS_SUMMARY.md (quick summary)
- Stable-Baselines3 TD3 best practices
- OpenAI Spinning Up monitoring guidelines

Author: Daniel Terra
Date: November 3, 2025
"""

import sys
import os
sys.path.insert(0, '/workspace/av_td3_system')

import numpy as np
import torch
import pytest

from src.agents.td3_agent import TD3Agent


class TestGetStats:
    """Test suite for get_stats() method (Bug #16 fix)"""
    
    @pytest.fixture
    def agent_standard(self):
        """Create TD3 agent with standard (non-Dict) buffer"""
        return TD3Agent(
            state_dim=10,
            action_dim=2,
            max_action=1.0,
            use_dict_buffer=False
        )
    
    @pytest.fixture
    def agent_dict(self):
        """Create TD3 agent with Dict buffer (separate CNNs)"""
        return TD3Agent(
            state_dim={'camera': (4, 84, 84), 'kinematics': 7},
            action_dim=2,
            max_action=1.0,
            use_dict_buffer=True
        )
    
    def test_basic_stats_present(self, agent_standard):
        """Test that all basic statistics are present in output"""
        stats = agent_standard.get_stats()
        
        # Check that all expected keys are present
        expected_basic_keys = [
            'total_iterations',
            'is_training',
            'exploration_phase',
            'replay_buffer_size',
            'replay_buffer_full',
            'buffer_utilization',
            'buffer_max_size',
            'use_dict_buffer',
            'actor_lr',
            'critic_lr',
            'discount',
            'tau',
            'policy_freq',
            'policy_noise',
            'noise_clip',
            'max_action',
            'learning_starts',
            'batch_size',
            'device'
        ]
        
        for key in expected_basic_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_network_stats_present(self, agent_standard):
        """Test that network parameter statistics are present"""
        stats = agent_standard.get_stats()
        
        expected_network_keys = [
            'actor_param_mean',
            'actor_param_std',
            'actor_param_max',
            'actor_param_min',
            'critic_param_mean',
            'critic_param_std',
            'critic_param_max',
            'critic_param_min',
            'target_actor_param_mean',
            'target_critic_param_mean',
        ]
        
        for key in expected_network_keys:
            assert key in stats, f"Missing network stat: {key}"
            assert isinstance(stats[key], float), f"{key} should be float"
    
    def test_cnn_stats_in_dict_buffer(self, agent_dict):
        """Test that CNN statistics are present when using Dict buffer"""
        stats = agent_dict.get_stats()
        
        # Check CNN-specific keys
        expected_cnn_keys = [
            'actor_cnn_lr',
            'critic_cnn_lr',
            'actor_cnn_param_mean',
            'actor_cnn_param_std',
            'actor_cnn_param_max',
            'actor_cnn_param_min',
            'critic_cnn_param_mean',
            'critic_cnn_param_std',
            'critic_cnn_param_max',
            'critic_cnn_param_min',
        ]
        
        for key in expected_cnn_keys:
            assert key in stats, f"Missing CNN stat: {key}"
            # CNN stats can be None or float
            if stats[key] is not None:
                assert isinstance(stats[key], float), f"{key} should be float or None"
    
    def test_cnn_stats_not_in_standard_buffer(self, agent_standard):
        """Test that CNN statistics are NOT present when using standard buffer"""
        stats = agent_standard.get_stats()
        
        # These keys should NOT be present
        cnn_keys = [
            'actor_cnn_lr',
            'critic_cnn_lr',
            'actor_cnn_param_mean',
            'critic_cnn_param_mean',
        ]
        
        for key in cnn_keys:
            assert key not in stats, f"CNN stat should not be present: {key}"
    
    def test_training_phase_indicator(self, agent_standard):
        """Test that training phase indicators are correct"""
        stats = agent_standard.get_stats()
        
        # Initially should be in exploration phase
        assert stats['is_training'] == False, "Should not be training initially"
        assert stats['exploration_phase'] == True, "Should be in exploration phase"
        assert stats['total_iterations'] == 0, "Should start at iteration 0"
        
        # Simulate some training steps
        agent_standard.total_it = agent_standard.learning_starts + 100
        stats = agent_standard.get_stats()
        
        assert stats['is_training'] == True, "Should be training after learning_starts"
        assert stats['exploration_phase'] == False, "Should not be in exploration phase"
    
    def test_buffer_utilization(self, agent_standard):
        """Test that buffer utilization is calculated correctly"""
        stats = agent_standard.get_stats()
        
        # Initially buffer should be empty
        assert stats['buffer_utilization'] == 0.0, "Buffer should be empty initially"
        
        # Add some transitions
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randn(2)
            next_state = np.random.randn(10)
            reward = np.random.randn()
            done = False
            agent_standard.replay_buffer.add(state, action, next_state, reward, done)
        
        stats = agent_standard.get_stats()
        expected_util = 100 / agent_standard.replay_buffer.max_size
        assert abs(stats['buffer_utilization'] - expected_util) < 1e-6, "Buffer utilization should match"
    
    def test_learning_rates_match_optimizers(self, agent_standard):
        """Test that reported learning rates match actual optimizer learning rates"""
        stats = agent_standard.get_stats()
        
        # Check that LRs match optimizer state
        actual_actor_lr = agent_standard.actor_optimizer.param_groups[0]['lr']
        actual_critic_lr = agent_standard.critic_optimizer.param_groups[0]['lr']
        
        assert stats['actor_lr'] == actual_actor_lr, "Actor LR mismatch"
        assert stats['critic_lr'] == actual_critic_lr, "Critic LR mismatch"
    
    def test_td3_hyperparameters(self, agent_standard):
        """Test that TD3 hyperparameters are correctly reported"""
        stats = agent_standard.get_stats()
        
        # These should match agent's internal state
        assert stats['discount'] == agent_standard.discount
        assert stats['tau'] == agent_standard.tau
        assert stats['policy_freq'] == agent_standard.policy_freq
        assert stats['policy_noise'] == agent_standard.policy_noise
        assert stats['noise_clip'] == agent_standard.noise_clip
        assert stats['max_action'] == agent_standard.max_action
    
    def test_type_hint_fix(self, agent_standard):
        """Test that return type is Dict[str, Any] (Bug #16 fix)"""
        stats = agent_standard.get_stats()
        
        # Should return a dictionary
        assert isinstance(stats, dict), "Should return dict"
        
        # Check that it contains mixed types (demonstrating 'Any')
        assert isinstance(stats['total_iterations'], int)
        assert isinstance(stats['is_training'], bool)
        assert isinstance(stats['actor_lr'], float)
        assert isinstance(stats['device'], str)


class TestGetGradientStats:
    """Test suite for get_gradient_stats() method"""
    
    @pytest.fixture
    def agent_standard(self):
        """Create TD3 agent with standard buffer"""
        agent = TD3Agent(
            state_dim=10,
            action_dim=2,
            max_action=1.0,
            use_dict_buffer=False
        )
        
        # Add some transitions to enable training
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randn(2)
            next_state = np.random.randn(10)
            reward = np.random.randn()
            done = False
            agent.replay_buffer.add(state, action, next_state, reward, done)
        
        agent.total_it = agent.learning_starts + 1
        return agent
    
    def test_gradient_stats_structure(self, agent_standard):
        """Test that gradient statistics have correct structure"""
        # Perform one training step to generate gradients
        agent_standard.train(batch_size=32)
        
        # Now get gradient stats (should have gradients from training)
        grad_stats = agent_standard.get_gradient_stats()
        
        # Check basic keys
        assert 'actor_grad_norm' in grad_stats
        assert 'critic_grad_norm' in grad_stats
    
    def test_gradient_norms_are_positive(self, agent_standard):
        """Test that gradient norms are non-negative"""
        agent_standard.train(batch_size=32)
        grad_stats = agent_standard.get_gradient_stats()
        
        assert grad_stats['actor_grad_norm'] >= 0, "Actor grad norm should be non-negative"
        assert grad_stats['critic_grad_norm'] >= 0, "Critic grad norm should be non-negative"
    
    def test_gradient_norms_change_during_training(self, agent_standard):
        """Test that gradient norms change as training progresses"""
        # Train for a few steps and collect gradient norms
        norms = []
        for _ in range(10):
            agent_standard.train(batch_size=32)
            grad_stats = agent_standard.get_gradient_stats()
            norms.append(grad_stats['critic_grad_norm'])
        
        # Gradient norms should vary (not all identical)
        assert len(set(norms)) > 1, "Gradient norms should change during training"
    
    def test_cnn_gradient_stats_in_dict_buffer(self):
        """Test CNN gradient statistics when using Dict buffer"""
        agent = TD3Agent(
            state_dim={'camera': (4, 84, 84), 'kinematics': 7},
            action_dim=2,
            max_action=1.0,
            use_dict_buffer=True
        )
        
        # Add Dict transitions
        for _ in range(100):
            state = {
                'camera': np.random.randn(4, 84, 84),
                'kinematics': np.random.randn(7)
            }
            action = np.random.randn(2)
            next_state = {
                'camera': np.random.randn(4, 84, 84),
                'kinematics': np.random.randn(7)
            }
            reward = np.random.randn()
            done = False
            agent.replay_buffer.add(state, action, next_state, reward, done)
        
        agent.total_it = agent.learning_starts + 1
        agent.train(batch_size=32)
        
        grad_stats = agent.get_gradient_stats()
        
        # Check CNN gradient keys
        assert 'actor_cnn_grad_norm' in grad_stats
        assert 'critic_cnn_grad_norm' in grad_stats
        
        # Should be float or 0.0
        assert isinstance(grad_stats['actor_cnn_grad_norm'], float)
        assert isinstance(grad_stats['critic_cnn_grad_norm'], float)


class TestComparison:
    """Test comparison with previous (limited) implementation"""
    
    def test_metric_count_improvement(self):
        """Test that we now have significantly more metrics (Bug #16 fix)"""
        agent = TD3Agent(
            state_dim=10,
            action_dim=2,
            max_action=1.0,
            use_dict_buffer=False
        )
        
        stats = agent.get_stats()
        
        # Previous implementation had only 4 metrics
        # New implementation should have 25+ metrics
        assert len(stats) >= 25, f"Should have 25+ metrics, got {len(stats)}"
        
        print(f"\n[IMPROVEMENT] Metrics expanded from 4 to {len(stats)}")
        print(f"  Gap closed: {(len(stats) - 4) / 4 * 100:.0f}% improvement")
    
    def test_learning_rate_visibility(self):
        """Test that learning rates are now visible (Phase 22 finding)"""
        agent = TD3Agent(
            state_dim={'camera': (4, 84, 84), 'kinematics': 7},
            action_dim=2,
            max_action=1.0,
            use_dict_buffer=True
        )
        
        stats = agent.get_stats()
        
        # These are now visible (Phase 22 LR imbalance would be obvious!)
        assert 'actor_lr' in stats
        assert 'critic_lr' in stats
        assert 'actor_cnn_lr' in stats
        assert 'critic_cnn_lr' in stats
        
        print(f"\n[PHASE 22 FIX] Learning rates now visible:")
        print(f"  Actor:      {stats['actor_lr']:.6f}")
        print(f"  Critic:     {stats['critic_lr']:.6f}")
        print(f"  Actor CNN:  {stats['actor_cnn_lr']:.6f}")
        print(f"  Critic CNN: {stats['critic_cnn_lr']:.6f}")
        
        # Phase 22 issue: CNN LR was 0.0001 vs 0.0003
        # This would now be IMMEDIATELY visible!


if __name__ == "__main__":
    print("Testing get_stats() improvements (Bug #16 fix)...")
    print("="*70)
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

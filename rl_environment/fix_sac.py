#!/usr/bin/env python
"""
Fixed SAC implementation that addresses the backward pass issue.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def fix_sac_update_method():
    """
    Create a modified SAC update method to fix the backward pass issue.

    This function returns code that can be used to replace the existing update method.
    """
    return """
    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update the SAC networks.

        Args:
            batch_size: Batch size for update, defaults to self.batch_size

        Returns:
            Dictionary of loss values
        """
        if batch_size is None:
            batch_size = self.batch_size

        if not self.memory.can_sample(batch_size):
            return {'actor_loss': 0, 'critic_loss': 0, 'alpha_loss': 0}

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Extract features - make sure to detach to avoid backward issues
        features = self.feature_extractor(states)
        next_features = self.feature_extractor(next_states).detach()

        # Get current alpha value
        alpha = self.log_alpha.exp().item() if self.auto_entropy else self.alpha

        # Update critic
        with torch.no_grad():
            # Sample actions from policy
            next_actions, next_log_probs, _ = self.policy.sample(next_features)

            # Compute target Q values
            q1_next = self.q1_target(next_features, next_actions)
            q2_next = self.q2_target(next_features, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Compute current Q values
        q1 = self.q1(features, actions)
        q2 = self.q2(features, actions)

        # Compute critic loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss

        # Update critics
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Record critic loss
        self.critic_losses.append(critic_loss.item())

        # Get fresh features for actor update to avoid backward issue
        with torch.no_grad():
            features_actor = self.feature_extractor(states)

        # Update actor
        actions_pi, log_probs, _ = self.policy.sample(features_actor)
        q1_pi = self.q1(features_actor, actions_pi)
        q2_pi = self.q2(features_actor, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss is expectation of Q - entropy
        actor_loss = (alpha * log_probs - q_pi).mean()

        # Update actor
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # Record actor loss
        self.actor_losses.append(actor_loss.item())

        # Update alpha if using automatic entropy tuning
        alpha_loss = 0
        if self.auto_entropy:
            # Get fresh log_probs for alpha update to avoid backward issue
            with torch.no_grad():
                features_alpha = self.feature_extractor(states)

            _, log_probs_alpha, _ = self.policy.sample(features_alpha)
            alpha_loss = -(self.log_alpha * (log_probs_alpha + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Record alpha loss
            self.alpha_losses.append(alpha_loss.item())

        # Soft update target networks
        self._soft_update_targets()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item()
        }
    """

def main():
    """
    Print instructions for fixing the SAC implementation.
    """
    fix_code = fix_sac_update_method()

    print("To fix the backward pass issue in the SAC implementation, replace the update method")
    print("in simple_sac.py with the following code:")
    print()
    print(fix_code)
    print()
    print("This modified update method avoids the backward pass issue by:")
    print("1. Detaching next_features to avoid gradient flow through the target network")
    print("2. Using separate feature extractions for actor and alpha updates")
    print("3. Adding proper detach() calls to prevent double backward")

if __name__ == "__main__":
    main()

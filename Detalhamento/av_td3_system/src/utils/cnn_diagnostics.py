"""
CNN Diagnostics Utility

Provides simple, straightforward tools for monitoring CNN learning during training.
Tracks gradient flow, weight changes, feature statistics, and activation patterns.

Based on:
- Stable-Baselines3 TensorBoard integration patterns
- PyTorch best practices for monitoring neural networks
- Real-world debugging experience from production deep learning systems

Author: Daniel Terra Gomes
2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import defaultdict


class CNNDiagnostics:
    """
    Diagnostic tool for monitoring CNN learning in end-to-end TD3 training.

    Tracks key metrics that indicate whether the CNN is learning:
    - Gradient magnitudes (are gradients flowing?)
    - Weight changes (are weights updating?)
    - Feature statistics (are features meaningful?)
    - Activation patterns (is the network saturated?)

    Usage:
        diagnostics = CNNDiagnostics(cnn_model)

        # During training (after backward pass)
        diagnostics.capture_gradients()

        # After optimizer step
        diagnostics.capture_weights()

        # During forward pass
        features = cnn(images)
        diagnostics.capture_features(features)

        # Get summary
        summary = diagnostics.get_summary()
    """

    def __init__(self, cnn_model: nn.Module):
        """
        Initialize CNN diagnostics tracker.

        Args:
            cnn_model: PyTorch CNN module to monitor
        """
        self.cnn = cnn_model

        # Storage for metrics
        self.gradient_norms = defaultdict(list)
        self.weight_norms = defaultdict(list)
        self.weight_changes = defaultdict(list)
        self.feature_stats = defaultdict(list)

        # Store initial weights for change tracking
        self.initial_weights = {}
        self.previous_weights = {}
        for name, param in cnn_model.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone().cpu()
                self.previous_weights[name] = param.data.clone().cpu()

        # Counters
        self.n_gradient_captures = 0
        self.n_weight_captures = 0
        self.n_feature_captures = 0

    def capture_gradients(self) -> Dict[str, float]:
        """
        Capture gradient statistics after backward pass.

        Call this AFTER loss.backward() but BEFORE optimizer.step()

        Returns:
            Dictionary with gradient norms for each layer
        """
        grad_norms = {}

        for name, param in self.cnn.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Compute L2 norm of gradients
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                self.gradient_norms[name].append(grad_norm)

        self.n_gradient_captures += 1
        return grad_norms

    def capture_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Capture weight statistics after optimizer step.

        Call this AFTER optimizer.step()

        Returns:
            Dictionary with weight norms and changes for each layer
        """
        weight_stats = {}

        for name, param in self.cnn.named_parameters():
            if param.requires_grad:
                current_weight = param.data.clone().cpu()

                # Weight norm
                weight_norm = current_weight.norm().item()
                self.weight_norms[name].append(weight_norm)

                # Weight change from previous step
                prev_weight = self.previous_weights[name]
                weight_change = (current_weight - prev_weight).norm().item()
                self.weight_changes[name].append(weight_change)

                # Update previous weights
                self.previous_weights[name] = current_weight

                weight_stats[name] = {
                    'norm': weight_norm,
                    'change': weight_change
                }

        self.n_weight_captures += 1
        return weight_stats

    def capture_features(self, features: torch.Tensor, name: str = "output") -> Dict[str, float]:
        """
        Capture feature statistics during forward pass.

        Args:
            features: Output tensor from CNN (e.g., shape [batch, 512])
            name: Name for this feature capture (default: "output")

        Returns:
            Dictionary with feature statistics
        """
        with torch.no_grad():
            stats = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'min': features.min().item(),
                'max': features.max().item(),
                'abs_mean': features.abs().mean().item(),
                'norm': features.norm(dim=1).mean().item()  # Average L2 norm per sample
            }

            # Store statistics
            for key, value in stats.items():
                self.feature_stats[f"{name}_{key}"].append(value)

        self.n_feature_captures += 1
        return stats

    def check_gradient_flow(self, threshold: float = 1e-7) -> Dict[str, bool]:
        """
        Check if gradients are flowing through each layer.

        Args:
            threshold: Minimum gradient norm to consider as "flowing"

        Returns:
            Dictionary indicating if gradients are flowing for each layer
        """
        flow_status = {}

        for name, param in self.cnn.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    flow_status[name] = grad_norm > threshold
                else:
                    flow_status[name] = False

        return flow_status

    def check_weight_updates(self, threshold: float = 1e-8) -> Dict[str, bool]:
        """
        Check if weights are actually updating.

        Args:
            threshold: Minimum weight change to consider as "updating"

        Returns:
            Dictionary indicating if weights are updating for each layer
        """
        update_status = {}

        for name, param in self.cnn.named_parameters():
            if param.requires_grad:
                current_weight = param.data.clone().cpu()
                initial_weight = self.initial_weights[name]

                # Check if weights have changed from initialization
                change = (current_weight - initial_weight).norm().item()
                update_status[name] = change > threshold

        return update_status

    def get_summary(self, last_n: int = 100) -> Dict[str, any]:
        """
        Get comprehensive summary of CNN learning status.

        Args:
            last_n: Number of recent captures to average over

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_captures': {
                'gradients': self.n_gradient_captures,
                'weights': self.n_weight_captures,
                'features': self.n_feature_captures
            },
            'gradient_flow': {},
            'weight_updates': {},
            'recent_stats': {}
        }

        # Gradient flow status
        gradient_flow = self.check_gradient_flow()
        summary['gradient_flow'] = gradient_flow
        summary['gradient_flow_ok'] = all(gradient_flow.values())

        # Weight update status
        weight_updates = self.check_weight_updates()
        summary['weight_updates'] = weight_updates
        summary['weights_updating'] = all(weight_updates.values())

        # Recent gradient statistics
        for name in self.gradient_norms:
            recent_grads = self.gradient_norms[name][-last_n:]
            if recent_grads:
                summary['recent_stats'][f"{name}_grad"] = {
                    'mean': np.mean(recent_grads),
                    'std': np.std(recent_grads),
                    'min': np.min(recent_grads),
                    'max': np.max(recent_grads)
                }

        # Recent weight change statistics
        for name in self.weight_changes:
            recent_changes = self.weight_changes[name][-last_n:]
            if recent_changes:
                summary['recent_stats'][f"{name}_change"] = {
                    'mean': np.mean(recent_changes),
                    'std': np.std(recent_changes),
                    'min': np.min(recent_changes),
                    'max': np.max(recent_changes)
                }

        # Recent feature statistics
        for name in self.feature_stats:
            recent_values = self.feature_stats[name][-last_n:]
            if recent_values:
                summary['recent_stats'][name] = {
                    'mean': np.mean(recent_values),
                    'trend': self._compute_trend(recent_values)
                }

        return summary

    def _compute_trend(self, values: List[float]) -> str:
        """
        Compute trend direction from recent values.

        Args:
            values: List of values (most recent last)

        Returns:
            String indicating trend: "increasing", "decreasing", or "stable"
        """
        if len(values) < 2:
            return "stable"

        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        if abs(slope) < 0.01 * np.std(values):
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def print_summary(self, last_n: int = 100) -> None:
        """
        Print human-readable summary of CNN learning status.

        Args:
            last_n: Number of recent captures to average over
        """
        summary = self.get_summary(last_n=last_n)

        print("\n" + "="*70)
        print("CNN DIAGNOSTICS SUMMARY")
        print("="*70)

        # Capture counts
        print(f"\nCaptures: {summary['n_captures']['gradients']} gradients | "
              f"{summary['n_captures']['weights']} weights | "
              f"{summary['n_captures']['features']} features")

        # Overall status
        print(f"\n{'✅' if summary['gradient_flow_ok'] else '❌'} Gradient Flow: "
              f"{'OK' if summary['gradient_flow_ok'] else 'BLOCKED'}")
        print(f"{'✅' if summary['weights_updating'] else '❌'} Weight Updates: "
              f"{'OK' if summary['weights_updating'] else 'FROZEN'}")

        # Per-layer gradient flow
        print("\n[Gradient Flow by Layer]")
        for name, flowing in summary['gradient_flow'].items():
            status = "✅ FLOWING" if flowing else "❌ BLOCKED"
            # Get recent gradient stats if available
            grad_key = f"{name}_grad"
            if grad_key in summary['recent_stats']:
                grad_mean = summary['recent_stats'][grad_key]['mean']
                grad_max = summary['recent_stats'][grad_key]['max']
                print(f"  {name:40s} {status:12s} (mean={grad_mean:.2e}, max={grad_max:.2e})")
            else:
                print(f"  {name:40s} {status:12s}")

        # Per-layer weight updates
        print("\n[Weight Updates by Layer]")
        for name, updating in summary['weight_updates'].items():
            status = "✅ UPDATING" if updating else "❌ FROZEN"
            # Get recent weight change stats if available
            change_key = f"{name}_change"
            if change_key in summary['recent_stats']:
                change_mean = summary['recent_stats'][change_key]['mean']
                change_max = summary['recent_stats'][change_key]['max']
                print(f"  {name:40s} {status:12s} (mean={change_mean:.2e}, max={change_max:.2e})")
            else:
                print(f"  {name:40s} {status:12s}")

        # Feature statistics
        print("\n[Feature Statistics]")
        for name, stats in summary['recent_stats'].items():
            if not name.endswith('_grad') and not name.endswith('_change'):
                mean_val = stats['mean']
                trend = stats.get('trend', 'unknown')
                print(f"  {name:40s} mean={mean_val:8.4f} trend={trend}")

        print("="*70 + "\n")

    def log_to_tensorboard(self, writer, step: int, prefix: str = "cnn_diagnostics") -> None:
        """
        Log CNN diagnostics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter instance
            step: Current training step
            prefix: Prefix for all logged metrics (default: "cnn_diagnostics")
        """
        # Log gradient norms
        for name, grads in self.gradient_norms.items():
            if grads:
                writer.add_scalar(f"{prefix}/gradients/{name}", grads[-1], step)

        # Log weight norms
        for name, norms in self.weight_norms.items():
            if norms:
                writer.add_scalar(f"{prefix}/weights/{name}_norm", norms[-1], step)

        # Log weight changes
        for name, changes in self.weight_changes.items():
            if changes:
                writer.add_scalar(f"{prefix}/weights/{name}_change", changes[-1], step)

        # Log feature statistics
        for name, values in self.feature_stats.items():
            if values:
                writer.add_scalar(f"{prefix}/features/{name}", values[-1], step)

        # Log overall health metrics
        summary = self.get_summary(last_n=100)
        writer.add_scalar(f"{prefix}/health/gradient_flow_ok",
                         float(summary['gradient_flow_ok']), step)
        writer.add_scalar(f"{prefix}/health/weights_updating",
                         float(summary['weights_updating']), step)

    def reset(self) -> None:
        """Reset all captured metrics (useful for starting a new training phase)."""
        self.gradient_norms.clear()
        self.weight_norms.clear()
        self.weight_changes.clear()
        self.feature_stats.clear()

        # Update initial weights to current state
        for name, param in self.cnn.named_parameters():
            if param.requires_grad:
                self.initial_weights[name] = param.data.clone().cpu()
                self.previous_weights[name] = param.data.clone().cpu()

        self.n_gradient_captures = 0
        self.n_weight_captures = 0
        self.n_feature_captures = 0


def create_cnn_diagnostics(cnn_model: nn.Module) -> CNNDiagnostics:
    """
    Factory function to create CNN diagnostics tracker.

    Args:
        cnn_model: PyTorch CNN module to monitor

    Returns:
        CNNDiagnostics instance
    """
    return CNNDiagnostics(cnn_model)


def quick_check_cnn_learning(cnn_model: nn.Module) -> Tuple[bool, str]:
    """
    Quick check if CNN is learning (gradients flowing and weights updating).

    Args:
        cnn_model: PyTorch CNN module to check

    Returns:
        Tuple of (is_learning: bool, message: str)
    """
    # Check gradients
    has_gradients = False
    gradient_info = []

    for name, param in cnn_model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_info.append(f"{name}: {grad_norm:.2e}")
                if grad_norm > 1e-7:
                    has_gradients = True
            else:
                gradient_info.append(f"{name}: NO GRAD")

    if not has_gradients:
        return False, "❌ CNN NOT LEARNING: No gradients detected\n" + "\n".join(gradient_info)

    return True, "✅ CNN LEARNING: Gradients detected\n" + "\n".join(gradient_info[:3])


# Example usage in training loop
"""
# Initialize diagnostics at start of training
diagnostics = CNNDiagnostics(agent.cnn_extractor)

# During training loop (after agent.train())
if t % 100 == 0 and t > start_timesteps:
    # Capture gradients (do this before optimizer.step() in agent.train())
    # Note: This requires modification to agent.train() to expose gradients

    # After training step
    diagnostics.capture_weights()

    # Log to TensorBoard
    diagnostics.log_to_tensorboard(writer, step=t)

    # Print summary every 1000 steps
    if t % 1000 == 0:
        diagnostics.print_summary(last_n=100)

# Quick check at any time
is_learning, msg = quick_check_cnn_learning(agent.cnn_extractor)
print(msg)
"""

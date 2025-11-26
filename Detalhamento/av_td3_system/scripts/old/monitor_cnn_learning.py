"""
CNN Learning Monitor Script

Simple utility script to monitor CNN learning during TD3 training.
Can be run standalone or imported as a module to add monitoring to training loops.

Usage:
    # As standalone script
    python scripts/monitor_cnn_learning.py --checkpoint checkpoints/td3_10k.pth

    # In training script
    from scripts.monitor_cnn_learning import setup_cnn_monitoring, log_cnn_diagnostics

    # Setup at training start
    setup_cnn_monitoring(agent, writer)

    # Log during training loop
    if t % 100 == 0:
        log_cnn_diagnostics(agent, writer, step=t)

Author: Daniel Terra
Date: 2025-01-XX
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_cnn_monitoring(agent, writer: Optional[SummaryWriter] = None) -> bool:
    """
    Setup CNN diagnostics monitoring for the agent.

    Args:
        agent: TD3Agent instance with CNN extractor
        writer: TensorBoard SummaryWriter (optional)

    Returns:
        True if diagnostics enabled successfully, False otherwise
    """
    if not hasattr(agent, 'cnn_extractor') or agent.cnn_extractor is None:
        print("[CNN MONITOR] ❌ No CNN extractor found in agent")
        return False

    # Enable diagnostics
    agent.enable_diagnostics()

    if agent.cnn_diagnostics is not None:
        print("[CNN MONITOR] ✅ CNN diagnostics enabled")
        if writer is not None:
            print("[CNN MONITOR] ✅ TensorBoard logging enabled")
        return True
    else:
        print("[CNN MONITOR] ❌ Failed to enable diagnostics")
        return False


def log_cnn_diagnostics(
    agent,
    writer: Optional[SummaryWriter] = None,
    step: int = 0,
    print_summary: bool = False,
    print_interval: int = 1000
) -> None:
    """
    Log CNN diagnostics to TensorBoard and optionally print summary.

    Args:
        agent: TD3Agent instance with diagnostics enabled
        writer: TensorBoard SummaryWriter (optional)
        step: Current training step
        print_summary: If True, print diagnostics summary
        print_interval: Print summary every N steps (default: 1000)
    """
    if agent.cnn_diagnostics is None:
        return

    # Log to TensorBoard
    if writer is not None:
        agent.cnn_diagnostics.log_to_tensorboard(writer, step=step)

    # Print summary if requested
    if print_summary and step % print_interval == 0:
        agent.print_diagnostics(last_n=100)


def quick_check_cnn_learning(agent) -> None:
    """
    Quick check if CNN is learning (gradients flowing and weights updating).

    Args:
        agent: TD3Agent instance with CNN extractor
    """
    if not hasattr(agent, 'cnn_extractor') or agent.cnn_extractor is None:
        print("[CNN CHECK] ❌ No CNN extractor found")
        return

    # Import quick check function
    try:
        from src.utils.cnn_diagnostics import quick_check_cnn_learning
        is_learning, msg = quick_check_cnn_learning(agent.cnn_extractor)
        print(f"[CNN CHECK] {msg}")
    except ImportError:
        print("[CNN CHECK] ❌ Could not import cnn_diagnostics")


def analyze_checkpoint(checkpoint_path: str) -> None:
    """
    Analyze a saved checkpoint to inspect CNN state.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    print(f"\n{'='*70}")
    print(f"CNN CHECKPOINT ANALYSIS")
    print(f"{'='*70}\n")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check if CNN state is present
    if 'cnn_state_dict' not in checkpoint:
        print("❌ No CNN state found in checkpoint")
        print("   This checkpoint was saved without end-to-end CNN training")
        return

    print("✅ CNN state found in checkpoint")

    # Analyze CNN state
    cnn_state = checkpoint['cnn_state_dict']
    print(f"\n[CNN Layers]")

    total_params = 0
    for name, param in cnn_state.items():
        param_count = param.numel()
        total_params += param_count

        # Get parameter statistics
        param_mean = param.mean().item()
        param_std = param.std().item()
        param_min = param.min().item()
        param_max = param.max().item()
        param_norm = param.norm().item()

        print(f"  {name:40s} shape={str(tuple(param.shape)):20s} "
              f"params={param_count:>8,} | "
              f"norm={param_norm:8.4f} | "
              f"mean={param_mean:+.4f} | "
              f"std={param_std:.4f}")

    print(f"\n[Total CNN Parameters] {total_params:,}")

    # Check optimizer state if available
    if 'cnn_optimizer_state_dict' in checkpoint:
        print("\n✅ CNN optimizer state found")
        print("   This indicates CNN was actively being trained")
    else:
        print("\n⚠️  No CNN optimizer state found")

    # Check training iteration
    if 'total_it' in checkpoint:
        print(f"\n[Training Iterations] {checkpoint['total_it']:,}")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Monitor CNN learning in TD3 agent"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file to analyze'
    )
    parser.add_argument(
        '--quick-check',
        action='store_true',
        help='Perform quick learning check (requires loaded agent)'
    )

    args = parser.parse_args()

    if args.checkpoint:
        analyze_checkpoint(args.checkpoint)
    elif args.quick_check:
        print("Quick check requires an active training session")
        print("Use this in your training script instead:")
        print("  from scripts.monitor_cnn_learning import quick_check_cnn_learning")
        print("  quick_check_cnn_learning(agent)")
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Analyze checkpoint")
        print("  python scripts/monitor_cnn_learning.py --checkpoint checkpoints/td3_10k.pth")
        print("\n  # In training script:")
        print("  from scripts.monitor_cnn_learning import setup_cnn_monitoring, log_cnn_diagnostics")
        print("  setup_cnn_monitoring(agent, writer)")
        print("  if t % 100 == 0:")
        print("      log_cnn_diagnostics(agent, writer, step=t, print_summary=(t % 1000 == 0))")


if __name__ == "__main__":
    main()

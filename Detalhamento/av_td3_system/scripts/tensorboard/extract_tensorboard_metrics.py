#!/usr/bin/env python3
"""
TensorBoard Metrics Extraction and Analysis Tool

Extracts metrics from TensorBoard event files and performs systematic analysis
to validate TD3 training behavior at early stages (5k steps) before scaling to 1M steps.

Usage:
    python extract_tensorboard_metrics.py <event_file> [--output REPORT.md]

References:
    - TD3 Paper (Fujimoto et al., 2018): https://arxiv.org/abs/1802.09477
    - OpenAI Spinning Up TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    - Stable-Baselines3 TD3: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: TensorBoard not installed. Install with: pip install tensorboard")
    sys.exit(1)


class TensorBoardMetricsExtractor:
    """Extract and analyze metrics from TensorBoard event files."""
    
    def __init__(self, event_file: str):
        """Initialize the extractor with an event file path."""
        self.event_file = Path(event_file)
        if not self.event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file}")
        
        # Load the event file
        print(f"Loading TensorBoard event file: {self.event_file.name}")
        self.ea = event_accumulator.EventAccumulator(str(self.event_file))
        self.ea.Reload()
        
        # Store available tags
        self.scalar_tags = self.ea.Tags().get('scalars', [])
        print(f"Found {len(self.scalar_tags)} scalar metrics")
    
    def get_scalar_data(self, tag: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract scalar data for a given tag.
        
        Returns:
            Tuple of (steps, values) as numpy arrays
        """
        if tag not in self.scalar_tags:
            return np.array([]), np.array([])
        
        events = self.ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values
    
    def compute_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Compute statistical summary of values."""
        if len(values) == 0:
            return {
                'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'q25': 0, 'median': 0, 'q75': 0
            }
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'median': np.median(values),
            'q75': np.percentile(values, 75),
        }
    
    def analyze_training_phase(self) -> Dict[str, any]:
        """
        Analyze which training phase the run is in based on timesteps.
        
        Returns:
            Dictionary with phase information
        """
        # Get timestep data (use any available metric to determine max step)
        max_step = 0
        for tag in self.scalar_tags:
            steps, _ = self.get_scalar_data(tag)
            if len(steps) > 0:
                max_step = max(max_step, steps[-1])
        
        # Determine phase based on configuration
        # From config: learning_starts=1000, train_freq=50
        learning_starts = 1000
        
        if max_step < learning_starts:
            phase = "exploration"
            phase_desc = f"Random exploration (steps 0-{learning_starts})"
        else:
            phase = "learning"
            phase_desc = f"TD3 learning (steps {learning_starts}+)"
        
        return {
            'max_step': int(max_step),
            'phase': phase,
            'description': phase_desc,
            'learning_starts': learning_starts,
            'expected_updates': int((max_step - learning_starts) / 50) if max_step > learning_starts else 0
        }
    
    def analyze_q_values(self) -> Dict[str, any]:
        """Analyze Q-value metrics to detect overestimation."""
        analysis = {}
        
        # Q1 and Q2 values from twin critics
        for q_idx in [1, 2]:
            tag = f'train/q{q_idx}_value'
            steps, values = self.get_scalar_data(tag)
            
            if len(values) > 0:
                stats = self.compute_statistics(values)
                analysis[f'q{q_idx}'] = {
                    'stats': stats,
                    'steps': steps,
                    'values': values
                }
        
        # Actor Q-values (Q(s, μ(s)))
        tag = 'debug/actor_q_mean'
        steps, values = self.get_scalar_data(tag)
        if len(values) > 0:
            analysis['actor_q'] = {
                'stats': self.compute_statistics(values),
                'steps': steps,
                'values': values
            }
        
        # Twin critic difference |Q1 - Q2|
        tag = 'debug/q1_q2_diff'
        steps, values = self.get_scalar_data(tag)
        if len(values) > 0:
            analysis['q1_q2_diff'] = {
                'stats': self.compute_statistics(values),
                'steps': steps,
                'values': values
            }
        
        return analysis
    
    def analyze_losses(self) -> Dict[str, any]:
        """Analyze actor and critic losses."""
        analysis = {}
        
        # Critic loss
        tag = 'train/critic_loss'
        steps, values = self.get_scalar_data(tag)
        if len(values) > 0:
            analysis['critic'] = {
                'stats': self.compute_statistics(values),
                'steps': steps,
                'values': values
            }
        
        # Actor loss
        tag = 'train/actor_loss'
        steps, values = self.get_scalar_data(tag)
        if len(values) > 0:
            analysis['actor'] = {
                'stats': self.compute_statistics(values),
                'steps': steps,
                'values': values
            }
        
        return analysis
    
    def analyze_rewards(self) -> Dict[str, any]:
        """Analyze reward metrics."""
        analysis = {}
        
        # Episode reward
        tag = 'rollout/ep_reward'
        steps, values = self.get_scalar_data(tag)
        if len(values) > 0:
            analysis['episode'] = {
                'stats': self.compute_statistics(values),
                'steps': steps,
                'values': values
            }
        
        # Step reward (debug)
        tag = 'debug/reward_mean'
        steps, values = self.get_scalar_data(tag)
        if len(values) > 0:
            analysis['step_mean'] = {
                'stats': self.compute_statistics(values),
                'steps': steps,
                'values': values
            }
        
        return analysis
    
    def analyze_gradients(self) -> Dict[str, any]:
        """Analyze gradient norms to detect instability."""
        analysis = {}
        
        gradient_tags = [
            'debug/actor_grad_norm',
            'debug/critic_grad_norm',
            'debug/actor_cnn_grad_norm',
            'debug/critic_cnn_grad_norm'
        ]
        
        for tag in gradient_tags:
            steps, values = self.get_scalar_data(tag)
            if len(values) > 0:
                network_name = tag.split('/')[-1].replace('_grad_norm', '')
                analysis[network_name] = {
                    'stats': self.compute_statistics(values),
                    'steps': steps,
                    'values': values
                }
        
        return analysis
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive analysis report."""
        
        # Analyze all components
        phase_info = self.analyze_training_phase()
        q_analysis = self.analyze_q_values()
        loss_analysis = self.analyze_losses()
        reward_analysis = self.analyze_rewards()
        grad_analysis = self.analyze_gradients()
        
        # Generate markdown report
        report_lines = [
            "# TensorBoard Metrics Analysis Report",
            "",
            f"**Event File**: `{self.event_file.name}`",
            f"**Analysis Date**: {Path(__file__).stat().st_mtime}",
            f"**Total Metrics**: {len(self.scalar_tags)}",
            "",
            "---",
            "",
            "## 1. Training Phase Analysis",
            "",
            f"- **Maximum Step**: {phase_info['max_step']:,}",
            f"- **Current Phase**: {phase_info['phase'].upper()}",
            f"- **Description**: {phase_info['description']}",
            f"- **Expected Updates**: {phase_info['expected_updates']} (train_freq=50)",
            "",
        ]
        
        # Expected behavior at 5k steps
        report_lines.extend([
            "### Expected Behavior at 5k Steps",
            "",
            "Based on TD3 paper and OpenAI Spinning Up:",
            "",
            "1. **Steps 0-1,000**: Random exploration",
            "   - No policy training",
            "   - Replay buffer filling",
            "   - No Q-values/losses logged",
            "",
            "2. **Steps 1,001-5,000**: Early learning",
            "   - ~80 training updates (every 50 steps)",
            "   - Q-values: Should be LOW and NOISY (lots of variance)",
            "   - Critic loss: Should be HIGH (learning from scratch)",
            "   - Actor loss: Should be NEGATIVE (policy gradient)",
            "   - Gradients: Should be MODERATE (not exploding)",
            "",
            "---",
            "",
        ])
        
        # Q-value analysis
        report_lines.extend([
            "## 2. Q-Value Analysis (Overestimation Detection)",
            "",
        ])
        
        if 'q1' in q_analysis or 'q2' in q_analysis:
            report_lines.append("### Twin Critic Q-Values")
            report_lines.append("")
            
            for q_idx in [1, 2]:
                if f'q{q_idx}' in q_analysis:
                    stats = q_analysis[f'q{q_idx}']['stats']
                    report_lines.extend([
                        f"**Q{q_idx} (from replay buffer samples)**:",
                        f"- Count: {stats['count']}",
                        f"- Mean: {stats['mean']:.2f}",
                        f"- Std Dev: {stats['std']:.2f}",
                        f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]",
                        f"- Median: {stats['median']:.2f}",
                        "",
                    ])
        
        if 'actor_q' in q_analysis:
            stats = q_analysis['actor_q']['stats']
            report_lines.extend([
                "### Actor Q-Values (Overestimation Check)",
                "",
                "**Q(s, μ(s)) - Q-value of current policy actions**:",
                f"- Count: {stats['count']}",
                f"- Mean: {stats['mean']:.2e}",
                f"- Std Dev: {stats['std']:.2e}",
                f"- Range: [{stats['min']:.2e}, {stats['max']:.2e}]",
                f"- Median: {stats['median']:.2e}",
                "",
            ])
            
            # Overestimation detection
            if stats['mean'] > 1e6:
                report_lines.extend([
                    "⚠️ **CRITICAL: Q-Value Explosion Detected**",
                    "",
                    f"Actor Q-values are {stats['mean']:.2e}, which is abnormally high!",
                    "",
                    "**Expected at 5k steps**: Q-values should be < 1,000",
                    "**Observed**: Q-values > 1,000,000 (1000× higher than expected)",
                    "",
                    "This indicates severe overestimation bias NOT being controlled by TD3's twin critics.",
                    "",
                ])
            elif stats['mean'] > 1000:
                report_lines.extend([
                    "⚠️ **WARNING: High Q-Values**",
                    "",
                    f"Actor Q-values are {stats['mean']:.2f}, which is higher than expected.",
                    "",
                    "**Expected at 5k steps**: Q-values should be < 500",
                    "**Observed**: Q-values > 1,000",
                    "",
                ])
            else:
                report_lines.extend([
                    "✅ **Q-Values Within Expected Range**",
                    "",
                    f"Actor Q-values are {stats['mean']:.2f}, which is reasonable for early training.",
                    "",
                ])
        
        if 'q1_q2_diff' in q_analysis:
            stats = q_analysis['q1_q2_diff']['stats']
            report_lines.extend([
                "### Twin Critic Divergence |Q1 - Q2|",
                "",
                f"- Mean: {stats['mean']:.2f}",
                f"- Max: {stats['max']:.2f}",
                "",
                "**Expected**: Low divergence (<10% of Q-value magnitude) indicates twin critics are converging.",
                "",
            ])
        
        report_lines.append("---")
        report_lines.append("")
        
        # Loss analysis
        report_lines.extend([
            "## 3. Loss Analysis",
            "",
        ])
        
        if 'critic' in loss_analysis:
            stats = loss_analysis['critic']['stats']
            report_lines.extend([
                "### Critic Loss (MSE)",
                "",
                f"- Count: {stats['count']}",
                f"- Mean: {stats['mean']:.2f}",
                f"- Std Dev: {stats['std']:.2f}",
                f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]",
                "",
                "**Expected at 5k steps**: High and volatile (network learning from scratch).",
                "",
            ])
        
        if 'actor' in loss_analysis:
            stats = loss_analysis['actor']['stats']
            report_lines.extend([
                "### Actor Loss (Policy Gradient)",
                "",
                f"- Count: {stats['count']}",
                f"- Mean: {stats['mean']:.2e}",
                f"- Std Dev: {stats['std']:.2e}",
                f"- Range: [{stats['min']:.2e}, {stats['max']:.2e}]",
                "",
                "**Expected**: Negative values (maximizing Q). Should be stable, not exploding.",
                "",
            ])
            
            if abs(stats['mean']) > 1e6:
                report_lines.extend([
                    "⚠️ **CRITICAL: Actor Loss Explosion**",
                    "",
                    f"Actor loss magnitude is {abs(stats['mean']):.2e}, indicating divergence!",
                    "",
                ])
        
        report_lines.append("---")
        report_lines.append("")
        
        # Gradient analysis
        report_lines.extend([
            "## 4. Gradient Norm Analysis",
            "",
        ])
        
        for network_name, data in grad_analysis.items():
            stats = data['stats']
            report_lines.extend([
                f"### {network_name.replace('_', ' ').title()}",
                "",
                f"- Mean: {stats['mean']:.2e}",
                f"- Max: {stats['max']:.2e}",
                f"- Std Dev: {stats['std']:.2e}",
                "",
            ])
            
            # Detect gradient explosion
            if stats['max'] > 1e6:
                report_lines.extend([
                    f"⚠️ **CRITICAL: Gradient Explosion in {network_name}**",
                    "",
                    f"Max gradient norm: {stats['max']:.2e}",
                    "**Fix**: Gradient clipping is MANDATORY (see config: gradient_clipping.enabled)",
                    "",
                ])
            elif stats['max'] > 100:
                report_lines.extend([
                    f"⚠️ **WARNING: High Gradients in {network_name}**",
                    "",
                    f"Max gradient norm: {stats['max']:.2e}",
                    "**Recommendation**: Consider gradient clipping.",
                    "",
                ])
        
        report_lines.append("---")
        report_lines.append("")
        
        # Reward analysis
        report_lines.extend([
            "## 5. Reward Analysis",
            "",
        ])
        
        if 'episode' in reward_analysis:
            stats = reward_analysis['episode']['stats']
            report_lines.extend([
                "### Episode Reward",
                "",
                f"- Count: {stats['count']} episodes",
                f"- Mean: {stats['mean']:.2f}",
                f"- Std Dev: {stats['std']:.2f}",
                f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]",
                "",
            ])
        
        if 'step_mean' in reward_analysis:
            stats = reward_analysis['step_mean']['stats']
            report_lines.extend([
                "### Step Reward (Mean per Update)",
                "",
                f"- Mean: {stats['mean']:.2f}",
                f"- Std Dev: {stats['std']:.2f}",
                "",
            ])
        
        report_lines.extend([
            "---",
            "",
            "## 6. Readiness Assessment for 1M Step Training",
            "",
        ])
        
        # Readiness checklist
        issues = []
        warnings = []
        
        # Check Q-values
        if 'actor_q' in q_analysis:
            if q_analysis['actor_q']['stats']['mean'] > 1e6:
                issues.append("Q-value explosion (>1M)")
            elif q_analysis['actor_q']['stats']['mean'] > 1000:
                warnings.append("Q-values higher than expected (>1K)")
        
        # Check gradients
        for network_name, data in grad_analysis.items():
            if data['stats']['max'] > 1e6:
                issues.append(f"Gradient explosion in {network_name}")
            elif data['stats']['max'] > 100:
                warnings.append(f"High gradients in {network_name}")
        
        # Check actor loss
        if 'actor' in loss_analysis:
            if abs(loss_analysis['actor']['stats']['mean']) > 1e6:
                issues.append("Actor loss divergence")
        
        # Generate assessment
        if len(issues) > 0:
            report_lines.extend([
                "### ❌ NOT READY FOR 1M TRAINING",
                "",
                "**Critical Issues**:",
                "",
            ])
            for issue in issues:
                report_lines.append(f"- {issue}")
            report_lines.append("")
        elif len(warnings) > 0:
            report_lines.extend([
                "### ⚠️ PROCEED WITH CAUTION",
                "",
                "**Warnings**:",
                "",
            ])
            for warning in warnings:
                report_lines.append(f"- {warning}")
            report_lines.append("")
            report_lines.extend([
                "**Recommendation**: Fix warnings before scaling to 1M steps.",
                "",
            ])
        else:
            report_lines.extend([
                "### ✅ READY FOR 1M TRAINING",
                "",
                "All metrics are within expected ranges for early training (5k steps).",
                "",
            ])
        
        # Available metrics
        report_lines.extend([
            "---",
            "",
            "## 7. Available Metrics",
            "",
            "```",
        ])
        report_lines.extend(sorted(self.scalar_tags))
        report_lines.extend([
            "```",
            "",
        ])
        
        # Join report
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            print(f"\n✅ Report saved to: {output_path}")
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze TensorBoard metrics for TD3 training validation"
    )
    parser.add_argument(
        'event_file',
        type=str,
        help='Path to TensorBoard event file (events.out.tfevents.*)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output markdown file path (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create extractor and generate report
        extractor = TensorBoardMetricsExtractor(args.event_file)
        report = extractor.generate_report(args.output)
        
        # Print to stdout if no output file specified
        if not args.output:
            print("\n" + "="*80)
            print(report)
            print("="*80)
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

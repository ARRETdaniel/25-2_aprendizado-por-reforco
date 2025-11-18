#!/usr/bin/env python3
"""
Systematic Analysis of 5K Validation Run (November 18, 2025)

Analyzes the TensorBoard event file and text log to assess:
1. Gradient explosion status (should be FIXED after all patches)
2. Episode length improvements
3. Learning stability
4. Q-value behavior
5. Reward balance

Validates against issues in:
- COMPREHENSIVE_LOG_ANALYSIS_5K_POST_FIXES.md
- FINAL_VERDICT_Q_VALUE_EXPLOSION.md
- FIXES_APPLIED_SUMMARY.md

Author: Daniel Terra
Date: 2025-11-18
"""
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tensorboard.backend.event_processing import event_accumulator


class Nov18_5K_Analyzer:
    """Analyzer for November 18, 2025 5K validation run."""

    def __init__(self, event_file: str, log_file: str):
        """
        Initialize analyzer.

        Args:
            event_file: Path to TensorBoard event file
            log_file: Path to text log file
        """
        self.event_file = event_file
        self.log_file = log_file
        self.metrics = defaultdict(list)

        print("="*80)
        print("5K VALIDATION RUN ANALYSIS - November 18, 2025")
        print("="*80)
        print(f"Event file: {event_file}")
        print(f"Log file: {log_file}")
        print("="*80)

    def load_tensorboard_events(self):
        """Load TensorBoard event file."""
        print("\n📊 Loading TensorBoard events...")

        ea = event_accumulator.EventAccumulator(
            self.event_file,
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 0,
            }
        )
        ea.Reload()

        tags = ea.Tags()
        print(f"✅ Loaded {len(tags['scalars'])} scalar metrics")

        # Extract all scalar metrics
        for tag in tags['scalars']:
            events = ea.Scalars(tag)
            self.metrics[tag] = [(e.step, e.value) for e in events]

        return self.metrics

    def analyze_gradient_norms(self) -> Dict:
        """Analyze gradient norms to check for explosion."""
        print("\n" + "="*80)
        print("GRADIENT NORM ANALYSIS")
        print("="*80)

        gradient_tags = {
            'actor_cnn': 'train/actor_cnn_grad_norm',
            'critic_cnn': 'train/critic_cnn_grad_norm',
            'actor_mlp': 'train/actor_mlp_grad_norm',
            'critic_mlp': 'train/critic_mlp_grad_norm',
        }

        results = {}

        for name, tag in gradient_tags.items():
            if tag in self.metrics:
                values = [v for _, v in self.metrics[tag]]
                if values:
                    results[name] = {
                        'mean': np.mean(values),
                        'max': np.max(values),
                        'min': np.min(values),
                        'std': np.std(values),
                        'final': values[-1] if values else None,
                    }

                    print(f"\n{name.upper()}:")
                    print(f"  Mean: {results[name]['mean']:.2f}")
                    print(f"  Max:  {results[name]['max']:.2f}")
                    print(f"  Min:  {results[name]['min']:.2f}")
                    print(f"  Std:  {results[name]['std']:.2f}")
                    print(f"  Final: {results[name]['final']:.2f}")

                    # Check for explosion (based on FINAL_VERDICT_Q_VALUE_EXPLOSION.md)
                    if name == 'actor_cnn':
                        if results[name]['max'] > 100_000:
                            print(f"  ❌ EXPLOSION DETECTED! Max gradient = {results[name]['max']:.0f}")
                        elif results[name]['max'] > 10_000:
                            print(f"  ⚠️ High gradients (max = {results[name]['max']:.0f})")
                        else:
                            print(f"  ✅ Gradients healthy (max = {results[name]['max']:.0f})")

        return results

    def analyze_episode_lengths(self) -> Dict:
        """Analyze episode lengths from TensorBoard."""
        print("\n" + "="*80)
        print("EPISODE LENGTH ANALYSIS")
        print("="*80)

        if 'rollout/ep_len_mean' in self.metrics:
            ep_lengths = self.metrics['rollout/ep_len_mean']
            values = [v for _, v in ep_lengths]

            if values:
                result = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'final': values[-1],
                    'trend': 'improving' if values[-1] > values[0] else 'declining',
                }

                print(f"\nEpisode Length Statistics:")
                print(f"  Mean: {result['mean']:.1f} steps")
                print(f"  Max:  {result['max']:.1f} steps")
                print(f"  Min:  {result['min']:.1f} steps")
                print(f"  Final: {result['final']:.1f} steps")
                print(f"  Trend: {result['trend']}")

                # Literature expectations (from COMPREHENSIVE_LOG_ANALYSIS_5K_POST_FIXES.md)
                print(f"\nLiterature Expectations at 5K steps:")
                print(f"  Expected: 5-20 steps (pipeline validation)")
                if result['final'] >= 5 and result['final'] <= 20:
                    print(f"  ✅ EXPECTED (final = {result['final']:.1f})")
                elif result['final'] < 5:
                    print(f"  ⚠️ Below expected (final = {result['final']:.1f})")
                else:
                    print(f"  ✅ Above expected! (final = {result['final']:.1f})")

                return result

        return {}

    def analyze_actor_loss(self) -> Dict:
        """Analyze actor loss for Q-value explosion."""
        print("\n" + "="*80)
        print("ACTOR LOSS ANALYSIS (Q-Value Explosion Check)")
        print("="*80)

        if 'train/actor_loss' in self.metrics:
            actor_losses = self.metrics['train/actor_loss']
            values = [v for _, v in actor_losses]

            if values:
                result = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'final': values[-1],
                }

                print(f"\nActor Loss Statistics:")
                print(f"  Mean: {result['mean']:.2e}")
                print(f"  Max:  {result['max']:.2e}")
                print(f"  Min:  {result['min']:.2e}")
                print(f"  Final: {result['final']:.2e}")

                # Check for explosion (from FINAL_VERDICT_Q_VALUE_EXPLOSION.md)
                if abs(result['min']) > 1_000_000_000:  # -1B or worse
                    print(f"  ❌ EXTREME Q-VALUE EXPLOSION! Min = {result['min']:.2e}")
                elif abs(result['min']) > 1_000_000:  # -1M or worse
                    print(f"  ❌ Q-VALUE EXPLOSION! Min = {result['min']:.2e}")
                elif abs(result['min']) > 100_000:
                    print(f"  ⚠️ High negative values (min = {result['min']:.2e})")
                else:
                    print(f"  ✅ Actor loss healthy (min = {result['min']:.2e})")

                return result

        return {}

    def analyze_log_file(self) -> Dict:
        """Analyze text log file for episode statistics."""
        print("\n" + "="*80)
        print("LOG FILE ANALYSIS")
        print("="*80)

        with open(self.log_file, 'r') as f:
            log_content = f.read()

        # Extract episode information
        import re
        
        # Pattern: "Episode  XXX | Ep Step  YYY"
        episode_pattern = r'Episode\s+(\d+)\s+\|\s+Ep Step\s+(\d+)'
        episodes = re.findall(episode_pattern, log_content)

        if episodes:
            episode_lengths = [int(length) for _, length in episodes]
            
            result = {
                'total_episodes': len(episodes),
                'mean_length': np.mean(episode_lengths),
                'max_length': np.max(episode_lengths),
                'min_length': np.min(episode_lengths),
                'episodes_under_10': sum(1 for l in episode_lengths if l < 10),
                'episodes_under_20': sum(1 for l in episode_lengths if l < 20),
            }

            print(f"\nEpisode Statistics from Log:")
            print(f"  Total episodes: {result['total_episodes']}")
            print(f"  Mean length: {result['mean_length']:.1f} steps")
            print(f"  Max length: {result['max_length']} steps")
            print(f"  Min length: {result['min_length']} steps")
            print(f"  Episodes <10 steps: {result['episodes_under_10']} ({100*result['episodes_under_10']/result['total_episodes']:.1f}%)")
            print(f"  Episodes <20 steps: {result['episodes_under_20']} ({100*result['episodes_under_20']/result['total_episodes']:.1f}%)")

            return result

        return {}

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        
        # Load data
        self.load_tensorboard_events()

        # Run analyses
        gradient_results = self.analyze_gradient_norms()
        episode_results = self.analyze_episode_lengths()
        actor_loss_results = self.analyze_actor_loss()
        log_results = self.analyze_log_file()

        # Generate summary
        print("\n" + "="*80)
        print("SUMMARY AND GO/NO-GO ASSESSMENT")
        print("="*80)

        issues = []
        passes = []

        # Check gradients
        if gradient_results.get('actor_cnn', {}).get('max', 0) > 100_000:
            issues.append("❌ Actor CNN gradient explosion detected")
        else:
            passes.append("✅ Actor CNN gradients healthy")

        # Check actor loss
        if abs(actor_loss_results.get('min', 0)) > 1_000_000:
            issues.append("❌ Actor loss shows Q-value explosion")
        else:
            passes.append("✅ Actor loss within expected range")

        # Check episode lengths
        final_ep_len = episode_results.get('final', 0)
        if final_ep_len >= 5 and final_ep_len <= 20:
            passes.append(f"✅ Episode length expected for 5K steps ({final_ep_len:.1f})")
        elif final_ep_len < 5:
            issues.append(f"⚠️ Episode length below expected ({final_ep_len:.1f})")
        else:
            passes.append(f"✅ Episode length above expected ({final_ep_len:.1f})")

        print("\nPASSES:")
        for p in passes:
            print(f"  {p}")

        print("\nISSUES:")
        if issues:
            for i in issues:
                print(f"  {i}")
        else:
            print("  None detected!")

        # Final decision
        print("\n" + "="*80)
        if not issues:
            print("✅ GO FOR 50K VALIDATION RUN")
            print("All critical issues resolved. System ready for extended training.")
        else:
            print("❌ NO-GO - Issues detected")
            print("Review and fix issues before proceeding to 50K.")
        print("="*80)

        return {
            'gradient_results': gradient_results,
            'episode_results': episode_results,
            'actor_loss_results': actor_loss_results,
            'log_results': log_results,
            'issues': issues,
            'passes': passes,
        }


def main():
    """Main entry point."""
    event_file = "/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs/TD3_scenario_0_npcs_20_20251118-110409/events.out.tfevents.1763463849.danielterra.1.0"
    log_file = "/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/validation_5k_post_all_fixes_2_20251118_080401.log"

    analyzer = Nov18_5K_Analyzer(event_file, log_file)
    results = analyzer.generate_report()

    # Save results to JSON
    import json
    output_path = Path(project_root) / "docs" / "day-18" / "5K_VALIDATION_ANALYSIS_NOV18.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)

    print(f"\n📄 Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

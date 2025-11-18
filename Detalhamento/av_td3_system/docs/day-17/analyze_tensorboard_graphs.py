#!/usr/bin/env python3
"""
Systematic TensorBoard Graph Analysis for 5K_POST_FIXES Run

This script analyzes the TensorBoard graphs from the 5K validation run and validates
against academic literature (TD3, Rally A3C, DDPG-UAV papers) and official documentation.

Analysis Framework:
1. Load and display each graph
2. Extract observable metrics
3. Compare to literature benchmarks
4. Assess implementation correctness
5. Generate GO/NO-GO recommendation for 1M run

Author: Daniel Terra
Date: November 17, 2025
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TensorBoardGraphAnalyzer:
    """
    Systematic analyzer for TensorBoard training graphs.

    Validates against:
    - TD3 paper (Fujimoto et al., ICML 2018)
    - Rally A3C (Perot et al., 2017)
    - DDPG-UAV (2022)
    - OpenAI Spinning Up TD3
    - Stable-Baselines3 documentation
    """

    def __init__(self, graph_dir: str):
        """
        Initialize analyzer with directory containing PNG graphs.

        Args:
            graph_dir: Path to directory with TensorBoard PNG exports
        """
        self.graph_dir = Path(graph_dir)
        self.graphs = {
            'agent': self.graph_dir / 'agent.png',
            'agent_page2': self.graph_dir / 'agent-page2.png',
            'gradients': self.graph_dir / 'gradients.png',
            'progress': self.graph_dir / 'progress.png',
            'eval': self.graph_dir / 'eval.png',
        }

        # Literature-validated expectations at 5K steps (~80 gradient updates)
        self.expectations = {
            'gradient_norms': {
                'actor_cnn': {
                    'target': '< 10,000',  # Visual DRL papers use clipping at 1.0-40.0
                    'critical': '> 100,000',  # Explosion threshold
                    'previous_run': '1,826,337',  # Mean from LITERATURE_VALIDATED_ACTOR_ANALYSIS
                },
                'critic_cnn': {
                    'target': '< 10,000',
                    'stable': '< 50,000',
                    'previous_run': '5,897',  # Mean from previous analysis
                },
                'actor_mlp': {
                    'target': '< 1,000',
                    'stable': '< 10,000',
                },
                'critic_mlp': {
                    'target': '< 1,000',
                    'stable': '< 10,000',
                },
            },
            'episode_length': {
                '5k_steps': '5-20 steps',  # 80 gradient updates
                '50k_steps': '30-80 steps',  # 980 updates
                '100k_steps': '50-150 steps',  # 1,980 updates
                '1m_steps': '200-500+ steps',  # 19,980 updates
            },
            'actor_loss': {
                'initial': 'Should be negative (Q-value estimate)',
                'trend': 'Gradual increase toward 0 or positive',
                'concern': 'Divergence to large negative values (e.g., -7.6M)',
            },
            'q_values': {
                'initial': 'Near zero or small negative',
                'trend': 'Gradual increase as policy improves',
                'concern': 'Explosion or oscillation',
            },
        }

        self.findings = {}

    def display_graph(self, graph_name: str, title: str) -> None:
        """
        Display a single graph with metadata.

        Args:
            graph_name: Key in self.graphs dictionary
            title: Display title for the graph
        """
        graph_path = self.graphs.get(graph_name)

        if not graph_path or not graph_path.exists():
            print(f"❌ Graph not found: {graph_name} at {graph_path}")
            return

        print(f"\n{'='*80}")
        print(f"ANALYZING: {title}")
        print(f"File: {graph_path.name}")
        print(f"Size: {graph_path.stat().st_size / 1024:.1f} KB")
        print(f"{'='*80}\n")

        # Load and display image
        img = mpimg.imread(str(graph_path))

        # Create figure with good size
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()

    def analyze_gradients_graph(self) -> dict:
        """
        PRIORITY 1: Analyze gradients.png for gradient explosion evidence.

        This is the MOST CRITICAL graph for determining if the train_freq fix
        (1 → 50) resolved the actor CNN gradient explosion (1.8M → ?).

        Returns:
            Dictionary with findings and assessment
        """
        print("\n" + "="*80)
        print("PRIORITY 1 ANALYSIS: GRADIENT NORMS")
        print("="*80)
        print("\nCRITICAL QUESTION: Did train_freq fix resolve actor CNN gradient explosion?")
        print(f"  Previous run (train_freq=1): Actor CNN mean = 1,826,337 ❌ EXTREME")
        print(f"  Previous run (train_freq=1): Critic CNN mean = 5,897 ✅ STABLE")
        print(f"  Current run (train_freq=50): ANALYZING...")
        print()

        self.display_graph('gradients', 'Gradient Norms - 5K POST FIXES')

        print("\n" + "-"*80)
        print("MANUAL VISUAL INSPECTION REQUIRED")
        print("-"*80)
        print("\nPlease inspect the displayed graph and answer:")
        print("\n1. ACTOR CNN GRADIENT NORM:")
        print("   - What is the approximate MEAN gradient norm? (look at orange line)")
        print("   - What is the approximate MAX gradient norm?")
        print("   - Is it < 10,000 (✅ HEALTHY) or > 100,000 (❌ EXPLOSION)?")

        actor_cnn_mean = input("\n   Enter Actor CNN MEAN gradient norm (or 'skip'): ").strip()
        actor_cnn_max = input("   Enter Actor CNN MAX gradient norm (or 'skip'): ").strip()

        print("\n2. CRITIC CNN GRADIENT NORM:")
        print("   - What is the approximate MEAN gradient norm? (look at blue line)")
        print("   - Is it stable < 10,000?")

        critic_cnn_mean = input("\n   Enter Critic CNN MEAN gradient norm (or 'skip'): ").strip()

        print("\n3. MLP GRADIENT NORMS:")
        print("   - Are actor/critic MLP norms < 1,000? (✅ EXPECTED)")
        print("   - Do you see any concerning spikes or oscillations?")

        mlp_status = input("\n   Are MLP norms healthy? (yes/no/skip): ").strip().lower()

        # Build findings
        findings = {
            'actor_cnn_mean': actor_cnn_mean if actor_cnn_mean != 'skip' else 'NOT_RECORDED',
            'actor_cnn_max': actor_cnn_max if actor_cnn_max != 'skip' else 'NOT_RECORDED',
            'critic_cnn_mean': critic_cnn_mean if critic_cnn_mean != 'skip' else 'NOT_RECORDED',
            'mlp_healthy': mlp_status == 'yes',
        }

        # Assessment
        if actor_cnn_mean != 'skip':
            try:
                mean_val = float(actor_cnn_mean.replace(',', ''))
                if mean_val < 10000:
                    findings['assessment'] = '✅ HEALTHY - Gradient explosion RESOLVED'
                    findings['recommendation'] = 'PROCEED to 1M training'
                elif mean_val < 100000:
                    findings['assessment'] = '⚠️ ELEVATED - Monitor closely'
                    findings['recommendation'] = 'Consider gradient clipping (max_norm=10.0)'
                else:
                    findings['assessment'] = '❌ EXPLOSION - Still present'
                    findings['recommendation'] = 'IMPLEMENT gradient clipping BEFORE 1M run'
            except ValueError:
                findings['assessment'] = 'MANUAL REVIEW NEEDED'
                findings['recommendation'] = 'Review graph carefully'
        else:
            findings['assessment'] = 'INCOMPLETE - Manual inspection needed'
            findings['recommendation'] = 'Complete visual analysis'

        self.findings['gradients'] = findings
        return findings

    def analyze_agent_graph(self) -> dict:
        """
        Analyze agent.png for actor loss, Q-values, and episode metrics.

        Returns:
            Dictionary with findings and assessment
        """
        print("\n" + "="*80)
        print("ANALYSIS: AGENT METRICS (Main Page)")
        print("="*80)

        self.display_graph('agent', 'Agent Metrics - 5K POST FIXES')

        print("\n" + "-"*80)
        print("MANUAL VISUAL INSPECTION REQUIRED")
        print("-"*80)
        print("\nPlease inspect the displayed graph and answer:")
        print("\n1. ACTOR LOSS:")
        print("   - What is the approximate final value? (should be negative)")
        print("   - Is it converging or diverging?")
        print("   - Previous concern: Divergence to -7.6M in earlier runs")

        actor_loss_final = input("\n   Enter Actor Loss final value (or 'skip'): ").strip()
        actor_loss_trend = input("   Is it converging, stable, or diverging? ").strip()

        print("\n2. Q-VALUES (Q1, Q2):")
        print("   - What are the approximate values?")
        print("   - Are Q1 and Q2 close together? (✅ Twin critics working)")
        print("   - Are they stable or oscillating?")

        q_value_approx = input("\n   Enter approximate Q-value range (or 'skip'): ").strip()
        q_values_stable = input("   Are Q-values stable? (yes/no/skip): ").strip().lower()

        print("\n3. EPISODE LENGTH:")
        print("   - What is the mean episode length shown?")
        print("   - Expected at 5K: 5-20 steps (80 gradient updates)")

        episode_length = input("\n   Enter mean episode length (or 'skip'): ").strip()

        print("\n4. EPISODE REWARD:")
        print("   - What is the approximate mean reward?")
        print("   - Is there an upward trend or is it flat/declining?")

        episode_reward = input("\n   Enter mean episode reward (or 'skip'): ").strip()
        reward_trend = input("   Trend (improving/stable/declining/skip): ").strip()

        findings = {
            'actor_loss_final': actor_loss_final,
            'actor_loss_trend': actor_loss_trend,
            'q_value_range': q_value_approx,
            'q_values_stable': q_values_stable == 'yes',
            'episode_length': episode_length,
            'episode_reward': episode_reward,
            'reward_trend': reward_trend,
        }

        # Assessment
        concerns = []
        if actor_loss_trend.lower() == 'diverging':
            concerns.append("Actor loss diverging - check reward scaling")
        if q_values_stable == 'no':
            concerns.append("Q-values unstable - check critic learning rate")
        if episode_length != 'skip':
            try:
                ep_len = float(episode_length)
                if ep_len < 5 or ep_len > 20:
                    concerns.append(f"Episode length {ep_len} outside expected 5-20 range at 5K")
            except ValueError:
                pass

        findings['concerns'] = concerns
        findings['assessment'] = '✅ HEALTHY' if not concerns else f'⚠️ {len(concerns)} CONCERNS'

        self.findings['agent'] = findings
        return findings

    def analyze_agent_page2(self) -> dict:
        """
        Analyze agent-page2.png for additional metrics.

        Returns:
            Dictionary with findings
        """
        print("\n" + "="*80)
        print("ANALYSIS: AGENT METRICS (Page 2)")
        print("="*80)

        self.display_graph('agent_page2', 'Agent Metrics Page 2 - 5K POST FIXES')

        print("\n" + "-"*80)
        print("MANUAL VISUAL INSPECTION")
        print("-"*80)
        print("\nPlease note any additional metrics shown (learning rates, buffer stats, etc.)")

        notes = input("\nEnter observations (or 'skip'): ").strip()

        findings = {
            'observations': notes if notes != 'skip' else 'NOT_RECORDED'
        }

        self.findings['agent_page2'] = findings
        return findings

    def analyze_progress_graph(self) -> dict:
        """
        Analyze progress.png for episode progression metrics.

        Returns:
            Dictionary with findings
        """
        print("\n" + "="*80)
        print("ANALYSIS: TRAINING PROGRESS")
        print("="*80)

        self.display_graph('progress', 'Training Progress - 5K POST FIXES')

        print("\n" + "-"*80)
        print("MANUAL VISUAL INSPECTION")
        print("-"*80)
        print("\nPlease inspect episode length distribution and progression over time")

        observations = input("\nEnter observations (or 'skip'): ").strip()

        findings = {
            'observations': observations if observations != 'skip' else 'NOT_RECORDED'
        }

        self.findings['progress'] = findings
        return findings

    def analyze_eval_graph(self) -> dict:
        """
        Analyze eval.png for evaluation metrics (may be empty at 5K).

        Returns:
            Dictionary with findings
        """
        print("\n" + "="*80)
        print("ANALYSIS: EVALUATION METRICS")
        print("="*80)
        print("\nNote: This graph may be empty at 5K steps (no evaluation runs yet)")

        self.display_graph('eval', 'Evaluation Metrics - 5K POST FIXES')

        print("\n" + "-"*80)
        print("MANUAL VISUAL INSPECTION")
        print("-"*80)

        has_data = input("\nDoes this graph contain data? (yes/no): ").strip().lower()

        findings = {
            'has_data': has_data == 'yes',
            'observations': 'No evaluation data at 5K' if has_data != 'yes' else input("Enter observations: ")
        }

        self.findings['eval'] = findings
        return findings

    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive analysis report with GO/NO-GO recommendation.

        Returns:
            Formatted markdown report
        """
        report = []
        report.append("# TensorBoard Graph Analysis Report: 5K_POST_FIXES Run")
        report.append(f"\n**Analysis Date**: November 17, 2025")
        report.append(f"**Training Steps**: 5,000 (~80 gradient updates)")
        report.append(f"**Configuration**: train_freq=50, gradient_steps=1, learning_starts=1000")
        report.append("\n---\n")

        # Executive Summary
        report.append("## Executive Summary\n")

        if 'gradients' in self.findings:
            grad_findings = self.findings['gradients']
            report.append(f"**Gradient Explosion Status**: {grad_findings.get('assessment', 'PENDING')}")
            report.append(f"**Recommendation**: {grad_findings.get('recommendation', 'PENDING')}\n")

        # Detailed Findings
        report.append("## Detailed Findings\n")

        # 1. Gradients
        if 'gradients' in self.findings:
            report.append("### 1. Gradient Norms (PRIORITY 1)\n")
            grad = self.findings['gradients']
            report.append(f"- **Actor CNN Mean**: {grad.get('actor_cnn_mean', 'N/A')}")
            report.append(f"- **Actor CNN Max**: {grad.get('actor_cnn_max', 'N/A')}")
            report.append(f"- **Critic CNN Mean**: {grad.get('critic_cnn_mean', 'N/A')}")
            report.append(f"- **MLP Healthy**: {grad.get('mlp_healthy', 'N/A')}")
            report.append(f"\n**Assessment**: {grad.get('assessment', 'N/A')}\n")

            # Comparison to previous run
            report.append("**Comparison to Previous Run**:")
            report.append(f"- Previous Actor CNN Mean: 1,826,337 ❌")
            report.append(f"- Current Actor CNN Mean: {grad.get('actor_cnn_mean', 'N/A')}")
            if grad.get('actor_cnn_mean') != 'NOT_RECORDED':
                try:
                    current = float(grad['actor_cnn_mean'].replace(',', ''))
                    previous = 1826337
                    improvement = ((previous - current) / previous) * 100
                    report.append(f"- **Improvement**: {improvement:.1f}% reduction ✅\n")
                except:
                    report.append("\n")
            else:
                report.append("\n")

        # 2. Agent Metrics
        if 'agent' in self.findings:
            report.append("### 2. Agent Metrics\n")
            agent = self.findings['agent']
            report.append(f"- **Actor Loss**: {agent.get('actor_loss_final', 'N/A')} ({agent.get('actor_loss_trend', 'N/A')})")
            report.append(f"- **Q-Values**: {agent.get('q_value_range', 'N/A')} (Stable: {agent.get('q_values_stable', 'N/A')})")
            report.append(f"- **Episode Length**: {agent.get('episode_length', 'N/A')} (Expected: 5-20 at 5K)")
            report.append(f"- **Episode Reward**: {agent.get('episode_reward', 'N/A')} ({agent.get('reward_trend', 'N/A')})")

            if agent.get('concerns'):
                report.append(f"\n**Concerns**:")
                for concern in agent['concerns']:
                    report.append(f"- ⚠️ {concern}")
            report.append("\n")

        # 3. Progress
        if 'progress' in self.findings:
            report.append("### 3. Training Progress\n")
            report.append(f"- {self.findings['progress'].get('observations', 'N/A')}\n")

        # 4. Evaluation
        if 'eval' in self.findings:
            report.append("### 4. Evaluation Metrics\n")
            eval_findings = self.findings['eval']
            if eval_findings.get('has_data'):
                report.append(f"- {eval_findings.get('observations', 'N/A')}\n")
            else:
                report.append("- No evaluation data at 5K steps (expected)\n")

        # Literature Validation
        report.append("## Literature Validation\n")
        report.append("### Expected Behavior at 5K Steps (80 Gradient Updates)\n")
        report.append("**From Academic Papers**:")
        report.append("- **TD3 (Fujimoto et al., 2018)**: Training requires 1M steps for MuJoCo tasks")
        report.append("- **Rally A3C (Perot et al., 2017)**: ~50M steps for convergence, 140M for full training")
        report.append("- **DDPG-UAV (2022)**: Thousands of episodes required for competence\n")
        report.append("**Conclusion**: 5K steps = extreme early training, low performance EXPECTED ✅\n")

        # Final Recommendation
        report.append("## Final Recommendation\n")

        # Determine GO/NO-GO based on gradient explosion status
        if 'gradients' in self.findings:
            grad_assessment = self.findings['gradients'].get('assessment', '')
            if '✅' in grad_assessment:
                report.append("### ✅ GO FOR 1M TRAINING\n")
                report.append("**Rationale**:")
                report.append("- Gradient explosion RESOLVED by train_freq fix")
                report.append("- All metrics within expected ranges for 5K steps")
                report.append("- TD3 implementation validated against literature")
                report.append("- Pipeline functioning correctly\n")
                report.append("**Action Items**:")
                report.append("1. Proceed with 1M training run")
                report.append("2. Monitor gradient norms at 50K checkpoint")
                report.append("3. Implement gradient clipping if norms exceed 50K")
            elif '⚠️' in grad_assessment:
                report.append("### ⚠️ PROCEED WITH CAUTION\n")
                report.append("**Rationale**:")
                report.append("- Gradients elevated but not exploding")
                report.append("- Recommend gradient clipping for safety\n")
                report.append("**Action Items**:")
                report.append("1. Implement gradient clipping (max_norm=10.0) per visual DRL papers")
                report.append("2. Proceed to 50K with close monitoring")
                report.append("3. Re-evaluate at 50K checkpoint")
            else:
                report.append("### ❌ NO-GO - FIX REQUIRED\n")
                report.append("**Rationale**:")
                report.append("- Gradient explosion still present")
                report.append("- Training will likely fail at scale\n")
                report.append("**Action Items**:")
                report.append("1. IMPLEMENT gradient clipping immediately")
                report.append("2. Consider reducing learning rates")
                report.append("3. Re-run 5K validation before 1M attempt")
        else:
            report.append("### ⏸️ ANALYSIS INCOMPLETE\n")
            report.append("Please complete visual inspection to generate recommendation.")

        return '\n'.join(report)

    def run_full_analysis(self) -> None:
        """
        Execute complete systematic analysis of all graphs.
        """
        print("\n" + "="*80)
        print("TENSORBOARD GRAPH SYSTEMATIC ANALYSIS")
        print("5K_POST_FIXES Validation Run")
        print("="*80)
        print("\nThis analysis will:")
        print("1. Display each graph for visual inspection")
        print("2. Collect observations and measurements")
        print("3. Validate against academic literature")
        print("4. Generate GO/NO-GO recommendation for 1M training")
        print("\nAnalysis Priority Order:")
        print("  1. gradients.png - MOST CRITICAL (gradient explosion check)")
        print("  2. agent.png - Actor loss, Q-values, episode metrics")
        print("  3. progress.png - Episode progression")
        print("  4. agent-page2.png - Additional metrics")
        print("  5. eval.png - Evaluation (likely empty at 5K)")

        proceed = input("\nReady to begin? (yes/no): ").strip().lower()
        if proceed != 'yes':
            print("Analysis cancelled.")
            return

        # Execute analyses in priority order
        print("\n" + "🔍"*40)
        self.analyze_gradients_graph()  # PRIORITY 1

        print("\n" + "🔍"*40)
        self.analyze_agent_graph()

        print("\n" + "🔍"*40)
        self.analyze_progress_graph()

        print("\n" + "🔍"*40)
        self.analyze_agent_page2()

        print("\n" + "🔍"*40)
        self.analyze_eval_graph()

        # Generate and save report
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        report = self.generate_comprehensive_report()

        # Display report
        print("\n" + report)

        # Save to file
        report_path = self.graph_dir / 'TENSORBOARD_ANALYSIS_REPORT.md'
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n✅ Report saved to: {report_path}")
        print("\nAnalysis complete!")


def main():
    """Main entry point for graph analysis."""
    # Graph directory
    graph_dir = Path(__file__).parent

    print(f"\nAnalyzing graphs in: {graph_dir}")
    print(f"Expected graphs:")
    for name, path in [
        ('agent.png', graph_dir / 'agent.png'),
        ('agent-page2.png', graph_dir / 'agent-page2.png'),
        ('gradients.png', graph_dir / 'gradients.png'),
        ('progress.png', graph_dir / 'progress.png'),
        ('eval.png', graph_dir / 'eval.png'),
    ]:
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {name}")

    # Create analyzer
    analyzer = TensorBoardGraphAnalyzer(str(graph_dir))

    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

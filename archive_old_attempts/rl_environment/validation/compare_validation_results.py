"""
Compare validation results from multiple runs.

This script allows comparing validation results from multiple runs
to track progress over time or compare different environment configurations.

Usage:
    python compare_validation_results.py --results results_run1.json results_run2.json --output comparison.md
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ResultComparator")

class ValidationResultComparator:
    """
    Compare validation results from multiple runs.
    """

    def __init__(self, results_files: List[str]):
        """
        Initialize the comparator with result files.

        Args:
            results_files: List of paths to result JSON files
        """
        self.results_files = results_files
        self.results = []
        self.labels = []

        # Load results
        for file_path in results_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Generate label from file or timestamp
                label = os.path.basename(file_path).replace("validation_results_", "").replace(".json", "")
                if "timestamp" in data.get("summary", {}):
                    ts = data["summary"]["timestamp"]
                    try:
                        # Try to parse ISO format timestamp
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        label = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass

                self.results.append(data)
                self.labels.append(label)

            except Exception as e:
                logger.error(f"Error loading results from {file_path}: {e}")

    def compare_summary(self) -> pd.DataFrame:
        """
        Compare summary statistics across runs.

        Returns:
            DataFrame with summary comparison
        """
        comparison = {
            "Run": self.labels,
            "Tests Passed": [r["summary"].get("passed", 0) for r in self.results],
            "Tests Failed": [r["summary"].get("failed", 0) for r in self.results],
            "Total Tests": [r["summary"].get("total_tests", 0) for r in self.results],
            "Pass Rate (%)": [
                round(100 * r["summary"].get("passed", 0) / r["summary"].get("total_tests", 1), 1)
                for r in self.results
            ],
            "Duration (s)": [r["summary"].get("duration", 0) for r in self.results],
            "Overall Status": [r["summary"].get("overall_passed", False) for r in self.results]
        }

        return pd.DataFrame(comparison)

    def compare_test_results(self) -> pd.DataFrame:
        """
        Compare individual test results across runs.

        Returns:
            DataFrame with test result comparison
        """
        # Collect all test names across all runs
        all_tests = set()
        for result in self.results:
            all_tests.update(result.get("tests", {}).keys())

        all_tests = sorted(all_tests)

        # Create comparison data
        comparison = {
            "Test": all_tests,
        }

        for i, (label, result) in enumerate(zip(self.labels, self.results)):
            column = f"Run {i+1}: {label}"
            comparison[column] = []

            for test in all_tests:
                if test in result.get("tests", {}):
                    comparison[column].append(result["tests"][test].get("passed", False))
                else:
                    comparison[column].append(None)

        return pd.DataFrame(comparison)

    def compare_performance(self) -> Dict[str, pd.DataFrame]:
        """
        Compare performance metrics across runs.

        Returns:
            Dictionary of DataFrames with performance metrics comparison
        """
        performance_dfs = {}

        # Find all performance categories
        perf_categories = set()
        for result in self.results:
            perf_categories.update(result.get("performance", {}).keys())

        # Compare each performance category
        for category in perf_categories:
            metrics = set()
            for result in self.results:
                if category in result.get("performance", {}):
                    metrics.update(result["performance"][category].keys())

            metrics = sorted(metrics)
            comparison = {"Metric": metrics}

            for i, (label, result) in enumerate(zip(self.labels, self.results)):
                column = f"Run {i+1}: {label}"
                comparison[column] = []

                for metric in metrics:
                    if (category in result.get("performance", {}) and
                        metric in result["performance"][category]):
                        comparison[column].append(result["performance"][category][metric])
                    else:
                        comparison[column].append(None)

            performance_dfs[category] = pd.DataFrame(comparison)

        return performance_dfs

    def generate_comparison_report(self, output_file: str) -> None:
        """
        Generate a markdown report comparing validation results.

        Args:
            output_file: Path to save the markdown report
        """
        logger.info(f"Generating comparison report to {output_file}")

        # Generate charts
        charts_dir = os.path.join(os.path.dirname(output_file), "validation_charts")
        os.makedirs(charts_dir, exist_ok=True)

        summary_chart = os.path.join(charts_dir, "summary_comparison.png")
        self._plot_summary_comparison(summary_chart)

        category_chart = os.path.join(charts_dir, "category_comparison.png")
        self._plot_category_comparison(category_chart)

        # Start building report
        report = []
        report.append("# CARLA Environment Validation Comparison Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Add summary comparison
        report.append("\n## Summary Comparison")
        summary_df = self.compare_summary()
        report.append(f"\n{summary_df.to_markdown(index=False)}")

        # Add summary chart
        rel_summary_chart = os.path.relpath(summary_chart, os.path.dirname(output_file))
        report.append(f"\n![Summary Comparison]({rel_summary_chart})")

        # Add category chart
        rel_category_chart = os.path.relpath(category_chart, os.path.dirname(output_file))
        report.append(f"\n![Category Comparison]({rel_category_chart})")

        # Add test results comparison
        report.append("\n## Test Results Comparison")
        test_df = self.compare_test_results()
        report.append(f"\n{test_df.to_markdown(index=False)}")

        # Add performance comparison
        report.append("\n## Performance Comparison")
        perf_dfs = self.compare_performance()
        for category, df in perf_dfs.items():
            report.append(f"\n### {category.title()}")
            report.append(f"\n{df.to_markdown(index=False)}")

            # Add performance chart for this category
            perf_chart = os.path.join(charts_dir, f"{category}_comparison.png")
            self._plot_performance_comparison(category, perf_chart)
            rel_perf_chart = os.path.relpath(perf_chart, os.path.dirname(output_file))
            report.append(f"\n![{category.title()} Comparison]({rel_perf_chart})")

        # Add improvement analysis
        report.append("\n## Improvement Analysis")
        if len(self.results) >= 2:
            first_run = self.results[0]
            last_run = self.results[-1]

            first_passed = first_run["summary"].get("passed", 0)
            last_passed = last_run["summary"].get("passed", 0)

            # Only analyze improvement if runs have same number of tests
            if first_run["summary"].get("total_tests", 0) == last_run["summary"].get("total_tests", 0):
                improvement = last_passed - first_passed

                if improvement > 0:
                    report.append(f"\n✅ **Improvement detected**: +{improvement} passing tests since first run")

                    # Find which tests improved
                    improved_tests = []
                    for test_name, test_data in last_run.get("tests", {}).items():
                        if test_data.get("passed", False):
                            if test_name in first_run.get("tests", {}) and not first_run["tests"][test_name].get("passed", False):
                                improved_tests.append(test_name)

                    if improved_tests:
                        report.append("\n### Fixed Tests")
                        for test in improved_tests:
                            report.append(f"\n- ✅ `{test}`")

                elif improvement < 0:
                    report.append(f"\n⚠️ **Regression detected**: {improvement} fewer passing tests since first run")

                    # Find which tests regressed
                    regressed_tests = []
                    for test_name, test_data in first_run.get("tests", {}).items():
                        if test_data.get("passed", False):
                            if test_name in last_run.get("tests", {}) and not last_run["tests"][test_name].get("passed", False):
                                regressed_tests.append(test_name)

                    if regressed_tests:
                        report.append("\n### Regressed Tests")
                        for test in regressed_tests:
                            report.append(f"\n- ❌ `{test}`")

                else:
                    report.append("\nNo change in number of passing tests since first run")
            else:
                report.append("\nCannot compare improvement due to different number of tests between runs")

            # Compare performance
            report.append("\n### Performance Changes")
            if all(r.get("performance") for r in [first_run, last_run]):
                # Compare reset time
                if all("reset_time" in r.get("performance", {}) for r in [first_run, last_run]):
                    first_reset = first_run["performance"]["reset_time"].get("mean", 0)
                    last_reset = last_run["performance"]["reset_time"].get("mean", 0)

                    if first_reset > 0 and last_reset > 0:
                        pct_change = ((last_reset - first_reset) / first_reset) * 100
                        if pct_change <= -5:  # Improved by at least 5%
                            report.append(f"\n✅ **Reset time improved by {abs(pct_change):.1f}%**: {first_reset:.3f}s → {last_reset:.3f}s")
                        elif pct_change >= 5:  # Worsened by at least 5%
                            report.append(f"\n⚠️ **Reset time worsened by {pct_change:.1f}%**: {first_reset:.3f}s → {last_reset:.3f}s")

                # Compare step time
                if all("step_time" in r.get("performance", {}) for r in [first_run, last_run]):
                    first_step = first_run["performance"]["step_time"].get("mean", 0)
                    last_step = last_run["performance"]["step_time"].get("mean", 0)

                    if first_step > 0 and last_step > 0:
                        pct_change = ((last_step - first_step) / first_step) * 100
                        if pct_change <= -5:  # Improved by at least 5%
                            report.append(f"\n✅ **Step time improved by {abs(pct_change):.1f}%**: {first_step:.3f}s → {last_step:.3f}s")
                        elif pct_change >= 5:  # Worsened by at least 5%
                            report.append(f"\n⚠️ **Step time worsened by {pct_change:.1f}%**: {first_step:.3f}s → {last_step:.3f}s")
        else:
            report.append("\nNot enough runs to analyze improvement")

        # Write report
        with open(output_file, 'w') as f:
            f.write("\n".join(report))

        logger.info(f"Report saved to {output_file}")

    def _plot_summary_comparison(self, output_file: str) -> None:
        """
        Plot summary comparison chart.

        Args:
            output_file: Path to save the chart image
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.labels))
        width = 0.35

        passed = [r["summary"].get("passed", 0) for r in self.results]
        failed = [r["summary"].get("failed", 0) for r in self.results]

        ax.bar(x, passed, width, label='Passed', color='#4CAF50')
        ax.bar(x, failed, width, bottom=passed, label='Failed', color='#F44336')

        ax.set_title('Validation Tests Summary')
        ax.set_xlabel('Run')
        ax.set_ylabel('Number of Tests')
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=45, ha="right")
        ax.legend()

        fig.tight_layout()
        plt.savefig(output_file)

    def _plot_category_comparison(self, output_file: str) -> None:
        """
        Plot test category comparison chart.

        Args:
            output_file: Path to save the chart image
        """
        # Group tests by category
        categories = ["init", "reset", "action", "step", "reward", "termination", "cleanup"]

        category_results = []
        for result in self.results:
            cat_data = {}
            for category in categories:
                cat_tests = [t for t in result.get("tests", {}) if t.startswith(f"{category}_")]
                passed = sum(1 for t in cat_tests if result["tests"][t].get("passed", False))
                total = len(cat_tests)
                cat_data[category] = (passed, total)
            category_results.append(cat_data)

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(categories))
        width = 0.8 / len(self.results)

        for i, (label, cat_data) in enumerate(zip(self.labels, category_results)):
            pass_rates = [
                100 * cat_data.get(cat, (0, 0))[0] / max(cat_data.get(cat, (0, 1))[1], 1)
                for cat in categories
            ]
            ax.bar(x + i*width - width*(len(self.results)-1)/2,
                   pass_rates, width, label=label)

        ax.set_title('Pass Rate by Test Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Pass Rate (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylim(0, 105)
        ax.legend()

        fig.tight_layout()
        plt.savefig(output_file)

    def _plot_performance_comparison(self, category: str, output_file: str) -> None:
        """
        Plot performance comparison chart for a category.

        Args:
            category: Performance category name
            output_file: Path to save the chart image
        """
        metrics = ["mean", "std", "min", "max"]
        values = []

        for result in self.results:
            if category in result.get("performance", {}):
                values.append([
                    result["performance"][category].get(metric, 0)
                    for metric in metrics
                ])
            else:
                values.append([0] * len(metrics))

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics))
        width = 0.8 / len(self.results)

        for i, (label, vals) in enumerate(zip(self.labels, values)):
            ax.bar(x + i*width - width*(len(self.results)-1)/2,
                   vals, width, label=label)

        ax.set_title(f'{category.title()} Performance')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()

        fig.tight_layout()
        plt.savefig(output_file)

def main():
    """Run the result comparator."""
    parser = argparse.ArgumentParser(description="CARLA Environment Validation Result Comparator")
    parser.add_argument("--results", type=str, nargs='+', required=True,
                      help="Paths to JSON results files")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to save the comparison report")

    args = parser.parse_args()

    missing_files = [f for f in args.results if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Results files not found: {', '.join(missing_files)}")
        return 1

    try:
        comparator = ValidationResultComparator(args.results)
        comparator.generate_comparison_report(args.output)

        logger.info(f"Comparison report saved to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error comparing results: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

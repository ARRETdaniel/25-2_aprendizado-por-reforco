"""
Validation result viewer for CARLA environment wrapper.

This script provides a simple interface for viewing validation test results
in a more readable format.

Usage:
    python view_validation_results.py --results validation_results_20250815_120000.json
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ResultViewer")

def status_str(status: bool) -> str:
    """Get colorful status string (works in terminals with ANSI color support)."""
    return f"\033[92mPASS\033[0m" if status else f"\033[91mFAIL\033[0m"

def display_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """
    Display validation results in a readable format.

    Args:
        results: Dictionary with test results
        verbose: Whether to show detailed information for each test
    """
    # Display summary
    summary = results["summary"]

    print("\n" + "="*80)
    print(f"CARLA ENVIRONMENT VALIDATION SUMMARY")
    print("="*80)

    print(f"\nTimestamp: {summary.get('timestamp', 'Unknown')}")
    print(f"Duration: {summary.get('duration', 0):.2f} seconds")
    print(f"Tests: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed")
    print(f"Overall status: {status_str(summary.get('overall_passed', False))}")

    # Display environment info
    env_info = summary.get("environment_info", {})
    print("\nEnvironment:")
    for key, value in env_info.items():
        print(f"- {key}: {value}")

    # Group tests by category
    tests_by_category = {}
    for test_name, test_data in results["tests"].items():
        category = test_name.split('_')[0]
        if category not in tests_by_category:
            tests_by_category[category] = []
        tests_by_category[category].append((test_name, test_data))

    # Display test results by category
    print("\n" + "="*80)
    print("TEST RESULTS BY CATEGORY")
    print("="*80)

    for category, tests in tests_by_category.items():
        passed_count = sum(1 for _, data in tests if data["passed"])
        print(f"\n{category.upper()} ({passed_count}/{len(tests)} passed):")

        for test_name, test_data in tests:
            status = status_str(test_data["passed"])
            print(f"  - {test_name}: {status}")

            if verbose:
                print(f"    Duration: {test_data.get('duration', 0):.2f}s")

                details = test_data.get("details", {})
                if "message" in details:
                    print(f"    Message: {details['message']}")
                elif "error" in details:
                    print(f"    Error: {details['error']}")

                # Show iteration results if available
                iterations = test_data.get("iterations", [])
                if iterations:
                    print(f"    Iterations: {len(iterations)}")
                    passed_iterations = sum(1 for it in iterations if it["passed"])
                    print(f"    Passed iterations: {passed_iterations}/{len(iterations)}")

    # Display performance metrics if available
    perf = results.get("performance", {})
    if perf:
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)

        for metric, values in perf.items():
            print(f"\n{metric.upper()}:")
            for key, value in values.items():
                print(f"  - {key}: {value}")

    # Display failed tests for quick reference
    failed_tests = [(name, data) for name, data in results["tests"].items() if not data["passed"]]

    if failed_tests:
        print("\n" + "="*80)
        print("FAILED TESTS")
        print("="*80)

        for test_name, test_data in failed_tests:
            print(f"\n{test_name}:")
            details = test_data.get("details", {})
            if "error" in details:
                print(f"  Error: {details['error']}")
            elif isinstance(details, str):
                print(f"  Details: {details}")

    print("\n" + "="*80)

def main():
    """Run the result viewer."""
    parser = argparse.ArgumentParser(description="CARLA Environment Validation Result Viewer")
    parser.add_argument("--results", type=str, required=True,
                      help="Path to the JSON results file")
    parser.add_argument("--verbose", action="store_true",
                      help="Show detailed information for each test")

    args = parser.parse_args()

    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        return 1

    try:
        with open(args.results, 'r') as f:
            results = json.load(f)

        display_results(results, args.verbose)
        return 0
    except Exception as e:
        logger.error(f"Error displaying results: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

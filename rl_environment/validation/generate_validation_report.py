"""
Generate validation report from validation results.

This script processes the JSON validation results and creates a markdown report
based on the provided template.

Usage:
    python generate_validation_report.py --results results_20250815_120000.json --template validation_report_template.md
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import string
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ReportGenerator")

def status_to_emoji(status: bool) -> str:
    """Convert boolean status to emoji string."""
    return "✅ PASS" if status else "❌ FAIL"

def generate_report(results_file: str, template_file: str, output_file: Optional[str] = None) -> str:
    """
    Generate a validation report from results and template.

    Args:
        results_file: Path to the JSON results file
        template_file: Path to the markdown template file
        output_file: Path to save the generated report (optional)

    Returns:
        Path to the generated report
    """
    logger.info(f"Generating report from {results_file} using template {template_file}")

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load template
    with open(template_file, 'r') as f:
        template_content = f.read()

    # Prepare template variables
    variables = {
        "DATE": datetime.now().strftime("%Y-%m-%d"),
        "OVERALL_STATUS": status_to_emoji(results["summary"].get("overall_passed", False)),
        "OVERALL_NOTES": f"{results['summary']['passed']}/{results['summary']['total_tests']} tests passed"
    }

    # Add component statuses
    components = {
        "INIT": "init_",
        "RESET": "reset_",
        "ACTION": "action_",
        "STEP": "step_",
        "REWARD": "reward_",
        "TERM": "termination_",
        "CLEANUP": "cleanup_"
    }

    for key, prefix in components.items():
        component_tests = [name for name in results["tests"] if name.startswith(prefix)]
        if component_tests:
            passed = sum(1 for name in component_tests if results["tests"][name]["passed"])
            total = len(component_tests)
            variables[f"{key}_STATUS"] = status_to_emoji(passed == total)
            variables[f"{key}_NOTES"] = f"{passed}/{total} tests passed"
        else:
            variables[f"{key}_STATUS"] = "⚠️ NOT TESTED"
            variables[f"{key}_NOTES"] = "No tests executed"

    # Add individual test results
    for test_name, test_data in results["tests"].items():
        status_var = f"{test_name.upper()}_STATUS"
        details_var = f"{test_name.upper()}_DETAILS"

        variables[status_var] = status_to_emoji(test_data["passed"])

        # Format details
        if test_data["passed"]:
            if "message" in test_data["details"]:
                variables[details_var] = test_data["details"]["message"]
            else:
                variables[details_var] = "Test passed successfully"
        else:
            variables[details_var] = test_data["details"].get("error", "Unknown error")

    # Add performance metrics
    # These would come from more detailed results
    # For now, set placeholders
    perf_vars = [
        "RESET_AVG_TIME", "RESET_STD_TIME", "RESET_MIN_TIME", "RESET_MAX_TIME",
        "STEP_AVG_TIME", "STEP_STD_TIME", "STEP_MIN_TIME", "STEP_MAX_TIME",
        "MEMORY_PEAK"
    ]
    for var in perf_vars:
        variables[var] = "N/A"  # Not available in current results

    # Add issues and recommendations sections
    failed_tests = [name for name, data in results["tests"].items() if not data["passed"]]
    if failed_tests:
        variables["CRITICAL_ISSUES"] = "\n".join(f"- **{name}**: {results['tests'][name]['details'].get('error', 'Unknown error')}"
                                             for name in failed_tests)
    else:
        variables["CRITICAL_ISSUES"] = "No critical issues identified."

    variables["NON_CRITICAL_ISSUES"] = "No non-critical issues identified."

    # Add recommendations based on failed tests
    if failed_tests:
        variables["RECOMMENDATIONS"] = """
1. Fix the identified issues in the environment wrapper implementation.
2. Re-run the validation tests to verify the fixes.
3. Consider adding more extensive tests for specific components that failed.
"""
    else:
        variables["RECOMMENDATIONS"] = """
1. Add more tests for edge cases in the environment interaction.
2. Consider performance optimization for step and reset operations.
3. Implement additional test coverage for different weather conditions and towns.
"""

    # Add conclusion
    if results["summary"].get("overall_passed", False):
        variables["CONCLUSION"] = """
The environment wrapper has passed all validation tests and is considered ready for use in reinforcement learning experiments. The tests covered all major components of the environment wrapper, including initialization, reset, action handling, step function, reward calculation, termination conditions, and resource management.

The wrapper provides a reliable and consistent interface between reinforcement learning agents and the CARLA simulator, enabling efficient and effective training of autonomous driving policies.
"""
    else:
        variables["CONCLUSION"] = f"""
The environment wrapper has failed {results['summary']['failed']} out of {results['summary']['total_tests']} validation tests. The failing tests indicate issues that need to be addressed before the wrapper can be considered reliable for reinforcement learning experiments.

The issues are primarily related to the following components:
{', '.join(key for key, prefix in components.items() if any(not results["tests"][name]["passed"] for name in results["tests"] if name.startswith(prefix) and results["tests"].get(name, {}).get("passed") is not None))}

These issues should be addressed before proceeding with RL algorithm development and training.
"""

    # Fill in template
    template = string.Template(template_content)
    report_content = template.safe_substitute(variables)

    # Handle missing placeholders
    # Use a second pass with empty string replacements for any remaining placeholders
    template = string.Template(report_content)
    report_content = template.safe_substitute({key: "" for key in template.get_identifiers()})

    # Save report
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(results_file), f"validation_report_{timestamp}.md")

    with open(output_file, 'w') as f:
        f.write(report_content)

    logger.info(f"Report saved to {output_file}")
    return output_file

def main():
    """Run the report generator."""
    parser = argparse.ArgumentParser(description="CARLA Environment Validation Report Generator")
    parser.add_argument("--results", type=str, required=True,
                      help="Path to the JSON results file")
    parser.add_argument("--template", type=str, default=None,
                      help="Path to the markdown template file (default: validation_report_template.md in same dir as script)")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save the generated report (default: auto-generated based on results file)")

    args = parser.parse_args()

    # Use default template if none provided
    if args.template is None:
        args.template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_report_template.md")

    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        return 1

    if not os.path.exists(args.template):
        logger.error(f"Template file not found: {args.template}")
        return 1

    try:
        output_file = generate_report(args.results, args.template, args.output)
        logger.info(f"Report generation complete: {output_file}")
        return 0
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

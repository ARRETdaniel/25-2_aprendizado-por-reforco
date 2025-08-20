"""
Analyze validation results and provide explanations and suggested fixes.

This script analyzes validation results and provides detailed explanations
of any issues found, along with suggested fixes.

Usage:
    python analyze_validation_results.py --results validation_results_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ResultAnalyzer")

# Define issue explanations and fixes
ISSUE_KNOWLEDGE_BASE = {
    "init_environment_creation": {
        "explanation": "This test checks if the environment wrapper can be created without errors.",
        "possible_issues": [
            "Missing required dependencies",
            "Environment class not properly defined",
            "Initialization parameters not properly handled"
        ],
        "fix_suggestions": [
            "Verify all required dependencies are installed",
            "Check for syntax errors in the environment class definition",
            "Ensure all required parameters have proper default values or validation"
        ]
    },
    "init_carla_connection": {
        "explanation": "This test verifies that the environment can connect to the CARLA server.",
        "possible_issues": [
            "CARLA server not running",
            "Connection parameters (host, port) are incorrect",
            "Network issues blocking the connection",
            "Version mismatch between CARLA client and server"
        ],
        "fix_suggestions": [
            "Ensure CARLA server is running on the specified host and port",
            "Check network connectivity between the client and server",
            "Verify that the CARLA client and server versions are compatible",
            "Try increasing the connection timeout value"
        ]
    },
    "init_action_space": {
        "explanation": "This test checks if the action space is correctly defined.",
        "possible_issues": [
            "Action space not defined or incorrectly defined",
            "Action space bounds are inappropriate",
            "Missing action space validation"
        ],
        "fix_suggestions": [
            "Define the action space as a gym.spaces object (e.g., Box or Discrete)",
            "Ensure action bounds are reasonable for the task",
            "Implement validation to check if actions are within bounds"
        ]
    },
    "init_observation_space": {
        "explanation": "This test checks if the observation space is correctly defined.",
        "possible_issues": [
            "Observation space not defined or incorrectly defined",
            "Observation space dimensions don't match the actual observations",
            "Inappropriate bounds for observation values"
        ],
        "fix_suggestions": [
            "Define the observation space as a gym.spaces object that matches the actual observations",
            "Ensure observation dimensions are consistent",
            "Set appropriate bounds for observation values"
        ]
    },
    "reset_initial_state": {
        "explanation": "This test verifies that the reset method returns a valid initial state.",
        "possible_issues": [
            "Reset method not returning an observation",
            "Initial state format doesn't match the defined observation space",
            "Reset method raising exceptions"
        ],
        "fix_suggestions": [
            "Ensure the reset method returns a valid observation",
            "Make sure the returned observation matches the defined observation space",
            "Handle potential exceptions in the reset method"
        ]
    },
    "reset_vehicle_position": {
        "explanation": "This test checks if the vehicle is properly positioned after reset.",
        "possible_issues": [
            "Vehicle spawning at incorrect locations",
            "Vehicle position not being properly set",
            "Collision at spawn point preventing proper positioning"
        ],
        "fix_suggestions": [
            "Verify spawn point coordinates and orientation",
            "Ensure the vehicle is properly created and positioned",
            "Check for obstacles at spawn points",
            "Add retry logic for spawning in case of collisions"
        ]
    },
    "reset_consistent_outputs": {
        "explanation": "This test checks if reset produces consistent observation structures across multiple calls.",
        "possible_issues": [
            "Non-deterministic reset behavior",
            "Observation structure changing between resets",
            "Memory leaks causing inconsistent behavior over time"
        ],
        "fix_suggestions": [
            "Add deterministic seeding for random elements",
            "Ensure observation structure is consistent across resets",
            "Fix any memory leaks or resource issues"
        ]
    },
    "action_throttle_response": {
        "explanation": "This test verifies that the vehicle responds correctly to throttle actions.",
        "possible_issues": [
            "Throttle actions not being properly applied",
            "Vehicle physics issues preventing proper movement",
            "Throttle scaling or conversion issues"
        ],
        "fix_suggestions": [
            "Verify the throttle action mapping is correct",
            "Check if the action is properly scaled before sending to CARLA",
            "Ensure the vehicle is in a drivable state",
            "Check for physics timeouts or step delays"
        ]
    },
    "action_steering_response": {
        "explanation": "This test verifies that the vehicle responds correctly to steering actions.",
        "possible_issues": [
            "Steering actions not being properly applied",
            "Vehicle physics issues affecting steering",
            "Steering scaling or conversion issues",
            "Inappropriate steering limits"
        ],
        "fix_suggestions": [
            "Verify the steering action mapping is correct",
            "Check if the action is properly scaled before sending to CARLA",
            "Adjust steering limits to be appropriate for the vehicle",
            "Ensure steering actions are smoothed if needed"
        ]
    },
    "action_continuous_validity": {
        "explanation": "This test checks if the environment correctly handles continuous action values.",
        "possible_issues": [
            "Continuous actions not properly handled",
            "Action clipping or normalization issues",
            "Invalid action value handling"
        ],
        "fix_suggestions": [
            "Ensure continuous actions are properly normalized and scaled",
            "Implement proper clipping for action values",
            "Add validation and error handling for invalid actions"
        ]
    },
    "step_observation_format": {
        "explanation": "This test checks if the step method returns observations in the correct format.",
        "possible_issues": [
            "Step method not returning proper observation format",
            "Inconsistency between reset and step observation formats",
            "Missing or extra observation components"
        ],
        "fix_suggestions": [
            "Ensure step returns observations in the same format as reset",
            "Verify all required observation components are included",
            "Check for data type consistency across steps"
        ]
    },
    "step_reward_range": {
        "explanation": "This test verifies that rewards are within the expected range.",
        "possible_issues": [
            "Reward calculation errors",
            "Unbounded or extreme reward values",
            "Inconsistent reward scaling"
        ],
        "fix_suggestions": [
            "Implement bounds checking for reward values",
            "Normalize rewards to a consistent range",
            "Fix any calculation errors in the reward function",
            "Ensure reward components are properly weighted"
        ]
    },
    "step_done_flag": {
        "explanation": "This test checks if the done flag is correctly set on termination conditions.",
        "possible_issues": [
            "Termination conditions not properly detected",
            "Done flag not set correctly on collision or goal achievement",
            "Inconsistent termination logic"
        ],
        "fix_suggestions": [
            "Verify all termination conditions are properly detected",
            "Ensure the done flag is set to True when conditions are met",
            "Implement consistent termination logic across different scenarios"
        ]
    },
    "step_info_contents": {
        "explanation": "This test verifies that the info dictionary contains required information.",
        "possible_issues": [
            "Missing required info dictionary keys",
            "Incorrect data types in info dictionary",
            "Inconsistent info dictionary structure"
        ],
        "fix_suggestions": [
            "Add all required keys to the info dictionary",
            "Ensure consistent data types for info values",
            "Document the expected info dictionary structure"
        ]
    },
    "reward_progress_component": {
        "explanation": "This test checks if the progress component of the reward function is correctly calculated.",
        "possible_issues": [
            "Progress not properly tracked",
            "Progress component not increasing with actual progress",
            "Progress calculation errors"
        ],
        "fix_suggestions": [
            "Verify the progress tracking logic",
            "Ensure progress component increases as the vehicle moves toward the goal",
            "Fix any calculation errors in the progress reward component"
        ]
    },
    "reward_collision_penalty": {
        "explanation": "This test verifies that collision penalties are correctly applied.",
        "possible_issues": [
            "Collisions not properly detected",
            "Penalty not proportional to collision intensity",
            "Missing collision types"
        ],
        "fix_suggestions": [
            "Improve collision detection logic",
            "Scale penalty based on collision intensity",
            "Ensure all relevant collision types are handled",
            "Add debugging information for collision events"
        ]
    },
    "reward_consistency": {
        "explanation": "This test checks if the reward function is consistent across multiple runs.",
        "possible_issues": [
            "Non-deterministic reward calculation",
            "Inconsistent reward scaling",
            "State-dependent reward inconsistencies"
        ],
        "fix_suggestions": [
            "Add deterministic seeding for random elements in reward",
            "Ensure consistent reward scaling across runs",
            "Fix any state-dependent inconsistencies"
        ]
    },
    "termination_collision_detection": {
        "explanation": "This test verifies that the episode correctly terminates on vehicle collision.",
        "possible_issues": [
            "Collisions not properly detected",
            "Termination not triggered on collision",
            "Inconsistent collision handling"
        ],
        "fix_suggestions": [
            "Improve collision detection logic",
            "Ensure episode terminates on significant collisions",
            "Make collision detection sensitivity configurable",
            "Add different handling for minor versus major collisions"
        ]
    },
    "termination_goal_detection": {
        "explanation": "This test verifies that the episode correctly terminates when the goal is reached.",
        "possible_issues": [
            "Goal achievement not properly detected",
            "Termination not triggered on goal achievement",
            "Inconsistent goal detection criteria"
        ],
        "fix_suggestions": [
            "Improve goal detection logic",
            "Ensure episode terminates when goal is reached",
            "Make goal detection criteria more robust",
            "Consider using a distance threshold for goal achievement"
        ]
    },
    "cleanup_resources": {
        "explanation": "This test checks if all resources are properly released on environment close.",
        "possible_issues": [
            "Resources not properly released",
            "Memory leaks",
            "CARLA client connections not closed"
        ],
        "fix_suggestions": [
            "Ensure all resources are released in the close method",
            "Explicitly disconnect from CARLA and destroy actors",
            "Implement proper exception handling during cleanup",
            "Add debug logging for resource cleanup"
        ]
    }
}

def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze validation results and provide explanations and fix suggestions.

    Args:
        results: Dictionary with validation test results

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "summary": {
            "passed": results["summary"].get("passed", 0),
            "failed": results["summary"].get("failed", 0),
            "total_tests": results["summary"].get("total_tests", 0),
            "overall_passed": results["summary"].get("overall_passed", False)
        },
        "failed_tests": [],
        "explanations": {},
        "fix_suggestions": {}
    }

    # Analyze failed tests
    for test_name, test_data in results.get("tests", {}).items():
        if not test_data.get("passed", True):
            analysis["failed_tests"].append(test_name)

            # Add explanation and fix suggestions from knowledge base
            if test_name in ISSUE_KNOWLEDGE_BASE:
                analysis["explanations"][test_name] = ISSUE_KNOWLEDGE_BASE[test_name]["explanation"]
                analysis["fix_suggestions"][test_name] = {
                    "possible_issues": ISSUE_KNOWLEDGE_BASE[test_name]["possible_issues"],
                    "suggestions": ISSUE_KNOWLEDGE_BASE[test_name]["fix_suggestions"]
                }
            else:
                # Generic explanation for unknown tests
                analysis["explanations"][test_name] = "This test checks a specific aspect of the environment functionality."
                analysis["fix_suggestions"][test_name] = {
                    "possible_issues": ["Unknown issue with the test"],
                    "suggestions": ["Review the test code and environment implementation"]
                }

    # Provide overall assessment
    if not analysis["failed_tests"]:
        analysis["assessment"] = "All tests passed! The environment appears to be functioning correctly."
    else:
        severity = len(analysis["failed_tests"]) / analysis["summary"]["total_tests"]
        if severity < 0.2:
            assessment = "Minor issues detected. The environment is mostly functional but has a few issues to fix."
        elif severity < 0.5:
            assessment = "Moderate issues detected. Several important components of the environment have issues."
        else:
            assessment = "Critical issues detected. The environment has major functionality problems."

        analysis["assessment"] = assessment

    return analysis

def print_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print analysis results in a readable format.

    Args:
        analysis: Dictionary with analysis results
    """
    print("\n" + "="*80)
    print("CARLA ENVIRONMENT VALIDATION ANALYSIS")
    print("="*80 + "\n")

    # Print summary
    print(f"Tests: {analysis['summary']['passed']}/{analysis['summary']['total_tests']} passed")
    print(f"Overall status: {'PASSED' if analysis['summary']['overall_passed'] else 'FAILED'}")
    print(f"\nAssessment: {analysis['assessment']}")

    # Print failed tests with explanations and fixes
    if analysis["failed_tests"]:
        print("\n" + "="*80)
        print("FAILED TESTS ANALYSIS")
        print("="*80 + "\n")

        for test_name in analysis["failed_tests"]:
            print(f"\n## {test_name}")

            if test_name in analysis["explanations"]:
                print(f"\nExplanation: {analysis['explanations'][test_name]}")

            if test_name in analysis["fix_suggestions"]:
                print("\nPossible Issues:")
                for issue in analysis["fix_suggestions"][test_name]["possible_issues"]:
                    print(f"- {issue}")

                print("\nSuggested Fixes:")
                for fix in analysis["fix_suggestions"][test_name]["suggestions"]:
                    print(f"- {fix}")

    # Print next steps
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80 + "\n")

    if not analysis["failed_tests"]:
        print("1. Consider adding more test coverage for edge cases")
        print("2. Proceed to using the environment for reinforcement learning experiments")
        print("3. Monitor performance during training")
    else:
        print("1. Address the issues in the failed tests, focusing on:")
        for i, test in enumerate(analysis["failed_tests"][:3]):
            print(f"   - {test}")
        if len(analysis["failed_tests"]) > 3:
            print(f"   - And {len(analysis['failed_tests'])-3} more...")
        print("2. Re-run validation to verify fixes")
        print("3. Focus on critical functionality first (reset, step, termination conditions)")

def save_analysis(analysis: Dict[str, Any], output_file: str) -> None:
    """
    Save analysis results to a markdown file.

    Args:
        analysis: Dictionary with analysis results
        output_file: Path to save the analysis report
    """
    logger.info(f"Saving analysis to {output_file}")

    with open(output_file, 'w') as f:
        f.write("# CARLA Environment Validation Analysis\n\n")

        # Write summary
        f.write("## Summary\n\n")
        f.write(f"- Tests: {analysis['summary']['passed']}/{analysis['summary']['total_tests']} passed\n")
        f.write(f"- Overall status: **{'PASSED' if analysis['summary']['overall_passed'] else 'FAILED'}**\n")
        f.write(f"\n**Assessment**: {analysis['assessment']}\n")

        # Write failed tests with explanations and fixes
        if analysis["failed_tests"]:
            f.write("\n## Failed Tests Analysis\n\n")

            for test_name in analysis["failed_tests"]:
                f.write(f"\n### {test_name}\n\n")

                if test_name in analysis["explanations"]:
                    f.write(f"**Explanation**: {analysis['explanations'][test_name]}\n\n")

                if test_name in analysis["fix_suggestions"]:
                    f.write("**Possible Issues**:\n\n")
                    for issue in analysis["fix_suggestions"][test_name]["possible_issues"]:
                        f.write(f"- {issue}\n")

                    f.write("\n**Suggested Fixes**:\n\n")
                    for fix in analysis["fix_suggestions"][test_name]["suggestions"]:
                        f.write(f"- {fix}\n")

        # Write next steps
        f.write("\n## Recommended Next Steps\n\n")

        if not analysis["failed_tests"]:
            f.write("1. Consider adding more test coverage for edge cases\n")
            f.write("2. Proceed to using the environment for reinforcement learning experiments\n")
            f.write("3. Monitor performance during training\n")
        else:
            f.write("1. Address the issues in the failed tests, focusing on:\n")
            for i, test in enumerate(analysis["failed_tests"][:3]):
                f.write(f"   - `{test}`\n")
            if len(analysis["failed_tests"]) > 3:
                f.write(f"   - And {len(analysis['failed_tests'])-3} more...\n")
            f.write("2. Re-run validation to verify fixes\n")
            f.write("3. Focus on critical functionality first (reset, step, termination conditions)\n")

def main():
    """Run the result analyzer."""
    parser = argparse.ArgumentParser(description="CARLA Environment Validation Result Analyzer")
    parser.add_argument("--results", type=str, required=True,
                      help="Path to the JSON results file")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save the analysis report (default: auto-generated)")

    args = parser.parse_args()

    if not os.path.exists(args.results):
        logger.error(f"Results file not found: {args.results}")
        return 1

    try:
        # Load results
        with open(args.results, 'r') as f:
            results = json.load(f)

        # Analyze results
        analysis = analyze_results(results)

        # Print analysis
        print_analysis(analysis)

        # Save analysis if output file provided
        if args.output:
            output_file = args.output
        else:
            # Auto-generate output filename
            base_name = os.path.basename(args.results)
            output_file = os.path.join(os.path.dirname(args.results),
                                    base_name.replace("validation_results_", "validation_analysis_")
                                             .replace(".json", ".md"))

        save_analysis(analysis, output_file)
        logger.info(f"Analysis saved to {output_file}")

        return 0
    except Exception as e:
        logger.error(f"Error analyzing results: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

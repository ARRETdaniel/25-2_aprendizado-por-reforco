"""
Advanced validation runner for the CARLA environment wrapper.

This script provides a more advanced interface for running validation tests
on the CARLA environment wrapper with additional options.

Usage:
    python run_advanced_validation.py --tests all --iterations 5 --timeout 300
"""

import os
import sys
import argparse
import logging
import time
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path to import rl_environment module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validation modules
from environment_validator import EnvironmentValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("ValidationRunner")

# Define test categories
TEST_CATEGORIES = {
    "init": ["init_environment_creation", "init_carla_connection",
             "init_action_space", "init_observation_space"],
    "reset": ["reset_initial_state", "reset_vehicle_position",
              "reset_consistent_outputs"],
    "action": ["action_throttle_response", "action_steering_response",
               "action_continuous_validity"],
    "step": ["step_observation_format", "step_reward_range",
             "step_done_flag", "step_info_contents"],
    "reward": ["reward_progress_component", "reward_collision_penalty",
               "reward_consistency"],
    "termination": ["termination_collision_detection", "termination_goal_detection"],
    "cleanup": ["cleanup_resources"]
}

class ValidationRunner:
    """
    Advanced runner for environment validation tests.

    Features:
    - Selective test execution
    - Test iterations for stability checking
    - Timeout handling
    - Detailed reporting
    """

    def __init__(self,
                 carla_host: str = "localhost",
                 carla_port: int = 2000,
                 results_dir: str = None):
        """
        Initialize the validation runner.

        Args:
            carla_host: Host address for CARLA server
            carla_port: Port for CARLA server
            results_dir: Directory to store result files (default: same directory as script)
        """
        self.carla_host = carla_host
        self.carla_port = carla_port

        if results_dir is None:
            self.results_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.results_dir = results_dir
            os.makedirs(self.results_dir, exist_ok=True)

        self.validator = EnvironmentValidator(carla_host, carla_port)

    def _prepare_tests(self, test_selection: str) -> List[str]:
        """
        Prepare the list of tests to run based on selection.

        Args:
            test_selection: Which tests to run ('all', category name, or comma-separated list)

        Returns:
            List of test method names
        """
        if test_selection == "all":
            # Flatten all test categories
            return [test for category in TEST_CATEGORIES.values() for test in category]

        if test_selection in TEST_CATEGORIES:
            # Return tests for a specific category
            return TEST_CATEGORIES[test_selection]

        # Assume comma-separated list of test names
        return test_selection.split(",")

    def run_validation(self,
                       test_selection: str = "all",
                       iterations: int = 1,
                       timeout: int = 600,
                       check_carla: bool = True) -> Dict[str, Any]:
        """
        Run validation tests.

        Args:
            test_selection: Which tests to run ('all', category name, or comma-separated list)
            iterations: Number of times to run each test
            timeout: Maximum time in seconds for validation
            check_carla: Whether to check CARLA connection before running tests

        Returns:
            Dictionary with test results
        """
        start_time = time.time()

        # Check CARLA server connection
        if check_carla:
            logger.info(f"Checking CARLA connection at {self.carla_host}:{self.carla_port}")
            if not self.validator.check_carla_connection():
                logger.error("CARLA server not available. Please start CARLA before running tests.")
                return {
                    "summary": {
                        "timestamp": datetime.now().isoformat(),
                        "error": "CARLA server not available",
                        "overall_passed": False
                    }
                }

        # Prepare tests to run
        tests_to_run = self._prepare_tests(test_selection)
        logger.info(f"Running {len(tests_to_run)} tests with {iterations} iterations each")

        # Initialize results structure
        results = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "test_selection": test_selection,
                "iterations": iterations,
                "environment_info": {
                    "python_version": sys.version.split()[0],
                    "carla_host": self.carla_host,
                    "carla_port": self.carla_port
                }
            },
            "tests": {},
            "performance": {
                "reset_time": {},
                "step_time": {}
            }
        }

        # Run tests
        passed_count = 0
        total_tests = len(tests_to_run)

        for test_name in tests_to_run:
            if time.time() - start_time > timeout:
                logger.warning(f"Validation timed out after {timeout} seconds")
                break

            logger.info(f"Running test: {test_name}")

            # Track test results across iterations
            iteration_results = []

            for i in range(iterations):
                logger.debug(f"Iteration {i+1}/{iterations}")

                try:
                    # Get the test method from the validator
                    test_method = getattr(self.validator, test_name)
                    test_result = test_method()

                    # Store iteration result
                    iteration_results.append({
                        "passed": test_result.get("passed", False),
                        "details": test_result.get("details", {}),
                        "duration": test_result.get("duration", 0)
                    })

                except Exception as e:
                    logger.error(f"Error running test {test_name}: {e}", exc_info=True)
                    iteration_results.append({
                        "passed": False,
                        "details": {"error": str(e)},
                        "duration": 0
                    })

            # Aggregate results across iterations
            passed_iterations = sum(1 for res in iteration_results if res["passed"])
            all_passed = passed_iterations == iterations

            if all_passed:
                passed_count += 1

            # Store aggregated results
            results["tests"][test_name] = {
                "passed": all_passed,
                "details": {
                    "iterations_passed": passed_iterations,
                    "iterations_total": iterations,
                    "message": f"{passed_iterations}/{iterations} iterations passed" if all_passed else
                              iteration_results[-1]["details"].get("error", "Test failed in some iterations")
                },
                "duration": sum(res["duration"] for res in iteration_results) / len(iteration_results),
                "iterations": iteration_results
            }

        # Complete summary information
        duration = time.time() - start_time
        results["summary"].update({
            "duration": duration,
            "passed": passed_count,
            "failed": total_tests - passed_count,
            "total_tests": total_tests,
            "overall_passed": passed_count == total_tests
        })

        # Save results
        self._save_results(results)

        logger.info(f"Validation completed: {passed_count}/{total_tests} tests passed")
        return results

    def _save_results(self, results: Dict[str, Any]) -> str:
        """
        Save validation results to a JSON file.

        Args:
            results: Dictionary with test results

        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"validation_results_{timestamp}.json")

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        return results_file

def main():
    """Run the validation."""
    parser = argparse.ArgumentParser(description="CARLA Environment Advanced Validation Runner")
    parser.add_argument("--tests", type=str, default="all",
                      help="Tests to run: 'all', category name, or comma-separated list")
    parser.add_argument("--iterations", type=int, default=1,
                      help="Number of iterations for each test")
    parser.add_argument("--timeout", type=int, default=600,
                      help="Maximum time in seconds for validation")
    parser.add_argument("--host", type=str, default="localhost",
                      help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,
                      help="CARLA server port")
    parser.add_argument("--results-dir", type=str, default=None,
                      help="Directory to store result files")
    parser.add_argument("--no-check-carla", action="store_true",
                      help="Skip CARLA server connection check")
    parser.add_argument("--generate-report", action="store_true",
                      help="Generate report from results")
    parser.add_argument("--report-template", type=str, default=None,
                      help="Path to report template file")

    args = parser.parse_args()

    try:
        # Create and run validation
        runner = ValidationRunner(
            carla_host=args.host,
            carla_port=args.port,
            results_dir=args.results_dir
        )

        results = runner.run_validation(
            test_selection=args.tests,
            iterations=args.iterations,
            timeout=args.timeout,
            check_carla=not args.no_check_carla
        )

        # Generate report if requested
        if args.generate_report:
            try:
                # Import report generator
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from generate_validation_report import generate_report

                # Get the latest results file
                results_dir = args.results_dir or os.path.dirname(os.path.abspath(__file__))
                results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir)
                              if f.startswith("validation_results_") and f.endswith(".json")]
                latest_results = max(results_files, key=os.path.getmtime)

                # Use default template if none provided
                template_file = args.report_template
                if template_file is None:
                    template_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "validation_report_template.md")

                # Generate report
                report_file = generate_report(latest_results, template_file)
                logger.info(f"Report generated: {report_file}")
            except Exception as e:
                logger.error(f"Error generating report: {e}", exc_info=True)

        return 0 if results["summary"].get("overall_passed", False) else 1

    except Exception as e:
        logger.error(f"Error running validation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

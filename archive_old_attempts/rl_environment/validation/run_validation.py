"""
Command-line script to run the environment validator.

This script provides a simple way to execute the environment validation
tests from the command line.

Usage:
    python run_validation.py [--host HOST] [--port PORT] [--output-dir DIR]
    python -m rl_environment.validation.run_validation [--host HOST] [--port PORT] [--output-dir DIR]
"""

import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Conditional import to handle both direct execution and module execution
try:
    # First, try the relative import (when run directly)
    from environment_validator import EnvironmentValidator
except ImportError:
    try:
        # Then try the package import (when run as a module)
        from rl_environment.validation.environment_validator import EnvironmentValidator
    except ImportError as e:
        print(f"ERROR: Failed to import EnvironmentValidator: {e}")
        print("Make sure the rl_environment package is in your PYTHONPATH")
        sys.exit(1)

def main():
    """Execute environment validation tests."""
    import argparse

    parser = argparse.ArgumentParser(description="CARLA RL Environment Validator")
    parser.add_argument("--host", type=str, default="localhost",
                      help="CARLA server host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000,
                      help="CARLA server port (default: 2000)")
    parser.add_argument("--output-dir", type=str, default="./validation_results",
                      help="Output directory for validation results (default: ./validation_results)")

    args = parser.parse_args()

    # Create and run validator
    validator = EnvironmentValidator(
        output_dir=args.output_dir,
        host=args.host,
        port=args.port
    )

    success = validator.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

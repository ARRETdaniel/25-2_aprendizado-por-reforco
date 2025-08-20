#!/usr/bin/env python
"""
Launch script for CARLA testing.
This script will:
1. Start CARLA simulator
2. Run the test script
"""

import os
import sys
import time
import logging
import subprocess
import signal
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_carla_binary():
    """Find the CARLA binary in the expected paths."""
    base_dir = Path(__file__).parent.parent

    # Possible locations for CarlaUE4.exe
    possible_paths = [
        base_dir / "CarlaSimulator" / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4.exe",
        base_dir / "CarlaSimulator" / "CarlaUE4.exe",
        base_dir / "CarlaSimulator" / "WindowsNoEditor" / "CarlaUE4.exe"
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None

def start_carla(quality="Low", town="Town01", headless=False):
    """Start CARLA simulator."""
    carla_binary = find_carla_binary()

    if not carla_binary:
        logger.error("Could not find CARLA binary! Please check your installation.")
        return None

    logger.info(f"Starting CARLA simulator: {carla_binary}")

    cmd = [carla_binary]

    # Add quality setting
    if quality:
        cmd.append(f"-quality={quality}")

    # Add town setting
    if town:
        cmd.append(f"-map={town}")

    # Run headless if requested
    if headless:
        cmd.append("-RenderOffScreen")

    try:
        # Start CARLA as a subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Open in new window
        )

        # Give it some time to start
        logger.info("Waiting for CARLA to start...")
        time.sleep(10)

        # Check if process is still running
        if process.poll() is not None:
            logger.error("CARLA failed to start!")
            return None

        logger.info(f"CARLA started with PID: {process.pid}")
        return process

    except Exception as e:
        logger.error(f"Error starting CARLA: {e}")
        return None

def run_test_script(test_script):
    """Run the test script."""
    logger.info(f"Running test script: {test_script}")

    try:
        result = subprocess.run(
            [sys.executable, test_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        logger.info(f"Test script output:\n{result.stdout}")

        if result.stderr:
            logger.error(f"Test script errors:\n{result.stderr}")

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Error running test script: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='CARLA test launcher')

    parser.add_argument(
        '--quality',
        type=str,
        default="Low",
        choices=["Low", "Medium", "High", "Epic"],
        help='CARLA rendering quality'
    )

    parser.add_argument(
        '--town',
        type=str,
        default="Town01",
        help='CARLA town/map to load'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run CARLA in headless mode'
    )

    parser.add_argument(
        '--test',
        type=str,
        default="test_carla_connection.py",
        help='Test script to run'
    )

    args = parser.parse_args()

    # Start CARLA
    carla_process = start_carla(
        quality=args.quality,
        town=args.town,
        headless=args.headless
    )

    if not carla_process:
        logger.error("Failed to start CARLA. Exiting.")
        return 1

    try:
        # Run the test script
        test_script = os.path.join(os.path.dirname(__file__), args.test)
        success = run_test_script(test_script)

        if success:
            logger.info("Test completed successfully!")
        else:
            logger.error("Test failed!")

    finally:
        # Clean up: terminate CARLA
        logger.info("Terminating CARLA...")
        try:
            carla_process.terminate()
            carla_process.wait(timeout=10)
        except Exception as e:
            logger.error(f"Error terminating CARLA: {e}")
            try:
                carla_process.kill()
            except:
                pass

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

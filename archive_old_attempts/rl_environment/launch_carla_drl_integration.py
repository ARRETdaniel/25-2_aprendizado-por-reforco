#!/usr/bin/env python
"""
Launch script for CARLA DRL Integration.
This script will:
1. Start CARLA simulator
2. Run the CARLA sensor publisher
3. Run the test script
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

def run_sensor_publisher(host='localhost', port=2000, town='Town01', fps=20, steps=1000):
    """Run the sensor publisher."""
    logger.info("Starting CARLA sensor publisher...")

    script_path = os.path.join(os.path.dirname(__file__), "carla_sensor_publisher.py")

    try:
        cmd = [
            sys.executable,
            script_path,
            f"--host={host}",
            f"--port={port}",
            f"--town={town}",
            f"--fps={fps}",
            f"--steps={steps}"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Open in new window
        )

        logger.info(f"Sensor publisher started with PID: {process.pid}")
        return process

    except Exception as e:
        logger.error(f"Error starting sensor publisher: {e}")
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
    parser = argparse.ArgumentParser(description='CARLA DRL Integration')

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

    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Simulation frames per second'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of steps to run'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=2000,
        help='CARLA server port'
    )

    parser.add_argument(
        '--host',
        type=str,
        default="localhost",
        help='CARLA server host'
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
        # Start the sensor publisher
        publisher_process = run_sensor_publisher(
            host=args.host,
            port=args.port,
            town=args.town,
            fps=args.fps,
            steps=args.steps
        )

        if not publisher_process:
            logger.error("Failed to start sensor publisher. Exiting.")
            return 1

        # Wait a moment for the publisher to initialize
        time.sleep(5)

        # Run the test script
        test_script = os.path.join(os.path.dirname(__file__), args.test)
        success = run_test_script(test_script)

        if success:
            logger.info("Test completed successfully!")
        else:
            logger.error("Test failed!")

        # Wait for the sensor publisher to finish
        logger.info("Waiting for sensor publisher to complete...")
        publisher_process.wait(timeout=60)

        return 0 if success else 1

    finally:
        # Clean up processes
        logger.info("Cleaning up processes...")

        if 'publisher_process' in locals() and publisher_process:
            logger.info("Terminating sensor publisher...")
            try:
                publisher_process.terminate()
                publisher_process.wait(timeout=10)
            except Exception as e:
                logger.error(f"Error terminating sensor publisher: {e}")
                try:
                    publisher_process.kill()
                except:
                    pass

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

if __name__ == "__main__":
    sys.exit(main())

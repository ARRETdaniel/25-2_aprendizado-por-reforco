#!/usr/bin/env python
"""
Start CARLA server and run validation tests.

This script starts the CARLA server, waits for it to initialize,
and then runs the validation tests.
"""

import os
import sys
import time
import logging
import subprocess
import signal
import atexit
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_carla_executable():
    """
    Find the CARLA executable in the workspace.

    Returns:
        Path to CARLA executable or None if not found
    """
    base_path = Path(__file__).parent.parent
    carla_path = base_path / "CarlaSimulator"

    # Look for Windows executable
    carla_exe = carla_path / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4.exe"

    if carla_exe.exists():
        logger.info(f"Found CARLA executable at {carla_exe}")
        return str(carla_exe)

    logger.error("Could not find CARLA executable")
    return None

def start_carla_server(carla_exe, port=2000, low_quality=True):
    """
    Start the CARLA server.

    Args:
        carla_exe: Path to CARLA executable
        port: Port to start CARLA on
        low_quality: Whether to start in low quality mode

    Returns:
        Process object if server started, None otherwise
    """
    try:
        cmd = [carla_exe, "-carla-server", f"-carla-world-port={port}"]

        if low_quality:
            cmd.extend(["-quality-level=Low"])

        logger.info(f"Starting CARLA server with command: {' '.join(cmd)}")

        # Start CARLA without creating a window (headless)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        # Register cleanup function to kill CARLA when script exits
        def cleanup():
            if process.poll() is None:
                logger.info("Terminating CARLA server...")
                process.terminate()
                process.wait(timeout=10)

        atexit.register(cleanup)

        # Wait for server to start
        logger.info("Waiting for CARLA server to initialize...")
        time.sleep(10)  # Give CARLA time to start

        if process.poll() is not None:
            logger.error("CARLA server failed to start")
            return None

        logger.info("CARLA server started successfully")
        return process

    except Exception as e:
        logger.error(f"Error starting CARLA server: {e}")
        return None

def run_validation_tests():
    """
    Run the validation tests.

    Returns:
        True if validation succeeded, False otherwise
    """
    try:
        logger.info("Running validation tests...")
        validation_cmd = [sys.executable, "-m", "validation.run_validation"]

        # Run validation in the current process
        result = subprocess.run(
            validation_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False
        )

        success = result.returncode == 0
        logger.info(f"Validation {'succeeded' if success else 'failed'}")
        return success

    except Exception as e:
        logger.error(f"Error running validation tests: {e}")
        return False

def main():
    """
    Main function to start CARLA server and run validation.
    """
    carla_exe = find_carla_executable()
    if not carla_exe:
        logger.error("Cannot proceed without CARLA executable")
        return False

    # Start CARLA server
    carla_process = start_carla_server(carla_exe)
    if not carla_process:
        logger.error("Failed to start CARLA server")
        return False

    try:
        # Run validation tests
        success = run_validation_tests()

        # Keep CARLA running for 5 seconds to allow for graceful shutdown
        time.sleep(5)

        return success

    finally:
        # Terminate CARLA server
        if carla_process.poll() is None:
            logger.info("Terminating CARLA server...")
            carla_process.terminate()
            try:
                carla_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("CARLA did not terminate gracefully, forcing kill...")
                carla_process.kill()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

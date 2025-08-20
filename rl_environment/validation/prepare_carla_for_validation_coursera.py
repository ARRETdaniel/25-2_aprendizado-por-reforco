"""
CARLA Simulator Launch Script for RL Environment Testing

This script launches the CARLA simulator and sets it up for RL environment validation.
It handles starting the simulator with appropriate settings and ensuring it's ready
for the validation tests to connect.

This version is specifically adapted for the Coursera modified CARLA 0.8.4 version.

Usage:
    python prepare_carla_for_validation_coursera.py
    python prepare_carla_for_validation_coursera.py --check-only
    python prepare_carla_for_validation_coursera.py --map /Game/Maps/Town02 --fps 60

Author: Autonomous Driving Research Team
Date: August 15, 2025
"""

import os
import sys
import time
import logging
import subprocess
import argparse
import signal
import socket
import platform
from typing import Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("CARLA_Setup")

def find_carla_executable() -> Optional[str]:
    """
    Find the CARLA executable in the project directory structure.

    Returns:
        Path to the CARLA executable or None if not found
    """
    # Get the root directory of the project
    current_dir = Path(os.path.abspath(__file__))
    project_root = current_dir.parent.parent.parent

    # Path structures to check
    if platform.system() == "Windows":
        possible_paths = [
            # Path for Coursera modified CARLA 0.8.4 (Windows)
            project_root / "CarlaSimulator" / "CarlaUE4.exe",

            # Standard CARLA paths
            project_root / "CarlaSimulator" / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4.exe",
            project_root / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4.exe",

            # CARLA installation paths
            Path("C:/CARLA_0.8.4") / "CarlaUE4.exe"
        ]
    else:  # Linux
        possible_paths = [
            # Path for Coursera modified CARLA 0.8.4 (Linux)
            project_root / "CarlaSimulator" / "CarlaUE4.sh",

            # Standard CARLA paths
            project_root / "CarlaUE4" / "CarlaUE4.sh",

            # CARLA installation paths
            Path("/opt/carla") / "CarlaUE4.sh"
        ]

    # Check each path
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found CARLA executable at: {path}")
            return str(path)

    # If not found in any of the expected locations
    return None

def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use.

    Args:
        port: Port number to check

    Returns:
        True if the port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_carla_server(port: int, timeout: int = 60) -> bool:
    """
    Wait for the CARLA server to start and listen on the specified port.

    Args:
        port: Port to check
        timeout: Maximum time to wait in seconds

    Returns:
        True if the server started successfully, False otherwise
    """
    start_time = time.time()

    logger.info(f"Waiting for CARLA server to start on port {port}...")

    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            logger.info(f"CARLA server is running on port {port}")
            return True
        time.sleep(1)

    logger.error(f"CARLA server did not start within {timeout} seconds")
    return False

def launch_carla(
    executable_path: str,
    port: int = 2000,
    quality_level: str = "Low",
    windowed: bool = True,
    width: int = 800,
    height: int = 600,
    map_name: str = "/Game/Maps/Course4",
    fps: int = 30,
    benchmark: bool = True
) -> Optional[subprocess.Popen]:
    """
    Launch the CARLA simulator with specified settings.

    Args:
        executable_path: Path to CARLA executable
        port: Port for CARLA server
        quality_level: Graphics quality level ("Low" or "Epic")
        windowed: Whether to run in windowed mode
        width: Window width
        height: Window height
        map_name: Name of the map to load (e.g., "/Game/Maps/Course4")
        fps: Target frames per second
        benchmark: Whether to use benchmark mode

    Returns:
        Subprocess object for the CARLA process, or None if launch failed
    """
    try:
        # Build command line arguments for Coursera modified CARLA 0.8.4
        args = [executable_path]

        # Add map name (must come first for CARLA 0.8.4)
        if map_name:
            args.append(map_name)

        # Add standard CARLA arguments
        if windowed:
            args.append("-windowed")

        # Essential for CARLA 0.8.4
        args.append("-carla-server")

        # Additional arguments
        if benchmark:
            args.append("-benchmark")

        args.append(f"-fps={fps}")

        # Quality level (using appropriate format)
        if quality_level == "Low":
            args.append("-quality-level=Low")

        logger.info(f"Launching CARLA with: {' '.join(args)}")

        # Launch CARLA process
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Wait for CARLA to start
        if wait_for_carla_server(port):
            logger.info("CARLA simulator started successfully")
            return process
        else:
            logger.error("Failed to start CARLA simulator")
            if process.poll() is None:
                process.terminate()
            return None

    except Exception as e:
        logger.error(f"Error launching CARLA: {e}")
        return None

def run_validation_tests(
    host: str = "localhost",
    port: int = 2000,
    output_dir: str = "./validation_results"
) -> int:
    """
    Run the environment validation tests.

    Args:
        host: CARLA server host
        port: CARLA server port
        output_dir: Output directory for validation results

    Returns:
        Exit code from the validation tests
    """
    try:
        # Get script directory and project root
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build the command to run validation tests directly
        cmd = [
            sys.executable,  # Current Python interpreter
            os.path.join(script_dir, "run_validation.py"),  # Direct script path
            f"--host={host}",
            f"--port={port}",
            f"--output-dir={output_dir}"
        ]

        logger.info(f"Running validation tests: {' '.join(cmd)}")

        # Add current directory to PYTHONPATH to ensure imports work
        env = os.environ.copy()
        python_path = env.get('PYTHONPATH', '')
        project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
        if python_path:
            env['PYTHONPATH'] = f"{project_root}{os.pathsep}{python_path}"
        else:
            env['PYTHONPATH'] = project_root

        # Run validation tests with the enhanced environment
        result = subprocess.run(cmd, check=False, env=env)

        return result.returncode

    except Exception as e:
        logger.error(f"Error running validation tests: {e}")
        return 1

def main():
    """Main function to launch CARLA and run validation tests."""
    parser = argparse.ArgumentParser(description="Launch CARLA and run RL environment validation tests")

    parser.add_argument("--port", type=int, default=2000,
                      help="Port for CARLA server (default: 2000)")
    parser.add_argument("--quality", type=str, choices=["Low", "Epic"], default="Low",
                      help="Graphics quality level (default: Low)")
    parser.add_argument("--no-window", action="store_true",
                      help="Run CARLA in headless mode")
    parser.add_argument("--width", type=int, default=800,
                      help="Window width (default: 800)")
    parser.add_argument("--height", type=int, default=600,
                      help="Window height (default: 600)")
    parser.add_argument("--carla-path", type=str, default=None,
                      help="Path to CARLA executable (will try to find automatically if not provided)")
    parser.add_argument("--output-dir", type=str, default="./validation_results",
                      help="Output directory for validation results (default: ./validation_results)")
    parser.add_argument("--skip-tests", action="store_true",
                      help="Skip running validation tests after launching CARLA")
    parser.add_argument("--map", type=str, default="/Game/Maps/Course4",
                      help="Name of the map to load (default: /Game/Maps/Course4)")
    parser.add_argument("--fps", type=int, default=30,
                      help="Target frames per second (default: 30)")
    parser.add_argument("--check-only", action="store_true",
                      help="Only check if CARLA server is running, don't start it")

    args = parser.parse_args()

    # If check-only is specified, just check if CARLA is running
    if args.check_only:
        if is_port_in_use(args.port):
            logger.info(f"CARLA server is already running on port {args.port}")
            return 0
        else:
            logger.info(f"CARLA server is not running on port {args.port}")
            return 1

    # Find CARLA executable
    carla_path = args.carla_path
    if carla_path is None:
        carla_path = find_carla_executable()
        if carla_path is None:
            logger.error("Could not find CARLA executable. Please provide the path with --carla-path")
            return 1

    # Check if port is already in use
    if is_port_in_use(args.port):
        logger.error(f"Port {args.port} is already in use. Please choose a different port or close the existing process.")
        return 1

    # Launch CARLA
    process = launch_carla(
        executable_path=carla_path,
        port=args.port,
        quality_level=args.quality,
        windowed=not args.no_window,
        width=args.width,
        height=args.height,
        map_name=args.map,
        fps=args.fps,
        benchmark=True  # Always use benchmark mode for consistent timing
    )

    if process is None:
        return 1

    try:
        if not args.skip_tests:
            # Run validation tests
            test_result = run_validation_tests(
                port=args.port,
                output_dir=args.output_dir
            )

            if test_result == 0:
                logger.info("Validation tests passed successfully")
            else:
                logger.error(f"Validation tests failed with exit code {test_result}")

            return test_result
        else:
            logger.info("CARLA is running. Press Ctrl+C to exit.")
            # Wait for Ctrl+C
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping CARLA...")
    finally:
        # Terminate CARLA process
        if process.poll() is None:
            logger.info("Terminating CARLA process...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("CARLA did not terminate gracefully, forcing...")
                process.kill()

    return 0

if __name__ == "__main__":
    sys.exit(main())

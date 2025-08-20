#!/usr/bin/env python
"""
CARLA DRL Camera Visualization Launcher

This script will start the CARLA simulator and then run our
DRL training with camera visualization.
"""

import os
import sys
import argparse
import subprocess
import time
import logging
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch CARLA and DRL visualizer")

    # CARLA settings
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--quality', default='Low', choices=['Low', 'Epic'],
                        help='CARLA graphics quality')

    # Camera settings
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')

    # Training settings
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--sync', action='store_true', help='Use synchronous mode')

    return parser.parse_args()

def find_carla_executable():
    """Find the CARLA executable path."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Default CARLA paths based on OS
    if platform.system() == "Windows":
        carla_path = project_root / "CarlaSimulator" / "CarlaUE4" / "Binaries" / "Win64" / "CarlaUE4.exe"
    else:  # Linux/Mac
        carla_path = project_root / "CarlaSimulator" / "CarlaUE4.sh"

    if carla_path.exists():
        logger.info(f"Found CARLA executable at {carla_path}")
        return str(carla_path)
    else:
        # Try alternative location
        if platform.system() == "Windows":
            alt_path = project_root / "CarlaSimulator" / "CarlaUE4.exe"
            if alt_path.exists():
                logger.info(f"Found CARLA executable at {alt_path}")
                return str(alt_path)

        logger.warning(f"CARLA executable not found at {carla_path}")
        return None

def start_carla_server(carla_path, port=2000, quality="Low"):
    """Start the CARLA server process."""
    if not carla_path:
        logger.error("CARLA executable path not provided")
        return None

    # Build the command
    if platform.system() == "Windows":
        cmd = [
            carla_path,
            "-carla-server",
            f"-carla-world-port={port}",
            f"-quality-level={quality}"
        ]
    else:  # Linux/Mac
        cmd = [
            "bash", carla_path,
            "-carla-server",
            f"-carla-world-port={port}",
            f"-quality-level={quality}"
        ]

    logger.info(f"Starting CARLA server with command: {' '.join(cmd)}")

    try:
        # Start CARLA as a subprocess
        process = subprocess.Popen(cmd)

        # Wait for CARLA to initialize
        logger.info("Waiting for CARLA server to initialize...")
        time.sleep(10)  # Wait for server to start

        if process.poll() is not None:
            # Process has terminated
            logger.error(f"CARLA server failed to start. Return code: {process.returncode}")
            return None

        logger.info("CARLA server started successfully")
        return process
    except Exception as e:
        logger.error(f"Error starting CARLA server: {e}")
        return None

def start_drl_visualizer(args):
    """Start the DRL visualizer script."""
    # Path to visualizer script
    visualizer_path = Path(__file__).parent / "carla_drl_visualizer.py"

    if not visualizer_path.exists():
        logger.error(f"Visualizer script not found at {visualizer_path}")
        return None

    # Build command
    cmd = [
        sys.executable,  # Current Python interpreter
        str(visualizer_path),
        f"--host={args.host}",
        f"--port={args.port}",
        f"--width={args.width}",
        f"--height={args.height}",
        f"--episodes={args.episodes}"
    ]

    if args.sync:
        cmd.append("--sync")

    logger.info(f"Starting DRL visualizer with command: {' '.join(cmd)}")

    try:
        # Start visualizer as a subprocess
        process = subprocess.Popen(cmd)
        logger.info("DRL visualizer started successfully")
        return process
    except Exception as e:
        logger.error(f"Error starting DRL visualizer: {e}")
        return None

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Find CARLA executable
    carla_path = find_carla_executable()
    if not carla_path:
        logger.error("CARLA executable not found. Please install CARLA following instructions in CARLA_SETUP_INSTRUCTIONS.py")
        return 1

    # Start CARLA server
    carla_process = start_carla_server(carla_path, args.port, args.quality)
    if not carla_process:
        logger.error("Failed to start CARLA server")
        return 1

    try:
        # Start DRL visualizer
        visualizer_process = start_drl_visualizer(args)
        if not visualizer_process:
            logger.error("Failed to start DRL visualizer")
            carla_process.terminate()
            return 1

        # Wait for visualizer to finish
        visualizer_process.wait()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping processes...")
    finally:
        # Clean up processes
        logger.info("Stopping CARLA server...")
        if 'carla_process' in locals() and carla_process is not None:
            carla_process.terminate()
            carla_process.wait(timeout=10)

        logger.info("Stopping DRL visualizer...")
        if 'visualizer_process' in locals() and visualizer_process is not None and visualizer_process.poll() is None:
            visualizer_process.terminate()
            visualizer_process.wait(timeout=10)

    logger.info("All processes stopped. Exiting.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

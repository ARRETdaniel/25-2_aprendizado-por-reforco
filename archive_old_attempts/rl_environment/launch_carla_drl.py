#!/usr/bin/env python
"""
CARLA launcher script.

This script starts the CARLA server and the DRL visualizer.
"""

import os
import sys
import time
import logging
import subprocess
import signal
import argparse
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            carla_path,
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

def start_drl_visualizer(host='localhost', port=2000, width=640, height=480, episodes=100, sync=False):
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
        f"--host={host}",
        f"--port={port}",
        f"--width={width}",
        f"--height={height}",
        f"--episodes={episodes}"
    ]

    if sync:
        cmd.append("--sync")

    logger.info(f"Starting DRL visualizer with command: {' '.join(cmd)}")

    try:
        # Start the visualizer as a subprocess
        process = subprocess.Popen(cmd)

        logger.info("DRL visualizer started")
        return process
    except Exception as e:
        logger.error(f"Error starting DRL visualizer: {e}")
        return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CARLA with DRL launcher")

    # CARLA server options
    parser.add_argument('--carla-path', help='Path to CARLA executable')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--quality', choices=['Low', 'Medium', 'High', 'Epic'], default='Low',
                      help='CARLA quality level')

    # Visualizer options
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')

    # Other options
    parser.add_argument('--visualizer-only', action='store_true',
                      help='Only start the visualizer (assume CARLA is already running)')

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()

    carla_process = None
    visualizer_process = None

    try:
        # Start CARLA server if needed
        if not args.visualizer_only:
            carla_path = args.carla_path if args.carla_path else find_carla_executable()
            if not carla_path:
                logger.error("CARLA executable not found")
                return False

            carla_process = start_carla_server(carla_path, args.port, args.quality)
            if not carla_process:
                return False

        # Start DRL visualizer
        visualizer_process = start_drl_visualizer(
            port=args.port,
            width=args.width,
            height=args.height,
            episodes=args.episodes,
            sync=args.sync
        )

        if not visualizer_process:
            return False

        # Wait for visualizer to finish
        visualizer_process.wait()

        logger.info("DRL training completed")
        return True

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return True
    finally:
        # Clean up processes
        if visualizer_process and visualizer_process.poll() is None:
            logger.info("Terminating visualizer process...")
            visualizer_process.terminate()
            visualizer_process.wait()

        if carla_process and carla_process.poll() is None:
            logger.info("Terminating CARLA server...")
            carla_process.terminate()
            carla_process.wait()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

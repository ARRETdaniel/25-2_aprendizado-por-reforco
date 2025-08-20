"""
CARLA DRL Launcher

This script launches all the components needed for DRL training with CARLA:
1. CARLA simulator (if not already running)
2. CARLA camera visualizer with ROS 2 bridge (Python 3.6)
3. DRL trainer (Python 3.12)

Usage:
    python run_carla_drl.py --quality Low --episodes 100
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import signal
import atexit
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CARLA_SERVER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'CarlaSimulator', 'CarlaUE4', 'Binaries', 'Win64', 'CarlaUE4.exe')
PYTHON36_PATH = 'python'  # Assuming Python 3.6 is in PATH as 'python'
PYTHON312_PATH = 'python'  # Assuming Python 3.12 is in PATH as 'python'


class ProcessManager:
    """Manages subprocesses and ensures they are properly terminated."""

    def __init__(self):
        """Initialize the process manager."""
        self.processes = []
        atexit.register(self.cleanup)

    def start_process(self, args, **kwargs):
        """Start a subprocess.

        Args:
            args: Arguments for subprocess.Popen
            **kwargs: Additional keyword arguments for subprocess.Popen

        Returns:
            Subprocess object
        """
        try:
            process = subprocess.Popen(args, **kwargs)
            self.processes.append(process)
            return process
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            return None

    def cleanup(self):
        """Clean up all subprocesses."""
        logger.info("Cleaning up subprocesses")

        for process in self.processes:
            try:
                if process.poll() is None:
                    logger.info(f"Terminating process {process.pid}")
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Failed to terminate process: {e}")
                try:
                    process.kill()
                except:
                    pass


def is_carla_running():
    """Check if CARLA is already running.

    Returns:
        bool: True if CARLA is running, False otherwise
    """
    try:
        # Check for CarlaUE4 process
        output = subprocess.check_output('tasklist /FI "IMAGENAME eq CarlaUE4.exe"', shell=True)
        return b'CarlaUE4.exe' in output
    except:
        return False


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CARLA DRL Launcher")
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--timeout', default=10, type=int, help='CARLA server timeout')
    parser.add_argument('--quality', default='Low', choices=['Low', 'Epic'], help='Graphics quality')
    parser.add_argument('--start-pos', default=0, type=int, help='Player start position')
    parser.add_argument('--vehicles', default=0, type=int, help='Number of vehicles')
    parser.add_argument('--pedestrians', default=0, type=int, help='Number of pedestrians')
    parser.add_argument('--weather', default=0, type=int, help='Weather preset ID')
    parser.add_argument('--episodes', default=100, type=int, help='Number of episodes')
    parser.add_argument('--max-steps', default=1000, type=int, help='Maximum steps per episode')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--no-ros', action='store_true', help='Disable ROS bridge (use file-based communication)')
    parser.add_argument('--no-display', action='store_true', help='Disable OpenCV visualization')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agent instead of training')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--no-server', action='store_true', help='Do not start CARLA server')

    args = parser.parse_args()

    # Initialize process manager
    process_manager = ProcessManager()

    try:
        # Start CARLA server if not already running and not explicitly disabled
        carla_server_process = None
        if not is_carla_running() and not args.no_server:
            logger.info("Starting CARLA server")
            carla_server_args = [CARLA_SERVER_PATH]

            if args.quality == 'Low':
                carla_server_args.append('-quality-level=Low')

            carla_server_process = process_manager.start_process(
                carla_server_args,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

            if carla_server_process is None:
                logger.error("Failed to start CARLA server")
                return 1

            logger.info("Waiting for CARLA server to start")
            time.sleep(5)  # Give CARLA time to start

        # Start CARLA camera visualizer with ROS 2 bridge (Python 3.6)
        logger.info("Starting CARLA camera visualizer")
        visualizer_args = [
            PYTHON36_PATH,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carla_camera_visualizer_ros.py'),
            f'--host={args.host}',
            f'--port={args.port}',
            f'--timeout={args.timeout}',
            f'--quality={args.quality}',
            f'--start-pos={args.start_pos}',
            f'--vehicles={args.vehicles}',
            f'--pedestrians={args.pedestrians}',
            f'--weather={args.weather}'
        ]

        if args.no_ros:
            visualizer_args.append('--no-ros')

        if args.no_display:
            visualizer_args.append('--no-display')

        visualizer_process = process_manager.start_process(
            visualizer_args,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        if visualizer_process is None:
            logger.error("Failed to start CARLA camera visualizer")
            return 1

        logger.info("Waiting for CARLA camera visualizer to initialize")
        time.sleep(2)  # Give visualizer time to connect to CARLA

        # Start DRL trainer (Python 3.12)
        logger.info("Starting DRL trainer")
        trainer_args = [
            PYTHON312_PATH,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carla_drl_trainer.py'),
            f'--episodes={args.episodes}',
            f'--max-steps={args.max_steps}',
            f'--seed={args.seed}'
        ]

        if args.checkpoint:
            trainer_args.append(f'--checkpoint={args.checkpoint}')

        if args.no_ros:
            trainer_args.append('--no-ros')

        if args.no_display:
            trainer_args.append('--no-render')

        if args.evaluate:
            trainer_args.append('--evaluate')

        trainer_process = process_manager.start_process(
            trainer_args,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        if trainer_process is None:
            logger.error("Failed to start DRL trainer")
            return 1

        # Wait for trainer to finish
        logger.info("All components started, waiting for training to complete")
        trainer_process.wait()

        # When the trainer finishes, clean up everything else
        process_manager.cleanup()

        return 0

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, stopping")
        process_manager.cleanup()
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        process_manager.cleanup()
        return 1


if __name__ == '__main__':
    sys.exit(main())

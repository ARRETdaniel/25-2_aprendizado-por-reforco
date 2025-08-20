"""
Setup Script for CARLA DRL with ROS 2

This script sets up the environment for training a DRL agent with CARLA.
It checks for dependencies and installs them if necessary.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_command(command):
    """Check if a command is available.

    Args:
        command: Command to check

    Returns:
        bool: True if the command is available, False otherwise
    """
    try:
        subprocess.check_call([command, '--version'],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def install_dependencies_py36():
    """Install Python 3.6 dependencies."""
    logger.info("Installing Python 3.6 dependencies")

    requirements = [
        'numpy',
        'opencv-python',
        'matplotlib',
        'pillow'
    ]

    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            logger.info(f"Successfully installed {package}")
        except subprocess.SubprocessError:
            logger.error(f"Failed to install {package}")


def install_dependencies_py312():
    """Install Python 3.12 dependencies."""
    logger.info("Installing Python 3.12 dependencies")

    requirements = [
        'numpy',
        'torch',
        'opencv-python',
        'matplotlib',
        'pillow',
        'dataclasses'
    ]

    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            logger.info(f"Successfully installed {package}")
        except subprocess.SubprocessError:
            logger.error(f"Failed to install {package}")


def install_ros2_windows():
    """Install ROS 2 on Windows."""
    logger.info("Installing ROS 2 on Windows")
    logger.info("Please follow the instructions at: https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html")
    logger.info("After installing ROS 2, run the following commands:")
    logger.info("    pip install -U rosdep")
    logger.info("    rosdep init")
    logger.info("    rosdep update")
    logger.info("    pip install -U cv_bridge")
    logger.info("    pip install -U sensor_msgs")
    logger.info("    pip install -U geometry_msgs")
    logger.info("    pip install -U std_msgs")
    logger.info("    pip install -U rclpy")

    # Check if installation instructions are understood
    understood = input("Have you installed ROS 2 following the instructions above? (y/N): ")
    return understood.lower() in ['y', 'yes']


def setup_fallback_communication():
    """Set up fallback communication mechanism."""
    logger.info("Setting up fallback communication mechanism")

    # Create directory for file-based communication
    comm_dir = Path.home() / ".carla_drl_bridge"
    comm_dir.mkdir(exist_ok=True)

    logger.info(f"Created directory for file-based communication: {comm_dir}")


def setup_carla_environment():
    """Set up CARLA environment."""
    logger.info("Setting up CARLA environment")

    # Check if CARLA is installed
    carla_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'CarlaSimulator')

    if not os.path.exists(carla_path):
        logger.error(f"CARLA not found at {carla_path}")
        return False

    # Check if Python API is present
    python_api_path = os.path.join(carla_path, 'PythonAPI')
    if not os.path.exists(python_api_path):
        logger.error(f"CARLA Python API not found at {python_api_path}")
        return False

    logger.info("CARLA environment is set up")
    return True


def main():
    """Main function."""
    logger.info("Setting up CARLA DRL with ROS 2")

    # Check Python version
    python_version = platform.python_version_tuple()
    logger.info(f"Python version: {platform.python_version()}")

    # Determine which dependencies to install based on Python version
    if python_version[0] == '3' and int(python_version[1]) == 6:
        install_dependencies_py36()
    elif python_version[0] == '3' and int(python_version[1]) >= 10:
        install_dependencies_py312()
    else:
        logger.warning(f"Unexpected Python version {platform.python_version()}")
        logger.warning("CARLA requires Python 3.6, and the DRL trainer works best with Python 3.10+")

        # Ask whether to continue
        continue_anyway = input("Do you want to continue anyway? (y/N): ")
        if continue_anyway.lower() not in ['y', 'yes']:
            return 1

    # Check and set up CARLA environment
    if not setup_carla_environment():
        logger.error("Failed to set up CARLA environment")
        return 1

    # Check if ROS 2 is installed
    if not check_command('ros2'):
        logger.warning("ROS 2 not found")

        # Ask whether to install ROS 2
        install_ros = input("Do you want to install ROS 2? (y/N): ")
        if install_ros.lower() in ['y', 'yes']:
            if platform.system() == 'Windows':
                if not install_ros2_windows():
                    logger.warning("ROS 2 installation not completed")
                    setup_fallback_communication()
            else:
                logger.error(f"ROS 2 installation on {platform.system()} not supported by this script")
                setup_fallback_communication()
        else:
            setup_fallback_communication()
    else:
        logger.info("ROS 2 is already installed")

    logger.info("Setup complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())

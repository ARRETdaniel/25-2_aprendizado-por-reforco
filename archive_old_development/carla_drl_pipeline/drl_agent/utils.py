"""
Utility Functions for CARLA DRL Pipeline

This module provides common utility functions for configuration management,
logging setup, file operations, and other helper functions used throughout
the pipeline.
"""

import os
import sys
import yaml
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import psutil
from datetime import datetime, timedelta
import platform


def setup_logging(log_file: Optional[Path] = None, 
                 level: str = 'INFO',
                 format_string: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete - Level: {level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return root_logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration loaded from {config_path}")
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration saved to {config_path}")


def create_directories(base_path: Path):
    """Create standard directory structure for experiments.
    
    Args:
        base_path: Base experiment directory
    """
    directories = [
        'checkpoints',
        'models',
        'logs',
        'plots',
        'configs',
        'data',
        'tensorboard',
        'videos'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Directory structure created in {base_path}")


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements for CARLA DRL pipeline.
    
    Returns:
        System information dictionary
    """
    logger = logging.getLogger(__name__)
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'disk_free_gb': round(shutil.disk_usage('.').free / (1024**3), 2),
        'torch_available': False,
        'cuda_available': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpu_memory_gb': 0
    }
    
    # Check PyTorch
    try:
        import torch
        system_info['torch_available'] = True
        system_info['torch_version'] = torch.__version__
        
        if torch.cuda.is_available():
            system_info['cuda_available'] = True
            system_info['cuda_version'] = torch.version.cuda
            system_info['gpu_count'] = torch.cuda.device_count()
            
            if system_info['gpu_count'] > 0:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                system_info['gpu_memory_gb'] = round(gpu_memory / (1024**3), 2)
    except ImportError:
        logger.warning("PyTorch not available")
    
    # Check recommended requirements
    warnings = []
    
    if system_info['memory_gb'] < 8:
        warnings.append("Recommended: At least 8GB RAM")
    
    if system_info['disk_free_gb'] < 50:
        warnings.append("Recommended: At least 50GB free disk space")
    
    if not system_info['cuda_available']:
        warnings.append("Recommended: CUDA-compatible GPU for training")
    
    if system_info['gpu_memory_gb'] < 4 and system_info['cuda_available']:
        warnings.append("Recommended: At least 4GB GPU memory")
    
    system_info['warnings'] = warnings
    
    # Log system information
    logger.info("System Requirements Check:")
    logger.info(f"  Platform: {system_info['platform']}")
    logger.info(f"  Python: {system_info['python_version']}")
    logger.info(f"  CPU Cores: {system_info['cpu_count']}")
    logger.info(f"  Memory: {system_info['memory_gb']} GB")
    logger.info(f"  PyTorch: {system_info.get('torch_version', 'Not available')}")
    logger.info(f"  CUDA: {system_info['cuda_available']}")
    if system_info['cuda_available']:
        logger.info(f"  GPU Count: {system_info['gpu_count']}")
        logger.info(f"  GPU Memory: {system_info['gpu_memory_gb']} GB")
    
    for warning in warnings:
        logger.warning(warning)
    
    return system_info


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available.
    
    Returns:
        Dictionary of dependency availability
    """
    logger = logging.getLogger(__name__)
    
    dependencies = {
        'torch': False,
        'numpy': False,
        'opencv': False,
        'yaml': False,
        'zmq': False,
        'msgpack': False,
        'matplotlib': False,
        'gym': False,
        'psutil': False,
        'pydantic': False,
        'rclpy': False,
        'cv_bridge': False
    }
    
    # Check each dependency
    packages = {
        'torch': 'torch',
        'numpy': 'numpy',
        'opencv': 'cv2',
        'yaml': 'yaml',
        'zmq': 'zmq',
        'msgpack': 'msgpack',
        'matplotlib': 'matplotlib',
        'gym': 'gym',
        'psutil': 'psutil',
        'pydantic': 'pydantic',
        'rclpy': 'rclpy',
        'cv_bridge': 'cv_bridge'
    }
    
    for dep_name, module_name in packages.items():
        try:
            __import__(module_name)
            dependencies[dep_name] = True
        except ImportError:
            dependencies[dep_name] = False
    
    # Log results
    logger.info("Dependency Check:")
    for dep_name, available in dependencies.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {dep_name}")
    
    # Check critical dependencies
    critical_deps = ['torch', 'numpy', 'opencv', 'yaml', 'zmq', 'msgpack']
    missing_critical = [dep for dep in critical_deps if not dependencies[dep]]
    
    if missing_critical:
        logger.error(f"Missing critical dependencies: {missing_critical}")
        return dependencies
    
    # Check optional dependencies
    optional_deps = ['rclpy', 'cv_bridge']
    missing_optional = [dep for dep in optional_deps if not dependencies[dep]]
    
    if missing_optional:
        logger.warning(f"Missing optional dependencies (ROS2 features disabled): {missing_optional}")
    
    return dependencies


def check_carla_connection(host: str = 'localhost', port: int = 2000, timeout: float = 5.0) -> bool:
    """Check if CARLA server is running and accessible.
    
    Args:
        host: CARLA server host
        port: CARLA server port
        timeout: Connection timeout
        
    Returns:
        True if CARLA is accessible
    """
    logger = logging.getLogger(__name__)
    
    try:
        import socket
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info(f"CARLA server accessible at {host}:{port}")
            return True
        else:
            logger.warning(f"CARLA server not accessible at {host}:{port}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to check CARLA connection: {e}")
        return False


def monitor_system_resources(log_interval: float = 60.0) -> Dict[str, float]:
    """Monitor system resources during training.
    
    Args:
        log_interval: Logging interval in seconds
        
    Returns:
        Current resource usage
    """
    logger = logging.getLogger(__name__)
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # Disk usage
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # GPU usage (if available)
        gpu_percent = 0
        gpu_memory_percent = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        except:
            pass
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_gb': round(memory_available_gb, 2),
            'disk_percent': disk_percent,
            'disk_free_gb': round(disk_free_gb, 2),
            'gpu_memory_percent': gpu_memory_percent,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log if resources are high
        if (cpu_percent > 90 or memory_percent > 90 or 
            disk_percent > 95 or gpu_memory_percent > 90):
            logger.warning(f"High resource usage - CPU: {cpu_percent:.1f}%, "
                          f"Memory: {memory_percent:.1f}%, "
                          f"Disk: {disk_percent:.1f}%, "
                          f"GPU Memory: {gpu_memory_percent:.1f}%")
        
        return resources
        
    except Exception as e:
        logger.error(f"Failed to monitor system resources: {e}")
        return {}


def cleanup_old_experiments(experiments_dir: Path, 
                          max_age_days: int = 30,
                          max_count: int = 50) -> int:
    """Clean up old experiment directories.
    
    Args:
        experiments_dir: Base experiments directory
        max_age_days: Maximum age in days
        max_count: Maximum number of experiments to keep
        
    Returns:
        Number of directories cleaned up
    """
    logger = logging.getLogger(__name__)
    
    if not experiments_dir.exists():
        return 0
    
    try:
        # Get all experiment directories
        exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
        
        # Sort by modification time (newest first)
        exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for i, exp_dir in enumerate(exp_dirs):
            # Keep recent experiments
            if i < max_count:
                continue
            
            # Check age
            mod_time = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            if mod_time > cutoff_time:
                continue
            
            # Remove old experiment
            try:
                shutil.rmtree(exp_dir)
                cleaned_count += 1
                logger.info(f"Cleaned up old experiment: {exp_dir.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {exp_dir.name}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old experiments")
        
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old experiments: {e}")
        return 0


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate configuration against schema.
    
    Args:
        config: Configuration to validate
        schema: Validation schema
        
    Returns:
        List of validation errors
    """
    errors = []
    
    def validate_dict(cfg: Dict, sch: Dict, path: str = ""):
        for key, expected_type in sch.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in cfg:
                errors.append(f"Missing required key: {current_path}")
                continue
            
            value = cfg[key]
            
            if isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    errors.append(f"Expected dict for {current_path}, got {type(value).__name__}")
                else:
                    validate_dict(value, expected_type, current_path)
            elif isinstance(expected_type, type):
                if not isinstance(value, expected_type):
                    errors.append(f"Expected {expected_type.__name__} for {current_path}, got {type(value).__name__}")
            elif isinstance(expected_type, list):
                if not isinstance(value, list):
                    errors.append(f"Expected list for {current_path}, got {type(value).__name__}")
    
    validate_dict(config, schema)
    return errors


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_training_time(episodes_completed: int,
                          total_episodes: int,
                          elapsed_time: float) -> str:
    """Estimate remaining training time.
    
    Args:
        episodes_completed: Number of episodes completed
        total_episodes: Total number of episodes
        elapsed_time: Elapsed time in seconds
        
    Returns:
        Estimated remaining time string
    """
    if episodes_completed == 0:
        return "Unknown"
    
    episodes_remaining = total_episodes - episodes_completed
    time_per_episode = elapsed_time / episodes_completed
    estimated_remaining = episodes_remaining * time_per_episode
    
    return format_duration(estimated_remaining)


def create_video_from_images(image_dir: Path,
                           output_path: Path,
                           fps: int = 30,
                           image_pattern: str = "*.png") -> bool:
    """Create video from sequence of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        image_pattern: Image filename pattern
        
    Returns:
        True if successful
    """
    logger = logging.getLogger(__name__)
    
    try:
        import cv2
        
        # Get list of images
        images = sorted(list(image_dir.glob(image_pattern)))
        
        if not images:
            logger.warning(f"No images found in {image_dir} with pattern {image_pattern}")
            return False
        
        # Read first image to get dimensions
        first_image = cv2.imread(str(images[0]))
        height, width, layers = first_image.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Add images to video
        for image_path in images:
            image = cv2.imread(str(image_path))
            video_writer.write(image)
        
        video_writer.release()
        
        logger.info(f"Video created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        return False


# Global configuration schema for validation
CONFIG_SCHEMA = {
    'algorithm': {
        'learning_rate': float,
        'n_steps': int,
        'batch_size': int,
        'n_epochs': int,
        'gamma': float,
        'gae_lambda': float,
        'clip_range': float
    },
    'environment': {
        'max_episode_steps': int,
        'observation': dict,
        'reward': dict,
        'communication': dict
    },
    'training': {
        'max_episodes': int,
        'eval_frequency': int,
        'save_frequency': int,
        'log_frequency': int
    }
}

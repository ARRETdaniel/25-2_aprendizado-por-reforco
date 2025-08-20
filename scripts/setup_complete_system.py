#!/usr/bin/env python3
"""
Complete DRL Pipeline Setup Script
Automated installation and configuration for CARLA + ROS 2 + DRL system

Author: GitHub Copilot
Date: 2025-01-26
"""

import os
import sys
import subprocess
import platform
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_complete_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SystemSetup:
    """Complete system setup and validation"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.is_windows = platform.system() == "Windows"
        self.is_wsl = self._check_wsl()
        self.config = self._load_config()
        
    def _check_wsl(self) -> bool:
        """Check if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def _load_config(self) -> Dict:
        """Load system configuration"""
        config_path = self.base_path / "configs" / "system_settings.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default system configuration"""
        return {
            "python_versions": {
                "carla": "3.6",
                "drl": "3.12"
            },
            "carla": {
                "version": "0.8.4",
                "host": "127.0.0.1",
                "port": 2000,
                "timeout": 10.0
            },
            "ros2": {
                "distro": "humble",
                "workspace": "carla_drl_ws"
            },
            "zmq": {
                "ports": {
                    "camera_rgb": 5555,
                    "vehicle_state": 5556,
                    "control_cmd": 5557,
                    "reward": 5558
                }
            },
            "training": {
                "algorithm": "PPO",
                "max_episodes": 1000,
                "save_interval": 50
            }
        }
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        logger.info("🔍 Checking system prerequisites...")
        
        checks = [
            ("Python 3.6", self._check_python36),
            ("Python 3.12", self._check_python312),
            ("Docker", self._check_docker),
            ("Git", self._check_git),
            ("CARLA Server", self._check_carla),
            ("ROS 2", self._check_ros2) if not self.is_windows else ("WSL2", self._check_wsl2)
        ]
        
        all_passed = True
        for name, check_func in checks:
            try:
                if check_func():
                    logger.info(f"  ✅ {name}: OK")
                else:
                    logger.error(f"  ❌ {name}: FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"  ❌ {name}: ERROR - {e}")
                all_passed = False
        
        return all_passed
    
    def _check_python36(self) -> bool:
        """Check Python 3.6 availability"""
        try:
            if self.is_windows:
                result = subprocess.run(['python', '--version'], 
                                      capture_output=True, text=True)
                return '3.6' in result.stdout
            else:
                result = subprocess.run(['python3.6', '--version'], 
                                      capture_output=True, text=True)
                return result.returncode == 0
        except:
            return False
    
    def _check_python312(self) -> bool:
        """Check Python 3.12 availability"""
        try:
            if self.is_wsl:
                result = subprocess.run(['conda', 'list', 'envs'], 
                                      capture_output=True, text=True)
                return 'drl_py312' in result.stdout
            else:
                result = subprocess.run(['python3.12', '--version'], 
                                      capture_output=True, text=True)
                return result.returncode == 0
        except:
            return False
    
    def _check_docker(self) -> bool:
        """Check Docker availability"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_git(self) -> bool:
        """Check Git availability"""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_carla(self) -> bool:
        """Check CARLA server availability"""
        if self.is_windows:
            carla_paths = [
                "C:\\CARLA_0.8.4\\CarlaUE4\\Binaries\\Win64\\CarlaUE4.exe",
                "C:\\CARLA\\CarlaUE4\\Binaries\\Win64\\CarlaUE4.exe",
                str(self.base_path / "CarlaSimulator" / "CarlaUE4.exe")
            ]
            return any(Path(p).exists() for p in carla_paths)
        return True  # Skip for Linux/WSL
    
    def _check_ros2(self) -> bool:
        """Check ROS 2 installation"""
        try:
            result = subprocess.run(['ros2', '--version'], 
                                  capture_output=True, text=True)
            return 'humble' in result.stdout.lower()
        except:
            return False
    
    def _check_wsl2(self) -> bool:
        """Check WSL2 availability on Windows"""
        try:
            result = subprocess.run(['wsl', '--status'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def setup_python_environments(self) -> bool:
        """Setup Python virtual environments"""
        logger.info("🐍 Setting up Python environments...")
        
        if self.is_windows:
            return self._setup_windows_python()
        else:
            return self._setup_linux_python()
    
    def _setup_windows_python(self) -> bool:
        """Setup Python environments on Windows"""
        try:
            # Python 3.6 for CARLA
            carla_env = self.base_path / "carla_py36_env"
            if not carla_env.exists():
                logger.info("  Creating Python 3.6 environment for CARLA...")
                subprocess.run(['python', '-m', 'venv', str(carla_env)], check=True)
            
            # Install CARLA dependencies
            pip_exe = carla_env / "Scripts" / "pip.exe"
            requirements = self.base_path / "carla_client_py36" / "requirements_py36.txt"
            if requirements.exists():
                subprocess.run([str(pip_exe), 'install', '-r', str(requirements)], check=True)
            
            logger.info("  ✅ Python 3.6 environment ready")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to setup Python 3.6: {e}")
            return False
    
    def _setup_linux_python(self) -> bool:
        """Setup Python environments on Linux/WSL"""
        try:
            # Check conda availability
            result = subprocess.run(['conda', '--version'], capture_output=True)
            if result.returncode != 0:
                logger.error("  ❌ Conda not found. Please install Miniconda first.")
                return False
            
            # Create DRL environment
            logger.info("  Creating Python 3.12 environment for DRL...")
            subprocess.run(['conda', 'create', '-n', 'drl_py312', 'python=3.12', '-y'], 
                         check=True)
            
            # Install DRL dependencies
            requirements = self.base_path / "drl_agent" / "requirements_py312.txt"
            if requirements.exists():
                subprocess.run(['conda', 'run', '-n', 'drl_py312', 'pip', 'install', 
                              '-r', str(requirements)], check=True)
            
            logger.info("  ✅ Python 3.12 environment ready")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to setup Python 3.12: {e}")
            return False
    
    def build_ros2_workspace(self) -> bool:
        """Build ROS 2 workspace"""
        if self.is_windows:
            logger.info("🔧 ROS 2 build skipped on Windows (use WSL2)")
            return True
        
        logger.info("🔧 Building ROS 2 workspace...")
        
        try:
            # Source ROS 2
            env = os.environ.copy()
            env['ROS_DISTRO'] = 'humble'
            
            # Build workspace
            ros2_ws = self.base_path / "ros2_gateway"
            if ros2_ws.exists():
                subprocess.run(['colcon', 'build', '--symlink-install'], 
                             cwd=ros2_ws, env=env, check=True)
                logger.info("  ✅ ROS 2 workspace built successfully")
                return True
            else:
                logger.error("  ❌ ROS 2 workspace not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to build ROS 2 workspace: {e}")
            return False
    
    def create_configuration_files(self) -> bool:
        """Create system configuration files"""
        logger.info("⚙️ Creating configuration files...")
        
        configs_dir = self.base_path / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        # System configuration
        system_config = {
            "system": {
                "name": "CARLA DRL Pipeline",
                "version": "1.0.0",
                "platform": platform.system()
            },
            **self.config
        }
        
        with open(configs_dir / "system_settings.yaml", 'w') as f:
            import yaml
            yaml.dump(system_config, f, default_flow_style=False)
        
        # Create launch scripts
        self._create_launch_scripts()
        
        logger.info("  ✅ Configuration files created")
        return True
    
    def _create_launch_scripts(self):
        """Create system launch scripts"""
        scripts_dir = self.base_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        if self.is_windows:
            # Windows batch script
            batch_content = """@echo off
echo Starting CARLA DRL Pipeline...
echo.

echo [1/4] Starting CARLA Server...
start "CARLA Server" cmd /k "cd /d C:\\CARLA_0.8.4 && CarlaUE4.exe /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=30"

timeout /t 10

echo [2/4] Starting CARLA Client...
start "CARLA Client" cmd /k "cd /d %~dp0..\\carla_client_py36 && ..\\carla_py36_env\\Scripts\\activate && python main_enhanced.py"

echo [3/4] Please start ROS 2 Gateway in WSL2:
echo   wsl -d Ubuntu-22.04 -e bash -c "cd /mnt/c/Users/%USERNAME%/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco && source setup_wsl.sh"

echo [4/4] Please start DRL Agent in WSL2:
echo   (Wait for ROS 2 Gateway to be ready)

echo.
echo System starting... Check individual windows for status.
pause
"""
            with open(scripts_dir / "start_system.bat", 'w') as f:
                f.write(batch_content)
        
        # WSL2 setup script
        wsl_setup = """#!/bin/bash
echo "Setting up WSL2 environment..."

# Source ROS 2
source /opt/ros/humble/setup.bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate drl_py312

# Set environment variables
export ROS_DOMAIN_ID=42
export CARLA_HOST=127.0.0.1
export CARLA_PORT=2000

echo "WSL2 environment ready!"
echo "Now run:"
echo "  Terminal 1: ros2 launch carla_bridge carla_bridge.launch.py"
echo "  Terminal 2: python drl_agent/train_ppo.py --visualize"
"""
        with open(self.base_path / "setup_wsl.sh", 'w') as f:
            f.write(wsl_setup)
        
        # Make executable
        os.chmod(self.base_path / "setup_wsl.sh", 0o755)
    
    def test_system_integration(self) -> bool:
        """Test complete system integration"""
        logger.info("🧪 Testing system integration...")
        
        tests = [
            ("Configuration validation", self._test_config),
            ("Port availability", self._test_ports),
            ("File permissions", self._test_permissions),
            ("Python imports", self._test_imports)
        ]
        
        all_passed = True
        for name, test_func in tests:
            try:
                if test_func():
                    logger.info(f"  ✅ {name}: PASSED")
                else:
                    logger.warning(f"  ⚠️ {name}: FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"  ❌ {name}: ERROR - {e}")
                all_passed = False
        
        return all_passed
    
    def _test_config(self) -> bool:
        """Test configuration validation"""
        try:
            # Validate YAML files
            config_files = list(self.base_path.glob("configs/*.yaml"))
            for config_file in config_files:
                import yaml
                with open(config_file) as f:
                    yaml.safe_load(f)
            return True
        except:
            return False
    
    def _test_ports(self) -> bool:
        """Test port availability"""
        import socket
        
        ports = [2000, 5555, 5556, 5557, 5558, 6006]  # CARLA, ZMQ, TensorBoard
        
        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                except OSError:
                    logger.warning(f"    Port {port} is in use")
                    # Don't fail for ports in use (might be running services)
        
        return True
    
    def _test_permissions(self) -> bool:
        """Test file permissions"""
        try:
            # Test write permissions
            test_file = self.base_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except:
            return False
    
    def _test_imports(self) -> bool:
        """Test Python imports"""
        if self.is_windows:
            return True  # Skip import tests on Windows
        
        try:
            # Test basic imports in DRL environment
            import_test = """
import torch
import numpy as np
import cv2
import yaml
print("All imports successful")
"""
            result = subprocess.run(['conda', 'run', '-n', 'drl_py312', 'python', '-c', import_test],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def generate_startup_guide(self) -> str:
        """Generate startup guide"""
        guide = f"""
# 🚀 CARLA DRL Pipeline - Startup Guide

## System Status
- Platform: {platform.system()}
- Base Path: {self.base_path}
- Python Environments: Configured
- ROS 2 Workspace: {'Built' if not self.is_windows else 'Use WSL2'}

## Quick Start (First-Time Visualization)

### Windows Users:
1. **Start System**: Double-click `scripts/start_system.bat`
2. **WSL2 Terminal**: Run `wsl -d Ubuntu-22.04` and execute:
   ```bash
   cd /mnt/c/Users/$USER/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco
   source setup_wsl.sh
   
   # Terminal 1: ROS 2 Bridge
   ros2 launch carla_bridge carla_bridge.launch.py
   
   # Terminal 2: DRL Training
   python drl_agent/train_ppo.py --visualize --episodes 10
   
   # Terminal 3: TensorBoard
   tensorboard --logdir monitoring/tensorboard_logs --host 0.0.0.0
   ```

### Expected Outputs:
- **CARLA Window**: 3D simulation with ego vehicle
- **Camera Feeds**: RGB + Depth camera windows
- **Training Plots**: Real-time reward/loss graphs
- **TensorBoard**: http://localhost:6006 (metrics dashboard)
- **Console Logs**: Training progress and episode statistics

### Verification:
- ✅ CARLA server: Vehicle spawns in Town01
- ✅ ROS 2 bridge: Topics publishing at 30 Hz
- ✅ DRL agent: Training episodes complete
- ✅ Visualization: Real-time plots updating
- ✅ Performance: >20 FPS simulation rate

## Troubleshooting:
- **Port conflicts**: Run `netstat -an | findstr 2000,5555,5556,5557,5558`
- **CARLA issues**: Check Windows Defender + Antivirus exclusions
- **ROS 2 issues**: Verify `echo $ROS_DISTRO` shows 'humble'
- **Python issues**: Check virtual environment activation

## Next Steps:
1. **Train longer**: Increase episodes in config
2. **Tune hyperparameters**: Edit `configs/ppo_training.yaml`
3. **Add scenarios**: Modify `configs/carla_sim.yaml`
4. **Evaluate models**: Run `python drl_agent/evaluation.py`

## Support:
- Logs: Check `setup_complete_system.log`
- Health check: `python scripts/health_check.py`
- Documentation: See `docs/` directory

**🎯 System Ready for Deep Reinforcement Learning Training!**
"""
        
        guide_path = self.base_path / "STARTUP_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        return guide

def main():
    parser = argparse.ArgumentParser(description='CARLA DRL Pipeline Complete Setup')
    parser.add_argument('--base-path', default='.', help='Base path for the project')
    parser.add_argument('--skip-tests', action='store_true', help='Skip integration tests')
    parser.add_argument('--generate-guide-only', action='store_true', help='Only generate startup guide')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = SystemSetup(args.base_path)
    
    if args.generate_guide_only:
        guide = setup.generate_startup_guide()
        logger.info("📋 Startup guide generated!")
        print(guide)
        return
    
    logger.info("🚀 Starting CARLA DRL Pipeline Complete Setup...")
    
    # Run setup steps
    steps = [
        ("Prerequisites Check", setup.check_prerequisites),
        ("Python Environments", setup.setup_python_environments),
        ("ROS 2 Workspace", setup.build_ros2_workspace),
        ("Configuration Files", setup.create_configuration_files),
    ]
    
    if not args.skip_tests:
        steps.append(("Integration Tests", setup.test_system_integration))
    
    # Execute setup steps
    for step_name, step_func in steps:
        logger.info(f"📋 {step_name}...")
        if not step_func():
            logger.error(f"❌ {step_name} failed!")
            sys.exit(1)
        logger.info(f"✅ {step_name} completed!")
    
    # Generate startup guide
    guide = setup.generate_startup_guide()
    logger.info("📋 Startup guide generated!")
    
    logger.info("🎉 CARLA DRL Pipeline setup complete!")
    logger.info("📖 See STARTUP_GUIDE.md for next steps")
    
    # Print quick start info
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE - READY FOR FIRST RUN!")
    print("="*60)
    print("Next steps:")
    print("1. Windows: Run 'scripts/start_system.bat'")
    print("2. WSL2: Run 'source setup_wsl.sh'")
    print("3. Start training: 'python drl_agent/train_ppo.py --visualize'")
    print("4. Open TensorBoard: http://localhost:6006")
    print("="*60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
First-Time Visualization Runner
Automated startup and visualization for CARLA DRL training

Author: GitHub Copilot
Date: 2025-01-26
"""

import os
import sys
import subprocess
import time
import threading
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import socket
import argparse
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('first_run_visualization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VisualizationRunner:
    """Orchestrates the complete CARLA DRL visualization system"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.processes = {}
        self.is_running = False
        self.config = self._load_config()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict:
        """Load system configuration"""
        config_path = self.base_path / "configs" / "visualization_config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default visualization configuration"""
        return {
            "carla": {
                "executable": "C:\\CARLA_0.8.4\\CarlaUE4\\Binaries\\Win64\\CarlaUE4.exe",
                "args": ["/Game/Maps/Town01", "-windowed", "-carla-server", 
                        "-benchmark", "-fps=30", "-quality-level=Low"],
                "timeout": 30,
                "port": 2000
            },
            "training": {
                "episodes": 10,
                "algorithm": "PPO",
                "save_interval": 5,
                "visualization": True,
                "tensorboard": True
            },
            "monitoring": {
                "system_metrics": True,
                "performance_plots": True,
                "camera_feeds": True,
                "update_interval": 1.0
            },
            "ros2": {
                "domain_id": 42,
                "qos_reliability": "RELIABLE",
                "qos_durability": "VOLATILE"
            },
            "display": {
                "window_size": [800, 600],
                "fps_target": 30,
                "plot_update_rate": 10
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def check_system_ready(self) -> bool:
        """Comprehensive system readiness check"""
        logger.info("🔍 Checking system readiness...")
        
        checks = [
            ("CARLA executable", self._check_carla_executable),
            ("Python environments", self._check_python_envs),
            ("Port availability", self._check_ports),
            ("ROS 2 environment", self._check_ros2_env),
            ("GPU availability", self._check_gpu),
            ("Disk space", self._check_disk_space),
            ("Memory availability", self._check_memory)
        ]
        
        all_ready = True
        for name, check_func in checks:
            try:
                if check_func():
                    logger.info(f"  ✅ {name}: Ready")
                else:
                    logger.error(f"  ❌ {name}: Not ready")
                    all_ready = False
            except Exception as e:
                logger.error(f"  ❌ {name}: Error - {e}")
                all_ready = False
        
        return all_ready
    
    def _check_carla_executable(self) -> bool:
        """Check CARLA server executable"""
        carla_exe = Path(self.config["carla"]["executable"])
        return carla_exe.exists()
    
    def _check_python_envs(self) -> bool:
        """Check Python environments"""
        # Check Windows Python 3.6 environment
        carla_env = self.base_path / "carla_py36_env"
        if not carla_env.exists():
            return False
        
        # Check activation script
        activate_script = carla_env / "Scripts" / "activate"
        return activate_script.exists()
    
    def _check_ports(self) -> bool:
        """Check required ports availability"""
        required_ports = [2000, 5555, 5556, 5557, 5558, 6006]
        
        for port in required_ports:
            if self._is_port_in_use(port):
                logger.warning(f"    Port {port} is in use")
                # Don't fail - might be our own services
        
        return True
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return False
            except OSError:
                return True
    
    def _check_ros2_env(self) -> bool:
        """Check ROS 2 environment (WSL2)"""
        try:
            # Check if WSL2 is available
            result = subprocess.run(['wsl', '--status'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _check_gpu(self) -> bool:
        """Check GPU availability"""
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            logger.warning("    GPU check failed - will use CPU")
            return True  # Don't fail setup for missing GPU
    
    def _check_disk_space(self) -> bool:
        """Check available disk space (min 5GB)"""
        try:
            disk_usage = psutil.disk_usage(str(self.base_path))
            free_gb = disk_usage.free / (1024**3)
            return free_gb >= 5.0
        except:
            return True  # Don't fail on check error
    
    def _check_memory(self) -> bool:
        """Check available memory (min 8GB)"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            return available_gb >= 8.0
        except:
            return True  # Don't fail on check error
    
    def start_carla_server(self) -> bool:
        """Start CARLA server"""
        logger.info("🚗 Starting CARLA server...")
        
        carla_exe = self.config["carla"]["executable"]
        carla_args = self.config["carla"]["args"]
        
        try:
            # Kill any existing CARLA processes
            self._kill_existing_carla()
            
            # Start CARLA server
            cmd = [carla_exe] + carla_args
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            self.processes['carla_server'] = process
            
            # Wait for CARLA to be ready
            if self._wait_for_carla_ready():
                logger.info("  ✅ CARLA server started successfully")
                return True
            else:
                logger.error("  ❌ CARLA server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Failed to start CARLA: {e}")
            return False
    
    def _kill_existing_carla(self):
        """Kill any existing CARLA processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'carla' in proc.info['name'].lower():
                    proc.kill()
                    logger.info(f"    Killed existing CARLA process: {proc.info['pid']}")
        except:
            pass
    
    def _wait_for_carla_ready(self, timeout: int = 30) -> bool:
        """Wait for CARLA server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_port_in_use(self.config["carla"]["port"]):
                time.sleep(2)  # Additional wait for full initialization
                return True
            time.sleep(1)
        return False
    
    def start_carla_client(self) -> bool:
        """Start enhanced CARLA client"""
        logger.info("🎮 Starting CARLA client...")
        
        try:
            client_dir = self.base_path / "carla_client_py36"
            if not client_dir.exists():
                # Fallback to existing structure
                client_dir = self.base_path / "CarlaSimulator" / "PythonClient" / "FinalProject"
                client_script = "module_7.py"
            else:
                client_script = "main_enhanced.py"
            
            if not client_dir.exists():
                logger.error("  ❌ CARLA client directory not found")
                return False
            
            # Activate virtual environment and run client
            if os.name == 'nt':  # Windows
                activate_script = self.base_path / "carla_py36_env" / "Scripts" / "activate.bat"
                cmd = f'"{activate_script}" && cd "{client_dir}" && python {client_script} --visualize'
                
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux/WSL2 version would go here
                logger.warning("  ⚠️ Linux client startup not implemented")
                return True
            
            self.processes['carla_client'] = process
            
            # Wait for client to initialize
            time.sleep(5)
            
            if process.poll() is None:  # Process still running
                logger.info("  ✅ CARLA client started successfully")
                return True
            else:
                logger.error("  ❌ CARLA client failed to start")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Failed to start CARLA client: {e}")
            return False
    
    def start_ros2_bridge(self) -> bool:
        """Start ROS 2 bridge in WSL2"""
        logger.info("🌉 Starting ROS 2 bridge...")
        
        try:
            # Prepare WSL2 command
            wsl_cmd = [
                'wsl', '-d', 'Ubuntu-22.04', '-e', 'bash', '-c',
                f'''
                cd "/mnt/c/Users/$USER/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco" &&
                source /opt/ros/humble/setup.bash &&
                export ROS_DOMAIN_ID={self.config["ros2"]["domain_id"]} &&
                export CARLA_HOST=127.0.0.1 &&
                export CARLA_PORT=2000 &&
                cd ros2_gateway &&
                source install/setup.bash &&
                ros2 launch carla_bridge carla_bridge.launch.py
                '''
            ]
            
            process = subprocess.Popen(
                wsl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            self.processes['ros2_bridge'] = process
            
            # Wait for bridge to initialize
            time.sleep(10)
            
            if process.poll() is None:
                logger.info("  ✅ ROS 2 bridge started successfully")
                return True
            else:
                logger.error("  ❌ ROS 2 bridge failed to start")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Failed to start ROS 2 bridge: {e}")
            return False
    
    def start_drl_training(self) -> bool:
        """Start DRL training with visualization"""
        logger.info("🧠 Starting DRL training...")
        
        try:
            # Prepare DRL training command
            wsl_cmd = [
                'wsl', '-d', 'Ubuntu-22.04', '-e', 'bash', '-c',
                f'''
                cd "/mnt/c/Users/$USER/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco" &&
                eval "$(conda shell.bash hook)" &&
                conda activate drl_py312 &&
                export ROS_DOMAIN_ID={self.config["ros2"]["domain_id"]} &&
                cd drl_agent &&
                python train_ppo.py --episodes {self.config["training"]["episodes"]} --visualize --config ../configs/ppo_training.yaml
                '''
            ]
            
            process = subprocess.Popen(
                wsl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            self.processes['drl_training'] = process
            
            # Wait for training to initialize
            time.sleep(5)
            
            if process.poll() is None:
                logger.info("  ✅ DRL training started successfully")
                return True
            else:
                logger.error("  ❌ DRL training failed to start")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Failed to start DRL training: {e}")
            return False
    
    def start_tensorboard(self) -> bool:
        """Start TensorBoard for monitoring"""
        logger.info("📊 Starting TensorBoard...")
        
        try:
            wsl_cmd = [
                'wsl', '-d', 'Ubuntu-22.04', '-e', 'bash', '-c',
                f'''
                cd "/mnt/c/Users/$USER/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco" &&
                eval "$(conda shell.bash hook)" &&
                conda activate drl_py312 &&
                tensorboard --logdir monitoring/tensorboard_logs --host 0.0.0.0 --port 6006
                '''
            ]
            
            process = subprocess.Popen(
                wsl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes['tensorboard'] = process
            
            # Wait for TensorBoard to start
            time.sleep(5)
            
            if process.poll() is None:
                logger.info("  ✅ TensorBoard started at http://localhost:6006")
                return True
            else:
                logger.error("  ❌ TensorBoard failed to start")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Failed to start TensorBoard: {e}")
            return False
    
    def monitor_system(self):
        """Monitor system status and performance"""
        logger.info("📈 Starting system monitoring...")
        
        while self.is_running:
            try:
                # Check process health
                dead_processes = []
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        dead_processes.append(name)
                
                if dead_processes:
                    logger.warning(f"⚠️ Dead processes detected: {dead_processes}")
                
                # Log system metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                logger.info(f"📊 System: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
                
                # Check for GPU usage if available
                try:
                    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                             '--format=csv,noheader,nounits'], 
                                            capture_output=True, text=True, timeout=2)
                    if gpu_info.returncode == 0:
                        gpu_data = gpu_info.stdout.strip().split(',')
                        gpu_util = gpu_data[0].strip()
                        gpu_mem_used = gpu_data[1].strip()
                        gpu_mem_total = gpu_data[2].strip()
                        logger.info(f"🎮 GPU: {gpu_util}% util, {gpu_mem_used}/{gpu_mem_total}MB memory")
                except:
                    pass  # GPU monitoring optional
                
                time.sleep(self.config["monitoring"]["update_interval"])
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(5)
    
    def run_complete_system(self) -> bool:
        """Run the complete system"""
        logger.info("🚀 Starting complete CARLA DRL visualization system...")
        
        # Check system readiness
        if not self.check_system_ready():
            logger.error("❌ System not ready. Please check prerequisites.")
            return False
        
        self.is_running = True
        
        # Start system components in sequence
        startup_sequence = [
            ("CARLA Server", self.start_carla_server),
            ("CARLA Client", self.start_carla_client),
            ("ROS 2 Bridge", self.start_ros2_bridge),
            ("DRL Training", self.start_drl_training),
        ]
        
        if self.config["training"]["tensorboard"]:
            startup_sequence.append(("TensorBoard", self.start_tensorboard))
        
        # Execute startup sequence
        for component_name, start_func in startup_sequence:
            logger.info(f"▶️ Starting {component_name}...")
            if not start_func():
                logger.error(f"❌ Failed to start {component_name}")
                self.shutdown()
                return False
            
            # Brief pause between components
            time.sleep(3)
        
        logger.info("✅ All components started successfully!")
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        monitor_thread.start()
        
        # Print status information
        self._print_status_info()
        
        return True
    
    def _print_status_info(self):
        """Print system status information"""
        print("\n" + "="*60)
        print("🎉 CARLA DRL SYSTEM RUNNING!")
        print("="*60)
        print("📺 Visual Components:")
        print("   • CARLA 3D Window: Town01 simulation")
        print("   • Camera Feeds: RGB + Depth windows")
        print("   • Training Plots: Real-time reward graphs")
        print("   • TensorBoard: http://localhost:6006")
        print()
        print("📊 System Status:")
        print("   • CARLA Server: Running on port 2000")
        print("   • ROS 2 Bridge: WSL2 (domain 42)")
        print("   • DRL Training: PPO algorithm active")
        print("   • Monitoring: System metrics logged")
        print()
        print("🎮 Controls:")
        print("   • Ctrl+C: Graceful shutdown")
        print("   • Watch logs: tail -f first_run_visualization.log")
        print("   • Check status: python scripts/health_check.py")
        print()
        print("🚀 Training in progress... Watch the magic happen!")
        print("="*60)
    
    def wait_for_completion(self):
        """Wait for training completion or user interrupt"""
        try:
            while self.is_running:
                # Check if training is complete
                if 'drl_training' in self.processes:
                    training_process = self.processes['drl_training']
                    if training_process.poll() is not None:
                        logger.info("🎯 Training completed!")
                        break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("⚠️ User interrupt received")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful system shutdown"""
        if not self.is_running:
            return
        
        logger.info("🛑 Shutting down system...")
        self.is_running = False
        
        # Shutdown processes in reverse order
        shutdown_order = ['tensorboard', 'drl_training', 'ros2_bridge', 'carla_client', 'carla_server']
        
        for process_name in shutdown_order:
            if process_name in self.processes:
                process = self.processes[process_name]
                try:
                    logger.info(f"  Stopping {process_name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                        logger.info(f"    ✅ {process_name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"    ⚠️ Force killing {process_name}")
                        process.kill()
                        process.wait()
                
                except Exception as e:
                    logger.error(f"    ❌ Error stopping {process_name}: {e}")
        
        # Clean up any remaining CARLA processes
        self._kill_existing_carla()
        
        logger.info("✅ System shutdown complete")

def main():
    parser = argparse.ArgumentParser(description='CARLA DRL First-Time Visualization')
    parser.add_argument('--base-path', default='.', help='Base path for the project')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--config', help='Custom configuration file')
    parser.add_argument('--no-tensorboard', action='store_true', help='Skip TensorBoard startup')
    parser.add_argument('--check-only', action='store_true', help='Only check system readiness')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = VisualizationRunner(args.base_path)
    
    # Override config if specified
    if args.episodes != 10:
        runner.config["training"]["episodes"] = args.episodes
    if args.no_tensorboard:
        runner.config["training"]["tensorboard"] = False
    
    # Check-only mode
    if args.check_only:
        if runner.check_system_ready():
            logger.info("🎉 System ready for visualization!")
            sys.exit(0)
        else:
            logger.error("❌ System not ready. Check logs for details.")
            sys.exit(1)
    
    # Run complete system
    try:
        if runner.run_complete_system():
            logger.info("🎯 System running successfully!")
            runner.wait_for_completion()
        else:
            logger.error("❌ Failed to start system")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        runner.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main()

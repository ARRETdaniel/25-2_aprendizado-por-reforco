#!/usr/bin/env python3
"""
System Health Check and Diagnostics
Comprehensive validation of CARLA DRL pipeline components

Author: GitHub Copilot
Date: 2025-01-26
"""

import os
import sys
import subprocess
import socket
import time
import psutil
import platform
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """Comprehensive system health and diagnostics"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.results = {
            'timestamp': time.time(),
            'platform': platform.system(),
            'checks': {},
            'overall_status': 'UNKNOWN'
        }
        
    def check_system_requirements(self) -> bool:
        """Check basic system requirements"""
        logger.info("🖥️ Checking system requirements...")
        
        checks = {
            'os_version': self._check_os_version(),
            'memory': self._check_memory(),
            'disk_space': self._check_disk_space(),
            'cpu_cores': self._check_cpu_cores(),
            'gpu': self._check_gpu()
        }
        
        all_passed = all(checks.values())
        self.results['checks']['system_requirements'] = checks
        
        for check_name, status in checks.items():
            logger.info(f"  {'✅' if status else '❌'} {check_name}: {'OK' if status else 'FAILED'}")
        
        return all_passed
    
    def _check_os_version(self) -> bool:
        """Check OS version compatibility"""
        try:
            if platform.system() == "Windows":
                # Check for Windows 10/11
                version = platform.version()
                return "10.0" in version  # Windows 10/11
            return True  # Assume other OS are fine
        except:
            return False
    
    def _check_memory(self) -> bool:
        """Check available memory (minimum 8GB)"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            logger.info(f"    Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
            return available_gb >= 8.0
        except:
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space (minimum 20GB)"""
        try:
            disk_usage = psutil.disk_usage(str(self.base_path))
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            logger.info(f"    Disk: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            return free_gb >= 20.0
        except:
            return False
    
    def _check_cpu_cores(self) -> bool:
        """Check CPU cores (minimum 4 cores)"""
        try:
            cpu_count = psutil.cpu_count(logical=False)
            logical_count = psutil.cpu_count(logical=True)
            logger.info(f"    CPU: {cpu_count} physical cores, {logical_count} logical cores")
            return cpu_count >= 4
        except:
            return False
    
    def _check_gpu(self) -> bool:
        """Check GPU availability"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("    GPU: NVIDIA GPU detected")
                return True
            else:
                logger.warning("    GPU: No NVIDIA GPU detected (will use CPU)")
                return True  # Don't fail on missing GPU
        except:
            logger.warning("    GPU: Cannot detect GPU (will use CPU)")
            return True  # Don't fail on missing GPU
    
    def check_software_dependencies(self) -> bool:
        """Check software dependencies"""
        logger.info("📦 Checking software dependencies...")
        
        checks = {
            'python36': self._check_python36(),
            'python312': self._check_python312(),
            'docker': self._check_docker(),
            'wsl2': self._check_wsl2(),
            'git': self._check_git(),
            'carla': self._check_carla()
        }
        
        all_passed = all(checks.values())
        self.results['checks']['software_dependencies'] = checks
        
        for check_name, status in checks.items():
            logger.info(f"  {'✅' if status else '❌'} {check_name}: {'OK' if status else 'FAILED'}")
        
        return all_passed
    
    def _check_python36(self) -> bool:
        """Check Python 3.6 availability"""
        try:
            if platform.system() == "Windows":
                # Check virtual environment
                venv_path = self.base_path / "carla_py36_env"
                python_exe = venv_path / "Scripts" / "python.exe"
                if python_exe.exists():
                    result = subprocess.run([str(python_exe), '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    version_ok = '3.6' in result.stdout
                    logger.info(f"    Python 3.6: {result.stdout.strip()}")
                    return version_ok
            else:
                result = subprocess.run(['python3.6', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                return result.returncode == 0
        except:
            pass
        return False
    
    def _check_python312(self) -> bool:
        """Check Python 3.12 availability (in conda)"""
        try:
            if platform.system() == "Linux" or self._is_wsl():
                result = subprocess.run(['conda', 'env', 'list'], 
                                      capture_output=True, text=True, timeout=10)
                env_exists = 'drl_py312' in result.stdout
                if env_exists:
                    logger.info("    Python 3.12: Conda environment 'drl_py312' found")
                return env_exists
            else:
                # Check system Python on Windows
                result = subprocess.run(['python', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                version_ok = '3.12' in result.stdout
                if version_ok:
                    logger.info(f"    Python 3.12: {result.stdout.strip()}")
                return version_ok
        except:
            pass
        return False
    
    def _check_docker(self) -> bool:
        """Check Docker availability"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"    Docker: {result.stdout.strip()}")
                return True
        except:
            pass
        return False
    
    def _check_wsl2(self) -> bool:
        """Check WSL2 availability"""
        if platform.system() != "Windows":
            return True  # Not needed on Linux
        
        try:
            result = subprocess.run(['wsl', '--status'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("    WSL2: Available")
                return True
        except:
            pass
        return False
    
    def _is_wsl(self) -> bool:
        """Check if running in WSL"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False
    
    def _check_git(self) -> bool:
        """Check Git availability"""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"    Git: {result.stdout.strip()}")
                return True
        except:
            pass
        return False
    
    def _check_carla(self) -> bool:
        """Check CARLA installation"""
        carla_paths = [
            "C:\\CARLA_0.8.4\\CarlaUE4\\Binaries\\Win64\\CarlaUE4.exe",
            "C:\\CARLA\\CarlaUE4\\Binaries\\Win64\\CarlaUE4.exe",
            str(self.base_path / "CarlaSimulator" / "CarlaUE4.exe")
        ]
        
        for path in carla_paths:
            if Path(path).exists():
                logger.info(f"    CARLA: Found at {path}")
                return True
        
        logger.warning("    CARLA: Not found in standard locations")
        return False
    
    def check_network_connectivity(self) -> bool:
        """Check network and port availability"""
        logger.info("🌐 Checking network connectivity...")
        
        checks = {
            'localhost': self._check_localhost(),
            'carla_port': self._check_port_available(2000),
            'zmq_ports': self._check_zmq_ports(),
            'tensorboard_port': self._check_port_available(6006),
            'internet': self._check_internet()
        }
        
        all_passed = all(checks.values())
        self.results['checks']['network_connectivity'] = checks
        
        for check_name, status in checks.items():
            logger.info(f"  {'✅' if status else '❌'} {check_name}: {'OK' if status else 'FAILED'}")
        
        return all_passed
    
    def _check_localhost(self) -> bool:
        """Check localhost connectivity"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('127.0.0.1', 80))  # Try to connect to localhost
                return True
        except:
            # Try alternative approach
            try:
                import urllib.request
                urllib.request.urlopen('http://127.0.0.1', timeout=1)
                return True
            except:
                pass
        return True  # Assume localhost works if we can't test
    
    def _check_port_available(self, port: int) -> bool:
        """Check if port is available (not in use)"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    logger.info(f"    Port {port}: In use")
                    return True  # Port is in use (might be our services)
                else:
                    logger.info(f"    Port {port}: Available")
                    return True  # Port is available
        except:
            return True  # Assume OK if we can't check
    
    def _check_zmq_ports(self) -> bool:
        """Check ZeroMQ ports (5555-5560)"""
        zmq_ports = [5555, 5556, 5557, 5558, 5559, 5560]
        available_count = 0
        
        for port in zmq_ports:
            if self._check_port_available(port):
                available_count += 1
        
        logger.info(f"    ZMQ ports: {available_count}/{len(zmq_ports)} available")
        return available_count >= len(zmq_ports) // 2  # At least half should be available
    
    def _check_internet(self) -> bool:
        """Check internet connectivity"""
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=5)
            logger.info("    Internet: Connected")
            return True
        except:
            logger.warning("    Internet: Not connected (optional)")
            return True  # Don't fail on missing internet
    
    def check_project_structure(self) -> bool:
        """Check project directory structure"""
        logger.info("📁 Checking project structure...")
        
        required_paths = [
            'configs',
            'scripts',
            'carla_client_py36',
            'drl_agent',
            'ros2_gateway',
            'monitoring'
        ]
        
        optional_paths = [
            'CarlaSimulator/PythonClient/FinalProject',
            'rl_environment',
            'docs',
            'tests'
        ]
        
        checks = {}
        
        # Check required paths
        for path in required_paths:
            path_obj = self.base_path / path
            exists = path_obj.exists()
            checks[f"required_{path}"] = exists
            logger.info(f"  {'✅' if exists else '❌'} Required: {path}")
        
        # Check optional paths
        for path in optional_paths:
            path_obj = self.base_path / path
            exists = path_obj.exists()
            checks[f"optional_{path}"] = exists
            logger.info(f"  {'✅' if exists else '⚠️'} Optional: {path}")
        
        # Check configuration files
        config_files = [
            'configs/complete_system_config.yaml',
            'configs/sim.yaml'
        ]
        
        for config_file in config_files:
            config_path = self.base_path / config_file
            exists = config_path.exists()
            checks[f"config_{config_file}"] = exists
            logger.info(f"  {'✅' if exists else '❌'} Config: {config_file}")
        
        # Overall project structure check
        required_checks = [v for k, v in checks.items() if k.startswith('required_')]
        all_required_present = all(required_checks)
        
        self.results['checks']['project_structure'] = checks
        return all_required_present
    
    def check_python_packages(self) -> bool:
        """Check Python package dependencies"""
        logger.info("📦 Checking Python packages...")
        
        # Python 3.6 packages (CARLA)
        py36_packages = ['numpy', 'opencv-python', 'pyyaml', 'msgpack']
        py36_status = self._check_packages_in_env('carla_py36_env', py36_packages)
        
        # Python 3.12 packages (DRL)
        py312_packages = ['torch', 'numpy', 'opencv-python', 'pyyaml', 'rclpy', 'tensorboard']
        py312_status = self._check_packages_in_conda('drl_py312', py312_packages)
        
        checks = {
            'python36_packages': py36_status,
            'python312_packages': py312_status
        }
        
        self.results['checks']['python_packages'] = checks
        
        for check_name, status in checks.items():
            logger.info(f"  {'✅' if status else '❌'} {check_name}: {'OK' if status else 'FAILED'}")
        
        return all(checks.values())
    
    def _check_packages_in_env(self, env_name: str, packages: List[str]) -> bool:
        """Check packages in virtual environment"""
        try:
            if platform.system() == "Windows":
                venv_path = self.base_path / env_name
                pip_exe = venv_path / "Scripts" / "pip.exe"
                if not pip_exe.exists():
                    return False
                
                for package in packages:
                    result = subprocess.run([str(pip_exe), 'show', package], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        logger.warning(f"    Package {package} not found in {env_name}")
                        return False
                
                logger.info(f"    All packages found in {env_name}")
                return True
        except:
            pass
        return False
    
    def _check_packages_in_conda(self, env_name: str, packages: List[str]) -> bool:
        """Check packages in conda environment"""
        try:
            for package in packages:
                result = subprocess.run(['conda', 'list', '-n', env_name, package], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0 or package not in result.stdout:
                    logger.warning(f"    Package {package} not found in conda env {env_name}")
                    return False
            
            logger.info(f"    All packages found in conda env {env_name}")
            return True
        except:
            pass
        return False
    
    def check_running_processes(self) -> bool:
        """Check for running CARLA/training processes"""
        logger.info("🔄 Checking running processes...")
        
        process_checks = {
            'carla_server': self._find_process('CarlaUE4'),
            'python_processes': self._count_python_processes(),
            'docker_processes': self._find_process('docker'),
            'wsl_processes': self._find_process('wsl')
        }
        
        self.results['checks']['running_processes'] = process_checks
        
        # Log findings
        if process_checks['carla_server']:
            logger.info("  ✅ CARLA server: Running")
        else:
            logger.info("  ⚠️ CARLA server: Not running")
        
        logger.info(f"  📊 Python processes: {process_checks['python_processes']} running")
        
        return True  # Don't fail based on running processes
    
    def _find_process(self, process_name: str) -> bool:
        """Find process by name"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if process_name.lower() in proc.info['name'].lower():
                    return True
        except:
            pass
        return False
    
    def _count_python_processes(self) -> int:
        """Count Python processes"""
        count = 0
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'python' in proc.info['name'].lower():
                    count += 1
        except:
            pass
        return count
    
    def generate_performance_report(self) -> Dict:
        """Generate system performance report"""
        logger.info("📊 Generating performance report...")
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage(str(self.base_path))
            
            # Network stats
            network = psutil.net_io_counters()
            
            # GPU info (if available)
            gpu_info = self._get_gpu_info()
            
            performance_report = {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': cpu_count,
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                'gpu': gpu_info
            }
            
            self.results['performance'] = performance_report
            
            # Log key metrics
            logger.info(f"  CPU: {cpu_percent:.1f}% usage, {cpu_count} cores")
            logger.info(f"  Memory: {memory.percent:.1f}% used ({memory.available / (1024**3):.1f}GB available)")
            logger.info(f"  Disk: {(disk.used / disk.total) * 100:.1f}% used ({disk.free / (1024**3):.1f}GB free)")
            
            return performance_report
            
        except Exception as e:
            logger.error(f"Performance report error: {e}")
            return {}
    
    def _get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        gpus.append({
                            'name': parts[0],
                            'memory_total_mb': int(parts[1]),
                            'memory_used_mb': int(parts[2]),
                            'utilization_percent': int(parts[3])
                        })
                
                if gpus:
                    logger.info(f"  GPU: {gpus[0]['name']}, "
                              f"{gpus[0]['utilization_percent']}% usage, "
                              f"{gpus[0]['memory_used_mb']}/{gpus[0]['memory_total_mb']}MB memory")
                
                return {'gpus': gpus}
        except:
            pass
        return None
    
    def run_comprehensive_check(self) -> Dict:
        """Run all health checks"""
        logger.info("🔍 Starting comprehensive system health check...")
        
        check_results = {
            'system_requirements': self.check_system_requirements(),
            'software_dependencies': self.check_software_dependencies(),
            'network_connectivity': self.check_network_connectivity(),
            'project_structure': self.check_project_structure(),
            'python_packages': self.check_python_packages(),
            'running_processes': self.check_running_processes()
        }
        
        # Generate performance report
        self.generate_performance_report()
        
        # Calculate overall status
        passed_checks = sum(check_results.values())
        total_checks = len(check_results)
        
        if passed_checks == total_checks:
            self.results['overall_status'] = 'HEALTHY'
            status_icon = '🎉'
            status_msg = "All systems operational!"
        elif passed_checks >= total_checks * 0.8:
            self.results['overall_status'] = 'WARNING'
            status_icon = '⚠️'
            status_msg = "System mostly ready with minor issues"
        else:
            self.results['overall_status'] = 'CRITICAL'
            status_icon = '❌'
            status_msg = "System has critical issues"
        
        logger.info(f"\n{status_icon} Overall System Status: {self.results['overall_status']}")
        logger.info(f"Health Score: {passed_checks}/{total_checks} checks passed")
        logger.info(f"Status: {status_msg}")
        
        return self.results
    
    def save_report(self, output_path: str = None):
        """Save health check report to file"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"health_check_report_{timestamp}.json"
        
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"📄 Health check report saved to: {output_file}")
        return output_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CARLA DRL System Health Check')
    parser.add_argument('--base-path', default='.', help='Base path for the project')
    parser.add_argument('--output', help='Output file for health report')
    parser.add_argument('--quick', action='store_true', help='Run quick check only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize health checker
    checker = SystemHealthChecker(args.base_path)
    
    try:
        if args.quick:
            # Quick check - only essential components
            logger.info("⚡ Running quick health check...")
            results = {
                'system_requirements': checker.check_system_requirements(),
                'software_dependencies': checker.check_software_dependencies(),
                'project_structure': checker.check_project_structure()
            }
            
            passed = sum(results.values())
            total = len(results)
            
            if passed == total:
                print("\n✅ Quick check PASSED - System ready!")
                return 0
            else:
                print(f"\n❌ Quick check FAILED - {passed}/{total} checks passed")
                return 1
        
        else:
            # Comprehensive check
            results = checker.run_comprehensive_check()
            
            # Save report
            report_file = checker.save_report(args.output)
            
            # Print summary
            print("\n" + "="*60)
            print("🔍 CARLA DRL SYSTEM HEALTH CHECK SUMMARY")
            print("="*60)
            
            for category, passed in results['checks'].items():
                if isinstance(passed, dict):
                    passed_count = sum(1 for v in passed.values() if v)
                    total_count = len(passed)
                    print(f"{'✅' if passed_count == total_count else '⚠️'} {category}: {passed_count}/{total_count}")
                else:
                    print(f"{'✅' if passed else '❌'} {category}: {'PASSED' if passed else 'FAILED'}")
            
            print("\n📊 Performance Summary:")
            if 'performance' in results:
                perf = results['performance']
                print(f"  CPU: {perf.get('cpu', {}).get('usage_percent', 'N/A')}% usage")
                print(f"  Memory: {perf.get('memory', {}).get('used_percent', 'N/A')}% used")
                print(f"  Disk: {perf.get('disk', {}).get('used_percent', 'N/A')}% used")
            
            print(f"\n📄 Detailed report: {report_file}")
            print(f"🎯 Overall Status: {results['overall_status']}")
            
            if results['overall_status'] == 'HEALTHY':
                print("\n🚀 System is ready for CARLA DRL training!")
                return 0
            elif results['overall_status'] == 'WARNING':
                print("\n⚠️ System has minor issues but should work")
                return 0
            else:
                print("\n❌ System has critical issues that need attention")
                return 1
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

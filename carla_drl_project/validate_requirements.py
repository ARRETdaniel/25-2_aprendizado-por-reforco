#!/usr/bin/env python3
"""
🔍 CARLA DRL System Requirements Validator
This script comprehensively validates all requirements for the DRL + CARLA + ROS 2 system
"""

import sys
import subprocess
import importlib
import os
import json
from datetime import datetime
import platform

class RequirementsValidator:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'python_packages': {},
            'ros2_packages': {},
            'carla_status': {},
            'gpu_info': {},
            'recommendations': [],
            'critical_issues': [],
            'next_steps': []
        }
        
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
        
    def print_status(self, item, status, details=""):
        icons = {"✅": "✅", "❌": "❌", "⚠️": "⚠️", "🔍": "🔍"}
        status_icon = icons.get(status, status)
        detail_str = f" - {details}" if details else ""
        print(f"{status_icon} {item:<30} {detail_str}")
        
    def check_system_info(self):
        """Check basic system information"""
        self.print_header("System Information")
        
        try:
            # OS Information
            os_info = platform.platform()
            python_version = platform.python_version()
            architecture = platform.architecture()[0]
            
            self.results['system_info'] = {
                'os': os_info,
                'python_version': python_version,
                'architecture': architecture,
                'hostname': platform.node()
            }
            
            self.print_status("Operating System", "✅", os_info)
            self.print_status("Python Version", "✅", python_version)
            self.print_status("Architecture", "✅", architecture)
            
            # Check if we're on Ubuntu 20.04 (optimal for ROS 2 Foxy)
            if "Ubuntu-20.04" in os_info:
                self.print_status("Ubuntu 20.04", "✅", "Optimal for ROS 2 Foxy")
            else:
                self.print_status("Ubuntu 20.04", "⚠️", "Non-optimal OS version")
                self.results['recommendations'].append("Consider using Ubuntu 20.04 for best ROS 2 Foxy compatibility")
                
        except Exception as e:
            self.print_status("System Info", "❌", f"Error: {e}")
            
    def check_python_packages(self):
        """Check critical Python packages"""
        self.print_header("Python Package Dependencies")
        
        critical_packages = [
            ('carla', 'CARLA Python API'),
            ('torch', 'PyTorch'),
            ('stable_baselines3', 'Stable-Baselines3'),
            ('gymnasium', 'Gymnasium'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('rclpy', 'ROS 2 Python'),
            ('matplotlib', 'Matplotlib'),
            ('tensorboard', 'TensorBoard')
        ]
        
        optional_packages = [
            ('scipy', 'SciPy'),
            ('pandas', 'Pandas'),
            ('yaml', 'PyYAML'),
            ('zmq', 'PyZMQ'),
        ]
        
        # Check critical packages
        for package, name in critical_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                self.results['python_packages'][package] = {
                    'status': 'installed',
                    'version': version
                }
                self.print_status(name, "✅", f"v{version}")
            except ImportError:
                self.results['python_packages'][package] = {
                    'status': 'missing',
                    'version': None
                }
                self.print_status(name, "❌", "Not installed")
                self.results['critical_issues'].append(f"Missing critical package: {name}")
                
        # Check optional packages
        print(f"\n📦 Optional Packages:")
        for package, name in optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'Unknown')
                self.results['python_packages'][package] = {
                    'status': 'installed',
                    'version': version
                }
                self.print_status(name, "✅", f"v{version}")
            except ImportError:
                self.results['python_packages'][package] = {
                    'status': 'missing',
                    'version': None
                }
                self.print_status(name, "⚠️", "Optional - not installed")
                
    def check_gpu_and_cuda(self):
        """Check GPU and CUDA availability"""
        self.print_header("GPU and CUDA Status")
        
        try:
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            self.results['gpu_info']['cuda_available'] = cuda_available
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                cuda_version = torch.version.cuda
                
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
                
                self.results['gpu_info'].update({
                    'device_count': device_count,
                    'current_device': current_device,
                    'device_name': device_name,
                    'cuda_version': cuda_version,
                    'memory_total_gb': memory_total,
                    'memory_allocated_gb': memory_allocated
                })
                
                self.print_status("CUDA Available", "✅", f"Version {cuda_version}")
                self.print_status("GPU Device", "✅", device_name)
                self.print_status("VRAM Total", "✅", f"{memory_total:.1f} GB")
                self.print_status("VRAM Used", "✅", f"{memory_allocated:.2f} GB")
                
                # Check if RTX 2060 with sufficient memory
                if "RTX 2060" in device_name:
                    if memory_total >= 5.5:
                        self.print_status("RTX 2060 Memory", "✅", "Sufficient for optimized training")
                    else:
                        self.print_status("RTX 2060 Memory", "⚠️", "Limited - requires optimization")
                        self.results['recommendations'].append("Use headless mode and low-resolution images for memory optimization")
                        
            else:
                self.print_status("CUDA Available", "❌", "No CUDA support")
                self.results['critical_issues'].append("CUDA not available - GPU acceleration disabled")
                
        except ImportError:
            self.print_status("PyTorch", "❌", "Not installed - cannot check CUDA")
            
        # Try to get nvidia-smi output
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')[0].split(', ')
                self.print_status("nvidia-smi", "✅", f"GPU: {gpu_info[0]}, VRAM: {gpu_info[1]}MiB")
            else:
                self.print_status("nvidia-smi", "❌", "Command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.print_status("nvidia-smi", "❌", "Not available")
            
    def check_ros2_status(self):
        """Check ROS 2 installation and packages"""
        self.print_header("ROS 2 Status")
        
        # Check if ROS 2 is sourced
        ros_distro = os.environ.get('ROS_DISTRO')
        if ros_distro:
            self.print_status("ROS 2 Environment", "✅", f"Distro: {ros_distro}")
            self.results['ros2_packages']['distro'] = ros_distro
        else:
            self.print_status("ROS 2 Environment", "❌", "Not sourced")
            self.results['critical_issues'].append("ROS 2 environment not sourced")
            
        # Try to check ROS 2 packages
        try:
            result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                packages = result.stdout.strip().split('\n')
                carla_packages = [pkg for pkg in packages if 'carla' in pkg.lower()]
                
                self.print_status("ROS 2 Commands", "✅", f"Found {len(packages)} packages")
                
                if carla_packages:
                    self.print_status("CARLA ROS Packages", "✅", f"Found {len(carla_packages)} packages")
                    for pkg in carla_packages:
                        self.print_status("  " + pkg, "✅", "")
                    self.results['ros2_packages']['carla_packages'] = carla_packages
                else:
                    self.print_status("CARLA ROS Packages", "❌", "Not found")
                    self.results['critical_issues'].append("CARLA ROS 2 bridge packages not installed")
                    
            else:
                self.print_status("ROS 2 Commands", "❌", "ros2 command failed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.print_status("ROS 2 Commands", "❌", "ros2 command not available")
            
    def check_carla_status(self):
        """Check CARLA installation and server status"""
        self.print_header("CARLA Status")
        
        # Check CARLA Python API
        try:
            import carla
            version = carla.Client('localhost', 2000, 1.0).get_client_version()
            self.results['carla_status']['python_api'] = True
            self.results['carla_status']['version'] = version
            self.print_status("CARLA Python API", "✅", f"Version {version}")
        except ImportError:
            self.print_status("CARLA Python API", "❌", "Not installed")
            self.results['carla_status']['python_api'] = False
            self.results['critical_issues'].append("CARLA Python API not available")
        except Exception as e:
            self.print_status("CARLA Python API", "✅", "Installed (server not running)")
            self.results['carla_status']['python_api'] = True
            
        # Check if CARLA server is running
        try:
            import carla
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)
            world = client.get_world()
            map_name = world.get_map().name
            self.print_status("CARLA Server", "✅", f"Running - Map: {map_name}")
            self.results['carla_status']['server_running'] = True
            self.results['carla_status']['current_map'] = map_name
        except Exception as e:
            self.print_status("CARLA Server", "❌", "Not running")
            self.results['carla_status']['server_running'] = False
            
        # Check for CARLA executable
        home_dir = os.path.expanduser("~")
        possible_carla_paths = [
            os.path.join(home_dir, "CarlaUE4.sh"),
            os.path.join(home_dir, "CARLA_0.9.16", "CarlaUE4.sh"),
            "/opt/carla-simulator/CarlaUE4.sh",
        ]
        
        carla_executable = None
        for path in possible_carla_paths:
            if os.path.exists(path):
                carla_executable = path
                break
                
        if carla_executable:
            self.print_status("CARLA Executable", "✅", carla_executable)
            self.results['carla_status']['executable_path'] = carla_executable
        else:
            self.print_status("CARLA Executable", "⚠️", "Not found in common locations")
            self.results['recommendations'].append("Locate CARLA executable path for server startup")
            
    def generate_recommendations(self):
        """Generate specific recommendations based on findings"""
        self.print_header("Recommendations & Next Steps")
        
        # Priority 1: Critical missing packages
        missing_critical = [pkg for pkg, info in self.results['python_packages'].items() 
                          if info.get('status') == 'missing' and pkg in ['torch', 'stable_baselines3', 'carla']]
        
        if missing_critical:
            self.print_status("CRITICAL", "❌", f"Install missing packages: {', '.join(missing_critical)}")
            self.results['next_steps'].append({
                'priority': 'CRITICAL',
                'action': 'Install missing Python packages',
                'command': f'pip3 install {" ".join(missing_critical)}'
            })
            
        # Priority 2: ROS 2 bridge
        if not self.results['ros2_packages'].get('carla_packages'):
            self.print_status("HIGH", "⚠️", "Install CARLA ROS 2 bridge packages")
            self.results['next_steps'].append({
                'priority': 'HIGH',
                'action': 'Install CARLA ROS 2 bridge',
                'command': 'sudo apt install ros-foxy-carla-ros-bridge'
            })
            
        # Priority 3: CARLA server
        if not self.results['carla_status'].get('server_running'):
            self.print_status("MEDIUM", "🔍", "Start CARLA server for testing")
            self.results['next_steps'].append({
                'priority': 'MEDIUM',
                'action': 'Start CARLA server',
                'command': './CarlaUE4.sh -RenderOffScreen'
            })
            
        # Memory optimization for RTX 2060
        gpu_info = self.results.get('gpu_info', {})
        if gpu_info.get('device_name') and 'RTX 2060' in gpu_info['device_name']:
            if gpu_info.get('memory_total_gb', 0) < 7:
                self.print_status("OPTIMIZATION", "⚠️", "Use memory optimization for RTX 2060")
                self.results['next_steps'].append({
                    'priority': 'OPTIMIZATION',
                    'action': 'Configure memory optimization',
                    'details': 'Use headless mode, low resolution, small batch sizes'
                })
                
    def save_report(self):
        """Save detailed report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"requirements_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📄 Detailed report saved to: {report_file}")
        return report_file
        
    def run_validation(self):
        """Run complete validation"""
        print("🔍 CARLA DRL System Requirements Validation")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.check_system_info()
        self.check_python_packages()
        self.check_gpu_and_cuda()
        self.check_ros2_status()
        self.check_carla_status()
        self.generate_recommendations()
        
        # Summary
        self.print_header("Validation Summary")
        
        critical_count = len(self.results['critical_issues'])
        recommendations_count = len(self.results['recommendations'])
        
        if critical_count == 0:
            self.print_status("Overall Status", "✅", "Ready for development!")
        elif critical_count <= 2:
            self.print_status("Overall Status", "⚠️", f"{critical_count} critical issues to resolve")
        else:
            self.print_status("Overall Status", "❌", f"{critical_count} critical issues - setup needed")
            
        self.print_status("Critical Issues", "📊", str(critical_count))
        self.print_status("Recommendations", "📊", str(recommendations_count))
        
        # Show critical issues
        if self.results['critical_issues']:
            print("\n🚨 Critical Issues:")
            for issue in self.results['critical_issues']:
                print(f"   ❌ {issue}")
                
        # Show next steps
        if self.results['next_steps']:
            print("\n🎯 Immediate Next Steps:")
            for i, step in enumerate(self.results['next_steps'], 1):
                priority = step['priority']
                action = step['action']
                command = step.get('command', step.get('details', ''))
                print(f"   {i}. [{priority}] {action}")
                if command:
                    print(f"      → {command}")
                    
        return self.save_report()

if __name__ == "__main__":
    validator = RequirementsValidator()
    report_file = validator.run_validation()
    
    print(f"\n🎯 Next: Run the installation script with:")
    print(f"   ./install_dependencies.sh")
    print(f"\n📋 For detailed analysis, see: {report_file}")

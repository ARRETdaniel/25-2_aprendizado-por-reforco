#!/usr/bin/env python3
"""
Debug and development utilities for CARLA DRL pipeline.
Provides diagnostic tools, performance monitoring, and troubleshooting utilities.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import zmq
import cv2
import psutil
import signal
import subprocess

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

@dataclass
class DiagnosticResult:
    """Container for diagnostic test results."""
    test_name: str
    status: str  # "PASS", "FAIL", "WARNING", "SKIP"
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None

class CarlaDRLDiagnostics:
    """Comprehensive diagnostic suite for CARLA DRL pipeline."""
    
    def __init__(self):
        self.results: List[DiagnosticResult] = []
        self.zmq_context = None
        self.ros_node = None
        
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete diagnostic suite."""
        print("🔍 Starting CARLA DRL Pipeline Diagnostics...")
        print("=" * 60)
        
        # System checks
        self._check_system_requirements()
        self._check_python_environments()
        self._check_dependencies()
        
        # CARLA checks
        self._check_carla_installation()
        self._check_carla_server()
        
        # ROS 2 checks
        self._check_ros2_environment()
        self._check_ros2_gateway()
        
        # ZeroMQ checks
        self._check_zeromq_communication()
        
        # Pipeline integration checks
        self._check_pipeline_integration()
        
        return self._generate_diagnostic_report()
    
    def _check_system_requirements(self):
        """Check system resources and requirements."""
        print("\n📊 System Requirements Check")
        print("-" * 30)
        
        # CPU check
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_count >= 4:
            status = "PASS" if cpu_usage < 80 else "WARNING"
            message = f"CPU: {cpu_count} cores, {cpu_usage:.1f}% usage"
        else:
            status = "WARNING"
            message = f"CPU: {cpu_count} cores (recommended: ≥4)"
        
        self.results.append(DiagnosticResult(
            "system_cpu", status, message, 0.0,
            {"cpu_count": cpu_count, "cpu_usage": cpu_usage}
        ))
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_usage = memory.percent
        
        if memory_gb >= 8:
            status = "PASS" if memory_usage < 80 else "WARNING"
            message = f"Memory: {memory_gb:.1f}GB total, {memory_usage:.1f}% used"
        else:
            status = "WARNING"
            message = f"Memory: {memory_gb:.1f}GB (recommended: ≥8GB)"
        
        self.results.append(DiagnosticResult(
            "system_memory", status, message, 0.0,
            {"memory_gb": memory_gb, "memory_usage": memory_usage}
        ))
        
        # GPU check (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                message = f"GPU: {gpu.name}, {gpu.memoryTotal}MB VRAM"
                status = "PASS"
            else:
                message = "No GPU detected (CPU training will be slow)"
                status = "WARNING"
        except ImportError:
            message = "GPU detection unavailable (install gputil)"
            status = "SKIP"
        
        self.results.append(DiagnosticResult(
            "system_gpu", status, message, 0.0
        ))
        
        print(f"✓ System check completed ({len([r for r in self.results if r.test_name.startswith('system')]) } tests)")
    
    def _check_python_environments(self):
        """Check Python environment configurations."""
        print("\n🐍 Python Environment Check")
        print("-" * 30)
        
        # Current Python version
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        message = f"Current Python: {current_version}"
        status = "PASS" if current_version >= "3.8" else "WARNING"
        
        self.results.append(DiagnosticResult(
            "python_version", status, message, 0.0,
            {"version": current_version}
        ))
        
        # Check for multiple Python installations
        python_paths = self._find_python_installations()
        if len(python_paths) > 1:
            message = f"Multiple Python installations found: {len(python_paths)}"
            status = "WARNING"
        else:
            message = "Single Python installation detected"
            status = "PASS"
        
        self.results.append(DiagnosticResult(
            "python_installations", status, message, 0.0,
            {"paths": python_paths}
        ))
        
        print(f"✓ Python environment check completed")
    
    def _check_dependencies(self):
        """Check required Python packages."""
        print("\n📦 Dependency Check")
        print("-" * 30)
        
        # Core dependencies
        core_deps = [
            "numpy", "opencv-python", "zmq", "pyyaml", "pydantic",
            "stable-baselines3", "tensorboard", "matplotlib"
        ]
        
        # ROS 2 dependencies
        ros_deps = ["rclpy", "sensor_msgs", "geometry_msgs", "std_msgs"]
        
        # CARLA dependencies
        carla_deps = ["carla"]  # Note: CARLA client requires specific installation
        
        all_deps = core_deps + ros_deps + carla_deps
        missing_deps = []
        
        for dep in all_deps:
            try:
                __import__(dep.replace("-", "_"))
                status = "PASS"
                message = f"{dep}: Available"
            except ImportError:
                status = "FAIL" if dep in core_deps else "WARNING"
                message = f"{dep}: Missing"
                missing_deps.append(dep)
            
            self.results.append(DiagnosticResult(
                f"dependency_{dep}", status, message, 0.0
            ))
        
        # Summary
        total_deps = len(all_deps)
        available_deps = total_deps - len(missing_deps)
        print(f"✓ Dependencies: {available_deps}/{total_deps} available")
        
        if missing_deps:
            print(f"⚠️  Missing: {', '.join(missing_deps)}")
    
    def _check_carla_installation(self):
        """Check CARLA installation and configuration."""
        print("\n🚗 CARLA Installation Check")
        print("-" * 30)
        
        # Check for CARLA binary
        carla_paths = [
            "C:/carla/CarlaUE4.exe",
            "./CarlaUE4/Binaries/Win64/CarlaUE4.exe",
            "./carla/CarlaUE4.exe"
        ]
        
        carla_found = False
        carla_path = None
        
        for path in carla_paths:
            if os.path.exists(path):
                carla_found = True
                carla_path = path
                break
        
        if carla_found:
            status = "PASS"
            message = f"CARLA executable found: {carla_path}"
        else:
            status = "FAIL"
            message = "CARLA executable not found in standard locations"
        
        self.results.append(DiagnosticResult(
            "carla_executable", status, message, 0.0,
            {"path": carla_path}
        ))
        
        # Check CARLA Python API
        try:
            import carla
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)
            
            try:
                world = client.get_world()
                status = "PASS"
                message = f"CARLA Python API: Connected to {world.get_map().name}"
            except Exception as e:
                status = "WARNING"
                message = f"CARLA Python API available but server not running: {str(e)}"
                
        except ImportError:
            status = "FAIL"
            message = "CARLA Python API not available"
        
        self.results.append(DiagnosticResult(
            "carla_python_api", status, message, 0.0
        ))
        
        print(f"✓ CARLA installation check completed")
    
    def _check_carla_server(self):
        """Check if CARLA server is running and accessible."""
        print("\n🌐 CARLA Server Check")
        print("-" * 30)
        
        try:
            import carla
            client = carla.Client('localhost', 2000)
            client.set_timeout(5.0)
            
            start_time = time.time()
            world = client.get_world()
            duration = (time.time() - start_time) * 1000
            
            # Get world info
            map_name = world.get_map().name
            actors = world.get_actors()
            
            status = "PASS"
            message = f"CARLA server responding: {map_name}, {len(actors)} actors"
            
            details = {
                "map_name": map_name,
                "actor_count": len(actors),
                "response_time_ms": duration
            }
            
        except Exception as e:
            status = "FAIL"
            message = f"CARLA server not accessible: {str(e)}"
            duration = 0.0
            details = None
        
        self.results.append(DiagnosticResult(
            "carla_server", status, message, duration, details
        ))
        
        print(f"✓ CARLA server check completed")
    
    def _check_ros2_environment(self):
        """Check ROS 2 environment and installation."""
        print("\n🤖 ROS 2 Environment Check")
        print("-" * 30)
        
        # Check ROS 2 installation
        ros_distro = os.environ.get('ROS_DISTRO')
        if ros_distro:
            status = "PASS"
            message = f"ROS 2 distro: {ros_distro}"
        else:
            status = "FAIL"
            message = "ROS_DISTRO environment variable not set"
        
        self.results.append(DiagnosticResult(
            "ros2_installation", status, message, 0.0,
            {"distro": ros_distro}
        ))
        
        # Check ROS 2 daemon
        try:
            result = subprocess.run(
                ['ros2', 'node', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                status = "PASS"
                message = f"ROS 2 daemon running: {len(result.stdout.strip().split())} nodes"
            else:
                status = "WARNING"
                message = "ROS 2 daemon not responding properly"
                
        except Exception as e:
            status = "FAIL"
            message = f"ROS 2 daemon check failed: {str(e)}"
        
        self.results.append(DiagnosticResult(
            "ros2_daemon", status, message, 0.0
        ))
        
        print(f"✓ ROS 2 environment check completed")
    
    def _check_ros2_gateway(self):
        """Check ROS 2 gateway node status."""
        print("\n🌉 ROS 2 Gateway Check")
        print("-" * 30)
        
        try:
            # Initialize ROS 2 client
            if not rclpy.ok():
                rclpy.init()
            
            node = rclpy.create_node('diagnostic_client')
            
            # Check for gateway node
            node_names = node.get_node_names()
            gateway_found = 'carla_gateway' in node_names
            
            if gateway_found:
                status = "PASS"
                message = "Gateway node is running"
                
                # Check topics
                topic_names = [name for name, _ in node.get_topic_names_and_types()]
                carla_topics = [t for t in topic_names if 'carla' in t.lower()]
                
                details = {
                    "node_found": True,
                    "carla_topics": carla_topics,
                    "topic_count": len(carla_topics)
                }
                
                if len(carla_topics) > 0:
                    message += f", {len(carla_topics)} CARLA topics"
                else:
                    status = "WARNING"
                    message += ", no CARLA topics found"
            else:
                status = "FAIL"
                message = "Gateway node not found"
                details = {"node_found": False}
            
            node.destroy_node()
            
        except Exception as e:
            status = "FAIL"
            message = f"ROS 2 gateway check failed: {str(e)}"
            details = None
        
        self.results.append(DiagnosticResult(
            "ros2_gateway", status, message, 0.0, details
        ))
        
        print(f"✓ ROS 2 gateway check completed")
    
    def _check_zeromq_communication(self):
        """Check ZeroMQ communication channels."""
        print("\n⚡ ZeroMQ Communication Check")
        print("-" * 30)
        
        try:
            # Initialize ZeroMQ context
            context = zmq.Context()
            
            # Test subscriber socket
            subscriber = context.socket(zmq.SUB)
            subscriber.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
            subscriber.connect("tcp://localhost:5555")
            subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            
            start_time = time.time()
            
            try:
                # Try to receive a message
                message = subscriber.recv_string()
                duration = (time.time() - start_time) * 1000
                
                status = "PASS"
                message_content = f"ZeroMQ communication active, received: {message[:50]}..."
                
                details = {
                    "message_received": True,
                    "response_time_ms": duration,
                    "message_preview": message[:100]
                }
                
            except zmq.Again:
                status = "WARNING"
                message_content = "ZeroMQ socket accessible but no messages received"
                duration = 2000.0
                details = {"message_received": False}
            
            subscriber.close()
            context.term()
            
        except Exception as e:
            status = "FAIL"
            message_content = f"ZeroMQ communication failed: {str(e)}"
            duration = 0.0
            details = None
        
        self.results.append(DiagnosticResult(
            "zeromq_communication", status, message_content, duration, details
        ))
        
        print(f"✓ ZeroMQ communication check completed")
    
    def _check_pipeline_integration(self):
        """Check end-to-end pipeline integration."""
        print("\n🔗 Pipeline Integration Check")
        print("-" * 30)
        
        # This would run a quick integration test
        # For now, we'll check if all components are ready
        
        component_status = {}
        
        # Check each component
        components = ['carla_server', 'carla_python_api', 'ros2_gateway', 'zeromq_communication']
        
        for component in components:
            result = next((r for r in self.results if r.test_name == component), None)
            if result:
                component_status[component] = result.status == "PASS"
        
        all_ready = all(component_status.values())
        ready_count = sum(component_status.values())
        total_count = len(component_status)
        
        if all_ready:
            status = "PASS"
            message = "All pipeline components ready for integration"
        elif ready_count > total_count // 2:
            status = "WARNING"
            message = f"Partial integration possible: {ready_count}/{total_count} components ready"
        else:
            status = "FAIL"
            message = f"Integration not possible: {ready_count}/{total_count} components ready"
        
        self.results.append(DiagnosticResult(
            "pipeline_integration", status, message, 0.0,
            {"component_status": component_status, "ready_count": ready_count}
        ))
        
        print(f"✓ Pipeline integration check completed")
    
    def _find_python_installations(self) -> List[str]:
        """Find all Python installations on the system."""
        python_paths = []
        
        # Common Python installation paths
        common_paths = [
            "C:/Python*/python.exe",
            "C:/Users/*/AppData/Local/Programs/Python/Python*/python.exe",
            "C:/ProgramData/Anaconda3/python.exe",
            "C:/ProgramData/Miniconda3/python.exe"
        ]
        
        import glob
        for pattern in common_paths:
            python_paths.extend(glob.glob(pattern))
        
        # Add current Python
        python_paths.append(sys.executable)
        
        return list(set(python_paths))  # Remove duplicates
    
    def _generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC REPORT SUMMARY")
        print("=" * 60)
        
        # Count results by status
        status_counts = {
            "PASS": len([r for r in self.results if r.status == "PASS"]),
            "WARNING": len([r for r in self.results if r.status == "WARNING"]),
            "FAIL": len([r for r in self.results if r.status == "FAIL"]),
            "SKIP": len([r for r in self.results if r.status == "SKIP"])
        }
        
        total_tests = len(self.results)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ PASS: {status_counts['PASS']}")
        print(f"⚠️  WARNING: {status_counts['WARNING']}")
        print(f"❌ FAIL: {status_counts['FAIL']}")
        print(f"⏭️  SKIP: {status_counts['SKIP']}")
        
        # Overall status
        if status_counts['FAIL'] == 0 and status_counts['WARNING'] <= 2:
            overall_status = "HEALTHY"
            print(f"\n🎉 Overall Status: {overall_status}")
            print("✓ Pipeline is ready for operation!")
        elif status_counts['FAIL'] <= 2:
            overall_status = "PARTIAL"
            print(f"\n⚠️  Overall Status: {overall_status}")
            print("⚠️  Some issues detected, partial functionality available")
        else:
            overall_status = "CRITICAL"
            print(f"\n❌ Overall Status: {overall_status}")
            print("❌ Critical issues detected, pipeline not operational")
        
        # Detailed results
        print("\n📊 DETAILED RESULTS:")
        print("-" * 60)
        
        for result in self.results:
            status_icon = {
                "PASS": "✅",
                "WARNING": "⚠️ ",
                "FAIL": "❌",
                "SKIP": "⏭️ "
            }[result.status]
            
            print(f"{status_icon} {result.test_name}: {result.message}")
            
            if result.details and result.status in ["WARNING", "FAIL"]:
                for key, value in result.details.items():
                    print(f"    {key}: {value}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 60)
        
        fail_results = [r for r in self.results if r.status == "FAIL"]
        warning_results = [r for r in self.results if r.status == "WARNING"]
        
        if fail_results:
            print("Critical Issues (must fix):")
            for result in fail_results:
                print(f"  • {result.test_name}: {result.message}")
        
        if warning_results:
            print("Improvements (recommended):")
            for result in warning_results:
                print(f"  • {result.test_name}: {result.message}")
        
        if not fail_results and not warning_results:
            print("✓ No issues detected. System is ready!")
        
        return {
            "overall_status": overall_status,
            "status_counts": status_counts,
            "total_tests": total_tests,
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in self.results
            ],
            "timestamp": time.time()
        }

class CarlaDebugTools:
    """Advanced debugging tools for CARLA DRL development."""
    
    def __init__(self):
        self.recording_data = []
        self.performance_metrics = {}
        
    def start_performance_monitoring(self):
        """Start system performance monitoring."""
        print("📊 Starting performance monitoring...")
        
        # Monitor CPU, memory, GPU usage during training
        # Log to files for analysis
        pass
    
    def visualize_sensor_data(self, data_path: str):
        """Visualize recorded sensor data."""
        print(f"🖼️  Visualizing sensor data from {data_path}")
        
        # Load and display camera images, lidar point clouds
        # Create visualization dashboard
        pass
    
    def analyze_training_logs(self, log_path: str):
        """Analyze training performance logs."""
        print(f"📈 Analyzing training logs from {log_path}")
        
        # Parse TensorBoard logs
        # Generate performance reports
        # Identify training issues
        pass
    
    def debug_action_space(self, model_path: str):
        """Debug agent action space and decision making."""
        print(f"🎯 Debugging action space for model {model_path}")
        
        # Visualize action distributions
        # Analyze decision patterns
        # Identify action space issues
        pass

def main():
    """Main entry point for diagnostic tools."""
    parser = argparse.ArgumentParser(description="CARLA DRL Pipeline Diagnostics")
    
    parser.add_argument(
        '--mode', 
        choices=['diagnostics', 'performance', 'debug', 'all'],
        default='diagnostics',
        help='Diagnostic mode to run'
    )
    
    parser.add_argument(
        '--output',
        default='diagnostic_report.json',
        help='Output file for diagnostic report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['diagnostics', 'all']:
        diagnostics = CarlaDRLDiagnostics()
        report = diagnostics.run_full_diagnostics()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {args.output}")
    
    if args.mode in ['debug', 'all']:
        debug_tools = CarlaDebugTools()
        # Run debugging tools based on additional arguments
        pass
    
    print("\n🎯 Diagnostics completed!")

if __name__ == "__main__":
    main()

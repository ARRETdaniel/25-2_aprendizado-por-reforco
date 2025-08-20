#!/usr/bin/env python3
"""
Health monitoring and diagnostics system for CARLA DRL Pipeline
Comprehensive health checks, performance monitoring, and troubleshooting tools
"""

import os
import sys
import time
import json
import logging
import subprocess
import requests
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status for a component"""
    component: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ComponentHealthChecker:
    """Health checker for individual components"""
    
    def __init__(self):
        self.checks = {
            "carla-server": self._check_carla_server,
            "ros2-gateway": self._check_ros2_gateway,
            "drl-trainer": self._check_drl_trainer,
            "monitoring": self._check_monitoring,
            "redis": self._check_redis,
            "postgres": self._check_postgres
        }
    
    def check_component(self, component: str) -> HealthStatus:
        """Check health of a specific component"""
        try:
            if component in self.checks:
                return self.checks[component]()
            else:
                return HealthStatus(
                    component=component,
                    status="unknown",
                    message=f"No health check defined for {component}",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthStatus(
                component=component,
                status="critical",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_carla_server(self) -> HealthStatus:
        """Check CARLA server health"""
        try:
            # Check if container is running
            container_status = self._get_container_status("carla-server")
            if not container_status["running"]:
                return HealthStatus(
                    component="carla-server",
                    status="critical",
                    message="Container not running",
                    timestamp=datetime.now()
                )
            
            # Check CARLA API endpoint
            try:
                response = requests.get("http://localhost:2000", timeout=5)
                if response.status_code == 200:
                    status = "healthy"
                    message = "CARLA server responding"
                else:
                    status = "warning"
                    message = f"CARLA server returned status {response.status_code}"
            except requests.RequestException:
                status = "warning"
                message = "CARLA server not responding to HTTP requests"
            
            # Get performance metrics
            metrics = self._get_container_metrics("carla-server")
            
            return HealthStatus(
                component="carla-server",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
        except Exception as e:
            return HealthStatus(
                component="carla-server",
                status="critical",
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_ros2_gateway(self) -> HealthStatus:
        """Check ROS 2 gateway health"""
        try:
            container_status = self._get_container_status("ros2-gateway")
            if not container_status["running"]:
                return HealthStatus(
                    component="ros2-gateway",
                    status="critical",
                    message="Container not running",
                    timestamp=datetime.now()
                )
            
            # Check ROS 2 nodes
            cmd = ["docker-compose", "exec", "-T", "ros2-gateway", "ros2", "node", "list"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                nodes = result.stdout.strip().split('\n')
                active_nodes = [node for node in nodes if node.strip()]
                
                if len(active_nodes) >= 2:  # Expect at least carla_client and gateway nodes
                    status = "healthy"
                    message = f"ROS 2 gateway active with {len(active_nodes)} nodes"
                else:
                    status = "warning"
                    message = f"ROS 2 gateway running but only {len(active_nodes)} nodes active"
                
                metrics = {
                    "active_nodes": len(active_nodes),
                    "node_list": active_nodes
                }
            else:
                status = "warning"
                message = "Could not query ROS 2 nodes"
                metrics = {}
            
            # Add container metrics
            container_metrics = self._get_container_metrics("ros2-gateway")
            metrics.update(container_metrics)
            
            return HealthStatus(
                component="ros2-gateway",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
        except Exception as e:
            return HealthStatus(
                component="ros2-gateway",
                status="critical",
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_drl_trainer(self) -> HealthStatus:
        """Check DRL trainer health"""
        try:
            container_status = self._get_container_status("drl-trainer")
            if not container_status["running"]:
                return HealthStatus(
                    component="drl-trainer",
                    status="critical",
                    message="Container not running",
                    timestamp=datetime.now()
                )
            
            # Check training process
            metrics = {}
            
            # Check if training is active
            cmd = ["docker-compose", "exec", "-T", "drl-trainer", "pgrep", "-f", "python.*train"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            training_active = result.returncode == 0
            
            # Check GPU utilization if available
            try:
                cmd = ["docker-compose", "exec", "-T", "drl-trainer", "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_util = float(result.stdout.strip())
                    metrics["gpu_utilization"] = gpu_util
            except:
                pass
            
            # Check recent checkpoint
            try:
                cmd = ["docker-compose", "exec", "-T", "drl-trainer", "find", "/opt/checkpoints", "-name", "*.pth", "-mtime", "-1"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                recent_checkpoints = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                metrics["recent_checkpoints"] = recent_checkpoints
            except:
                metrics["recent_checkpoints"] = 0
            
            # Determine status
            if training_active:
                status = "healthy"
                message = "DRL training active"
            elif metrics.get("recent_checkpoints", 0) > 0:
                status = "warning"
                message = "DRL training not active but recent checkpoints found"
            else:
                status = "warning"
                message = "DRL training not active and no recent checkpoints"
            
            # Add container metrics
            container_metrics = self._get_container_metrics("drl-trainer")
            metrics.update(container_metrics)
            
            return HealthStatus(
                component="drl-trainer",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
        except Exception as e:
            return HealthStatus(
                component="drl-trainer",
                status="critical",
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_monitoring(self) -> HealthStatus:
        """Check monitoring stack health"""
        try:
            # Check Prometheus
            prometheus_healthy = False
            try:
                response = requests.get("http://localhost:9090/-/healthy", timeout=5)
                prometheus_healthy = response.status_code == 200
            except:
                pass
            
            # Check Grafana
            grafana_healthy = False
            try:
                response = requests.get("http://localhost:3000/api/health", timeout=5)
                grafana_healthy = response.status_code == 200
            except:
                pass
            
            metrics = {
                "prometheus_healthy": prometheus_healthy,
                "grafana_healthy": grafana_healthy
            }
            
            if prometheus_healthy and grafana_healthy:
                status = "healthy"
                message = "Monitoring stack fully operational"
            elif prometheus_healthy or grafana_healthy:
                status = "warning"
                message = "Monitoring stack partially operational"
            else:
                status = "critical"
                message = "Monitoring stack not responding"
            
            return HealthStatus(
                component="monitoring",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics
            )
        except Exception as e:
            return HealthStatus(
                component="monitoring",
                status="critical",
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_redis(self) -> HealthStatus:
        """Check Redis health"""
        try:
            cmd = ["docker-compose", "exec", "-T", "redis", "redis-cli", "ping"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and "PONG" in result.stdout:
                # Get Redis info
                cmd = ["docker-compose", "exec", "-T", "redis", "redis-cli", "info", "memory"]
                info_result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                metrics = {}
                if info_result.returncode == 0:
                    for line in info_result.stdout.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            if key in ['used_memory_human', 'used_memory_peak_human']:
                                metrics[key] = value.strip()
                
                return HealthStatus(
                    component="redis",
                    status="healthy",
                    message="Redis responding to commands",
                    timestamp=datetime.now(),
                    metrics=metrics
                )
            else:
                return HealthStatus(
                    component="redis",
                    status="critical",
                    message="Redis not responding to ping",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthStatus(
                component="redis",
                status="critical",
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_postgres(self) -> HealthStatus:
        """Check PostgreSQL health"""
        try:
            cmd = ["docker-compose", "exec", "-T", "postgres", "pg_isready", "-U", "carla_user"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Get database size
                cmd = ["docker-compose", "exec", "-T", "postgres", "psql", "-U", "carla_user", "-d", "carla_db", "-c", "SELECT pg_size_pretty(pg_database_size('carla_db'));"]
                size_result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                metrics = {}
                if size_result.returncode == 0:
                    lines = size_result.stdout.strip().split('\n')
                    if len(lines) >= 3:
                        metrics["database_size"] = lines[2].strip()
                
                return HealthStatus(
                    component="postgres",
                    status="healthy",
                    message="PostgreSQL accepting connections",
                    timestamp=datetime.now(),
                    metrics=metrics
                )
            else:
                return HealthStatus(
                    component="postgres",
                    status="critical",
                    message="PostgreSQL not ready",
                    timestamp=datetime.now()
                )
        except Exception as e:
            return HealthStatus(
                component="postgres",
                status="critical",
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _get_container_status(self, service_name: str) -> Dict[str, Any]:
        """Get container status information"""
        try:
            cmd = ["docker-compose", "ps", "-q", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                container_id = result.stdout.strip()
                
                # Get container details
                cmd = ["docker", "inspect", container_id]
                inspect_result = subprocess.run(cmd, capture_output=True, text=True)
                
                if inspect_result.returncode == 0:
                    container_info = json.loads(inspect_result.stdout)[0]
                    state = container_info["State"]
                    
                    return {
                        "running": state["Running"],
                        "status": state["Status"],
                        "started_at": state.get("StartedAt"),
                        "finished_at": state.get("FinishedAt"),
                        "exit_code": state.get("ExitCode")
                    }
            
            return {"running": False, "status": "not found"}
        except Exception:
            return {"running": False, "status": "unknown"}
    
    def _get_container_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get container resource metrics"""
        try:
            cmd = ["docker", "stats", "--no-stream", "--format", "json", f"$(docker-compose ps -q {service_name})"]
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                stats = json.loads(result.stdout)
                return {
                    "cpu_percent": stats.get("CPUPerc", "0%").replace("%", ""),
                    "memory_usage": stats.get("MemUsage", "0B / 0B"),
                    "memory_percent": stats.get("MemPerc", "0%").replace("%", ""),
                    "network_io": stats.get("NetIO", "0B / 0B"),
                    "block_io": stats.get("BlockIO", "0B / 0B")
                }
            
            return {}
        except Exception:
            return {}

class SystemHealthMonitor:
    """System-wide health monitoring"""
    
    def __init__(self):
        self.component_checker = ComponentHealthChecker()
        self.monitoring = True
        self.health_history = []
        self.performance_history = []
    
    def get_system_health(self) -> Dict[str, HealthStatus]:
        """Get health status for all components"""
        components = [
            "carla-server", "ros2-gateway", "drl-trainer",
            "monitoring", "redis", "postgres"
        ]
        
        health_status = {}
        for component in components:
            health_status[component] = self.component_checker.check_component(component)
        
        return health_status
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get system performance metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        
        # GPU metrics (if available)
        gpu_percent = 0.0
        gpu_memory_percent = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_percent = gpu.load * 100
                gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except ImportError:
            pass
        
        return PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=(disk.used / disk.total) * 100,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            network_sent_mb=network.bytes_sent / 1024 / 1024,
            network_recv_mb=network.bytes_recv / 1024 / 1024
        )
    
    def monitor_continuously(self, interval: int = 60):
        """Monitor system health continuously"""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        while self.monitoring:
            try:
                # Collect health status
                health_status = self.get_system_health()
                self.health_history.append({
                    "timestamp": datetime.now(),
                    "status": health_status
                })
                
                # Collect performance metrics
                performance = self.get_performance_metrics()
                self.performance_history.append(performance)
                
                # Keep only recent history (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.health_history = [
                    h for h in self.health_history 
                    if h["timestamp"] > cutoff_time
                ]
                self.performance_history = [
                    p for p in self.performance_history 
                    if p.timestamp > cutoff_time
                ]
                
                # Log critical issues
                critical_components = [
                    name for name, status in health_status.items()
                    if status.status == "critical"
                ]
                
                if critical_components:
                    logger.error(f"Critical components detected: {', '.join(critical_components)}")
                
                # Check performance thresholds
                if performance.cpu_percent > 90:
                    logger.warning(f"High CPU usage: {performance.cpu_percent:.1f}%")
                
                if performance.memory_percent > 90:
                    logger.warning(f"High memory usage: {performance.memory_percent:.1f}%")
                
                if performance.disk_percent > 90:
                    logger.warning(f"High disk usage: {performance.disk_percent:.1f}%")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        logger.info("Monitoring stopped")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        current_health = self.get_system_health()
        current_performance = self.get_performance_metrics()
        
        # Calculate health summary
        healthy_count = sum(1 for status in current_health.values() if status.status == "healthy")
        warning_count = sum(1 for status in current_health.values() if status.status == "warning")
        critical_count = sum(1 for status in current_health.values() if status.status == "critical")
        
        overall_status = "healthy"
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        
        # Performance trends (if history available)
        trends = {}
        if len(self.performance_history) > 1:
            recent_perf = self.performance_history[-10:]  # Last 10 measurements
            
            cpu_trend = np.mean([p.cpu_percent for p in recent_perf])
            memory_trend = np.mean([p.memory_percent for p in recent_perf])
            
            trends = {
                "avg_cpu_percent": cpu_trend,
                "avg_memory_percent": memory_trend,
                "measurements": len(recent_perf)
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "healthy": healthy_count,
                "warning": warning_count,
                "critical": critical_count,
                "total": len(current_health)
            },
            "components": {
                name: status.to_dict() 
                for name, status in current_health.items()
            },
            "performance": {
                "cpu_percent": current_performance.cpu_percent,
                "memory_percent": current_performance.memory_percent,
                "disk_percent": current_performance.disk_percent,
                "gpu_percent": current_performance.gpu_percent,
                "gpu_memory_percent": current_performance.gpu_memory_percent
            },
            "trends": trends,
            "recommendations": self._generate_recommendations(current_health, current_performance)
        }
    
    def _generate_recommendations(self, health_status: Dict[str, HealthStatus], 
                                performance: PerformanceMetrics) -> List[str]:
        """Generate recommendations based on current status"""
        recommendations = []
        
        # Health-based recommendations
        for name, status in health_status.items():
            if status.status == "critical":
                recommendations.append(f"URGENT: Investigate {name} - {status.message}")
            elif status.status == "warning":
                recommendations.append(f"Check {name} - {status.message}")
        
        # Performance-based recommendations
        if performance.cpu_percent > 80:
            recommendations.append("Consider scaling DRL training workers or reducing batch size")
        
        if performance.memory_percent > 80:
            recommendations.append("Monitor memory usage - consider increasing available memory")
        
        if performance.disk_percent > 80:
            recommendations.append("Clean up old logs and checkpoints to free disk space")
        
        if performance.gpu_percent < 20 and health_status.get("drl-trainer", {}).status == "healthy":
            recommendations.append("GPU utilization is low - check if training is using GPU acceleration")
        
        return recommendations

def main():
    """Main health monitoring CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CARLA DRL Pipeline Health Monitor")
    parser.add_argument("action", choices=[
        "check", "monitor", "report", "component"
    ], help="Action to perform")
    parser.add_argument("--component", help="Specific component to check")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Monitoring interval in seconds")
    parser.add_argument("--output", type=Path, 
                       help="Output file for reports")
    
    args = parser.parse_args()
    
    monitor = SystemHealthMonitor()
    
    try:
        if args.action == "check":
            health_status = monitor.get_system_health()
            
            print("\n=== CARLA DRL Pipeline Health Check ===")
            for name, status in health_status.items():
                status_icon = {
                    "healthy": "✓",
                    "warning": "⚠",
                    "critical": "✗",
                    "unknown": "?"
                }.get(status.status, "?")
                
                print(f"{status_icon} {name}: {status.status.upper()} - {status.message}")
        
        elif args.action == "component":
            if not args.component:
                print("Error: --component is required for component action")
                sys.exit(1)
            
            checker = ComponentHealthChecker()
            status = checker.check_component(args.component)
            
            print(f"\n=== {args.component} Health Check ===")
            print(f"Status: {status.status.upper()}")
            print(f"Message: {status.message}")
            print(f"Timestamp: {status.timestamp}")
            
            if status.metrics:
                print("\nMetrics:")
                for key, value in status.metrics.items():
                    print(f"  {key}: {value}")
        
        elif args.action == "monitor":
            print(f"Starting continuous monitoring (Ctrl+C to stop)")
            
            def signal_handler(sig, frame):
                monitor.stop_monitoring()
                sys.exit(0)
            
            import signal
            signal.signal(signal.SIGINT, signal_handler)
            
            monitor_thread = threading.Thread(
                target=monitor.monitor_continuously,
                args=(args.interval,)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            try:
                monitor_thread.join()
            except KeyboardInterrupt:
                monitor.stop_monitoring()
        
        elif args.action == "report":
            report = monitor.generate_health_report()
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Health report saved to {args.output}")
            else:
                print(json.dumps(report, indent=2))
    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

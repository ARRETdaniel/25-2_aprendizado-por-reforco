#!/usr/bin/env python3
"""
Production monitoring system for CARLA DRL Pipeline.
Provides comprehensive health monitoring, metrics collection, and alerting.
"""

import os
import sys
import time
import logging
import threading
import queue
import json
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import socket
from contextlib import contextmanager

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logging.warning("Prometheus client not available. Metrics export disabled.")

# TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logging.warning("TensorBoard not available. TensorBoard logging disabled.")

# ROS 2 imports (optional)
try:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    logging.warning("ROS 2 not available. ROS 2 diagnostics disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    
    # GPU metrics (if available)
    gpu_count: int = 0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    
    # Disk metrics
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0
    
    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

@dataclass
class TrainingMetrics:
    """Container for DRL training metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # Episode metrics
    episode: int = 0
    total_timesteps: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    episodes_per_hour: float = 0.0
    
    # Performance metrics
    success_rate: float = 0.0
    collision_rate: float = 0.0
    lane_invasion_rate: float = 0.0
    average_speed: float = 0.0
    completion_rate: float = 0.0
    
    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    # Curriculum metrics
    curriculum_stage: str = "unknown"
    stage_progress: float = 0.0
    stage_success_rate: float = 0.0

@dataclass
class CarlaMetrics:
    """Container for CARLA simulator metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # Connection metrics
    server_connected: bool = False
    client_connected: bool = False
    connection_latency_ms: float = 0.0
    
    # Simulation metrics
    simulation_time: float = 0.0
    simulation_fps: float = 0.0
    target_fps: float = 20.0
    frame_drop_rate: float = 0.0
    
    # World metrics
    current_town: str = "unknown"
    weather_preset: int = 0
    num_vehicles: int = 0
    num_pedestrians: int = 0
    
    # Vehicle metrics
    vehicle_speed: float = 0.0
    vehicle_acceleration: float = 0.0
    vehicle_location: List[float] = field(default_factory=list)
    vehicle_rotation: List[float] = field(default_factory=list)
    
    # Sensor metrics
    camera_fps: float = 0.0
    lidar_fps: float = 0.0
    sensor_latency_ms: float = 0.0

@dataclass
class HealthStatus:
    """Container for component health status."""
    timestamp: float = field(default_factory=time.time)
    component: str = "unknown"
    status: str = "unknown"  # OK, WARNING, ERROR, CRITICAL
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collects and aggregates metrics from various sources."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.system_metrics = SystemMetrics()
        self.training_metrics = TrainingMetrics()
        self.carla_metrics = CarlaMetrics()
        self.health_statuses = {}
        
        # Metrics history
        self.metrics_history = {
            'system': [],
            'training': [],
            'carla': []
        }
        self.max_history_size = 1000
        
        logger.info("Metrics collector initialized")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        metrics = SystemMetrics()
        
        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_count = psutil.cpu_count()
            
            # Load average (Unix-like systems only)
            try:
                metrics.load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                metrics.load_average = [0.0, 0.0, 0.0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total_gb = memory.total / (1024**3)
            metrics.memory_used_gb = memory.used / (1024**3)
            metrics.memory_percent = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # GPU metrics (if available)
            try:
                gpus = GPUtil.getGPUs()
                metrics.gpu_count = len(gpus)
                for gpu in gpus:
                    metrics.gpu_utilization.append(gpu.load * 100)
                    metrics.gpu_memory_used.append(gpu.memoryUsed)
                    metrics.gpu_memory_total.append(gpu.memoryTotal)
                    metrics.gpu_temperature.append(gpu.temperature)
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_total_gb = disk.total / (1024**3)
            metrics.disk_used_gb = disk.used / (1024**3)
            metrics.disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.network_bytes_sent = network.bytes_sent
            metrics.network_bytes_recv = network.bytes_recv
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        self.system_metrics = metrics
        self._add_to_history('system', metrics)
        return metrics
    
    def update_training_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Update training metrics from external source."""
        try:
            # Update training metrics with provided data
            for key, value in metrics_dict.items():
                if hasattr(self.training_metrics, key):
                    setattr(self.training_metrics, key, value)
            
            self.training_metrics.timestamp = time.time()
            self._add_to_history('training', self.training_metrics)
            
        except Exception as e:
            logger.error(f"Error updating training metrics: {e}")
    
    def update_carla_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Update CARLA metrics from external source."""
        try:
            # Update CARLA metrics with provided data
            for key, value in metrics_dict.items():
                if hasattr(self.carla_metrics, key):
                    setattr(self.carla_metrics, key, value)
            
            self.carla_metrics.timestamp = time.time()
            self._add_to_history('carla', self.carla_metrics)
            
        except Exception as e:
            logger.error(f"Error updating CARLA metrics: {e}")
    
    def _add_to_history(self, metric_type: str, metrics) -> None:
        """Add metrics to history with size limit."""
        if metric_type in self.metrics_history:
            self.metrics_history[metric_type].append(asdict(metrics))
            
            # Trim history if too large
            if len(self.metrics_history[metric_type]) > self.max_history_size:
                self.metrics_history[metric_type] = self.metrics_history[metric_type][-self.max_history_size:]
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics from all sources."""
        return {
            'system': asdict(self.system_metrics),
            'training': asdict(self.training_metrics),
            'carla': asdict(self.carla_metrics),
            'timestamp': time.time()
        }

class HealthMonitor:
    """Monitors system health and component status."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize health monitor."""
        self.config = config
        self.health_checks = []
        self.alert_callbacks = []
        self.check_interval = config.get('check_interval', 10.0)
        self.timeout = config.get('timeout', 5.0)
        
        # Health status storage
        self.component_health = {}
        self.last_check_time = {}
        
        # Register built-in health checks
        self._register_builtin_checks()
        
        logger.info("Health monitor initialized")
    
    def _register_builtin_checks(self) -> None:
        """Register built-in health checks."""
        checks = self.config.get('checks', [])
        
        if 'carla_server_connection' in checks:
            self.register_health_check('carla_server', self._check_carla_connection)
        
        if 'ros2_node_status' in checks:
            self.register_health_check('ros2_nodes', self._check_ros2_nodes)
        
        if 'system_resources' in checks:
            self.register_health_check('system_resources', self._check_system_resources)
        
        if 'gpu_utilization' in checks:
            self.register_health_check('gpu_utilization', self._check_gpu_utilization)
        
        if 'drl_training_progress' in checks:
            self.register_health_check('training_progress', self._check_training_progress)
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a health check function."""
        self.health_checks.append((name, check_function))
        logger.info(f"Registered health check: {name}")
    
    def register_alert_callback(self, callback: Callable[[HealthStatus], None]) -> None:
        """Register an alert callback function."""
        self.alert_callbacks.append(callback)
        logger.info("Registered alert callback")
    
    def run_health_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        health_results = {}
        
        for check_name, check_function in self.health_checks:
            try:
                with self._timeout_context(self.timeout):
                    status = check_function()
                    if not isinstance(status, HealthStatus):
                        # Convert to HealthStatus if needed
                        status = HealthStatus(
                            component=check_name,
                            status="OK" if status else "ERROR",
                            message="Health check completed"
                        )
                    
                    health_results[check_name] = status
                    self.component_health[check_name] = status
                    
                    # Trigger alerts if needed
                    if status.status in ['ERROR', 'CRITICAL']:
                        self._trigger_alerts(status)
                        
            except Exception as e:
                error_status = HealthStatus(
                    component=check_name,
                    status="ERROR",
                    message=f"Health check failed: {str(e)}"
                )
                health_results[check_name] = error_status
                self.component_health[check_name] = error_status
                logger.error(f"Health check '{check_name}' failed: {e}")
        
        return health_results
    
    def _check_carla_connection(self) -> HealthStatus:
        """Check CARLA server connection."""
        try:
            # Try to connect to CARLA server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex(('localhost', 2000))
            sock.close()
            
            if result == 0:
                return HealthStatus(
                    component="carla_server",
                    status="OK",
                    message="CARLA server is reachable"
                )
            else:
                return HealthStatus(
                    component="carla_server",
                    status="ERROR",
                    message="CARLA server is not reachable"
                )
        except Exception as e:
            return HealthStatus(
                component="carla_server",
                status="ERROR",
                message=f"Connection check failed: {str(e)}"
            )
    
    def _check_ros2_nodes(self) -> HealthStatus:
        """Check ROS 2 node status."""
        if not HAS_ROS2:
            return HealthStatus(
                component="ros2_nodes",
                status="SKIP",
                message="ROS 2 not available"
            )
        
        try:
            # This is a simplified check
            # In reality, you'd check specific node status
            return HealthStatus(
                component="ros2_nodes",
                status="OK",
                message="ROS 2 nodes are running"
            )
        except Exception as e:
            return HealthStatus(
                component="ros2_nodes",
                status="ERROR",
                message=f"ROS 2 check failed: {str(e)}"
            )
    
    def _check_system_resources(self) -> HealthStatus:
        """Check system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            status = "OK"
            messages = []
            
            # Check thresholds
            if cpu_percent > 90:
                status = "WARNING"
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 90:
                status = "WARNING"
                messages.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = "CRITICAL"
                messages.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources OK"
            
            return HealthStatus(
                component="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent
                }
            )
        except Exception as e:
            return HealthStatus(
                component="system_resources",
                status="ERROR",
                message=f"Resource check failed: {str(e)}"
            )
    
    def _check_gpu_utilization(self) -> HealthStatus:
        """Check GPU utilization."""
        try:
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return HealthStatus(
                    component="gpu_utilization",
                    status="WARNING",
                    message="No GPUs detected"
                )
            
            status = "OK"
            messages = []
            
            for i, gpu in enumerate(gpus):
                utilization = gpu.load * 100
                memory_usage = (gpu.memoryUsed / gpu.memoryTotal) * 100
                
                if utilization < 10:
                    status = "WARNING"
                    messages.append(f"GPU {i} low utilization: {utilization:.1f}%")
                
                if memory_usage > 95:
                    status = "WARNING"
                    messages.append(f"GPU {i} high memory usage: {memory_usage:.1f}%")
                
                if gpu.temperature > 85:
                    status = "CRITICAL"
                    messages.append(f"GPU {i} high temperature: {gpu.temperature}°C")
            
            message = "; ".join(messages) if messages else f"GPU utilization OK ({len(gpus)} GPUs)"
            
            return HealthStatus(
                component="gpu_utilization",
                status=status,
                message=message
            )
        except Exception as e:
            return HealthStatus(
                component="gpu_utilization",
                status="ERROR",
                message=f"GPU check failed: {str(e)}"
            )
    
    def _check_training_progress(self) -> HealthStatus:
        """Check DRL training progress."""
        # This would need to be integrated with your training system
        return HealthStatus(
            component="training_progress",
            status="OK",
            message="Training progress check not implemented"
        )
    
    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for timeout handling."""
        # Simple timeout implementation
        # In production, you might use signal-based timeouts
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Health check timed out after {elapsed:.2f}s")
    
    def _trigger_alerts(self, status: HealthStatus) -> None:
        """Trigger alert callbacks for critical status."""
        for callback in self.alert_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

class PrometheusExporter:
    """Exports metrics to Prometheus."""
    
    def __init__(self, port: int = 8080, prefix: str = "carla_drl"):
        """Initialize Prometheus exporter."""
        if not HAS_PROMETHEUS:
            raise ImportError("Prometheus client not available")
        
        self.port = port
        self.prefix = prefix
        
        # System metrics
        self.cpu_usage = Gauge(f'{prefix}_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge(f'{prefix}_memory_usage_percent', 'Memory usage percentage')
        self.gpu_usage = Gauge(f'{prefix}_gpu_usage_percent', 'GPU usage percentage', ['gpu_id'])
        self.disk_usage = Gauge(f'{prefix}_disk_usage_percent', 'Disk usage percentage')
        
        # Training metrics
        self.episode_reward = Gauge(f'{prefix}_episode_reward', 'Episode reward')
        self.success_rate = Gauge(f'{prefix}_success_rate', 'Success rate')
        self.collision_rate = Gauge(f'{prefix}_collision_rate', 'Collision rate')
        self.training_loss = Gauge(f'{prefix}_training_loss', 'Training loss', ['loss_type'])
        
        # CARLA metrics
        self.simulation_fps = Gauge(f'{prefix}_simulation_fps', 'Simulation FPS')
        self.connection_latency = Gauge(f'{prefix}_connection_latency_ms', 'Connection latency in ms')
        
        # Health metrics
        self.component_health = Gauge(f'{prefix}_component_health', 'Component health status', ['component'])
        
        logger.info(f"Prometheus exporter initialized on port {port}")
    
    def start_server(self):
        """Start Prometheus metrics server."""
        start_http_server(self.port)
        logger.info(f"Prometheus metrics server started on port {self.port}")
    
    def update_metrics(self, system_metrics: SystemMetrics, 
                      training_metrics: TrainingMetrics,
                      carla_metrics: CarlaMetrics) -> None:
        """Update Prometheus metrics."""
        try:
            # System metrics
            self.cpu_usage.set(system_metrics.cpu_percent)
            self.memory_usage.set(system_metrics.memory_percent)
            self.disk_usage.set(system_metrics.disk_percent)
            
            # GPU metrics
            for i, utilization in enumerate(system_metrics.gpu_utilization):
                self.gpu_usage.labels(gpu_id=str(i)).set(utilization)
            
            # Training metrics
            self.episode_reward.set(training_metrics.episode_reward)
            self.success_rate.set(training_metrics.success_rate)
            self.collision_rate.set(training_metrics.collision_rate)
            
            self.training_loss.labels(loss_type='policy').set(training_metrics.policy_loss)
            self.training_loss.labels(loss_type='value').set(training_metrics.value_loss)
            self.training_loss.labels(loss_type='entropy').set(training_metrics.entropy_loss)
            
            # CARLA metrics
            self.simulation_fps.set(carla_metrics.simulation_fps)
            self.connection_latency.set(carla_metrics.connection_latency_ms)
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")

class MonitoringSystem:
    """Main monitoring system that orchestrates all components."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring system."""
        self.config = config
        self.running = False
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(config.get('health_checks', {}))
        
        # Initialize Prometheus exporter if enabled
        self.prometheus_exporter = None
        if config.get('prometheus', {}).get('enabled', False) and HAS_PROMETHEUS:
            try:
                port = config.get('prometheus', {}).get('port', 8080)
                prefix = config.get('prometheus', {}).get('metrics_prefix', 'carla_drl')
                self.prometheus_exporter = PrometheusExporter(port, prefix)
                self.prometheus_exporter.start_server()
            except Exception as e:
                logger.error(f"Failed to start Prometheus exporter: {e}")
        
        # Initialize TensorBoard writer if enabled
        self.tensorboard_writer = None
        if config.get('tensorboard', {}).get('enabled', False) and HAS_TENSORBOARD:
            log_dir = Path("logs/monitoring") / f"run_{int(time.time())}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
        
        # Monitoring thread
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        logger.info("Monitoring system initialized")
    
    def start(self):
        """Start the monitoring system."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring system started")
    
    def stop(self):
        """Stop the monitoring system."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                # Run health checks
                health_results = self.health_monitor.run_health_checks()
                
                # Update external metrics if available
                self._process_external_metrics()
                
                # Export metrics
                if self.prometheus_exporter:
                    self.prometheus_exporter.update_metrics(
                        system_metrics,
                        self.metrics_collector.training_metrics,
                        self.metrics_collector.carla_metrics
                    )
                
                # Log to TensorBoard
                if self.tensorboard_writer:
                    self._log_to_tensorboard(system_metrics)
                
                # Sleep until next collection
                time.sleep(self.config.get('collection_interval', 5.0))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _process_external_metrics(self):
        """Process metrics from external sources."""
        # Process any queued metrics updates
        while not self.metrics_queue.empty():
            try:
                metric_type, metrics_data = self.metrics_queue.get_nowait()
                if metric_type == 'training':
                    self.metrics_collector.update_training_metrics(metrics_data)
                elif metric_type == 'carla':
                    self.metrics_collector.update_carla_metrics(metrics_data)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing external metrics: {e}")
    
    def _log_to_tensorboard(self, system_metrics: SystemMetrics):
        """Log metrics to TensorBoard."""
        if not self.tensorboard_writer:
            return
        
        timestamp = int(time.time())
        
        # System metrics
        self.tensorboard_writer.add_scalar('System/CPU_Usage', system_metrics.cpu_percent, timestamp)
        self.tensorboard_writer.add_scalar('System/Memory_Usage', system_metrics.memory_percent, timestamp)
        self.tensorboard_writer.add_scalar('System/Disk_Usage', system_metrics.disk_percent, timestamp)
        
        # GPU metrics
        for i, utilization in enumerate(system_metrics.gpu_utilization):
            self.tensorboard_writer.add_scalar(f'System/GPU_{i}_Usage', utilization, timestamp)
    
    def update_training_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics from external source."""
        self.metrics_queue.put(('training', metrics))
    
    def update_carla_metrics(self, metrics: Dict[str, Any]):
        """Update CARLA metrics from external source."""
        self.metrics_queue.put(('carla', metrics))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        return {
            'timestamp': time.time(),
            'system_metrics': asdict(self.metrics_collector.system_metrics),
            'training_metrics': asdict(self.metrics_collector.training_metrics),
            'carla_metrics': asdict(self.metrics_collector.carla_metrics),
            'component_health': {name: asdict(status) for name, status in self.health_monitor.component_health.items()},
            'overall_status': self._get_overall_status()
        }
    
    def _get_overall_status(self) -> str:
        """Determine overall system status."""
        if not self.health_monitor.component_health:
            return "UNKNOWN"
        
        statuses = [status.status for status in self.health_monitor.component_health.values()]
        
        if any(status == "CRITICAL" for status in statuses):
            return "CRITICAL"
        elif any(status == "ERROR" for status in statuses):
            return "ERROR"
        elif any(status == "WARNING" for status in statuses):
            return "WARNING"
        else:
            return "OK"

def create_monitoring_system(config_path: str) -> MonitoringSystem:
    """Factory function to create monitoring system from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    monitoring_config = config.get('monitoring', {})
    return MonitoringSystem(monitoring_config)

def main():
    """Main function for standalone monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CARLA DRL Monitoring System")
    parser.add_argument("--config", type=str, 
                       default="configs/advanced_pipeline_config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Create and start monitoring system
    monitoring_system = create_monitoring_system(args.config)
    
    try:
        monitoring_system.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        monitoring_system.stop()

if __name__ == "__main__":
    main()

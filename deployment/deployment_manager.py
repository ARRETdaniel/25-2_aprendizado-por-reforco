#!/usr/bin/env python3
"""
Advanced deployment and automation scripts for CARLA DRL Pipeline
Production-ready deployment automation with comprehensive error handling
"""

import os
import sys
import subprocess
import time
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import requests
import psutil
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    name: str
    port: int
    health_endpoint: str
    startup_timeout: int = 60
    depends_on: List[str] = None
    required: bool = True

class DeploymentManager:
    """Advanced deployment manager for CARLA DRL Pipeline"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.compose_file = project_root / "docker-compose.yml"
        self.env_file = project_root / ".env"
        
        # Service definitions
        self.services = {
            "carla-server": ServiceConfig(
                name="carla-server",
                port=2000,
                health_endpoint="http://localhost:2000",
                startup_timeout=120
            ),
            "redis": ServiceConfig(
                name="redis",
                port=6379,
                health_endpoint="redis://localhost:6379",
                startup_timeout=30
            ),
            "postgres": ServiceConfig(
                name="postgres",
                port=5432,
                health_endpoint="postgresql://localhost:5432",
                startup_timeout=45
            ),
            "ros2-gateway": ServiceConfig(
                name="ros2-gateway",
                port=8080,
                health_endpoint="http://localhost:8080/health",
                startup_timeout=60,
                depends_on=["carla-server"]
            ),
            "monitoring": ServiceConfig(
                name="monitoring",
                port=9090,
                health_endpoint="http://localhost:9090/-/healthy",
                startup_timeout=30
            ),
            "drl-trainer": ServiceConfig(
                name="drl-trainer",
                port=8081,
                health_endpoint="http://localhost:8081/health",
                startup_timeout=90,
                depends_on=["carla-server", "ros2-gateway", "redis"]
            )
        }
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites for deployment"""
        logger.info("Checking prerequisites...")
        
        checks = {
            "Docker": self._check_docker,
            "Docker Compose": self._check_docker_compose,
            "NVIDIA Docker": self._check_nvidia_docker,
            "System Resources": self._check_system_resources,
            "Network Ports": self._check_network_ports
        }
        
        results = {}
        for check_name, check_func in checks.items():
            try:
                results[check_name] = check_func()
                logger.info(f"✓ {check_name}: {'PASS' if results[check_name] else 'FAIL'}")
            except Exception as e:
                results[check_name] = False
                logger.error(f"✗ {check_name}: {str(e)}")
        
        return all(results.values())
    
    def _check_docker(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return "Docker version" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_docker_compose(self) -> bool:
        """Check if Docker Compose is available"""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return "docker-compose version" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_nvidia_docker(self) -> bool:
        """Check if NVIDIA Docker runtime is available"""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8-base-ubuntu20.04", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("NVIDIA Docker runtime not available - GPU acceleration disabled")
            return True  # Non-blocking for CPU-only deployments
    
    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources"""
        # Check available memory (minimum 8GB)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            logger.error(f"Insufficient memory: {memory_gb:.1f}GB (minimum 8GB required)")
            return False
        
        # Check available disk space (minimum 50GB)
        disk_gb = psutil.disk_usage('/').free / (1024**3)
        if disk_gb < 50:
            logger.error(f"Insufficient disk space: {disk_gb:.1f}GB (minimum 50GB required)")
            return False
        
        return True
    
    def _check_network_ports(self) -> bool:
        """Check if required ports are available"""
        required_ports = [2000, 3000, 5432, 6006, 6379, 8080, 8081, 8888, 9090]
        
        for port in required_ports:
            if self._is_port_in_use(port):
                logger.warning(f"Port {port} is already in use")
                # Non-blocking warning
        
        return True
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a specific port is in use"""
        connections = psutil.net_connections()
        return any(conn.laddr.port == port for conn in connections if conn.laddr)
    
    def setup_environment(self) -> bool:
        """Setup environment variables and configuration"""
        logger.info("Setting up environment...")
        
        env_vars = {
            "COMPOSE_PROJECT_NAME": "carla-drl-prod",
            "NVIDIA_VISIBLE_DEVICES": "0",
            "DISPLAY": "host.docker.internal:0.0",
            "CARLA_VERSION": "0.8.4",
            "ROS_DISTRO": "humble",
            "PYTHONPATH": "/opt/workspace:/opt/carla",
            "CUDA_VISIBLE_DEVICES": "0"
        }
        
        try:
            with open(self.env_file, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            logger.info(f"Environment file created: {self.env_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create environment file: {str(e)}")
            return False
    
    def build_images(self) -> bool:
        """Build all Docker images"""
        logger.info("Building Docker images...")
        
        try:
            cmd = ["docker-compose", "-f", str(self.compose_file), "build", "--no-cache"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Docker images built successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build images: {e.stderr}")
            return False
    
    def deploy_services(self, services: Optional[List[str]] = None) -> bool:
        """Deploy services in correct order"""
        logger.info("Deploying services...")
        
        if services is None:
            # Deploy in dependency order
            deploy_order = [
                ["redis", "postgres"],  # Infrastructure
                ["carla-server"],       # CARLA simulator
                ["monitoring"],         # Monitoring stack
                ["ros2-gateway"],       # ROS 2 gateway
                ["drl-trainer"]         # DRL training
            ]
        else:
            deploy_order = [services]
        
        for service_group in deploy_order:
            if not self._deploy_service_group(service_group):
                return False
            
            # Wait for services to be healthy
            if not self._wait_for_services_healthy(service_group):
                return False
        
        logger.info("All services deployed successfully")
        return True
    
    def _deploy_service_group(self, services: List[str]) -> bool:
        """Deploy a group of services"""
        logger.info(f"Deploying services: {', '.join(services)}")
        
        try:
            cmd = ["docker-compose", "-f", str(self.compose_file), "up", "-d"] + services
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy services {services}: {e.stderr}")
            return False
    
    def _wait_for_services_healthy(self, services: List[str], timeout: int = 300) -> bool:
        """Wait for services to become healthy"""
        logger.info(f"Waiting for services to become healthy: {', '.join(services)}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for service_name in services:
                if service_name not in self.services:
                    continue
                
                service = self.services[service_name]
                if not self._check_service_health(service):
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info(f"All services healthy: {', '.join(services)}")
                return True
            
            time.sleep(10)
        
        logger.error(f"Timeout waiting for services to become healthy: {', '.join(services)}")
        return False
    
    def _check_service_health(self, service: ServiceConfig) -> bool:
        """Check if a service is healthy"""
        try:
            if service.health_endpoint.startswith("http"):
                response = requests.get(service.health_endpoint, timeout=5)
                return response.status_code == 200
            elif service.health_endpoint.startswith("redis"):
                # Simple Redis ping check
                cmd = ["docker-compose", "exec", "-T", service.name, "redis-cli", "ping"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                return "PONG" in result.stdout
            elif service.health_endpoint.startswith("postgresql"):
                # Simple PostgreSQL check
                cmd = ["docker-compose", "exec", "-T", service.name, "pg_isready"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            else:
                # Check if container is running
                cmd = ["docker-compose", "ps", "-q", service.name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return bool(result.stdout.strip())
        except Exception:
            return False
    
    def verify_deployment(self) -> Dict[str, bool]:
        """Verify all services are working correctly"""
        logger.info("Verifying deployment...")
        
        verification_results = {}
        
        for service_name, service in self.services.items():
            try:
                # Check if container is running
                is_running = self._is_container_running(service_name)
                
                # Check service health
                is_healthy = self._check_service_health(service) if is_running else False
                
                verification_results[service_name] = {
                    "running": is_running,
                    "healthy": is_healthy,
                    "status": "OK" if is_running and is_healthy else "FAILED"
                }
                
                logger.info(f"{service_name}: {verification_results[service_name]['status']}")
            except Exception as e:
                verification_results[service_name] = {
                    "running": False,
                    "healthy": False,
                    "status": f"ERROR: {str(e)}"
                }
                logger.error(f"{service_name}: {verification_results[service_name]['status']}")
        
        return verification_results
    
    def _is_container_running(self, service_name: str) -> bool:
        """Check if a container is running"""
        try:
            cmd = ["docker-compose", "ps", "-q", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                # Check container status
                container_id = result.stdout.strip()
                cmd = ["docker", "inspect", "--format", "{{.State.Running}}", container_id]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.stdout.strip() == "true"
            
            return False
        except Exception:
            return False
    
    def cleanup(self, remove_volumes: bool = False) -> bool:
        """Clean up deployment"""
        logger.info("Cleaning up deployment...")
        
        try:
            cmd = ["docker-compose", "-f", str(self.compose_file), "down"]
            if remove_volumes:
                cmd.extend(["--volumes", "--remove-orphans"])
            
            subprocess.run(cmd, check=True)
            logger.info("Cleanup completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return False
    
    def export_logs(self, output_dir: Path) -> bool:
        """Export logs from all services"""
        logger.info(f"Exporting logs to {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for service_name in self.services.keys():
                log_file = output_dir / f"{service_name}.log"
                cmd = ["docker-compose", "logs", "--no-color", service_name]
                
                with open(log_file, 'w') as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
                
                logger.info(f"Exported logs for {service_name}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to export logs: {str(e)}")
            return False
    
    def backup_data(self, backup_dir: Path) -> bool:
        """Backup persistent data"""
        logger.info(f"Backing up data to {backup_dir}")
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        volumes_to_backup = [
            "carla-drl-pipeline_drl_checkpoints",
            "carla-drl-pipeline_prometheus_data",
            "carla-drl-pipeline_grafana_data",
            "carla-drl-pipeline_postgres_data"
        ]
        
        try:
            for volume in volumes_to_backup:
                backup_file = backup_dir / f"{volume.split('_')[-1]}.tar.gz"
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{volume}:/source:ro",
                    "-v", f"{backup_dir}:/backup",
                    "alpine",
                    "tar", "czf", f"/backup/{backup_file.name}", "-C", "/source", "."
                ]
                subprocess.run(cmd, check=True)
                logger.info(f"Backed up volume: {volume}")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Backup failed: {str(e)}")
            return False

def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="CARLA DRL Pipeline Deployment Manager")
    parser.add_argument("action", choices=[
        "deploy", "verify", "cleanup", "backup", "logs", "check"
    ], help="Action to perform")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--services", nargs="+", help="Specific services to deploy")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"),
                       help="Output directory for logs and backups")
    parser.add_argument("--remove-volumes", action="store_true",
                       help="Remove volumes during cleanup")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(args.project_root)
    
    try:
        if args.action == "check":
            success = deployment_manager.check_prerequisites()
            sys.exit(0 if success else 1)
        
        elif args.action == "deploy":
            # Full deployment pipeline
            if not deployment_manager.check_prerequisites():
                logger.error("Prerequisites check failed")
                sys.exit(1)
            
            if not deployment_manager.setup_environment():
                logger.error("Environment setup failed")
                sys.exit(1)
            
            if not deployment_manager.build_images():
                logger.error("Image build failed")
                sys.exit(1)
            
            if not deployment_manager.deploy_services(args.services):
                logger.error("Service deployment failed")
                sys.exit(1)
            
            logger.info("Deployment completed successfully!")
        
        elif args.action == "verify":
            results = deployment_manager.verify_deployment()
            failed_services = [
                name for name, status in results.items() 
                if status.get("status") != "OK"
            ]
            
            if failed_services:
                logger.error(f"Verification failed for services: {', '.join(failed_services)}")
                sys.exit(1)
            else:
                logger.info("All services verified successfully!")
        
        elif args.action == "cleanup":
            success = deployment_manager.cleanup(args.remove_volumes)
            sys.exit(0 if success else 1)
        
        elif args.action == "backup":
            success = deployment_manager.backup_data(args.output_dir)
            sys.exit(0 if success else 1)
        
        elif args.action == "logs":
            success = deployment_manager.export_logs(args.output_dir)
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

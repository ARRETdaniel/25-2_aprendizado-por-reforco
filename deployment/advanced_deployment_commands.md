# Advanced Deployment Commands and Automation Scripts
# Production-ready deployment for Windows 11 with WSL2/Docker support

## Prerequisites Setup

### 1. Windows PowerShell Setup (Run as Administrator)
```powershell
# Enable WSL 2 and install Ubuntu
wsl --install -d Ubuntu-22.04

# Install Docker Desktop with WSL 2 backend
winget install Docker.DockerDesktop

# Install NVIDIA Container Toolkit for GPU support
# Download from: https://github.com/NVIDIA/nvidia-docker
```

### 2. WSL 2 Environment Setup
```bash
# Update WSL Ubuntu
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y curl wget git build-essential

# Install Docker CLI in WSL (if needed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA drivers in WSL (for GPU passthrough)
# Follow NVIDIA WSL 2 CUDA guide
```

## Deployment Commands

### Quick Start (Windows PowerShell)
```powershell
# Clone repository and navigate
git clone https://github.com/your-repo/carla-drl-pipeline.git
cd carla-drl-pipeline

# Build and start all services
docker-compose up -d --build

# Monitor startup
docker-compose logs -f

# Check service health
docker-compose ps
```

### Production Deployment (Full Stack)
```powershell
# Set environment variables
$env:COMPOSE_PROJECT_NAME = "carla-drl-prod"
$env:NVIDIA_VISIBLE_DEVICES = "0"
$env:DISPLAY = "host.docker.internal:0.0"

# Build all images
docker-compose build --no-cache

# Start core services first
docker-compose up -d carla-server redis postgres

# Wait for core services to be healthy
Start-Sleep -Seconds 30

# Start ROS 2 gateway
docker-compose up -d ros2-gateway

# Start monitoring stack
docker-compose up -d monitoring

# Finally start DRL training
docker-compose up -d drl-trainer

# Verify all services are running
docker-compose ps --services --filter "status=running"
```

### Development Setup (with Jupyter)
```powershell
# Start with development profile
docker-compose --profile development up -d

# Access Jupyter Notebook
Start-Process "http://localhost:8888/?token=carla-drl-token"

# Access monitoring dashboards
Start-Process "http://localhost:3000"      # Grafana
Start-Process "http://localhost:9090"      # Prometheus
Start-Process "http://localhost:6006"      # TensorBoard
```

### WSL 2 Deployment (Linux commands)
```bash
#!/bin/bash
# Run from WSL 2 Ubuntu terminal

# Navigate to project directory
cd /mnt/c/Users/$(whoami)/Documents/carla-drl-pipeline

# Set environment variables
export COMPOSE_PROJECT_NAME="carla-drl-prod"
export NVIDIA_VISIBLE_DEVICES="0"
export DISPLAY=:0

# Build and deploy
docker-compose up -d --build

# Monitor logs
docker-compose logs -f --tail=50

# Health check script
./scripts/health_check.sh
```

## Service Management Commands

### Individual Service Control
```powershell
# Start specific service
docker-compose up -d carla-server

# Stop specific service
docker-compose stop drl-trainer

# Restart service
docker-compose restart ros2-gateway

# View service logs
docker-compose logs -f carla-server

# Execute command in service
docker-compose exec drl-trainer python /opt/tools/diagnostics.py
```

### Scaling and Updates
```powershell
# Scale training workers (if configured)
docker-compose up -d --scale drl-trainer=3

# Update specific service
docker-compose build --no-cache drl-trainer
docker-compose up -d drl-trainer

# Rolling update with zero downtime
docker-compose up -d --force-recreate --no-deps drl-trainer
```

### Data Management
```powershell
# Backup trained models
docker run --rm -v carla-drl-pipeline_drl_checkpoints:/source -v ${PWD}/backup:/backup alpine tar czf /backup/models_$(Get-Date -Format "yyyy-MM-dd_HH-mm-ss").tar.gz -C /source .

# Restore from backup
docker run --rm -v carla-drl-pipeline_drl_checkpoints:/target -v ${PWD}/backup:/backup alpine tar xzf /backup/models_latest.tar.gz -C /target

# Export training logs
docker-compose exec drl-trainer tar czf /tmp/logs.tar.gz /opt/logs
docker cp $(docker-compose ps -q drl-trainer):/tmp/logs.tar.gz ./logs_export.tar.gz
```

## Monitoring and Debugging

### Performance Monitoring
```powershell
# Real-time resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# GPU utilization (requires nvidia-smi in container)
docker-compose exec drl-trainer nvidia-smi -l 5

# Custom monitoring dashboard
Start-Process "http://localhost:3000/d/carla-drl-overview"
```

### Debugging Commands
```powershell
# Enter container for debugging
docker-compose exec carla-server bash
docker-compose exec drl-trainer /bin/bash

# Check container health
docker-compose exec carla-server curl -f http://localhost:2000 || echo "CARLA not ready"

# Inspect network connectivity
docker network inspect carla-drl-pipeline_carla-network

# View container configurations
docker-compose config
```

### Log Analysis
```powershell
# Aggregate logs from all services
docker-compose logs --since=1h > ./logs/pipeline_logs.txt

# Filter logs by service and level
docker-compose logs carla-server | Select-String "ERROR|WARN"

# Real-time log monitoring with filtering
docker-compose logs -f | Select-String "collision|success|reward"

# Export structured logs for analysis
docker-compose exec monitoring curl -s "http://localhost:9090/api/v1/query?query=carla_drl_episode_reward" | ConvertFrom-Json
```

## Troubleshooting

### Common Issues and Solutions
```powershell
# Issue: CARLA server not starting
# Solution: Check GPU drivers and X11 forwarding
docker-compose logs carla-server
# Ensure NVIDIA Docker runtime is installed

# Issue: ROS 2 nodes not communicating
# Solution: Check DDS configuration
docker-compose exec ros2-gateway ros2 node list
docker-compose exec ros2-gateway ros2 topic list

# Issue: Training not progressing
# Solution: Check GPU availability and memory
docker-compose exec drl-trainer python -c "import torch; print(torch.cuda.is_available())"

# Issue: High memory usage
# Solution: Check for memory leaks and adjust limits
docker-compose exec drl-trainer python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### Performance Optimization
```powershell
# Optimize Docker for performance
# Add to Docker Desktop settings or daemon.json:
# {
#   "experimental": true,
#   "storage-driver": "overlay2",
#   "log-driver": "json-file",
#   "log-opts": {
#     "max-size": "100m",
#     "max-file": "3"
#   }
# }

# Increase shared memory for large datasets
docker-compose run --shm-size=2g drl-trainer python train.py

# Use tmpfs for high-speed temporary storage
# Add to docker-compose.yml:
# tmpfs:
#   - /tmp:size=1G,uid=1000
```

## Cleanup and Maintenance

### Regular Cleanup
```powershell
# Stop all services
docker-compose down

# Remove unused containers and images
docker system prune -f

# Remove specific project containers
docker-compose down --volumes --remove-orphans

# Clean up unused volumes
docker volume ls | Select-String "carla-drl" | ForEach-Object { docker volume rm $_.ToString().Split()[1] }
```

### Backup and Restore
```powershell
# Complete system backup
$backupDir = "./backup/$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
New-Item -ItemType Directory -Path $backupDir

# Backup volumes
docker run --rm -v carla-drl-pipeline_drl_checkpoints:/source -v ${PWD}/${backupDir}:/backup alpine tar czf /backup/checkpoints.tar.gz -C /source .
docker run --rm -v carla-drl-pipeline_prometheus_data:/source -v ${PWD}/${backupDir}:/backup alpine tar czf /backup/prometheus.tar.gz -C /source .

# Backup configurations
Copy-Item -Recurse ./configs ${backupDir}/
Copy-Item docker-compose.yml ${backupDir}/

# Create restore script
@"
# Restore script
docker-compose down --volumes
docker volume create carla-drl-pipeline_drl_checkpoints
docker volume create carla-drl-pipeline_prometheus_data
docker run --rm -v carla-drl-pipeline_drl_checkpoints:/target -v ${PWD}:/backup alpine tar xzf /backup/checkpoints.tar.gz -C /target
docker run --rm -v carla-drl-pipeline_prometheus_data:/target -v ${PWD}:/backup alpine tar xzf /backup/prometheus.tar.gz -C /target
"@ | Out-File -FilePath "${backupDir}/restore.ps1"
```

## VS Code Integration

### Development Container Configuration
```json
// .devcontainer/devcontainer.json
{
    "name": "CARLA DRL Development",
    "dockerComposeFile": "../docker-compose.yml",
    "service": "drl-trainer",
    "workspaceFolder": "/opt/workspace",
    "shutdownAction": "none",
    "extensions": [
        "ms-python.python",
        "ms-python.debugpy",
        "ms-toolsai.jupyter"
    ],
    "settings": {
        "python.defaultInterpreterPath": "/opt/conda/bin/python"
    },
    "forwardPorts": [6006, 8888, 3000, 9090],
    "postCreateCommand": "pip install -r requirements.txt"
}
```

### VS Code Tasks
```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Deploy Full Stack",
            "type": "shell",
            "command": "docker-compose",
            "args": ["up", "-d", "--build"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Stop All Services",
            "type": "shell",
            "command": "docker-compose",
            "args": ["down"],
            "group": "build"
        },
        {
            "label": "View Logs",
            "type": "shell",
            "command": "docker-compose",
            "args": ["logs", "-f", "--tail=100"],
            "group": "test"
        },
        {
            "label": "Health Check",
            "type": "shell",
            "command": "docker-compose",
            "args": ["exec", "drl-trainer", "python", "/opt/tools/diagnostics.py"],
            "group": "test"
        }
    ]
}
```

## Production Checklist

### Pre-deployment Verification
- [ ] NVIDIA drivers installed and working
- [ ] Docker Desktop with WSL 2 backend configured
- [ ] NVIDIA Container Toolkit installed
- [ ] Sufficient disk space (50GB+)
- [ ] GPU memory available (6GB+)
- [ ] Network ports available (2000, 3000, 6006, 8080, 9090)

### Post-deployment Verification
- [ ] All services healthy: `docker-compose ps`
- [ ] CARLA server responding: `curl localhost:2000`
- [ ] ROS 2 nodes running: `docker-compose exec ros2-gateway ros2 node list`
- [ ] Training metrics available: `curl localhost:8080/metrics`
- [ ] Dashboards accessible: Grafana (3000), TensorBoard (6006)
- [ ] GPU utilization visible in monitoring
- [ ] Log files being generated
- [ ] Model checkpoints being saved

This comprehensive deployment guide provides everything needed to deploy the CARLA DRL pipeline in a production environment with monitoring, scalability, and maintainability.

# Docker Setup Guide: TD3-Based Autonomous Vehicle System

## Overview

This guide provides step-by-step instructions for setting up the Docker-based TD3 Autonomous Vehicle system for the CARLA simulator and ROS 2 ecosystem.

**Project**: End-to-End Visual Autonomous Navigation with Twin Delayed DDPG
**Reference**: [detalhamento_RL_25_2_IEEE.md](../../../contextual/detalhamento_RL_25_2_IEEE.md)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Quick Start](#quick-start)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)
7. [Documentation References](#documentation-references)

---

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04 LTS
- **RAM**: 16GB minimum (32GB+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 2060)
- **Disk Space**: 50GB+ (20GB for CARLA image + 30GB+ for training data)

### Software Requirements

The installation process will set up:
- **Docker Engine**: 20.10+
- **NVIDIA Container Toolkit**: 1.17.9+
- **CARLA**: 0.9.16
- **ROS 2**: Foxy

---

## Installation Steps

### Step 1: Install Docker Engine

```bash
# Remove old Docker versions (if any)
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker using official script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (avoid sudo requirement)
sudo usermod -aG docker $USER

# Activate new group membership
newgrp docker

# Verify installation
docker run hello-world
```

**Expected Output**:
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

### Step 2: Install NVIDIA Container Toolkit

```bash
# Set up distribution identifier
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Add NVIDIA GPG key
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

# Add NVIDIA Docker repository
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon to register nvidia runtime
sudo systemctl restart docker

# Verify GPU access in containers
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

**Expected Output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07                |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 2060    Off  | 00:1F.0     Off |                  N/A |
|  0%   40C    P8     9W /  80W |     50MiB /  6144MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

### Step 3: Pull CARLA 0.9.16 Docker Image

```bash
# Pull the CARLA 0.9.16 image (20.7GB)
docker pull carlasim/carla:0.9.16

# Verify image was pulled
docker images | grep carla
```

**Expected Output**:
```
carlasim/carla       0.9.16    a1b2c3d4e5f6    2 weeks ago   20.7GB
```

### Step 4: Build the TD3 AV System Image

```bash
# Navigate to project directory
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Build the image (this takes 5-15 minutes depending on internet speed)
docker build -t td3-av-system:v1.0 .

# Tag as latest for convenience
docker tag td3-av-system:v1.0 td3-av-system:latest

# Verify images
docker images | grep td3-av-system
```

**Expected Output**:
```
td3-av-system       latest       7d309da4e5a5   3 minutes ago   30.6GB
td3-av-system       v1.0         7d309da4e5a5   3 minutes ago   30.6GB
```

---

## Verification

### Test 1: GPU Access

Verify that the container can access GPU:

```bash
docker run --rm --runtime=nvidia --gpus all --entrypoint python3 td3-av-system:v1.0 \
  -c "
import torch
print('PyTorch Version:', torch.__version__)
print('GPU Available:', torch.cuda.is_available())
print('GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"
```

**Expected Output**:
```
PyTorch Version: 2.4.1+cu121
GPU Available: True
GPU Device: NVIDIA GeForce RTX 2060
```

### Test 2: Python Imports

Verify that all critical packages are available:

```bash
docker run --rm --runtime=nvidia --gpus all --entrypoint python3 td3-av-system:v1.0 \
  -c "
import torch
import cv2
import numpy
import scipy
import matplotlib
import seaborn
import pandas
import wandb
print('✅ PyTorch:', torch.__version__)
print('✅ OpenCV:', cv2.__version__)
print('✅ NumPy:', numpy.__version__)
print('✅ SciPy:', scipy.__version__)
print('✅ Matplotlib:', matplotlib.__version__)
print('✅ Seaborn:', seaborn.__version__)
print('✅ Pandas:', pandas.__version__)
print('✅ Weights & Biases:', wandb.__version__)
print('✅ All imports successful!')
"
```

**Expected Output**:
```
✅ PyTorch: 2.4.1+cu121
✅ OpenCV: 4.8.1.78
✅ NumPy: 1.24.3
✅ SciPy: 1.10.1
✅ Matplotlib: 3.7.3
✅ Seaborn: 0.12.2
✅ Pandas: 2.0.3
✅ Weights & Biases: 0.22.2
✅ All imports successful!
```

### Test 3: Docker Compose Configuration

Verify that docker-compose configuration is valid:

```bash
docker compose config > /dev/null && echo "✅ docker-compose.yml is valid" || echo "❌ Configuration error"
```

---

## Quick Start

### Method 1: Using docker-compose (Recommended)

```bash
# Navigate to project directory
cd av_td3_system

# Start TD3 training container
docker compose up td3-training

# In another terminal, monitor training
docker logs -f td3_training
```

### Method 2: Using docker run directly

```bash
# Simple container startup
docker run --rm --runtime=nvidia --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/config:/workspace/av_td3_system/config \
  --net=host \
  -e AGENT_TYPE=TD3 \
  -e EPISODES=2000 \
  td3-av-system:latest \
  python3 /workspace/av_td3_system/scripts/train_td3.py
```

### Method 3: Using helper scripts

```bash
# Make scripts executable
chmod +x scripts/docker_build.sh
chmod +x scripts/docker_run_train.sh

# Run training
./scripts/docker_run_train.sh TD3 2000 50
```

---

## Advanced Usage

### Multi-GPU Training (TD3 + DDPG)

Train TD3 and DDPG in parallel on different GPUs:

```bash
# Start both agents (requires 2 NVIDIA GPUs)
docker compose --profile multi-gpu up td3-training ddpg-training
```

### Evaluation Mode

Run evaluation on trained checkpoints:

```bash
docker compose --profile evaluation up evaluation
```

### Monitoring with TensorBoard

View training progress in real-time:

```bash
# Start TensorBoard container
docker compose --profile monitoring up tensorboard

# Access at http://localhost:6006
```

### Custom Configuration

Modify training parameters by editing configuration files:

```bash
# Edit TD3 configuration
nano config/td3_config.yaml

# Edit CARLA simulation parameters
nano config/carla_config.yaml

# Rebuild image to apply changes
docker compose build td3-training
docker compose up td3-training
```

---

## Troubleshooting

### Issue 1: "docker: command not found"

**Solution**: Docker not installed or not in PATH
```bash
# Verify installation
which docker

# If not found, reinstall Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Issue 2: "permission denied while trying to connect to Docker daemon"

**Solution**: User not in docker group
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Activate changes
newgrp docker

# Verify
docker ps
```

### Issue 3: "NVIDIA Driver not found" or "GPU not detected"

**Solution**: NVIDIA runtime not configured
```bash
# Verify NVIDIA Container Toolkit
nvidia-ctk --version

# Reconfigure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Issue 4: "Error response from daemon: Unknown runtime specified nvidia"

**Solution**: Docker service not restarted after toolkit installation
```bash
sudo systemctl restart docker
docker run --rm --gpus all --entrypoint python3 td3-av-system:v1.0 -c "import torch; print(torch.cuda.is_available())"
```

### Issue 5: Build fails with "Cannot uninstall PyYAML"

**Solution**: System package conflicts (already resolved in current build)
- PyYAML removed from requirements.txt (already in base image)
- TensorBoard removed (use wandb instead)

### Issue 6: Out of disk space during build

**Solution**: Docker images consuming too much space
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Check disk space
du -sh /var/lib/docker/
```

### Issue 7: CARLA server not starting inside container

**Current Status**: Handled by `docker-entrypoint.sh`
- Script waits 15 seconds for CARLA initialization
- Tests Python connection before proceeding
- See logs with: `docker logs -f <container_name>`

---

## Monitoring & Debugging

### View Container Logs

```bash
# Real-time logs
docker logs -f td3_training

# Last 100 lines
docker logs --tail 100 td3_training

# With timestamps
docker logs -t td3_training
```

### Execute Commands in Running Container

```bash
# Interactive shell
docker exec -it td3_training bash

# Single command
docker exec td3_training python3 -c "import torch; print(torch.cuda.is_available())"
```

### Check Container Resource Usage

```bash
# CPU and memory usage
docker stats td3_training

# GPU usage (inside container)
docker exec td3_training nvidia-smi
```

---

## Important Paths & Volumes

| Path | Purpose | Host Mount |
|------|---------|-----------|
| `/workspace/data` | Training data, checkpoints | `./data` |
| `/workspace/results` | Output results, logs | `./results` |
| `/workspace/av_td3_system/config` | Configuration files | `./config` |
| `/home/carla/carla` | CARLA simulator (read-only) | (base image) |
| `/opt/ros/foxy` | ROS 2 Foxy (read-only) | (base image) |

---

## Performance Optimization

### Memory Usage

- **Recommended**: Run one training container per GPU
- **RTX 2060**: Typically uses 4-6GB VRAM during training
- Monitor with: `docker stats`

### Training Time

- **Initial setup**: 5-15 minutes
- **Warm-up period**: ~10 minutes (first CARLA connection)
- **Training per episode**: ~2-5 minutes
- **Total for 2000 episodes**: 60-150 hours

### Data Storage

- **CARLA checkpoints**: ~100MB per model
- **Training logs**: ~500MB per run
- **Results**: ~1GB per evaluation

---

## Documentation References

### Official Documentation
- **CARLA 0.9.16**: https://carla.readthedocs.io/en/latest/
- **CARLA Docker Guide**: https://carla.readthedocs.io/en/latest/build_docker/
- **CARLA 0.9.16 Release Notes**: https://carla.org/2025/09/16/release-0.9.16/
- **ROS 2 Foxy**: https://docs.ros.org/en/foxy/

### Deep Learning References
- **TD3 Algorithm**: https://spinningup.openai.com/en/latest/algorithms/td3.html
- **Stable Baselines3 TD3**: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
- **PyTorch**: https://pytorch.org/docs/

### Project Documentation
- **IEEE Paper Detailing**: [detalhamento_RL_25_2_IEEE.md](../../../contextual/detalhamento_RL_25_2_IEEE.md)
- **Development Plan**: [PRE_DEVELOPMENT_PLAN_DOCKER.md](../PRE_DEVELOPMENT_PLAN_DOCKER.md)
- **Docker Build Fixes**: [DOCKER_BUILD_FIXES.md](../DOCKER_BUILD_FIXES.md)

---

## Support & Issues

For issues with:
- **CARLA**: See CARLA documentation at https://github.com/carla-simulator/carla/discussions
- **ROS 2**: See ROS 2 documentation at https://docs.ros.org/
- **Docker**: See Docker documentation at https://docs.docker.com/
- **Project**: Check `DOCKER_BUILD_FIXES.md` for recent fixes and solutions

---

**Last Updated**: October 20, 2025
**Docker Image Version**: td3-av-system:v1.0 (30.6GB)
**Status**: ✅ Fully Functional with GPU Support Verified

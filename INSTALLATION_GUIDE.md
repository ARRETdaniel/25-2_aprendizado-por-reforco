# Complete Installation & First-Run Guide

## System Architecture Overview

```
carla_drl_complete/
├── 🚗 carla_client_py36/           # CARLA Client (Python 3.6)
│   ├── main_enhanced.py            # Enhanced module_7.py with ROS bridge
│   ├── ros_communication.py        # ZeroMQ bridge to ROS 2 gateway
│   ├── sensor_manager.py           # Sensor data processing
│   ├── yolo_integration.py         # YOLO detection integration
│   ├── performance_tracker.py      # Performance monitoring
│   └── requirements_py36.txt       # Python 3.6 dependencies
│
├── 🌉 ros2_gateway/                # High-Performance C++ Bridge
│   ├── src/
│   │   ├── carla_bridge_node.cpp   # Main ROS 2 node
│   │   ├── zmq_bridge.cpp          # ZeroMQ communication
│   │   └── message_converter.cpp   # Message serialization
│   ├── include/
│   │   ├── carla_bridge_node.hpp   # Node header
│   │   └── zmq_bridge.hpp          # ZMQ header
│   ├── package.xml                 # ROS 2 package manifest
│   ├── CMakeLists.txt              # Build configuration
│   └── launch/
│       └── carla_bridge.launch.py  # Launch configuration
│
├── 🧠 drl_agent/                   # DRL Training (Python 3.12)
│   ├── train_ppo.py                # PPO training script
│   ├── ppo_agent.py                # PPO implementation
│   ├── network_architectures.py    # Neural network models
│   ├── environment_wrapper.py      # Gym environment wrapper
│   ├── reward_functions.py         # Reward computation
│   ├── evaluation.py               # Model evaluation
│   ├── visualization.py            # Training visualization
│   └── requirements_py312.txt      # Python 3.12 dependencies
│
├── ⚙️ configs/                     # Configuration Files
│   ├── carla_sim.yaml              # CARLA simulation config
│   ├── ppo_training.yaml           # PPO hyperparameters
│   ├── ros2_bridge.yaml            # ROS 2 bridge config
│   ├── system_settings.yaml        # System configuration
│   └── validation_models.py        # Pydantic config models
│
├── 🔧 scripts/                     # Automation Scripts
│   ├── install_windows.ps1         # Windows installation
│   ├── install_wsl2.sh             # WSL2 setup
│   ├── setup_environments.py       # Python env setup
│   ├── start_system.py             # System launcher
│   ├── stop_system.py              # Graceful shutdown
│   └── health_check.py             # System diagnostics
│
├── 🧪 tests/                       # Testing Framework
│   ├── test_integration.py         # End-to-end tests
│   ├── test_carla_client.py        # CARLA client tests
│   ├── test_ros2_bridge.py         # ROS 2 bridge tests
│   ├── test_drl_agent.py           # DRL agent tests
│   ├── test_performance.py         # Performance benchmarks
│   └── test_data/                  # Test datasets
│
├── 📊 monitoring/                  # System Monitoring
│   ├── tensorboard_logs/           # TensorBoard logs
│   ├── performance_metrics/        # System metrics
│   ├── checkpoints/                # Model checkpoints
│   └── evaluation_results/         # Evaluation data
│
├── 🐳 docker/                      # Docker Configuration
│   ├── Dockerfile.carla_client     # CARLA client container
│   ├── Dockerfile.ros2_gateway     # ROS 2 gateway container
│   ├── Dockerfile.drl_agent        # DRL agent container
│   ├── docker-compose.yml          # Complete system
│   └── .dockerignore               # Docker ignore file
│
├── 📚 docs/                        # Documentation
│   ├── API_REFERENCE.md            # API documentation
│   ├── TROUBLESHOOTING.md          # Common issues
│   ├── PERFORMANCE_TUNING.md       # Optimization guide
│   └── RESEARCH_NOTES.md           # Research findings
│
└── 🔧 Development Tools
    ├── .vscode/                    # VS Code configuration
    │   ├── launch.json             # Debug configurations
    │   ├── tasks.json              # Build tasks
    │   └── settings.json           # Workspace settings
    ├── .gitignore                  # Git ignore rules
    ├── environment.yml             # Conda environment
    ├── requirements.txt            # Global requirements
    ├── setup.py                    # Package setup
    └── README.md                   # Project overview
```

## Prerequisites

### Hardware Requirements
- **OS**: Windows 11 x64 (with WSL2 enabled)
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores recommended)
- **GPU**: NVIDIA RTX 2060 or better (4GB+ VRAM)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB+ free space (SSD recommended)

### Software Requirements
- **Visual Studio Code** (latest)
- **Python 3.6** (for CARLA)
- **Python 3.12** (for DRL/ROS2)
- **Docker Desktop** (with WSL2 backend)
- **Git** (latest)
- **CARLA 0.8.4** (Coursera binary)

## Installation Steps

### 1. System Preparation

#### Enable WSL2 (PowerShell as Administrator)
```powershell
# Enable WSL2
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart required
shutdown /r /t 5

# After restart, set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu 22.04
wsl --install -d Ubuntu-22.04
```

#### Install Docker Desktop
```powershell
# Download and install Docker Desktop with WSL2 backend
# https://docs.docker.com/docker-for-windows/install/
# Ensure WSL2 integration is enabled
```

### 2. Repository Setup

```bash
# Clone the repository
git clone <repository-url>
cd carla_drl_complete

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py
```

### 3. Python Environment Setup

#### Python 3.6 (CARLA Client)
```powershell
# Windows PowerShell
# Download and install Python 3.6.8 from python.org
# Add to PATH during installation

# Create virtual environment
python -m venv carla_py36_env
carla_py36_env\Scripts\activate

# Install CARLA client dependencies
cd carla_client_py36
pip install -r requirements_py36.txt
cd ..
```

#### Python 3.12 (DRL Agent) - WSL2
```bash
# In WSL2 Ubuntu
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create DRL environment
conda create -n drl_py312 python=3.12 -y
conda activate drl_py312

# Install DRL dependencies
cd drl_agent
pip install -r requirements_py312.txt
cd ..
```

### 4. ROS 2 Installation (WSL2)

```bash
# In WSL2 Ubuntu
# Add ROS 2 Humble repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop -y
sudo apt install python3-colcon-common-extensions -y

# Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install additional dependencies
sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs -y
sudo apt install libzmq3-dev -y
```

### 5. Build ROS 2 Gateway

```bash
# In WSL2, activate ROS 2
source /opt/ros/humble/setup.bash

# Build the gateway
cd ros2_gateway
colcon build --symlink-install
source install/setup.bash
cd ..
```

### 6. CARLA Setup

```powershell
# In Windows PowerShell
# Ensure CARLA 0.8.4 is installed at:
# C:\CARLA_0.8.4\CarlaUE4\Binaries\Win64\CarlaUE4.exe

# Test CARLA server
cd C:\CARLA_0.8.4
CarlaUE4.exe /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=30
```

### 7. Configuration

#### Update Configuration Files
```yaml
# configs/carla_sim.yaml
carla:
  host: "127.0.0.1"  # localhost for Windows-WSL2 communication
  port: 2000
  timeout: 10.0

# configs/ros2_bridge.yaml  
bridge:
  zmq_ports:
    camera_rgb: 5555
    vehicle_state: 5556
    control_cmd: 5557
    reward: 5558
```

### 8. First System Test

#### Terminal 1: CARLA Server (Windows PowerShell)
```powershell
cd C:\CARLA_0.8.4
CarlaUE4.exe /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=30 -quality-level=Low
```

#### Terminal 2: ROS 2 Gateway (WSL2)
```bash
# Activate ROS 2
source /opt/ros/humble/setup.bash
cd carla_drl_complete/ros2_gateway
source install/setup.bash

# Launch bridge
ros2 launch carla_bridge carla_bridge.launch.py
```

#### Terminal 3: CARLA Client (Windows PowerShell)
```powershell
# Activate Python 3.6 environment
carla_py36_env\Scripts\activate
cd carla_drl_complete\carla_client_py36

# Launch enhanced client
python main_enhanced.py --config ../configs/carla_sim.yaml
```

#### Terminal 4: DRL Agent (WSL2)
```bash
# Activate DRL environment
conda activate drl_py312
cd carla_drl_complete/drl_agent

# Start training with visualization
python train_ppo.py --config ../configs/ppo_training.yaml --visualize
```

#### Terminal 5: Monitoring (WSL2)
```bash
# TensorBoard
conda activate drl_py312
tensorboard --logdir ../monitoring/tensorboard_logs --host 0.0.0.0

# Access at: http://localhost:6006
```

## Verification Checklist

### ✅ System Health Checks

```bash
# Run automated health check
python scripts/health_check.py --full

# Expected outputs:
# ✓ CARLA server connection: OK
# ✓ ROS 2 bridge: OK  
# ✓ ZeroMQ communication: OK
# ✓ DRL agent: OK
# ✓ GPU acceleration: OK
# ✓ All topics publishing: OK
```

### ✅ Performance Validation

```bash
# Performance benchmark
python tests/test_performance.py --duration 60

# Expected metrics:
# - Message latency: <10ms
# - Frame rate: 30 FPS
# - Memory usage: <8GB
# - CPU usage: <80%
# - GPU usage: 60-90%
```

### ✅ Training Validation

```bash
# Short training test (10 episodes)
python drl_agent/train_ppo.py --episodes 10 --test-mode

# Expected results:
# - Episodes complete successfully
# - Reward values logged
# - Model checkpoints saved
# - TensorBoard graphs updated
```

## Troubleshooting

### Common Issues

#### 1. CARLA Connection Failed
```bash
# Check CARLA server status
netstat -an | findstr 2000  # Windows
ss -tulpn | grep 2000       # Linux

# Restart CARLA server
taskkill /f /im CarlaUE4.exe  # Windows
```

#### 2. ROS 2 Bridge Not Starting
```bash
# Check ROS 2 environment
echo $ROS_DISTRO
ros2 node list

# Rebuild bridge
cd ros2_gateway
colcon build --symlink-install --cmake-clean-cache
```

#### 3. Python Environment Issues
```bash
# Verify Python versions
python --version  # Should be 3.6 in CARLA env
python --version  # Should be 3.12 in DRL env

# Reinstall dependencies
pip install -r requirements_py36.txt --force-reinstall
pip install -r requirements_py312.txt --force-reinstall
```

#### 4. ZeroMQ Communication Errors
```bash
# Check port availability
netstat -an | findstr 555   # Windows
ss -tulpn | grep 555        # Linux

# Test ZeroMQ directly
python tests/test_zmq_communication.py
```

#### 5. GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Update CUDA if needed
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Next Steps

### Training Your First Model

1. **Start System**: Use the automated launcher
   ```bash
   python scripts/start_system.py --mode training
   ```

2. **Monitor Progress**: 
   - TensorBoard: http://localhost:6006
   - System metrics: `python scripts/monitor_system.py`

3. **Evaluate Model**:
   ```bash
   python drl_agent/evaluation.py --checkpoint checkpoints/latest_model.pth
   ```

### Advanced Configuration

- **Hyperparameter Tuning**: Edit `configs/ppo_training.yaml`
- **Reward Engineering**: Modify `drl_agent/reward_functions.py`
- **Custom Scenarios**: Add to `configs/carla_sim.yaml`
- **Performance Optimization**: See `docs/PERFORMANCE_TUNING.md`

## Support

- **Issues**: Create GitHub issue with system logs
- **Documentation**: See `docs/` directory
- **Community**: Join project discussions
- **Updates**: Check for new releases regularly

---

**🎯 Installation Status: Ready for First Training Run**

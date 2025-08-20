# CARLA DRL Pipeline - Deployment Commands

## Windows PowerShell Commands

### 1. Setup CARLA Python 3.6 Environment
```powershell
# Create Python 3.6 virtual environment
python -m venv carla_py36_env
.\carla_py36_env\Scripts\Activate.ps1

# Install CARLA client dependencies
pip install pygame numpy opencv-python==4.5.5.64 zmq pyyaml

# Verify CARLA connection
cd CarlaSimulator\PythonAPI\carla
python setup.py install

# Test CARLA client
cd ..\..\..\carla_client_py36
python main.py --config ../configs/sim.yaml
```

### 2. Launch CARLA Server
```powershell
# Navigate to CARLA executable
cd CarlaSimulator\CarlaUE4\Binaries\Win64

# Launch CARLA with Town02, synchronous mode, 50Hz
.\CarlaUE4.exe /Game/Maps/Town02 -windowed -carla-server -benchmark -fps=50 -quality-level=Low

# Alternative: Launch with specific settings
.\CarlaUE4.exe -windowed -carla-server -benchmark -fps=50 -RenderOffScreen
```

### 3. VS Code Tasks
```powershell
# Launch VS Code with workspace
code carla_drl_pipeline.code-workspace

# Run CARLA client from VS Code terminal
Ctrl+Shift+P -> "Tasks: Run Task" -> "Launch CARLA Client"

# Run training from VS Code terminal  
Ctrl+Shift+P -> "Tasks: Run Task" -> "Train DRL Agent"
```

## WSL2 Commands

### 1. Setup WSL2 Environment
```bash
# Update WSL2
wsl --update

# Install Ubuntu 22.04 if not present
wsl --install -d Ubuntu-22.04

# Enter WSL2
wsl -d Ubuntu-22.04

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential cmake git curl wget
```

### 2. Install ROS 2 Humble
```bash
# Setup ROS 2 sources
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-dev-tools

# Install additional packages
sudo apt install -y ros-humble-cv-bridge ros-humble-image-transport
sudo apt install -y libzmq3-dev libjsoncpp-dev

# Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3. Build ROS 2 Workspace
```bash
# Create workspace
mkdir -p ~/carla_drl_ws/src
cd ~/carla_drl_ws

# Clone/copy the ros2_gateway package
cp -r /mnt/c/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/ros2_gateway src/

# Install dependencies
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace
echo "source ~/carla_drl_ws/install/setup.bash" >> ~/.bashrc
source ~/carla_drl_ws/install/setup.bash
```

## Docker Commands

### 1. Docker Setup
```powershell
# Build ROS 2 + DRL Docker image
docker build -t carla-drl-ros2 -f docker/Dockerfile.ros2 .

# Run with GPU support (if available)
docker run --gpus all -it --rm --name carla-drl-container `
  --network host `
  -v ${PWD}:/workspace `
  -v ${PWD}/logs:/workspace/logs `
  -v ${PWD}/models:/workspace/models `
  carla-drl-ros2

# Run without GPU
docker run -it --rm --name carla-drl-container `
  --network host `
  -v ${PWD}:/workspace `
  carla-drl-ros2
```

### 2. Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild services
docker-compose build --no-cache
```

## ROS 2 Specific Commands

### 1. Launch Gateway Node
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash
source ~/carla_drl_ws/install/setup.bash

# Launch gateway node
ros2 run ros2_gateway gateway_node

# Launch with launch file
ros2 launch ros2_gateway gateway.launch.py

# Monitor topics
ros2 topic list
ros2 topic echo /carla/ego_vehicle/camera/image
ros2 topic hz /carla/ego_vehicle/odometry
```

### 2. Debug ROS 2 Communication
```bash
# Check node status
ros2 node list
ros2 node info /carla_gateway_node

# Monitor topic data rates
ros2 topic hz /carla/ego_vehicle/camera/image
ros2 topic bw /carla/ego_vehicle/camera/image

# Record data for analysis
ros2 bag record -a -o carla_session_$(date +%Y%m%d_%H%M%S)

# Play back recorded data
ros2 bag play carla_session_20250818_120000
```

### 3. Launch DRL Training
```bash
# Activate conda environment
conda activate carla_drl

# Install dependencies
pip install -r docker/requirements.txt

# Run training
cd drl_agent
python train.py --config ../configs/train.yaml

# Run inference
python infer.py --model ../models/checkpoints/ppo_carla_1000000_steps.zip --episodes 10
```

## VS Code Configuration Files

### launch.json
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch CARLA Client",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/carla_client_py36/main.py",
            "args": ["--config", "${workspaceFolder}/configs/sim.yaml"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/carla_client_py36",
            "env": {"PYTHONPATH": "${workspaceFolder}/CarlaSimulator/PythonAPI/carla"}
        },
        {
            "name": "Train DRL Agent",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/drl_agent/train.py",
            "args": ["--config", "${workspaceFolder}/configs/train.yaml"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/drl_agent"
        },
        {
            "name": "Run Inference",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/drl_agent/infer.py",
            "args": [
                "--model", "${workspaceFolder}/models/checkpoints/best_model.zip",
                "--config", "${workspaceFolder}/configs/train.yaml",
                "--episodes", "5"
            ],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/drl_agent"
        }
    ]
}
```

### tasks.json
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Launch CARLA Server",
            "type": "shell",
            "command": "./CarlaUE4.exe",
            "args": ["/Game/Maps/Town02", "-windowed", "-carla-server", "-benchmark", "-fps=50"],
            "options": {
                "cwd": "${workspaceFolder}/CarlaSimulator/CarlaUE4/Binaries/Win64"
            },
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            },
            "isBackground": true
        },
        {
            "label": "Build ROS2 Gateway",
            "type": "shell",
            "command": "colcon",
            "args": ["build", "--packages-select", "ros2_gateway"],
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "group": "build"
        },
        {
            "label": "Launch Gateway Node",
            "type": "shell",
            "command": "ros2",
            "args": ["run", "ros2_gateway", "gateway_node"],
            "group": "build",
            "dependsOrder": "sequence",
            "dependsOn": "Build ROS2 Gateway"
        },
        {
            "label": "Train DRL Agent",
            "type": "shell",
            "command": "python",
            "args": ["train.py", "--config", "../configs/train.yaml"],
            "options": {
                "cwd": "${workspaceFolder}/drl_agent"
            },
            "group": "build"
        }
    ]
}
```

## Quick Start Script (PowerShell)
```powershell
# setup_pipeline.ps1
Write-Host "Setting up CARLA DRL Pipeline..." -ForegroundColor Green

# 1. Start CARLA Server
Write-Host "Starting CARLA Server..." -ForegroundColor Yellow
Start-Process -FilePath "CarlaSimulator\CarlaUE4\Binaries\Win64\CarlaUE4.exe" -ArgumentList "/Game/Maps/Town02", "-windowed", "-carla-server", "-benchmark", "-fps=50" -WindowStyle Normal

Start-Sleep -Seconds 10

# 2. Start ROS 2 Gateway (in WSL2)
Write-Host "Starting ROS 2 Gateway..." -ForegroundColor Yellow
wsl -d Ubuntu-22.04 -e bash -c "source /opt/ros/humble/setup.bash && source ~/carla_drl_ws/install/setup.bash && ros2 run ros2_gateway gateway_node"

Start-Sleep -Seconds 5

# 3. Start CARLA Client
Write-Host "Starting CARLA Client..." -ForegroundColor Yellow
& "carla_py36_env\Scripts\python.exe" "carla_client_py36\main.py" --config "configs\sim.yaml"

Write-Host "Pipeline setup complete!" -ForegroundColor Green
```

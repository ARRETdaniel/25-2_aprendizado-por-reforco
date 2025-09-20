#!/bin/bash

# 🚀 CARLA DRL System - Dependency Installation Script
# This script installs all required dependencies for the DRL + CARLA + ROS 2 system

set -e  # Exit on any error

echo "🎯 Starting CARLA DRL System Dependency Installation..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo ""
print_status "Checking prerequisites..."

# Check Python 3.8
if ! command_exists python3; then
    print_error "Python 3 not found!"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $python_version"

# Check pip3
if ! command_exists pip3; then
    print_error "pip3 not found! Installing..."
    sudo apt update && sudo apt install -y python3-pip
fi

# Check if we're in the project directory
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found. Ensure you're in the carla_drl_project directory."
    cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/carla_drl_project
fi

echo ""
print_status "=== PHASE 1: Core ML Dependencies ==="

# Install PyTorch with CUDA support (for RTX 2060)
print_status "Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
print_status "Verifying PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" && print_success "PyTorch installed successfully!" || print_error "PyTorch installation failed!"

echo ""
print_status "=== PHASE 2: Deep Reinforcement Learning Dependencies ==="

# Install Stable Baselines3 and related packages
print_status "Installing Stable-Baselines3..."
pip3 install stable-baselines3[extra]

# Install additional ML libraries
print_status "Installing additional ML libraries..."
pip3 install gymnasium opencv-python numpy scipy matplotlib tensorboard

# Install project requirements
if [ -f "requirements.txt" ]; then
    print_status "Installing project requirements..."
    pip3 install -r requirements.txt
else
    print_warning "requirements.txt not found, skipping..."
fi

echo ""
print_status "=== PHASE 3: ROS 2 and CARLA Integration ==="

# Source ROS 2 environment
if [ -f "/opt/ros/foxy/setup.bash" ]; then
    print_status "Sourcing ROS 2 Foxy environment..."
    source /opt/ros/foxy/setup.bash
    print_success "ROS 2 environment sourced"
else
    print_error "ROS 2 Foxy not found! Please install ROS 2 Foxy first."
    exit 1
fi

# Try to install carla_ros_bridge via apt first
print_status "Attempting to install carla-ros-bridge via apt..."
if sudo apt install -y ros-foxy-carla-ros-bridge ros-foxy-carla-msgs; then
    print_success "carla-ros-bridge installed via apt"
else
    print_warning "APT installation failed, will need to build from source"
    
    # Prepare for source build
    print_status "Preparing for source build of carla-ros-bridge..."
    
    # Create workspace if it doesn't exist
    CARLA_WS_DIR="$HOME/carla_ws"
    if [ ! -d "$CARLA_WS_DIR" ]; then
        mkdir -p "$CARLA_WS_DIR/src"
        print_status "Created workspace at $CARLA_WS_DIR"
    fi
    
    # Clone the repository
    cd "$CARLA_WS_DIR/src"
    if [ ! -d "ros-bridge" ]; then
        print_status "Cloning carla-ros-bridge repository..."
        git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git
    else
        print_status "carla-ros-bridge repository already exists"
    fi
    
    # Install dependencies
    cd "$CARLA_WS_DIR"
    print_status "Installing ROS dependencies..."
    rosdep update
    rosdep install --from-paths src --ignore-src -r -y
    
    # Build the workspace
    print_status "Building carla-ros-bridge from source..."
    source /opt/ros/foxy/setup.bash
    colcon build --packages-select carla_ros_bridge carla_msgs carla_waypoint_publisher
    
    if [ $? -eq 0 ]; then
        print_success "carla-ros-bridge built successfully!"
        echo "export CARLA_WS_PATH=$CARLA_WS_DIR" >> ~/.bashrc
        echo "source $CARLA_WS_DIR/install/setup.bash" >> ~/.bashrc
    else
        print_error "carla-ros-bridge build failed!"
    fi
fi

echo ""
print_status "=== PHASE 4: Installation Verification ==="

# Test Python imports
print_status "Testing Python imports..."

python3 -c "
import sys
import traceback

packages = [
    ('carla', 'CARLA Python API'),
    ('torch', 'PyTorch'),
    ('stable_baselines3', 'Stable-Baselines3'),
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('gymnasium', 'Gymnasium'),
    ('rclpy', 'ROS 2 Python'),
]

print('=' * 50)
print('Python Package Verification')
print('=' * 50)

all_good = True
for package, name in packages:
    try:
        __import__(package)
        print(f'✅ {name:<20} - OK')
    except ImportError as e:
        print(f'❌ {name:<20} - MISSING')
        all_good = False
    except Exception as e:
        print(f'⚠️  {name:<20} - ERROR: {e}')
        all_good = False

print('=' * 50)
if all_good:
    print('🎉 All packages imported successfully!')
else:
    print('⚠️  Some packages are missing or have errors')
print('=' * 50)
"

# Test CUDA availability
print_status "Testing CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA is available')
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA is not available')
"

# Test ROS 2 packages
print_status "Testing ROS 2 packages..."
source /opt/ros/foxy/setup.bash
if [ -f "$HOME/carla_ws/install/setup.bash" ]; then
    source "$HOME/carla_ws/install/setup.bash"
fi

ros2 pkg list | grep carla && print_success "CARLA ROS 2 packages found" || print_warning "CARLA ROS 2 packages not found"

echo ""
print_status "=== PHASE 5: System Information ==="

# Display system information
print_status "System Information:"
echo "   OS: $(lsb_release -d | cut -f2)"
echo "   Kernel: $(uname -r)"
echo "   Python: $(python3 --version)"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Not detected')"
echo "   VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'Not detected') MiB"

echo ""
print_status "=== INSTALLATION COMPLETE ==="

echo ""
print_success "🎉 Dependency installation completed!"
echo ""
print_status "Next steps:"
echo "1. Restart your terminal to ensure all environment variables are loaded"
echo "2. Run 'source ~/.bashrc' to load CARLA workspace paths"
echo "3. Test CARLA server: ./CarlaUE4.sh -RenderOffScreen"
echo "4. Begin implementing the CARLA interface (see NEXT_STEPS_ANALYSIS.md)"
echo ""
print_status "For troubleshooting, check the error messages above."
print_status "Documentation available in carla_drl_project/docs/"

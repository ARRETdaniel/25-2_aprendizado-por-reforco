# System Architecture

┌─────────────────────┐     ┌──────────────────────┐
│ CARLA Client (3.6)  │     │ Detection Server (3.12)
│                     │     │                      │
│ - CARLA integration │ ──► │ - YOLOv8             │
│ - Vehicle control   │     │ - GPU acceleration   │
│ - Trajectory plan   │ ◄── │ - Driver assistance  │
│ - Visualization     │     │ - High FPS proc.
└─────────────────────┘     └──────────────────────┘
                      Socket

# YOLOv8 GPU Detection Server Setup

## System Requirements

- NVIDIA GPU (RTX 2060 or better)
- NVIDIA Drivers (535.0 or newer)
- CUDA Toolkit 11.8
- cuDNN 8.7.0 for CUDA 11.x
- Windows 10/11 or Linux

## Step 1: Install NVIDIA Drivers & CUDA Toolkit

1. Download and install the latest NVIDIA drivers for your GPU:
   https://www.nvidia.com/Download/index.aspx

2. Download and install CUDA Toolkit 11.8:
   https://developer.nvidia.com/cuda-11-8-0-download-archive

3. Download and install cuDNN 8.7 for CUDA 11.x:
   https://developer.nvidia.com/cudnn

## Step 2: Set Up Conda Environment

```bash
# Create a new conda environment with Python 3.12
conda create -n yolo_gpu python=3.12 -y
conda activate yolo_gpu

# Install PyTorch with CUDA support
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics==8.3.143 opencv-python==4.8.1.78 msgpack==1.0.7 msgpack-numpy==0.4.8


cd CarlaSimulator\PythonClient\FinalProject\detector_socket
conda activate yolo_gpu
python detector_server.py

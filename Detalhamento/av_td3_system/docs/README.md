# TD3-Based Autonomous Vehicle System

End-to-End Visual Navigation in CARLA + ROS 2 with Docker

## Project Overview

This project implements a TD3 (Twin Delayed DDPG) based autonomous vehicle system for end-to-end visual navigation in the CARLA simulator using ROS 2.

## Quick Start

### Prerequisites
- Docker Engine
- NVIDIA Container Toolkit
- NVIDIA GPU with drivers

### Building the Docker Image
```bash
cd av_td3_system
docker build -t td3-av-system:latest .
```

### Running Training
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  --net=host \
  td3-av-system:latest
```

## Project Structure

See `PRE_DEVELOPMENT_PLAN_DOCKER.md` for complete documentation.

## Author

Daniel Terra Gomes
Federal University of Minas Gerais (UFMG)
Master's Research - Reinforcement Learning (2025-2)

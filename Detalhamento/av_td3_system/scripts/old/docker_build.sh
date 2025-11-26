#!/bin/bash
# scripts/docker_build.sh
# Helper script to build TD3 AV System Docker image

set -e

echo "========================================"
echo "Building TD3 AV System Docker Image"
echo "========================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Build the image with multiple tags
echo "Building image: td3-av-system:v2.0-python310 (also tagged as :latest)"
docker build -t td3-av-system:v2.0-python310 -t td3-av-system:latest .

echo ""
echo "========================================"
echo "Testing image..."
echo "========================================"

# Test imports
docker run --rm td3-av-system:latest python3 -c "
import sys
print('Python version:', sys.version)
print('Testing imports...')

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
    print('  CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('✗ PyTorch import failed:', e)

try:
    import carla
    print('✓ CARLA API: OK')
except Exception as e:
    print('✗ CARLA API import failed:', e)

try:
    import rclpy
    print('✓ ROS 2 (rclpy): OK')
except Exception as e:
    print('✗ ROS 2 import failed:', e)

try:
    import cv2
    print('✓ OpenCV:', cv2.__version__)
except Exception as e:
    print('✗ OpenCV import failed:', e)

try:
    import numpy
    print('✓ NumPy:', numpy.__version__)
except Exception as e:
    print('✗ NumPy import failed:', e)

print('All core imports completed!')
"

echo ""
echo "========================================"
echo "Build successful! 🎉"
echo "========================================"
echo "Images: td3-av-system:v2.0-python310, td3-av-system:latest"
echo ""
echo "Next steps:"
echo "  1. Run training:    ./scripts/docker_run_train.sh"
echo "  2. Or use docker-compose: docker-compose up td3-training"
echo ""

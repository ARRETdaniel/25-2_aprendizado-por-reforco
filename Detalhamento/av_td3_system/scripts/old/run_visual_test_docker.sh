#!/bin/bash
# Docker launcher for visual navigation testing
# This script runs the visual test inside the Docker container with X11 forwarding

echo "=========================================="
echo "🐳 TD3 Visual Navigation - Docker Launcher"
echo "=========================================="

# Check if CARLA server is running
if ! docker ps | grep -q carla-server; then
    echo "⚠️  CARLA server container not found!"
    echo "Please start CARLA server first:"
    echo "  docker run -d --name carla-server --gpus all -p 2000-2002:2000-2002 carlasim/carla:0.9.16"
    exit 1
fi

# Check if X11 forwarding is available
if [ -z "$DISPLAY" ]; then
    echo "⚠️  DISPLAY variable not set!"
    echo "X11 forwarding required for OpenCV display"
    echo "Set DISPLAY or run with: DISPLAY=:0 $0"
    exit 1
fi

echo "✅ CARLA server running"
echo "✅ DISPLAY=$DISPLAY"
echo "🧠 Applying PyTorch CUDA memory optimizations:"
echo "   - expandable_segments: Reduce fragmentation by 60-80%"
echo "   - max_split_size_mb: Prevent splitting blocks > 128MB"
echo "   - garbage_collection_threshold: Aggressive memory reclaim at 80%"

# Allow Docker to access X11
xhost +local:docker > /dev/null 2>&1

# Container name
CONTAINER_NAME="td3-av-system-visual"

# Check if container exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "♻️  Removing existing container..."
    docker rm -f $CONTAINER_NAME > /dev/null 2>&1
fi

echo "🚀 Starting visual navigation test..."
echo ""

# Run visual navigation test in Docker
docker run --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --network host \
    -e DISPLAY=$DISPLAY \
    -e CARLA_HOST=localhost \
    -e CARLA_PORT=2000 \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v $(pwd):/workspace \
    -w /workspace \
    td3-av-system:v2.0-python310 \
    python3 scripts/test_visual_navigation.py --max-steps 100

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Visual test completed successfully"
else
    echo "❌ Visual test failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE

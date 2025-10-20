#!/bin/bash
# scripts/docker_run_train.sh
# Helper script to run training in Docker container

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default values
AGENT=${1:-TD3}  # Default: TD3
EPISODES=${2:-100}  # Default: 100 episodes
TRAFFIC_DENSITY=${3:-50}  # Default: 50 NPCs

echo "========================================"
echo "Starting TD3 AV Training in Docker"
echo "========================================"
echo "Agent: $AGENT"
echo "Episodes: $EPISODES"
echo "Traffic Density: $TRAFFIC_DENSITY NPCs"
echo "========================================"
echo ""

# Create data directories if they don't exist
mkdir -p "$PROJECT_ROOT/data/waypoints"
mkdir -p "$PROJECT_ROOT/data/checkpoints"
mkdir -p "$PROJECT_ROOT/data/logs"
mkdir -p "$PROJECT_ROOT/results"

# Run the container
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -v "$PROJECT_ROOT/data":/workspace/data \
  -v "$PROJECT_ROOT/results":/workspace/results \
  -v "$PROJECT_ROOT/config":/workspace/av_td3_system/config \
  --net=host \
  -e AGENT_TYPE="$AGENT" \
  -e EPISODES="$EPISODES" \
  -e TRAFFIC_DENSITY="$TRAFFIC_DENSITY" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name "td3_training_${AGENT,,}_$$" \
  td3-av-system:latest \
  python3 /workspace/av_td3_system/scripts/train_${AGENT,,}.py \
    --config /workspace/av_td3_system/config/${AGENT,,}_config.yaml \
    --episodes "$EPISODES" \
    --traffic-density "$TRAFFIC_DENSITY"

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo "Results saved to: $PROJECT_ROOT/results"
echo "Checkpoints saved to: $PROJECT_ROOT/data/checkpoints"
echo ""

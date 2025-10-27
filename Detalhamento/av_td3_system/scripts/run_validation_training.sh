#!/bin/bash
###############################################################################
# TD3 Validation Training Run (20,000 steps)
#
# Purpose: Comprehensive validation of TD3 solution before full 1M-step run
# on supercomputer. Collects critical data to verify:
#   1. Reward function is working correctly (no "stand still" exploit)
#   2. CNN feature extraction is providing useful visual information
#   3. Waypoint data is properly integrated into state representation
#   4. Agent is learning to navigate (improving over time)
#   5. All components (CARLA, environment, agent) work together
#
# Data Collection:
#   - TensorBoard logs (metrics, learning curves)
#   - Detailed debug logs (waypoints, CNN features, rewards)
#   - Checkpoint at 20k steps for resumption
#   - Episode statistics and success rates
#
# Expected Runtime: ~2-3 hours on laptop (20k steps)
# Expected Storage: ~500 MB (logs + checkpoint)
#
# Author: Daniel Terra
# Date: October 26, 2024
###############################################################################

set -e  # Exit on error

# Configuration
SCENARIO=0                    # 20 NPCs (light traffic for faster validation)
MAX_TIMESTEPS=20000           # 20k steps for validation
EVAL_FREQ=2000                # Evaluate every 2k steps (10 evaluations total)
EVAL_EPISODES=5               # 5 episodes per evaluation (faster than 10)
CHECKPOINT_FREQ=5000          # Save checkpoints every 5k steps (4 total)
SEED=42                       # Fixed seed for reproducibility
DEVICE="cpu"                  # CPU for laptop (change to "cuda" if GPU available)
DEBUG_MODE="--debug"          # Enable visual debugging and detailed logs

# Directories
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/data/logs"
CHECKPOINT_DIR="${PROJECT_ROOT}/data/checkpoints"
VALIDATION_LOG="validation_training_20k_$(date +%Y%m%d_%H%M%S).log"

# Docker configuration
DOCKER_IMAGE="td3-av-system:v2.0-python310"
WORKSPACE="/workspace"  # Path inside container
CARLA_HOST="host.docker.internal"  # Or "172.17.0.1" on Linux
CARLA_PORT=2000

###############################################################################
# Pre-flight Checks
###############################################################################

echo "=========================================================================="
echo "TD3 VALIDATION TRAINING RUN (20,000 STEPS)"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Scenario:           ${SCENARIO} (20 NPCs)"
echo "  Max Timesteps:      ${MAX_TIMESTEPS}"
echo "  Evaluation Freq:    ${EVAL_FREQ} steps (${EVAL_EPISODES} episodes each)"
echo "  Checkpoint Freq:    ${CHECKPOINT_FREQ} steps"
echo "  Device:             ${DEVICE}"
echo "  Debug Mode:         Enabled"
echo "  Seed:               ${SEED}"
echo ""
echo "Expected Runtime:    ~2-3 hours"
echo "Expected Storage:    ~500 MB"
echo ""
echo "=========================================================================="
echo ""

# Check if CARLA server is running
echo "[CHECK] Verifying CARLA server is running..."
if docker ps | grep -q carla-server; then
    echo "[CHECK] ✓ CARLA server is running"
else
    echo "[CHECK] ✗ CARLA server is NOT running!"
    echo "[CHECK] Please start CARLA server first:"
    echo ""
    echo "  docker run -d --name carla-server --rm \\"
    echo "    --network host --gpus all \\"
    echo "    -e SDL_VIDEODRIVER=offscreen \\"
    echo "    carlasim/carla:0.9.16 \\"
    echo "    /bin/bash ./CarlaUE4.sh -RenderOffScreen"
    echo ""
    exit 1
fi

# Check if Docker image exists
echo "[CHECK] Verifying Docker image exists..."
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${DOCKER_IMAGE}$"; then
    echo "[CHECK] ✓ Docker image found: ${DOCKER_IMAGE}"
else
    echo "[CHECK] ✗ Docker image NOT found: ${DOCKER_IMAGE}"
    echo "[CHECK] Please build the image first"
    exit 1
fi

# Create log directory
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Enable X11 forwarding for debug visualization
echo "[CHECK] Enabling X11 forwarding for debug visualization..."
xhost +local:docker 2>/dev/null || echo "[WARNING] xhost not available, debug visualization may not work"

echo ""
echo "=========================================================================="
echo "STARTING VALIDATION TRAINING"
echo "=========================================================================="
echo ""
echo "Training will log to: ${LOG_DIR}/${VALIDATION_LOG}"
echo "Press Ctrl+C to stop training gracefully"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOG_DIR}/${VALIDATION_LOG}"
echo ""
echo "Monitor TensorBoard with:"
echo "  cd av_td3_system && tensorboard --logdir data/logs"
echo ""
echo "=========================================================================="
echo ""

# Wait 3 seconds for user to cancel if needed
sleep 3

###############################################################################
# Run Training
###############################################################################

cd "$(dirname "$0")/.." || exit 1  # Go to av_td3_system root

docker run --rm \
  --network host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY="${DISPLAY}" \
  -e CARLA_HOST="${CARLA_HOST}" \
  -e CARLA_PORT="${CARLA_PORT}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd):/workspace" \
  -w /workspace \
  "${DOCKER_IMAGE}" \
  python3 scripts/train_td3.py \
    --scenario "${SCENARIO}" \
    --max-timesteps "${MAX_TIMESTEPS}" \
    --eval-freq "${EVAL_FREQ}" \
    --num-eval-episodes "${EVAL_EPISODES}" \
    --checkpoint-freq "${CHECKPOINT_FREQ}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    ${DEBUG_MODE} \
  2>&1 | tee "${LOG_DIR}/${VALIDATION_LOG}"

EXIT_CODE=$?

echo ""
echo "=========================================================================="
echo "VALIDATION TRAINING COMPLETED"
echo "=========================================================================="
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[SUCCESS] Training completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Review training logs:     less ${LOG_DIR}/${VALIDATION_LOG}"
    echo "  2. Check TensorBoard:        tensorboard --logdir ${LOG_DIR}"
    echo "  3. Analyze checkpoints:      ls -lh ${CHECKPOINT_DIR}/"
    echo "  4. Run validation analysis:  python3 scripts/analyze_validation_run.py"
    echo ""
    echo "If validation is successful, proceed with full 1M-step training on supercomputer."
else
    echo "[ERROR] Training failed with exit code ${EXIT_CODE}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs:               less ${LOG_DIR}/${VALIDATION_LOG}"
    echo "  2. Verify CARLA is running:  docker ps | grep carla"
    echo "  3. Check GPU availability:   nvidia-smi"
    echo ""
fi

echo "=========================================================================="

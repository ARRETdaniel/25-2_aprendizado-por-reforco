#!/bin/bash
# Minimal ROS Bridge Test Script
# Following official CARLA ROS Bridge documentation
#
# This script:
# 1. Launches CARLA + minimal ROS bridge (NO manual control)
# 2. Spawns ego vehicle via ROS service
# 3. Runs control test
#
# Based on: https://carla.readthedocs.io/projects/ros-bridge/en/latest/

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Minimal ROS Bridge Control Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Navigate to docker directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/../docker"
cd "${DOCKER_DIR}"

# Step 1: Clean up any existing containers
echo -e "${YELLOW}[1/6]${NC} Cleaning up existing containers..."
docker compose -f docker-compose.minimal-test.yml down 2>/dev/null || true
sleep 2

# Step 2: Launch CARLA + ROS Bridge
echo -e "${YELLOW}[2/6]${NC} Launching CARLA server and minimal ROS bridge..."
docker compose -f docker-compose.minimal-test.yml up -d

# Step 3: Wait for bridge and ego vehicle to be ready
echo -e "${YELLOW}[3/5]${NC} Waiting for ROS bridge to spawn ego vehicle (45 seconds)..."
sleep 45

# Step 4: Verify bridge is running and ego vehicle topics exist
echo -e "${YELLOW}[4/5]${NC} Verifying ROS bridge and ego vehicle status..."
docker exec ros2-bridge-minimal bash -c "source /ros_entrypoint.sh && ros2 topic list | grep -q carla" && \
    echo -e "${GREEN}✓${NC} ROS bridge is publishing topics" || \
    { echo -e "${RED}✗${NC} ROS bridge not ready!"; exit 1; }

echo -e "${BLUE}Checking for ego vehicle topics...${NC}"
docker exec ros2-bridge-minimal bash -c "source /ros_entrypoint.sh && ros2 topic list | grep ego_vehicle" || \
    { echo -e "${RED}✗${NC} Ego vehicle topics not found!"; exit 1; }

# Verify odometry is publishing
echo -e "${BLUE}Verifying odometry is publishing...${NC}"
docker exec ros2-bridge-minimal bash -c "source /ros_entrypoint.sh && timeout 3 ros2 topic echo /carla/ego_vehicle/odometry --once" >/dev/null 2>&1 && \
    echo -e "${GREEN}✓${NC} Odometry is publishing!" || \
    { echo -e "${RED}✗${NC} Odometry not publishing!"; exit 1; }

# Step 5: Run control test
echo -e "${YELLOW}[5/5]${NC} Running vehicle control test..."
echo -e "${BLUE}Copying test script into container...${NC}"

# Copy test script to container
TEST_SCRIPT="${SCRIPT_DIR}/../tests/test_minimal_ros_bridge.py"
docker cp "${TEST_SCRIPT}" ros2-bridge-minimal:/tmp/test_control.py

echo -e "${BLUE}Executing test (this will take ~10 seconds)...${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"

# Run test and capture exit code
docker exec ros2-bridge-minimal bash -c "source /ros_entrypoint.sh && python3 /tmp/test_control.py"
TEST_EXIT_CODE=$?

echo -e "${BLUE}========================================${NC}"
echo ""

# Report results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✓ TEST PASSED SUCCESSFULLY!        ║${NC}"
    echo -e "${GREEN}║   Vehicle control via ROS Bridge OK  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Vehicle control verified ✓"
    echo "  2. Ready to extract PID + Pure Pursuit controllers"
    echo "  3. Proceed to Phase 2.3"
else
    echo -e "${RED}╔═══════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ✗ TEST FAILED                      ║${NC}"
    echo -e "${RED}║   Vehicle did not respond to control ║${NC}"
    echo -e "${RED}╚═══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  - Check bridge logs: docker logs ros2-bridge-minimal"
    echo "  - Check CARLA logs: docker logs carla-minimal-test"
    echo "  - Verify topics: docker exec ros2-bridge-minimal bash -c 'source /ros_entrypoint.sh && ros2 topic list'"
fi

echo ""
echo -e "${BLUE}Containers are still running for inspection.${NC}"
echo -e "${BLUE}To stop: cd ${DOCKER_DIR} && docker compose -f docker-compose.minimal-test.yml down${NC}"
echo ""

exit $TEST_EXIT_CODE

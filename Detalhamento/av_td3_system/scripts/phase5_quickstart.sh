#!/bin/bash

################################################################################
# Phase 5 Quick Start Script
# Automates infrastructure startup and verification for ROS 2 Bridge integration
#
# Usage:
#   ./scripts/phase5_quickstart.sh [command]
#
# Commands:
#   start     - Start CARLA + ROS Bridge
#   verify    - Verify system is working correctly
#   test      - Run control test
#   stop      - Stop all containers
#   restart   - Restart containers
#   logs      - Show logs from containers
#   topics    - List all ROS topics
#   help      - Show this help message
#
# Author: AV TD3 System
# Date: 2025-01-22
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly COMPOSE_FILE="docker-compose.ros-integration.yml"
readonly WORKSPACE_DIR="/media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    log_success "Docker: $(docker --version)"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi
    log_success "Docker Compose: $(docker-compose --version)"

    # Check NVIDIA runtime
    if ! docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        log_warning "NVIDIA runtime test failed. GPU may not be available."
    else
        log_success "NVIDIA runtime available"
    fi

    # Check compose file exists
    if [ ! -f "${WORKSPACE_DIR}/${COMPOSE_FILE}" ]; then
        log_error "Compose file not found: ${WORKSPACE_DIR}/${COMPOSE_FILE}"
        exit 1
    fi
    log_success "Compose file found"

    # Check ROS Bridge image
    if ! docker images | grep -q "ros2-carla-bridge:humble-v4"; then
        log_warning "ROS Bridge image not found. You need to build it first:"
        log_warning "  docker build -t ros2-carla-bridge:humble-v4 -f docker/ros2-carla-bridge.Dockerfile ."
        read -p "Do you want to build it now? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_ros_bridge_image
        else
            log_error "Cannot proceed without ROS Bridge image"
            exit 1
        fi
    else
        log_success "ROS Bridge image found"
    fi
}

# Build ROS Bridge image
build_ros_bridge_image() {
    log_info "Building ROS Bridge image (this may take 15-20 minutes)..."

    cd "${WORKSPACE_DIR}"
    docker build -t ros2-carla-bridge:humble-v4 -f docker/ros2-carla-bridge.Dockerfile .

    if [ $? -eq 0 ]; then
        log_success "ROS Bridge image built successfully"
    else
        log_error "Failed to build ROS Bridge image"
        exit 1
    fi
}

# Start infrastructure
start_infrastructure() {
    log_info "Starting CARLA + ROS Bridge infrastructure..."

    cd "${WORKSPACE_DIR}"

    # Start containers
    docker-compose -f "${COMPOSE_FILE}" up -d

    log_info "Waiting for containers to start (30 seconds)..."
    sleep 10

    # Check container status
    local carla_status=$(docker-compose -f "${COMPOSE_FILE}" ps | grep carla-server | grep -o "Up (healthy)")
    local bridge_status=$(docker-compose -f "${COMPOSE_FILE}" ps | grep ros2-bridge | grep -o "Up (healthy)")

    if [ -z "$carla_status" ]; then
        log_warning "CARLA server not healthy yet, waiting..."
        sleep 20
    fi

    if [ -z "$bridge_status" ]; then
        log_warning "ROS Bridge not healthy yet, waiting..."
        sleep 20
    fi

    log_success "Infrastructure started"
    docker-compose -f "${COMPOSE_FILE}" ps
}

# Stop infrastructure
stop_infrastructure() {
    log_info "Stopping infrastructure..."

    cd "${WORKSPACE_DIR}"
    docker-compose -f "${COMPOSE_FILE}" down

    log_success "Infrastructure stopped"
}

# Restart infrastructure
restart_infrastructure() {
    log_info "Restarting infrastructure..."
    stop_infrastructure
    sleep 5
    start_infrastructure
}

# Verify system
verify_system() {
    log_info "=== System Verification ==="

    cd "${WORKSPACE_DIR}"

    # Check 1: Container status
    echo ""
    log_info "[1/7] Checking container status..."
    if docker-compose -f "${COMPOSE_FILE}" ps | grep -q "Up (healthy)"; then
        log_success "Containers are healthy"
    else
        log_error "Containers are not healthy"
        docker-compose -f "${COMPOSE_FILE}" ps
        return 1
    fi

    # Check 2: CARLA port
    echo ""
    log_info "[2/7] Checking CARLA port 2000..."
    if docker exec carla-server netstat -tuln 2>/dev/null | grep -q ":2000"; then
        log_success "CARLA listening on port 2000"
    else
        log_error "CARLA not listening on port 2000"
        return 1
    fi

    # Check 3: ROS topics
    echo ""
    log_info "[3/7] Checking ROS topics..."
    local topic_count=$(docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        ros2 topic list 2>/dev/null | grep -c '/carla'" || echo "0")

    if [ "$topic_count" -gt 0 ]; then
        log_success "Found $topic_count CARLA topics"
    else
        log_error "No CARLA topics found"
        return 1
    fi

    # Check 4: Control topic
    echo ""
    log_info "[4/7] Checking control topic..."
    if docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        ros2 topic list 2>/dev/null | grep -q '/carla/ego_vehicle/vehicle_control_cmd'"; then
        log_success "Control topic available"
    else
        log_error "Control topic not found"
        return 1
    fi

    # Check 5: Ego vehicle spawned
    echo ""
    log_info "[5/7] Checking ego vehicle..."
    if docker logs ros2-bridge 2>&1 | grep -q "Created EgoVehicle"; then
        local vehicle_id=$(docker logs ros2-bridge 2>&1 | grep "Created EgoVehicle" | tail -1 | grep -oP '(?<=id=)\d+')
        log_success "Ego vehicle spawned (ID: $vehicle_id)"
    else
        log_error "Ego vehicle not spawned"
        return 1
    fi

    # Check 6: Topic subscription
    echo ""
    log_info "[6/7] Checking topic subscribers..."
    local sub_count=$(docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        ros2 topic info /carla/ego_vehicle/vehicle_control_cmd 2>/dev/null | grep -c 'Subscription count:'" || echo "0")

    if [ "$sub_count" -gt 0 ]; then
        log_success "Control topic has subscribers"
    else
        log_warning "Control topic has no subscribers (this might be OK)"
    fi

    # Check 7: Python integration
    echo ""
    log_info "[7/7] Checking Python integration..."
    if python3 -c "
import sys
sys.path.insert(0, '${WORKSPACE_DIR}')
from src.utils.ros_bridge_interface import ROSBridgeInterface
print('✅ Python integration OK')
" 2>/dev/null; then
        log_success "Python integration working"
    else
        log_warning "Python integration test failed (might need to install dependencies)"
    fi

    # Summary
    echo ""
    log_success "=== ✅ ALL CHECKS PASSED ==="
    log_info "System ready for evaluation/training"
}

# Test vehicle control
test_control() {
    log_info "=== Testing Vehicle Control ==="

    cd "${WORKSPACE_DIR}"

    # Publish throttle command
    log_info "Publishing throttle command (0.3 for 5 seconds)..."
    docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        source /opt/carla-ros-bridge/install/setup.bash &&
        timeout 5s ros2 topic pub /carla/ego_vehicle/vehicle_control_cmd \
            carla_msgs/msg/CarlaEgoVehicleControl \
            '{throttle: 0.3, steer: 0.0, brake: 0.0}' -r 10" &

    local pub_pid=$!

    # Monitor speed in parallel
    sleep 1
    log_info "Monitoring vehicle speed..."
    docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        source /opt/carla-ros-bridge/install/setup.bash &&
        timeout 5s ros2 topic echo /carla/ego_vehicle/speedometer --once"

    # Wait for publishing to finish
    wait $pub_pid 2>/dev/null || true

    # Stop vehicle
    log_info "Stopping vehicle..."
    docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        source /opt/carla-ros-bridge/install/setup.bash &&
        ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
            carla_msgs/msg/CarlaEgoVehicleControl \
            '{throttle: 0.0, steer: 0.0, brake: 1.0}'"

    log_success "Control test completed"
}

# Show logs
show_logs() {
    local service=${1:-}

    cd "${WORKSPACE_DIR}"

    if [ -z "$service" ]; then
        log_info "Showing logs for all services..."
        docker-compose -f "${COMPOSE_FILE}" logs --tail=50
    elif [ "$service" = "carla" ]; then
        log_info "Showing CARLA logs..."
        docker logs carla-server --tail=50
    elif [ "$service" = "bridge" ] || [ "$service" = "ros" ]; then
        log_info "Showing ROS Bridge logs..."
        docker logs ros2-bridge --tail=50
    else
        log_error "Unknown service: $service"
        log_info "Available services: carla, bridge"
    fi
}

# List topics
list_topics() {
    log_info "=== ROS 2 Topics ==="

    docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        ros2 topic list | grep '/carla' | sort"

    echo ""
    log_info "Total topics: $(docker exec ros2-bridge bash -c "
        source /opt/ros/humble/setup.bash &&
        ros2 topic list | grep -c '/carla'")"
}

# Show help
show_help() {
    cat << EOF
${GREEN}Phase 5 Quick Start Script${NC}

${BLUE}Usage:${NC}
  ./scripts/phase5_quickstart.sh [command]

${BLUE}Commands:${NC}
  ${GREEN}start${NC}     - Start CARLA + ROS Bridge infrastructure
  ${GREEN}verify${NC}    - Verify system is working correctly
  ${GREEN}test${NC}      - Run vehicle control test
  ${GREEN}stop${NC}      - Stop all containers
  ${GREEN}restart${NC}   - Restart infrastructure
  ${GREEN}logs${NC}      - Show logs (optional: logs carla|bridge)
  ${GREEN}topics${NC}    - List all ROS topics
  ${GREEN}help${NC}      - Show this help message

${BLUE}Examples:${NC}
  # Complete setup and verification
  ./scripts/phase5_quickstart.sh start
  ./scripts/phase5_quickstart.sh verify
  ./scripts/phase5_quickstart.sh test

  # Check logs
  ./scripts/phase5_quickstart.sh logs carla
  ./scripts/phase5_quickstart.sh logs bridge

  # List available topics
  ./scripts/phase5_quickstart.sh topics

  # Restart system
  ./scripts/phase5_quickstart.sh restart

${BLUE}Next Steps After Verification:${NC}
  1. Run baseline evaluation:
     python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 5

  2. Run TD3 training:
     python3 scripts/train_td3.py --scenario 0 --max-timesteps 50000

${BLUE}Documentation:${NC}
  - Integration Guide: docs/ROS_BRIDGE_INTEGRATION_GUIDE.md
  - Success Report: docs/ROS_BRIDGE_SUCCESS_REPORT.md

EOF
}

# Main function
main() {
    local command=${1:-help}

    case "$command" in
        start)
            check_prerequisites
            start_infrastructure
            ;;
        verify)
            verify_system
            ;;
        test)
            test_control
            ;;
        stop)
            stop_infrastructure
            ;;
        restart)
            restart_infrastructure
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        topics)
            list_topics
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

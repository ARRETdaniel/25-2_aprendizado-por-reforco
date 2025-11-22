#!/bin/bash
# Build script for ROS 2 + CARLA Bridge Docker image
# Usage: ./build_ros2_bridge.sh [OPTIONS]
# Options:
#   -r, --ros-distro    ROS 2 distribution (default: foxy)
#   -c, --carla-version CARLA version (default: 0.9.16)
#   -h, --help          Show this help message

set -e

# Default values
ROS_DISTRO="foxy"
CARLA_VERSION="0.9.16"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Build ROS 2 + CARLA Bridge Docker Image

Usage: $0 [OPTIONS]

Options:
    -r, --ros-distro DISTRO     ROS 2 distribution (default: foxy)
    -c, --carla-version VERSION CARLA version (default: 0.9.16)
    -h, --help                  Show this help message

Examples:
    $0
    $0 --ros-distro foxy --carla-version 0.9.16
    $0 -r humble -c 0.9.16

Supported ROS 2 distributions:
    - foxy (Ubuntu 20.04, default)
    - humble (Ubuntu 22.04)
    - iron (Ubuntu 22.04)

Supported CARLA versions:
    - 0.9.16 (default, latest)
    - 0.9.15
    - 0.9.14
    - 0.9.13

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--ros-distro)
            ROS_DISTRO="$2"
            shift 2
            ;;
        -c|--carla-version)
            CARLA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Verify Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Verify nvidia-docker is installed (for GPU support)
if ! docker run --rm --runtime=nvidia nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    print_warn "NVIDIA Docker runtime not detected. GPU support may not work."
    print_warn "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Print build configuration
print_info "Build Configuration:"
echo "  ROS 2 Distribution: ${ROS_DISTRO}"
echo "  CARLA Version: ${CARLA_VERSION}"
echo "  Project Root: ${PROJECT_ROOT}"
echo "  Dockerfile: ${PROJECT_ROOT}/docker/ros2-carla-bridge.Dockerfile"
echo ""

# Confirm build
read -p "Proceed with build? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Build cancelled."
    exit 0
fi

# Start build
print_info "Starting Docker build..."
print_info "This may take 10-20 minutes on first build."
echo ""

# Build command
DOCKER_BUILDKIT=1 docker build \
    -t "ros2-carla-bridge:${ROS_DISTRO}" \
    -f "${PROJECT_ROOT}/docker/ros2-carla-bridge.Dockerfile" \
    --build-arg ROS_DISTRO="${ROS_DISTRO}" \
    --build-arg CARLA_VERSION="${CARLA_VERSION}" \
    --progress=plain \
    "${PROJECT_ROOT}"

BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    print_info "✅ Build successful!"
    echo ""
    print_info "Image: ros2-carla-bridge:${ROS_DISTRO}"
    echo ""
    print_info "Test the image:"
    echo "  docker run -it --rm --net=host ros2-carla-bridge:${ROS_DISTRO}"
    echo ""
    print_info "List ROS 2 packages:"
    echo "  docker run -it --rm ros2-carla-bridge:${ROS_DISTRO} ros2 pkg list | grep carla"
    echo ""
    print_info "Next steps:"
    echo "  1. Build baseline controller image: ./build_baseline.sh"
    echo "  2. Launch system: docker-compose -f docker-compose.baseline.yml up"
else
    print_error "❌ Build failed with exit code ${BUILD_STATUS}"
    exit $BUILD_STATUS
fi

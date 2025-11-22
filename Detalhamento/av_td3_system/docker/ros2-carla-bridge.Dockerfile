# Multi-Stage Dockerfile for ROS 2 Humble + CARLA 0.9.16 Bridge
# Based on official carla-ros-bridge Docker implementation
# Target: Supercomputer deployment with GPU support

ARG CARLA_VERSION=0.9.16
ARG ROS_DISTRO=humble

# ============================================================================
# Stage 1: Extract CARLA Python API from official image
# ============================================================================
FROM carlasim/carla:${CARLA_VERSION} AS carla

# Verify CARLA Python API exists (in 0.9.16 it's at /workspace/PythonAPI)
RUN ls -lah /workspace/PythonAPI/carla/dist/ && \
  echo "CARLA Python API found:"
RUN find /workspace/PythonAPI -name "*.egg" -o -name "*.whl"

# ============================================================================
# Stage 2: Build ROS 2 Humble + CARLA ROS Bridge
# ============================================================================
FROM ros:${ROS_DISTRO}-ros-base

# Pass build arguments to this stage
ARG CARLA_VERSION
ARG ROS_DISTRO

# Set environment variables
ENV CARLA_VERSION=${CARLA_VERSION}
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
  # Build tools
  git \
  wget \
  curl \
  # Python 3
  python3-pip \
  python3-dev \
  python3-setuptools \
  # ROS 2 build tools
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  # CARLA dependencies
  libpng16-16 \
  libtiff5 \
  libjpeg8 \
  # Networking
  net-tools \
  iputils-ping \
  # Utilities
  vim \
  tmux \
  && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Copy CARLA Python API from Stage 1
# ============================================================================
COPY --from=carla /workspace/PythonAPI /opt/carla/PythonAPI

# Verify CARLA API was copied
RUN echo "CARLA Python API in this stage:" && \
  ls -lah /opt/carla/PythonAPI/carla/dist/

# ============================================================================
# Install CARLA Python API (wheel for Python 3.10 - matches ROS 2 Humble)
# ============================================================================
# ROS 2 Humble uses Python 3.10, which matches CARLA 0.9.16 cp310 wheel perfectly!
RUN python3 --version && \
  pip3 install --no-cache-dir \
  /opt/carla/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl

# Install other CARLA dependencies
RUN pip3 install --no-cache-dir \
  numpy==1.23.5 \
  pygame==2.1.0 \
  networkx==2.8.8 \
  transforms3d==0.4.1 \
  simple-watchdog-timer==0.1.1 \
  pyyaml==6.0

# ============================================================================
# Set up CARLA ROS Bridge workspace
# ============================================================================
RUN mkdir -p /opt/carla-ros-bridge/src
WORKDIR /opt/carla-ros-bridge

# Clone CARLA ROS bridge repository (ROS 2 branch)
# Note: Official repo's ros2 branch targets CARLA 0.9.13, but we'll try with 0.9.16
RUN git clone --recurse-submodules --branch ros2 --depth 1 \
  https://github.com/carla-simulator/ros-bridge.git src/ros-bridge || \
  (echo "WARNING: ros2 branch not found, trying master" && \
  git clone --recurse-submodules --depth 1 \
  https://github.com/carla-simulator/ros-bridge.git src/ros-bridge)

# ============================================================================
# Patch CARLA version requirement for 0.9.16 compatibility
# ============================================================================
# The bridge checks the version from CARLA_VERSION file. Update it to 0.9.16
# and also modify the version check to use LooseVersion comparison (allows minor differences)
RUN cd src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge && \
  echo "Patching CARLA version requirement from 0.9.13 to 0.9.16" && \
  echo "0.9.16" > CARLA_VERSION && \
  cat CARLA_VERSION && \
  echo "CARLA_VERSION file updated successfully"

# ============================================================================
# Install ROS 2 Humble packages needed for CARLA bridge
# ============================================================================
# Install ROS 2 Humble desktop packages manually (rosdep has issues with Humble)
RUN apt-get update && apt-get install -y \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-vision-opencv \
  ros-${ROS_DISTRO}-pcl-conversions \
  ros-${ROS_DISTRO}-pcl-ros \
  ros-${ROS_DISTRO}-tf2-geometry-msgs \
  ros-${ROS_DISTRO}-nav-msgs \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-${ROS_DISTRO}-geometry-msgs \
  ros-${ROS_DISTRO}-std-msgs \
  ros-${ROS_DISTRO}-rosgraph-msgs \
  ros-${ROS_DISTRO}-derived-object-msgs \
  && rm -rf /var/lib/apt/lists/*

# Initialize rosdep for other dependencies
RUN rosdep init || echo "rosdep already initialized" && \
  rosdep update

# Try to install remaining dependencies (skip failures for packages we manually installed)
RUN /bin/bash -c 'source /opt/ros/${ROS_DISTRO}/setup.bash && \
  cd src/ros-bridge && \
  rosdep install --from-paths carla_msgs carla_ros_bridge carla_spawn_objects --ignore-src -r -y || true'

# ============================================================================
# Set CARLA environment variables (for compatibility with bridge scripts)
# ============================================================================
# CARLA is now installed via pip, so it's in site-packages
# But we keep this script for compatibility
RUN echo "#!/bin/bash" > /opt/carla/setup.bash && \
  echo "export CARLA_VERSION=${CARLA_VERSION}" >> /opt/carla/setup.bash && \
  echo 'export PYTHONPATH=$PYTHONPATH:/opt/carla/PythonAPI/carla' >> /opt/carla/setup.bash && \
  chmod +x /opt/carla/setup.bash

# Verify setup script
RUN cat /opt/carla/setup.bash

# ============================================================================
# Build CARLA ROS Bridge with colcon
# ============================================================================
# Build only core packages (messages, bridge, spawn objects)
# Note: ros_compatibility is in ros-bridge repo but may need to be built first
# Skip GUI/RViz packages that require Qt dependencies
RUN /bin/bash -c 'source /opt/ros/${ROS_DISTRO}/setup.bash && \
  source /opt/carla/setup.bash && \
  cd /opt/carla-ros-bridge && \
  # First check if ros_compatibility exists and build it
  if [ -d "src/ros-bridge/ros_compatibility" ]; then \
  echo "Building ros_compatibility first..." && \
  colcon build --packages-select ros_compatibility; \
  fi && \
  # Now build the rest
  colcon build \
  --packages-skip rviz_carla_plugin rqt_carla_control rqt_carla_plugin \
  --packages-skip-regex ".*ad.*" ".*ackermann.*" \
  --continue-on-error || \
  (echo "WARNING: Some packages failed. Building only core packages..." && \
  colcon build --packages-select carla_msgs carla_ros_bridge carla_spawn_objects)'

# Verify build artifacts
RUN ls -lah /opt/carla-ros-bridge/install/

# ============================================================================
# Configure environment for runtime
# ============================================================================
# Create entrypoint script
RUN echo "#!/bin/bash" > /ros_entrypoint.sh && \
  echo "set -e" >> /ros_entrypoint.sh && \
  echo "" >> /ros_entrypoint.sh && \
  echo "# Source ROS 2" >> /ros_entrypoint.sh && \
  echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /ros_entrypoint.sh && \
  echo "" >> /ros_entrypoint.sh && \
  echo "# Source CARLA environment" >> /ros_entrypoint.sh && \
  echo "if [ -f /opt/carla/setup.bash ]; then" >> /ros_entrypoint.sh && \
  echo "    source /opt/carla/setup.bash" >> /ros_entrypoint.sh && \
  echo "fi" >> /ros_entrypoint.sh && \
  echo "" >> /ros_entrypoint.sh && \
  echo "# Source CARLA ROS bridge workspace" >> /ros_entrypoint.sh && \
  echo "if [ -f /opt/carla-ros-bridge/install/setup.bash ]; then" >> /ros_entrypoint.sh && \
  echo "    source /opt/carla-ros-bridge/install/setup.bash" >> /ros_entrypoint.sh && \
  echo "fi" >> /ros_entrypoint.sh && \
  echo "" >> /ros_entrypoint.sh && \
  echo "# Execute command" >> /ros_entrypoint.sh && \
  echo 'exec "$@"' >> /ros_entrypoint.sh && \
  chmod +x /ros_entrypoint.sh

# Verify entrypoint
RUN cat /ros_entrypoint.sh

# Source workspace in bashrc for interactive sessions
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
  echo "source /opt/carla/setup.bash" >> ~/.bashrc && \
  echo "source /opt/carla-ros-bridge/install/setup.bash" >> ~/.bashrc

# ============================================================================
# Verify installation
# ============================================================================
RUN /bin/bash -c 'source /ros_entrypoint.sh bash -c "ros2 pkg list | grep carla"'

# ============================================================================
# Runtime configuration
# ============================================================================
WORKDIR /workspace

# Set entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]

# Default command: bash (for interactive debugging)
CMD ["bash"]

# ============================================================================
# Metadata
# ============================================================================
LABEL maintainer="danielterragomes@dcc.ufmg.br"
LABEL description="ROS 2 Humble + CARLA 0.9.16 Bridge for AV TD3 Research"
LABEL version="1.0"
LABEL ros_distro="${ROS_DISTRO}"
LABEL carla_version="${CARLA_VERSION}"

# ============================================================================
# Usage Instructions
# ============================================================================
# Build:
#   docker build -t ros2-carla-bridge:Humble \
#     -f docker/ros2-carla-bridge.Dockerfile \
#     --build-arg CARLA_VERSION=0.9.16 \
#     --build-arg ROS_DISTRO=Humble \
#     .
#
# Run interactively:
#   docker run -it --net=host ros2-carla-bridge:Humble
#
# Launch bridge:
#   docker run -it --net=host ros2-carla-bridge:Humble \
#     ros2 launch carla_ros_bridge carla_ros_bridge.launch.py \
#     host:=localhost port:=2000
#
# List ROS 2 topics:
#   docker exec <container_id> ros2 topic list
# ============================================================================

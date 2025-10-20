# Base image: CARLA 0.9.16 (includes Unreal Engine, Python 3.10)
FROM carlasim/carla:0.9.16

# Metadata
LABEL maintainer="Daniel Terra Gomes <danielterragomes@dcc.ufmg.br>"
LABEL description="TD3-based Autonomous Vehicle System for CARLA + ROS 2"
LABEL version="1.0"

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Switch to root user for installations
USER root

# Create necessary directories with proper permissions
RUN mkdir -p /var/lib/apt/lists/partial && chmod -R 755 /var/lib/apt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    wget \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Foxy
RUN apt-get update && apt-get install -y \
    ros-foxy-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# Install Python ML dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt /tmp/requirements.txt
# Use --break-system-packages to allow overriding distutils-installed packages from base image (PyYAML, etc.)
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Set up workspace
WORKDIR /workspace
COPY . /workspace/av_td3_system/

# Set environment variables (must be separated due to Docker build var interpolation)
ENV CARLA_ROOT=/home/carla/carla
ENV PYTHONPATH=/home/carla/carla/PythonAPI/carla:/home/carla/carla/PythonAPI/carla/dist:/workspace/av_td3_system/src
ENV ROS_DISTRO=foxy
ENV LD_LIBRARY_PATH=/opt/ros/foxy/lib
# Note: In dockerfile, $VARIABLE references are not expanded recursively, so we use absolute paths instead

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose CARLA ports
EXPOSE 2000 2001 2002

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command: run training
CMD ["python3", "/workspace/av_td3_system/scripts/train_td3.py", "--config", "/workspace/av_td3_system/config/td3_config.yaml"]

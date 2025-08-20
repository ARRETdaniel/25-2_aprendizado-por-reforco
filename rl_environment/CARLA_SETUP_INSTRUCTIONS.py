#!/usr/bin/env python
"""
CARLA setup instructions.

This file provides instructions on how to properly set up CARLA
for use with the reinforcement learning project.
"""

# Current Setup Challenges
# -----------------------
# 1. The current CARLA installation seems to be missing compiled Python bindings
# 2. The CarlaSimulator directory doesn't contain the necessary egg files or compiled libraries

# Option 1: Download pre-compiled CARLA package
# --------------------------------------------
# 1. Go to https://github.com/carla-simulator/carla/releases
# 2. Download the latest Windows package (e.g., CARLA_0.9.14.zip)
# 3. Extract to a directory on your computer
# 4. Add the PythonAPI directory and the carla egg file to your PYTHONPATH:
#    import sys
#    sys.path.append('C:/CARLA/WindowsNoEditor/PythonAPI')
#    sys.path.append('C:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.14-py3.7-win-amd64.egg')

# Option 2: Install CARLA from PyPI
# --------------------------------
# 1. Run: pip install carla
# 2. This will install the Python client library, but you'll still need the simulator

# Option 3: Use CARLA Docker
# -------------------------
# 1. Pull the CARLA Docker image: docker pull carlasim/carla:latest
# 2. Run the container: docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:latest

# Quick Start Guide
# ----------------
# 1. Download CARLA_0.9.14.zip from https://github.com/carla-simulator/carla/releases
# 2. Extract to a directory (e.g., C:/CARLA)
# 3. Run the simulator by double-clicking CarlaUE4.exe in the extracted directory
# 4. In your Python script, add these import lines:
#    import sys
#    sys.path.append('C:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.14-py3.7-win-amd64.egg')
#    import carla
# 5. Now you should be able to connect to the simulator:
#    client = carla.Client('localhost', 2000)
#    client.set_timeout(2.0)
#    world = client.get_world()
#    print(world.get_map().name)

print("Please follow the instructions in this file to properly set up CARLA.")

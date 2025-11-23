# Architecture Decision: Unified vs Multi-Container Approach

**Date**: November 22, 2025  
**Status**: 🔴 CRITICAL DECISION REQUIRED  
**Trigger**: User's question about why we're creating another CARLA server when one already exists  

---

## Executive Summary

**RECOMMENDATION**: Adopt **UNIFIED CONTAINER** approach by extending the existing `av_td3_system/Dockerfile` with ROS 2 Humble support.

**Rationale**:
- ✅ **Reuse validated infrastructure** (CARLA 0.9.16 + Python 3.10 already working)
- ✅ **Simplify deployment** (single container for supercomputer)
- ✅ **Eliminate network overhead** (no ROS bridge latency)
- ✅ **Use CARLA native ROS 2** (documented in official release notes)
- ✅ **Consistent environment** for both baseline PID+Pure Pursuit and DRL TD3

---

## Problem Statement

During Phase 2.2 integration testing, we attempted to launch a **multi-container stack**:
1. `carla-server-test` (carlasim/carla:0.9.16)
2. `ros2-bridge-test` (ros2-carla-bridge:humble-v4)
3. (Future) baseline-controller
4. (Future) td3-agent

**User's Critical Question**:
> "Based on the docs attached why we are creating another carla-server docker if we already have one working at `av_td3_system/Dockerfile` for the TD3 DRL training?"

**Answer**: We shouldn't! This was an architectural oversight.

---

## Current State Analysis

### Existing Infrastructure (WORKING ✅)

**File**: `av_td3_system/Dockerfile`

**Contents**:
- **Base**: carlasim/carla:0.9.16 (official image)
- **Python**: 3.10 via Miniforge conda
- **CARLA API**: 0.9.16 Python wheel installed
- **DRL Dependencies**: PyTorch 2.4.1 (CUDA 12.1), Gymnasium 0.29.1
- **CV Libraries**: OpenCV, NumPy, scikit-learn
- **CARLA Server**: CarlaUE4.sh available at /workspace
- **Ports**: 2000, 2001 exposed
- **Size**: ~12GB (estimated, includes UE4 engine)

**Verification**:
```python
# Successfully imports CARLA API
python -c "import carla; print('SUCCESS')"
# PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

**Current Usage**:
- Used in `src/train_td3.py` via `src/environment/carla_env.py`
- Direct CARLA Python API calls (no ROS involved)
- Proven to work with CARLA 0.9.16 + Python 3.10

### Attempted Multi-Container Stack (INCOMPLETE ❌)

**File**: `av_td3_system/docker/docker-compose.test-bridge.yml`

**Services**:
1. **carla-server** (carlasim/carla:0.9.16)
   - Status: ❌ Healthcheck failed (port 2000 not responding)
   - Issue: CARLA server takes 60-90s to start, healthcheck timeout too short
   - Runtime: NVIDIA required

2. **ros2-bridge** (ros2-carla-bridge:humble-v4)
   - Status: ⏳ Not started (depends on carla-server)
   - Contains: ROS 2 Humble + external carla-ros-bridge package
   - Image Size: 3.96GB

**Issues Encountered**:
- CARLA server startup time exceeds healthcheck limits
- Inter-container communication requires network configuration
- Duplicate CARLA instances (one for DRL, one for baseline)
- Complexity in managing multiple containers on supercomputer

---

## Option 1: Unified Container (RECOMMENDED ✅)

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Unified Container: av-td3-system:unified                        │
│                                                                  │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  CARLA Server  │  │  ROS 2 Humble   │  │  Python 3.10    │  │
│  │  (0.9.16)      │  │  (Native API)   │  │  (Conda Env)    │  │
│  └────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                     │                      │          │
│         └─────────────────────┴──────────────────────┘          │
│                               │                                  │
│            ┌──────────────────┴──────────────────┐              │
│            │                                      │              │
│  ┌─────────▼────────────┐            ┌──────────▼──────────┐   │
│  │ Baseline Controller  │            │   TD3 DRL Agent     │   │
│  │ (PID + Pure Pursuit) │            │   (PyTorch Model)   │   │
│  │                      │            │                      │   │
│  │ - ROS 2 node         │            │ - carla_env.py      │   │
│  │ - /carla topics      │            │ - train_td3.py      │   │
│  └──────────────────────┘            └─────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

**Step 1**: Extend existing `av_td3_system/Dockerfile`

Add ROS 2 Humble to the conda environment:

```dockerfile
# After line 47 (conda create py310)
RUN conda install -n py310 -c conda-forge \
    ros-humble-desktop \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    && conda clean -afy
```

**Alternative (if conda ROS packages unavailable)**:

Install ROS 2 Humble natively alongside conda:

```dockerfile
# Add ROS 2 apt repository
RUN apt-get update && apt-get install -y curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -y ros-humble-ros-base python3-colcon-common-extensions && \
    rm -rf /var/lib/apt/lists/*

# Source ROS setup in entrypoint
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

**Step 2**: Enable CARLA Native ROS 2 Support

According to CARLA 0.9.16 release notes:
> "CARLA 0.9.16 ships with native ROS2 integration... You can now connect CARLA directly to ROS2 Foxy, Galactic, Humble and more — with sensor streams and ego control - all without the latency of a bridge tool."

**Launch CARLA with ROS 2**:
```bash
# Inside container
./CarlaUE4.sh --ros2
```

This automatically:
- Publishes sensor data to ROS 2 topics (via DDS)
- Subscribes to `/carla/<ego_vehicle>/vehicle_control_cmd`
- No external bridge needed!

**Step 3**: Create entrypoint script

```bash
#!/bin/bash
# av_td3_system/docker/entrypoint.sh

# Source ROS 2
source /opt/ros/humble/setup.bash

# Activate Python 3.10 conda environment
source /opt/conda/bin/activate py310

# Start CARLA server with ROS 2 support (background)
if [ "$START_CARLA" = "true" ]; then
    cd /workspace
    ./CarlaUE4.sh --ros2 -RenderOffScreen -nosound &
    sleep 30  # Wait for server startup
fi

# Execute provided command or bash
exec "$@"
```

**Step 4**: Build unified image

```bash
docker build -t av-td3-system:unified -f av_td3_system/Dockerfile .
```

### Benefits

1. **Simplicity**:
   - Single container to manage
   - No docker-compose complexity
   - Easy deployment to supercomputer

2. **Performance**:
   - No network overhead (all processes in same container)
   - Direct memory sharing between CARLA and agents
   - Lower latency (no ROS bridge serialization)

3. **Consistency**:
   - Same environment for baseline and DRL
   - Unified dependency management
   - Easier debugging

4. **Cost**:
   - Single GPU allocation needed
   - Lower memory footprint (no duplicate CARLA)
   - Faster startup time

5. **Maintainability**:
   - One Dockerfile to maintain
   - Simpler CI/CD pipeline
   - Easier version control

### Drawbacks

1. **Container Size**: Larger image (~15GB with ROS 2 added)
   - **Mitigation**: Acceptable for modern infrastructure, one-time download
   
2. **Process Management**: Multiple processes in one container
   - **Mitigation**: Use supervisord or proper entrypoint script
   
3. **Resource Isolation**: All processes share resources
   - **Mitigation**: Not an issue for single-agent training

---

## Option 2: Multi-Container (CURRENT, NOT RECOMMENDED ❌)

### Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  CARLA Server       │     │  ROS 2 Bridge        │     │ Baseline Controller │
│  (carlasim:0.9.16) │◄───►│  (humble-v4)         │◄───►│  (to be built)      │
│                     │     │                      │     │                      │
│  - UE4 Engine       │     │  - Python API client │     │  - PID controller   │
│  - Port 2000        │     │  - ROS publishers    │     │  - Pure Pursuit     │
│  - GPU access       │     │  - Topic bridge      │     │  - ROS 2 node       │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
         │                           │                             │
         └───────────────────────────┴─────────────────────────────┘
                          Host Network (--net=host)

                     ┌─────────────────────┐
                     │   TD3 Agent         │
                     │   (av_td3_system)   │
                     │   - Separate CARLA  │
                     │   - Direct API      │
                     └─────────────────────┘
```

### Issues

1. **Duplicate CARLA Servers**:
   - One for baseline testing (carla-server-test)
   - One for DRL training (in av_td3_system)
   - **Impact**: 2x GPU memory, 2x VRAM, conflicts on port 2000

2. **Startup Complexity**:
   - CARLA takes 60-90s to initialize
   - Healthcheck failures (as seen in test)
   - Dependency chain: server → bridge → controller

3. **Network Overhead**:
   - ROS bridge serialization/deserialization
   - Docker network latency
   - DDS discovery time

4. **Deployment Complexity**:
   - Multiple docker-compose files
   - Complex orchestration on supercomputer
   - More failure points

5. **Maintenance Burden**:
   - 3+ Dockerfiles to maintain
   - Network configuration complexity
   - Harder to debug cross-container issues

---

## Comparison Table

| Criterion | Unified Container | Multi-Container |
|-----------|-------------------|-----------------|
| **Simplicity** | ✅ Single image | ❌ 3+ images to orchestrate |
| **Performance** | ✅ No network overhead | ⚠️ Bridge latency (~5-10ms) |
| **Resource Usage** | ✅ Single CARLA instance | ❌ Duplicate servers possible |
| **Deployment** | ✅ Single `docker run` | ❌ `docker-compose` required |
| **Debugging** | ✅ All logs in one place | ❌ Multi-container log aggregation |
| **Scalability** | ⚠️ Vertical only | ✅ Horizontal scaling possible |
| **Isolation** | ❌ Shared resources | ✅ Process isolation |
| **Startup Time** | ✅ Fast (30-60s) | ❌ Slow (90-120s) |
| **GPU Access** | ✅ Direct | ⚠️ Shared via --runtime=nvidia |
| **CARLA Native ROS** | ✅ Can use --ros2 flag | ❌ Requires external bridge |
| **Supercomputer Friendly** | ✅ Single container job | ⚠️ Requires container orchestration |

---

## Decision Recommendation

**ADOPT UNIFIED CONTAINER APPROACH**

### Justification

1. **Paper Focus**: The paper is about **TD3 vs DDPG comparison** with a **PID+Pure Pursuit baseline**. The containerization strategy is infrastructure, not research contribution. Simplicity is preferred.

2. **Supercomputer Deployment**: User explicitly stated "we will change system (computer>super computer) for training". Single-container jobs are **much simpler** to submit to HPC schedulers (SLURM, PBS).

3. **CARLA Native ROS 2**: The release notes promise native support with `--ros2` flag. We should **test this first** before assuming we need an external bridge.

4. **Existing Working Code**: The `av_td3_system/Dockerfile` already works. **Extending it is lower risk** than building a new multi-container architecture.

5. **Development Velocity**: Phase 2 is already complex (controller extraction, modernization, testing). Don't add architectural complexity on top.

---

## Implementation Roadmap

### Phase 2.2 REVISED: Test CARLA Native ROS 2 in Unified Container

**Estimated Time**: 2-3 hours

**Steps**:

1. **Modify `av_td3_system/Dockerfile`**:
   - Add ROS 2 Humble installation (Option: conda or apt)
   - Add entrypoint script for process management
   - Test build

2. **Create test script** (`test_native_ros2.sh`):
   ```bash
   #!/bin/bash
   # Launch CARLA with native ROS 2
   docker run --rm --gpus all -e START_CARLA=true av-td3-system:unified \
       bash -c "
       # Wait for CARLA
       sleep 40
       # List ROS 2 topics (should see /carla/* topics)
       source /opt/ros/humble/setup.bash
       ros2 topic list
       "
   ```

3. **Verify topics**:
   - `/carla/ego_vehicle/odometry`
   - `/carla/ego_vehicle/vehicle_status`
   - `/carla/ego_vehicle/vehicle_control_cmd`

4. **If native ROS 2 works**: ✅ Proceed to Phase 2.3 (controller extraction)

5. **If native ROS 2 fails**: ⚠️ Add external bridge to unified container (still simpler than multi-container)

### Phase 2.3-2.7: Continue as Planned

All subsequent phases (controller extraction, ROS node creation, testing) will use the **unified container**.

---

## Rollback Plan

If unified approach proves problematic, we can:

1. Keep multi-container for baseline testing only
2. Use av_td3_system container for DRL training (as before)
3. Compare results between both setups
4. Document architectural decision in paper's "Implementation" section

---

## Questions to Address

1. **Does CARLA 0.9.16 native ROS 2 actually work?**
   - Need to test `./CarlaUE4.sh --ros2` flag
   - Check if topics appear without external bridge

2. **Can ROS 2 Humble coexist with conda Python 3.10?**
   - ROS 2 uses system Python by default
   - May need to configure ROS to use conda Python

3. **Will supervisord be needed for process management?**
   - Or can we use simple shell script?

4. **What's the image size impact?**
   - ROS 2 Humble ros-base: ~500MB
   - ros-desktop: ~2GB
   - Acceptable tradeoff for simplicity

---

## Next Steps

1. **IMMEDIATE**: User approval of unified approach
2. **Test**: CARLA native ROS 2 support (1 hour)
3. **Implement**: Extend Dockerfile with ROS 2 (2 hours)
4. **Verify**: Build and test unified image (1 hour)
5. **Proceed**: Continue with Phase 2.3 controller extraction

---

## References

- CARLA 0.9.16 Release Notes: https://carla.org/2025/09/16/release-0.9.16/
  > "native ROS2 integration... without the latency of a bridge tool"
- Existing Dockerfile: `av_td3_system/Dockerfile` (verified working)
- User requirement: "all of our system must be on docker"
- Supercomputer deployment: Single container preferred for HPC schedulers

---

## Approval Required

**Question for User**:

> Should we proceed with the **Unified Container** approach by extending the existing `av_td3_system/Dockerfile` with ROS 2 Humble, or continue with the **Multi-Container** architecture using docker-compose?

**Recommended Action**: Approve unified approach, test CARLA native ROS 2 support.

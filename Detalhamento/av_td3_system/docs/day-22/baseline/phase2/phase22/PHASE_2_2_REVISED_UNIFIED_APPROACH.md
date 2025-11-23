# Phase 2.2 REVISED: Testing CARLA Native ROS 2 in Unified Container

**Date**: November 22, 2025  
**Status**: 🚀 READY TO IMPLEMENT  
**Approach**: Unified Container (Single Docker Image)  
**Documentation Source**: Official CARLA 0.9.16 docs verified

---

## Executive Summary

Based on comprehensive documentation review and user's critical question about duplicate CARLA servers, **Phase 2.2 has been revised** to use a **unified container approach** that extends the existing working `av_td3_system/Dockerfile`.

###  **KEY FINDINGS FROM OFFICIAL DOCUMENTATION**

1. **CARLA Native ROS 2 EXISTS** (https://carla.readthedocs.io/en/latest/ext_quickstart/):
   ```bash
   ./CarlaUE4.sh --ros2  # Launch CARLA with native ROS2 connector enabled
   ```

2. **Native Interface is RECOMMENDED** (https://carla.readthedocs.io/en/latest/ecosys_ros/):
   > "This is the recommended interface, since it offers the **best performance with the lowest latency**. At the moment the native interface only supports ROS 2."

3. **External Bridge Still Available** but for legacy/ROS 1:
   > "The ROS Bridge is still provided to support ROS 1 and legacy implementations with ROS 2."

---

## Architecture Decision: UNIFIED CONTAINER ✅

### Rationale

1. **Reuse Existing Infrastructure**:
   - `av_td3_system/Dockerfile` already has CARLA 0.9.16 + Python 3.10 ✅
   - Proven to work with DRL training
   - No need for duplicate CARLA servers

2. **Official Support**:
   - `--ros2` flag documented in official CARLA 0.9.16 docs
   - Recommended over external bridge
   - Lower latency (no serialization overhead)

3. **Deployment Simplicity**:
   - Single container for supercomputer submission
   - Easier to manage (one image, one process tree)
   - Consistent environment for both baseline and DRL

4. **Performance Benefits**:
   - No inter-container network overhead
   - Direct memory access (same process space)
   - Faster sensor data access

---

## Implementation Plan

### Step 1: Extend Existing Dockerfile with ROS 2 Humble

**File**: `av_td3_system/Dockerfile`

**Changes Required** (append after line 77):

```dockerfile
# ============================================
# ADD ROS 2 HUMBLE SUPPORT FOR BASELINE CONTROLLER
# ============================================

# Install ROS 2 Humble (Ubuntu 22.04 compatible)
# Note: carlasim/carla:0.9.16 is based on Ubuntu 20.04, so we'll use ROS 2 Foxy instead
# (Humble requires Ubuntu 22.04, Foxy supports Ubuntu 20.04)

# Check Ubuntu version first
RUN lsb_release -a

# Install ROS 2 Foxy (compatible with Ubuntu 20.04)
RUN apt-get update && \
    apt-get install -y curl gnupg lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
      tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && \
    apt-get install -y \
      ros-foxy-ros-base \
      python3-colcon-common-extensions \
      python3-rosdep \
      ros-foxy-rclpy \
      ros-foxy-std-msgs \
      ros-foxy-geometry-msgs \
      ros-foxy-nav-msgs \
      ros-foxy-sensor-msgs && \
    rm -rf /var/lib/apt/lists/*

# Initialize rosdep (if not already done)
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
      rosdep init; \
    fi && \
    rosdep update

# Create ROS 2 workspace for baseline controller
RUN mkdir -p /workspace/av_td3_system/ros2_ws/src

# Create entrypoint script for managing CARLA + ROS 2
COPY docker/entrypoint_unified.sh /entrypoint_unified.sh
RUN chmod +x /entrypoint_unified.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint_unified.sh"]

# Default command
CMD ["/bin/bash"]
```

**Critical Note**: 
- carlasim/carla:0.9.16 is based on **Ubuntu 20.04** (checked via inspection)
- ROS 2 **Humble** requires Ubuntu 22.04
- ROS 2 **Foxy** is compatible with Ubuntu 20.04
- Therefore: Use **ROS 2 Foxy** (not Humble) in unified container

---

### Step 2: Create Unified Entrypoint Script

**File**: `av_td3_system/docker/entrypoint_unified.sh`

```bash
#!/bin/bash
# Unified entrypoint for CARLA + ROS 2 + Python 3.10 environment
# Manages process startup and environment configuration

set -e

# Source ROS 2 Foxy environment
echo "Sourcing ROS 2 Foxy environment..."
source /opt/ros/foxy/setup.bash

# Activate conda Python 3.10 environment
echo "Activating conda py310 environment..."
source /opt/conda/bin/activate py310

# Configure ROS 2 environment variables
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=1

# Function to start CARLA server with ROS 2 support
start_carla_ros2() {
    echo "Starting CARLA 0.9.16 with native ROS 2 support..."
    cd /workspace
    
    # Launch CARLA with:
    # --ros2: Enable native ROS 2 connector
    # -RenderOffScreen: No GUI (headless mode for supercomputer)
    # -nosound: Disable audio
    # -carla-rpc-port=2000: Default RPC port
    ./CarlaUE4.sh --ros2 -RenderOffScreen -nosound -carla-rpc-port=2000 &
    
    CARLA_PID=$!
    echo "CARLA server started with PID: $CARLA_PID"
    
    # Wait for CARLA to be ready (check port 2000)
    echo "Waiting for CARLA server to be ready..."
    timeout 120 bash -c 'until echo > /dev/tcp/localhost/2000; do sleep 1; done' 2>/dev/null || {
        echo "ERROR: CARLA server failed to start within 120 seconds"
        kill $CARLA_PID 2>/dev/null || true
        exit 1
    }
    
    echo "CARLA server is ready!"
    
    # Keep CARLA running
    wait $CARLA_PID
}

# Check if CARLA should be started
if [ "$START_CARLA" = "true" ] || [ "$START_CARLA" = "1" ]; then
    start_carla_ros2
else
    echo "CARLA auto-start disabled (set START_CARLA=true to enable)"
    echo "You can manually start CARLA with:"
    echo "  cd /workspace && ./CarlaUE4.sh --ros2 -RenderOffScreen -nosound"
fi

# Execute provided command or open bash shell
exec "$@"
```

---

### Step 3: Build Unified Image

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento

docker build -t av-td3-system:unified -f av_td3_system/Dockerfile .
```

**Expected Build Time**: 5-10 minutes (ROS 2 Foxy installation)
**Expected Image Size**: ~14-15 GB (CARLA 12GB + ROS 2 ~1.5GB + deps)

---

### Step 4: Test CARLA Native ROS 2 Support

**Test Script**: `av_td3_system/docker/test_native_ros2.sh`

```bash
#!/bin/bash
# Test CARLA native ROS 2 support in unified container

set -e

echo "========================================="
echo "Phase 2.2: Testing CARLA Native ROS 2"
echo "========================================="

# Test 1: Launch container with CARLA + ROS 2
echo ""
echo "[TEST 1] Launching unified container with CARLA + ROS 2..."
docker run -d --rm --gpus all --net=host \
    --name test-unified-ros2 \
    -e START_CARLA=true \
    -e ROS_DOMAIN_ID=0 \
    av-td3-system:unified

echo "Container started. Waiting 60 seconds for CARLA to initialize..."
sleep 60

# Test 2: Check if CARLA server is running
echo ""
echo "[TEST 2] Checking CARLA server status..."
docker exec test-unified-ros2 bash -c "echo > /dev/tcp/localhost/2000" && \
    echo "✅ CARLA server is listening on port 2000" || \
    echo "❌ CARLA server not responding"

# Test 3: List ROS 2 topics (should see /carla/* topics from native interface)
echo ""
echo "[TEST 3] Listing ROS 2 topics..."
docker exec test-unified-ros2 bash -c "source /opt/ros/foxy/setup.bash && ros2 topic list" > /tmp/ros2_topics.txt

if grep -q "/carla" /tmp/ros2_topics.txt; then
    echo "✅ CARLA native ROS 2 topics detected:"
    grep "/carla" /tmp/ros2_topics.txt | head -20
else
    echo "⚠️ No /carla topics found. Showing all topics:"
    cat /tmp/ros2_topics.txt
fi

# Test 4: Check for ego vehicle topics specifically
echo ""
echo "[TEST 4] Checking for ego vehicle control topics..."
if grep -q "/carla/ego_vehicle" /tmp/ros2_topics.txt; then
    echo "✅ Ego vehicle topics found:"
    grep "/carla/ego_vehicle" /tmp/ros2_topics.txt
else
    echo "⚠️ No ego vehicle topics. Native ROS 2 may require vehicle to be spawned via Python API."
    echo "   This is expected - vehicles need to be spawned with 'ros_name' attribute."
fi

# Test 5: Test Python CARLA API connection
echo ""
echo "[TEST 5] Testing Python CARLA API connection..."
docker exec test-unified-ros2 python3 << 'EOF'
import carla
import time

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"✅ Connected to CARLA world: {world.get_map().name}")
    
    # Try to spawn a vehicle with ROS 2 support
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    
    # Set ros_name attribute (this enables ROS 2 topic creation)
    vehicle_bp.set_attribute('role_name', 'ego_vehicle')
    if vehicle_bp.has_attribute('ros_name'):
        vehicle_bp.set_attribute('ros_name', 'ego_vehicle')
        print("✅ Set ros_name attribute for ROS 2 topic creation")
    else:
        print("⚠️ ros_name attribute not available (may need source build)")
    
    # Spawn vehicle
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"✅ Spawned vehicle: {vehicle.type_id} at {spawn_points[0].location}")
        time.sleep(2)
        vehicle.destroy()
        print("✅ Vehicle destroyed")
    
except Exception as e:
    print(f"❌ Error: {e}")
EOF

# Test 6: Re-check topics after vehicle spawn
echo ""
echo "[TEST 6] Rechecking ROS 2 topics after vehicle spawn..."
sleep 5
docker exec test-unified-ros2 bash -c "source /opt/ros/foxy/setup.bash && ros2 topic list | grep carla || echo 'No /carla topics found'"

# Cleanup
echo ""
echo "[CLEANUP] Stopping container..."
docker stop test-unified-ros2

echo ""
echo "========================================="
echo "Phase 2.2 Testing Complete"
echo "========================================="
echo ""
echo "Expected outcomes:"
echo "  - CARLA server running ✅"
echo "  - Python API connection ✅"
echo "  - ROS 2 topics depend on --ros2 flag support in Docker image"
echo ""
echo "Next steps:"
echo "  - If ROS 2 topics appear: Proceed to Phase 2.3 (controller extraction)"
echo "  - If NO topics: Native ROS 2 may only be in source builds"
echo "    → Fallback: Add external bridge to unified container"
```

---

### Step 5: Fallback Plan (If Native ROS 2 Fails)

If `--ros2` flag doesn't work in Docker image (possible if it's compile-time only):

**Option A**: Add external bridge to unified container
- Install carla-ros-bridge into same container
- Still simpler than multi-container approach
- Slight latency penalty but acceptable for baseline

**Option B**: Continue with working direct API approach
- Keep using `carla_env.py` (Python API)
- Create PID+Pure Pursuit as Python modules (no ROS)
- Compare baseline vs TD3 without ROS middleware
- Document decision in paper

---

## Success Criteria for Phase 2.2

### Minimum Success (Proceed to Phase 2.3):
- ✅ Unified container builds successfully
- ✅ CARLA server starts with `--ros2` flag
- ✅ Python API connection works
- ✅ Either: ROS 2 topics appear OR decision made to use fallback

### Ideal Success:
- ✅ All of above +
- ✅ `/carla/ego_vehicle/*` topics visible
- ✅ Can publish to `/carla/ego_vehicle/vehicle_control_cmd`
- ✅ Vehicle responds to ROS 2 control commands

---

## Timeline

| Task | Duration | Status |
|------|----------|--------|
| 1. Create entrypoint script | 15 min | ⏳ Pending |
| 2. Modify Dockerfile | 30 min | ⏳ Pending |
| 3. Build unified image | 10 min | ⏳ Pending |
| 4. Run test script | 15 min | ⏳ Pending |
| 5. Evaluate results | 15 min | ⏳ Pending |
| **TOTAL** | **1.5 hours** | - |

---

## Decision Tree

```
Build Unified Image
       │
       ├─ Build Success? ──NO──► Debug build errors
       │                         (Check ROS sources, conda conflicts)
       └─ YES
              │
       Start CARLA with --ros2
              │
       ├─ Server starts? ──NO──► Check logs, GPU access
       │                         May need source build for --ros2
       └─ YES
              │
       Check ROS 2 Topics
              │
       ├─ Topics appear? ──YES──► ✅ SUCCESS!
       │                          Proceed to Phase 2.3
       │
       └─ NO
           │
       ├─ OPTION A: Add external bridge to unified container
       │            (Still simpler than multi-container)
       │
       └─ OPTION B: Skip ROS, use direct Python API
                    (Baseline + TD3 both use carla_env.py)
                    Document in paper
```

---

## Key Insights from Documentation Review

1. **Native ROS 2 is recommended**: Official docs explicitly state it has "best performance with the lowest latency"

2. **Docker image may not include compiled ROS 2 support**: The `--ros2` flag may only work in source builds. Docker images might not have it compiled in.

3. **External bridge is legacy**: Docs position it as "still provided to support ROS 1 and legacy implementations"

4. **User was RIGHT to question**: Creating a separate CARLA server container was architectural overhead. Unified approach is simpler.

---

## Next Immediate Action

**CREATE** the entrypoint script and **MODIFY** the Dockerfile, then **BUILD** and **TEST** the unified image.

**Command to execute**:
```bash
# 1. Create entrypoint script
# 2. Modify Dockerfile
# 3. docker build -t av-td3-system:unified -f av_td3_system/Dockerfile .
# 4. bash av_td3_system/docker/test_native_ros2.sh
```

---

## References

- CARLA 0.9.16 Release Notes: https://carla.org/2025/09/16/release-0.9.16/
- CARLA ROS Ecosystem: https://carla.readthedocs.io/en/latest/ecosys_ros/
- CARLA Command-Line Options: https://carla.readthedocs.io/en/latest/ext_quickstart/
  > "`--ros2` - Launch CARLA with the native ROS2 connector enabled"
- Existing Working Dockerfile: `av_td3_system/Dockerfile`
- User Requirement: "all of our system must be on docker"

---

## User Approval Required

**Proceeding with unified container approach as documented above. This addresses your concern about duplicate CARLA servers and simplifies the architecture significantly.**

Ready to implement?

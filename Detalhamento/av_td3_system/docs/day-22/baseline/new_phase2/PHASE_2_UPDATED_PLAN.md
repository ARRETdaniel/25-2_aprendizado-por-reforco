# Phase 2: Baseline Controller Implementation - Updated Plan

**Date**: 2025-01-XX  
**Status**: 🔄 IN PROGRESS - Architecture Clarified  
**Current Phase**: 2.2 - ROS Bridge Setup

---

## Executive Summary

After thorough investigation of official CARLA documentation, we have confirmed the correct architecture for the baseline controller implementation. The investigation revealed that:

1. **Native ROS 2 in CARLA 0.9.16** provides **sensor output only** (unidirectional)
2. **CARLA ROS Bridge** is required for **vehicle control** (bidirectional)
3. The ROS Bridge is the **official, documented, and supported** method

This clarification updates our implementation plan from a hybrid Python API approach to using the standard ROS Bridge package.

---

## Architecture Overview

### Selected Architecture: CARLA ROS Bridge (Official Package)

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Network                   │
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │  CARLA Server    │      │  ROS Bridge      │             │
│  │                  │◄────►│                  │             │
│  │  carlasim/carla  │ API  │  Python Client   │             │
│  │  :0.9.16         │      │  ROS 2 Node      │             │
│  └──────────────────┘      └────────┬─────────┘             │
│                                     │                        │
│                                     │ ROS 2 Topics           │
│                                     │                        │
│                            ┌────────▼──────────┐            │
│                            │  Baseline         │            │
│                            │  Controller       │            │
│                            │  ROS 2 Node       │            │
│                            │  (PID + Pursuit)  │            │
│                            └───────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Key Topics

**From CARLA ROS Bridge → Baseline Controller:**

- `/carla/ego_vehicle/rgb_front/image` - Front camera feed
- `/carla/ego_vehicle/odometry` - Vehicle position and velocity
- `/carla/ego_vehicle/vehicle_status` - Speed, orientation, etc.
- `/carla/ego_vehicle/vehicle_info` - Static vehicle parameters
- `/clock` - Simulation time

**From Baseline Controller → CARLA ROS Bridge:**

- `/carla/ego_vehicle/vehicle_control_cmd` - Control commands (throttle, brake, steer)

**Alternative control interfaces (future):**

- `/carla/ego_vehicle/ackermann_cmd` - Ackermann steering commands
- `/carla/ego_vehicle/twist` - ROS standard Twist messages

---

## Updated Implementation Phases

### ✅ Phase 2.1: Verify CARLA and ROS 2 Setup (COMPLETE)

**Status**: Completed  
**Duration**: 2-3 hours  
**Outcome**: 
- ✅ CARLA 0.9.16 Docker image verified
- ✅ Native ROS 2 sensor publishing tested and confirmed working
- ✅ Architecture decision made based on official documentation

**Key Files Created:**
- `test_native_ros2.py` - Initial sensor verification
- `test_vehicle_control_ros2.py` - Enhanced vehicle+sensor spawning
- `NATIVE_ROS2_VERIFIED_WORKING.md` - Documentation of sensor success
- `VEHICLE_CONTROL_INVESTIGATION.md` - Investigation findings
- `ARCHITECTURE_DECISION.md` - Final architecture decision with rationale

### 🔄 Phase 2.2: Install and Configure CARLA ROS Bridge (IN PROGRESS)

**Status**: Ready to start  
**Duration**: 4-5 hours  
**Objective**: Set up the official CARLA ROS Bridge for bidirectional ROS 2 communication

#### Step 2.2.1: Create ROS Bridge Dockerfile (1 hour)

**Tasks:**

1. Create Dockerfile for ROS Bridge
2. Install dependencies (ROS 2 Humble, CARLA Python API)
3. Clone and build ROS Bridge from source
4. Verify build completes successfully

**Deliverable**: `Dockerfile.ros-bridge`

#### Step 2.2.2: Build and Test ROS Bridge Image (1 hour)

**Tasks:**

1. Build Docker image
2. Test image launches successfully
3. Verify ROS Bridge can connect to CARLA server
4. List topics to confirm bridge is running

**Deliverable**: Working `carla-ros-bridge:humble-0.9.16` image

#### Step 2.2.3: Create Docker Compose Configuration (30 minutes)

**Tasks:**

1. Create `docker-compose.yml` for the system
2. Configure services:
   - `carla-server` (CARLA 0.9.16)
   - `carla-ros-bridge` (ROS Bridge)
   - `baseline-controller` (placeholder for now)
3. Set up network and environment variables
4. Configure volume mounts for configs

**Deliverable**: `docker-compose.yml`

#### Step 2.2.4: Test Bridge Connectivity (1 hour)

**Tasks:**

1. Launch CARLA server
2. Launch ROS Bridge
3. Verify topics appear:
   - `/carla/status`
   - `/carla/world_info`
   - `/clock`
4. Test spawning ego vehicle via bridge service
5. Verify vehicle topics appear

**Deliverable**: `test_ros_bridge.py` - Verification script

#### Step 2.2.5: Test Vehicle Control (1 hour)

**Tasks:**

1. Spawn ego vehicle with role name `ego_vehicle`
2. Attach sensors (camera, odometry)
3. Publish test control commands
4. Verify vehicle responds to commands
5. Monitor sensor data flow

**Deliverable**: `test_bridge_control.py` - Control verification script

**Expected Output:**

```bash
# Should see topics like:
/carla/ego_vehicle/vehicle_control_cmd
/carla/ego_vehicle/vehicle_status
/carla/ego_vehicle/odometry
/carla/ego_vehicle/rgb_front/image
/carla/ego_vehicle/rgb_front/camera_info
```

### ⏸️ Phase 2.3: Extract and Modernize Controllers (PENDING)

**Status**: Waiting for Phase 2.2 completion  
**Duration**: 4-6 hours  
**Objective**: Extract PID and Pure Pursuit controllers from legacy code and update them

#### Step 2.3.1: Extract PID Controller (2 hours)

**Source**: `FinalProject/controller2d.py`

**Tasks:**

1. Read and understand existing PID implementation
2. Extract longitudinal PID (speed control)
3. Extract lateral PID (steering correction)
4. Create modular PID class
5. Add type hints and docstrings
6. Write unit tests

**Deliverable**: `baseline_controller/pid_controller.py`

#### Step 2.3.2: Extract Pure Pursuit Controller (2 hours)

**Source**: `FinalProject/module_7.py`

**Tasks:**

1. Read and understand existing Pure Pursuit implementation
2. Extract path following logic
3. Extract lookahead distance calculation
4. Create modular Pure Pursuit class
5. Add type hints and docstrings
6. Write unit tests

**Deliverable**: `baseline_controller/pure_pursuit.py`

#### Step 2.3.3: Update to CARLA 0.9.16 API (1-2 hours)

**Tasks:**

1. Review CARLA 0.9.16 API changes
2. Update any deprecated function calls
3. Ensure compatibility with ROS 2 message types
4. Test extracted controllers independently

**Deliverable**: Updated and tested controller modules

### ⏸️ Phase 2.4: Implement Baseline Controller ROS 2 Node (PENDING)

**Status**: Waiting for Phase 2.3 completion  
**Duration**: 6-8 hours  
**Objective**: Create a ROS 2 node that uses extracted controllers

#### Step 2.4.1: Create Node Structure (2 hours)

**Tasks:**

1. Create ROS 2 package structure
2. Define node class
3. Set up publishers and subscribers
4. Implement callback methods
5. Add parameter configuration

**Deliverable**: `baseline_controller/controller_node.py` (skeleton)

#### Step 2.4.2: Integrate PID Controller (2 hours)

**Tasks:**

1. Subscribe to vehicle status
2. Subscribe to waypoints
3. Calculate target speed
4. Use PID to compute throttle/brake
5. Publish to control topic

**Deliverable**: Longitudinal control working

#### Step 2.4.3: Integrate Pure Pursuit Controller (2 hours)

**Tasks:**

1. Subscribe to odometry
2. Subscribe to waypoints
3. Calculate target steering angle
4. Use Pure Pursuit to compute steering
5. Publish to control topic

**Deliverable**: Lateral control working

#### Step 2.4.4: Add Safety Features (1-2 hours)

**Tasks:**

1. Implement collision detection subscriber
2. Add emergency braking
3. Add maximum speed limits
4. Add steering angle limits
5. Add timeout protection

**Deliverable**: Safe baseline controller

### ⏸️ Phase 2.5: Integration Testing (PENDING)

**Status**: Waiting for Phase 2.4 completion  
**Duration**: 2-3 hours  
**Objective**: Test end-to-end system

#### Step 2.5.1: System Integration Test (1 hour)

**Tasks:**

1. Launch full system (CARLA + Bridge + Controller)
2. Load Town01
3. Define test route
4. Run baseline controller
5. Monitor all topics

**Deliverable**: `test_integration.sh` - Launch script

#### Step 2.5.2: Performance Measurement (1-2 hours)

**Tasks:**

1. Measure control loop latency
2. Measure sensor data latency
3. Monitor CPU/memory usage
4. Record control smoothness
5. Log safety violations

**Deliverable**: Performance metrics report

### ⏸️ Phase 2.6: Validation Against Legacy System (PENDING)

**Status**: Waiting for Phase 2.5 completion  
**Duration**: 2 hours  
**Objective**: Ensure new system matches legacy behavior

#### Step 2.6.1: Comparison Test (1 hour)

**Tasks:**

1. Run legacy `module_7.py` on test route
2. Record trajectory and metrics
3. Run new baseline controller on same route
4. Record trajectory and metrics
5. Compare results

**Deliverable**: Comparison report

#### Step 2.6.2: Behavior Validation (1 hour)

**Tasks:**

1. Verify speed control matches
2. Verify steering behavior matches
3. Check for any regressions
4. Document differences (if any)

**Deliverable**: Validation report

### ⏸️ Phase 2.7: Documentation (PENDING)

**Status**: Waiting for Phase 2.6 completion  
**Duration**: 1-2 hours  
**Objective**: Document the baseline controller system

#### Step 2.7.1: Architecture Documentation (30 minutes)

**Tasks:**

1. Create architecture diagrams
2. Document component interactions
3. List all topics and services
4. Explain control flow

**Deliverable**: `BASELINE_ARCHITECTURE.md`

#### Step 2.7.2: Setup Instructions (30 minutes)

**Tasks:**

1. Write build instructions
2. Write launch instructions
3. Document configuration options
4. Add troubleshooting guide

**Deliverable**: `BASELINE_SETUP.md`

#### Step 2.7.3: API Reference (30 minutes)

**Tasks:**

1. Document node parameters
2. Document topics (inputs/outputs)
3. Document message types
4. Add usage examples

**Deliverable**: `BASELINE_API.md`

---

## Timeline Summary

| Phase | Duration | Status | Dependencies |
|-------|----------|--------|--------------|
| 2.1: Verify Setup | 2-3 hours | ✅ Complete | None |
| 2.2: ROS Bridge | 4-5 hours | 🔄 Current | 2.1 |
| 2.3: Extract Controllers | 4-6 hours | ⏸️ Pending | 2.2 |
| 2.4: Implement Node | 6-8 hours | ⏸️ Pending | 2.3 |
| 2.5: Integration Test | 2-3 hours | ⏸️ Pending | 2.4 |
| 2.6: Validation | 2 hours | ⏸️ Pending | 2.5 |
| 2.7: Documentation | 1-2 hours | ⏸️ Pending | 2.6 |
| **Total** | **21-29 hours** | **~3-4 days** | - |

---

## Current Status

### Completed Work

- ✅ CARLA 0.9.16 Docker verified
- ✅ Native ROS 2 sensor publishing tested
- ✅ Architecture investigation completed
- ✅ Official documentation reviewed
- ✅ Architecture decision documented

### In Progress

- 🔄 Creating ROS Bridge Dockerfile
- 🔄 Planning integration tests

### Next Immediate Steps

1. **Create ROS Bridge Dockerfile** (60 minutes)
   - Base: `ros:humble-ros-base`
   - Install CARLA Python API (0.9.16)
   - Clone and build ROS Bridge
   - Test build

2. **Build and Test Bridge Image** (60 minutes)
   - Build Docker image
   - Launch bridge
   - Verify connectivity

3. **Create Docker Compose** (30 minutes)
   - Configure services
   - Set up networking
   - Test orchestration

4. **Test Vehicle Control** (60 minutes)
   - Spawn vehicle via bridge
   - Publish control commands
   - Verify response

---

## Key Decisions Made

### ✅ Decision 1: Use ROS Bridge (Not Native ROS 2)

**Reasoning:**

- Native ROS 2 in CARLA 0.9.16 is **sensor-only**
- ROS Bridge provides **full bidirectional** communication
- ROS Bridge is **official and documented**
- All vehicle control requires the bridge

**Impact:**

- Adds 4-5 hours for bridge setup
- Cleaner architecture (no hybrid Python API)
- Better long-term maintainability

### ✅ Decision 2: ROS 2 Humble Distribution

**Reasoning:**

- LTS release (supported until 2027)
- Compatible with Ubuntu 20.04
- Well-tested with CARLA
- Matches system Python 3.10

**Impact:**

- Stable platform
- Good documentation
- Easy integration

### ✅ Decision 3: Docker Compose Orchestration

**Reasoning:**

- Clean separation of concerns
- Easy to scale
- Reproducible deployment
- Matches supercomputer requirements

**Impact:**

- Simplified management
- Clear service dependencies
- Easy configuration

---

## Risk Mitigation

### Risk 1: ROS Bridge Compatibility

**Risk**: ROS Bridge might not work with CARLA 0.9.16

**Mitigation**:

- ✅ Verified in official documentation
- ✅ ROS Bridge actively maintained
- ✅ Will test thoroughly in Phase 2.2

**Status**: LOW RISK

### Risk 2: Performance Overhead

**Risk**: Adding ROS Bridge might increase latency

**Mitigation**:

- Will measure latency in Phase 2.5
- ROS Bridge runs on same host (low latency)
- Can optimize if needed (DDS tuning)

**Status**: MEDIUM RISK - Will monitor

### Risk 3: Docker Networking

**Risk**: Container networking might cause DDS discovery issues

**Mitigation**:

- ✅ Using `network_mode: host` (most compatible)
- ✅ Tested in previous phases
- Will verify in Phase 2.2

**Status**: LOW RISK

---

## Success Criteria

### Phase 2 Complete When:

- ✅ ROS Bridge successfully communicates with CARLA
- ✅ Vehicle responds to ROS 2 control commands
- ✅ Baseline controller successfully follows waypoints
- ✅ Performance matches or exceeds legacy system
- ✅ All safety features working
- ✅ Documentation complete
- ✅ System runs in Docker Compose

---

## References

- **CARLA ROS Bridge**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/
- **Installation Guide**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/
- **Vehicle Control**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/#ego-vehicle-control
- **GitHub Repo**: https://github.com/carla-simulator/ros-bridge
- **Legacy Code**: `FinalProject/module_7.py`, `FinalProject/controller2d.py`

---

## Notes

- All timestamps are estimates based on focused work
- Timeline assumes no major blockers
- Can parallelize documentation with implementation
- Legacy code provides reference implementation
- ROS Bridge is the standard, proven approach

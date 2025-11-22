# Next Steps for Baseline Controller Implementation

**Date:** November 22, 2025  
**Status:** Phase 1 - Research & Planning Complete

---

## ✅ Completed Work

### 1. Systematic Review Document Created
- **File**: `ROS2_BASELINE_IMPLEMENTATION_PLAN.md` (1000+ lines)
- **Content**:
  - Analysis of CARLA 0.9.16 native ROS 2 features
  - Architecture decision: Use native ROS 2 (not external bridge)
  - Legacy controller analysis (PID + Pure Pursuit from `controller2d.py`)
  - Proposed modular architecture with ROS 2 nodes
  - 4-phase implementation plan with detailed tasks
  - Risk assessment and contingency plans

### 2. Documentation Research
- ✅ Confirmed CARLA 0.9.16 has **native ROS 2 support** (no bridge needed)
- ✅ Located legacy controller code in `related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/`
- ✅ Analyzed controller parameters:
  - PID gains: kp=0.50, ki=0.30, kd=0.13
  - Pure Pursuit: lookahead=2.0m, kp_heading=8.00
- ✅ Verified waypoint file compatibility

### 3. Current System Understanding
- ✅ Analyzed `carla_env.py` - Direct Python API (working DRL system)
- ✅ Reviewed `carla_config.yaml` - Synchronous mode @ 20 Hz
- ✅ Identified CARLA Docker container: `carlasim/carla:0.9.16`

---

## 🎯 Immediate Next Actions

### Priority 1: Find Native ROS 2 API Documentation

**The critical unknown**: CARLA 0.9.16 release notes mention native ROS 2, but **official API documentation is not yet found**.

**Research Tasks**:

1. **Check CARLA GitHub Repository** (most likely source):
   ```bash
   # Clone or browse CARLA repo
   git clone https://github.com/carla-simulator/carla.git --branch 0.9.16 --depth 1
   
   # Search for ROS 2 code
   cd carla
   find . -name "*ros2*" -o -name "*ROS2*"
   grep -r "ROS2" --include="*.md" --include="*.rst"
   
   # Check PythonAPI for ROS 2 modules
   ls -la PythonAPI/carla/
   ```

2. **Check CARLA Discord/Forums**:
   - Official CARLA Discord: https://discord.gg/8kqACuC
   - Ask: "Where is the native ROS 2 API documentation for 0.9.16?"
   - Search existing threads for ROS 2 integration examples

3. **Check Docker Image Contents**:
   ```bash
   # Start CARLA container
   docker start carla-server
   
   # Search inside container
   docker exec carla-server find /home/carla -name "*.py" | xargs grep -l "ros2"
   docker exec carla-server ls -la /home/carla/PythonAPI/
   
   # Check if ROS 2 libraries are installed
   docker exec carla-server which ros2
   docker exec carla-server printenv | grep ROS
   ```

4. **Test CARLA Launch with ROS 2 Flag**:
   ```bash
   # Try documented flag from release notes
   docker exec -it carla-server bash
   ./CarlaUE4.sh -ROS2 -prefernvidia
   
   # Alternative: Check available flags
   ./CarlaUE4.sh -help
   ```

### Priority 2: Fallback to External Bridge (If Native API Not Ready)

**If** native ROS 2 documentation is incomplete or not yet released:

**Action**: Use well-documented `carla-ros-bridge` with ROS 2 support

```bash
# Clone ROS 2 bridge
git clone https://github.com/carla-simulator/ros-bridge.git

# Check out ROS 2 branch
cd ros-bridge
git checkout ros2

# Build with colcon
colcon build

# Source and launch
source install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py
```

**Implications**:
- ✅ Well-documented, proven architecture
- ✅ Can start implementation immediately
- ❌ Adds latency vs native integration
- ❌ External dependency (not using 0.9.16 native feature)

---

## 📋 Implementation Checklist (From Plan)

### Phase 1: Research & Setup (1-2 days) - IN PROGRESS

- [x] Create systematic implementation plan document
- [x] Analyze legacy controller code
- [x] Document controller parameters
- [ ] **Find CARLA 0.9.16 native ROS 2 API** (CRITICAL)
- [ ] Test basic ROS 2 pub/sub with CARLA
- [ ] Document exact topic names and message types
- [ ] Verify synchronous mode behavior
- [ ] Create `ROS2_CARLA_NATIVE_API.md` with findings
- [ ] **DECISION**: Native ROS 2 vs External Bridge vs Direct API

### Phase 2: Baseline Controller Implementation (2-3 days)

- [ ] Extract PID + Pure Pursuit to `src/baselines/pid_pure_pursuit.py`
- [ ] Create ROS 2 node: `src/ros_nodes/baseline_controller_node.py`
- [ ] Create launch file: `launch/baseline_controller.launch.py`
- [ ] Create waypoint loader: `src/utils/waypoint_loader.py`
- [ ] Test baseline controller in simulation
- [ ] Validate waypoint following accuracy

### Phase 3: Evaluation Infrastructure (2-3 days)

- [ ] Create `scripts/evaluate_baseline.py`
- [ ] Run baseline evaluation (20, 50, 100 NPCs)
- [ ] Collect metrics (success rate, speed, collisions, jerk)
- [ ] Generate comparison report vs TD3
- [ ] Update `BASELINE_CONTROLLER_ANALYSIS.md` with results

### Phase 4: Paper Updates (1 day)

- [ ] Update Section IV.B with PID + Pure Pursuit description
- [ ] Add ROS 2 architecture diagram
- [ ] Include baseline comparison results
- [ ] Add references to controller parameters

---

## 🚨 Risk Mitigation

### Risk 1: Native ROS 2 API Not Fully Documented

**Likelihood**: HIGH (release notes mention feature, but no docs found yet)

**Mitigation Strategies**:

**Option A** (Preferred): **Reverse-engineer from Docker image**
```bash
# If ROS 2 topics exist, we can infer API
docker start carla-server
docker exec carla-server bash -c "source /opt/ros/foxy/setup.bash && ros2 topic list"
docker exec carla-server bash -c "source /opt/ros/foxy/setup.bash && ros2 interface list | grep carla"
```

**Option B**: **Use External Bridge** (fallback)
- Implement with `carla-ros-bridge` ROS 2 branch
- Document performance comparison (native would be future work)

**Option C**: **Direct Python API Wrapper** (last resort)
- Create Python wrapper around `controller2d.py`
- No ROS 2, but fastest to implement
- Still valid for paper comparison

### Risk 2: Controller Tuning Required

**Likelihood**: MEDIUM (CARLA physics may differ from legacy version)

**Mitigation**:
- Start with documented gains from `controller2d.py`
- Add parameter tuning script
- Document any changes made

---

## 📊 Decision Matrix

| Approach | Implementation Time | Performance | Documentation Quality | Paper Contribution |
|----------|---------------------|-------------|----------------------|-------------------|
| **Native ROS 2** | 5-9 days | ⭐⭐⭐⭐⭐ Best | ⭐⭐ Unknown | ⭐⭐⭐⭐⭐ Novel |
| **External Bridge** | 3-5 days | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Standard |
| **Direct API Wrapper** | 2-3 days | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐ Fair | ⭐⭐ Limited |

**Recommendation**: 
1. Spend 1 day researching native ROS 2 API
2. If found → proceed with native integration (best for paper)
3. If not found → use external bridge (proven, documented)

---

## 🔧 Technical Specifications Reference

### Controller Parameters (from `controller2d.py`)

```yaml
# PID Longitudinal Controller
pid:
  kp: 0.50
  ki: 0.30
  kd: 0.13
  integrator_min: 0.0
  integrator_max: 10.0

# Pure Pursuit Lateral Controller
pure_pursuit:
  lookahead_distance: 2.0  # meters
  kp_heading: 8.00
  k_speed_crosstrack: 0.00
  cross_track_deadband: 0.01  # meters
```

### System Timing

```yaml
carla_simulation_rate: 20  # Hz (0.05s fixed_delta_seconds)
ros2_node_rate: 20         # Hz (match CARLA)
control_loop_rate: 20      # Hz
```

### Expected ROS 2 Topics (Based on Bridge Conventions)

```
Publishers (Baseline → CARLA):
  /carla/ego_vehicle/vehicle_control_cmd
    Type: carla_msgs/CarlaEgoVehicleControl
    Rate: 20 Hz

Subscribers (CARLA → Baseline):
  /carla/ego_vehicle/vehicle_status
    Type: carla_msgs/CarlaEgoVehicleStatus
    Rate: 20 Hz
    
  /carla/ego_vehicle/odometry
    Type: nav_msgs/Odometry
    Rate: 20 Hz
```

---

## 📝 Communication Plan

### User Status Updates

**After Phase 1 Research** (1-2 days):
- Report findings on native ROS 2 API availability
- Present decision: Which architecture to use
- Updated timeline based on chosen approach

**After Phase 2 Implementation** (2-3 days):
- Demo baseline controller following waypoints
- Performance metrics (smooth tracking, no collisions)
- Code review: Controller module + ROS 2 node

**After Phase 3 Evaluation** (2-3 days):
- Baseline vs TD3 comparison results
- Success rates, collision metrics, comfort metrics
- Recommendation for paper inclusion

---

## 🎓 Learning Objectives

This implementation provides hands-on experience with:

1. **ROS 2 Architecture**: Modern robotics middleware
2. **Control Systems**: PID + Pure Pursuit for autonomous vehicles
3. **System Integration**: Connecting simulator with control software
4. **Scientific Evaluation**: Fair baseline comparison methodology
5. **Technical Writing**: Documenting system architecture

---

## 📚 Resources

### Primary References
- CARLA 0.9.16 Release: https://carla.org/2025/09/16/release-0.9.16/
- CARLA Documentation: https://carla.readthedocs.io/en/latest/
- ROS 2 Foxy Docs: https://docs.ros.org/en/foxy/
- CARLA ROS Bridge: https://github.com/carla-simulator/ros-bridge

### Code References
- Legacy Controller: `related_works/exemple-carlaClient-openCV-YOLO-town01-waypoints/controller2d.py`
- Current DRL Env: `av_td3_system/src/environment/carla_env.py`
- Configuration: `av_td3_system/config/carla_config.yaml`

---

## ✅ Success Criteria

**Phase 1 Complete** when:
- [ ] Native ROS 2 API documented OR decision made on alternative
- [ ] Test script successfully connects to CARLA
- [ ] Topic names and message types confirmed
- [ ] Timeline updated based on chosen approach

**Phase 2 Complete** when:
- [ ] Baseline controller follows waypoints smoothly
- [ ] No crashes or unexpected behavior
- [ ] Code passes review (clean, documented)
- [ ] Unit tests for controller logic

**Phase 3 Complete** when:
- [ ] Evaluation runs complete (60 episodes total)
- [ ] Metrics match format of TD3 evaluation
- [ ] Comparison table generated
- [ ] Results ready for paper inclusion

**Project Complete** when:
- [ ] Paper Section IV.B updated
- [ ] All code committed to repository
- [ ] Documentation complete
- [ ] Baseline results validated

---

**Current Status**: 📍 **Phase 1 - Research Native ROS 2 API**

**Estimated Completion**: 
- Phase 1: November 23-24, 2025
- Full Implementation: November 29-December 1, 2025


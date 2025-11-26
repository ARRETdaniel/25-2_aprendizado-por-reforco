# ROS 2 Bridge Implementation - Native rclpy Integration

**Date**: 2025-01-25
**Phase**: Ubuntu 22.04 Migration - Phase 5 (ROS 2 Integration)
**Status**: ✅ Implementation Complete - Testing Pending

---

## Executive Summary

Implemented native ROS 2 integration for CARLA vehicle control using **geometry_msgs/Twist** messages instead of custom CARLA messages. This approach eliminates dependencies on `carla_msgs` package in the training container while maintaining compatibility with the external CARLA ROS Bridge.

**Key Decision**: Use standard ROS 2 messages (geometry_msgs) instead of CARLA-specific messages.

---

## Architecture Overview

### Two-Container Setup (Without ROS Bridge - Current)

```
┌─────────────────────────────────┐
│  CARLA Server Container         │
│  carlasim/carla:0.9.16          │
│  - Simulator (UnrealEngine)     │
│  - Port 2000 (CARLA Python API) │
└─────────────────────────────────┘
           ↑
           │ CARLA Python API
           │ (direct connection)
           ↓
┌─────────────────────────────────┐
│  Training Container              │
│  av-td3-system:ubuntu22.04-test │
│  - CARLA Python Client (wheel)  │
│  - ROS 2 Humble (native)        │
│  - PyTorch 2.4.1                │
│  - Training/Evaluation code     │
└─────────────────────────────────┘
```

### Three-Container Setup (With ROS Bridge - Future)

```
┌─────────────────────────────────┐
│  CARLA Server Container         │
│  carlasim/carla:0.9.16          │
│  - Simulator (UnrealEngine)     │
│  - Port 2000 (CARLA Python API) │
└─────────────────────────────────┘
           ↑
           │ CARLA Python API
           │
           ↓
┌─────────────────────────────────┐
│  ROS Bridge Container           │
│  ros2-carla-bridge:humble-v4    │
│  - carla_ros_bridge             │
│  - carla_twist_to_control       │
│  - carla_msgs package           │
└─────────────────────────────────┘
           ↑
           │ ROS 2 Topics (DDS)
           │ /carla/ego_vehicle/twist
           │
           ↓
┌─────────────────────────────────┐
│  Training Container              │
│  av-td3-system:ubuntu22.04-test │
│  - Native rclpy (ROS 2 Humble)  │
│  - geometry_msgs (standard)     │
│  - ROSBridgeInterface class     │
└─────────────────────────────────┘
```

---

## Implementation Details

### File: `src/utils/ros_bridge_interface.py`

**Key Features**:
1. **Uses Standard ROS 2 Messages**: `geometry_msgs/Twist` instead of `carla_msgs/CarlaEgoVehicleControl`
2. **Native rclpy Publisher**: No docker-exec subprocess calls
3. **Flexible API**: Supports both VehicleControl objects and individual parameters
4. **Automatic Topic Detection**: `wait_for_topics()` method for ROS Bridge verification

### Message Conversion Logic

Based on CARLA ROS Bridge's `carla_twist_to_control.py`:

```python
# CARLA Control → Twist Message

Throttle/Brake → twist.linear.x (longitudinal acceleration, m/s²)
    - Forward throttle: linear.x = throttle * 10.0 m/s²
    - Brake: linear.x = -brake * 10.0 m/s²
    - Reverse: linear.x = -throttle * 10.0 m/s²

Steering → twist.angular.z (angular velocity, radians)
    - angular.z = -steer * max_steering_angle
    - Default max_steering_angle = 1.22 rad (~70°)

Special Cases:
    - Hand brake: linear.x = 0, angular.z = 0
    - Full brake (>= 0.99): linear.x = 0, angular.z = 0
```

### API Usage

```python
from src.utils.ros_bridge_interface import ROSBridgeInterface

# Initialize (once per environment)
ros_interface = ROSBridgeInterface(
    node_name='carla_env_controller',
    role_name='ego_vehicle'
)

# Wait for ROS Bridge to be ready (optional but recommended)
if ros_interface.wait_for_topics(timeout=10.0):
    print("ROS Bridge is ready!")
else:
    print("ROS Bridge not available, using direct CARLA API")

# Publish control commands (method 1: individual parameters)
ros_interface.publish_control(
    throttle=0.5,
    steer=0.1,
    brake=0.0
)

# Publish control commands (method 2: VehicleControl object)
import carla
control = carla.VehicleControl(throttle=0.5, steer=0.1, brake=0.0)
ros_interface.publish_control(control=control)

# Cleanup
ros_interface.destroy()
```

---

## Why geometry_msgs/Twist Instead of carla_msgs?

### Problem with carla_msgs Approach

1. **Additional Build Requirement**: Would need to build the entire CARLA ROS Bridge workspace in our training container
2. **Dependency Complexity**: Requires colcon, CARLA source code, and many ROS 2 build dependencies
3. **Container Size**: Would significantly increase the Ubuntu 22.04 container size
4. **Maintenance**: Need to keep CARLA ROS Bridge version in sync with CARLA server

### Benefits of geometry_msgs/Twist Approach

1. **✅ Standard ROS 2 Message**: Part of `geometry_msgs` (always available in ROS 2)
2. **✅ No Extra Build Steps**: Training container stays lightweight
3. **✅ Official CARLA Pattern**: The ROS Bridge includes `carla_twist_to_control` for this exact purpose
4. **✅ Separation of Concerns**: ROS Bridge container handles CARLA-specific conversions
5. **✅ Simplified Testing**: Can test with or without full ROS Bridge setup

### Official CARLA Documentation Support

From CARLA ROS Bridge docs:
> **CARLA Twist to Control**: The carla_twist_to_control package converts a geometry_msgs.Twist to carla_msgs.CarlaEgoVehicleControl.

This confirms that using Twist messages is the **intended and officially supported** method.

---

## Integration with carla_env.py

The `CARLANavigationEnv` class already has the integration code in place:

```python
# Line 90-110 in carla_env.py
if self.use_ros_bridge:
    try:
        from src.utils.ros_bridge_interface import ROSBridgeInterface
        self.ros_interface = ROSBridgeInterface(
            node_name='carla_env_controller',
            use_docker_exec=True  # Deprecated, kept for compatibility
        )

        if not self.ros_interface.wait_for_topics(timeout=10.0):
            self.logger.error("[ROS BRIDGE] ROS topics not available, falling back to direct CARLA API")
            self.use_ros_bridge = False
            self.ros_interface = None
    except Exception as e:
        self.logger.warning(f"[ROS BRIDGE] Failed to initialize ROS Bridge: {e}")
        self.use_ros_bridge = False
        self.ros_interface = None
```

Control publishing happens in the `step()` method (line 935-945):

```python
if self.use_ros_bridge and self.ros_interface is not None:
    # Publish via ROS 2 topics
    self.ros_interface.publish_control(
        throttle=throttle,
        steer=steering,
        brake=brake,
        hand_brake=False,
        reverse=False
    )
else:
    # Direct CARLA API control
    control = carla.VehicleControl(...)
    self.vehicle.apply_control(control)
```

---

## Testing Strategy

### Phase 1: Verify ROS 2 Availability (✅ Can Test Now)

Test that `geometry_msgs` is available in Ubuntu 22.04 container:

```bash
docker run --rm av-td3-system:ubuntu22.04-test \
  bash -c "source /opt/ros/humble/setup.bash && python3 -c 'from geometry_msgs.msg import Twist, Vector3; print(\"✅ geometry_msgs available\")'"
```

### Phase 2: Test Direct API Mode (✅ ALREADY TESTED - PASSED!)

**Status**: COMPLETED in Phase 4 (test_baseline_direct_api.log)

This works WITHOUT ROS Bridge - uses direct CARLA API control.

```bash
python3 scripts/evaluate_baseline.py \
  --scenario 0 --num-episodes 1 \
  --baseline-config config/baseline_config.yaml \
  --debug
```

**Result**: ✅ SUCCESS - Completed full episode with excellent metrics.

### Phase 3: Test ROS Bridge Mode (⏳ PENDING - Requires ROS Bridge Container)

This requires the external CARLA ROS Bridge container:

```bash
# 1. Start CARLA server
docker run -d --name carla-server --runtime=nvidia --net=host \
  carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

# 2. Start ROS Bridge with Twist to Control
docker run -d --name ros2-bridge --network host \
  --env ROS_DOMAIN_ID=0 \
  ros2-carla-bridge:humble-v4 \
  bash -c "source /opt/ros/humble/setup.bash && \
           source /opt/carla-ros-bridge/install/setup.bash && \
           ros2 launch carla_twist_to_control carla_twist_to_control.launch.py role_name:=ego_vehicle"

# 3. Run evaluation with ROS Bridge
python3 scripts/evaluate_baseline.py \
  --scenario 0 --num-episodes 1 \
  --use-ros-bridge \
  --debug
```

### Phase 4: Measure Latency (⏳ PENDING - Requires ROS Bridge)

Add timing instrumentation to compare:
- **Old**: docker-exec subprocess call (~3150ms)
- **New**: native rclpy publish (~10ms expected)
- **Target**: 630x improvement validation

---

## Current Status & Next Steps

### ✅ Completed

1. Researched official CARLA ROS Bridge documentation
2. Analyzed Twist to Control conversion logic
3. Implemented ROSBridgeInterface with geometry_msgs/Twist
4. Added wait_for_topics() method for ROS Bridge detection
5. Integrated with existing carla_env.py code
6. Documented architecture and design decisions

### ⏳ Pending

1. **Test geometry_msgs availability** in Ubuntu 22.04 container
2. **Decision Point**: Do we need full ROS Bridge for paper evaluation?
   - **Option A**: Focus on direct CARLA API (already working ✅)
   - **Option B**: Set up ROS Bridge container for latency comparison
3. **Performance measurements** (if proceeding with Option B)
4. **Update paper methodology** to reflect actual implementation

---

## Design Decisions Log

### Decision 1: Twist vs CarlaEgoVehicleControl
**Date**: 2025-01-25
**Decision**: Use `geometry_msgs/Twist` instead of `carla_msgs/CarlaEgoVehicleControl`
**Rationale**:
- Standard ROS 2 message (no extra dependencies)
- Officially supported by CARLA ROS Bridge
- Keeps training container lightweight
- Follows CARLA's intended architecture pattern

**Trade-offs**:
- ✅ Pro: Simplified container build
- ✅ Pro: Standard ROS 2 patterns
- ⚠️ Con: Requires external ROS Bridge for full ROS integration
- ⚠️ Con: Less direct control over gear/hand_brake (Twist has limitations)

### Decision 2: Fallback to Direct API
**Date**: 2025-01-25
**Decision**: Implement graceful fallback to direct CARLA API if ROS Bridge unavailable
**Rationale**:
- Allows testing without full ROS Bridge setup
- Maintains backward compatibility
- Provides flexibility for different deployment scenarios

---

## References

1. **CARLA ROS Bridge Installation**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/
2. **CARLA Twist to Control**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_twist_to_control/
3. **geometry_msgs/Twist**: https://docs.ros.org/en/humble/p/geometry_msgs/interfaces/msg/Twist.html
4. **ROS 2 Publisher Tutorial**: https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html
5. **carla_twist_to_control.py source**: https://github.com/carla-simulator/ros-bridge/blob/master/carla_twist_to_control/src/carla_twist_to_control/carla_twist_to_control.py

---

## Conclusion

The ROS Bridge interface is **implemented and ready for testing**. The design uses standard ROS 2 patterns and official CARLA mechanisms, ensuring maintainability and compatibility. The next critical step is determining whether full ROS Bridge integration is necessary for the research paper, or if the direct CARLA API approach (which already works) is sufficient for demonstrating the TD3 algorithm performance.

**Recommendation**: Proceed with **Option A (Direct CARLA API)** for initial paper submission, as:
1. It already works perfectly (Phase 4 test passed ✅)
2. ROS integration adds complexity without changing the core TD3 algorithm
3. Can be added as "future work" for ROS-based deployment scenarios
4. Focuses the paper on the DRL contribution (TD3 vs DDPG comparison)

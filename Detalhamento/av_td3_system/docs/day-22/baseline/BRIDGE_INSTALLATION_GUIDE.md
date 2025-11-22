# CARLA ROS Bridge Installation & Setup Guide

**Target System**: Ubuntu 20.04
**ROS Version**: ROS 2 Foxy
**CARLA Version**: 0.9.16 (Docker)
**Bridge Version**: ros2 branch (latest)

---

## Prerequisites Verification

### 1. System Requirements

✅ **Operating System**: Ubuntu 20.04 (Focal Fossa)
✅ **CARLA**: 0.9.16 running in Docker container `carla-server`
❌ **ROS 2 Foxy**: NOT installed (need to install)

**Current Status**: ROS 2 is missing. Will install in Step 2.

---

## Installation Steps

### Step 1: Install ROS 2 Foxy

**Official Documentation**: https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html

```bash
# 1.1 Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 1.2 Setup sources
sudo apt install software-properties-common
sudo add-apt-repository universe

# 1.3 Add ROS 2 GPG key
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

# 1.4 Add repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 1.5 Install ROS 2 Foxy Desktop (includes RViz, demos, tutorials)
sudo apt update
sudo apt install ros-foxy-desktop

# 1.6 Install additional development tools
sudo apt install python3-colcon-common-extensions python3-rosdep

# 1.7 Initialize rosdep
sudo rosdep init
rosdep update

# 1.8 Verify installation
source /opt/ros/foxy/setup.bash
ros2 --version  # Should output: ros2 cli version: 0.9.X
```

**Expected Output**:
```
ros2 cli version: 0.9.9
ros client library version: 1.1.14
```

**Add to .bashrc** (permanent ROS 2 sourcing):
```bash
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Clone CARLA ROS Bridge Repository

```bash
# 2.1 Create workspace
mkdir -p ~/carla-ros-bridge/src
cd ~/carla-ros-bridge

# 2.2 Clone bridge (with submodules)
git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git src/ros-bridge

# 2.3 Verify clone
ls -la src/ros-bridge/
# Expected: carla_ros_bridge/, carla_msgs/, carla_ackermann_control/, etc.
```

### Step 3: Install Dependencies

```bash
# 3.1 Ensure ROS 2 environment is sourced
source /opt/ros/foxy/setup.bash

# 3.2 Update rosdep database
rosdep update

# 3.3 Install bridge dependencies
cd ~/carla-ros-bridge
rosdep install --from-paths src --ignore-src -r -y

# Expected dependencies:
# - ros-foxy-rclpy
# - ros-foxy-std-msgs
# - ros-foxy-geometry-msgs
# - ros-foxy-nav-msgs
# - ros-foxy-sensor-msgs
# - ros-foxy-tf2-ros
# - python3-numpy
# - python3-pygame (for manual control)
```

### Step 4: Set CARLA Python API Path

**Critical**: Bridge needs access to CARLA's Python API.

```bash
# 4.1 Verify CARLA container is running
docker ps | grep carla-server
# Should show: carla-server ... Up X minutes

# 4.2 Find CARLA .egg file in container
docker exec carla-server find /home/carla -name "carla-*.egg" 2>/dev/null

# Expected output (example):
# /home/carla/PythonAPI/carla/dist/carla-0.9.16-py3.7-linux-x86_64.egg
```

**Option A**: Extract .egg from container (recommended):
```bash
# 4.3 Create local CARLA Python API directory
mkdir -p ~/carla_python_api

# 4.4 Copy .egg file from container
docker cp carla-server:/home/carla/PythonAPI/carla/dist/carla-0.9.16-py3.7-linux-x86_64.egg \
  ~/carla_python_api/

# 4.5 Also copy carla module
docker cp carla-server:/home/carla/PythonAPI/carla \
  ~/carla_python_api/

# 4.6 Set PYTHONPATH
export CARLA_ROOT=~/carla_python_api
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/carla-0.9.16-py3.7-linux-x86_64.egg:$CARLA_ROOT/carla
```

**Add to .bashrc** (permanent):
```bash
cat >> ~/.bashrc << 'EOF'

# CARLA Python API
export CARLA_ROOT=~/carla_python_api
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/carla-0.9.16-py3.7-linux-x86_64.egg:$CARLA_ROOT/carla
EOF

source ~/.bashrc
```

**Verify**:
```bash
python3 -c 'import carla; print("✅ CARLA Python API imported successfully")'
# Should print success message without errors
```

### Step 5: Build CARLA ROS Bridge

```bash
# 5.1 Navigate to workspace
cd ~/carla-ros-bridge

# 5.2 Source ROS 2
source /opt/ros/foxy/setup.bash

# 5.3 Build with colcon
colcon build

# Expected output:
# Starting >>> carla_msgs
# Finished <<< carla_msgs [XX.Xs]
# Starting >>> carla_ros_bridge
# Finished <<< carla_ros_bridge [XX.Xs]
# ... (other packages)
#
# Summary: X packages finished [XXX.Xs]
```

**If build fails**, check:
- ROS 2 Foxy sourced: `echo $ROS_DISTRO` → should print `foxy`
- Dependencies installed: `rosdep check --from-paths src --ignore-src`
- Python version: `python3 --version` → should be 3.8.x

**Troubleshooting Build Errors**:

**Error**: `Package 'carla_msgs' not found`
```bash
# Build only carla_msgs first
colcon build --packages-select carla_msgs
# Then build everything else
colcon build
```

**Error**: `No module named 'carla'`
```bash
# Verify PYTHONPATH
echo $PYTHONPATH | grep carla
# Should contain path to .egg file
```

### Step 6: Source Workspace

```bash
# 6.1 Source the newly built workspace
source ~/carla-ros-bridge/install/setup.bash

# 6.2 Verify packages are found
ros2 pkg list | grep carla

# Expected output:
# carla_ackermann_control
# carla_ad_agent
# carla_msgs
# carla_ros_bridge
# carla_spawn_objects
# ... etc
```

**Add to .bashrc** (permanent):
```bash
echo "source ~/carla-ros-bridge/install/setup.bash" >> ~/.bashrc
```

---

## Testing the Bridge

### Test 1: Launch Bridge (Basic)

**Terminal 1** - Ensure CARLA is running:
```bash
docker ps | grep carla-server
# Should show: Up X minutes

# If not running:
docker start carla-server
sleep 5  # Wait for startup
```

**Terminal 2** - Launch bridge:
```bash
source ~/carla-ros-bridge/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py
```

**Expected Output**:
```
[INFO] [carla_ros_bridge]: Trying to connect to CARLA at localhost:2000
[INFO] [carla_ros_bridge]: Connected to CARLA 0.9.16
[INFO] [carla_ros_bridge]: Synchronous mode: ON
[INFO] [carla_ros_bridge]: Fixed delta seconds: 0.05
```

**If connection fails**:
- Check CARLA is running: `docker ps | grep carla`
- Check port 2000 is open: `netstat -tuln | grep 2000`
- Verify host parameter: Add `host:=localhost` to launch command

### Test 2: Verify Topics

**Terminal 3** (while bridge is running):
```bash
# List all CARLA topics
ros2 topic list | grep carla

# Expected topics:
# /carla/status
# /carla/world_info
# /carla/actor_list
# /carla/traffic_lights
# (no ego vehicle yet - needs to be spawned)
```

### Test 3: Spawn Ego Vehicle

**Terminal 4**:
```bash
# Stop the basic bridge (Ctrl+C in Terminal 2)

# Launch bridge WITH example ego vehicle
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py
```

**Expected Output**:
```
[INFO] [carla_ros_bridge]: Connected to CARLA 0.9.16
[INFO] [carla_spawn_objects]: Spawning ego vehicle...
[INFO] [carla_spawn_objects]: Ego vehicle spawned with ID: 123
[INFO] [carla_ros_bridge]: Publishing sensor data...
```

**Terminal 5** - Verify ego vehicle topics:
```bash
ros2 topic list | grep ego_vehicle

# Expected topics:
# /carla/ego_vehicle/camera/rgb/image_raw
# /carla/ego_vehicle/camera/rgb/camera_info
# /carla/ego_vehicle/collision
# /carla/ego_vehicle/imu
# /carla/ego_vehicle/lane_invasion
# /carla/ego_vehicle/odometry
# /carla/ego_vehicle/vehicle_control_cmd  (subscriber)
# /carla/ego_vehicle/vehicle_status
# /carla/ego_vehicle/gnss
```

### Test 4: Monitor Odometry

```bash
# Echo odometry topic (should publish at 20 Hz)
ros2 topic echo /carla/ego_vehicle/odometry

# Expected output (continuously updating):
# header:
#   stamp:
#     sec: 1234567890
#     nanosec: 123456789
#   frame_id: map
# child_frame_id: ego_vehicle
# pose:
#   pose:
#     position:
#       x: 123.45
#       y: 67.89
#       z: 0.3
#     orientation:
#       x: 0.0
#       y: 0.0
#       z: 0.707
#       w: 0.707
# twist:
#   twist:
#     linear:
#       x: 0.0  # velocity in m/s
#       y: 0.0
#       z: 0.0
#     angular:
#       x: 0.0
#       y: 0.0
#       z: 0.0  # yaw rate
```

**Check publish rate**:
```bash
ros2 topic hz /carla/ego_vehicle/odometry

# Expected: average rate: 20.000
```

### Test 5: Send Control Command

```bash
# Publish a manual control command (move forward slowly)
ros2 topic pub --once /carla/ego_vehicle/vehicle_control_cmd \
  carla_msgs/msg/CarlaEgoVehicleControl \
  "{throttle: 0.3, steer: 0.0, brake: 0.0, hand_brake: false, reverse: false, gear: 1, manual_gear_shift: false}"

# Vehicle in CARLA should start moving forward!
```

**Expected Behavior**:
- Vehicle accelerates to ~30% throttle
- Moves straight (steer=0.0)
- Can be stopped with: `{throttle: 0.0, brake: 1.0}`

### Test 6: Synchronous Mode Verification

```bash
# Subscribe to clock topic
ros2 topic echo /clock

# Expected output (only advances when bridge ticks):
# clock:
#   sec: 0
#   nanosec: 50000000  # 0.05s increments (20 Hz)
```

**Critical**: If `/clock` is publishing, synchronous mode is active. All nodes should use sim time.

---

## Message Type Details (For Implementation)

### Control Message (Publisher)

**Topic**: `/carla/ego_vehicle/vehicle_control_cmd`
**Type**: `carla_msgs/msg/CarlaEgoVehicleControl`

**Fields**:
```python
std_msgs/Header header
float32 throttle          # [0.0, 1.0] - accelerator pedal position
float32 steer             # [-1.0, 1.0] - steering wheel angle (-1=full left, +1=full right)
float32 brake             # [0.0, 1.0] - brake pedal position
bool hand_brake           # emergency/parking brake
bool reverse              # reverse gear engaged
int32 gear                # manual gear selection (if manual_gear_shift=true)
bool manual_gear_shift    # disable automatic transmission
```

**Conversion from Baseline Controller**:

Our PID+Pure Pursuit outputs:
- `throttle_brake`: float in [-1, 1] (negative=brake, positive=throttle)
- `steer`: float in [-1, 1]

**Mapping**:
```python
def convert_to_carla_control(throttle_brake: float, steer: float) -> CarlaEgoVehicleControl:
    """Convert baseline controller output to CARLA message."""
    msg = CarlaEgoVehicleControl()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "ego_vehicle"

    # Split throttle/brake
    if throttle_brake >= 0.0:
        msg.throttle = float(np.clip(throttle_brake, 0.0, 1.0))
        msg.brake = 0.0
    else:
        msg.throttle = 0.0
        msg.brake = float(np.clip(-throttle_brake, 0.0, 1.0))

    msg.steer = float(np.clip(steer, -1.0, 1.0))
    msg.hand_brake = False
    msg.reverse = False
    msg.gear = 1
    msg.manual_gear_shift = False

    return msg
```

### State Messages (Subscribers)

#### 1. Odometry (Primary State Source)

**Topic**: `/carla/ego_vehicle/odometry`
**Type**: `nav_msgs/msg/Odometry` (standard ROS message)
**Rate**: 20 Hz

**Fields**:
```python
std_msgs/Header header
string child_frame_id
geometry_msgs/PoseWithCovariance pose
  geometry_msgs/Pose pose
    geometry_msgs/Point position     # x, y, z in map frame
    geometry_msgs/Quaternion orientation  # orientation as quaternion
  float64[36] covariance
geometry_msgs/TwistWithCovariance twist
  geometry_msgs/Twist twist
    geometry_msgs/Vector3 linear    # velocity (m/s) in vehicle frame
    geometry_msgs/Vector3 angular   # angular velocity (rad/s)
  float64[36] covariance
```

**Extract State Variables**:
```python
def extract_state(odometry_msg) -> dict:
    """Extract x, y, yaw, speed from odometry."""
    # Position
    x = odometry_msg.pose.pose.position.x
    y = odometry_msg.pose.pose.position.y

    # Orientation (quaternion to yaw)
    quat = odometry_msg.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (quat.w * quat.z + quat.x * quat.y),
        1.0 - 2.0 * (quat.y**2 + quat.z**2)
    )

    # Velocity (magnitude of linear velocity)
    vx = odometry_msg.twist.twist.linear.x
    vy = odometry_msg.twist.twist.linear.y
    speed = math.sqrt(vx**2 + vy**2)

    return {
        'x': x,
        'y': y,
        'yaw': yaw,
        'speed': speed
    }
```

#### 2. Vehicle Status (Alternative State Source)

**Topic**: `/carla/ego_vehicle/vehicle_status`
**Type**: `carla_msgs/msg/CarlaEgoVehicleStatus`
**Rate**: 20 Hz

**Fields**:
```python
std_msgs/Header header
float32 velocity                           # m/s (scalar speed)
geometry_msgs/Accel acceleration
  geometry_msgs/Vector3 linear            # m/s²
  geometry_msgs/Vector3 angular
geometry_msgs/Quaternion orientation       # current orientation
carla_msgs/CarlaEgoVehicleControl control  # echo of last control command
```

**Use Case**: If we need acceleration for jerk calculation:
```python
def calculate_jerk(current_accel, previous_accel, dt):
    """Jerk = d(acceleration)/dt"""
    return (current_accel - previous_accel) / dt
```

#### 3. Collision Events

**Topic**: `/carla/ego_vehicle/collision`
**Type**: `carla_msgs/msg/CarlaCollisionEvent`
**Rate**: Event-driven (only on collision)

**Fields**:
```python
std_msgs/Header header
uint32 other_actor_id      # ID of collided object
geometry_msgs/Vector3 normal_impulse  # collision force vector
```

**Subscriber**:
```python
def collision_callback(msg):
    """Handle collision events."""
    self.collision_count += 1
    self.episode_done = True
    logger.warning(f"Collision detected with actor {msg.other_actor_id}")
```

#### 4. Lane Invasion Events

**Topic**: `/carla/ego_vehicle/lane_invasion`
**Type**: `carla_msgs/msg/CarlaLaneInvasionEvent`
**Rate**: Event-driven

**Fields**:
```python
std_msgs/Header header
int32[] crossed_lane_markings  # BROKEN=1, SOLID=2, OTHER=0
```

**Use Case**: Track lane departures (penalty in reward function)

---

## Launch File Configuration

### Custom Launch File for Baseline Controller

**File**: `~/carla-ros-bridge/src/ros-bridge/carla_ros_bridge/launch/baseline_carla_bridge.launch.py`

```python
#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch CARLA ROS bridge for baseline controller testing.

    Configured for:
    - Synchronous mode: ON
    - Fixed delta: 0.05s (20 Hz)
    - Ego vehicle: Spawned automatically
    - Sensors: RGB camera, collision, IMU, GNSS, odometry
    """

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument(
            'host',
            default_value='localhost',
            description='CARLA server host'
        ),
        DeclareLaunchArgument(
            'port',
            default_value='2000',
            description='CARLA server port'
        ),
        DeclareLaunchArgument(
            'synchronous_mode',
            default_value='True',
            description='Enable synchronous simulation'
        ),
        DeclareLaunchArgument(
            'fixed_delta_seconds',
            default_value='0.05',
            description='Simulation timestep (20 Hz)'
        ),
        DeclareLaunchArgument(
            'town',
            default_value='Town01',
            description='CARLA map to load'
        ),

        # CARLA ROS Bridge Node
        Node(
            package='carla_ros_bridge',
            executable='carla_ros_bridge',
            name='carla_ros_bridge',
            output='screen',
            parameters=[{
                'host': LaunchConfiguration('host'),
                'port': LaunchConfiguration('port'),
                'synchronous_mode': LaunchConfiguration('synchronous_mode'),
                'fixed_delta_seconds': LaunchConfiguration('fixed_delta_seconds'),
                'town': LaunchConfiguration('town'),
                'use_sim_time': True,
            }]
        ),

        # Ego Vehicle Spawner Node
        Node(
            package='carla_spawn_objects',
            executable='carla_spawn_objects',
            name='carla_spawn_objects',
            output='screen',
            parameters=[{
                'objects_definition_file': '/path/to/baseline_ego_vehicle.json',
                'use_sim_time': True,
            }]
        ),
    ])
```

**Ego Vehicle Definition** (`baseline_ego_vehicle.json`):
```json
{
  "objects": [
    {
      "type": "vehicle.tesla.model3",
      "id": "ego_vehicle",
      "sensors": [
        {
          "type": "sensor.camera.rgb",
          "id": "front_camera",
          "x": 2.0,
          "y": 0.0,
          "z": 1.5,
          "roll": 0.0,
          "pitch": 0.0,
          "yaw": 0.0,
          "width": 800,
          "height": 600,
          "fov": 90
        },
        {
          "type": "sensor.other.collision",
          "id": "collision"
        },
        {
          "type": "sensor.other.imu",
          "id": "imu"
        },
        {
          "type": "sensor.other.gnss",
          "id": "gnss"
        },
        {
          "type": "sensor.other.lane_invasion",
          "id": "lane_invasion"
        }
      ]
    }
  ]
}
```

---

## Synchronous Mode Details

### How Synchronous Mode Works with ROS 2

**Without Synchronous Mode** (asynchronous):
- CARLA runs as fast as possible
- ROS topics publish at arbitrary rates
- Non-deterministic simulation
- ❌ **NOT suitable for RL training**

**With Synchronous Mode**:
1. Bridge connects to CARLA, sets `synchronous_mode=True`
2. CARLA pauses, waits for `world.tick()` command
3. Bridge publishes `/clock` topic with current sim time
4. All ROS 2 nodes use sim time (via `use_sim_time=True` parameter)
5. Bridge calls `world.tick()` at fixed rate (20 Hz)
6. CARLA advances one physics step (0.05s)
7. CARLA publishes new sensor/state data
8. Bridge forwards to ROS topics
9. Controller nodes receive data, compute control
10. Control commands sent back to CARLA
11. Repeat from step 5

**Result**: Deterministic simulation, reproducible experiments

### Synchronous Mode Configuration Check

```bash
# While bridge is running, check CARLA status
ros2 topic echo /carla/status

# Expected output:
# frame: 12345
# fixed_delta_seconds: 0.05
# synchronous_mode: true
# synchronous_mode_running: true
```

---

## Common Issues & Solutions

### Issue 1: "ImportError: no module named 'carla'"

**Cause**: PYTHONPATH not set correctly

**Solution**:
```bash
# Find .egg file in container
docker exec carla-server find /home/carla -name "*.egg"

# Copy to host
docker cp carla-server:/home/carla/PythonAPI/carla/dist/carla-0.9.16-py3.7-linux-x86_64.egg ~/carla_python_api/

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/carla_python_api/carla-0.9.16-py3.7-linux-x86_64.egg

# Test
python3 -c 'import carla; print("Success")'
```

### Issue 2: Bridge Can't Connect to CARLA

**Symptoms**:
```
[ERROR] [carla_ros_bridge]: Could not connect to CARLA at localhost:2000
```

**Solutions**:
1. **Check CARLA is running**:
   ```bash
   docker ps | grep carla-server
   # Should show "Up X minutes"
   ```

2. **Check port 2000**:
   ```bash
   docker exec carla-server netstat -tuln | grep 2000
   # Should show: tcp 0.0.0.0:2000 LISTEN
   ```

3. **Try explicit host parameter**:
   ```bash
   ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=127.0.0.1
   ```

4. **Check Docker network mode**:
   ```bash
   docker inspect carla-server | grep NetworkMode
   # Should be: "NetworkMode": "host"

   # If not, recreate container with --net=host
   ```

### Issue 3: Low Topic Publish Rate

**Symptoms**:
```bash
ros2 topic hz /carla/ego_vehicle/odometry
# average rate: 5.000  ❌ (should be 20.000)
```

**Causes**:
- CARLA running too slow (insufficient GPU)
- Synchronous mode disabled
- Fixed delta seconds too large

**Solutions**:
1. **Verify synchronous mode**:
   ```bash
   ros2 topic echo /carla/status --once
   # Check: synchronous_mode: true
   ```

2. **Check CARLA performance**:
   ```bash
   docker exec carla-server nvidia-smi
   # Verify GPU utilization < 90%
   ```

3. **Reduce sensor load** (if using cameras):
   - Lower resolution: 800x600 → 400x300
   - Reduce FOV: 90 → 60 degrees
   - Disable unused sensors

### Issue 4: Topic Not Found

**Symptoms**:
```bash
ros2 topic list | grep ego_vehicle
# (no output)
```

**Cause**: Ego vehicle not spawned

**Solution**:
```bash
# Launch bridge WITH ego vehicle
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py

# Or manually spawn:
ros2 run carla_spawn_objects carla_spawn_objects \
  --objects-definition-file /path/to/ego_vehicle.json
```

---

## Next Steps

After successful bridge installation and testing:

1. ✅ **Create Baseline Controller Package**
   - Package: `av_td3_baseline`
   - Node: `baseline_controller_node`
   - Files: `pid_pure_pursuit.py`, `waypoint_loader.py`

2. ✅ **Implement Controller**
   - Subscribe: `/carla/ego_vehicle/odometry`
   - Publish: `/carla/ego_vehicle/vehicle_control_cmd`
   - Algorithm: PID (longitudinal) + Pure Pursuit (lateral)

3. ✅ **Test Waypoint Following**
   - Load waypoints from `config/waypoints.txt`
   - Run on Town01
   - Verify lane keeping and speed control

4. ✅ **Evaluation**
   - Collect metrics: success rate, collisions, speed, jerk
   - Compare with TD3 agent

---

## References

### Official Documentation
1. **ROS 2 Foxy Installation**: https://docs.ros.org/en/foxy/Installation.html
2. **CARLA ROS Bridge**: https://carla.readthedocs.io/projects/ros-bridge/
3. **Bridge ROS 2 Installation**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/
4. **Bridge Messages**: https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_msgs/
5. **CARLA Python API**: https://carla.readthedocs.io/en/latest/python_api/

### Code Examples
- Bridge launch files: `~/carla-ros-bridge/src/ros-bridge/carla_ros_bridge/launch/`
- Message definitions: `~/carla-ros-bridge/src/ros-bridge/carla_msgs/msg/`
- Example controllers: `~/carla-ros-bridge/src/ros-bridge/carla_ackermann_control/`

---

**Document Status**: ✅ Ready for implementation

**Tested**: NOT YET (documentation only - will test in next steps)

**Last Updated**: $(date)

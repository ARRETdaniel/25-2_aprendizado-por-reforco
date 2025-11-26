# TD3

docker stop carla-server
docker stop ros2-bridge

docker start ros2-bridge
docker start carla-server
docker rm carla-server

docker run -d --name carla-server --runtime=nvidia --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

## headless

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia   -e NVIDIA_VISIBLE_DEVICES=all   -e NVIDIA_DRIVER_CAPABILITIES=all   -e PYTHONUNBUFFERED=1   -e PYTHONPATH=/workspace   -v $(pwd):/workspace   -w /workspace   av-td3-system:ubuntu22.04-test   python3 scripts/train_td3.py     --scenario 0     --max-timesteps 100000     --eval-freq 20000     --checkpoint-freq 10000     --seed 42  --device cpu     2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-25/run-validation-runtest_$(date +%Y%m%d_%H%M%S).log


## DEBUG - CAMERA:



xhost +local:docker 2>/dev/null || echo "xhost not available, proceeding anyway"

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  av-td3-system:ubuntu22.04-test   python3 scripts/train_td3.py     --scenario 0     --max-timesteps 5000     --eval-freq 3001     --checkpoint-freq 5001     --seed 42  --debug   --device cpu     2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-25/RewardProgress.log


# then
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system &&tensorboard --logdir data/logs --port 6006

# BASELINE

xhost +local:docker 2>/dev/null || echo "xhost not available, proceeding anyway"

docker run -d --name carla-server --runtime=nvidia --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

## python API
 cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/workspace -w /workspace --privileged av-td3-system:ubuntu22.04-test bash -c "source /opt/ros/humble/setup.bash && python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 6 --baseline-config config/baseline_config.yaml --debug" 2>&1 | tee docs/day-25/migration/test_baseline_direct_api.log

## ROS 2 humble:
### 1.
docker run -d --name ros2-bridge \
  --network host \
  --env ROS_DOMAIN_ID=0 \
  ros2-carla-bridge:humble-v4 \
  bash -c "
    source /opt/ros/humble/setup.bash && \
    source /opt/carla-ros-bridge/install/setup.bash && \
    ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py timeout:=10 & \
    sleep 5 && \
    ros2 launch carla_twist_to_control carla_twist_to_control.launch.py role_name:=ego_vehicle & \
    wait
  "
### 2.

xhost +local:docker 2>/dev/null || echo "xhost not available, proceeding anyway"

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --gpus all --network=host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e AMENT_PREFIX_PATH=/opt/ros/humble \
  -e ROS_DISTRO=humble \
  -e ROS_VERSION=2 \
  -e ROS_PYTHON_VERSION=3 \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  -e PYTHONPATH=/workspace:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages \
  -e LD_LIBRARY_PATH=/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  av-td3-system:ubuntu22.04-test python3 scripts/evaluate_baseline.py --use-ros-bridge --num-episodes 1 --debug 2>&1 | tee test_ros2_bridge_WITH_DIRECT_CONTROL.log


# MANUAL VALIDATION WITH DEBUG LOGGING (CORRECTED):

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && \
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  bash -c "pip install pygame --quiet && python3 scripts/validate_rewards_manual.py --config config/baseline_config.yaml --output-dir validation_logs/debug_test --max-steps 100000 --log-level DEBUG" 2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-25/progress-manual.log

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system &&  docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/workspace -w /workspace --privileged td3-av-system:v2.0-python310 bash -c "pip install pygame --quiet && python3 scripts/validate_rewards_manual.py --config config/baseline_config.yaml --output-dir validation_logs/quick_test --max-steps 100000 --log-level DEBUG"

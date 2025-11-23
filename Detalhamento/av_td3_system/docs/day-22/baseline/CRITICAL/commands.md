# home:
/workspace/PythonAPI

# ros in docker:
docker run -d --name carla-server --runtime=nvidia --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all carlasim/carla:0.9.16 bash CarlaUE4.sh --ros2 -RenderOffScreen -nosound

# td3 DRL in docker:
## with debug window CV2
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
  td3-av-system:v2.0-python310   python3 scripts/train_td3.py     --scenario 0     --max-timesteps 5000     --eval-freq 3001     --checkpoint-freq 5001     --seed 42  --debug   --device cpu     2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-21/run6/run_RewardProgressdelete.log


# NOOTES:
After spawning a sensor with ros_name, we need to explicitly call .enable_for_ros() on the sensor to activate the ROS 2 publisher!

# ATTETION - read exemples files in docker api / ros2

docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all carlasim/carla:0.9.16 find /workspace -name "*ros2*" -o -name "*native*" 2>/dev/null | head -20
/workspace/PythonAPI/examples/ros2
/workspace/PythonAPI/examples/ros2/rviz/ros2_native.rviz
/workspace/PythonAPI/examples/ros2/ros2_native.py

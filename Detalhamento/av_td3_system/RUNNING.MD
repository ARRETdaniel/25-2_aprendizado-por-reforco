
# NEED TO CHECK MAY HAVE CHANGED:
docker run -d \
    --name carla-server \
    --runtime=nvidia \
    --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

# OR

docker start carla-server && sleep 5 && docker ps | grep carla-server


docker ps -a


# DEBUG MODE:

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/workspace -w /workspace td3-av-system:v2.0-python310 python3 scripts/train_td3.py --scenario 0 --max-timesteps 2000 --debug 2>&1 | tee debug_2000_steps



# REAL TRAINING

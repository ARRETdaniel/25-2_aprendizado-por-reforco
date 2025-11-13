docker ps -a | grep carla
docker stop carla-server
docker rm carla-server


docker rm carla-server 2>/dev/null || true

docker run -d --name carla-server --runtime=nvidia --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

xhost +local:docker 2>/dev/null || echo "xhost not available, proceeding anyway"

docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system:/workspace -w /workspace td3-av-system:v2.0-python310 python3 scripts/train_td3.py --scenario 0 --max-timesteps 1000 --eval-freq 500 --debug --device cpu 2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/DAY_FIVE-runAndDebug/debug_validation_$(date +%Y%m%d_%H%M%S).log

read: QUICKSTART.md
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

sudo lsof -i :8000 -i :8050


# tensorboard

tensorboard --logdir /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/data/logs --port 6008 &




cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -v $(pwd):/workspace -w /workspace td3-av-system:v2.0-python310 python3 scripts/train_td3.py --scenario 0 --max-timesteps 5000 --eval-freq 3001 --checkpoint-freq 1000 --seed 42 --device cpu 2>&1 | tee training-test-tensor_5k-3.log

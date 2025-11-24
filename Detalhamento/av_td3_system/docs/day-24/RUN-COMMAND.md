
### Step 1: Clear Python Cache
```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Remove all Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "✅ Python cache cleared"
```

docker stop carla-server
docker start carla-server
docker rm carla-server

docker run -d --name carla-server --runtime=nvidia --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia   -e NVIDIA_VISIBLE_DEVICES=all   -e NVIDIA_DRIVER_CAPABILITIES=all   -e PYTHONUNBUFFERED=1   -e PYTHONPATH=/workspace   -v $(pwd):/workspace   -w /workspace   td3-av-system:v2.0-python310   python3 scripts/train_td3.py     --scenario 0     --max-timesteps 100000     --eval-freq 20000     --checkpoint-freq 10000     --seed 42  --device cpu     2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-22/run1/run-validation-runtest_$(date +%Y%m%d_%H%M%S).log


DEBUG - CAMERA:



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
  td3-av-system:v2.0-python310   python3 scripts/train_td3.py     --scenario 0     --max-timesteps 5000     --eval-freq 3001     --checkpoint-freq 5001     --seed 42  --debug   --device cpu     2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-21/run6/run_RewardProgress6.log


# then
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system &&tensorboard --logdir data/logs --port 6006

# BASELINE

xhost +local:docker 2>/dev/null || echo "xhost not available, proceeding anyway"

docker run -d --name carla-server --runtime=nvidia --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/workspace -w /workspace --privileged td3-av-system:v2.0-python310 python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 1 --baseline-config config/baseline_config.yaml --debug 2>&1 | tee av_td3_system/docs/day-23/test.log


---
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -v $(pwd):/workspace -w /workspace td3-av-system:v2.0-python310 python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 1 --baseline-config config/baseline_config.yaml --debug 2>&1


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
  bash -c "pip install pygame --quiet && python3 scripts/validate_rewards_manual.py --config config/baseline_config.yaml --output-dir validation_logs/debug_test --max-steps 100000 --log-level DEBUG" 2>&1 | tee /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system/docs/day-24/progress.log

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system &&  docker run --rm --network host --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $(pwd):/workspace -w /workspace --privileged td3-av-system:v2.0-python310 bash -c "pip install pygame --quiet && python3 scripts/validate_rewards_manual.py --config config/baseline_config.yaml --output-dir validation_logs/quick_test --max-steps 100000 --log-level DEBUG"

# testes

docker run -d --name carla-server --runtime=nvidia --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all carlasim/carla:0.9.16 bash CarlaUE4.sh --ros2 -RenderOffScreen -nosound

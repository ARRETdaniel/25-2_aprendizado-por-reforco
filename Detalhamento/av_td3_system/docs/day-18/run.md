
docker stop carla-server
docker rm carla-server

docker run -d --name carla-server --runtime=nvidia --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound

cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system && docker run --rm --network host --runtime nvidia   -e NVIDIA_VISIBLE_DEVICES=all   -e NVIDIA_DRIVER_CAPABILITIES=all   -e PYTHONUNBUFFERED=1   -e PYTHONPATH=/workspace   -v $(pwd):/workspace   -w /workspace   td3-av-system:v2.0-python310   python3 scripts/train_td3.py     --scenario 0     --max-timesteps 5000     --eval-freq 3001     --checkpoint-freq 1000     --seed 42     --device cpu     2>&1 | tee validation_5k_post_all_fixes_2_$(date +%Y%m%d_%H%M%S).log

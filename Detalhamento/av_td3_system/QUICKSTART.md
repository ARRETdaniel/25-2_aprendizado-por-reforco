# TD3 Training Quick Start Guide

## 1. Start CARLA Server (Docker)

```bash
docker run -d \
    --name carla-server \
    --runtime=nvidia \
    --net=host \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    carlasim/carla:0.9.16 \
    bash CarlaUE4.sh -RenderOffScreen -nosound
```

**Stop CARLA:**
```bash
docker stop carla-server
docker rm carla-server
```

---

## 2. Check if CARLA is Running

```bash
# Check if CARLA Docker container is running
docker ps -a

# Should show: carla-server with status "Up"

# Check CARLA logs
docker logs carla-server

# Test connection with Python
python3 -c "import carla; client = carla.Client('localhost', 2000); client.set_timeout(2.0); print('CARLA is running!')"
```

---

## 3. Training Modes

### Debug Mode (Short Runs with Visual Feedback)

**Use for:** Testing, debugging, short runs (< 1000 steps)

```bash
cd av_td3_system

docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 1000 \
    --debug \
    --device cpu
```

**Features:**
- OpenCV window with camera view
- Detailed step-by-step logging
- Press 'q' to quit, 'p' to pause

---

### Normal Mode (Full Training)

**Use for:** Real training runs (30k+ steps)

```bash
cd av_td3_system

docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 30000 \
    --eval-freq 5000 \
    --checkpoint-freq 10000 \
    --seed 42 \
    --device cpu \
    2>&1 | tee training_log.log
```

**Parameters:**
- `--scenario`: Traffic density (0=20, 1=50, 2=100 NPCs)
- `--max-timesteps`: Total training steps (30000 = ~2-3 hours)
- `--eval-freq`: Evaluate every N steps
- `--checkpoint-freq`: Save checkpoint every N steps
- `--device`: Use `cpu` for RTX 2060, `cuda` for dedicated GPU

---

## 4. Monitor Training Progress

### View TensorBoard

```bash
# Start TensorBoard (from av_td3_system directory)
cd av_td3_system
tensorboard --logdir data/logs --port 6006

# Open in browser
# http://localhost:6006
```

**⚠️ Important: Data Availability Timeline**

TensorBoard data appears gradually as training progresses:

| Time | Step | Metrics Available |
|------|------|-------------------|
| **Now** | 0-100 | `progress/*` metrics (buffer, speed, reward) |
| **~30 min** | ~1,000 | First episode completes → `train/episode_*` metrics |
| **~2 hours** | 10,000 | Learning starts → `train/critic_loss`, `train/actor_loss` |
| **~2.5 hours** | 15,000 | First evaluation → `eval/*` metrics |

**Key metrics to watch:**
- `progress/buffer_size` - Buffer filling (0 → 10,000) ✅ **Available now**
- `progress/speed_kmh` - Vehicle speed during training ✅ **Available now**
- `progress/current_reward` - Step reward ✅ **Available now**
- `train/episode_reward` - Should increase over time ⏳ **After first episode**
- `train/critic_loss` - Should decrease ⏳ **After step 10,000**
- `eval/mean_reward` - Should improve at evaluations ⏳ **Every 5k steps**
- `eval/success_rate` - Should increase ⏳ **Every 5k steps**

**If TensorBoard is empty:**
1. ✅ Check training is running: `docker ps`
2. ✅ Wait ~5 minutes for first data to appear
3. ✅ Refresh browser (Ctrl+R or Cmd+R)
4. ✅ Verify correct directory: `ls -lh data/logs/TD3_scenario_0_*/`

---

### Monitor Live Training Logs

```bash
# Watch last 30 lines (updates every 5 seconds)
watch -n 5 'tail -30 training_log.log'

# Check current step
grep "Step.*/" training_log.log | tail -1

# Count completed steps
grep -c "Processing step" training_log.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

---

## 5. Expected Timeline

| Steps | Time (RTX 2060 + CPU) | Phase |
|-------|----------------------|-------|
| 0-150 | ~2 minutes | Initialization |
| 1-10,000 | ~1-2 hours | Exploration (random actions) |
| 10,001-30,000 | ~2-4 hours | Learning (policy updates) |
| **Total** | **~3-6 hours** | **30k steps** |

---

## 6. After Training

### Check Results

```bash
# View checkpoints
ls -lh data/checkpoints/

# View final results
cat data/logs/TD3_scenario_0_*/results.json

# Evaluate trained agent
python3 scripts/evaluate.py \
  --checkpoint data/checkpoints/td3_scenario_0_step_30000.pth \
  --scenario 0 \
  --num-episodes 10
```

---

## Troubleshooting

**Training seems stuck?**
- First `env.reset()` takes 1-5 minutes (spawning NPCs)
- Look for `[EXPLORATION] Processing step X/30,000...` logs

**CUDA out of memory?**
- Use `--device cpu` (CARLA needs GPU on RTX 2060)

**No logs appearing?**
- Check if CARLA server is running
- Verify Docker container is running: `docker ps`

**Need faster training?**
- Use university supercomputer with `--device cuda`
- Training will be 10-20x faster on dedicated GPU

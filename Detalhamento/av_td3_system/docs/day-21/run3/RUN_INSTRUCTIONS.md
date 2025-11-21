# Running TD3 Training After Logger Fix

## Problem
Docker containers may cache Python bytecode (`.pyc` files) and not pick up the latest code changes, even with volume mounts.

## Solution: Clear Cache and Run

### Step 1: Clear Python Cache
```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Remove all Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "✅ Python cache cleared"
```

### Step 2: Run Training with Fresh Code
```bash
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 500 \
    --eval-freq 5001 \
    --checkpoint-freq 5000 \
    --seed 42 \
    --debug \
    --device cpu 2>&1 | tee docs/day-21/run3/run_diagnostic_500steps.log
```

**Key Addition**: `-e PYTHONDONTWRITEBYTECODE=1` prevents Docker from creating new `.pyc` files

### Step 3: Verify Logger Works
Look for these in the output:
```
✅ No AttributeError
✅ Action statistics logged every 1000 steps (when buffer has data)
✅ TensorBoard metrics under debug/action_*
```

## Alternative: Rebuild Docker Image (If Above Doesn't Work)

If clearing cache doesn't help, the Docker image itself might have cached the old code:

```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Rebuild the Docker image
docker build -t td3-av-system:v2.0-python310 .

# Then run as normal
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 500 \
    --eval-freq 5001 \
    --seed 42 \
    --debug \
    --device cpu
```

## Verification Checklist

After running, verify:
- [ ] No `AttributeError: 'TD3TrainingPipeline' object has no attribute 'logger'`
- [ ] Console shows action statistics every 1000 steps (after step 1000)
- [ ] TensorBoard has 8 new metrics: `debug/action_steering_mean`, `debug/action_steering_std`, etc.
- [ ] Training continues past step 5000 without errors

## Expected Output Pattern

```
[LEARNING] Processing step   1000/500...
   ACTION STATISTICS (last 100 actions):
   ══════════════════════════════════════
   STEERING:
      Mean: +0.023  (expect ~0.0, no bias)
      Std:  0.245   (expect 0.1-0.3, exploration)
      Range: [-0.856, +0.891]
   ──────────────────────────────────────
   THROTTLE/BRAKE:
      Mean: +0.156  (expect 0.0-0.5, forward bias)
      Std:  0.312   (expect 0.1-0.3, exploration)
      Range: [-0.923, +0.987]
   ══════════════════════════════════════
```

## Troubleshooting

### Issue: Still Getting AttributeError
**Cause**: Volume mount isn't reflecting host changes  
**Solution**: Use `docker exec` to verify the file inside the container:
```bash
# Start container in background
docker run -d --name td3_test \
  -v $(pwd):/workspace \
  td3-av-system:v2.0-python310 \
  sleep 3600

# Check if logger line exists inside container
docker exec td3_test grep -n "self.logger = logging.getLogger" /workspace/scripts/train_td3.py

# Should show: 109:        self.logger = logging.getLogger(self.__class__.__name__)

# Clean up
docker stop td3_test
docker rm td3_test
```

If the grep **doesn't find the line**, the volume mount is not working. Check:
1. File permissions on host
2. Docker volume mount syntax
3. Try absolute path: `-v /absolute/path/to/av_td3_system:/workspace`

### Issue: Action Statistics Not Showing
**Cause**: Buffer needs 100 actions before stats are meaningful  
**Expected**: Stats appear after step 100 for TensorBoard, step 1000 for console

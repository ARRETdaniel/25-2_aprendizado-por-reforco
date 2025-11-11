# TD3 Validation Training - Quick Reference Card

**Goal:** Validate TD3 solution with 20k-step training run before full deployment

---

## 🚀 Quick Start (Copy-Paste These Commands)

### 1. Start CARLA Server (if not running)
```bash
docker run -d --name carla-server --rm \
  --network host --gpus all \
  -e SDL_VIDEODRIVER=offscreen \
  carlasim/carla:0.9.16 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen

# Wait 30 seconds
sleep 30
```

### 2. Run Validation Training (20k steps, ~2-3 hours)
```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

./scripts/run_validation_training.sh
```

### 3. Monitor Progress (in another terminal)
```bash
# Option A: Watch logs
tail -f data/logs/validation_training_20k_*.log

# Option B: TensorBoard
cd av_td3_system
tensorboard --logdir data/logs
# Open: http://localhost:6006
```

### 4. Analyze Results (after training completes)
```bash
python3 scripts/analyze_validation_run.py \
  --log-file data/logs/validation_training_20k_*.log \
  --output-dir data/validation_analysis

# View report
less data/validation_analysis/validation_report.txt

# View plots
xdg-open data/validation_analysis/validation_analysis.png
```

---

## ✅ Expected Behavior

### Phase 1: Exploration (Steps 1-10,000)
- **Vehicle:** Mostly stationary (0-5 km/h) ✓
- **Reward:** Consistently ~-53 (negative) ✓
- **Action:** Random (not learned) ✓
- **This is NORMAL** - filling replay buffer

### Phase 2: Learning (Steps 10,001-20,000)
- **Vehicle:** Speed increases (5 → 15+ km/h) ✓
- **Reward:** Improves (-53 → -20 or better) ✓
- **Action:** Learned policy (not random) ✓
- **This shows LEARNING** - agent improving

---

## 🛑 Red Flags (Stop Immediately If You See These)

❌ **Standing still gives POSITIVE reward** → Reward bug regression
❌ **Safety component POSITIVE when stationary** → Sign bug
❌ **CNN L2 norm constant (no variation)** → Degenerate features
❌ **All waypoints behind vehicle (x < 0)** → Coordinate bug
❌ **Training crashes repeatedly** → Configuration issue

---

## 📊 Key Metrics to Watch (TensorBoard)

| Metric | Expected Trend | What It Means |
|--------|----------------|---------------|
| `train/episode_reward` | ↗ Increasing | Agent learning to navigate |
| `progress/speed_kmh` | ↗ Increasing | Vehicle moving faster |
| `train/critic_loss` | ↘ Decreasing | Networks converging |
| `eval/success_rate` | ↗ Increasing | More goals reached |
| `train/collisions_per_episode` | ↘ Decreasing | Safer navigation |

---

## 📁 Output Files

After completion, you'll have:

```
data/
├── logs/
│   ├── validation_training_20k_YYYYMMDD_HHMMSS.log  ← Training log
│   └── TD3_scenario_0_npcs_20_YYYYMMDD_HHMMSS/      ← TensorBoard events
├── checkpoints/
│   ├── td3_scenario_0_step_5000.pth                 ← Checkpoint 1
│   ├── td3_scenario_0_step_10000.pth                ← Checkpoint 2
│   ├── td3_scenario_0_step_15000.pth                ← Checkpoint 3
│   └── td3_scenario_0_step_20000.pth                ← Checkpoint 4 (final)
└── validation_analysis/
    ├── validation_report.txt                        ← Pass/fail report
    └── validation_analysis.png                      ← Visualization plots
```

---

## ✅ Decision Tree

### IF validation_report.txt shows "PASS" ✅
→ **Proceed with full 1M-step training on supercomputer**

### IF validation_report.txt shows "FAIL" 🛑
→ **DO NOT proceed**
→ Fix issues in report
→ Re-run validation
→ Only proceed after PASS

---

## 🐛 Quick Troubleshooting

**Problem:** CARLA connection timeout
**Fix:** `docker restart carla-server && sleep 30`

**Problem:** Docker image not found
**Fix:** `docker build -t td3-av-system:v2.0-python310 -f docker/Dockerfile .`

**Problem:** Out of memory
**Fix:** Use `--device cpu` instead of `cuda` in script

**Problem:** Reward still positive when stationary
**Fix:** Check `config/training_config.yaml` → `safety.weight` must be **+100.0**

---

## 📞 Help

**Full Guide:** `docs/VALIDATION_TRAINING_GUIDE.md`
**Technical Details:** `docs/REWARD_FUNCTION_VALIDATION_ANALYSIS.md`
**Task Tracking:** `docs/TODO.md`

---

**Version:** 1.0
**Date:** October 26, 2024
**Estimated Time:** ~2-3 hours
**Ready to Execute:** ✅

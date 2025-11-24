# Quick Reference: info Dict Enhancement

## What Changed?

**File:** `src/environment/carla_env.py`
**Method:** `step()`
**Line:** ~787

### Before
```python
info = {
    "reward_breakdown": reward_dict["breakdown"],  # Only this
    # ...
}
```

### After
```python
info = {
    "reward_breakdown": reward_dict["breakdown"],  # Kept
    "reward_components": {  # NEW: Simple format
        "total": reward,
        "efficiency": ...,
        "lane_keeping": ...,
        # ...
    },
    "state": {  # NEW: Metrics
        "velocity": ...,
        "lateral_deviation": ...,
        # ...
    },
    # ...
}
```

---

## Why?

| Reason | Explanation |
|--------|-------------|
| **Official Standard** | Gymnasium docs explicitly recommend including reward components in `info` dict |
| **Scientific Validation** | Papers require evidence that reward function works correctly |
| **Bug Detection** | Catches issues like components not summing to total |
| **Manual Testing** | Enables real-time HUD display during manual control |
| **Paper Figures** | Generates data for component comparison plots |
| **Reproducibility** | Logs exact calculations for peer review |

---

## How to Use

### In Manual Validation Script

```python
obs, reward, term, trunc, info = env.step(action)

# ✅ Access components (simple!)
efficiency = info['reward_components']['efficiency']
velocity = info['state']['velocity']

# ✅ Display on HUD
print(f"Efficiency: {efficiency} | Speed: {velocity} km/h")

# ✅ Validate summation
components = info['reward_components']
calculated = sum([components[k] for k in components if k != 'total'])
assert abs(calculated - reward) < 0.001
```

### In Analysis Script

```python
# Load logged data
with open('validation_logs/session_01/reward_validation_*.json') as f:
    snapshots = json.load(f)

# Extract time series
lateral_dev = [s['lateral_deviation'] for s in snapshots]
lane_keep = [s['lane_keeping_reward'] for s in snapshots]

# Analyze correlation
r = pearson(lateral_dev, lane_keep)
print(f"Correlation: {r:.3f}")  # Should be negative

# Generate plot
plt.scatter(lateral_dev, lane_keep)
plt.xlabel('Lateral Deviation (m)')
plt.ylabel('Lane Keeping Reward')
plt.savefig('correlation.png')
```

### In Paper

```latex
\subsection{Reward Function Validation}

We validated the reward function using manual control sessions
(n=15 scenarios, 1,247 steps). Statistical analysis confirmed:

\begin{itemize}
    \item Lane keeping correlates with deviation ($r=-0.85$, $p<0.001$)
    \item Components sum correctly (residual $<10^{-4}$)
    \item Safety penalties activate reliably (100\% detection)
\end{itemize}
```

---

## Format Reference

### reward_components (NEW)

**Type:** `Dict[str, float]`

**Keys:**
- `"total"`: Total reward (should equal env.step() return value)
- `"efficiency"`: Speed tracking component
- `"lane_keeping"`: Lane centering component
- `"comfort"`: Smooth driving component
- `"safety"`: Collision/off-road penalties
- `"progress"`: Waypoint advancement

**Example:**
```python
{
    "total": -0.0845,
    "efficiency": 0.0245,
    "lane_keeping": -0.0012,
    "comfort": -0.0078,
    "safety": 0.0000,
    "progress": 0.0100
}
```

### state (NEW)

**Type:** `Dict[str, float]`

**Keys:**
- `"velocity"`: Current speed (km/h)
- `"lateral_deviation"`: Distance from lane center (m)
- `"heading_error"`: Angle difference from lane heading (rad)
- `"distance_to_goal"`: Remaining distance to goal (m)

**Example:**
```python
{
    "velocity": 28.5,
    "lateral_deviation": 0.15,
    "heading_error": 0.02,
    "distance_to_goal": 450.3
}
```

### reward_breakdown (EXISTING)

**Type:** `Dict[str, Tuple[float, float, float]]`

**Format:** `{component: (weight, raw_value, weighted_contribution)}`

**Example:**
```python
{
    "efficiency": (0.5, 0.049, 0.0245),
    "lane_keeping": (0.3, -0.004, -0.0012),
    # ...
}
```

**Notes:**
- Index [0]: Weight parameter (from config)
- Index [1]: Raw normalized value (before weighting)
- Index [2]: Weighted contribution (this goes into `reward_components`)

---

## Validation Checklist

Before proceeding to training, verify:

- [ ] Unit tests pass (`test_reward_components.py`)
- [ ] Manual validation completed (drive through scenarios)
- [ ] Components sum to total (residual < 0.001)
- [ ] Correlations match expectations:
  - [ ] Lane keeping ↔ lateral deviation: r < -0.7
  - [ ] Efficiency ↔ speed: peaks near target (30 km/h)
  - [ ] Safety penalties activate in unsafe scenarios
- [ ] No critical issues in analysis report
- [ ] Validation logs archived for paper

---

## Common Issues

### Issue: KeyError 'reward_components'

**Cause:** Old version of `carla_env.py` loaded (cache)

**Fix:**
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Verify modification
grep -A 5 "reward_components" src/environment/carla_env.py
```

### Issue: Components don't sum to total

**Cause:** Bug in reward calculation logic

**Diagnosis:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation
obs, reward, term, trunc, info = env.step(action)

# Check breakdown
print("Breakdown:", info['reward_breakdown'])
print("Components:", info['reward_components'])
print("Total:", reward)
```

**Fix:** Review `src/environment/reward_functions.py` calculation

### Issue: PyGame window not showing

**Cause:** X11 forwarding not enabled

**Fix:**
```bash
# Enable X11
xhost +local:docker

# Test with simple app
docker run --rm -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ubuntu xeyes  # Should show window
```

---

## Next Steps

### 1. Quick Test (15 min)

```bash
# Start CARLA
docker run -d --name carla-server --runtime=nvidia --net=host \
    carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen

# Run unit tests
docker run --rm --network host --runtime nvidia \
  -v $(pwd):/workspace -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/test_reward_components.py
```

### 2. Manual Validation (2-3 hours)

```bash
# Run manual control
xhost +local:docker
docker run --rm --network host --runtime nvidia \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/validate_rewards_manual.py \
    --config config/baseline_config.yaml \
    --output-dir validation_logs/session_01
```

### 3. Analysis (30 min)

```bash
# Analyze results
docker run --rm --network host --runtime nvidia \
  -v $(pwd):/workspace -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/analyze_reward_validation.py \
    --log validation_logs/session_01/reward_validation_*.json \
    --output-dir validation_logs/session_01/analysis
```

### 4. Review Report

```bash
# Check for issues
cat validation_logs/session_01/analysis/validation_report_*.md

# View plots
eog validation_logs/session_01/analysis/*.png
```

### 5. Proceed to Training (Only After Validation)

```bash
# Start TD3 training
docker run --rm --network host --runtime nvidia \
  -v $(pwd):/workspace -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 100000 \
    --eval-freq 20000
```

---

## Documentation

### Generated
- `docs/day-23/manual-evaluation/NEXT_STEPS_REWARD_VALIDATION.md` - Complete workflow
- `docs/day-23/manual-evaluation/WHY_INFO_DICT_ENHANCEMENT.md` - Detailed explanation
- `docs/day-23/manual-evaluation/ARCHITECTURE_DIAGRAM.md` - Visual architecture
- `docs/day-23/manual-evaluation/QUICK_REFERENCE.md` - This document

### Related
- `docs/README_REWARD_VALIDATION.md` - Quick start guide
- `docs/reward_validation_guide.md` - Complete validation guide
- `docs/REWARD_VALIDATION_SUMMARY.md` - Technical summary

### Scripts
- `scripts/validate_rewards_manual.py` - Manual control interface
- `scripts/analyze_reward_validation.py` - Analysis tool
- `scripts/test_reward_components.py` - Unit tests

---

## Official References

- **Gymnasium step() API**: https://gymnasium.farama.org/api/env/#gymnasium.Env.step
- **CARLA 0.9.16 Docs**: https://carla.readthedocs.io/en/latest/
- **TD3 Paper**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods"

---

**Status:** ✅ Code modification complete, ready for testing

**Quick Start:**
```bash
# Test CARLA connection
docker logs carla-server

# Run unit tests
cd av_td3_system && \
docker run --rm --network host --runtime nvidia \
  -v $(pwd):/workspace -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/test_reward_components.py
```

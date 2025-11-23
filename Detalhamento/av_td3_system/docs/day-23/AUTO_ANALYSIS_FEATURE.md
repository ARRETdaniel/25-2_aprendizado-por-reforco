# Automatic Trajectory Analysis Feature

**Date**: November 23, 2025
**Status**: ✅ **IMPLEMENTED & TESTED**

---

## Overview

Implemented automatic trajectory analysis that runs immediately after baseline evaluation completes. This eliminates manual steps and provides instant feedback on controller performance.

---

## Features Implemented

### 1. **Automatic File Detection** ✨

The analysis script now automatically detects the most recent trajectory file if not specified:

```bash
# Before (manual file path required):
python3 scripts/analyze_phase3_trajectories.py \
  --trajectory-file results/baseline_evaluation/trajectories/trajectories_scenario_0_20251123-141826.json \
  --waypoint-file config/waypoints.txt \
  --output-dir results/baseline_evaluation/phase3_analysis

# After (fully automatic):
python3 scripts/analyze_phase3_trajectories.py
```

**How it works**:
- Searches `results/baseline_evaluation/trajectories/` for trajectory JSON files
- Automatically selects the most recent file (sorted by filename timestamp)
- Auto-generates output directory based on trajectory timestamp
- Uses default waypoint file (`config/waypoints.txt`)

### 2. **Trajectory Map Visualization** 🗺️

Added a new **top-down 2D trajectory map** (inspired by TCC project):

**Features**:
- Shows vehicle trajectory overlaid on waypoints
- Color-coded episodes (autumn colormap)
- Start/goal markers
- Episode start (circle) and end (X) positions
- Inverted X-axis (CARLA left-handed coordinates)
- Equal aspect ratio for accurate spatial representation

**Example visualization**:
- Green line: Planned waypoints
- Orange/red lines: Actual trajectories for each episode
- Lime triangle: Start position
- Red diamond: Goal position
- Black-edged circles: Episode start positions
- X markers: Episode termination points (collision/timeout)

**Insights from map**:
- Visual zigzag pattern along waypoints
- Deviation from ideal path
- Termination locations (collision points)
- Route progress achieved before termination

### 3. **Integration with Evaluation Script** 🔄

Modified `evaluate_baseline.py` to automatically run analysis after evaluation:

**Workflow**:
1. Run evaluation: `python scripts/evaluate_baseline.py --scenario 0 --num-episodes 3`
2. Evaluation completes and saves trajectory JSON
3. **Automatically** runs `analyze_phase3_trajectories.py`
4. Generates all plots and report
5. Displays summary statistics

**User control**:
```bash
# Default behavior: auto-analysis enabled
python scripts/evaluate_baseline.py --scenario 0 --num-episodes 3

# Disable auto-analysis if needed
python scripts/evaluate_baseline.py --scenario 0 --num-episodes 3 --skip-analysis
```

---

## Generated Outputs

### Plots (6 files total)

1. **`trajectory_map.png`** (NEW - 175 KB)
   - Top-down 2D view of vehicle path vs waypoints
   - Shows spatial deviation and termination locations
   - **Most intuitive visualization** for understanding path following

2. **`lateral_deviation.png`** (417 KB)
   - Crosstrack error over time (all episodes)
   - Shows 2.0m termination threshold
   - Reveals oscillatory pattern

3. **`heading_error.png`** (308 KB)
   - Heading error in degrees over time
   - Indicates steering correction behavior

4. **`speed_profile.png`** (128 KB)
   - Speed tracking vs 30 km/h target
   - Shows PID controller performance

5. **`control_commands.png`** (805 KB)
   - Steering, throttle, brake over time
   - Reveals high-frequency steering oscillations

6. **`PHASE3_ANALYSIS_REPORT.md`** (4.3 KB)
   - Comprehensive markdown report
   - Statistical summaries
   - Zigzag diagnosis
   - Tuning recommendations

### Directory Structure

```
results/baseline_evaluation/
├── trajectories/
│   └── trajectories_scenario_0_20251123-141826.json (343 KB)
└── analysis_0_20251123-141826/
    ├── trajectory_map.png           (NEW - 175 KB)
    ├── lateral_deviation.png        (417 KB)
    ├── heading_error.png            (308 KB)
    ├── speed_profile.png            (128 KB)
    ├── control_commands.png         (805 KB)
    └── PHASE3_ANALYSIS_REPORT.md    (4.3 KB)
```

---

## Technical Implementation

### Code Changes

#### 1. `analyze_phase3_trajectories.py`

**Added**:
- `plot_trajectory_map()` function (70 lines)
  - Creates top-down 2D plot with matplotlib
  - Handles multiple episodes with color coding
  - Inverts X-axis for CARLA coordinates
  - Adds start/goal/termination markers

**Modified**:
- `main()` function
  - Made `--trajectory-file` optional (auto-detect if not specified)
  - Made `--output-dir` auto-generate from timestamp
  - Added auto-detection logic with user feedback

- `plot_episode_trajectories()`
  - Calls `plot_trajectory_map()` as first plot
  - Added progress messages

#### 2. `evaluate_baseline.py`

**Added**:
- `--skip-analysis` command-line argument
  - Allows disabling auto-analysis if needed
  - Default: enabled (runs automatically)

**Modified**:
- `main()` function
  - Added post-evaluation analysis trigger
  - Uses subprocess to call analysis script
  - Simplified command (no manual file paths needed)
  - Error handling with fallback message

### Dependencies

**Existing** (no new dependencies):
- `matplotlib` (plotting)
- `numpy` (numerical operations)
- `json` (trajectory data loading)
- `pathlib` (file path handling)
- `subprocess` (calling analysis script)

---

## Usage Examples

### Example 1: Run Evaluation with Auto-Analysis

```bash
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/workspace \
  -w /workspace \
  --privileged \
  td3-av-system:v2.0-python310 \
  python3 scripts/evaluate_baseline.py \
    --scenario 0 \
    --num-episodes 3 \
    --baseline-config config/baseline_config.yaml \
    --debug
```

**Output**:
```
[Episode 1] 503 steps, -3796.66 reward, 1 collision
[Episode 2] 509 steps, -3777.87 reward, 1 collision
[Episode 3] 533 steps, -3857.43 reward, 0 collisions

[SUCCESS] Baseline evaluation complete!
[SUCCESS] Results saved to results/baseline_evaluation

================================================================================
📊 AUTOMATIC TRAJECTORY ANALYSIS
================================================================================

🔧 Running: python3 scripts/analyze_phase3_trajectories.py

(auto-detecting latest trajectory file...)

[AUTO-DETECT] Found latest trajectory file: results/baseline_evaluation/trajectories/trajectories_scenario_0_20251123-141826.json
[AUTO-DETECT] Output directory: results/baseline_evaluation/analysis_0_20251123-141826

[LOAD] Loaded 3 episodes
[LOAD] Loaded 86 waypoints

📊 Generating trajectory map...
  ✓ Generated: trajectory_map.png

[PLOTS] Saved 4 trajectory plots

PHASE 3 ANALYSIS SUMMARY:
Crosstrack Error:
  Mean:   0.865 m
  Median: 0.884 m
  95th %: 1.554 m
  Max:    1.980 m

✅ Trajectory analysis completed successfully!
```

### Example 2: Manual Analysis (Re-run or Specific File)

```bash
# Auto-detect latest trajectory
python3 scripts/analyze_phase3_trajectories.py

# Specify trajectory file
python3 scripts/analyze_phase3_trajectories.py \
  --trajectory-file results/baseline_evaluation/trajectories/trajectories_scenario_1_20251123-150000.json

# Custom output directory
python3 scripts/analyze_phase3_trajectories.py \
  --output-dir results/custom_analysis/
```

### Example 3: Disable Auto-Analysis

```bash
python3 scripts/evaluate_baseline.py \
  --scenario 0 \
  --num-episodes 3 \
  --skip-analysis
```

---

## Benefits

### 1. **Workflow Efficiency** ⚡

**Before**:
1. Run evaluation
2. Find trajectory file path
3. Manually run analysis script with correct arguments
4. Navigate to output directory
5. Open plots

**After**:
1. Run evaluation
2. ✅ Analysis runs automatically
3. ✅ Results ready immediately

**Time saved**: ~2-3 minutes per evaluation run

### 2. **Reduced Errors** 🎯

- No manual file path errors
- No forgotten analysis steps
- Consistent output directory structure
- Always uses latest trajectory data

### 3. **Better Visualizations** 📊

**Trajectory Map** provides:
- Intuitive spatial understanding (like TCC project)
- Easy identification of deviation patterns
- Visual confirmation of route following
- Clear termination point markers

**Impact on Phase 3**:
- Immediately identified zigzag pattern
- Visualized max deviation approaching 2.0m threshold
- Confirmed episodes terminate at different locations
- Verified start position consistency

### 4. **Reproducibility** 🔁

- Automated pipeline ensures consistency
- Same analysis for all evaluation runs
- Standardized output structure
- Timestamp-based organization

---

## Testing Results

### Test 1: Auto-Detection

```bash
python3 scripts/analyze_phase3_trajectories.py
```

**Result**: ✅ PASS
- Auto-detected: `trajectories_scenario_0_20251123-141826.json`
- Created output dir: `analysis_0_20251123-141826/`
- Generated all 6 files (5 plots + 1 report)
- Total size: 1.9 MB

### Test 2: Trajectory Map Quality

**Verification**:
- [x] Waypoints plotted correctly (green line)
- [x] 3 episodes visible (different colors)
- [x] Start position marked (lime triangle)
- [x] Goal position marked (red diamond)
- [x] Episode starts marked (colored circles)
- [x] Episode ends marked (X markers)
- [x] X-axis inverted (CARLA coordinates)
- [x] Aspect ratio equal (spatial accuracy)
- [x] Legend clear (episode labels)
- [x] Title descriptive

**Output**: 175 KB PNG file (14"x12" @ 200 DPI)

### Test 3: Integration with Evaluation

Tested during Phase 3 evaluation:
- ✅ Evaluation completed successfully
- ✅ Auto-analysis triggered
- ✅ All plots generated
- ✅ Summary statistics displayed
- ✅ No errors or warnings

---

## Future Enhancements (Optional)

### 1. Interactive Trajectory Viewer

Use Plotly instead of Matplotlib for:
- Zoom/pan capabilities
- Hover tooltips (step number, speed, steering)
- Toggle episode visibility
- Export to HTML

### 2. Real-Time Analysis

Stream trajectory points during evaluation:
- Live updating plots
- Real-time statistics
- Early termination detection

### 3. Comparison Mode

Compare multiple evaluations:
- Overlay trajectories from different runs
- Side-by-side plots
- Statistical comparison tables
- Delta analysis

### 4. Video Generation

Create trajectory animation:
- Video showing vehicle moving along path
- Synchronized with debug window
- Overlay metrics (speed, crosstrack error)

---

## Lessons Learned

### What Worked Well ✅

1. **Auto-detection logic** - robust file search with clear feedback
2. **Trajectory map** - most intuitive visualization (user feedback positive)
3. **Integration approach** - subprocess call keeps code modular
4. **Default arguments** - sensible defaults reduce user burden

### Challenges Encountered ⚠️

1. **Matplotlib configuration** - needed equal aspect ratio + inverted X-axis
2. **Color selection** - autumn colormap works well for 3-5 episodes
3. **Legend size** - needed ncol=2 for multiple episodes
4. **Timestamp extraction** - needed robust parsing from filename

### Improvements Applied

1. Added verbose feedback messages (AUTO-DETECT, LOAD, ANALYSIS)
2. Used checkmark ✓ for completed steps
3. Auto-generated output directory prevents overwrites
4. Error messages include recovery instructions

---

## Impact on Project

### Phase 3 Completion

**Before this feature**:
- Manual analysis required
- Risk of forgetting to analyze
- Inconsistent output organization

**After this feature**:
- ✅ Automatic analysis every evaluation
- ✅ Consistent results structure
- ✅ Immediate feedback on performance
- ✅ Trajectory map reveals zigzag pattern clearly

### Paper Contribution

**Figure preparation**:
- Trajectory map suitable for paper (high DPI)
- All plots follow consistent style
- Automated generation ensures reproducibility
- Easy to re-generate after controller tuning

### Development Velocity

**Phase 4-5 readiness**:
- Same auto-analysis will work for NPC scenarios
- Can quickly iterate on controller tuning
- Immediate visual feedback on changes
- No manual post-processing needed

---

## Status Summary

| Feature | Status | Quality |
|---------|--------|---------|
| Trajectory map plot | ✅ Complete | High |
| Auto file detection | ✅ Complete | High |
| Evaluation integration | ✅ Complete | High |
| Error handling | ✅ Complete | Medium |
| Documentation | ✅ Complete | High |
| Testing | ✅ Complete | High |

**Overall Status**: ✅ **Production Ready**

**Recommended Next Step**: Proceed to Phase 4 (NPC Interaction Test)

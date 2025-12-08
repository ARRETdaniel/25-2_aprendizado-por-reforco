# Visual Control Flow Diagram

**Date:** December 3, 2025  
**Purpose:** Clear visualization of who controls the CARLA vehicle

---

## THE ANSWER IN ONE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│              WHO CONTROLS THE CAR? → TD3 ACTOR NETWORK                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


                          🚗 CARLA VEHICLE
                                  ↑
                                  │ VehicleControl
                                  │ (throttle, brake, steer)
                                  │
                        ┌─────────┴─────────┐
                        │                   │
                        │   CARLA ENV       │
                        │   _apply_action() │
                        │                   │
                        └─────────┬─────────┘
                                  ↑
                                  │ action[2]
                                  │ [steering, throttle/brake]
                                  │ Range: [-1, +1]²
                                  │
        ┌─────────────────────────┴──────────────────────────┐
        │                                                     │
        │           TD3 ACTOR NETWORK (MLP)                   │
        │           ═══════════════════════                   │
        │                                                     │
        │  Input: state (565-dim)                            │
        │    ↓                                               │
        │  FC1: Linear(565, 256) + ReLU                      │
        │    ↓                                               │
        │  FC2: Linear(256, 256) + ReLU                      │
        │    ↓                                               │
        │  FC3: Linear(256, 2) + Tanh                        │
        │    ↓                                               │
        │  Output: action (2-dim) ← THIS CONTROLS THE CAR!   │
        │          [steering, throttle/brake]                │
        │                                                     │
        └─────────────────────────┬──────────────────────────┘
                                  ↑
                                  │ state[565]
                                  │ = CNN features + vector obs
                                  │
                        ┌─────────┴─────────┐
                        │                   │
                        │   CONCATENATION   │
                        │                   │
                        └─────────┬─────────┘
                                  ↑
                    ┌─────────────┴─────────────┐
                    │                           │
        ┌───────────┴───────────┐   ┌───────────┴──────────┐
        │                       │   │                      │
        │  CNN FEATURE EXTRACTOR│   │   VECTOR OBS (53)    │
        │  (NatureCNN)          │   │                      │
        │                       │   │  - velocity (1)      │
        │  Conv1 (8×8, s=4)     │   │  - lat_dev (1)       │
        │  LayerNorm + ReLU     │   │  - heading_err (1)   │
        │  Conv2 (4×4, s=2)     │   │  - waypoints (50)    │
        │  LayerNorm + ReLU     │   │                      │
        │  Conv3 (3×3, s=1)     │   │  Total: 53-dim       │
        │  LayerNorm + ReLU     │   │                      │
        │  Flatten + Linear     │   │                      │
        │                       │   │                      │
        │  Output: 512 features │   │                      │
        │  (INTERNAL STATE)     │   │                      │
        │  NOT CONTROL!         │   │                      │
        │                       │   │                      │
        └───────────┬───────────┘   └──────────────────────┘
                    ↑
                    │ image[4,84,84]
                    │
          ┌─────────┴─────────┐
          │                   │
          │   CARLA CAMERA    │
          │   (Front View)    │
          │                   │
          └───────────────────┘


═══════════════════════════════════════════════════════════════════════

KEY INSIGHT:

  ❌ CNN does NOT control the car
     CNN outputs: 512-dim feature vector (INTERNAL REPRESENTATION)
     
  ✅ ACTOR controls the car
     Actor outputs: 2-dim action vector (CONTROL COMMAND)
     
═══════════════════════════════════════════════════════════════════════
```

---

## SIMPLIFIED FLOW

```
Camera Image (4×84×84)
    ↓
CNN Feature Extractor
    ↓
512 Features ──────────┐
                       │
Kinematic + Waypoints  │ ← Concatenate to 565-dim state
(53-dim) ──────────────┘
    ↓
TD3 Actor Network (MLP)
    ↓
Action (2-dim) ← THIS IS WHAT CARLA RECEIVES!
    ↓
CARLA Vehicle Control
```

---

## WHAT EACH COMPONENT DOES

### 1. CNN (NatureCNN)
- **Role:** Feature Extractor
- **Input:** Camera image (4, 84, 84)
- **Output:** 512-dimensional feature vector
- **Purpose:** Convert visual input into compact representation
- **Does it control car?** ❌ NO! It only processes images

### 2. Actor Network (MLP)
- **Role:** Policy / Controller
- **Input:** State (565-dim = 512 CNN + 53 vector)
- **Output:** Action (2-dim = [steering, throttle/brake])
- **Purpose:** Decide what action to take based on current state
- **Does it control car?** ✅ YES! Its output is sent to CARLA

### 3. CARLA Environment
- **Role:** Simulator / Executor
- **Input:** Action (2-dim from Actor)
- **Output:** Next observation + reward
- **Purpose:** Apply action to vehicle and return new state
- **Does it control car?** ⚙️ EXECUTES commands from Actor

---

## WHY THE CONFUSION?

### Log Interpretation Problem

**What you see in logs:**
```
[DEBUG] CNN L2 Norm: 1245.011
[INFO] Action: [+0.994, +1.000]
```

**What you might think:**
"CNN outputs 1245 → Car receives [0.994, 1.000] → CNN controls car"

**What actually happens:**
```
1. CNN extracts 512 features with L2 norm = 1245
2. Features concatenated with vector obs → 565-dim state
3. Actor MLP processes state → outputs [0.994, 1.000]
4. CARLA receives [0.994, 1.000] from Actor (NOT CNN!)
```

**The logs show TWO SEPARATE things:**
- CNN L2 norm = magnitude of INTERNAL features
- Action = CONTROL COMMAND from Actor

---

## TRACING THE PROBLEM

### Why Agent Outputs Hard Turns

```
Step 1: CNN weights grow unbounded (no weight decay)
           ↓
Step 2: CNN features explode (L2 norm: 15 → 1200)
           ↓
Step 3: Actor receives HUGE input values
           ↓
Step 4: Actor activations saturate (tanh → ±1)
           ↓
Step 5: Actor outputs saturate ([+0.994, +1.000])
           ↓
Step 6: CARLA receives saturated control
           ↓
Result: Hard right turn + full throttle (STUCK!)
```

**Root cause:** CNN explosion (not Actor malfunction!)  
**Fix:** Weight decay prevents CNN explosion → Actor gets reasonable inputs → Outputs become diverse

---

## THE FIX (Already Implemented)

### Weight Decay 1e-4

**What it does:**
```python
# In optimizer update:
Loss = Loss_actor + weight_decay * ||W_cnn||²

# Gradient includes weight decay term:
∇Loss = ∇Loss_actor + 2 * weight_decay * W_cnn

# Weights shrink each update:
W_new = W_old - lr * (∇Loss_actor + 2*weight_decay*W_old)
```

**Expected result:**
```
Before fix:
  CNN L2: 15 → 1200 (explosion!)
  Action: diverse → saturated (stuck!)
  Behavior: normal → hard turns only

After fix:
  CNN L2: 15 → 100 (stable!)
  Action: diverse → diverse (learning!)
  Behavior: normal → smooth driving
```

---

## VALIDATION CHECKLIST

After running training with weight_decay=1e-4:

- [ ] CNN L2 norm stays below 150 (currently ~1200)
- [ ] Actions are diverse (not stuck at ±1.0)
- [ ] Episode length increases (currently ~27 steps)
- [ ] Agent learns smooth steering (no hard turns)
- [ ] Success rate improves (currently ~0%)

**If all checked → Fix successful! 🎉**

---

## CONCLUSION

### The Direct Answer

**Q: Who controls the car?**  
**A: TD3 Actor Network (MLP with 2 hidden layers)**

**Q: What does CNN do?**  
**A: Extracts 512 visual features (internal representation, NOT control)**

**Q: Why hard turns?**  
**A: CNN features explode → Actor saturates → Outputs stuck at maximum**

**Q: Will weight decay fix it?**  
**A: Yes. Prevents CNN explosion → Actor gets healthy inputs → Outputs become diverse**

---

**Document Version:** 1.0  
**Created:** December 3, 2025  
**Purpose:** Clear visual explanation of control flow

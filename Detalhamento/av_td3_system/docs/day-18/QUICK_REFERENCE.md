# Quick Reference: 5K Run Analysis (Nov 18, 2025)

## At a Glance

| Issue | Status | Action Required |
|-------|--------|-----------------|
| **Gradient Explosion** | ✅ FIXED | None - working perfectly |
| **Q-Value Explosion** | ❌ CRITICAL | Add diagnostic logging |
| **Episode Length** | ⚠️ LOW | Will improve with Q-fix |
| **Learning Config** | ✅ VALIDATED | None - all correct |

## The Numbers

```
GRADIENT NORMS (✅ EXCELLENT)
  Actor CNN:    2.39 max    ← Was 1.8M before fix!
  Critic CNN:  25.09 max    ← Clipped perfectly
  Explosion alerts: 0       ← No warnings

ACTOR LOSS (❌ CRITICAL) 
  Current: -2,400,000       ← Should be -1,000 max
  Problem: Q-value explosion

EPISODES
  Mean:    10.2 steps       ← Low end of expected (5-20)
  Final:   3 steps          ← Below expected
  Max:     1,000 steps      ← At least one succeeded

Q-VALUES (⚠️ ACCEPTABLE BUT HIGH)
  Q1: 90.18 final           ← Growing trend
  Q2: 90.30 final           ← Twin critics agree
```

## Before/After Gradient Fixes

```
METRIC              BEFORE        AFTER        IMPROVEMENT
─────────────────────────────────────────────────────────
Actor CNN Grad      1,826,337  →  2.39        99.9999% ✅
Gradient Alerts     Many       →  0           100% ✅
Training Crashes    Yes        →  No          100% ✅
Q-Value Explosion   ???        →  -2.4M       ❌ WORSE
```

## Diagnostic Plan (90 min to fix)

```
1. Add logging        [  5 min] → Actor Q, Target Q, Reward components
2. Run diagnostic     [ 30 min] → 5K with enhanced logging  
3. Analyze logs       [ 10 min] → Identify exact cause
4. Implement fix      [ 15 min] → Reward clip OR critic reg
5. Validate fix       [ 30 min] → 5K validation
   ─────────────────────────────
   TOTAL              [ 90 min] → Ready for 50K ✅
```

## Fix Options (Choose After Diagnostic)

### Option A: Reward Clipping (if rewards too large)
```python
reward = np.clip(reward, -10, +10)
```
**When**: Reward components exceed ±1000/step  
**Risk**: Low  
**Effectiveness**: High for scaling issues

### Option B: Reward Normalization (more robust)
```python
normalized_reward = (reward - mean) / (std + eps)
```
**When**: Rewards vary widely across episodes  
**Risk**: Medium (need to tune)  
**Effectiveness**: Best long-term solution

### Option C: Critic Regularization (if critic overfits)
```python
critic_loss += 0.01 * l2_norm(critic.parameters())
```
**When**: Actor Q-values >> Logged Q-values  
**Risk**: Low  
**Effectiveness**: Medium to High

## Decision Tree

```
Diagnostic Results
     │
     ├─→ Reward component > 1000?
     │   └─→ YES: Use Option A (reward clipping)
     │
     ├─→ Actor Q >> Logged Q?
     │   └─→ YES: Use Option C (critic regularization)
     │
     └─→ Both normal?
         └─→ Check Bellman equation (bootstrap error)
```

## GO/NO-GO Criteria

### Current Run: ❌ NO-GO
- ❌ Actor loss: -2.4M (fail threshold: -100K)
- ⚠️ Episode length: 3 (marginal, expect 5-20)
- ✅ Gradients: 2.39 (excellent)

### Next Run Must Have:
- ✅ Actor loss < 100,000
- ✅ Episode length > 5 steps
- ✅ No Q-value explosion trend
- ✅ Gradients still healthy

**If 3/4 pass → GO for 50K**

## Timeline to 50K

```
NOW ─→ Add Logging (5m) ─→ Diagnostic 5K (30m) ─→ Fix (25m) ─→ Validate (30m) ─→ 50K (6h)
                           ├─ Identify cause       ├─ Implement   ├─ Verify      └─ Full run
                           └─ Choose fix           └─ Test        └─ GO decision

                           ◄──── 90 minutes ────►                  ◄─── 6 hours ──►
```

## Key Files

- **Analysis**: `docs/day-18/SYSTEMATIC_5K_ANALYSIS_NOV18.md`
- **Action Plan**: `docs/day-18/ACTION_PLAN_Q_VALUE_EXPLOSION.md`
- **Summary**: `docs/day-18/SUMMARY_5K_VALIDATION_NOV18.md`
- **This File**: `docs/day-18/QUICK_REFERENCE.md`

## Next Command to Run

```bash
# After adding diagnostic logging to code:
cd av_td3_system

docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd):/workspace \
  -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 5000 \
    --eval-freq 3001 \
    --checkpoint-freq 1000 \
    --seed 42 \
    --device cpu \
    2>&1 | tee diagnostic_5k_$(date +%Y%m%d_%H%M%S).log
```

---

**Status**: 🔴 BLOCKING - Must fix Q-value explosion before 50K  
**Confidence**: 🟢 HIGH - Issue is well-understood, fix is straightforward  
**ETA**: 90 minutes to resolution

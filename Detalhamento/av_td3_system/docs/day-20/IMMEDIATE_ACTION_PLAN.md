# IMMEDIATE ACTION PLAN: Fix TD3 CNN Training

**Date**: November 20, 2025  
**Status**: 🔴 **CRITICAL ISSUES - SYSTEM BROKEN**  
**Next Action**: Debug gradient clipping + fix hyperparameters

---

## Executive Summary

**❌ ANSWER: NO - More steps will NOT help. System has fundamental bugs that must be fixed first.**

### Critical Issues Identified

1. **🔥 GRADIENT CLIPPING FAILURE** (Blocks ALL learning)
   - Actor CNN: 2.42 (should be ≤1.0) - **242% OVER LIMIT**
   - Critic CNN: 24.69 (should be ≤10.0) - **247% OVER LIMIT**
   - Code LOOKS correct but doesn't work → requires debugging

2. **🔥 HYPERPARAMETERS WRONG** (Amplifies gradient issues)
   - Critic LR: 1e-4 vs 1e-3 (TD3 paper) = **10× TOO SLOW**
   - Batch size: 256 vs 100 (TD3 paper) = **2.56× TOO LARGE**
   - Gamma: 0.9 vs 0.99 (all papers) = **10% TOO LOW**
   - Tau: 0.001 vs 0.005 (TD3 paper) = **5× TOO SLOW**

3. **⚠️ CNN ARCHITECTURE SUBOPTIMAL** (After fixing above)
   - Missing max pooling (used in successful papers)
   - Uses LSTM (256 units) vs successful papers use GRU (48 units)

---

## PHASE 1: Emergency Fixes (Do Now) 🚨

### Task 1: Debug Gradient Clipping Failure

**File**: `src/agents/td3_agent.py`  
**Location**: Lines 815-830 (actor clipping), 636-650 (critic clipping)

**Add Logging Before/After Clipping:**

```python
# In td3_agent.py, train() method:

# ============ ACTOR GRADIENT CLIPPING DEBUG ============
if self.actor_cnn is not None:
    actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
    
    # BEFORE clipping - get current norm
    grad_norm_before = torch.nn.utils.clip_grad_norm_(
        actor_params, max_norm=float('inf')
    )
    
    # APPLY clipping
    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
    
    # AFTER clipping - verify it worked
    grad_norm_after = torch.nn.utils.clip_grad_norm_(
        actor_params, max_norm=float('inf')
    )
    
    # LOG results
    if self.total_it % 100 == 0:  # Every 100 steps
        self.logger.info(
            f"[Gradient Clipping Debug] Actor CNN:\n"
            f"  BEFORE: {grad_norm_before:.6f}\n"
            f"  AFTER:  {grad_norm_after:.6f}\n"
            f"  EXPECTED: ≤1.0\n"
            f"  STATUS: {'✅ PASS' if grad_norm_after <= 1.1 else '❌ FAIL'}"
        )
    
    # ASSERT to catch failures
    if grad_norm_after > 1.1:
        raise RuntimeError(
            f"Actor CNN gradient clipping FAILED!\n"
            f"  Expected: ≤1.0\n"
            f"  Actual: {grad_norm_after:.6f}\n"
            f"  This indicates a critical bug in gradient clipping."
        )

# ============ CRITIC GRADIENT CLIPPING DEBUG ============
if self.critic_cnn is not None:
    critic_params = list(self.critic.parameters()) + list(self.critic_cnn.parameters())
    
    # BEFORE clipping
    grad_norm_before = torch.nn.utils.clip_grad_norm_(
        critic_params, max_norm=float('inf')
    )
    
    # APPLY clipping
    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=10.0)
    
    # AFTER clipping
    grad_norm_after = torch.nn.utils.clip_grad_norm_(
        critic_params, max_norm=float('inf')
    )
    
    # LOG results
    if self.total_it % 100 == 0:
        self.logger.info(
            f"[Gradient Clipping Debug] Critic CNN:\n"
            f"  BEFORE: {grad_norm_before:.6f}\n"
            f"  AFTER:  {grad_norm_after:.6f}\n"
            f"  EXPECTED: ≤10.0\n"
            f"  STATUS: {'✅ PASS' if grad_norm_after <= 10.1 else '❌ FAIL'}"
        )
    
    # ASSERT to catch failures
    if grad_norm_after > 10.1:
        raise RuntimeError(
            f"Critic CNN gradient clipping FAILED!\n"
            f"  Expected: ≤10.0\n"
            f"  Actual: {grad_norm_after:.6f}\n"
            f"  This indicates a critical bug in gradient clipping."
        )
```

**Expected Output (if working):**
```
[Gradient Clipping Debug] Actor CNN:
  BEFORE: 15.234567
  AFTER:  1.000000
  EXPECTED: ≤1.0
  STATUS: ✅ PASS

[Gradient Clipping Debug] Critic CNN:
  BEFORE: 87.654321
  AFTER:  10.000000
  EXPECTED: ≤10.0
  STATUS: ✅ PASS
```

**If Assertion Fails:**
- Investigate why `clip_grad_norm_()` doesn't work
- Check if CNN parameters have gradients: `actor_cnn.parameters()` → `.grad` is not None
- Verify optimizer includes CNN parameters (see Task 2)

---

### Task 2: Verify Optimizer Includes CNN Parameters

**File**: `src/agents/td3_agent.py`  
**Location**: `__init__()` method where optimizers are created

**Add Diagnostic Logging:**

```python
# In td3_agent.py __init__():

# Create actor optimizer
self.actor_optimizer = torch.optim.Adam(
    list(self.actor.parameters()) + 
    (list(self.actor_cnn.parameters()) if self.actor_cnn is not None else []),
    lr=self.config['actor_lr']
)

# DIAGNOSTIC: Print all parameters in actor optimizer
self.logger.info("=" * 60)
self.logger.info("ACTOR OPTIMIZER PARAMETERS:")
self.logger.info("=" * 60)
actor_mlp_params = sum(p.numel() for p in self.actor.parameters())
actor_cnn_params = sum(p.numel() for p in self.actor_cnn.parameters()) if self.actor_cnn else 0
self.logger.info(f"Actor MLP parameters: {actor_mlp_params:,}")
self.logger.info(f"Actor CNN parameters: {actor_cnn_params:,}")
self.logger.info(f"Total actor parameters: {actor_mlp_params + actor_cnn_params:,}")

# Create critic optimizer
self.critic_optimizer = torch.optim.Adam(
    list(self.critic.parameters()) + 
    (list(self.critic_cnn.parameters()) if self.critic_cnn is not None else []),
    lr=self.config['critic_lr']
)

# DIAGNOSTIC: Print all parameters in critic optimizer
self.logger.info("=" * 60)
self.logger.info("CRITIC OPTIMIZER PARAMETERS:")
self.logger.info("=" * 60)
critic_mlp_params = sum(p.numel() for p in self.critic.parameters())
critic_cnn_params = sum(p.numel() for p in self.critic_cnn.parameters()) if self.critic_cnn else 0
self.logger.info(f"Critic MLP parameters: {critic_mlp_params:,}")
self.logger.info(f"Critic CNN parameters: {critic_cnn_params:,}")
self.logger.info(f"Total critic parameters: {critic_mlp_params + critic_cnn_params:,}")
self.logger.info("=" * 60)
```

**Expected Output:**
```
============================================================
ACTOR OPTIMIZER PARAMETERS:
============================================================
Actor MLP parameters: 123,456
Actor CNN parameters: 789,012
Total actor parameters: 912,468
============================================================
CRITIC OPTIMIZER PARAMETERS:
============================================================
Critic MLP parameters: 234,567
Critic CNN parameters: 789,012
Total critic parameters: 1,023,579
============================================================
```

**Red Flags:**
- ❌ Actor CNN parameters = 0 → CNN not in optimizer!
- ❌ Critic CNN parameters = 0 → CNN not in optimizer!
- ❌ Separate optimizers for CNN → they override clipping!

---

### Task 3: Check for Separate CNN Optimizers

**Search the codebase for:**

```bash
# In terminal:
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Search for separate CNN optimizers (SHOULD NOT EXIST):
grep -r "actor_cnn_optimizer" src/
grep -r "critic_cnn_optimizer" src/
grep -r "cnn_optimizer" src/

# Search for optimizer.step() calls (should be exactly 2: actor + critic):
grep -r "optimizer.step()" src/
```

**Expected:**
- ❌ NO matches for `actor_cnn_optimizer`
- ❌ NO matches for `critic_cnn_optimizer`
- ✅ Exactly 2 optimizer.step() calls: actor_optimizer, critic_optimizer

**If Found:**
- Remove separate CNN optimizers
- Merge CNN parameters into main optimizers (see Task 2)

---

### Task 4: Fix Hyperparameters

**File**: `config/td3_config.yaml`  
**Changes**: Match TD3 paper standard values

```yaml
td3:
  # ============================================================
  # CRITICAL FIXES (Match TD3 Paper - Fujimoto et al., 2018)
  # ============================================================
  
  # Learning Rates (was: critic_lr=1e-4)
  actor_lr: 0.001     # 1e-3 (TD3 paper standard)
  critic_lr: 0.001    # ❌ CHANGE from 1e-4 to 1e-3 (10× FASTER)
  
  # Batch Size (was: 256)
  batch_size: 100     # ❌ CHANGE from 256 to 100 (TD3 paper standard)
  
  # Discount Factor (was: 0.9)
  gamma: 0.99         # ❌ CHANGE from 0.9 to 0.99 (TD3 paper standard)
  
  # Target Network Update (was: tau=0.001)
  tau: 0.005          # ❌ CHANGE from 0.001 to 0.005 (5× FASTER)
  
  # ============================================================
  # KEEP THESE (Correct)
  # ============================================================
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  exploration_noise: 0.1
  
  # Replay Buffer
  buffer_size: 1000000
  start_steps: 25000
  
  # Gradient Clipping (KEEP - values are correct)
  actor_grad_clip: 1.0
  critic_grad_clip: 10.0
```

**Verify Changes Loaded:**

```python
# In td3_agent.py __init__():
self.logger.info("=" * 60)
self.logger.info("TD3 HYPERPARAMETERS VERIFICATION:")
self.logger.info("=" * 60)
self.logger.info(f"Actor LR:     {self.config['actor_lr']} (expected: 0.001)")
self.logger.info(f"Critic LR:    {self.config['critic_lr']} (expected: 0.001)")
self.logger.info(f"Batch Size:   {self.config['batch_size']} (expected: 100)")
self.logger.info(f"Gamma:        {self.config['gamma']} (expected: 0.99)")
self.logger.info(f"Tau:          {self.config['tau']} (expected: 0.005)")
self.logger.info("=" * 60)

# ASSERT critical values
assert self.config['critic_lr'] == 0.001, "Critic LR should be 1e-3!"
assert self.config['batch_size'] == 100, "Batch size should be 100!"
assert self.config['gamma'] == 0.99, "Gamma should be 0.99!"
assert self.config['tau'] == 0.005, "Tau should be 0.005!"
```

---

### Task 5: Run 5K Validation Test

**Command:**
```bash
cd /media/danielterra/Windows-SSD/Users/danie/Documents/Documents/MESTRADO/25-2_aprendizado-por-reforco/Detalhamento/av_td3_system

# Run 5K training with fixed config
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 5000 \
  --eval-freq 5000 \
  --checkpoint-freq 5000 \
  --debug  # Enable detailed logging
```

**Expected Results (if fixes work):**

| Metric | Expected (5K steps) | Previous (BROKEN) | Status |
|--------|-------------------|-------------------|--------|
| Actor CNN Gradient | ≤1.0 | 2.42 | ✅ FIXED |
| Critic CNN Gradient | ≤10.0 | 24.69 | ✅ FIXED |
| Q-Values | 0-50 | 1,796,760 | ✅ FIXED |
| Episode Rewards | Noisy but NOT degrading | 721 → 7.6 | ✅ FIXED |
| Episode Length | Stable (~50 steps) | 50 → 2 | ✅ FIXED |

**If Results Still Bad:**
- Review gradient clipping debug logs (Task 1)
- Verify hyperparameters loaded (Task 4)
- Check optimizer parameters (Task 2)
- Investigate backward() → clip() → step() order

---

## PHASE 2: Architecture Improvements (After Phase 1 Works) 🔧

### Task 6: Add Max Pooling to CNN

**File**: `src/networks/cnn_extractor.py`  
**Reason**: Successful papers (Race Driving, UAV DDPG) use max pooling for smoother gradient flow

**Changes:**

```python
class NatureCNN(nn.Module):
    def __init__(self, input_channels=4, feature_dim=512):
        super().__init__()
        
        # ============================================================
        # UPDATED ARCHITECTURE: Add max pooling (like Race Driving paper)
        # ============================================================
        
        # Conv1: stride 4→1 (dense filtering) + max pool (dimension reduction)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=1, padding=0)  # ❌ CHANGE stride
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # ✅ ADD
        
        # Conv2: stride 2→1 + max pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0) # ❌ CHANGE stride
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # ✅ ADD
        
        # Conv3: stride 1 (keep) + max pool
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0) # ✅ KEEP
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                 # ✅ ADD
        
        # Activation
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # Compute new flat_size:
        # Input: 84×84
        # Conv1(8×8,s=1,p=0): (84-8+1) = 77×77 → Pool1(2×2): 38×38
        # Conv2(4×4,s=1,p=0): (38-4+1) = 35×35 → Pool2(2×2): 17×17
        # Conv3(3×3,s=1,p=0): (17-3+1) = 15×15 → Pool3(2×2): 7×7
        # Flatten: 64 × 7 × 7 = 3136 ✅ (SAME as before)
        
        # Fully connected
        self.fc = nn.Linear(3136, feature_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Conv1 + Pool1
        out = self.conv1(x)
        out = self.activation(out)
        out = self.pool1(out)  # ✅ ADD
        
        # Conv2 + Pool2
        out = self.conv2(out)
        out = self.activation(out)
        out = self.pool2(out)  # ✅ ADD
        
        # Conv3 + Pool3
        out = self.conv3(out)
        out = self.activation(out)
        out = self.pool3(out)  # ✅ ADD
        
        # Flatten + FC
        out = out.view(out.size(0), -1)
        features = self.fc(out)
        
        return features
```

**Test:**
```bash
# Run 5K validation with new CNN
python scripts/train_td3.py --scenario 0 --max-timesteps 5000 --debug
```

**Expected:**
- Faster convergence (like Race Driving paper)
- Smoother gradient flow
- Q-values still ≤50 at 5K steps

---

### Task 7: Replace LSTM with GRU

**Files**: Actor/Critic network definitions  
**Reason**: Successful papers use GRU (48 units) instead of LSTM (256 units)

**Changes:**

```python
# In actor/critic network definitions:

# OLD:
self.lstm = nn.LSTM(
    input_size=512,  # CNN features
    hidden_size=256,  # Large recurrent state
    batch_first=True
)

# NEW:
self.gru = nn.GRU(
    input_size=512,  # CNN features
    hidden_size=48,   # ❌ CHANGE: Small recurrent state (like Race Driving)
    batch_first=True
)
```

**Impact:**
- **5.3× fewer parameters** (256 → 48)
- **Faster convergence** (Race Driving paper evidence)
- **Less gradient accumulation** (smaller recurrent state)

**Note:** Old checkpoints will be incompatible (need to retrain from scratch)

---

## PHASE 3: Validation & Monitoring

### Task 8: Run 50K Training Test

**Command:**
```bash
# After Phase 1 + 2 fixes:
python scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 50000 \
  --eval-freq 5000 \
  --checkpoint-freq 10000 \
  --debug
```

**Monitor Metrics:**

| Metric | Expected at 50K | What to Watch |
|--------|----------------|---------------|
| Actor CNN Gradient | ≤1.0 | Should NEVER exceed 1.0 |
| Critic CNN Gradient | ≤10.0 | Should NEVER exceed 10.0 |
| Q-Values | ~500 | Linear growth (like TD3 paper) |
| Episode Rewards | Increasing | Noisy but positive trend |
| Episode Length | Stable | Should NOT collapse |

**Red Flags:**
- ❌ Gradient norms exceed limits → clipping still broken
- ❌ Q-values explode (>1000 at 50K) → hyperparameters still wrong
- ❌ Episode lengths collapse → policy divergence

---

## Success Criteria

### Phase 1 (Emergency Fixes)
- ✅ Gradient norms ≤ limits (1.0 actor, 10.0 critic)
- ✅ Q-values grow linearly (~0-50 at 5K steps)
- ✅ Episode lengths stable (no collapse)
- ✅ Rewards show learning signal (not degrading)

### Phase 2 (Architecture Improvements)
- ✅ Faster convergence than Phase 1
- ✅ Smoother gradient flow
- ✅ Lower memory usage (GRU vs LSTM)

### Phase 3 (Full Training)
- ✅ Q-values ~500 at 50K steps (matches TD3 paper trend)
- ✅ Rewards increasing over time
- ✅ Episode lengths stable for 50K steps
- ✅ Ready for 100K-500K full training

---

## Timeline Estimate

| Phase | Tasks | Time Estimate |
|-------|-------|--------------|
| Phase 1: Debug + Fix Hyperparameters | Tasks 1-5 | 2-4 hours |
| Phase 2: CNN + GRU Improvements | Tasks 6-7 | 2-3 hours |
| Phase 3: 50K Validation Test | Task 8 | 10-20 hours (on RTX 2060) |
| **Total** | | **14-27 hours** |

---

## References

- **Full Analysis**: `docs/day-20/CNN_END_TO_END_TRAINING_ANALYSIS.md`
- **5K Validation**: `docs/day-20/5K_RUN_VALIDATION_REPORT.md`
- **Related Work**: `docs/day-20/RELATED_WORK_CNN_GRADIENT_ANALYSIS.md`
- **PyTorch DQN Tutorial**: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- **TD3 Paper**: Fujimoto et al., 2018
- **Spinning Up TD3**: https://spinningup.openai.com/en/latest/algorithms/td3.html

---

**END OF ACTION PLAN**

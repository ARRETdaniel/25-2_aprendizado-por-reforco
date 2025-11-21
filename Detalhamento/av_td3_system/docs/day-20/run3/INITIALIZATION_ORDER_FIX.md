# Initialization Order Bug Fix (November 20, 2025)

## Summary

**Bug**: AttributeError during TD3Agent initialization
**Root Cause**: Accessing `self.actor_cnn` at line 155 before it was assigned at line 213
**Impact**: 5K validation test crashed immediately during agent creation
**Status**: ✅ FIXED

---

## Error Details

```
File "/workspace/src/agents/td3_agent.py", line 155, in __init__
  if self.actor_cnn is not None:
AttributeError: 'TD3Agent' object has no attribute 'actor_cnn'
```

### Why This Happened

During the gradient clipping fix (Nov 20, 2025), we merged CNN parameters into the main optimizers. This required checking `if self.actor_cnn is not None` to conditionally include CNN parameters. However, we added this check at line 155 (during optimizer creation), but the CNN assignment `self.actor_cnn = actor_cnn` remained at line 213 (after optimizer creation).

**Broken Initialization Order**:
```python
Line 145: self.actor = Actor(...)           # ✅ Create actor network
Line 153: self.actor_target = copy.deepcopy(self.actor)  # ✅ Create target
Line 155: if self.actor_cnn is not None:   # ❌ CHECK before assignment!
Line 213: self.actor_cnn = actor_cnn        # ⏰ ASSIGN (too late!)
```

### Secondary Issue

The logger was also initialized at line 293 (near the end of `__init__`), but used at line 180 (during optimizer creation). This worked due to Python's lenient nature, but was technically incorrect.

---

## Fix Applied

### Fix #1: Move CNN Assignment Before Optimizer Creation

**File**: `src/agents/td3_agent.py`
**Lines Modified**: 153-176

**Before** (BROKEN):
```python
self.actor_target = copy.deepcopy(self.actor)

# ❌ Checking actor_cnn BEFORE it's assigned
if self.actor_cnn is not None:
    actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
    self.logger.info(f"  Actor optimizer: ...")  # ❌ Also using logger before init
else:
    actor_params = list(self.actor.parameters())

self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
```

**After** (FIXED):
```python
self.actor_target = copy.deepcopy(self.actor)

# 1. Handle backward compatibility (deprecated cnn_extractor parameter)
if cnn_extractor is not None and (actor_cnn is None or critic_cnn is None):
    print("  WARNING: cnn_extractor parameter is DEPRECATED!")
    if actor_cnn is None:
        actor_cnn = cnn_extractor
    if critic_cnn is None:
        critic_cnn = cnn_extractor

# 2. ✅ ASSIGN CNN references FIRST
self.actor_cnn = actor_cnn
self.critic_cnn = critic_cnn
self.use_dict_buffer = use_dict_buffer

# 3. ✅ Initialize logger BEFORE using it
self.logger = logging.getLogger(__name__)

# 4. ✅ NOW safe to check and use
if self.actor_cnn is not None:
    actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
    self.logger.info(f"  Actor optimizer: {len(list(self.actor.parameters()))} MLP params + {len(list(self.actor_cnn.parameters()))} CNN params")
else:
    actor_params = list(self.actor.parameters())
    self.logger.info(f"  Actor optimizer: {len(actor_params)} MLP params only")

self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
```

### Fix #2: Remove Duplicate CNN Assignment

**File**: `src/agents/td3_agent.py`
**Lines Removed**: 226-239 (was duplicate of earlier code)

**Removed**:
```python
# Backward compatibility: accept old cnn_extractor parameter
if cnn_extractor is not None and (actor_cnn is None or critic_cnn is None):
    print("  WARNING: cnn_extractor parameter is DEPRECATED!")
    if actor_cnn is None:
        actor_cnn = cnn_extractor
    if critic_cnn is None:
        critic_cnn = cnn_extractor

# Store CNN references
self.actor_cnn = actor_cnn
self.critic_cnn = critic_cnn
self.use_dict_buffer = use_dict_buffer
```

**Reason**: This code was moved to lines 160-170, making this duplicate redundant.

### Fix #3: Remove Duplicate Logger Initialization

**File**: `src/agents/td3_agent.py`
**Line Modified**: 293

**Before**:
```python
# Initialize logger for debug mode
self.logger = logging.getLogger(__name__)
```

**After**:
```python
# Logger already initialized earlier (line 176)
# Removed duplicate: self.logger = logging.getLogger(__name__)
```

---

## Corrected Initialization Order

The `__init__` method now follows this logical sequence:

```python
# 1. Store basic parameters
self.device = device
self.discount = discount
self.tau = tau
...

# 2. Create actor network
self.actor = Actor(state_dim, action_dim, max_action, network_config).to(self.device)
self.actor_target = copy.deepcopy(self.actor)

# 3. Handle backward compatibility
if cnn_extractor is not None:
    # Map deprecated parameter to new parameters
    ...

# 4. ✅ ASSIGN CNN references (BEFORE first use)
self.actor_cnn = actor_cnn
self.critic_cnn = critic_cnn
self.use_dict_buffer = use_dict_buffer

# 5. ✅ Initialize logger (BEFORE first logging call)
self.logger = logging.getLogger(__name__)

# 6. Create actor optimizer (NOW safe to check self.actor_cnn)
if self.actor_cnn is not None:
    actor_params = list(self.actor.parameters()) + list(self.actor_cnn.parameters())
    self.logger.info(f"  Actor optimizer: ...")
else:
    actor_params = list(self.actor.parameters())
self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)

# 7. Create critic networks
self.critic = Critic(state_dim, action_dim, critic_config).to(self.device)
self.critic_target = copy.deepcopy(self.critic)

# 8. Create critic optimizer (uses self.critic_cnn)
if self.critic_cnn is not None:
    critic_params = list(self.critic.parameters()) + list(self.critic_cnn.parameters())
    self.logger.info(f"  Critic optimizer: ...")
else:
    critic_params = list(self.critic.parameters())
self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

# 9. ✅ NO separate CNN optimizers (deprecated Nov 20, 2025)
self.actor_cnn_optimizer = None  # REMOVED
self.critic_cnn_optimizer = None  # REMOVED

# 10. Initialize replay buffer, tracking, etc.
...
```

---

## Validation Checklist

To verify the fix works:

### 1. Agent Initialization
- [ ] Agent initializes without AttributeError
- [ ] CNNs are properly assigned (self.actor_cnn, self.critic_cnn)
- [ ] Logger is initialized before first logging call
- [ ] Optimizers include CNN parameters when CNNs are provided

### 2. Logging Output
Expected console output during initialization:
```
TD3Agent initialization:
  State dim (MLP): 6, Action dim: 2
  Using separate CNNs for actor and critic
  Actor optimizer: 132096 MLP params + 107658 CNN params
  Critic optimizer: 263168 MLP params + 215316 CNN params
  🔧 CRITICAL FIX (Nov 20, 2025): REMOVED separate CNN optimizers
  CNNs will be trained via MAIN optimizers (actor/critic) with gradient clipping
```

### 3. Training Start
- [ ] Exploration phase begins (random actions)
- [ ] No errors during state preprocessing
- [ ] No errors during action selection
- [ ] Episode loop starts successfully

### 4. TensorBoard Metrics (After 5K Steps)
- [ ] `debug/actor_grad_norm_AFTER_clip` ≤ 1.0
- [ ] `debug/critic_grad_norm_AFTER_clip` ≤ 10.0
- [ ] `debug/actor_cnn_grad_norm_AFTER_clip` ≤ 1.0
- [ ] `debug/critic_cnn_grad_norm_AFTER_clip` ≤ 10.0
- [ ] `train/q1_value`, `train/q2_value` in reasonable range (0-50)
- [ ] `train/episode_length` stable (not collapsing)

---

## Lessons Learned

### Why This Bug Was Hard to Spot

1. **Non-Adjacent Code**: The check (`if self.actor_cnn`) was at line 155, but the assignment was 58 lines later at line 213. Easy to miss during refactoring.

2. **Worked Before Nov 20**: Prior to the gradient clipping fix, the optimizer creation at line 155 didn't need to check `self.actor_cnn` because CNNs had separate optimizers created at line 242. The check was only added during Nov 20 refactoring.

3. **Python's Lenient Nature**: Python doesn't catch initialization order issues at parse time, only at runtime when the problematic line executes.

### Best Practices for Python `__init__`

1. **Assign Before Access**: Always assign instance variables BEFORE checking or using them.

2. **Group Related Operations**: Keep initialization steps together:
   ```python
   # GOOD:
   self.actor_cnn = actor_cnn  # Assign
   if self.actor_cnn is not None:  # Then check
       ...

   # BAD:
   if self.actor_cnn is not None:  # Check
       ...
   # ... 50 lines later ...
   self.actor_cnn = actor_cnn  # Assign (too late!)
   ```

3. **Initialize Dependencies Early**: If variable `A` depends on variable `B`, initialize `B` first.

4. **Comment Critical Order**: Add comments when order matters:
   ```python
   # Must assign BEFORE optimizer creation (which checks self.actor_cnn)
   self.actor_cnn = actor_cnn
   ```

5. **Avoid Duplicates**: Don't initialize the same variable twice (e.g., logger at lines 176 and 293).

### Refactoring Safety

When moving code that checks instance variables:
1. **Trace Dependencies**: Find ALL lines that assign/use the variable
2. **Preserve Order**: If you move a check up, move the assignment up too
3. **Test Immediately**: Run a quick smoke test after refactoring
4. **Pair Review**: Second set of eyes catches these easily

---

## Related Documents

- `docs/day-20/CNN_ARCHITECTURE_EXPLANATION.md`: Why we have separate CNN files
- `docs/day-20/GRADIENT_CLIPPING_FIX.md`: Nov 20 fix that introduced this bug
- `docs/day-20/HYPERPARAMETER_FIX.md`: Nov 20 hyperparameter corrections

---

## Next Steps

1. **Rerun 5K Validation Test**: Verify agent initializes successfully
2. **Monitor TensorBoard**: Check gradient norms after 5K steps
3. **Full 1M Training**: If 5K validation passes, proceed to full training
4. **Document Results**: Update progress tracking with validation outcomes

---

**Status**: ✅ FIXED (November 20, 2025)
**Tested**: Pending (awaiting 5K validation test retry)

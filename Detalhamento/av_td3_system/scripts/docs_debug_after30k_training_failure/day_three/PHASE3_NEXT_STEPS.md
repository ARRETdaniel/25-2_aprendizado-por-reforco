# Phase 3: Modify Training Loop - Implementation Guide

**Status:** ⏳ **READY TO IMPLEMENT**  
**Prerequisites:** ✅ Phase 1 (DictReplayBuffer) & Phase 2 (TD3Agent) COMPLETE  
**File to Modify:** `scripts/train_td3.py`

---

## Quick Summary

**What's Done:**
- ✅ DictReplayBuffer created (`src/utils/dict_replay_buffer.py`)
- ✅ TD3Agent modified to support CNN training (`src/agents/td3_agent.py`)

**What's Left:**
- ⏳ Modify `train_td3.py` to pass CNN to TD3Agent
- ⏳ Store Dict observations directly in replay buffer
- ⏳ Keep `flatten_dict_obs()` for inference only

**Expected Time:** 30-45 minutes

---

## Changes Required in train_td3.py

### Change 1: Pass CNN to TD3Agent Initialization

**Location:** Lines ~175-207 in `__init__()` method

**Current Code:**
```python
# Initialize agent
self.agent = TD3Agent(
    state_dim=535,
    action_dim=2,
    max_action=1.0,
    config=td3_config,
    device=args.device
)
```

**New Code:**
```python
# Initialize agent WITH CNN for end-to-end training
self.agent = TD3Agent(
    state_dim=535,
    action_dim=2,
    max_action=1.0,
    cnn_extractor=self.cnn_extractor,  # ← Pass CNN to agent!
    use_dict_buffer=True,               # ← Enable DictReplayBuffer!
    config=td3_config,
    device=args.device
)
```

### Change 2: Store Dict Observations Directly

**Location:** Lines ~530-580 in `train()` method

**Current Code:**
```python
# Store flattened state in replay buffer
flat_state = self.flatten_dict_obs(obs, enable_grad=False)
flat_next_state = self.flatten_dict_obs(next_obs, enable_grad=False)

self.agent.replay_buffer.add(
    state=flat_state,
    action=action,
    next_state=flat_next_state,
    reward=reward,
    done=done or truncated
)
```

**New Code:**
```python
# Store Dict observation directly in replay buffer (NO pre-flattening!)
# This enables gradient flow through CNN during training
self.agent.replay_buffer.add(
    obs_dict=obs,           # ← Dict observation!
    action=action,
    next_obs_dict=next_obs,  # ← Dict observation!
    reward=reward,
    done=done or truncated
)
```

### Change 3: Flatten Only for Action Selection

**Location:** Lines ~515-530 in `train()` method

**Current Code:**
```python
if t < self.agent.start_timesteps:
    # Exploration: random actions
    action = np.array([
        np.random.uniform(-1, 1),  # steering
        np.random.uniform(0, 1)     # throttle (forward bias)
    ])
else:
    # Exploitation: use policy
    flat_state = self.flatten_dict_obs(obs, enable_grad=False)
    action = self.agent.select_action(flat_state, noise=self.agent.expl_noise)
```

**New Code (NO CHANGE NEEDED):**
```python
# This is already correct!
# flatten_dict_obs() is ONLY used for action selection (inference)
# NOT used for replay buffer storage anymore
if t < self.agent.start_timesteps:
    action = np.array([
        np.random.uniform(-1, 1),
        np.random.uniform(0, 1)
    ])
else:
    flat_state = self.flatten_dict_obs(obs, enable_grad=False)
    action = self.agent.select_action(flat_state, noise=self.agent.expl_noise)
```

**Note:** `flatten_dict_obs()` is now ONLY for action selection, NOT for storage!

### Change 4: Update Evaluation Loop (Optional Improvement)

**Location:** Lines ~856-925 in `evaluate()` method

**Current Code:**
```python
for episode in range(self.num_eval_episodes):
    obs = self.env.reset()
    # ... evaluation loop ...
    flat_state = self.flatten_dict_obs(obs, enable_grad=False)
    action = self.agent.select_action(flat_state, noise=0)  # No noise for eval
```

**New Code (NO CHANGE NEEDED):**
```python
# This is already correct!
# Evaluation uses flatten_dict_obs() for action selection (inference mode)
for episode in range(self.num_eval_episodes):
    obs = self.env.reset()
    # ... evaluation loop ...
    flat_state = self.flatten_dict_obs(obs, enable_grad=False)
    action = self.agent.select_action(flat_state, noise=0)
```

---

## Exact Lines to Modify

### Modification 1: TD3Agent initialization (Line ~195)

**Find this block:**
```python
        # Initialize agent
        self.agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            config=td3_config,
            device=args.device
        )
```

**Replace with:**
```python
        # Initialize agent WITH CNN for end-to-end training (Bug #13 fix)
        self.agent = TD3Agent(
            state_dim=535,
            action_dim=2,
            max_action=1.0,
            cnn_extractor=self.cnn_extractor,  # Pass CNN to agent
            use_dict_buffer=True,               # Enable DictReplayBuffer
            config=td3_config,
            device=args.device
        )
        
        print(f"[AGENT] CNN passed to TD3Agent for end-to-end training")
        print(f"[AGENT] DictReplayBuffer enabled for gradient flow")
```

### Modification 2: Replay buffer storage (Line ~540-570)

**Find this block:**
```python
            # Store transition in replay buffer
            flat_state = self.flatten_dict_obs(obs, enable_grad=False)
            flat_next_state = self.flatten_dict_obs(next_obs, enable_grad=False)
            
            self.agent.replay_buffer.add(
                state=flat_state,
                action=action,
                next_state=flat_next_state,
                reward=reward,
                done=done or truncated
            )
```

**Replace with:**
```python
            # Store Dict observation directly in replay buffer (Bug #13 fix)
            # This enables gradient flow through CNN during training
            self.agent.replay_buffer.add(
                obs_dict=obs,           # Dict observation (not flattened!)
                action=action,
                next_obs_dict=next_obs,  # Dict observation (not flattened!)
                reward=reward,
                done=done or truncated
            )
```

---

## Configuration Update

Add CNN learning rate to `config/td3_config.yaml`:

```yaml
networks:
  cnn:
    learning_rate: 0.0001  # Conservative 1e-4 for CNN
  actor:
    hidden_layers: [256, 256]
    learning_rate: 0.0003
  critic:
    hidden_layers: [256, 256]
    learning_rate: 0.0003
```

---

## Testing After Implementation

### 1. Quick Syntax Check

```bash
cd av_td3_system
python -m py_compile scripts/train_td3.py
python -m py_compile src/agents/td3_agent.py
python -m py_compile src/utils/dict_replay_buffer.py
```

### 2. Import Test

```python
# Test in Python REPL
import sys
sys.path.insert(0, '/workspace/av_td3_system')

from src.utils.dict_replay_buffer import DictReplayBuffer
from src.agents.td3_agent import TD3Agent
from src.networks.cnn_extractor import NatureCNN

print("✓ All imports successful!")
```

### 3. Diagnostic Training Run (1000 steps)

```bash
# Start CARLA server first
docker start carla-server && sleep 5

# Run diagnostic training
cd av_td3_system
python scripts/train_td3.py --scenario 0 --max-timesteps 1000 --debug --device cpu
```

**Expected Output:**
```
[AGENT] CNN optimizer initialized with lr=0.0001
[AGENT] CNN mode: training (gradients enabled)
[AGENT] Using DictReplayBuffer for end-to-end CNN training
[AGENT] CNN passed to TD3Agent for end-to-end training
[AGENT] DictReplayBuffer enabled for gradient flow
```

**Success Criteria:**
- ✅ No errors during initialization
- ✅ Vehicle moves (speed > 0 km/h)
- ✅ Episode rewards vary (not constant -53)
- ✅ Replay buffer fills up correctly
- ✅ Training updates occur without errors

### 4. Full Training Run (30K steps)

```bash
cd av_td3_system
python scripts/train_td3.py --scenario 0 --max-timesteps 30000 --device cpu
```

**Compare with Previous Failure:**

| Metric | Old (Bug #13) | Expected (Fixed) |
|--------|---------------|------------------|
| Vehicle Speed | 0 km/h | > 5 km/h |
| Mean Reward | -52,700 | > -30,000 |
| Success Rate | 0% | > 5% |
| CNN Updates | 0 (frozen) | > 0 (learning) |

---

## Debugging Tips

### Issue: "Cannot sample X transitions from buffer with Y transitions"

**Cause:** DictReplayBuffer not filling up fast enough

**Fix:** Check that `replay_buffer.add()` is being called with Dict observations

### Issue: "KeyError: 'image'"

**Cause:** Trying to store flattened states in DictReplayBuffer

**Fix:** Verify you're passing `obs_dict` (not `flat_state`) to `replay_buffer.add()`

### Issue: CNN weights not changing

**Cause:** Check if CNN optimizer is None or gradients not flowing

**Debug Code:**
```python
# After first training update, check CNN gradients
for name, param in agent.cnn_extractor.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

### Issue: Memory usage too high

**Cause:** DictReplayBuffer stores images (larger than features)

**Fix:** Reduce buffer size in config:
```yaml
training:
  buffer_size: 250000  # Reduced from 1000000
```

---

## Rollback Plan

If Phase 3 implementation causes issues:

1. **Add flag to config:**
   ```yaml
   algorithm:
     use_end_to_end_cnn: false  # Disable Bug #13 fix temporarily
   ```

2. **Conditional initialization in train_td3.py:**
   ```python
   use_end_to_end = td3_config['algorithm'].get('use_end_to_end_cnn', True)
   
   self.agent = TD3Agent(
       state_dim=535,
       action_dim=2,
       max_action=1.0,
       cnn_extractor=self.cnn_extractor if use_end_to_end else None,
       use_dict_buffer=use_end_to_end,
       config=td3_config,
       device=args.device
   )
   ```

3. **Fall back to standard ReplayBuffer:**
   - TD3Agent will automatically use standard ReplayBuffer if `use_dict_buffer=False`
   - CNN won't be trained, but system will work

---

## Final Checklist

Before running full training:

- [ ] Modified `train_td3.py` to pass CNN to TD3Agent
- [ ] Modified replay buffer storage to use Dict observations
- [ ] Added CNN learning rate to config file
- [ ] Tested import of all modified files
- [ ] Ran 1000-step diagnostic training
- [ ] Verified vehicle moves and rewards vary
- [ ] Verified CNN weights change during training
- [ ] Checked memory usage is acceptable

---

## Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Modify train_td3.py | 15 min | ⏳ TODO |
| Update config file | 5 min | ⏳ TODO |
| Test imports | 5 min | ⏳ TODO |
| Run 1000-step diagnostic | 10 min | ⏳ TODO |
| Debug if needed | 15 min | ⏳ TODO |
| Run full 30K training | 2-4 hours | ⏳ TODO |
| **TOTAL** | **~3-4 hours** | ⏳ TODO |

---

## Success Metrics

After full implementation and 30K training:

✅ **Vehicle Speed:** > 5 km/h (currently: 0 km/h)  
✅ **Mean Reward:** > -30,000 (currently: -52,700)  
✅ **Success Rate:** > 5% (currently: 0%)  
✅ **CNN Learning:** Weights change significantly from initialization  
✅ **Episode Length:** > 100 steps on average  
✅ **Visual Features:** Meaningful activations in CNN

---

## Next Immediate Action

**RIGHT NOW:** Implement Modification 1 and 2 in `train_td3.py`

Use `replace_string_in_file` tool to make the exact changes documented above.

After modifications, run syntax check and diagnostic training to validate the fix works!

---

**Author:** GitHub Copilot  
**Date:** 2025-11-01  
**Status:** Phase 3 Implementation Guide - READY TO START ⏳

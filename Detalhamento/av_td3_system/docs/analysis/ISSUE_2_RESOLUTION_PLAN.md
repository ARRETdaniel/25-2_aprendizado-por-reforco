# Issue #2: Vector Observation Size Mismatch - Resolution Plan

**Issue:** Vector observation is 23 dimensions, expected 53 dimensions
**Root Cause:** Configuration mismatch - waypoint config produces 10 waypoints instead of 25
**Solution Type:** Configuration fix (no code changes to core logic)
**Estimated Time:** 2-3 hours (config + testing + validation)

---

## Problem Statement

**Current State:**
```
Vector observation = 3 kinematic + (10 waypoints × 2) = 23 dimensions
Actor/Critic input = 512 (CNN) + 23 (vector) = 535 dimensions
```

**Required State:**
```
Vector observation = 3 kinematic + (25 waypoints × 2) = 53 dimensions
Actor/Critic input = 512 (CNN) + 53 (vector) = 565 dimensions
```

**Delta:** Missing 30 dimensions (15 waypoints × 2 coordinates)

---

## Root Cause

### Configuration Mismatch in `carla_config.yaml`

**Current configuration (INCORRECT):**
```yaml
route:
  sampling_resolution: 2.0  # meters between waypoints
  lookahead_distance: 50.0  # meters ahead to look
  num_waypoints_ahead: 10   # ❌ HARDCODED, conflicts with dynamic calculation

# Legacy section (also conflicts)
waypoints:
  lookahead_distance: 5.0
  num_waypoints_ahead: 5
```

**Expected calculation:**
```python
num_waypoints = ceil(lookahead_distance / sampling_resolution)
num_waypoints = ceil(50.0 / 2.0) = 25 waypoints
```

**But the code is using the config value `10` instead of the calculated value `25`.**

---

## Solution

### Step 1: Fix Configuration File

**File:** `config/carla_config.yaml`

**Change 1: Remove hardcoded waypoint count**
```yaml
route:
  sampling_resolution: 2.0  # Keep
  lookahead_distance: 50.0  # Keep
  # DELETE: num_waypoints_ahead: 10
  # Let code calculate dynamically: ceil(50/2) = 25
```

**Change 2: Remove legacy waypoints section**
```yaml
# DELETE ENTIRE SECTION (lines 187-198):
# waypoints:
#   file_path: '/workspace/config/waypoints.txt'
#   lookahead_distance: 5.0
#   num_waypoints_ahead: 5
#   waypoint_spacing: 2.0
```

**Rationale:**
- The `WaypointManagerAdapter` already calculates `num_waypoints_ahead` dynamically (line 254)
- Removing the config value lets the dynamic calculation take effect
- The legacy `waypoints` section conflicts with the `route` section

### Step 2: Verify Code Logic (No Changes Needed)

**File:** `src/environment/carla_env.py`

The code **already has correct logic**:

```python
# _setup_spaces() - Line 377-381
lookahead_distance = self.carla_config.get("route", {}).get("lookahead_distance", 50.0)
sampling_resolution = self.carla_config.get("route", {}).get("sampling_resolution", 2.0)
num_waypoints_ahead = int(np.ceil(lookahead_distance / sampling_resolution))
# Result: ceil(50.0 / 2.0) = 25 ✅

# WaypointManagerAdapter.__init__() - Line 254
self.num_waypoints_ahead = int(np.ceil(lookahead_distance / sampling_resolution))
# Result: ceil(50.0 / 2.0) = 25 ✅
```

**BUT:** The adapter is being overridden somewhere. Need to trace:
1. Where is `num_waypoints_ahead` being read from config?
2. Why is it using `10` instead of the calculated `25`?

### Step 3: Update Network Dimensions

**File:** `src/networks/actor.py`

```python
# Current:
def __init__(self, state_dim, action_dim=2, max_action=1.0, hidden_size=256):
    self.fc1 = nn.Linear(state_dim, hidden_size)  # Currently: 535 → 256

# After config fix:
# state_dim will automatically become 565 (512 CNN + 53 vector)
# No code changes needed if state_dim is passed correctly
```

**File:** `src/networks/critic.py`

```python
# Current:
def __init__(self, state_dim, action_dim=2, hidden_size=256):
    self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)  # 535 + 2 = 537 → 256

# After config fix:
# state_dim will automatically become 565 (512 + 53)
# Input: 565 + 2 = 567 → 256
# No code changes needed if state_dim is passed correctly
```

**File:** `src/agents/td3_agent.py`

Need to verify `state_dim` calculation:
```python
# Should be:
cnn_output_dim = 512  # From CNN feature extractor
vector_obs_dim = 53   # From environment (after config fix)
state_dim = cnn_output_dim + vector_obs_dim  # 512 + 53 = 565
```

---

## Implementation Steps

### Phase 1: Configuration Fix (30 minutes)

1. **Backup config:**
   ```bash
   cp config/carla_config.yaml config/carla_config.yaml.backup
   ```

2. **Edit `config/carla_config.yaml`:**
   - Remove line `num_waypoints_ahead: 10` from `route` section
   - Remove entire `waypoints` section (lines 187-198)
   - Verify `lookahead_distance: 50.0` and `sampling_resolution: 2.0`

3. **Test observation space:**
   ```python
   from src.environment.carla_env import CARLANavigationEnv

   env = CARLANavigationEnv(
       carla_config_path='config/carla_config.yaml',
       td3_config_path='config/td3_config.yaml',
       training_config_path='config/training_config.yaml'
   )

   obs, info = env.reset()
   print(f"Vector shape: {obs['vector'].shape}")
   # Expected: Vector shape: (53,)

   assert obs['vector'].shape == (53,), f"Expected (53,), got {obs['vector'].shape}"
   ```

### Phase 2: Code Investigation (1 hour)

**If Phase 1 test fails (still returns 23 dims), investigate:**

1. **Trace waypoint manager initialization:**
   ```python
   # In carla_env.__init__(), add debug logging:
   self.logger.info(f"Route config: {self.carla_config.get('route', {})}")
   self.logger.info(f"Waypoint manager num_waypoints_ahead: {self.waypoint_manager.num_waypoints_ahead}")
   ```

2. **Check waypoint_manager.py:**
   ```bash
   grep -n "num_waypoints_ahead" src/environment/waypoint_manager.py
   ```
   - Is it reading from config directly?
   - Is it using a default value?

3. **Check dynamic_route_manager.py:**
   ```bash
   grep -n "num_waypoints_ahead" src/environment/dynamic_route_manager.py
   ```
   - Same questions as above

4. **Find where the override happens:**
   ```bash
   grep -rn "num_waypoints_ahead.*10" src/
   grep -rn "get.*num_waypoints" src/
   ```

### Phase 3: Network Dimension Update (30 minutes)

**Only if state_dim is hardcoded somewhere:**

1. **Search for hardcoded 535:**
   ```bash
   grep -rn "535" src/
   ```

2. **Update if found:**
   ```python
   # Change:
   state_dim = 535  # OLD

   # To:
   state_dim = 565  # NEW (512 CNN + 53 vector)
   ```

3. **Verify network initialization:**
   ```python
   from src.networks.actor import Actor
   from src.networks.critic import Critic

   actor = Actor(state_dim=565, action_dim=2)
   critic = Critic(state_dim=565, action_dim=2)

   # Test forward pass
   import torch
   state = torch.randn(1, 565)
   action = torch.randn(1, 2)

   actor_output = actor(state)
   q_value = critic(state, action)

   print(f"Actor output shape: {actor_output.shape}")  # Should be (1, 2)
   print(f"Critic output shape: {q_value.shape}")      # Should be (1, 1)
   ```

### Phase 4: Testing & Validation (1 hour)

1. **Unit tests:**
   ```bash
   pytest tests/test_carla_env.py::test_observation_dimensions -v
   pytest tests/test_networks.py::test_actor_forward_pass -v
   pytest tests/test_networks.py::test_critic_forward_pass -v
   ```

2. **Integration test:**
   ```python
   # Test full observation → network pipeline
   env = CARLANavigationEnv(...)
   obs, info = env.reset()

   # Extract components
   image_obs = obs['image']  # (4, 84, 84)
   vector_obs = obs['vector']  # (53,)

   # CNN forward pass
   cnn_features = cnn_extractor(torch.from_numpy(image_obs).unsqueeze(0))
   assert cnn_features.shape == (1, 512)

   # Concatenate state
   state = torch.cat([cnn_features, torch.from_numpy(vector_obs).unsqueeze(0)], dim=1)
   assert state.shape == (1, 565)

   # Actor forward pass
   action = actor(state)
   assert action.shape == (1, 2)

   # Critic forward pass
   q_value = critic(state, action)
   assert q_value.shape == (1, 1)

   print("✅ Full pipeline test passed!")
   ```

3. **Validation:**
   ```bash
   python scripts/validate_steps.py --steps 4,5,6,7,8
   ```

---

## Expected Outcomes

### After Configuration Fix

**Observation space:**
```python
observation_space = Dict({
    'image': Box(low=-1.0, high=1.0, shape=(4, 84, 84), dtype=float32),
    'vector': Box(low=-inf, high=inf, shape=(53,), dtype=float32)
})
```

**Vector breakdown:**
```python
vector_obs = [
    velocity_normalized,              # 1 dim
    lateral_deviation_normalized,     # 1 dim
    heading_error_normalized,         # 1 dim
    waypoints_normalized.flatten(),   # 50 dims (25 waypoints × 2 coords)
]
# Total: 53 dimensions ✅
```

**Network input:**
```python
state = concat([
    cnn_features,  # 512 dims
    vector_obs     # 53 dims
])
# Total: 565 dimensions ✅
```

### Validation Checkpoints

- [ ] `_get_observation()` returns 53-dim vector
- [ ] `_setup_spaces()` calculates 25 waypoints
- [ ] Waypoint manager returns exactly 25 waypoints
- [ ] Actor input layer expects 565 dims
- [ ] Critic input layer expects 567 dims (565 state + 2 action)
- [ ] Full pipeline test passes (env → CNN → actor → critic)
- [ ] Steps 4-8 validation all pass

---

## Risks & Mitigation

### Risk 1: Config change doesn't fix the issue

**Symptom:** After removing `num_waypoints_ahead` from config, still get 23 dims

**Mitigation:**
1. Add debug logging to trace where `num_waypoints_ahead` is read
2. Check if there's a default value in the code
3. Check if `td3_config.yaml` also has `num_waypoints` setting
4. Manually set `num_waypoints_ahead = 25` in adapter as temporary fix

### Risk 2: Network dimension mismatch after fix

**Symptom:** RuntimeError: size mismatch in network forward pass

**Mitigation:**
1. Search for all hardcoded `535` in codebase
2. Update to `565` or make it dynamic based on observation space
3. Add assertion in agent to verify dimensions match

### Risk 3: Training breaks due to state distribution change

**Symptom:** Training diverges or performs worse after adding 30 dimensions

**Mitigation:**
1. Normalize new waypoint features same as existing ones
2. Verify normalization: all features in [-1, 1] range
3. Use pre-existing checkpoint if available (train from scratch if not)
4. Monitor reward components to ensure balance

---

## Success Criteria

**Phase 1 Success:**
- ✅ Config file updated (no `num_waypoints_ahead`)
- ✅ Observation space shows `vector: (53,)`
- ✅ `_get_observation()` returns 53-dim vector

**Phase 2 Success:**
- ✅ All `num_waypoints_ahead` references traced
- ✅ Root cause of override identified and fixed
- ✅ Waypoint manager consistently returns 25 waypoints

**Phase 3 Success:**
- ✅ Actor accepts 565-dim input
- ✅ Critic accepts 565-dim state + 2-dim action
- ✅ No hardcoded dimensions in network code

**Phase 4 Success:**
- ✅ All unit tests pass
- ✅ Integration test passes (full pipeline)
- ✅ Steps 4-8 validation all green
- ✅ No dimension mismatches in logs

**Overall Success:**
- ✅ Issue #2 marked RESOLVED
- ✅ Documentation updated
- ✅ Ready to proceed with Steps 4-8 validation

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Config fix + test | 30 min | ⏳ Ready to start |
| 2 | Code investigation (if needed) | 1 hour | ⏳ Contingency |
| 3 | Network updates (if needed) | 30 min | ⏳ Contingency |
| 4 | Testing & validation | 1 hour | ⏳ After Phase 1-3 |
| **Total** | **End-to-end** | **2-3 hours** | ⏳ **Pending start** |

---

## Next Actions

1. **Review this plan** with the team
2. **Approve configuration changes**
3. **Execute Phase 1** (config fix + test)
4. **If Phase 1 succeeds:** Proceed to Phase 4 (testing)
5. **If Phase 1 fails:** Execute Phase 2 (investigation) then retry
6. **Document results** in `ISSUE_2_RESOLUTION.md`
7. **Update** `VALIDATION_PROGRESS.md`
8. **Proceed** with Steps 4-8 validation

---

**Status:** ✅ PLAN READY - Awaiting approval to execute
**Priority:** P0 - Critical blocker for Steps 4-8
**Assigned To:** [To be assigned]
**ETA:** 2-3 hours after start

---

## References

- **Analysis:** `ISSUE_2_VECTOR_OBSERVATION_ANALYSIS.md`
- **TD3 Paper:** [Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)
- **CARLA Docs:** https://carla.readthedocs.io/en/latest/python_api/
- **Validation:** `VALIDATION_PROGRESS.md`

# Evaluation Bug Analysis: "Destroyed Actor" Runtime Error

## Date
2025-11-05

## Bug Report
**Error Message**:
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  trying to operate on a destroyed actor; an actor's function was called, 
  but the actor is already destroyed.
```

**Occurrence**: During evaluation episode 3 reset, immediately after NPC spawning completes (19/19 successful).

**Log Reference**: `debug_validation_20251105_153049.log` lines ~450-500

---

## Root Cause Analysis (CARLA 0.9.16 Documentation-Based)

### 1. Traffic Manager Architecture (Single Instance Problem)

**From CARLA Documentation** ([adv_traffic_manager.md](https://carla.readthedocs.io/en/latest/adv_traffic_manager/)):

> **"ALSM (Agent Lifecycle and State Management):**
> - Scans the world to keep track of all vehicles and pedestrians
> - Updates the list of TM-controlled vehicles in the vehicle registry
> - The Vehicle registry stores vehicles registered to the TM in a separate array for iteration during the control loop"

**Problem**: Our code creates TWO CARLA environments (training + eval) but they share the **SAME Traffic Manager instance** (port 8000).

**Fatal Sequence**:

```python
# Step 1: Training environment creates TM instance
# File: carla_env.py, line 910
self.traffic_manager = self.client.get_trafficmanager()  # Port 8000 (default)
tm_port = self.traffic_manager.get_port()  # Returns 8000

# Step 2: Training env spawns NPCs, registers with TM
# File: carla_env.py, line 915
npc.set_autopilot(True, tm_port)  # NPCs A, B, C registered in TM at port 8000

# Step 3: evaluate() creates NEW environment (same CARLA server)
# File: train_td3.py, line 1032-1036
eval_env = CARLANavigationEnv(...)  # Uses SAME client/server

# Step 4: Eval env creates TM reference (SAME INSTANCE)
# File: carla_env.py, line 910
self.traffic_manager = self.client.get_trafficmanager()  # Gets EXISTING TM at port 8000

# Step 5: Eval env spawns NEW NPCs, registers with SAME TM
# File: carla_env.py, line 915
npc.set_autopilot(True, tm_port)  # NPCs D, E, F also registered in TM at port 8000
# NOW: TM vehicle registry has [A, B, C, D, E, F]

# Step 6: Eval env finishes episode 2, calls reset()
# File: carla_env.py, line 851-916 (_cleanup_episode)
for npc in self.npcs:
    npc.destroy()  # Destroys NPCs D, E, F
self.npcs = []

# Step 7: ⚠️ CRITICAL BUG ⚠️
# Traffic Manager's ALSM still has references to destroyed NPCs D, E, F
# TM registry: [A, B, C, D*, E*, F*] (* = destroyed but still referenced)

# Step 8: Eval env spawns NEW NPCs for episode 3
# File: carla_env.py, line 915
npc.set_autopilot(True, tm_port)  # NPCs G, H, I registered
# NOW: TM registry: [A, B, C, D*, E*, F*, G, H, I]

# Step 9: world.tick() triggers TM control loop
# File: carla_env.py, line 556
self.world.tick()

# Step 10: 💥 CRASH 💥
# TM's control loop iterates over vehicle registry [A, B, C, D*, E*, F*, G, H, I]
# When TM tries to apply control to D*, E*, or F* (destroyed actors):
# → std::runtime_error: trying to operate on a destroyed actor
```

---

### 2. CARLA Traffic Manager Documentation Evidence

#### 2.1 Control Loop Architecture
From [adv_traffic_manager.md](https://carla.readthedocs.io/en/latest/adv_traffic_manager/):

> **"Control loop:**
> - Receives an array of TM-controlled vehicles from the vehicle registry
> - Performs calculations for each vehicle separately by looping over the array
> - Sends the command array to the server when the last stages finish"

**Issue**: The control loop iterates over ALL vehicles in the registry, including destroyed ones that haven't been removed.

#### 2.2 Actor Lifecycle Management
From [core_actors.md](https://carla.readthedocs.io/en/latest/core_actors/):

> **"Actors are not destroyed when a Python script finishes. They have to explicitly destroy themselves."**
> 
> **"Destroying an actor blocks the simulator until the process finishes."**

**Issue**: While `actor.destroy()` removes the actor from the simulation, it does NOT automatically remove it from the Traffic Manager's internal registry.

#### 2.3 Synchronous Mode Requirements
From [adv_synchrony_timestep.md](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/):

> **"Data coming from GPU-based sensors, mostly cameras, is usually generated with a delay of a couple of frames. Synchrony is essential here."**
>
> **"If synchronous mode is enabled, and there is a Traffic Manager running, this must be set to sync mode too."**

**Issue**: We DO set TM to sync mode, but we don't allow enough time for TM's ALSM to update its registry after actor destruction.

#### 2.4 Multi-Client TM Behavior
From [adv_traffic_manager.md](https://carla.readthedocs.io/en/latest/adv_traffic_manager/):

> **"TM-Client:**
> A TM-Client is created when a TM connects to a port occupied by another TM (TM-Server). The TM-Client behavior will be dictated by the TM-Server."
>
> **Warning: Shutting down a TM-Server will shut down the TM-Clients connecting to it.**

**Issue**: Both training and eval environments create TM-Clients connected to the same TM-Server. When eval env destroys actors, the TM-Server's registry is not properly updated.

---

### 3. Why Cleanup Happens AFTER NPC Spawning

**From Log** (`debug_validation_20251105_153049.log`):
```
2025-11-05 18:34:01 - INFO - NPC spawning complete: 19/19 successful (100.0%)
terminate called after throwing an instance of 'std::runtime_error'
```

**Explanation**: 
1. Eval env episode 3 calls `reset()` → `_cleanup_episode()` → destroys NPCs D, E, F ✅
2. Eval env spawns NEW NPCs G, H, I → `npc.set_autopilot(True, tm_port)` ✅
3. **Critical**: `npc.set_autopilot()` IMMEDIATELY registers NPCs with TM
4. Eval env finishes reset, returns control to evaluate() loop
5. Next iteration calls `eval_env.step(action)` → `self.world.tick()`
6. **💥 CRASH**: TM control loop iterates over registry containing destroyed actors

**Why not during cleanup**:
- `actor.destroy()` is synchronous and blocks until complete (per CARLA docs)
- However, TM's ALSM update is **asynchronous** and happens during `world.tick()`
- Between cleanup and next tick, TM registry still has stale references

---

## Solution Options

### Option A: Separate Traffic Manager Ports (RECOMMENDED)

**Rationale**: CARLA Traffic Manager documentation explicitly supports multiple TM instances for multi-simulation scenarios.

**From Documentation**:
> **"Multi-TM simulations:**
> In a multi-TM simulation, multiple TM instances are created on distinct ports. Each TM instance will control its own behavior."

**Implementation**:

```python
# File: carla_env.py, __init__ method
class CARLANavigationEnv(Env):
    def __init__(
        self,
        ...
        tm_port: Optional[int] = None,  # NEW: Allow custom TM port
    ):
        ...
        # Get or create Traffic Manager on specified port
        if tm_port is None:
            self.traffic_manager = self.client.get_trafficmanager()  # Default port 8000
        else:
            self.traffic_manager = self.client.get_trafficmanager(tm_port)
        
        self.tm_port = self.traffic_manager.get_port()
        self.logger.info(f"Traffic Manager initialized on port {self.tm_port}")


# File: train_td3.py, evaluate() method
def evaluate(self) -> dict:
    # Use DIFFERENT TM port for evaluation to avoid registry conflicts
    eval_env = CARLANavigationEnv(
        self.carla_config_path,
        self.agent_config_path,
        self.training_config_path,
        tm_port=8050  # DIFFERENT from training env (8000)
    )
```

**Advantages**:
- ✅ Complete isolation between training and eval environments
- ✅ No shared state or actor registry conflicts
- ✅ Explicitly supported by CARLA documentation
- ✅ Minimal code changes

**Disadvantages**:
- ⚠️ Two TM instances running simultaneously (slight CPU overhead)
- ⚠️ May need to configure TM settings separately for each instance

---

### Option B: Explicit TM Registry Flush (ALTERNATIVE)

**Rationale**: Force Traffic Manager to update its actor registry immediately after cleanup.

**Implementation**:

```python
# File: carla_env.py, _cleanup_episode method
def _cleanup_episode(self):
    """Clean up vehicles and sensors from previous episode."""
    
    # ... existing sensor/vehicle cleanup ...
    
    # STEP 3: Destroy NPCs
    npc_failures = 0
    for i, npc in enumerate(self.npcs):
        try:
            # Unregister from Traffic Manager BEFORE destroying
            if hasattr(npc, 'set_autopilot'):
                npc.set_autopilot(False)  # Remove from TM registry
            
            time.sleep(0.001)  # Allow TM to process unregistration
            
            success = npc.destroy()
            if not success:
                npc_failures += 1
        except Exception as e:
            npc_failures += 1
            self.logger.debug(f"Failed to destroy NPC {i}: {e}")
    
    self.npcs = []
    
    # CRITICAL: Force TM to update its registry
    # Perform 2-3 ticks to allow ALSM to scan and clean up stale references
    if self.world and self.traffic_manager:
        for _ in range(3):
            self.world.tick()
            time.sleep(0.01)  # 10ms per tick for GPU sensor callback completion
        
        self.logger.debug("Traffic Manager registry flushed (3 ticks)")
```

**Advantages**:
- ✅ Single TM instance (lower resource usage)
- ✅ Addresses root cause (TM registry not updated)

**Disadvantages**:
- ⚠️ Adds 3 extra ticks to every reset (~60ms overhead at 20 FPS)
- ⚠️ Not explicitly documented as a solution by CARLA
- ⚠️ May not be reliable if TM ALSM timing changes in future versions

---

### Option C: Reuse Training Environment for Evaluation (NOT RECOMMENDED)

**Rationale**: Avoid creating a second environment entirely.

**Implementation**:

```python
# File: train_td3.py, evaluate() method
def evaluate(self) -> dict:
    # Reuse self.env instead of creating eval_env
    eval_rewards = []
    ...
    
    for episode in range(self.num_eval_episodes):
        obs_dict, _ = self.env.reset()  # Use training env
        ...
```

**Advantages**:
- ✅ No TM registry conflicts (single environment)
- ✅ Minimal code changes

**Disadvantages**:
- ❌ Breaks separation between training and evaluation
- ❌ RNG seed contamination (evaluation affects training randomness)
- ❌ Cannot run evaluation in parallel with training
- ❌ Violates DRL best practices (separate eval env)

---

## Recommended Solution: Option A (Separate TM Ports)

**Justification**:
1. **Explicitly supported by CARLA documentation** (Multi-TM simulations)
2. **Minimal code changes** (add tm_port parameter)
3. **Clean separation** between training and evaluation
4. **No timing hacks** or assumptions about TM internals
5. **Robust** to future CARLA version changes

**Implementation Steps**:

### Step 1: Modify `carla_env.py`

```python
# Add tm_port parameter to __init__
def __init__(
    self,
    carla_config_path: str,
    td3_config_path: str,
    training_config_path: str,
    host: str = "localhost",
    port: int = 2000,
    headless: bool = True,
    tm_port: Optional[int] = None,  # NEW: Custom TM port
):
```

```python
# Update _spawn_npc_traffic to use custom port
def _spawn_npc_traffic(self):
    try:
        # Get or create Traffic Manager on specified port
        if not hasattr(self, 'tm_port') or self.tm_port is None:
            # Default behavior: use default TM port (8000)
            self.traffic_manager = self.client.get_trafficmanager()
        else:
            # Custom port specified (e.g., for evaluation environment)
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        
        tm_port = self.traffic_manager.get_port()
        self.logger.info(f"Traffic Manager on port {tm_port}")
        
        # ... rest of spawning logic ...
```

### Step 2: Modify `train_td3.py`

```python
def __init__(self, ...):
    # Store TM port for evaluation
    self.training_tm_port = 8000  # Training uses default port
    self.eval_tm_port = 8050      # Evaluation uses separate port
```

```python
def evaluate(self) -> dict:
    print(f"[EVAL] Creating evaluation environment (TM port {self.eval_tm_port})...")
    eval_env = CARLANavigationEnv(
        self.carla_config_path,
        self.agent_config_path,
        self.training_config_path,
        tm_port=self.eval_tm_port  # Use separate TM instance
    )
    
    # ... evaluation loop ...
    
    eval_env.close()
```

---

## Testing Plan

### Test 1: Verify Separate TM Instances
```python
# After implementing Option A, add logging:
print(f"Training TM port: {self.env.traffic_manager.get_port()}")
print(f"Eval TM port: {eval_env.traffic_manager.get_port()}")
# Expected: Different ports (8000 vs 8050)
```

### Test 2: Evaluation Episode Transitions
```bash
# Run debug training with frequent evaluation
docker run --rm --network host \
  -v $(pwd):/workspace -w /workspace \
  td3-av-system:v2.0-python310 \
  python3 scripts/train_td3.py \
    --scenario 0 \
    --max-timesteps 10000 \
    --eval-freq 2000 \
    --num-eval-episodes 3 \
    --debug \
    --device cpu \
  2>&1 | tee logs/eval_fix_test.log
```

**Success Criteria**:
- ✅ No "destroyed actor" errors during evaluation
- ✅ Debug window remains open during [EVAL]
- ✅ All 3 evaluation episodes complete successfully
- ✅ Training resumes smoothly after evaluation

### Test 3: Long-Running Stability
```bash
# Run full training (200K steps with 40 evaluation cycles)
python3 scripts/train_td3.py \
  --scenario 0 \
  --max-timesteps 200000 \
  --eval-freq 5000 \
  --device cuda
```

**Success Criteria**:
- ✅ No crashes during 40 evaluation cycles
- ✅ Consistent evaluation metrics (no anomalies)
- ✅ No memory leaks or TM slowdown over time

---

## CARLA Documentation References

1. **Traffic Manager Architecture**:
   - https://carla.readthedocs.io/en/latest/adv_traffic_manager/
   - Section: "Architecture → ALSM → Vehicle registry"
   
2. **Multi-TM Simulations**:
   - https://carla.readthedocs.io/en/latest/adv_traffic_manager/
   - Section: "Running multiple Traffic Managers → Multi-TM simulations"
   
3. **Actor Lifecycle**:
   - https://carla.readthedocs.io/en/latest/core_actors/
   - Section: "Actor destruction"
   
4. **Synchronous Mode**:
   - https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
   - Section: "Synchronous mode requirements"

---

## Conclusion

The "destroyed actor" bug is caused by **Traffic Manager registry contamination** when training and evaluation environments share the same TM instance. The **recommended solution is Option A: Separate Traffic Manager Ports**, which is explicitly supported by CARLA documentation and provides clean isolation between environments.

**Next Steps**:
1. Implement Option A modifications to `carla_env.py` and `train_td3.py`
2. Run Test 1 to verify separate TM instances
3. Run Test 2 to confirm evaluation transitions work correctly
4. Run Test 3 for long-term stability validation
5. Update logging throttling documentation with evaluation fix notes

---

## Status
- [x] Root cause identified (TM registry shared between environments)
- [x] CARLA documentation reviewed and cited
- [ ] Solution implemented (Option A)
- [ ] Tests passed (1/3)
- [ ] Production validation (200K steps)

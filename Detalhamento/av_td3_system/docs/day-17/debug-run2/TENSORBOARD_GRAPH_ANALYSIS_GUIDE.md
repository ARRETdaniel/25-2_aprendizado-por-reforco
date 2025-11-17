# TensorBoard Graph Systematic Analysis Guide
**5K_POST_FIXES Validation Run**

**Analysis Date**: November 17, 2025  
**Training Steps**: 5,000 (~80 gradient updates)  
**Configuration**: train_freq=50, gradient_steps=1, learning_starts=1000, policy_freq=2  

---

## Purpose

This guide provides a systematic framework for manually analyzing the TensorBoard graphs from the 5K validation run. The analysis validates the implementation against academic literature and determines if the system is ready for the 1M production training run.

---

## Analysis Framework

For each graph, document:
1. **Observed Values**: What metrics show
2. **Literature Benchmark**: Expected ranges from papers  
3. **Comparison to Previous**: Improvement vs LITERATURE_VALIDATED_ACTOR_ANALYSIS
4. **Assessment**: ✅ Healthy / ⚠️ Concerning / ❌ Critical
5. **Evidence**: Quote relevant papers/documentation

---

## PRIORITY 1: gradients.png - GRADIENT NORMS

### Critical Question
**Did the train_freq fix (1 → 50) resolve the actor CNN gradient explosion?**

### Previous Run Metrics (BEFORE Fix)
```
Configuration: train_freq=1 (update every step)
Actor CNN Gradient Norms:
  - Mean: 1,826,337 ❌ EXTREME EXPLOSION
  - Max:  8,199,994 ❌ CATASTROPHIC
  - Pattern: EXPONENTIAL GROWTH
  
Critic CNN Gradient Norms:
  - Mean: 5,897 ✅ STABLE
  - Max:  ~50,000 ✅ ACCEPTABLE
  - Comparison: Actor 309× LARGER than critic
```

### Literature Benchmarks

**Visual DRL Papers (8 papers surveyed)**:
- ALL use gradient clipping: max_norm between 1.0 and 40.0
- Target gradient norms: < 10,000 (healthy)
- Critical threshold: > 100,000 (explosion)

**Specific Examples**:
1. **Rally A3C (Perot et al., 2017)**:
   - Gradient clipping: max_norm=40.0 for CNNs
   - Training: 140M steps with stable gradients
   
2. **TD3 Original (Fujimoto et al., 2018)**:
   - Uses MLP only (no CNN guidance)
   - No explicit gradient clipping mentioned
   - Note: MuJoCo tasks use state vectors, not images

3. **OpenAI Spinning Up TD3**:
   - Recommends update_every=50 (we now use this)
   - No gradient clipping in default config
   - Focus on hyperparameter tuning first

### Visual Inspection Checklist

Open `gradients.png` and locate the following curves:

#### 1. Actor CNN Gradient Norm (typically ORANGE line)
```
Current Value Observations:
┌────────────────────────────────────────────────┐
│ Mean Gradient Norm: _____________              │
│ Maximum Spike:      _____________              │
│ Trend: □ Stable  □ Growing  □ Oscillating     │
│ Range at end of training: _____________        │
└────────────────────────────────────────────────┘

Assessment:
□ ✅ < 10,000        → HEALTHY (proceed to 1M)
□ ⚠️ 10,000-100,000  → ELEVATED (add gradient clipping)
□ ❌ > 100,000        → EXPLOSION (must fix before 1M)

Comparison to Previous Run:
  Previous: 1,826,337 (mean)
  Current:  _____________ (mean)
  Improvement: _____% reduction
```

#### 2. Critic CNN Gradient Norm (typically BLUE line)
```
Current Value Observations:
┌────────────────────────────────────────────────┐
│ Mean Gradient Norm: _____________              │
│ Maximum Spike:      _____________              │
│ Stable throughout:  □ Yes  □ No               │
└────────────────────────────────────────────────┘

Assessment:
□ ✅ < 10,000  → HEALTHY
□ ⚠️ < 50,000  → ACCEPTABLE
□ ❌ > 50,000  → CONCERNING

Expected: Should remain stable (previous run: 5,897 mean)
```

#### 3. Actor MLP Gradient Norm
```
Current Value Observations:
┌────────────────────────────────────────────────┐
│ Mean: _____________                            │
│ Max:  _____________                            │
└────────────────────────────────────────────────┘

Assessment:
□ ✅ < 1,000   → HEALTHY
□ ⚠️ < 10,000  → ACCEPTABLE
□ ❌ > 10,000  → CONCERNING
```

#### 4. Critic MLP Gradient Norm
```
Current Value Observations:
┌────────────────────────────────────────────────┐
│ Mean: _____________                            │
│ Max:  _____________                            │
└────────────────────────────────────────────────┘

Assessment:
□ ✅ < 1,000   → HEALTHY
□ ⚠️ < 10,000  → ACCEPTABLE
□ ❌ > 10,000  → CONCERNING
```

### Gradient Analysis Summary
```
OVERALL GRADIENT HEALTH:
┌────────────────────────────────────────────────┐
│ Status: □ ✅ All gradients healthy             │
│         □ ⚠️ Some elevated, monitoring needed  │
│         □ ❌ Explosion detected, fix required  │
│                                                 │
│ PRIMARY FINDING:                                │
│ _____________________________________________  │
│ _____________________________________________  │
│                                                 │
│ RECOMMENDATION:                                 │
│ □ Proceed to 1M training                       │
│ □ Add gradient clipping (max_norm=_______)    │
│ □ Re-run 5K with fixes before 1M               │
└────────────────────────────────────────────────┘
```

---

## ANALYSIS 2: agent.png - AGENT METRICS

### Expected Behavior at 5K Steps

From literature and analysis document:
- **Episode Length**: 5-20 steps (80 gradient updates insufficient for learning)
- **Episode Reward**: Likely negative or near-zero (untrained policy)
- **Actor Loss**: Should be negative (Q-value of selected actions)
- **Q-Values**: Near zero or small negative initially

### Visual Inspection Checklist

#### 1. Episode Length
```
Observations:
┌────────────────────────────────────────────────┐
│ Mean Episode Length: _____________ steps       │
│ Trend: □ Stable  □ Increasing  □ Decreasing   │
│ Range (min-max): _____________ steps           │
└────────────────────────────────────────────────┘

Literature Validation:
  Expected at 5K:    5-20 steps ✅
  Expected at 50K:   30-80 steps
  Expected at 1M:    200-500+ steps

Assessment:
□ ✅ Within 5-20 range     → EXPECTED
□ ⚠️ Slightly outside range → ACCEPTABLE (early training variance)
□ ❌ < 3 or > 30 steps      → INVESTIGATE

Context from Log Analysis:
  Exploration Phase (1-2.5K): 56.5 steps (random actions)
  Learning Phase (2.5K-5K):   7.2 steps (policy training begins)
  Overall Mean:               12.1 steps ✅ EXPECTED
```

#### 2. Episode Reward
```
Observations:
┌────────────────────────────────────────────────┐
│ Mean Reward: _____________                     │
│ Trend: □ Improving  □ Stable  □ Declining     │
│ Variance: □ High  □ Medium  □ Low             │
└────────────────────────────────────────────────┘

Literature Context:
  "Early DRL training shows high variance in rewards"
  - Rally A3C: First 20M steps show erratic rewards
  - DDPG-UAV: Gradual improvement over thousands of episodes

Assessment:
□ ✅ Negative/near-zero with high variance → EXPECTED
□ ⚠️ Extremely negative (< -1000)         → Check reward scaling
□ ❌ Positive and increasing rapidly       → Suspicious (check logging)

Note: At 80 gradient updates, agent is essentially random.
Reward should reflect untrained exploration behavior.
```

#### 3. Actor Loss
```
Observations:
┌────────────────────────────────────────────────┐
│ Initial Value: _____________                   │
│ Final Value:   _____________                   │
│ Trend: □ Converging  □ Stable  □ Diverging    │
│ Oscillations: □ Minimal  □ Moderate  □ Severe │
└────────────────────────────────────────────────┘

TD3 Actor Loss Theory:
  Actor loss = -Q(s, μ(s)) (negative of Q-value)
  Goal: Maximize Q-value → Minimize actor loss
  
  Expected behavior:
  - Initial: Negative (Q-values of random actions)
  - Trend: Gradual increase toward 0 or positive
  - Convergence: Smooth, monotonic improvement

Previous Run Concern:
  Actor loss diverged to -7.6M in earlier run
  This indicated Q-value explosion or reward scaling issue

Assessment:
□ ✅ Negative, stable or slowly improving   → HEALTHY
□ ⚠️ Large magnitude but not diverging      → MONITOR
□ ❌ Diverging to large negative (< -1000)  → CRITICAL

If diverging: Check reward function scaling and Q-network clipping
```

#### 4. Q-Values (Q1, Q2)
```
Observations:
┌────────────────────────────────────────────────┐
│ Q1 Mean: _____________                         │
│ Q2 Mean: _____________                         │
│ Difference |Q1-Q2|: _____________              │
│ Trend: □ Stable  □ Increasing  □ Oscillating  │
└────────────────────────────────────────────────┘

TD3 Twin Critic Validation:
  Q1 and Q2 should be close together (twin critics)
  Target = min(Q1, Q2) to reduce overestimation
  
  Expected at 5K:
  - Values near zero or small negative
  - Q1 ≈ Q2 (difference < 10% of magnitude)
  - Gradual increase as policy improves

Assessment:
□ ✅ Q1 ≈ Q2, stable or gradually increasing → HEALTHY
□ ⚠️ Q1 and Q2 diverging moderately         → MONITOR
□ ❌ Q1 or Q2 exploding (> 1000)            → CRITICAL

Twin Critic Check:
  If Q1 ≈ Q2: ✅ Twin critics functioning correctly
  If Q1 >> Q2 or Q2 >> Q1: ⚠️ Potential training imbalance
```

#### 5. Training Iterations vs Episodes
```
Observations:
┌────────────────────────────────────────────────┐
│ Total Training Iterations: _____________       │
│ Total Episodes: _____________                  │
│ Ratio (iterations/episode): _____________      │
└────────────────────────────────────────────────┘

Configuration Validation:
  train_freq = 50 (update every 50 steps)
  Total steps = 5,000
  Expected iterations ≈ 5,000/50 = 100
  
  But: learning_starts = 1,000 (no updates first 1K steps)
  Actual learning steps = 5,000 - 1,000 = 4,000
  Expected iterations ≈ 4,000/50 = 80

Assessment:
□ ✅ ~80 iterations observed    → CORRECT
□ ⚠️ 60-100 iterations          → ACCEPTABLE
□ ❌ < 50 or > 200 iterations   → CONFIGURATION ERROR

If mismatch: Verify train_freq and learning_starts in logs
```

### Agent Metrics Summary
```
OVERALL AGENT HEALTH:
┌────────────────────────────────────────────────┐
│ Episode Length:  □ ✅  □ ⚠️  □ ❌             │
│ Episode Reward:  □ ✅  □ ⚠️  □ ❌             │
│ Actor Loss:      □ ✅  □ ⚠️  □ ❌             │
│ Q-Values:        □ ✅  □ ⚠️  □ ❌             │
│ Twin Critics:    □ ✅  □ ⚠️  □ ❌             │
│ Iterations:      □ ✅  □ ⚠️  □ ❌             │
│                                                 │
│ KEY FINDINGS:                                   │
│ _____________________________________________  │
│ _____________________________________________  │
│ _____________________________________________  │
└────────────────────────────────────────────────┘
```

---

## ANALYSIS 3: progress.png - TRAINING PROGRESS

### Expected Patterns

At 5K steps with 80 gradient updates:
- Episode length distribution should be BIMODAL:
  - Exploration phase (1-2.5K): Longer episodes (~50-60 steps)
  - Learning phase (2.5K-5K): Shorter episodes (~5-15 steps)
- This is EXPECTED and NORMAL (documented in COMPREHENSIVE_LOG_ANALYSIS)

### Visual Inspection Checklist

#### 1. Episode Length Over Time
```
Observations:
┌────────────────────────────────────────────────┐
│ Steps 0-2,500 (Exploration):                   │
│   Mean length: _____________ steps             │
│   Pattern: □ Stable  □ Variable                │
│                                                 │
│ Steps 2,501-5,000 (Learning):                  │
│   Mean length: _____________ steps             │
│   Pattern: □ Stable  □ Variable  □ Decreasing  │
│                                                 │
│ Overall trend:                                  │
│   □ Two distinct phases visible                │
│   □ Gradual transition                         │
│   □ Continuous decrease                        │
└────────────────────────────────────────────────┘

Literature Validation:
  From Log Analysis:
    Exploration: 56.5 steps (random actions, natural drift)
    Learning:    7.2 steps (untrained policy, rapid failures)
  
  This is EXPECTED because:
    - Random actions → gradual off-road drift → longer episodes
    - Trained policy (80 updates) → confident wrong actions → faster crashes
    - Performance "drop" is normal early-training dynamics

Assessment:
□ ✅ Bimodal distribution (exploration high, learning low) → EXPECTED
□ ✅ Gradual decrease from exploration to learning        → EXPECTED
□ ⚠️ Both phases very short (< 5 steps)                   → INVESTIGATE
□ ❌ Increasing trend (getting longer)                     → SUSPICIOUS
```

#### 2. Steps Per Episode Distribution
```
Observations:
┌────────────────────────────────────────────────┐
│ Most common episode length: _____________ steps│
│ Distribution shape:                             │
│   □ Single peak                                │
│   □ Two peaks (bimodal)                        │
│   □ Uniform/flat                               │
│   □ Heavy tail (some very long episodes)       │
└────────────────────────────────────────────────┘

Expected at 5K:
  Bimodal distribution:
  - First peak: ~55 steps (exploration phase)
  - Second peak: ~7 steps (learning phase)

Assessment:
□ ✅ Bimodal as expected
□ ⚠️ Single peak (mixed phases)
□ ❌ All episodes very short (< 3 steps)
```

#### 3. Episode Progression Timeline
```
Observations:
┌────────────────────────────────────────────────┐
│ Total episodes completed: _____________        │
│ Episodes in exploration (0-2.5K): _______      │
│ Episodes in learning (2.5K-5K): _______        │
│                                                 │
│ Learning phase acceleration:                    │
│   Episodes/1000 steps in exploration: ____     │
│   Episodes/1000 steps in learning: ____        │
│   Acceleration factor: ____x                   │
└────────────────────────────────────────────────┘

Expected from Log Analysis:
  Total episodes: 427
  Exploration: 42 episodes
  Learning: 385 episodes
  Acceleration: 385/42 = 9.2x more episodes in learning phase

This indicates: Policy failures happening much faster (expected at 80 updates)

Assessment:
□ ✅ Learning phase shows 5-10x more episodes  → EXPECTED
□ ⚠️ 2-5x acceleration                         → ACCEPTABLE
□ ❌ No acceleration or deceleration           → INVESTIGATE
```

### Progress Summary
```
TRAINING PROGRESS ASSESSMENT:
┌────────────────────────────────────────────────┐
│ Bimodal pattern observed:  □ Yes  □ No        │
│ Exploration phase visible: □ Yes  □ No        │
│ Learning phase visible:    □ Yes  □ No        │
│ Episode acceleration:      □ Yes  □ No        │
│                                                 │
│ INTERPRETATION:                                 │
│ _____________________________________________  │
│ _____________________________________________  │
│                                                 │
│ CONSISTENCY WITH LOG ANALYSIS:                 │
│ □ ✅ Matches documented 56.5 → 7.2 drop       │
│ □ ⚠️ Some discrepancies, investigate further  │
│ □ ❌ Major inconsistency, data integrity issue│
└────────────────────────────────────────────────┘
```

---

## ANALYSIS 4: agent-page2.png - ADDITIONAL METRICS

### Potential Metrics

This graph may contain:
- Learning rates over time
- Replay buffer statistics (size, utilization)
- Exploration noise decay
- Target network update frequency
- Additional loss components

### Visual Inspection Checklist

```
Metrics Visible:
┌────────────────────────────────────────────────┐
│ 1. ________________________________________   │
│    Value/Trend: ____________________________   │
│                                                 │
│ 2. ________________________________________   │
│    Value/Trend: ____________________________   │
│                                                 │
│ 3. ________________________________________   │
│    Value/Trend: ____________________________   │
│                                                 │
│ 4. ________________________________________   │
│    Value/Trend: ____________________________   │
└────────────────────────────────────────────────┘

Notable Observations:
___________________________________________________
___________________________________________________
___________________________________________________

Assessment:
□ ✅ All metrics within expected ranges
□ ⚠️ Some anomalies, note for monitoring
□ ❌ Critical issues detected
```

---

## ANALYSIS 5: eval.png - EVALUATION METRICS

### Expected State

At 5K steps, evaluation is typically NOT performed yet because:
- Training is in extreme early stage (80 gradient updates)
- Evaluation frequency often set to 10K-50K steps
- Early evaluation wastes computation on untrained policy

### Visual Inspection Checklist

```
Evaluation Status:
┌────────────────────────────────────────────────┐
│ Graph contains data: □ Yes  □ No               │
│                                                 │
│ If YES:                                         │
│   Number of evaluations: _____________         │
│   Mean eval reward: _____________              │
│   Mean eval episode length: _____________      │
│   Trend: □ Improving  □ Stable  □ Declining    │
│                                                 │
│ If NO:                                          │
│   Expected: ✅ No evaluation at 5K is normal   │
└────────────────────────────────────────────────┘

Assessment:
□ ✅ No data (expected)
□ ✅ Has data, shows early exploration behavior
□ ⚠️ Has data, shows unexpected good performance
```

---

## COMPREHENSIVE FINDINGS SUMMARY

### Critical Metrics Status

```
┌────────────────────────────────────────────────────────────┐
│                   METRIC ASSESSMENT MATRIX                  │
├────────────────────────────┬───────┬────────┬──────────────┤
│ Metric                     │ Status│ Value  │ Literature   │
├────────────────────────────┼───────┼────────┼──────────────┤
│ Actor CNN Gradient Norm    │       │        │ < 10K target │
│ Critic CNN Gradient Norm   │       │        │ < 10K target │
│ Actor MLP Gradient Norm    │       │        │ < 1K target  │
│ Critic MLP Gradient Norm   │       │        │ < 1K target  │
│ Episode Length (Learning)  │       │        │ 5-20 steps   │
│ Episode Length (Exploration│       │        │ 40-60 steps  │
│ Actor Loss Trend           │       │        │ Stable/Conv  │
│ Q-Values (Q1 ≈ Q2)         │       │        │ Difference<10%│
│ Training Iterations        │       │        │ ~80 expected │
└────────────────────────────┴───────┴────────┴──────────────┘

Legend: ✅ Healthy | ⚠️ Concerning | ❌ Critical | ⏸️ N/A
```

### Comparison to Previous Run

```
┌────────────────────────────────────────────────────────────┐
│           BEFORE vs AFTER FIX COMPARISON                    │
├────────────────────────┬──────────────┬────────────────────┤
│ Metric                 │ BEFORE (1)   │ AFTER (50)         │
├────────────────────────┼──────────────┼────────────────────┤
│ train_freq             │ 1 (wrong)    │ 50 (correct) ✅    │
│ gradient_steps         │ -1 (wrong)   │ 1 (correct) ✅     │
│ Actor CNN grad (mean)  │ 1,826,337 ❌ │ ____________       │
│ Critic CNN grad (mean) │ 5,897 ✅     │ ____________       │
│ Episode length         │ 7.2 steps ✅ │ ____________       │
│ Configuration match    │ NO ❌        │ YES ✅ (OpenAI)    │
└────────────────────────┴──────────────┴────────────────────┘

Improvement Analysis:
  Gradient explosion resolved:  □ Yes  □ No  □ Partial
  Configuration validated:      ✅ Yes (matches OpenAI standard)
  Metrics in expected ranges:   □ Yes  □ Mostly  □ No
```

---

## FINAL GO/NO-GO DECISION

### Decision Matrix

```
┌────────────────────────────────────────────────────────────┐
│                   GO/NO-GO DECISION TREE                    │
└────────────────────────────────────────────────────────────┘

QUESTION 1: Are gradient norms healthy?
  Actor CNN < 10K:  □ YES → Continue
                    □ NO  → Is it < 100K?
                              □ YES → Add clipping, mark CAUTION
                              □ NO  → STOP, fix required

QUESTION 2: Are agent metrics reasonable?
  Episode length 5-20:     □ YES → Continue
  Actor loss not diverging: □ YES → Continue
  Q-values stable:          □ YES → Continue
  
  Any NO answers? → Investigate before proceeding

QUESTION 3: Does progress match log analysis?
  Bimodal distribution:    □ YES → Continue
  56.5 → 7.2 drop visible: □ YES → Continue
  
  Any NO answers? → Data integrity check needed

QUESTION 4: Implementation validated?
  TD3 twin critics working: □ YES → Continue
  Configuration correct:     □ YES → Continue  
  ~80 iterations observed:   □ YES → Continue
```

### Final Recommendation

```
┌────────────────────────────────────────────────────────────┐
│                    FINAL RECOMMENDATION                     │
│                                                             │
│ Select ONE:                                                 │
│                                                             │
│ □ ✅ GO FOR 1M TRAINING                                    │
│    Conditions met:                                          │
│    - All gradients healthy (< 10K)                         │
│    - Agent metrics in expected ranges                      │
│    - Configuration validated against literature            │
│    - Train_freq fix resolved explosion                     │
│                                                             │
│    Action Items:                                            │
│    1. Proceed with 1M training run                         │
│    2. Monitor gradients at 50K checkpoint                  │
│    3. Implement clipping if norms exceed 50K               │
│                                                             │
│ □ ⚠️ PROCEED WITH CAUTION                                  │
│    Conditions:                                              │
│    - Gradients elevated (10K-100K) but not exploding       │
│    - Some minor concerns in agent metrics                  │
│                                                             │
│    Action Items:                                            │
│    1. Implement gradient clipping (max_norm=10.0)          │
│    2. Run 50K validation with monitoring                   │
│    3. Re-evaluate before full 1M run                       │
│                                                             │
│ □ ❌ NO-GO - FIXES REQUIRED                                │
│    Critical issues:                                         │
│    - Gradient explosion persists (> 100K)                  │
│    - Actor loss diverging severely                         │
│    - Configuration errors detected                         │
│                                                             │
│    Action Items:                                            │
│    1. Implement gradient clipping immediately              │
│    2. Review reward function scaling                       │
│    3. Consider reducing learning rates                     │
│    4. Re-run 5K validation before 1M attempt               │
│                                                             │
│ □ ⏸️ ANALYSIS INCOMPLETE                                   │
│    Complete visual inspection before deciding              │
└────────────────────────────────────────────────────────────┘
```

### Supporting Evidence

```
KEY FINDINGS SUMMARY:
1. ________________________________________________________
2. ________________________________________________________
3. ________________________________________________________

LITERATURE VALIDATION:
- TD3 (Fujimoto 2018): ____________________________________
- Rally A3C (Perot 2017): __________________________________
- OpenAI Spinning Up: _______________________________________

COMPARISON TO PREVIOUS RUN:
- Gradient explosion status: _______________________________
- Configuration improvements: ______________________________
- Training dynamics: ________________________________________

CONFIDENCE LEVEL:
□ High   (all data clear, decision straightforward)
□ Medium (some ambiguity, proceed with monitoring)
□ Low    (unclear data, recommend additional validation)
```

---

## Next Steps Based on Decision

### If GO ✅
1. Prepare 1M training run configuration
2. Set up continuous monitoring (TensorBoard + alerts)
3. Plan checkpoint intervals (50K, 100K, 500K)
4. Define early stopping criteria (gradient explosion, Q-value divergence)

### If CAUTION ⚠️
1. Implement gradient clipping in td3_agent.py:
   ```python
   # In TD3Agent.train() method, after backward():
   torch.nn.utils.clip_grad_norm_(
       self.actor_cnn.parameters(), 
       max_norm=10.0
   )
   ```
2. Run 50K validation with enhanced logging
3. Review logs and gradients at 50K before 1M commitment

### If NO-GO ❌
1. Priority fixes:
   - Gradient clipping (ALL networks)
   - Reward function balance check
   - Learning rate reduction (if needed)
2. Diagnostic 5K run with verbose logging
3. Full analysis before next attempt

---

## Analysis Completion Checklist

```
□ All 5 graphs inspected systematically
□ Gradient norms documented and assessed
□ Agent metrics validated against literature
□ Progress patterns confirmed with log analysis
□ Comparison to previous run completed
□ GO/NO-GO decision made with confidence
□ Next steps identified and documented
□ Findings saved for future reference
```

---

## Analyst Notes

```
Date: _____________
Analyst: _____________

Additional observations:
___________________________________________________
___________________________________________________
___________________________________________________
___________________________________________________

Questions for follow-up:
___________________________________________________
___________________________________________________
___________________________________________________

Confidence in recommendation: ___/10

Signature: _____________
```

---

**END OF ANALYSIS GUIDE**

*This systematic analysis framework ensures thorough validation of the 5K_POST_FIXES run against academic literature and implementation best practices. Use this guide to make an evidence-based GO/NO-GO decision for the 1M production training run.*

# CNN Learning Diagnostics Guide

This guide explains how to use the CNN diagnostics tools to monitor end-to-end learning during TD3 training.

## Overview

The CNN diagnostics system tracks three critical aspects of CNN learning:

1. **Gradient Flow**: Are gradients flowing through the CNN? (Essential for learning)
2. **Weight Updates**: Are CNN weights actually changing? (Indicates learning is happening)
3. **Feature Statistics**: Are CNN features meaningful? (Quality of learned representations)

## Quick Start

### 1. Enable Diagnostics in Training Script

Add to your training script (e.g., `scripts/train_td3.py`):

```python
from scripts.monitor_cnn_learning import setup_cnn_monitoring, log_cnn_diagnostics

# After creating agent
agent = TD3Agent(...)
setup_cnn_monitoring(agent, writer)  # writer is TensorBoard SummaryWriter

# In training loop (after agent.train())
if t % 100 == 0 and t > start_timesteps:
    log_cnn_diagnostics(
        agent, 
        writer, 
        step=t,
        print_summary=(t % 1000 == 0)  # Print every 1000 steps
    )
```

### 2. View Diagnostics in TensorBoard

```bash
tensorboard --logdir runs/
```

Navigate to:
- **Scalars → cnn_diagnostics/gradients**: Gradient magnitudes per layer
- **Scalars → cnn_diagnostics/weights**: Weight norms and changes per layer  
- **Scalars → cnn_diagnostics/features**: Feature statistics (mean, std, norm)
- **Scalars → cnn_diagnostics/health**: Overall health metrics

### 3. Analyze Checkpoints

```bash
python scripts/monitor_cnn_learning.py --checkpoint checkpoints/td3_10k.pth
```

This prints:
- CNN layer structure and parameter counts
- Weight statistics (norm, mean, std) per layer
- Whether CNN optimizer state is present
- Total training iterations

## Diagnostic Outputs

### Console Output Example

```
======================================================================
CNN DIAGNOSTICS SUMMARY
======================================================================

Captures: 950 gradients | 950 weights | 1900 features

✅ Gradient Flow: OK
✅ Weight Updates: OK

[Gradient Flow by Layer]
  features.0.weight                        ✅ FLOWING    (mean=1.23e-03, max=4.56e-03)
  features.0.bias                          ✅ FLOWING    (mean=5.67e-04, max=2.34e-03)
  features.2.weight                        ✅ FLOWING    (mean=2.34e-03, max=8.90e-03)
  ...

[Weight Updates by Layer]
  features.0.weight                        ✅ UPDATING   (mean=2.45e-04, max=1.23e-03)
  features.0.bias                          ✅ UPDATING   (mean=1.23e-04, max=5.67e-04)
  features.2.weight                        ✅ UPDATING   (mean=3.45e-04, max=1.67e-03)
  ...

[Feature Statistics]
  output_mean                               mean=  0.1234 trend=increasing
  output_std                                mean=  0.4567 trend=stable
  output_norm                               mean= 12.3456 trend=increasing
  ...
======================================================================
```

### What to Look For

#### ✅ Good Signs (CNN is Learning)
- **Gradient Flow**: All layers show ✅ FLOWING with non-zero gradients
- **Weight Updates**: All layers show ✅ UPDATING with increasing changes
- **Feature Norms**: Increasing trend (features becoming more discriminative)
- **Feature Std**: Stable or increasing (rich feature representations)

#### ❌ Bad Signs (CNN Not Learning)
- **Gradient Flow**: Any layer shows ❌ BLOCKED (gradients = 0)
- **Weight Updates**: Any layer shows ❌ FROZEN (no weight changes)
- **Feature Norms**: Decreasing or stuck at initialization values
- **Feature Std**: Decreasing (features collapsing)

## Common Issues and Solutions

### Issue 1: No Gradient Flow

**Symptoms:**
```
❌ Gradient Flow: BLOCKED
  features.0.weight                        ❌ BLOCKED
```

**Root Causes:**
- Observations flattened with `torch.no_grad()` (Bug #14)
- CNN not in training mode (`cnn.eval()` instead of `cnn.train()`)
- Using standard ReplayBuffer instead of DictReplayBuffer

**Solution:**
1. Verify `use_dict_buffer=True` in agent initialization
2. Check that training loop passes Dict observations directly:
   ```python
   # WRONG
   state = flatten_dict_obs(obs_dict)  # Breaks gradient flow!
   agent.select_action(state, noise=0.2)
   
   # CORRECT
   agent.select_action(obs_dict, noise=0.2)  # Gradients preserved!
   ```
3. Confirm CNN is in training mode:
   ```python
   print(agent.cnn_extractor.training)  # Should be True
   ```

### Issue 2: Gradients Flow But Weights Don't Update

**Symptoms:**
```
✅ Gradient Flow: OK
❌ Weight Updates: FROZEN
```

**Root Causes:**
- CNN optimizer not created or not being called
- Learning rate too low (weights change below detection threshold)
- Gradients clipped to zero

**Solution:**
1. Verify CNN optimizer exists:
   ```python
   print(agent.cnn_optimizer)  # Should not be None
   ```
2. Check learning rate:
   ```python
   print(agent.cnn_optimizer.param_groups[0]['lr'])  # Typically 1e-4
   ```
3. Increase learning rate if needed (edit `config/td3_config.yaml`):
   ```yaml
   networks:
     cnn:
       learning_rate: 0.0001  # Try 0.0003 or 0.0005 if stuck
   ```

### Issue 3: Features Not Changing

**Symptoms:**
```
✅ Gradient Flow: OK
✅ Weight Updates: OK
[Feature Statistics]
  output_norm    mean= 2.3456 trend=stable  # Not increasing!
```

**Root Causes:**
- Learning rate too low (weights update but slowly)
- Not enough training steps (CNN needs more time)
- Reward signal too noisy (CNN can't learn useful features)

**Solution:**
1. Train longer (CNN learning is slower than actor/critic)
2. Check episode rewards are improving
3. Increase CNN learning rate (carefully!)
4. Verify reward function provides useful learning signal

## Integration with Existing Training

### Minimal Integration (No Code Changes)

If you don't want to modify training code, use the checkpoint analyzer:

```bash
# After training
python scripts/monitor_cnn_learning.py --checkpoint checkpoints/td3_30k.pth
```

### Full Integration (Real-Time Monitoring)

Add to `scripts/train_td3.py`:

```python
# After agent initialization (line ~200)
if args.enable_cnn_diagnostics:
    from scripts.monitor_cnn_learning import setup_cnn_monitoring
    setup_cnn_monitoring(self.agent, self.writer)
    print("[CNN DIAGNOSTICS] Enabled real-time monitoring")

# In training loop (after agent.train(), around line ~790)
if t % 100 == 0 and t > start_timesteps:
    if self.agent.cnn_diagnostics is not None:
        from scripts.monitor_cnn_learning import log_cnn_diagnostics
        log_cnn_diagnostics(
            self.agent,
            self.writer,
            step=t,
            print_summary=(t % 1000 == 0)
        )
```

Add command-line argument:
```python
parser.add_argument(
    '--enable-cnn-diagnostics',
    action='store_true',
    help='Enable detailed CNN learning diagnostics'
)
```

## Performance Impact

- **Memory**: ~1-2 MB per 1000 training steps (negligible)
- **Compute**: ~0.1% overhead (gradient/weight norm calculations)
- **Storage**: ~10 MB TensorBoard events per 10k steps

**Recommendation**: Enable diagnostics for debugging, disable for production training.

## Interpreting TensorBoard Graphs

### Gradient Magnitudes

**Normal pattern:**
- Gradients start at ~1e-3 to 1e-2
- Decrease slightly as training stabilizes (1e-4 to 1e-3)
- Never drop to zero

**Warning signs:**
- Sudden drop to zero → Gradient flow broken
- Exponential decay → Vanishing gradients (check network depth)
- Explosion → Exploding gradients (reduce learning rate)

### Weight Changes

**Normal pattern:**
- Changes start at ~1e-3 to 1e-4
- Gradual decrease as training converges
- Never plateau at zero early in training

**Warning signs:**
- Zero from start → Weights frozen (optimizer issue)
- Oscillating wildly → Learning rate too high
- Stuck at constant value → Learning stalled

### Feature Norms

**Normal pattern:**
- Start at ~5-10 (random initialization)
- Increase to ~10-20 as CNN learns
- Stabilize when features are good

**Warning signs:**
- Stuck at initialization values → CNN not learning
- Decreasing → Features collapsing
- Exploding (>100) → Numerical instability

## Troubleshooting Checklist

- [ ] Diagnostics enabled: `agent.enable_diagnostics()` called?
- [ ] DictReplayBuffer used: `use_dict_buffer=True`?
- [ ] CNN in training mode: `agent.cnn_extractor.training == True`?
- [ ] CNN optimizer exists: `agent.cnn_optimizer is not None`?
- [ ] Gradient flow: All layers show ✅ FLOWING?
- [ ] Weight updates: All layers show ✅ UPDATING?
- [ ] Feature norms increasing: Trend shows "increasing"?
- [ ] Training steps sufficient: > 10k steps with policy learning?

## References

- **Stable-Baselines3 TensorBoard**: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
- **PyTorch Visualization**: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- **Gradient Flow Analysis**: https://towardsdatascience.com/how-to-visualize-gradient-flow-in-neural-networks-d3b9e3f3e7a4

## Support

For issues with CNN diagnostics:
1. Check console output for error messages
2. Verify imports work: `from src.utils.cnn_diagnostics import CNNDiagnostics`
3. Run standalone script: `python scripts/monitor_cnn_learning.py --checkpoint <path>`
4. Check TensorBoard logs: `tensorboard --logdir runs/`

**Last Updated:** 2025-01-XX  
**Author:** Daniel Terra

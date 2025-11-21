#!/usr/bin/env python3
"""
Extract and analyze the gradient metrics that WERE actually logged
"""
import sys
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

event_file = "docs/day-20/run3/TD3_scenario_0_npcs_20_20251120-190526/events.out.tfevents.1763665526.danielterra.1.0"

print("="*80)
print("GRADIENT METRICS EXTRACTION - What Was Actually Logged?")
print("="*80)

ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# Extract gradient metrics
gradient_tags = [
    'gradients/actor_cnn_norm',
    'gradients/critic_cnn_norm',
    'gradients/actor_mlp_norm',
    'gradients/critic_mlp_norm',
    'alerts/gradient_explosion_critical',
    'alerts/gradient_explosion_warning',
]

for tag in gradient_tags:
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        values = [e.value for e in events]

        print(f"\n{tag}:")
        print(f"  Points: {len(values)}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Last: {values[-1]:.4f}")

        # Check against limits
        if 'actor_cnn' in tag or 'actor_mlp' in tag:
            limit = 1.0
            violations = sum(1 for v in values if v > limit)
            if violations > 0:
                print(f"  ❌ VIOLATIONS: {violations}/{len(values)} exceed {limit}")
            else:
                print(f"  ✅ All ≤ {limit}")

        elif 'critic' in tag:
            limit = 10.0
            violations = sum(1 for v in values if v > limit)
            if violations > 0:
                print(f"  ❌ VIOLATIONS: {violations}/{len(values)} exceed {limit}")
            else:
                print(f"  ✅ All ≤ {limit}")

        elif 'alert' in tag:
            alerts_fired = sum(1 for v in values if v > 0)
            if alerts_fired > 0:
                print(f"  ⚠️  ALERTS FIRED: {alerts_fired}/{len(values)} times")
            else:
                print(f"  ✅ No alerts")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Also check actor loss
if 'train/actor_loss' in ea.Tags()['scalars']:
    events = ea.Scalars('train/actor_loss')
    values = [e.value for e in events]

    print(f"\nActor Loss Statistics:")
    print(f"  Mean: {np.mean(values):.2e}")
    print(f"  Min:  {np.min(values):.2e} (most negative)")
    print(f"  Max:  {np.max(values):.2e} (least negative)")

    if np.min(values) < -1e9:  # Less than -1 billion
        print(f"  ❌ EXTREME Q-VALUE EXPLOSION")
    elif np.min(values) < -1e6:  # Less than -1 million
        print(f"  ❌ Q-VALUE EXPLOSION")
    elif np.min(values) < -1000:
        print(f"  ⚠️  High negative values")
    else:
        print(f"  ✅ Reasonable range")

print("\n" + "="*80)

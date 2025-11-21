#!/usr/bin/env python3
"""
Validate gradient clipping fix by analyzing TensorBoard logs.

This script checks if the gradient norm calculation fix was successful by
validating that AFTER-clipping values respect the defined limits.

Usage: 
    python scripts/validate_gradient_fix.py <path_to_event_file>
    
Example:
    python scripts/validate_gradient_fix.py \\
        data/logs/TD3_scenario_0_npcs_20_20251121-100000/events.out.tfevents.1234567.host.1.0

Expected Outcome (After Fix):
    ✅ Actor CNN AFTER ≤ 1.0 (all updates)
    ✅ Critic CNN AFTER ≤ 10.0 (all updates)
    ✅ Actor MLP AFTER > 0.0 (network learning)
    ✅ Critic MLP AFTER > 0.0 (network learning)
    ✅ No gradient explosion alerts
    
Reference:
    - GRADIENT_CLIPPING_FIX_IMPLEMENTATION.md
    - GRADIENT_FIX_APPLIED_SUMMARY.md
"""

import sys
import os
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install tensorboard numpy")
    sys.exit(1)


def validate_gradients(event_file):
    """
    Validate gradient clipping metrics from TensorBoard event file.
    
    Args:
        event_file: Path to TensorBoard event file
        
    Returns:
        0 if all checks pass, 1 if issues found
    """
    if not os.path.exists(event_file):
        print(f"❌ Event file not found: {event_file}")
        return 1
    
    print("🔍 GRADIENT CLIPPING VALIDATION")
    print("=" * 90)
    print(f"Event file: {Path(event_file).name}")
    print("=" * 90)
    
    # Load event file
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
    except Exception as e:
        print(f"❌ Failed to load event file: {e}")
        return 1
    
    # Check if required metrics exist
    scalars = ea.Tags()['scalars']
    required_metrics = [
        'debug/actor_cnn_grad_norm_AFTER_clip',
        'debug/critic_cnn_grad_norm_AFTER_clip',
        'debug/actor_mlp_grad_norm_AFTER_clip',
        'debug/critic_mlp_grad_norm_AFTER_clip',
    ]
    
    missing = [m for m in required_metrics if m not in scalars]
    if missing:
        print(f"❌ Missing metrics: {missing}")
        print("   This event file may be from before the fix was applied.")
        return 1
    
    print(f"✅ Found all required metrics ({len(required_metrics)} total)\n")
    
    # Track issues
    issues = []
    warnings = []
    
    # 1. Actor CNN AFTER ≤ 1.0
    print("📊 Actor CNN Gradients (AFTER clipping)")
    print("-" * 90)
    actor_cnn = ea.Scalars('debug/actor_cnn_grad_norm_AFTER_clip')
    actor_cnn_values = [e.value for e in actor_cnn]
    
    if len(actor_cnn_values) == 0:
        warnings.append("⚠️  Actor CNN: No data points")
    else:
        violations = [v for v in actor_cnn_values if v > 1.0]
        mean_val = np.mean(actor_cnn_values)
        max_val = np.max(actor_cnn_values)
        
        print(f"   Data points: {len(actor_cnn_values)}")
        print(f"   Mean: {mean_val:.6f}")
        print(f"   Max:  {max_val:.6f}")
        print(f"   Limit: 1.0")
        
        if len(violations) > 0:
            pct = len(violations) / len(actor_cnn_values) * 100
            issues.append(f"❌ Actor CNN: {len(violations)}/{len(actor_cnn_values)} violations ({pct:.1f}%), max={max_val:.6f}")
            print(f"   ❌ VIOLATIONS: {len(violations)}/{len(actor_cnn_values)} ({pct:.1f}%)")
        else:
            print(f"   ✅ NO VIOLATIONS (all ≤ 1.0)")
    print()
    
    # 2. Critic CNN AFTER ≤ 10.0
    print("📊 Critic CNN Gradients (AFTER clipping)")
    print("-" * 90)
    critic_cnn = ea.Scalars('debug/critic_cnn_grad_norm_AFTER_clip')
    critic_cnn_values = [e.value for e in critic_cnn]
    
    if len(critic_cnn_values) == 0:
        warnings.append("⚠️  Critic CNN: No data points")
    else:
        violations = [v for v in critic_cnn_values if v > 10.0]
        mean_val = np.mean(critic_cnn_values)
        max_val = np.max(critic_cnn_values)
        
        print(f"   Data points: {len(critic_cnn_values)}")
        print(f"   Mean: {mean_val:.6f}")
        print(f"   Max:  {max_val:.6f}")
        print(f"   Limit: 10.0")
        
        if len(violations) > 0:
            pct = len(violations) / len(critic_cnn_values) * 100
            issues.append(f"❌ Critic CNN: {len(violations)}/{len(critic_cnn_values)} violations ({pct:.1f}%), max={max_val:.6f}")
            print(f"   ❌ VIOLATIONS: {len(violations)}/{len(critic_cnn_values)} ({pct:.1f}%)")
        else:
            print(f"   ✅ NO VIOLATIONS (all ≤ 10.0)")
    print()
    
    # 3. Actor MLP > 0.0
    print("📊 Actor MLP Gradients (AFTER clipping)")
    print("-" * 90)
    actor_mlp = ea.Scalars('debug/actor_mlp_grad_norm_AFTER_clip')
    actor_mlp_values = [e.value for e in actor_mlp]
    
    if len(actor_mlp_values) == 0:
        warnings.append("⚠️  Actor MLP: No data points")
    else:
        zero_count = sum(1 for v in actor_mlp_values if v == 0.0)
        mean_val = np.mean(actor_mlp_values)
        max_val = np.max(actor_mlp_values)
        
        print(f"   Data points: {len(actor_mlp_values)}")
        print(f"   Mean: {mean_val:.6f}")
        print(f"   Max:  {max_val:.6f}")
        print(f"   Zero count: {zero_count}/{len(actor_mlp_values)}")
        
        if max_val == 0.0:
            warnings.append("⚠️  Actor MLP: All gradients are zero (need investigation)")
            print(f"   ⚠️  ALL ZEROS (network may not be learning)")
        elif zero_count > 0:
            pct = zero_count / len(actor_mlp_values) * 100
            warnings.append(f"⚠️  Actor MLP: {zero_count}/{len(actor_mlp_values)} zero gradients ({pct:.1f}%)")
            print(f"   ⚠️  {zero_count} zero values ({pct:.1f}%) - may be due to policy_freq=2")
        else:
            print(f"   ✅ ALL NON-ZERO (network learning)")
    print()
    
    # 4. Critic MLP > 0.0
    print("📊 Critic MLP Gradients (AFTER clipping)")
    print("-" * 90)
    critic_mlp = ea.Scalars('debug/critic_mlp_grad_norm_AFTER_clip')
    critic_mlp_values = [e.value for e in critic_mlp]
    
    if len(critic_mlp_values) == 0:
        warnings.append("⚠️  Critic MLP: No data points")
    else:
        zero_count = sum(1 for v in critic_mlp_values if v == 0.0)
        mean_val = np.mean(critic_mlp_values)
        max_val = np.max(critic_mlp_values)
        
        print(f"   Data points: {len(critic_mlp_values)}")
        print(f"   Mean: {mean_val:.6f}")
        print(f"   Max:  {max_val:.6f}")
        print(f"   Zero count: {zero_count}/{len(critic_mlp_values)}")
        
        if max_val == 0.0:
            issues.append("❌ Critic MLP: All gradients are zero (network dead!)")
            print(f"   ❌ ALL ZEROS (network NOT learning)")
        elif zero_count > 0:
            pct = zero_count / len(critic_mlp_values) * 100
            warnings.append(f"⚠️  Critic MLP: {zero_count} zero gradients ({pct:.1f}%)")
            print(f"   ⚠️  {zero_count} zero values ({pct:.1f}%)")
        else:
            print(f"   ✅ ALL NON-ZERO (network learning)")
    print()
    
    # 5. Check for gradient explosion alerts
    print("🚨 Alert Analysis")
    print("-" * 90)
    alert_tags = [
        'alerts/gradient_explosion_warning',
        'alerts/gradient_explosion_critical',
        'alerts/critic_gradient_explosion_warning',
        'alerts/critic_gradient_explosion_critical',
    ]
    
    alert_found = False
    for tag in alert_tags:
        if tag in scalars:
            alerts = ea.Scalars(tag)
            if len(alerts) > 0:
                issues.append(f"❌ {tag}: {len(alerts)} fires")
                print(f"   ❌ {tag}: {len(alerts)} fires")
                alert_found = True
    
    if not alert_found:
        print(f"   ✅ No gradient explosion alerts")
    print()
    
    # 6. Check actor loss stability
    print("📈 Loss Analysis")
    print("-" * 90)
    if 'train/actor_loss' in scalars:
        actor_loss = ea.Scalars('train/actor_loss')
        actor_loss_values = [e.value for e in actor_loss]
        
        if len(actor_loss_values) > 0:
            mean_loss = np.mean(actor_loss_values)
            min_loss = np.min(actor_loss_values)
            max_loss = np.max(actor_loss_values)
            latest_loss = actor_loss_values[-1]
            
            print(f"   Actor Loss:")
            print(f"     Mean:   {mean_loss:,.2f}")
            print(f"     Min:    {min_loss:,.2f}")
            print(f"     Max:    {max_loss:,.2f}")
            print(f"     Latest: {latest_loss:,.2f}")
            
            if abs(max_loss) > 1e9:  # 1 billion
                issues.append(f"❌ Actor loss explosion: {max_loss:,.2f}")
                print(f"     ❌ EXPLOSION DETECTED (> 1 billion)")
            elif abs(max_loss) > 1e6:  # 1 million
                warnings.append(f"⚠️  Actor loss very high: {max_loss:,.2f}")
                print(f"     ⚠️  Very high (> 1 million)")
            else:
                print(f"     ✅ Stable (< 1 million)")
    print()
    
    # Summary
    print("=" * 90)
    print("📋 VALIDATION SUMMARY")
    print("=" * 90)
    
    if len(issues) == 0 and len(warnings) == 0:
        print("🎉 ALL CHECKS PASSED!")
        print("   ✅ Gradient clipping fix successful")
        print("   ✅ All AFTER-clipping values within limits")
        print("   ✅ Networks are learning (non-zero gradients)")
        print("   ✅ No gradient explosion alerts")
        print("   ✅ Actor loss stable")
        return 0
    else:
        if len(issues) > 0:
            print(f"❌ {len(issues)} CRITICAL ISSUE(S) FOUND:")
            for issue in issues:
                print(f"   {issue}")
            print()
        
        if len(warnings) > 0:
            print(f"⚠️  {len(warnings)} WARNING(S):")
            for warning in warnings:
                print(f"   {warning}")
            print()
        
        if len(issues) > 0:
            print("🔧 ACTION REQUIRED:")
            print("   1. Review GRADIENT_CLIPPING_BUG_ROOT_CAUSE.md")
            print("   2. Check if fix was applied correctly")
            print("   3. Investigate clipping implementation in td3_agent.py")
            return 1
        else:
            print("ℹ️  Warnings are informational - fix appears successful")
            print("   Actor MLP zeros may be due to policy_freq=2 (expected)")
            return 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_gradient_fix.py <event_file>")
        print("\nExample:")
        print("  python scripts/validate_gradient_fix.py \\")
        print("    data/logs/TD3_scenario_0_npcs_20_20251121-100000/events.out.tfevents.*")
        sys.exit(1)
    
    event_file = sys.argv[1]
    exit_code = validate_gradients(event_file)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

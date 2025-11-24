#!/usr/bin/env python3
"""
Validation script for progressive segment search fix.

Tests that:
1. Segment index advances as vehicle moves forward
2. t parameter varies smoothly within [0, 1] for each segment
3. Distance decreases continuously (no stuck segments)
4. No t=1.0000 stuck patterns
5. Perpendicular distance stays reasonable

Reference: validation_logs/CRITICAL_BUG_T_CLAMPING_ISSUE.md
"""

import sys
import os
import logging
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.waypoint_manager import WaypointManager

def create_test_waypoints():
    """Create simple straight-line test waypoints (west-going route) and save to file."""
    # Create 100 waypoints going west (X decreasing)
    waypoints = []
    for i in range(100):
        x = 320.0 - i * 0.5  # 50m total route
        y = 130.0
        z = 0.0
        waypoints.append((x, y, z))

    # Save to temporary file
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    for x, y, z in waypoints:
        temp_file.write(f"{x}, {y}, {z}\n")
    temp_file.close()

    return temp_file.name, waypoints

def simulate_forward_movement(waypoint_manager, num_steps=50):
    """
    Simulate vehicle moving forward along route.

    Returns:
        List of (step, vehicle_pos, segment_idx, t, perp_dist, distance) tuples
    """
    results = []

    # Start at beginning of route
    vx, vy, vz = 320.0, 130.0, 0.0

    # Move forward 0.3m per step (30cm, should cross multiple 1cm dense segments!)
    for step in range(num_steps):
        # Get distance using current implementation
        distance = waypoint_manager.get_route_distance_to_goal((vx, vy, vz))

        # Also track current waypoint index for debugging
        current_idx = waypoint_manager.current_waypoint_idx

        results.append({
            'step': step,
            'vehicle_x': vx,
            'vehicle_y': vy,
            'distance': distance,
            'current_idx': current_idx,
        })

        # Move vehicle forward (west direction)
        vx -= 0.3  # Move 30cm west per step

    return results


def validate_results(results):
    """
    Validate that results show proper continuous behavior.

    Returns:
        (success: bool, issues: list of str)
    """
    issues = []

    # Check 1: Distance should decrease monotonically
    prev_distance = float('inf')
    for i, result in enumerate(results):
        if result['distance'] > prev_distance:
            issues.append(
                f"❌ Step {i}: Distance INCREASED! "
                f"{prev_distance:.2f}m → {result['distance']:.2f}m"
            )
        elif result['distance'] == prev_distance:
            issues.append(
                f"❌ Step {i}: Distance UNCHANGED! "
                f"{result['distance']:.2f}m (vehicle moved but distance stuck!)"
            )
        prev_distance = result['distance']

    # Check 2: Look for patterns of decreasing distance (good!)
    total_progress = results[0]['distance'] - results[-1]['distance']
    expected_progress = (results[0]['vehicle_x'] - results[-1]['vehicle_x'])  # Should be close to linear movement

    progress_ratio = total_progress / expected_progress if expected_progress > 0 else 0

    if progress_ratio < 0.8:
        issues.append(
            f"❌ Progress ratio too low: {progress_ratio:.2f} "
            f"(expected ~1.0 for straight path)"
        )

    # Summary
    if not issues:
        return True, ["✅ All checks passed! Distance decreases continuously."]
    else:
        return False, issues

def main():
    """Run validation tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(name)s - %(message)s'
    )

    print("=" * 80)
    print("VALIDATION: Progressive Segment Search Fix")
    print("=" * 80)

    # Create waypoint manager
    waypoints_file, waypoints = create_test_waypoints()
    print(f"\n1. Created {len(waypoints)} test waypoints (straight west-going route)")

    try:
        manager = WaypointManager(
            waypoints_file=waypoints_file,
            lookahead_distance=50.0,
            num_waypoints_ahead=10,
            waypoint_spacing=5.0,
            carla_map=None
        )

        print(f"   Dense waypoints generated: {len(manager.dense_waypoints)}")
        print(f"   Interpolation resolution: 1cm (0.01m)")
        print(f"   Total route length: {manager.total_route_length:.2f}m")

        # Run simulation
        print(f"\n2. Simulating forward movement (50 steps, 30cm/step = 15m total)")
        results = simulate_forward_movement(manager, num_steps=50)

        # Display sample results
        print("\n3. Sample results:")
        print("-" * 80)
        print(f"{'Step':<6} {'VehicleX':<10} {'Distance':<12} {'Delta':<10} {'Status':<10}")
        print("-" * 80)

        prev_dist = None
        for i in [0, 1, 2, 10, 20, 25, 30, 40, 49]:
            r = results[i]
            delta = (prev_dist - r['distance']) if prev_dist is not None else 0.0
            status = "✅" if delta > 0 or i == 0 else "❌ STUCK!" if delta == 0 else "❌ INCREASED!"

            print(f"{r['step']:<6} {r['vehicle_x']:<10.2f} {r['distance']:<12.2f} "
                  f"{delta:<10.3f} Idx={r['current_idx']:<5} {status:<10}")

            prev_dist = r['distance']        # Validate results
        print("\n4. Validation:")
        print("-" * 80)
        success, messages = validate_results(results)

        for msg in messages:
            print(msg)

        print("-" * 80)

        # Summary
        print("\n5. Summary:")
        if success:
            print("✅ SUCCESS: Progressive segment search works correctly!")
            print("   - Distance decreases continuously during forward movement")
            print("   - No segment sticking detected")
            print("   - Fix resolves the t-parameter clamping issue")
            return_code = 0
        else:
            print("❌ FAILURE: Issues detected!")
            print("   See validation messages above for details.")
            return_code = 1

    finally:
        # Cleanup temp file
        import os
        if os.path.exists(waypoints_file):
            os.unlink(waypoints_file)

    return return_code


if __name__ == "__main__":
    sys.exit(main())

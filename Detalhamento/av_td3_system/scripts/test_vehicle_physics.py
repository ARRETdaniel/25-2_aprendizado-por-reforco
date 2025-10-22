#!/usr/bin/env python3
"""
Test Vehicle Physics - Verify CARLA vehicle can move with fixed controls.

This script verifies that the vehicle physics are working correctly by applying
fixed throttle values and observing the vehicle's movement.

Usage:
    python3 scripts/test_vehicle_physics.py --throttle 0.5 --duration 100

Requirements:
    - CARLA server running on port 2000
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add workspace to path
sys.path.insert(0, '/workspace')

from src.environment.carla_env import CARLANavigationEnv


def test_vehicle_physics(throttle: float, steering: float, duration: int):
    """
    Test vehicle physics with fixed control inputs.

    Args:
        throttle: Fixed throttle value [0.0, 1.0]
        steering: Fixed steering value [-1.0, 1.0]
        duration: Number of steps to run

    Returns:
        dict: Test results including max speed achieved
    """
    print("=" * 80)
    print("🚗 VEHICLE PHYSICS TEST")
    print("=" * 80)
    print(f"Test parameters:")
    print(f"  - Throttle: {throttle:.2f} (0.0 = none, 1.0 = full)")
    print(f"  - Steering: {steering:.2f} (-1.0 = full left, 0.0 = straight, 1.0 = full right)")
    print(f"  - Duration: {duration} steps")
    print()

    # Initialize environment
    print("🌍 Initializing CARLA environment...")

    env = CARLANavigationEnv(
        carla_config_path='/workspace/config/carla_config.yaml',
        td3_config_path='/workspace/config/td3_config.yaml',
        training_config_path='/workspace/config/training_config.yaml'
    )
    print("✅ Environment ready")

    # Reset environment
    print("\n🔄 Resetting environment...")
    obs_dict, info = env.reset()
    print("✅ Environment reset complete")

    # Run test with fixed actions
    print(f"\n🏁 Running {duration}-step test with fixed controls...")
    print()

    # Fixed action: [steering, throttle/brake]
    # throttle/brake > 0 = throttle, < 0 = brake
    action = np.array([steering, throttle], dtype=np.float32)

    max_speed = 0.0
    speeds = []

    for step in range(duration):
        # Apply fixed action
        obs_dict, reward, done, truncated, info = env.step(action)

        # Get current speed from info
        speed = info.get('speed', 0.0)
        speeds.append(speed)
        max_speed = max(max_speed, speed)

        # Log progress every 10 steps
        if step % 10 == 0 or step < 10:
            print(f"Step {step:3d}/{duration} | Speed: {speed:6.2f} km/h | "
                  f"Reward: {reward:7.3f} | Max Speed: {max_speed:6.2f} km/h")

        # Check termination
        if done or truncated:
            print(f"\n⚠️  Episode terminated at step {step}")
            print(f"   Reason: {'done' if done else 'truncated'}")
            break

    # Calculate statistics
    avg_speed = np.mean(speeds)
    final_speed = speeds[-1] if speeds else 0.0

    # Display results
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    print(f"Total steps: {len(speeds)}")
    print(f"Average speed: {avg_speed:.2f} km/h")
    print(f"Maximum speed: {max_speed:.2f} km/h")
    print(f"Final speed: {final_speed:.2f} km/h")
    print()

    # Evaluate results
    print("📈 EVALUATION:")
    if max_speed > 5.0:
        print(f"✅ SUCCESS - Vehicle moved! Max speed: {max_speed:.2f} km/h")
        print("   Physics are working correctly.")
    elif max_speed > 0.5:
        print(f"⚠️  PARTIAL - Minimal movement detected: {max_speed:.2f} km/h")
        print("   Vehicle might need higher throttle or longer duration.")
    else:
        print(f"❌ FAILURE - No movement detected (max speed: {max_speed:.2f} km/h)")
        print("   Possible issues:")
        print("   - Throttle too low to overcome friction")
        print("   - Vehicle physics configuration problem")
        print("   - Control application issue")
    print()

    # Cleanup
    env.close()

    return {
        'throttle': throttle,
        'steering': steering,
        'duration': len(speeds),
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'final_speed': final_speed,
        'success': max_speed > 5.0
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test vehicle physics with fixed control inputs"
    )
    parser.add_argument(
        '--throttle',
        type=float,
        default=0.5,
        help='Fixed throttle value [0.0, 1.0] (default: 0.5)'
    )
    parser.add_argument(
        '--steering',
        type=float,
        default=0.0,
        help='Fixed steering value [-1.0, 1.0] (default: 0.0 = straight)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=100,
        help='Number of steps to run (default: 100)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not 0.0 <= args.throttle <= 1.0:
        print(f"❌ Error: Throttle must be in range [0.0, 1.0], got {args.throttle}")
        return 1

    if not -1.0 <= args.steering <= 1.0:
        print(f"❌ Error: Steering must be in range [-1.0, 1.0], got {args.steering}")
        return 1

    if args.duration <= 0:
        print(f"❌ Error: Duration must be positive, got {args.duration}")
        return 1

    # Run test
    try:
        results = test_vehicle_physics(
            throttle=args.throttle,
            steering=args.steering,
            duration=args.duration
        )

        # Return exit code based on success
        return 0 if results['success'] else 1

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

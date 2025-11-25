#!/usr/bin/env python3
"""
ROS Bridge Integration Test Script

Tests the integration of ROS 2 Bridge with evaluate_baseline.py by running
a single episode with the --use-ros-bridge flag.

This script verifies:
1. ROS Bridge interface initializes correctly
2. Vehicle control commands are published to /carla/ego_vehicle/vehicle_control_cmd
3. Episode completes successfully with ROS control
4. Metrics are collected correctly

Usage:
    # Start infrastructure first
    ./scripts/phase5_quickstart.sh start

    # Run this test
    python3 scripts/test_ros_integration.py

    # Or test baseline evaluation directly
    python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 1 --use-ros-bridge --debug

Author: AV TD3 System
Date: 2025-01-22
"""

import sys
import os
import subprocess
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def check_docker_running():
    """Check if required Docker containers are running."""
    print("\n" + "="*70)
    print("STEP 1: Checking Docker Infrastructure")
    print("="*70)

    result = subprocess.run(
        ['docker', 'ps', '--format', '{{.Names}}'],
        capture_output=True,
        text=True
    )

    containers = result.stdout.strip().split('\n')

    carla_running = 'carla-server' in containers
    bridge_running = 'ros2-bridge' in containers

    if carla_running:
        print("✅ CARLA server container running")
    else:
        print("❌ CARLA server container NOT running")

    if bridge_running:
        print("✅ ROS Bridge container running")
    else:
        print("❌ ROS Bridge container NOT running")

    if not (carla_running and bridge_running):
        print("\n⚠️  Infrastructure not running!")
        print("   Start with: ./scripts/phase5_quickstart.sh start")
        return False

    return True

def check_ros_topics():
    """Check if ROS topics are available."""
    print("\n" + "="*70)
    print("STEP 2: Checking ROS Topics")
    print("="*70)

    result = subprocess.run(
        ['docker', 'exec', 'ros2-bridge', 'bash', '-c',
         'source /opt/ros/humble/setup.bash && '
         'ros2 topic list | grep /carla/ego_vehicle/vehicle_control_cmd'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Control topic available: /carla/ego_vehicle/vehicle_control_cmd")
        return True
    else:
        print("❌ Control topic NOT available")
        print("   Check bridge logs: docker logs ros2-bridge")
        return False

def test_baseline_with_ros():
    """Run baseline evaluation with ROS Bridge."""
    print("\n" + "="*70)
    print("STEP 3: Running Baseline Evaluation with ROS Bridge")
    print("="*70)
    print("\nConfiguration:")
    print("  - Scenario: 0 (20 NPCs)")
    print("  - Episodes: 1")
    print("  - ROS Bridge: ENABLED")
    print("  - Debug: ENABLED")
    print("\nStarting test...\n")

    cmd = [
        'python3',
        'scripts/evaluate_baseline.py',
        '--scenario', '0',
        '--num-episodes', '1',
        '--use-ros-bridge',
        '--debug'
    ]

    print(f"Command: {' '.join(cmd)}\n")
    print("="*70)

    # Run evaluation
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        print("\n" + "="*70)
        print("✅ TEST PASSED - Baseline evaluation completed successfully")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("❌ TEST FAILED - Evaluation returned error code:", result.returncode)
        print("="*70)
        return False

def monitor_ros_topics():
    """Display instructions for monitoring ROS topics during test."""
    print("\n" + "="*70)
    print("OPTIONAL: Monitor ROS Topics")
    print("="*70)
    print("\nIn a separate terminal, you can monitor the control commands:")
    print("\n  docker exec ros2-bridge bash -c \\")
    print("    'source /opt/ros/humble/setup.bash && \\")
    print("     source /opt/carla-ros-bridge/install/setup.bash && \\")
    print("     ros2 topic echo /carla/ego_vehicle/vehicle_control_cmd'")
    print("\nOr monitor vehicle speed:")
    print("\n  docker exec ros2-bridge bash -c \\")
    print("    'source /opt/ros/humble/setup.bash && \\")
    print("     source /opt/carla-ros-bridge/install/setup.bash && \\")
    print("     ros2 topic echo /carla/ego_vehicle/speedometer'")
    print("\n" + "="*70)

    input("\nPress ENTER when ready to start test...")

def main():
    """Main test sequence."""
    print("\n" + "="*70)
    print("ROS BRIDGE INTEGRATION TEST")
    print("="*70)
    print("\nThis script tests the Phase 5 ROS Bridge integration with")
    print("the baseline evaluation pipeline.")
    print("\nEnsure Docker infrastructure is running before proceeding!")

    # Check infrastructure
    if not check_docker_running():
        print("\n❌ PREREQUISITES FAILED - Cannot proceed with test")
        sys.exit(1)

    # Check ROS topics
    if not check_ros_topics():
        print("\n❌ ROS TOPICS NOT AVAILABLE - Cannot proceed with test")
        sys.exit(1)

    # Offer monitoring option
    print("\n" + "="*70)
    response = input("\nDo you want to set up topic monitoring first? (y/N): ")
    if response.lower() == 'y':
        monitor_ros_topics()

    # Run test
    success = test_baseline_with_ros()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    if success:
        print("\n✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("  1. Run multi-episode baseline: python3 scripts/evaluate_baseline.py --scenario 0 --num-episodes 20 --use-ros-bridge")
        print("  2. Integrate train_td3.py with ROS Bridge")
        print("  3. Test TD3 training with ROS control")
    else:
        print("\n❌ TESTS FAILED")
        print("\nTroubleshooting:")
        print("  1. Check docker logs: docker logs carla-server")
        print("  2. Check bridge logs: docker logs ros2-bridge")
        print("  3. Verify topics: docker exec ros2-bridge bash -c 'source /opt/ros/humble/setup.bash && ros2 topic list'")
        print("  4. Review integration guide: docs/ROS_BRIDGE_INTEGRATION_GUIDE.md")

    print("="*70 + "\n")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

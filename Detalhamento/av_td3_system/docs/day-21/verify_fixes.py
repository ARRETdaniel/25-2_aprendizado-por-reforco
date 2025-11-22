#!/usr/bin/env python3
"""
Verification script for right-turn bias fixes.

This script verifies that all three fixes have been properly implemented:
1. PBRS code removed from reward_functions.py
2. Route distance method implemented in waypoint_manager.py
3. Lane invasion penalty increased to -50.0 in config

Usage:
    python verify_fixes.py
"""

import os
import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, success=True):
    """Print colored status message."""
    color = Colors.GREEN if success else Colors.RED
    symbol = "✓" if success else "✗"
    print(f"{color}{symbol} {message}{Colors.ENDC}")

def print_section(title):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.ENDC}\n")

def check_file_exists(filepath):
    """Check if file exists."""
    if not os.path.exists(filepath):
        print_status(f"File not found: {filepath}", success=False)
        return False
    return True

def verify_fix1_pbrs_removed():
    """Verify Fix #1: PBRS code has been removed/commented."""
    print_section("Fix #1: PBRS Removal Verification")

    filepath = "src/environment/reward_functions.py"
    if not check_file_exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check that PBRS code is commented out (handle various comment styles)
    pbrs_active = False
    pbrs_commented = False

    for line in content.splitlines():
        # Check if line contains the PBRS assignment
        if "pbrs_reward = self.gamma * potential_current - potential_prev" in line:
            # Check if it's commented (handle spaces after #)
            if line.strip().startswith('#'):
                pbrs_commented = True
            else:
                pbrs_active = True
                break

    if pbrs_commented or not pbrs_active:
        print_status("PBRS code is properly commented out or removed", success=True)
        return True
    else:
        print_status("PBRS code is still active (not commented)!", success=False)
        return False

def verify_fix2_route_distance():
    """Verify Fix #2: Route distance method implemented."""
    print_section("Fix #2: Route Distance Implementation Verification")

    filepath = "src/environment/waypoint_manager.py"
    if not check_file_exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    checks = {
        "get_route_distance_to_goal method exists":
            "def get_route_distance_to_goal(" in content,
        "_find_nearest_waypoint_index helper exists":
            "def _find_nearest_waypoint_index(" in content,
        "Route distance documented with CARLA reference":
            "CARLA Waypoint API" in content or "core_map" in content,
        "Fallback to Euclidean for off-route":
            "return self.get_distance_to_goal(vehicle_location)" in content
    }

    all_passed = True
    for check_name, check_result in checks.items():
        print_status(check_name, success=check_result)
        if not check_result:
            all_passed = False

    # Check carla_env.py uses route distance
    env_filepath = "src/environment/carla_env.py"
    if check_file_exists(env_filepath):
        with open(env_filepath, 'r') as f:
            env_content = f.read()

        route_distance_used = "get_route_distance_to_goal(" in env_content
        print_status("carla_env.py uses get_route_distance_to_goal()", success=route_distance_used)

        if not route_distance_used:
            all_passed = False

    return all_passed

def verify_fix3_lane_penalty():
    """Verify Fix #3: Lane invasion penalty increased to -50.0."""
    print_section("Fix #3: Lane Invasion Penalty Verification")

    filepath = "config/carla_config.yaml"
    if not check_file_exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check for lane_invasion_penalty: -50.0
    penalty_found = "lane_invasion_penalty: -50.0" in content

    if penalty_found:
        print_status("Lane invasion penalty set to -50.0 in config", success=True)
    else:
        # Check if it's still -10.0
        old_penalty = "invasion_penalty: -10.0" in content or "lane_invasion_penalty: -10.0" in content
        if old_penalty:
            print_status("Lane invasion penalty still -10.0 (should be -50.0)!", success=False)
            return False
        else:
            print_status("Lane invasion penalty value not found in expected format", success=False)
            return False

    # Check default in reward_functions.py
    reward_filepath = "src/environment/reward_functions.py"
    if check_file_exists(reward_filepath):
        with open(reward_filepath, 'r') as f:
            reward_content = f.read()

        default_found = '"lane_invasion_penalty", -50.0' in reward_content
        print_status("Default lane_invasion_penalty is -50.0 in code", success=default_found)

    return penalty_found

def main():
    """Run all verification checks."""
    print(f"\n{Colors.BOLD}Right-Turn Bias Fixes - Verification Script{Colors.ENDC}")
    print(f"{Colors.BOLD}Date: 2025-11-21{Colors.ENDC}")

    # Change to av_td3_system directory if not already there
    if not os.path.exists("src/environment"):
        av_td3_path = "av_td3_system"
        if os.path.exists(av_td3_path):
            os.chdir(av_td3_path)
            print(f"\n{Colors.YELLOW}Changed directory to: {os.getcwd()}{Colors.ENDC}")
        else:
            print(f"\n{Colors.RED}Error: Cannot find av_td3_system directory!{Colors.ENDC}")
            sys.exit(1)

    results = {}

    # Run all checks
    results['Fix #1 (PBRS Removal)'] = verify_fix1_pbrs_removed()
    results['Fix #2 (Route Distance)'] = verify_fix2_route_distance()
    results['Fix #3 (Lane Penalty)'] = verify_fix3_lane_penalty()

    # Summary
    print_section("Verification Summary")

    all_passed = all(results.values())

    for fix_name, passed in results.items():
        print_status(f"{fix_name}: {'PASSED' if passed else 'FAILED'}", success=passed)

    print()
    if all_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.GREEN}✓ ALL FIXES VERIFIED SUCCESSFULLY{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.ENDC}")
        print(f"\n{Colors.GREEN}Status: Ready for testing{Colors.ENDC}")
        print(f"{Colors.GREEN}Next step: Run Phase 1 validation (1K steps){Colors.ENDC}\n")
        return 0
    else:
        print(f"{Colors.BOLD}{Colors.RED}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.RED}✗ VERIFICATION FAILED{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.RED}{'='*70}{Colors.ENDC}")
        print(f"\n{Colors.RED}Some fixes were not properly implemented.{Colors.ENDC}")
        print(f"{Colors.RED}Please review the failures above.{Colors.ENDC}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

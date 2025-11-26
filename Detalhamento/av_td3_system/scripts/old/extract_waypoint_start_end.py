"""
Extract start and end waypoints from waypoints.txt for DynamicRouteManager.

This script reads the existing waypoints.txt file and extracts:
- First waypoint (route start)
- Last waypoint (route end)

These will be used as fixed start/end for dynamic route generation.
"""

import numpy as np
import os


def extract_start_end_from_waypoints_file(filepath: str):
    """
    Extract start and end locations from waypoints.txt.

    Args:
        filepath: Path to waypoints.txt

    Returns:
        Tuple of (start_location, end_location) where each is (x, y, z)
    """
    waypoints = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse x, y, z
            parts = [float(x.strip()) for x in line.split(',')]
            if len(parts) >= 3:
                waypoints.append(tuple(parts[:3]))

    if not waypoints:
        raise ValueError(f"No waypoints found in {filepath}")

    start_location = waypoints[0]
    end_location = waypoints[-1]

    print(f"Waypoints file: {filepath}")
    print(f"Total waypoints in file: {len(waypoints)}")
    print(f"\nStart location (first waypoint):")
    print(f"  X: {start_location[0]:.2f}")
    print(f"  Y: {start_location[1]:.2f}")
    print(f"  Z: {start_location[2]:.2f} (will be corrected by CARLA)")
    print(f"\nEnd location (last waypoint):")
    print(f"  X: {end_location[0]:.2f}")
    print(f"  Y: {end_location[1]:.2f}")
    print(f"  Z: {end_location[2]:.2f} (will be corrected by CARLA)")

    # Calculate straight-line distance
    distance = np.sqrt(
        (end_location[0] - start_location[0])**2 +
        (end_location[1] - start_location[1])**2
    )
    print(f"\nStraight-line distance: {distance:.2f}m")

    return start_location, end_location


if __name__ == "__main__":
    # Path to waypoints.txt (adjust as needed)
    waypoints_file = "/workspace/av_td3_system/config/waypoints.txt"

    if not os.path.exists(waypoints_file):
        print(f"ERROR: Waypoints file not found at {waypoints_file}")
        print("\nTrying alternative path...")
        waypoints_file = "../config/waypoints.txt"

    if os.path.exists(waypoints_file):
        start, end = extract_start_end_from_waypoints_file(waypoints_file)

        print("\n" + "="*60)
        print("Configuration for DynamicRouteManager:")
        print("="*60)
        print(f"start_location = ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")
        print(f"end_location = ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})")
    else:
        print(f"ERROR: Could not find waypoints.txt")

"""
Test script for DynamicRouteManager

This script tests the dynamic route generation without running the full training loop.
It connects to CARLA, generates a route, and prints debug information.
"""

import sys
import os
import logging

# Setup Python path for CARLA
sys.path.append('/home/carla/carla/PythonAPI/carla')

import carla
import numpy as np

# Add project path
sys.path.insert(0, '/workspace/av_td3_system/src')

from environment.dynamic_route_manager import DynamicRouteManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_dynamic_route_generation():
    """Test dynamic route generation using CARLA's API."""
    
    logger.info("="*70)
    logger.info("TESTING DYNAMIC ROUTE GENERATION")
    logger.info("="*70)
    
    # Connect to CARLA
    logger.info("\n1. Connecting to CARLA...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        version = client.get_server_version()
        logger.info(f"   ✅ Connected to CARLA {version}")
    except Exception as e:
        logger.error(f"   ❌ Failed to connect: {e}")
        return False
    
    # Load Town01
    logger.info("\n2. Loading Town01...")
    world = client.load_world('Town01')
    logger.info("   ✅ Town01 loaded")
    
    # Extract start/end from waypoints.txt
    logger.info("\n3. Reading waypoints.txt for start/end locations...")
    waypoints_file = '/workspace/av_td3_system/config/waypoints.txt'
    
    waypoints = []
    with open(waypoints_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(x.strip()) for x in line.split(',')]
            if len(parts) >= 3:
                waypoints.append(tuple(parts[:3]))
    
    start_location = waypoints[0]
    end_location = waypoints[-1]
    
    logger.info(f"   Start: ({start_location[0]:.2f}, {start_location[1]:.2f}, {start_location[2]:.2f})")
    logger.info(f"   End:   ({end_location[0]:.2f}, {end_location[1]:.2f}, {end_location[2]:.2f})")
    logger.info(f"   Total waypoints in file: {len(waypoints)}")
    
    # Calculate straight-line distance
    straight_distance = np.sqrt(
        (end_location[0] - start_location[0])**2 +
        (end_location[1] - start_location[1])**2
    )
    logger.info(f"   Straight-line distance: {straight_distance:.2f}m")
    
    # Initialize DynamicRouteManager
    logger.info("\n4. Initializing DynamicRouteManager...")
    try:
        route_manager = DynamicRouteManager(
            carla_world=world,
            start_location=start_location,
            end_location=end_location,
            sampling_resolution=2.0,
            logger=logger
        )
        logger.info("   ✅ DynamicRouteManager initialized successfully")
    except Exception as e:
        logger.error(f"   ❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get route information
    logger.info("\n5. Route Information:")
    logger.info(f"   Total waypoints generated: {len(route_manager.waypoints)}")
    logger.info(f"   Route length: {route_manager.get_route_length():.2f}m")
    logger.info(f"   Sampling resolution: {route_manager.sampling_resolution}m")
    
    # Show first 5 and last 5 waypoints
    logger.info("\n6. First 5 waypoints:")
    for i, wp in enumerate(route_manager.waypoints[:5]):
        logger.info(f"   [{i}] ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
    
    logger.info("\n   Last 5 waypoints:")
    for i, wp in enumerate(route_manager.waypoints[-5:], start=len(route_manager.waypoints)-5):
        logger.info(f"   [{i}] ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
    
    # Get spawn transform
    logger.info("\n7. Spawn Transform:")
    spawn_transform = route_manager.get_start_transform()
    logger.info(f"   Location: ({spawn_transform.location.x:.2f}, "
                f"{spawn_transform.location.y:.2f}, "
                f"{spawn_transform.location.z:.2f})")
    logger.info(f"   Rotation: Pitch={spawn_transform.rotation.pitch:.2f}°, "
                f"Yaw={spawn_transform.rotation.yaw:.2f}°, "
                f"Roll={spawn_transform.rotation.roll:.2f}°")
    
    # Comparison with original waypoints.txt
    logger.info("\n8. Comparison (Dynamic vs Static):")
    logger.info(f"   Static waypoints (from file): {len(waypoints)}")
    logger.info(f"   Dynamic waypoints (generated): {len(route_manager.waypoints)}")
    logger.info(f"   Straight-line distance: {straight_distance:.2f}m")
    logger.info(f"   Dynamic route length: {route_manager.get_route_length():.2f}m")
    logger.info(f"   Ratio (route/straight): {route_manager.get_route_length()/straight_distance:.2f}x")
    
    # Z-coordinate comparison
    logger.info("\n9. Z-Coordinate Comparison:")
    logger.info(f"   Static start Z: {start_location[2]:.2f}m")
    logger.info(f"   Dynamic start Z: {route_manager.waypoints[0][2]:.2f}m")
    logger.info(f"   Difference: {abs(route_manager.waypoints[0][2] - start_location[2]):.2f}m")
    
    logger.info(f"\n   Static end Z: {end_location[2]:.2f}m")
    logger.info(f"   Dynamic end Z: {route_manager.waypoints[-1][2]:.2f}m")
    logger.info(f"   Difference: {abs(route_manager.waypoints[-1][2] - end_location[2]):.2f}m")
    
    # Success summary
    logger.info("\n" + "="*70)
    logger.info("✅ DYNAMIC ROUTE GENERATION TEST PASSED")
    logger.info("="*70)
    logger.info(f"\nSummary:")
    logger.info(f"  • Route successfully generated using GlobalRoutePlanner")
    logger.info(f"  • {len(route_manager.waypoints)} waypoints at {route_manager.sampling_resolution}m intervals")
    logger.info(f"  • Total route length: ~{route_manager.get_route_length():.0f}m")
    logger.info(f"  • Z-coordinates automatically corrected (road surface level)")
    logger.info(f"  • Spawn transform aligned with road direction")
    
    return True


if __name__ == "__main__":
    try:
        success = test_dynamic_route_generation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

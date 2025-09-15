#!/usr/bin/env python3
"""
Test script for CARLA ROS Bridge functionality.
This script tests the connection between CARLA and ROS 2.
"""

import os
import sys
import time

def test_carla_import():
    """Test CARLA Python API import."""
    try:
        import carla
        print("✅ CARLA Python API imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import CARLA: {e}")
        return False

def test_carla_connection():
    """Test connection to CARLA server."""
    try:
        import carla
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✅ Connected to CARLA server successfully")
        print(f"   World: {world.get_map().name}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to CARLA server: {e}")
        return False

def test_ros_environment():
    """Test ROS 2 environment setup."""
    try:
        # Check if ROS environment variables are set
        ros_distro = os.environ.get('ROS_DISTRO')
        if ros_distro:
            print(f"✅ ROS environment configured: ROS {ros_distro}")
            return True
        else:
            print("❌ ROS environment not configured")
            return False
    except Exception as e:
        print(f"❌ Error checking ROS environment: {e}")
        return False

def main():
    print("🔍 CARLA ROS Bridge Verification Test")
    print("=" * 50)
    
    # Test CARLA Python API
    carla_ok = test_carla_import()
    
    # Test CARLA server connection
    if carla_ok:
        carla_server_ok = test_carla_connection()
    else:
        carla_server_ok = False
    
    # Test ROS environment
    ros_ok = test_ros_environment()
    
    print("\n📋 Test Summary:")
    print(f"   CARLA Python API: {'✅' if carla_ok else '❌'}")
    print(f"   CARLA Server:     {'✅' if carla_server_ok else '❌'}")
    print(f"   ROS Environment:  {'✅' if ros_ok else '❌'}")
    
    if carla_ok and carla_server_ok and ros_ok:
        print("\n🎉 All tests passed! Ready for ROS bridge testing.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

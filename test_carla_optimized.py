#!/usr/bin/env python3
"""
CARLA Optimized Test for RTX 2060 6GB VRAM
Based on CARLA documentation for low-memory systems
"""

import carla
import time
import sys
import os

def test_carla_optimized():
    """Test CARLA connection with GPU memory optimizations"""
    
    print("Testing CARLA with optimized settings for RTX 2060 6GB...")
    
    try:
        # Connect to CARLA server
        print("Connecting to CARLA server on localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Get world and settings
        world = client.get_world()
        settings = world.get_settings()
        
        print(f"Connected to CARLA world: {world.get_map().name}")
        
        # Apply GPU memory optimizations
        print("Applying GPU memory optimizations...")
        
        # Reduce simulation quality
        settings.quality_level = carla.QualityLevel.Low
        
        # Reduce number of rendering threads
        settings.no_rendering_mode = False  # Keep rendering but optimized
        
        # Set fixed time step for stability
        settings.fixed_delta_seconds = 0.05  # 20 FPS instead of default
        
        # Disable synchronous mode initially
        settings.synchronous_mode = False
        
        # Apply settings
        world.apply_settings(settings)
        print("Applied low-memory settings successfully")
        
        # Get available maps and suggest lighter alternatives
        available_maps = client.get_available_maps()
        print(f"\nAvailable maps: {len(available_maps)}")
        
        # Recommend lighter maps for RTX 2060
        light_maps = [m for m in available_maps if any(light in m.lower() 
                     for light in ['town01', 'town02', 'town03'])]
        
        if light_maps:
            print(f"Recommended lighter maps for your GPU: {light_maps}")
        
        # Test basic world operations
        print("\nTesting basic world operations...")
        actors = world.get_actors()
        print(f"Current actors in world: {len(actors)}")
        
        # Get spawn points (this tests map loading)
        spawn_points = world.get_map().get_spawn_points()
        print(f"Available spawn points: {len(spawn_points)}")
        
        print("\n✅ CARLA optimized test completed successfully!")
        print("Your CARLA installation is working with GPU optimizations.")
        
        return True
        
    except Exception as e:
        print(f"❌ CARLA test failed: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Make sure CARLA server is running")
        print("2. Check if port 2000 is available")
        print("3. Try launching CARLA with: --opengl -quality-level=Low")
        print("4. Consider switching to a lighter map like Town01")
        return False

if __name__ == "__main__":
    success = test_carla_optimized()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
🚀 CARLA Integration Showcase
Demonstrates real-time CARLA camera feeds with performance monitoring.
"""

import os
import sys
import time
import numpy as np
import cv2

# Add CARLA path
sys.path.append('CarlaSimulator/PythonAPI/carla/dist/carla-0.8.4-py3.6-win-amd64.egg')
sys.path.append('CarlaSimulator/PythonClient')

print("🚀 CARLA INTEGRATION SHOWCASE")
print("=" * 50)
print("🎯 Features Demonstrated:")
print("   ✅ CARLA 0.8.4 Connection")
print("   ✅ Real-time Camera Feeds")
print("   ✅ Vehicle Control")
print("   ✅ Performance Monitoring")
print()

def main():
    """Main CARLA integration demonstration."""
    try:
        import carla
        
        print("🔗 Connecting to CARLA...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        print("🚗 Spawning vehicle...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.mustang.mustang')
        
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        print("📷 Setting up camera...")
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        
        def process_image(image):
            nonlocal frame_count
            frame_count += 1
            
            # Convert to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            array = array[:, :, ::-1]  # BGR to RGB
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add overlay
            display_frame = array.copy()
            cv2.putText(display_frame, f"🚀 CARLA Integration Demo", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"📊 FPS: {fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"🎮 Frames: {frame_count:,}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Vehicle info
            try:
                velocity = vehicle.get_velocity()
                speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                cv2.putText(display_frame, f"🚗 Speed: {speed:.1f} km/h", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except:
                pass
            
            cv2.putText(display_frame, "✅ Integration Complete", (10, 570),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press Q to quit", (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('🚀 CARLA Integration Demo', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 User requested quit")
                return False
            
            return True
        
        # Setup camera callback
        camera.listen(process_image)
        
        # Enable autopilot
        print("🚘 Enabling autopilot...")
        vehicle.set_autopilot(True)
        
        print("✅ Demo running! Press Q in the camera window to quit.")
        
        # Main loop
        running = True
        while running:
            time.sleep(0.1)
            
            # Check if window was closed
            try:
                cv2.getWindowProperty('🚀 CARLA Integration Demo', cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break
        
        print("🧹 Cleaning up...")
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        
        print("🎉 CARLA Integration Demo Complete!")
        print(f"📊 Performance: {frame_count:,} frames processed")
        
    except ImportError:
        print("❌ CARLA module not found. Make sure CARLA is installed.")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    main()

"""
Simplified Enhanced CARLA Client for Real-time Visualization
Building upon module_7.py patterns for maximum compatibility
"""

from __future__ import print_function, division
import sys
import os
import time
import cv2
import numpy as np
import argparse

# Add CARLA Python API to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CarlaSimulator', 'PythonClient')))

from carla.client import make_carla_client, VehicleControl
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.image_converter import to_rgb_array

def make_carla_settings():
    """Create CARLA settings similar to module_7.py"""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=2,
        NumberOfPedestrians=0,
        WeatherId=1,
        QualityLevel='Low'
    )
    
    # Add RGB camera
    camera_rgb = Camera('CameraRGB')
    camera_rgb.set_image_size(640, 480)
    camera_rgb.set_position(0.30, 0, 1.30)  # Similar to module_7.py
    camera_rgb.set_rotation(0, 0, 0)
    settings.add_sensor(camera_rgb)
    
    # Add depth camera
    camera_depth = Camera('CameraDepth', PostProcessing='Depth')
    camera_depth.set_image_size(640, 480)
    camera_depth.set_position(0.30, 0, 1.30)
    camera_depth.set_rotation(0, 0, 0)
    settings.add_sensor(camera_depth)
    
    return settings

def run_carla_client(host='localhost', port=2000):
    """Run the enhanced CARLA client with real-time visualization"""
    
    print("🚗 Starting Enhanced CARLA Client...")
    print(f"📡 Connecting to CARLA server at {host}:{port}")
    
    try:
        with make_carla_client(host, port) as client:
            print("✅ Connected to CARLA server")
            
            # Setup environment
            settings = make_carla_settings()
            scene = client.load_settings(settings)
            
            # Start episode
            client.start_episode(0)  # Random start position
            print("✅ Episode started")
            
            # Create OpenCV windows
            cv2.namedWindow('CARLA Camera RGB', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('CARLA Camera Depth', cv2.WINDOW_AUTOSIZE)
            
            print("🎥 Real-time camera visualization started")
            print("Press 'q' to quit, 'SPACE' for autopilot, 'r' to restart")
            
            frame_count = 0
            start_time = time.time()
            autopilot_enabled = False
            
            while True:
                try:
                    # Read sensor data
                    measurements, sensor_data = client.read_data()
                    
                    # Process camera data
                    if 'CameraRGB' in sensor_data:
                        rgb_image = sensor_data['CameraRGB']
                        rgb_array = to_rgb_array(rgb_image)
                        
                        # Convert RGB to BGR for OpenCV
                        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                        
                        # Add frame info overlay
                        fps = frame_count / (time.time() - start_time + 0.001)
                        info_text = f"Frame: {frame_count:04d} | FPS: {fps:.1f} | Auto: {autopilot_enabled}"
                        cv2.putText(bgr_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add vehicle state info
                        speed = measurements.player_measurements.forward_speed * 3.6  # Convert to km/h
                        pos = measurements.player_measurements.transform.location
                        state_text = f"Speed: {speed:.1f} km/h | Pos: ({pos.x:.1f}, {pos.y:.1f})"
                        cv2.putText(bgr_image, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        cv2.imshow('CARLA Camera RGB', bgr_image)
                    
                    # Process depth camera
                    if 'CameraDepth' in sensor_data:
                        depth_image = sensor_data['CameraDepth']
                        depth_array = to_rgb_array(depth_image)
                        depth_gray = cv2.cvtColor(depth_array, cv2.COLOR_RGB2GRAY)
                        cv2.imshow('CARLA Camera Depth', depth_gray)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("🛑 Quit requested")
                        break
                    elif key == ord(' '):  # Space for autopilot
                        autopilot_enabled = not autopilot_enabled
                        print(f"🤖 Autopilot: {'ON' if autopilot_enabled else 'OFF'}")
                    elif key == ord('r'):  # Restart episode
                        print("🔄 Restarting episode...")
                        client.start_episode(0)
                        frame_count = 0
                        start_time = time.time()
                    
                    # Send control commands
                    if autopilot_enabled:
                        # Use autopilot control
                        client.send_control(measurements.player_measurements.autopilot_control)
                    else:
                        # Manual control (stationary for now)
                        control = VehicleControl()
                        control.throttle = 0.0
                        control.steer = 0.0
                        control.brake = 1.0
                        control.hand_brake = True
                        control.reverse = False
                        client.send_control(control)
                    
                    frame_count += 1
                    
                    # Print status every 100 frames
                    if frame_count % 100 == 0:
                        print(f"📊 Status: Frame {frame_count}, FPS: {fps:.1f}")
                    
                except KeyboardInterrupt:
                    print("\n🛑 Interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Error during simulation: {e}")
                    break
            
            # Cleanup
            cv2.destroyAllWindows()
            print("✅ Camera windows closed")
            
    except Exception as e:
        print(f"❌ Failed to connect to CARLA: {e}")
        return False
    
    print("👋 Enhanced CARLA Client finished")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced CARLA Client with Real-time Visualization')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    
    args = parser.parse_args()
    
    success = run_carla_client(args.host, args.port)
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()

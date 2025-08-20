#!/usr/bin/env python3
"""
Real-time Visual Monitor for CARLA DRL Training
Connects to the running training session and displays camera feed
"""

import cv2
import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import time
import threading

# Enable msgpack numpy support
m.patch()

class CarlaVisualMonitor:
    """
    Visual monitor that connects to ZMQ bridge to display CARLA camera feed.
    """
    
    def __init__(self, zmq_port=5555):
        self.zmq_port = zmq_port
        self.running = False
        self.context = None
        self.socket = None
        self.last_frame = None
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0
        
    def connect(self):
        """Connect to ZMQ bridge."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.zmq_port}")
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            print(f"🌉 Connected to ZMQ bridge on port {self.zmq_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to ZMQ bridge: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ZMQ bridge."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("🔌 Disconnected from ZMQ bridge")
    
    def process_frame(self, carla_data):
        """Process CARLA data and extract camera frame."""
        try:
            sensors = carla_data.get('sensors', {})
            measurements = carla_data.get('measurements', {})
            
            # Find camera data
            camera_data = None
            for sensor_name, sensor_value in sensors.items():
                if 'Camera' in sensor_name:
                    if isinstance(sensor_value, list):
                        camera_array = np.array(sensor_value, dtype=np.uint8)
                        if len(camera_array.shape) == 3 and camera_array.shape[2] == 3:
                            if camera_array.shape[:2] == (64, 64):
                                camera_data = camera_array
                                break
                        elif len(camera_array.shape) == 1:
                            try:
                                camera_data = camera_array.reshape(64, 64, 3)
                                break
                            except:
                                continue
            
            if camera_data is None:
                return None
            
            # Scale up for display
            display_image = cv2.resize(camera_data, (640, 640))
            
            # Convert RGB to BGR for OpenCV
            if len(display_image.shape) == 3:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            
            # Create info panel
            info_panel = np.zeros((640, 400, 3), dtype=np.uint8)
            
            # Title
            cv2.putText(info_panel, "CARLA Live Camera Feed", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Vehicle measurements
            position = measurements.get('position', [0.0, 0.0, 0.0])
            rotation = measurements.get('rotation', [0.0, 0.0, 0.0])
            velocity = measurements.get('velocity', 0.0)
            acceleration = measurements.get('acceleration', [0.0, 0.0, 0.0])
            
            # Display vehicle info
            y_offset = 100
            cv2.putText(info_panel, "Vehicle Status:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            y_offset += 40
            cv2.putText(info_panel, f"Position:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  X: {position[0]:.2f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Y: {position[1]:.2f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Z: {position[2]:.2f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 40
            cv2.putText(info_panel, f"Rotation:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Yaw: {rotation[1]:.2f}°", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 40
            cv2.putText(info_panel, f"Speed: {velocity:.1f} m/s", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset += 40
            cv2.putText(info_panel, f"Acceleration:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  X: {acceleration[0]:.2f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Y: {acceleration[1]:.2f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Performance info
            y_offset += 60
            cv2.putText(info_panel, f"Monitor FPS: {self.current_fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Controls
            y_offset += 80
            cv2.putText(info_panel, "Controls:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_offset += 30
            cv2.putText(info_panel, "ESC/Q - Close window", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
            cv2.putText(info_panel, "S - Save screenshot", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Combine camera and info
            combined = np.hstack([display_image, info_panel])
            
            return combined
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return None
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_timer)
            self.fps_counter = 0
            self.fps_timer = current_time
    
    def run(self):
        """Main display loop."""
        if not self.connect():
            return
        
        self.running = True
        print("🖼️ Visual monitor started. Press ESC or Q to quit.")
        
        try:
            while self.running:
                try:
                    # Receive CARLA data
                    carla_data = self.socket.recv()
                    data = msgpack.unpackb(carla_data, raw=False)
                    
                    # Process and display frame
                    frame = self.process_frame(data)
                    if frame is not None:
                        cv2.imshow('CARLA DRL - Live Camera Feed', frame)
                        self.last_frame = frame
                        self.update_fps()
                    
                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC or Q
                        self.running = False
                        break
                    elif key == ord('s'):  # Save screenshot
                        if self.last_frame is not None:
                            timestamp = int(time.time())
                            filename = f"carla_screenshot_{timestamp}.jpg"
                            cv2.imwrite(filename, self.last_frame)
                            print(f"📸 Screenshot saved: {filename}")
                
                except zmq.Again:
                    # Timeout, continue
                    continue
                except Exception as e:
                    print(f"⚠️ Data receive error: {e}")
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            self.disconnect()

def main():
    """Main function."""
    print("🎥 CARLA DRL Visual Monitor")
    print("=" * 50)
    print("Connecting to CARLA ZMQ bridge...")
    
    monitor = CarlaVisualMonitor(zmq_port=5555)
    monitor.run()
    
    print("👋 Visual monitor closed.")

if __name__ == "__main__":
    main()

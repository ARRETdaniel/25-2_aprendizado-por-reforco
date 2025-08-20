#!/usr/bin/env python3
"""
Immediate Speed Boost for Current Training
"""

import zmq
import msgpack
import time
import threading

class ContinuousSpeedBoost:
    """Continuously send speed-enhancing actions."""
    
    def __init__(self, boost_factor=1.3, min_throttle=0.4):
        self.boost_factor = boost_factor
        self.min_throttle = min_throttle
        self.running = False
        
        # Setup ZMQ
        self.context = zmq.Context()
        self.action_socket = self.context.socket(zmq.PUB)
        self.action_socket.bind("tcp://*:5558")  # Alternative port
        
        # Data socket to monitor CARLA
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.data_socket.setsockopt(zmq.RCVTIMEO, 100)
        self.data_socket.connect("tcp://localhost:5555")
        
        time.sleep(0.5)
        
    def boost_action(self, original_action):
        """Apply speed boost to an action."""
        steering = original_action.get('steering', 0.0)
        throttle = original_action.get('throttle', 0.0)
        brake = original_action.get('brake', 0.0)
        
        # Apply speed boost
        if throttle > 0:
            throttle = max(throttle * self.boost_factor, self.min_throttle)
            throttle = min(throttle, 1.0)
            brake = 0.0  # No braking when boosting
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'speed_boost': True,
            'aggressive': True
        }
    
    def monitor_and_boost(self):
        """Monitor CARLA data and send boosted actions."""
        print("🚀 Starting continuous speed boost monitoring...")
        
        while self.running:
            try:
                # Try to receive CARLA data
                try:
                    packed_data = self.data_socket.recv(zmq.NOBLOCK)
                    carla_data = msgpack.unpackb(packed_data, raw=False)
                    
                    # Extract current measurements
                    measurements = carla_data.get('measurements', {})
                    velocity = measurements.get('velocity', 0.0)
                    
                    # If car is going too slow, send boost command
                    if velocity < 15.0:  # Less than 15 km/h
                        boost_action = {
                            'steering': 0.0,
                            'throttle': 0.8,  # High throttle
                            'brake': 0.0,
                            'speed_boost': True,
                            'aggressive': True,
                            'force_speed': True
                        }
                        
                        packed_boost = msgpack.packb(boost_action)
                        self.action_socket.send(packed_boost)
                        
                        print(f"⚡ Speed boost sent! Current velocity: {velocity:.1f} km/h")
                
                except zmq.Again:
                    pass  # No data available
                    
                time.sleep(0.1)  # 10 Hz monitoring
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(1.0)
    
    def start(self):
        """Start the speed boost system."""
        print("🏎️ Continuous Speed Boost System")
        print("=" * 40)
        print(f"🎯 Boost Factor: {self.boost_factor}x")
        print(f"🚀 Minimum Throttle: {self.min_throttle}")
        print("🔄 Monitoring CARLA data for slow speeds...")
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_and_boost)
        self.monitor_thread.start()
        
        try:
            # Main loop - send periodic boost commands
            while True:
                # Send general speed boost every 5 seconds
                boost_action = {
                    'steering': 0.0,
                    'throttle': 0.7,
                    'brake': 0.0,
                    'speed_boost': True,
                    'aggressive': True,
                    'periodic_boost': True
                }
                
                packed_boost = msgpack.packb(boost_action)
                self.action_socket.send(packed_boost)
                
                print("🚀 Periodic speed boost sent")
                time.sleep(5.0)
                
        except KeyboardInterrupt:
            print("\n⏹️ Speed boost stopped by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the speed boost system."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        self.action_socket.close()
        self.data_socket.close()
        self.context.term()
        print("🏁 Speed boost system stopped")

def main():
    """Run the continuous speed boost."""
    booster = ContinuousSpeedBoost(boost_factor=1.5, min_throttle=0.5)
    booster.start()

if __name__ == "__main__":
    main()

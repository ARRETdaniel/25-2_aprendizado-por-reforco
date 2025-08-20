#!/usr/bin/env python3
"""
Quick Speed Boost Script - Apply to running training
"""

import zmq
import msgpack
import time
import random
import numpy as np

def send_aggressive_actions():
    """Send more aggressive actions to boost car speed."""

    print("🏎️ Speed Boost Script - Sending Aggressive Actions")
    print("=" * 50)

    # Setup ZMQ connection
    context = zmq.Context()
    action_socket = context.socket(zmq.PUB)
    action_socket.bind("tcp://*:5556")

    time.sleep(0.5)  # Allow connection

    print("🚀 Sending high-speed commands...")

    try:
        for i in range(100):  # Send 100 aggressive actions
            # Generate aggressive driving action
            steering = random.uniform(-0.3, 0.3)  # Moderate steering
            throttle = random.uniform(900.6, 1000.0)   # High throttle
            brake = 0.0                           # No braking

            action_data = {
                'steering': steering,
                'throttle': throttle,
                'brake': brake,
                'aggressive': True,
                'speed_boost': True
            }

            packed_action = msgpack.packb(action_data)
            action_socket.send(packed_action)

            print(f"🏎️ Action {i+1:3d}: Throttle={throttle:.2f}, Steering={steering:.2f}")
            time.sleep(0.1)  # 10 Hz

    except KeyboardInterrupt:
        print("\n⏹️ Speed boost interrupted")

    finally:
        action_socket.close()
        context.term()
        print("🏁 Speed boost completed!")

if __name__ == "__main__":
    send_aggressive_actions()

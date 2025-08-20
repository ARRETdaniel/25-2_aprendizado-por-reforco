#!/usr/bin/env python3
"""
Quick ZMQ Connection Test
Tests communication between CARLA client and DRL trainer
"""

import zmq
import time
import json

def test_zmq_connection():
    """Test ZMQ connection between client and trainer."""
    
    print("🔍 Testing ZMQ Communication...")
    
    # Test if CARLA client is sending data
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
    
    try:
        socket.connect("tcp://localhost:5555")
        print("📡 Connected to CARLA client port 5555")
        
        for i in range(3):
            try:
                message = socket.recv()
                print(f"✅ Received message {i+1}: {len(message)} bytes")
                # Try to decode
                try:
                    data = json.loads(message.decode())
                    print(f"   📊 Message type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"   🔑 Keys: {list(data.keys())}")
                except:
                    print("   📝 Binary/msgpack data")
                break
            except zmq.Again:
                print(f"⏳ Waiting for message {i+1}...")
                time.sleep(1)
        else:
            print("❌ No messages received from CARLA client")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    test_zmq_connection()

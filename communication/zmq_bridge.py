#!/usr/bin/env python3
"""
ZeroMQ Communication Bridge for CARLA DRL Integration
Enables communication between Python 3.6 (CARLA) and Python 3.12 (DRL)
"""

import time
import json
import threading
import queue
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import zmq
    import msgpack
    import msgpack_numpy as m
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install pyzmq msgpack msgpack-numpy")
    raise

# Configure msgpack for numpy arrays
m.patch()

class ZMQCarlabridge:
    """
    ZeroMQ-based communication bridge for CARLA DRL integration.
    Handles bi-directional communication between CARLA client and DRL agent.
    """
    
    def __init__(self, 
                 carla_port: int = 5555,
                 drl_port: int = 5556,
                 timeout_ms: int = 1000):
        """
        Initialize ZMQ communication bridge.
        
        Args:
            carla_port: Port for CARLA client communication
            drl_port: Port for DRL agent communication  
            timeout_ms: Communication timeout in milliseconds
        """
        self.carla_port = carla_port
        self.drl_port = drl_port
        self.timeout_ms = timeout_ms
        
        # ZMQ Context
        self.context = zmq.Context()
        
        # Sockets
        self.carla_socket = None
        self.drl_socket = None
        
        # Threading
        self.running = False
        self.bridge_thread = None
        
        # Data queues
        self.carla_to_drl_queue = queue.Queue(maxsize=10)
        self.drl_to_carla_queue = queue.Queue(maxsize=10)
        
        # Performance tracking
        self.messages_sent = 0
        self.messages_received = 0
        self.start_time = time.time()
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    def start_carla_server(self):
        """Start ZMQ server for CARLA client connection."""
        try:
            self.carla_socket = self.context.socket(zmq.REP)  # Reply socket
            self.carla_socket.bind(f"tcp://*:{self.carla_port}")
            self.carla_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.carla_socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            
            self.logger.info(f"🌉 CARLA bridge server started on port {self.carla_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start CARLA server: {e}")
            return False
    
    def start_drl_client(self):
        """Start ZMQ client for DRL agent connection."""
        try:
            self.drl_socket = self.context.socket(zmq.REQ)  # Request socket
            self.drl_socket.connect(f"tcp://localhost:{self.drl_port}")
            self.drl_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.drl_socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            
            self.logger.info(f"🤖 DRL bridge client connected to port {self.drl_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect DRL client: {e}")
            return False
    
    def pack_carla_data(self, measurements, sensor_data: Dict[str, Any]) -> bytes:
        """Pack CARLA measurements and sensor data for transmission."""
        try:
            # Extract key measurements
            player = measurements.player_measurements
            
            data = {
                'timestamp': time.time(),
                'vehicle': {
                    'position': {
                        'x': player.transform.location.x,
                        'y': player.transform.location.y,
                        'z': player.transform.location.z
                    },
                    'rotation': {
                        'pitch': player.transform.rotation.pitch,
                        'yaw': player.transform.rotation.yaw,
                        'roll': player.transform.rotation.roll
                    },
                    'velocity': {
                        'forward': player.forward_speed,
                        'x': getattr(player, 'velocity_x', 0.0),
                        'y': getattr(player, 'velocity_y', 0.0),
                        'z': getattr(player, 'velocity_z', 0.0)
                    },
                    'acceleration': {
                        'x': player.acceleration.x,
                        'y': player.acceleration.y,
                        'z': player.acceleration.z
                    }
                },
                'sensors': {}
            }
            
            # Pack sensor data
            for sensor_name, sensor_value in sensor_data.items():
                if 'Camera' in sensor_name:
                    # Convert camera data to numpy array
                    if hasattr(sensor_value, 'data'):
                        # Handle CARLA 0.8.4 image format
                        img_array = np.frombuffer(sensor_value.data, dtype=np.uint8)
                        img_array = img_array.reshape((sensor_value.height, sensor_value.width, 4))
                        img_array = img_array[:, :, :3]  # Remove alpha channel
                        data['sensors'][sensor_name] = img_array
                    else:
                        self.logger.warning(f"Unknown camera data format for {sensor_name}")
                
                elif 'Lidar' in sensor_name:
                    # Handle LIDAR data
                    if hasattr(sensor_value, 'data'):
                        lidar_data = np.frombuffer(sensor_value.data, dtype=np.float32)
                        data['sensors'][sensor_name] = lidar_data
            
            # Pack with msgpack
            packed_data = msgpack.packb(data)
            return packed_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to pack CARLA data: {e}")
            return msgpack.packb({'error': str(e)})
    
    def unpack_drl_action(self, packed_data: bytes) -> Optional[Dict[str, float]]:
        """Unpack DRL action commands."""
        try:
            data = msgpack.unpackb(packed_data, raw=False)
            
            if 'error' in data:
                self.logger.error(f"DRL error: {data['error']}")
                return None
            
            # Expected DRL action format
            action = {
                'steering': float(data.get('steering', 0.0)),
                'throttle': float(data.get('throttle', 0.0)),
                'brake': float(data.get('brake', 0.0))
            }
            
            # Clamp values to valid ranges
            action['steering'] = np.clip(action['steering'], -1.0, 1.0)
            action['throttle'] = np.clip(action['throttle'], 0.0, 1.0)
            action['brake'] = np.clip(action['brake'], 0.0, 1.0)
            
            return action
            
        except Exception as e:
            self.logger.error(f"❌ Failed to unpack DRL action: {e}")
            return None
    
    def send_to_drl(self, measurements, sensor_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Send CARLA data to DRL agent and receive action."""
        if not self.drl_socket:
            return None
        
        try:
            # Pack and send data
            packed_data = self.pack_carla_data(measurements, sensor_data)
            self.drl_socket.send(packed_data)
            self.messages_sent += 1
            
            # Receive response
            response = self.drl_socket.recv()
            self.messages_received += 1
            
            # Unpack action
            action = self.unpack_drl_action(response)
            return action
            
        except zmq.Again:
            self.logger.warning("⏱️ DRL communication timeout")
            return None
        except Exception as e:
            self.logger.error(f"❌ DRL communication error: {e}")
            return None
    
    def handle_carla_request(self) -> bool:
        """Handle incoming CARLA client requests."""
        if not self.carla_socket:
            return False
        
        try:
            # Receive CARLA data
            message = self.carla_socket.recv()
            data = msgpack.unpackb(message, raw=False)
            
            # Process request
            if data.get('type') == 'sensor_data':
                # Add to queue for DRL processing
                try:
                    self.carla_to_drl_queue.put_nowait(data)
                except queue.Full:
                    self.logger.warning("🚫 CARLA to DRL queue full, dropping data")
                
                # Check for DRL response
                try:
                    response = self.drl_to_carla_queue.get_nowait()
                    self.carla_socket.send(msgpack.packb(response))
                except queue.Empty:
                    # Send default action
                    default_action = {'steering': 0.0, 'throttle': 0.1, 'brake': 0.0}
                    self.carla_socket.send(msgpack.packb(default_action))
            
            return True
            
        except zmq.Again:
            return True  # Timeout is expected
        except Exception as e:
            self.logger.error(f"❌ CARLA request handling error: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get communication performance statistics."""
        elapsed = time.time() - self.start_time
        return {
            'uptime_seconds': elapsed,
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'message_rate_hz': self.messages_sent / elapsed if elapsed > 0 else 0,
            'carla_queue_size': self.carla_to_drl_queue.qsize(),
            'drl_queue_size': self.drl_to_carla_queue.qsize()
        }
    
    def cleanup(self):
        """Clean up ZMQ resources."""
        self.running = False
        
        if self.carla_socket:
            self.carla_socket.close()
        if self.drl_socket:
            self.drl_socket.close()
        
        self.context.term()
        self.logger.info("🧹 ZMQ bridge cleaned up")


class CarlaBridgeClient:
    """
    Client interface for CARLA to communicate with DRL agent.
    To be used in Python 3.6 CARLA client.
    """
    
    def __init__(self, bridge_port: int = 5555):
        self.bridge_port = bridge_port
        self.context = zmq.Context()
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to the bridge server."""
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://localhost:{self.bridge_port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            self.socket.setsockopt(zmq.SNDTIMEO, 1000)
            
            self.connected = True
            print(f"✅ Connected to bridge server on port {self.bridge_port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to bridge: {e}")
            self.connected = False
            return False
    
    def send_sensor_data(self, measurements, sensor_data):
        """Send sensor data to DRL agent and receive control action."""
        if not self.connected or not self.socket:
            return None
        
        try:
            # Prepare data
            data = {
                'type': 'sensor_data',
                'timestamp': time.time(),
                'measurements': self._extract_measurements(measurements),
                'sensors': self._extract_sensor_data(sensor_data)
            }
            
            # Send and receive
            packed_data = msgpack.packb(data)
            self.socket.send(packed_data)
            
            response = self.socket.recv()
            action = msgpack.unpackb(response, raw=False)
            
            return action
            
        except zmq.Again:
            print("⏱️ Bridge communication timeout")
            return None
        except Exception as e:
            print(f"❌ Bridge communication error: {e}")
            return None
    
    def _extract_measurements(self, measurements):
        """Extract measurements from CARLA format."""
        player = measurements.player_measurements
        return {
            'position': [player.transform.location.x, player.transform.location.y, player.transform.location.z],
            'rotation': [player.transform.rotation.pitch, player.transform.rotation.yaw, player.transform.rotation.roll],
            'velocity': player.forward_speed,
            'acceleration': [player.acceleration.x, player.acceleration.y, player.acceleration.z]
        }
    
    def _extract_sensor_data(self, sensor_data):
        """Extract sensor data from CARLA format."""
        extracted = {}
        
        for sensor_name, sensor_value in sensor_data.items():
            if 'Camera' in sensor_name and hasattr(sensor_value, 'data'):
                # Convert to small image for fast transmission
                img_array = np.frombuffer(sensor_value.data, dtype=np.uint8)
                img_array = img_array.reshape((sensor_value.height, sensor_value.width, 4))
                img_array = img_array[:, :, :3]  # Remove alpha
                
                # Resize for efficiency (64x64 for DRL)
                import cv2
                small_img = cv2.resize(img_array, (64, 64))
                extracted[sensor_name] = small_img.tolist()  # Convert to list for JSON
        
        return extracted
    
    def disconnect(self):
        """Disconnect from bridge."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self.connected = False
        print("🔌 Disconnected from bridge")


class DRLBridgeServer:
    """
    Server interface for DRL agent to receive CARLA data and send actions.
    To be used in Python 3.12 DRL environment.
    """
    
    def __init__(self, port: int = 5556):
        self.port = port
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        
    def start(self):
        """Start the DRL bridge server."""
        try:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            
            self.running = True
            print(f"🤖 DRL bridge server started on port {self.port}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start DRL server: {e}")
            return False
    
    def receive_carla_data(self):
        """Receive CARLA sensor data."""
        if not self.running or not self.socket:
            return None
        
        try:
            message = self.socket.recv()
            data = msgpack.unpackb(message, raw=False)
            return data
            
        except zmq.Again:
            return None  # Timeout
        except Exception as e:
            print(f"❌ Error receiving CARLA data: {e}")
            return None
    
    def send_action(self, steering: float, throttle: float, brake: float = 0.0):
        """Send control action to CARLA."""
        if not self.running or not self.socket:
            return False
        
        try:
            action = {
                'steering': float(steering),
                'throttle': float(throttle), 
                'brake': float(brake)
            }
            
            packed_action = msgpack.packb(action)
            self.socket.send(packed_action)
            return True
            
        except Exception as e:
            print(f"❌ Error sending action: {e}")
            return False
    
    def stop(self):
        """Stop the DRL bridge server."""
        self.running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print("🛑 DRL bridge server stopped")


if __name__ == "__main__":
    # Test the bridge
    bridge = ZMQCarlabridge()
    
    print("🧪 Testing ZMQ Bridge...")
    if bridge.start_carla_server():
        print("✅ Bridge server started successfully")
        
        try:
            time.sleep(1)
            stats = bridge.get_performance_stats()
            print(f"📊 Bridge stats: {stats}")
            
        except KeyboardInterrupt:
            print("⚠️ Interrupted by user")
        finally:
            bridge.cleanup()
    else:
        print("❌ Failed to start bridge server")

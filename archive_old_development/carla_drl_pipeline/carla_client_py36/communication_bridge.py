"""
Communication Bridge for CARLA-ROS2 Pipeline

This module provides high-performance IPC communication between the CARLA client
(Python 3.6) and the ROS 2 gateway. It uses ZeroMQ for reliable, low-latency
message passing with fallback to file-based communication.
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List
from queue import Queue, Empty

logger = logging.getLogger(__name__)

# Try to import ZeroMQ for high-performance IPC
try:
    import zmq
    HAS_ZMQ = True
    logger.info("ZeroMQ available for IPC communication")
except ImportError:
    HAS_ZMQ = False
    logger.warning("ZeroMQ not available, using file-based communication")

# Try to import msgpack for efficient serialization
try:
    import msgpack
    HAS_MSGPACK = True
    logger.info("MessagePack available for serialization")
except ImportError:
    HAS_MSGPACK = False
    logger.warning("MessagePack not available, using JSON serialization")


class CommunicationBridge:
    """High-performance communication bridge for CARLA-ROS2 pipeline."""
    
    def __init__(self, 
                 address: str = "tcp://localhost:5555",
                 use_zmq: bool = True,
                 use_msgpack: bool = True,
                 timeout: float = 1.0):
        """Initialize communication bridge.
        
        Args:
            address: ZeroMQ address for communication
            use_zmq: Whether to use ZeroMQ (fallback to files if False)
            use_msgpack: Whether to use MessagePack for serialization
            timeout: Timeout for communication operations
        """
        self.address = address
        self.use_zmq = use_zmq and HAS_ZMQ
        self.use_msgpack = use_msgpack and HAS_MSGPACK
        self.timeout = timeout
        
        # ZeroMQ components
        self.context = None
        self.publisher = None
        self.subscriber = None
        
        # File-based communication
        self.sensor_data_file = "sensor_data.json"
        self.control_data_file = "control_data.json"
        self.reset_file = "reset_signal.txt"
        
        # Threading
        self.running = False
        self.receiver_thread = None
        self.control_queue = Queue(maxsize=10)
        self.reset_requested = False
        
        logger.info(f"Communication bridge initialized - "
                   f"ZMQ: {self.use_zmq}, MessagePack: {self.use_msgpack}")
    
    def connect(self) -> bool:
        """Connect communication bridge.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.use_zmq:
                return self._connect_zmq()
            else:
                return self._connect_file_based()
        except Exception as e:
            logger.error(f"Failed to connect communication bridge: {e}")
            return False
    
    def _connect_zmq(self) -> bool:
        """Connect ZeroMQ communication.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.context = zmq.Context()
            
            # Publisher for sensor data (CARLA -> ROS2)
            self.publisher = self.context.socket(zmq.PUB)
            pub_address = self.address
            self.publisher.bind(pub_address)
            
            # Subscriber for control commands (ROS2 -> CARLA)
            self.subscriber = self.context.socket(zmq.SUB)
            sub_address = self.address.replace("5555", "5556")
            self.subscriber.connect(sub_address)
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b"control")
            self.subscriber.setsockopt(zmq.SUBSCRIBE, b"reset")
            
            # Set timeouts
            self.publisher.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
            self.subscriber.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            
            # Start receiver thread
            self.running = True
            self.receiver_thread = threading.Thread(target=self._zmq_receiver_loop)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()
            
            logger.info(f"ZeroMQ communication connected - Pub: {pub_address}, Sub: {sub_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect ZeroMQ communication: {e}")
            return False
    
    def _connect_file_based(self) -> bool:
        """Connect file-based communication.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create communication files
            for filename in [self.sensor_data_file, self.control_data_file]:
                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        json.dump({}, f)
            
            # Start file monitor thread
            self.running = True
            self.receiver_thread = threading.Thread(target=self._file_receiver_loop)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()
            
            logger.info("File-based communication connected")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect file-based communication: {e}")
            return False
    
    def publish_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Publish sensor data to ROS2 gateway.
        
        Args:
            data: Sensor data dictionary
            
        Returns:
            True if publish successful, False otherwise
        """
        try:
            if self.use_zmq:
                return self._publish_zmq(data)
            else:
                return self._publish_file(data)
        except Exception as e:
            logger.error(f"Failed to publish sensor data: {e}")
            return False
    
    def _publish_zmq(self, data: Dict[str, Any]) -> bool:
        """Publish data via ZeroMQ.
        
        Args:
            data: Data to publish
            
        Returns:
            True if publish successful, False otherwise
        """
        try:
            # Serialize data
            if self.use_msgpack:
                # Convert numpy arrays to lists for msgpack
                serialized_data = self._prepare_data_for_msgpack(data)
                message_data = msgpack.packb(serialized_data)
            else:
                # Convert numpy arrays to lists for JSON
                serialized_data = self._prepare_data_for_json(data)
                message_data = json.dumps(serialized_data).encode('utf-8')
            
            # Send message
            self.publisher.send_multipart([b"sensor_data", message_data], zmq.NOBLOCK)
            return True
            
        except zmq.Again:
            logger.warning("ZeroMQ send timeout")
            return False
        except Exception as e:
            logger.error(f"ZeroMQ publish error: {e}")
            return False
    
    def _publish_file(self, data: Dict[str, Any]) -> bool:
        """Publish data via file.
        
        Args:
            data: Data to publish
            
        Returns:
            True if publish successful, False otherwise
        """
        try:
            # Convert numpy arrays to lists for JSON
            serialized_data = self._prepare_data_for_json(data)
            
            # Write to file atomically
            temp_file = self.sensor_data_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(serialized_data, f, indent=2)
            
            os.rename(temp_file, self.sensor_data_file)
            return True
            
        except Exception as e:
            logger.error(f"File publish error: {e}")
            return False
    
    def get_control_command(self) -> Optional[Dict[str, Any]]:
        """Get vehicle control command from ROS2 gateway.
        
        Returns:
            Control command dictionary or None
        """
        try:
            return self.control_queue.get_nowait()
        except Empty:
            return None
    
    def should_reset(self) -> bool:
        """Check if episode reset is requested.
        
        Returns:
            True if reset requested, False otherwise
        """
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False
    
    def _zmq_receiver_loop(self):
        """ZeroMQ receiver loop running in separate thread."""
        logger.info("ZeroMQ receiver loop started")
        
        while self.running:
            try:
                # Receive message with timeout
                message = self.subscriber.recv_multipart(zmq.NOBLOCK)
                
                if len(message) >= 2:
                    topic = message[0].decode('utf-8')
                    data_bytes = message[1]
                    
                    # Deserialize data
                    if self.use_msgpack:
                        data = msgpack.unpackb(data_bytes, raw=False)
                    else:
                        data = json.loads(data_bytes.decode('utf-8'))
                    
                    # Handle different message types
                    if topic == "control":
                        self._handle_control_message(data)
                    elif topic == "reset":
                        self._handle_reset_message(data)
                
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"ZeroMQ receiver error: {e}")
                time.sleep(0.1)
        
        logger.info("ZeroMQ receiver loop stopped")
    
    def _file_receiver_loop(self):
        """File-based receiver loop running in separate thread."""
        logger.info("File receiver loop started")
        
        last_control_mtime = 0
        last_reset_mtime = 0
        
        while self.running:
            try:
                # Check control file
                if os.path.exists(self.control_data_file):
                    mtime = os.path.getmtime(self.control_data_file)
                    if mtime > last_control_mtime:
                        last_control_mtime = mtime
                        
                        with open(self.control_data_file, 'r') as f:
                            data = json.load(f)
                        
                        if data:
                            self._handle_control_message(data)
                
                # Check reset file
                if os.path.exists(self.reset_file):
                    mtime = os.path.getmtime(self.reset_file)
                    if mtime > last_reset_mtime:
                        last_reset_mtime = mtime
                        self._handle_reset_message({})
                
                time.sleep(0.01)  # 100 Hz polling
                
            except Exception as e:
                logger.error(f"File receiver error: {e}")
                time.sleep(0.1)
        
        logger.info("File receiver loop stopped")
    
    def _handle_control_message(self, data: Dict[str, Any]):
        """Handle control message from ROS2 gateway.
        
        Args:
            data: Control message data
        """
        try:
            # Put control command in queue (drop old commands if queue is full)
            if self.control_queue.full():
                try:
                    self.control_queue.get_nowait()
                except Empty:
                    pass
            
            self.control_queue.put_nowait(data)
            
        except Exception as e:
            logger.error(f"Error handling control message: {e}")
    
    def _handle_reset_message(self, data: Dict[str, Any]):
        """Handle reset message from ROS2 gateway.
        
        Args:
            data: Reset message data
        """
        self.reset_requested = True
        logger.info("Episode reset requested")
    
    def _prepare_data_for_msgpack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for MessagePack serialization.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Data prepared for MessagePack
        """
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return {
                    '__numpy__': True,
                    'data': obj.tolist(),
                    'dtype': str(obj.dtype),
                    'shape': obj.shape
                }
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return convert_numpy(data)
    
    def _prepare_data_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON serialization.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Data prepared for JSON
        """
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        return convert_numpy(data)
    
    def disconnect(self):
        """Disconnect communication bridge."""
        logger.info("Disconnecting communication bridge")
        
        # Stop receiver thread
        self.running = False
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2.0)
        
        # Cleanup ZeroMQ
        if self.use_zmq:
            if self.publisher:
                self.publisher.close()
            if self.subscriber:
                self.subscriber.close()
            if self.context:
                self.context.term()
        
        logger.info("Communication bridge disconnected")


def test_communication_bridge():
    """Test communication bridge functionality."""
    import time
    import numpy as np
    
    logger.info("Testing communication bridge")
    
    # Create bridge
    bridge = CommunicationBridge()
    
    if not bridge.connect():
        logger.error("Failed to connect bridge")
        return False
    
    try:
        # Test sensor data publishing
        test_data = {
            'timestamp': time.time(),
            'episode': 1,
            'step': 10,
            'camera_rgb': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            'position': {'x': 1.0, 'y': 2.0, 'z': 3.0},
            'velocity': {'forward': 15.5}
        }
        
        success = bridge.publish_sensor_data(test_data)
        logger.info(f"Sensor data publish: {'Success' if success else 'Failed'}")
        
        # Test control command retrieval
        control = bridge.get_control_command()
        logger.info(f"Control command: {control}")
        
        # Test reset signal
        reset_requested = bridge.should_reset()
        logger.info(f"Reset requested: {reset_requested}")
        
        return True
        
    finally:
        bridge.disconnect()


if __name__ == '__main__':
    # Run test
    test_communication_bridge()

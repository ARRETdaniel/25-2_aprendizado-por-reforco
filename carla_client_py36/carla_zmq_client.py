#!/usr/bin/env python3
"""
Enhanced CARLA Client with ZMQ Bridge Integration
Connects to CARLA and communicates with DRL agent via ZMQ
Compatible with Python 3.6 and CARLA 0.8.4
"""

import sys
import os
import time
import numpy as np
import cv2

# Add CARLA paths - use relative paths from workspace
workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
carla_client_dir = os.path.join(workspace_dir, 'CarlaSimulator', 'PythonClient')
sys.path.append(carla_client_dir)

try:
    from carla.client import make_carla_client, VehicleControl
    from carla.settings import CarlaSettings
    from carla.sensor import Camera
    from carla.image_converter import to_rgb_array
    print("✅ CARLA modules imported successfully")
except ImportError as e:
    print(f"❌ CARLA import error: {e}")
    print("Make sure CARLA 0.8.4 is installed and paths are correct")
    raise

# Communication bridge (install with: pip install pyzmq msgpack msgpack-numpy)
try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()  # Enable numpy serialization
    print("✅ ZMQ communication modules imported")
except ImportError as e:
    print(f"❌ ZMQ import error: {e}")
    print("Install with: pip install pyzmq msgpack msgpack-numpy")
    raise

class CarlaZMQClient:
    """
    Enhanced CARLA Client with ZMQ bridge for DRL integration.
    Runs in Python 3.6 environment and communicates with DRL agent.
    """

    def __init__(self,
                 carla_host='localhost',
                 carla_port=2000,
                 zmq_port=5555,
                 timeout=10.0):
        """
        Initialize CARLA ZMQ Client.

        Args:
            carla_host: CARLA server host
            carla_port: CARLA server port
            zmq_port: ZMQ communication port
            timeout: Connection timeout
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.zmq_port = zmq_port
        self.timeout = timeout

        # CARLA connection
        self.client = None
        self.episode_active = False

        # ZMQ communication
        self.zmq_context = zmq.Context()
        self.zmq_socket = None
        self.zmq_connected = False

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_action = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}

        print(f"🚗 CARLA ZMQ Client initialized")
        print(f"📡 CARLA: {carla_host}:{carla_port}")
        print(f"🌉 ZMQ: port {zmq_port}")

    def setup_zmq_connection(self):
        """Setup ZMQ connection for DRL communication."""
        try:
            # Use PUB socket to publish CARLA data (DRL agent subscribes to this)
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(f"tcp://*:{self.zmq_port}")

            # Action receiver socket (for receiving actions from DRL)
            self.action_socket = self.zmq_context.socket(zmq.SUB)
            self.action_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.action_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            self.action_socket.connect(f"tcp://localhost:{self.zmq_port + 1}")

            # Small delay for socket binding
            time.sleep(0.1)

            self.zmq_connected = True
            print("✅ ZMQ bridge connection established")
            return True

        except Exception as e:
            print(f"⚠️ ZMQ setup failed: {e}")
            print("⚠️ Running in standalone mode")
            self.zmq_connected = False
            return False

    def create_carla_settings(self):
        """Create CARLA settings with sensors."""
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=False,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=10,
            WeatherId=1,
            QualityLevel='Low'  # Optimize for performance
        )

        # Add RGB camera
        camera = Camera('CameraRGB')
        camera.set_image_size(640, 480)
        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(camera)

        # Add depth camera
        depth_camera = Camera('CameraDepth', PostProcessing='Depth')
        depth_camera.set_image_size(640, 480)
        depth_camera.set_position(2.0, 0.0, 1.4)
        depth_camera.set_rotation(0.0, 0.0, 0.0)
        settings.add_sensor(depth_camera)

        return settings

    def connect_to_carla(self):
        """Connect to CARLA server."""
        try:
            print(f"🔗 Connecting to CARLA at {self.carla_host}:{self.carla_port}")
            # CARLA client is a context manager - we'll handle it in run_bridge()
            return True
        except Exception as e:
            print(f"❌ CARLA connection failed: {e}")
            return False

    def start_episode(self):
        """Start a new CARLA episode."""
        try:
            settings = self.create_carla_settings()
            scene = self.client.load_settings(settings)

            # Start episode with random position
            start_idx = 0  # You can randomize this
            self.client.start_episode(start_idx)

            self.episode_active = True
            print(f"✅ Episode started at position {start_idx}")
            return True

        except Exception as e:
            print(f"❌ Episode start failed: {e}")
            return False

    def pack_sensor_data(self, measurements, sensor_data):
        """Pack sensor data for ZMQ transmission."""
        try:
            # Extract vehicle measurements
            player = measurements.player_measurements

            data = {
                'type': 'sensor_data',
                'timestamp': time.time(),
                'measurements': {
                    'position': [player.transform.location.x,
                                player.transform.location.y,
                                player.transform.location.z],
                    'rotation': [player.transform.rotation.pitch,
                                player.transform.rotation.yaw,
                                player.transform.rotation.roll],
                    'velocity': player.forward_speed,
                    'acceleration': [player.acceleration.x,
                                   player.acceleration.y,
                                   player.acceleration.z],
                    # CRITICAL FIX: Add collision detection
                    'collision_vehicles': player.collision_vehicles,
                    'collision_pedestrians': player.collision_pedestrians,
                    'collision_other': player.collision_other,
                    'intersection_otherlane': player.intersection_otherlane,
                    'intersection_offroad': player.intersection_offroad
                },
                'sensors': {}
            }

            # Process camera data
            for sensor_name, sensor_value in sensor_data.items():
                if 'Camera' in sensor_name:
                    # Convert to numpy array
                    img_array = to_rgb_array(sensor_value)

                    # Resize to 64x64 for efficient transmission
                    small_img = cv2.resize(img_array, (64, 64))

                    # Convert to list for JSON serialization
                    data['sensors'][sensor_name] = small_img.tolist()

            return msgpack.packb(data)

        except Exception as e:
            print(f"❌ Error packing sensor data: {e}")
            return None

    def send_to_drl_agent(self, measurements, sensor_data):
        """Send sensor data to DRL agent via PUB socket."""
        if not self.zmq_connected:
            return None

        try:
            # Pack and send data via PUB socket
            packed_data = self.pack_sensor_data(measurements, sensor_data)
            if packed_data is None:
                return None

            self.zmq_socket.send(packed_data)

            # Try to receive action via SUB socket (non-blocking)
            try:
                action_data = self.action_socket.recv(zmq.NOBLOCK)
                action = msgpack.unpackb(action_data, raw=False)
                return action
            except zmq.Again:
                # No action available, return default/neutral action
                return {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}

        except Exception as e:
            print(f"⚠️ Communication error: {e}")
            return None

            # Receive action
            response = self.zmq_socket.recv()
            action = msgpack.unpackb(response, raw=False)

            return action

        except zmq.Again:
            print("⏱️ DRL agent timeout")
            return None
        except Exception as e:
            print(f"❌ DRL communication error: {e}")
            return None

    def apply_action(self, action):
        """Apply action to CARLA vehicle."""
        try:
            if action is None:
                # Use last action or default
                action = self.last_action

            # Create vehicle control
            control = VehicleControl()
            control.steer = float(action.get('steering', 0.0))
            control.throttle = float(action.get('throttle', 0.1))
            control.brake = float(action.get('brake', 0.0))
            control.hand_brake = False
            control.reverse = False

            # Enhanced aggressive mode handling
            if action.get('aggressive', False) or action.get('speed_boost', False):
                # Boost throttle for aggressive driving
                if control.throttle > 0:
                    control.throttle = max(control.throttle, 0.5)  # Minimum aggressive throttle
                    control.throttle = min(control.throttle * 1.2, 1.0)  # 20% boost

            # Clamp values
            control.steer = max(-1.0, min(1.0, control.steer))
            control.throttle = max(0.0, min(1.0, control.throttle))
            control.brake = max(0.0, min(1.0, control.brake))

            # Send to CARLA
            self.client.send_control(control)
            self.last_action = action

            return True

        except Exception as e:
            print(f"❌ Error applying action: {e}")
            return False

    def display_sensor_data(self, measurements, sensor_data):
        """Display sensor data with performance info."""
        try:
            self.frame_count += 1

            # Calculate FPS
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            # Process RGB camera
            if 'CameraRGB' in sensor_data:
                rgb_image = sensor_data['CameraRGB']
                rgb_array = to_rgb_array(rgb_image)
                bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

                # Add overlay information
                player = measurements.player_measurements
                speed = player.forward_speed * 3.6  # Convert to km/h

                # Performance info
                cv2.putText(bgr_image, f"🚗 CARLA ZMQ Bridge", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(bgr_image, f"📊 FPS: {fps:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(bgr_image, f"🎮 Frame: {self.frame_count}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Vehicle info
                cv2.putText(bgr_image, f"🚗 Speed: {speed:.1f} km/h", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Connection status
                status_color = (0, 255, 0) if self.zmq_connected else (0, 0, 255)
                status_text = "🤖 DRL Connected" if self.zmq_connected else "⚠️ DRL Disconnected"
                cv2.putText(bgr_image, status_text, (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Last action
                action_text = f"🎮 S:{self.last_action.get('steering', 0):.2f} T:{self.last_action.get('throttle', 0):.2f}"
                cv2.putText(bgr_image, action_text, (10, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Controls
                cv2.putText(bgr_image, "Controls: Q=Quit, R=Reset", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow('CARLA ZMQ Bridge', bgr_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False  # Quit
            elif key == ord('r'):
                print("🔄 Reset requested")
                self.episode_active = False
                return True

            return True

        except Exception as e:
            print(f"❌ Display error: {e}")
            return True

    def run(self):
        """Main run loop."""
        print("🚀 Starting CARLA ZMQ Bridge...")

        # Setup ZMQ connection
        self.setup_zmq_connection()

        try:
            # Use CARLA client as context manager
            with make_carla_client(self.carla_host, self.carla_port) as client:
                print("✅ Connected to CARLA server")
                self.client = client

                # Start episode
                if not self.start_episode():
                    print("❌ Failed to start episode")
                    return False

                print("🔄 Bridge running - Press Ctrl+C to stop")
                frame_count = 0

                while True:
                    try:
                        # Read sensor data
                        measurements, sensor_data = self.client.read_data()
                        frame_count += 1

                        # Debug info every 100 frames
                        if frame_count % 100 == 0:
                            print(f"📊 Processed {frame_count} frames")

                        # Send to DRL agent and get action
                        action = self.send_to_drl_agent(measurements, sensor_data)

                        # Debug ZMQ sending every 500 frames
                        if frame_count % 500 == 0:
                            if action:
                                print(f"📤 ZMQ: Sent data and received action at frame {frame_count}")
                            else:
                                print(f"📤 ZMQ: Sent data, no action received at frame {frame_count}")

                        # Apply action to vehicle
                        self.apply_action(action)

                        # Display visualization
                        if not self.display_sensor_data(measurements, sensor_data):
                            print("🛑 Display requested quit")
                            break  # User requested quit

                        # Small delay to prevent overwhelming the system
                        time.sleep(0.01)

                    except KeyboardInterrupt:
                        print("⚠️ Interrupted by user")
                        break
                    except Exception as e:
                        print(f"❌ Frame processing error: {e}")
                        # Continue to next frame
                        time.sleep(0.1)

        except Exception as e:
            print(f"❌ CARLA connection error: {e}")
            return False
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")

        try:
            if self.episode_active and self.client:
                self.client.disconnect()
        except:
            pass

        if self.zmq_socket:
            self.zmq_socket.close()

        if hasattr(self, 'action_socket') and self.action_socket:
            self.action_socket.close()

        self.zmq_context.term()
        cv2.destroyAllWindows()

        print("✅ Cleanup completed")


def main():
    """Main function."""
    print("🚗 CARLA ZMQ Bridge Client")
    print("=" * 50)

    # Create and run client
    client = CarlaZMQClient(
        carla_host='localhost',
        carla_port=2000,
        zmq_port=5555
    )

    client.run()


if __name__ == "__main__":
    main()

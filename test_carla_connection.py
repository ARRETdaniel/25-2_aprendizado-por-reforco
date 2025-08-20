"""
Quick test to verify CARLA client connection and camera visualization.
This uses Python 3.6 compatible code.
"""
import sys
import os
import time

# Add CARLA Python API to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'CarlaSimulator', 'PythonClient')))

try:
    from carla.client import make_carla_client, VehicleControl
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.image_converter import to_rgb_array
    print("✅ CARLA imports successful")
except ImportError as e:
    print(f"❌ CARLA import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print("✅ OpenCV import successful")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    sys.exit(1)

def test_carla_connection():
    """Test basic CARLA connection and camera setup."""
    print("\n🔍 Testing CARLA connection...")

    try:
        with make_carla_client('localhost', 2000, timeout=10) as client:
            print("✅ Connected to CARLA server")

            # Create basic settings
            settings = CarlaSettings()
            settings.set(
                SynchronousMode=True,
                SendNonPlayerAgentsInfo=False,
                NumberOfVehicles=0,
                NumberOfPedestrians=0,
                WeatherId=1,
                QualityLevel='Low'
            )

            # Add camera
            camera = Camera('RGB_Camera')
            camera.set_image_size(640, 480)
            camera.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera)

            print("✅ Camera sensor configured")

            # Load settings and start episode
            scene = client.load_settings(settings)
            client.start_episode(0)

            print("✅ Episode started, testing camera feed...")

            # Test a few frames
            for frame in range(5):
                print(f"Frame {frame + 1}/5...")

                measurements, sensor_data = client.read_data()

                if 'RGB_Camera' in sensor_data:
                    camera_data = sensor_data['RGB_Camera']
                    image_array = to_rgb_array(camera_data)

                    # Convert RGB to BGR for OpenCV
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                    # Display the image
                    cv2.imshow('CARLA Camera Test', image_bgr)
                    cv2.waitKey(1)

                    print(f"  ✅ Frame {frame + 1}: Camera image {image_array.shape} displayed")
                else:
                    print(f"  ❌ Frame {frame + 1}: No camera data received")

                # Send simple control
                client.send_control(
                    steer=0.0,
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                    reverse=False
                )

                time.sleep(0.1)

            cv2.destroyAllWindows()
            print("✅ Camera test completed successfully!")

    except Exception as e:
        print(f"❌ CARLA connection test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🚗 CARLA DRL System - Quick Connection Test")
    print("=" * 50)

    success = test_carla_connection()

    if success:
        print("\n🎉 All tests passed! System is ready for DRL training.")
    else:
        print("\n💥 Tests failed. Please check CARLA server and dependencies.")

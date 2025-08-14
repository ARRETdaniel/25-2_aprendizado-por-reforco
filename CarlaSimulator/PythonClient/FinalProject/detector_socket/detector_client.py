# Author: Daniel Terra Gomes
# Date: Jun 30, 2025
# detector_client.py (to be imported in module_7.py)

import socket
import time
import numpy as np
import msgpack
import msgpack_numpy as m
import cv2
import colorsys
from collections import deque
import threading
import subprocess
import os
import sys
m.patch()  # Enable numpy array serialization

# Global warning timers with persistence
warning_timers = {
    'person': 0,
    'car': 0,
    'truck': 0,
    'bus': 0,
    'stop sign': 0,
    'traffic light': 0,
    'bicycle': 0,
    'motorcycle': 0
}

# Persistence duration for each warning type
WARNING_PERSISTENCE = {
    'person': 15,       # Critical - keep warnings for 15 frames
    'stop sign': 15,    # Critical - keep warnings for 15 frames
    'traffic light': 10,# Important - keep warnings for 10 frames
    'default': 8        # Standard - keep warnings for 8 frames
}

# Pre-computed warning overlays
WARNING_OVERLAYS = {}

# Last audio warning timestamps
last_audio_warning = {}

class DetectionClient:
    def __init__(self, host="localhost", port=5555, reconnect_attempts=3):
        self.host = host
        self.port = port
        self.socket = None
        self.reconnect_attempts = reconnect_attempts
        self.connected = False
        self.labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Initialize FPS history tracker
        self.fps_history = deque(maxlen=30)  # Store FPS history for smoothing

        # Initialize warning overlays system
        self._initialize_warning_overlays()

        # Audio thread
        self.audio_thread = None

        # Define driving-relevant classes for client-side filtering and visualization
        self.driving_relevant_classes = {
            0: 'person',           # pedestrians
            1: 'bicycle',          # cyclists
            2: 'car',              # cars
            3: 'motorcycle',       # motorcycles
            5: 'bus',              # buses
            7: 'truck',            # trucks
            9: 'traffic light',    # traffic lights
            11: 'stop sign',       # stop signs
#            13: 'bench',           # roadside objects
            16: 'dog',             # animals on road
            17: 'horse',           # animals on road
            18: 'sheep',           # animals on road
            19: 'cow',             # animals on road
            24: 'backpack',        # pedestrian with backpack
            25: 'umbrella',        # pedestrian with umbrella
            28: 'suitcase',        # roadside objects
        }

        # Custom colors for each class (make more visible in driving scenarios)
        self.class_colors = {
            0: (0, 0, 255),      # person: red
            1: (0, 128, 255),    # bicycle: orange
            2: (0, 255, 255),    # car: yellow
            3: (0, 255, 128),    # motorcycle: yellow-green
            5: (255, 0, 0),      # bus: blue
            7: (255, 0, 128),    # truck: purple
            9: (255, 255, 0),    # traffic light: cyan
            11: (0, 255, 0),     # stop sign: green
            13: (128, 128, 128), # bench: gray
            16: (128, 0, 0),     # dog: dark blue
            17: (128, 0, 128),   # horse: dark purple
            18: (0, 0, 128),     # sheep: dark red
            19: (0, 128, 0),     # cow: dark green
            24: (128, 128, 0),   # backpack: dark cyan
            25: (192, 192, 192), # umbrella: light gray
            28: (255, 128, 128), # suitcase: light red
        }

        # Connect to server
        self.connect()

    def _initialize_warning_overlays(self):
        """Pre-compute warning overlays for different numbers of warnings"""
        global WARNING_OVERLAYS

        # Create warning overlays for 1-3 warnings
        for num_warnings in range(1, 4):
            height = min(num_warnings, 3) * 45 + 15

            # Create multiple sizes to handle different image resolutions
            for width in [800, 640, 320]:
                # Create base overlay (black background with transparency)
                overlay = np.zeros((height, width, 3), dtype=np.uint8)
                WARNING_OVERLAYS[(num_warnings, width)] = overlay

    def connect(self):
        """Connect to the detection server with retries"""
        attempts = 0
        max_attempts = self.reconnect_attempts
        retry_delay = 2  # seconds between retries

        while attempts < max_attempts:
            try:
                print(f"Attempting to connect to detection server ({attempts+1}/{max_attempts})...")
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(5)  # 5 second timeout for connection
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(None)  # Reset to blocking mode after connection
                self.connected = True
                print(f"Connected to detection server at {self.host}:{self.port}")
                return True
            except socket.error as e:
                attempts += 1
                print(f"Connection attempt {attempts} failed: {e}")
                if attempts < max_attempts:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Failed to connect to detection server after multiple attempts")

        return False

    def detect_objects(self, image_array):
        """Send image to server and get detection results with compression"""
        if not self.connected:
            self.connect()
            if not self.connected:
                return image_array, [], [], [], []

        try:
            # Compress the image before sending
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, compressed_image = cv2.imencode('.jpg', image_array, encode_param)
            compressed_data = compressed_image.tobytes()

            # Pack image data with compression flag - fix the structure
            data = {
                'image_compressed': compressed_data,
                'shape': image_array.shape,
                'vehicle_state': {
                    'speed': 0,
                    'near_intersection': False
                }
            }
            packed_data = msgpack.packb(data, default=m.encode)

            # Send data size followed by data
            self.socket.sendall(len(packed_data).to_bytes(4, byteorder='little'))
            self.socket.sendall(packed_data)

            # Receive result size
            result_size_bytes = self.socket.recv(4)
            if not result_size_bytes:
                raise ConnectionError("No data received from server")

            result_size = int.from_bytes(result_size_bytes, byteorder='little')

            # Receive results
            received_data = b''
            while len(received_data) < result_size:
                chunk = self.socket.recv(min(4096, result_size - len(received_data)))
                if not chunk:
                    break
                received_data += chunk

            # Unpack results
            results = msgpack.unpackb(received_data, raw=False, object_hook=m.decode)

            boxes = results['boxes']
            confidences = results['confidences']
            class_ids = results['class_ids']
            fps = results.get('fps', 0)
            processing_time = results.get('processing_time', 0)

            # Add to FPS history for smoothing
            self.fps_history.append(fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            # Print detailed performance metrics
            self._print_performance_metrics(results)

            # Draw bounding boxes on a copy of the image
            processed_img = np.copy(image_array)

            # Track detected classes for warnings
            detected_classes = set()

            # Draw boxes and identify warnings
            for i, box in enumerate(boxes):
                x, y, w, h = box
                conf = confidences[i]
                cls_id = class_ids[i]

                # Skip any non-driving relevant objects that might have slipped through
                if cls_id not in self.driving_relevant_classes:
                    continue

                label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"

                # Use custom color for this class
                color = self.class_colors.get(cls_id, [255, 255, 255])

                # Track detected classes for warnings
                if label in ['person', 'car', 'truck', 'bus', 'stop sign',
                            'traffic light', 'bicycle', 'motorcycle']:
                    detected_classes.add(label)

                # Draw bounding box
                processed_img = cv2.rectangle(processed_img, (x, y),
                                            (x + w, y + h), color, 2)

                # Draw label with confidence
                text = f"{label}: {conf:.2f}"
                # Add black background for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(processed_img, (x, y - 25), (x + text_size[0], y), color, -1)
                processed_img = cv2.putText(processed_img, text, (x, y - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add warnings based on detections
            processed_img = self._add_warnings(processed_img, boxes, confidences, class_ids, detected_classes)

            # Add FPS info
            cv2.rectangle(processed_img, (5, 5), (250, 40), (0, 0, 0), -1)  # Black background
            processed_img = cv2.putText(processed_img, f"Driving Detection: {avg_fps:.1f} FPS",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

            # Add risk level assessment
            # Get image dimensions
            height, width = processed_img.shape[:2]
            risk_level = self._calculate_traffic_risk(boxes, class_ids, confidences)
            if risk_level == "HIGH":
                risk_color = (0, 0, 255)  # Red for high risk
            elif risk_level == "MEDIUM":
                risk_color = (0, 165, 255)  # Orange for medium risk
            else:
                risk_color = (0, 255, 0)  # Green for low risk

            # Create risk level indicator
            cv2.rectangle(processed_img, (width - 200, 5), (width - 5, 40), (0, 0, 0), -1)  # Black background
            processed_img = cv2.putText(processed_img, f"Traffic Risk: {risk_level}",
                                    (width - 190, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, risk_color, 2)

            # Return processed image and detection results
            indices = np.arange(len(boxes))  # Simple indices for compatibility
            return processed_img, boxes, confidences, class_ids, indices

        except Exception as e:
            print(f"Error communicating with detection server: {e}")
            self.connected = False
            return image_array, [], [], [], []

    def _add_warnings(self, img, boxes, confidences, class_ids, detected_classes):
        """Add warning overlays to the image based on detections"""
        global warning_timers

        # Priority classes for warnings
        priority_classes = {
            'person': 'CAUTION: PEDESTRIAN DETECTED',
            'car': 'VEHICLE AHEAD',
            'truck': 'LARGE VEHICLE AHEAD',
            'bus': 'BUS AHEAD',
            'stop sign': 'APPROACHING STOP SIGN',
            'traffic light': 'TRAFFIC LIGHT AHEAD',
            'bicycle': 'CYCLIST NEARBY',
            'motorcycle': 'MOTORCYCLE NEARBY'
        }

        # Get image dimensions
        height, width = img.shape[:2]

        # Tracking for warnings
        critical_warnings = []
        standard_warnings = []

        # Process each detection
        for i, cls_id in enumerate(class_ids):
            if cls_id < len(self.labels):
                label = self.labels[cls_id]
                conf = confidences[i]
                x, y, w, h = boxes[i]

                # Check if this is a class we want to warn about
                if label in priority_classes:
                    warning_msg = priority_classes[label]
                    box_area = w * h  # Proxy for distance
                    conf_percent = int(conf * 100)

                    # Add to appropriate warning list
                    if label in ['person', 'stop sign']:
                        critical_warnings.append((warning_msg, conf_percent, box_area, label))
                        # Play audio warning
                        self._play_audio_warning(label)
                    else:
                        standard_warnings.append((warning_msg, conf_percent, box_area, label))

        # Update warning timers for detected classes
        for class_name in detected_classes:
            if class_name in warning_timers:
                persistence = WARNING_PERSISTENCE.get(class_name, WARNING_PERSISTENCE['default'])
                warning_timers[class_name] = persistence

        # Handle warning persistence for classes not detected in this frame
        for class_name, timer in list(warning_timers.items()):
            if class_name not in detected_classes and timer > 0:
                # Decrease timer
                warning_timers[class_name] = timer - 1

                # Add persisted warning with reduced confidence
                if timer > 2:  # Only show persisted warnings that are still relevant
                    fading_conf = max(40, int(70 * (timer / WARNING_PERSISTENCE.get(class_name, WARNING_PERSISTENCE['default']))))
                    warning_msg = priority_classes.get(class_name, "Warning")
                    size_factor = 0.75  # Smaller than direct detections

                    # Add to appropriate warning list with "(Persisted)" tag
                    if class_name in ['person', 'stop sign']:
                        critical_warnings.append((f"{warning_msg} (Persisted)", fading_conf, size_factor, class_name))
                    else:
                        standard_warnings.append((f"{warning_msg} (Persisted)", fading_conf, size_factor, class_name))

        # Display warnings if we have any
        if critical_warnings or standard_warnings:
            # Sort by importance (box size/proximity)
            critical_warnings.sort(key=lambda x: x[2], reverse=True)
            standard_warnings.sort(key=lambda x: x[2], reverse=True)

            # Combine warnings, prioritizing critical ones
            all_warnings = critical_warnings[:2] + standard_warnings[:1]

            if all_warnings:
                # Get the pre-computed overlay or create one if needed
                num_warnings = min(len(all_warnings), 3)

                # Find the closest width match
                if width >= 640:
                    overlay_width = 800
                elif width >= 320:
                    overlay_width = 640
                else:
                    overlay_width = 320

                # Get the overlay
                key = (num_warnings, overlay_width)
                if key not in WARNING_OVERLAYS:
                    # Create new overlay if needed
                    warning_height = num_warnings * 45 + 15
                    overlay = np.zeros((warning_height, overlay_width, 3), dtype=np.uint8)
                    WARNING_OVERLAYS[key] = overlay

                # Create a copy of the base overlay
                overlay = WARNING_OVERLAYS[key].copy()
                warning_height = overlay.shape[0]

                # Create a semi-transparent black background
                overlay_region = img[height - warning_height:height, 0:width].copy()
                cv2.addWeighted(overlay[:, :width], 0.7, overlay_region, 0.3, 0, overlay_region)
                img[height - warning_height:height, 0:width] = overlay_region

                # Display warnings
                y_offset = height - warning_height + 40
                for warning, conf, _, class_name in all_warnings[:3]:
                    # Choose icon and color based on warning type
                    if "PEDESTRIAN" in warning:
                        icon = "!"
                        text_color = (0, 0, 255)  # Red
                    elif "STOP SIGN" in warning:
                        icon = "!"
                        text_color = (0, 0, 255)  # Red
                    elif "VEHICLE" in warning or "TRUCK" in warning or "BUS" in warning:
                        icon = ">"
                        text_color = (255, 255, 0)  # Yellow
                    elif "CYCLIST" in warning or "MOTORCYCLE" in warning:
                        icon = ">"
                        text_color = (255, 255, 0)  # Yellow
                    elif "TRAFFIC LIGHT" in warning:
                        icon = ">"
                        text_color = (0, 255, 255)  # Light blue
                    else:
                        icon = "!"
                        text_color = (255, 255, 255)  # White

                    warning_with_icon = f"{warning} ({conf}%)"

                    # Draw text with shadow for better visibility
                    cv2.putText(img, f"{warning_with_icon}",
                              (22, y_offset+2), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, (0, 0, 0), 4)  # Shadow

                    cv2.putText(img, f"{warning_with_icon}",
                              (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, text_color, 2)

                    # Add separator line
                    if y_offset + 45 < height:
                        cv2.line(img, (30, y_offset + 15), (width - 30, y_offset + 15),
                               (200, 200, 200), 1)

                    y_offset += 45

        return img
    # Play audio warning for critical detections
    def _play_audio_warning(self, warning_type):
        """Play audio warning for critical detections"""
        global last_audio_warning

        # Check if we should play a warning (avoid too frequent warnings)
        current_time = time.time()
        last_time = last_audio_warning.get(warning_type, 0)

        # Only play every 3 seconds for each type
        if current_time - last_time < 3.0:
            return

        # Update last warning time
        last_audio_warning[warning_type] = current_time

        # Don't create a new thread if one is still running
        if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
            return

        # Path to the warning sound
        sound_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "sound_warning", "car_beeping.mp3")

        # Make sure the file exists
        if not os.path.exists(sound_file):
            print(f"Warning: Sound file not found: {sound_file}")
            return

        try:
            if sys.platform == 'win32':
                # On Windows, use a different approach for MP3 files
                import winsound
                # Use the built-in Windows alert sound as fallback
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

                # Optionally: If you want to play the actual MP3, use this instead
                # This requires the system's default MP3 player
                # self.audio_thread = threading.Thread(
                #     target=lambda: os.startfile(sound_file))
                # self.audio_thread.daemon = True
                # self.audio_thread.start()

                print(f"Playing warning sound for {warning_type}")
            else:
                # Linux/Mac platforms (requires mpg123 or similar)
                sound_command = f'mpg123 "{sound_file}" 2>/dev/null || ' \
                            f'ffplay -nodisp -autoexit -loglevel quiet "{sound_file}" || ' \
                            f'mplayer -really-quiet "{sound_file}"'

                # Create and start thread
                self.audio_thread = threading.Thread(
                    target=lambda: subprocess.run(sound_command, shell=True, timeout=5))
                self.audio_thread.daemon = True
                self.audio_thread.start()
                print(f"Playing warning sound for {warning_type}")
        except Exception as e:
            print(f"Warning: Audio playback error: {e}")

    def _print_performance_metrics(self, results):
        """Print detailed performance metrics about the detection"""
        # Skip if no performance data available
        if 'processing_time' not in results:
            return

        processing_time = results['processing_time']
        fps = results.get('fps', 1.0/processing_time if processing_time > 0 else 0)
        inference_time = results.get('inference_time', processing_time * 0.7)  # Estimate if not provided
        preprocess_time = results.get('preprocess_time', processing_time * 0.1)
        postprocess_time = results.get('postprocess_time', processing_time * 0.2)
        model_type = results.get('model_type', 'YOLOv8')

        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        # Create a cleaner, more readable performance report
        performance_str = (
            f"\n{'=' * 35}\n"
            f"{model_type} Detection Metrics:\n"
            f"  ⏱️ Preprocessing: {preprocess_time*1000:.1f}ms\n"
            f"  🧠 Inference:     {inference_time*1000:.1f}ms\n"
            f"  🔍 Postprocess:   {postprocess_time*1000:.1f}ms\n"
            f"  📊 Total time:    {processing_time*1000:.1f}ms\n"
            f"  🚀 Current FPS:   {fps:.1f}\n"
            f"  📈 Average FPS:   {avg_fps:.1f}\n"
            f"  🎯 Detections:    {len(results['boxes'])}\n"
            f"{'=' * 35}"
        )
        print(performance_str)

    def close(self):
        """Close the connection"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False

    def _calculate_traffic_risk(self, boxes, class_ids, confidences):
        """Calculate traffic risk based on detected objects and their positions"""
        if not boxes:
            return "LOW"

        # Get image dimensions
        height, width = 600, 800

        # Critical class indicators - presence immediately raises risk level
        critical_classes = {'person', 'stop sign'}
        high_risk_classes = {'traffic light', 'motorcycle', 'bicycle'}

        # Initialize counters and metrics
        risk_score = 0
        max_single_risk = 0
        critical_count = 0
        high_risk_count = 0
        vehicle_count = 0
        central_hazard = False

        # Analyze each detection
        for i, box in enumerate(boxes):
            if i >= len(confidences) or i >= len(class_ids):
                continue

            x, y, w, h = box
            cls_id = class_ids[i]
            conf = confidences[i]

            # Skip irrelevant classes or low confidence
            #if cls_id not in self.driving_relevant_classes or conf < 0.25:
            if cls_id not in self.driving_relevant_classes or conf < 0.4:
                continue

            label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"

            # Count by category
            if label in critical_classes:
                critical_count += 1
            elif label in high_risk_classes:
                high_risk_count += 1
            elif label in ['car', 'truck', 'bus']:
                vehicle_count += 1

            # Calculate position metrics
            box_area = w * h
            image_area = height * width
            size_ratio = box_area / image_area

            center_x = x + w/2
            center_y = y + h/2
            center_ratio_x = center_x / width  # 0 to 1 from left to right
            center_ratio_y = center_y / height  # 0 to 1 from top to bottom

            # Check for central hazards with substantial size
            in_center = (0.3 < center_ratio_x < 0.7)
            is_close = (center_ratio_y > 0.6)
            is_substantial = (size_ratio > 0.02)  # Object occupies at least 2% of frame

            if in_center and is_close and is_substantial and label in ['person', 'car', 'truck', 'motorcycle']:
                central_hazard = True

            # Base risk by object type (more aggressive values)
            base_risk = {
                'person': 1.0,      # Highest risk
                'stop sign': 0.9,   # Very important
                'traffic light': 0.8,
                'car': 0.6,
                'truck': 0.7,
                'bus': 0.7,
                'motorcycle': 0.8,
                'bicycle': 0.8,
            }.get(label, 0.2)

            # Add position factors (simplified)
            position_factor = 1.0
            if in_center:
                position_factor = 1.5  # Higher weight for central objects
            if is_close:
                position_factor *= 1.5  # Higher weight for close objects

            # Calculate object risk score
            object_risk = base_risk * position_factor * conf

            # Track highest single object risk
            max_single_risk = max(max_single_risk, object_risk)

            # Add to total risk score
            risk_score += object_risk

        # Rule-based decision
        if central_hazard or critical_count >= 1 or max_single_risk > 0.8:
            return "HIGH"
        elif (high_risk_count >= 1 or vehicle_count >= 2 or max_single_risk > 0.5 or
            risk_score > 1.0):  # Reduced threshold
            return "MEDIUM"
        else:
            return "LOW"

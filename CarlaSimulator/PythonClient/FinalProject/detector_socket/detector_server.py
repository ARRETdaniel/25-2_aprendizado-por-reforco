# Author: Daniel Terra Gomes
# Date: Jun 30, 2025
# detector_server.py (Python 3.12)

import socket
import json
import numpy as np
import cv2
from ultralytics import YOLO
import time
import threading
import msgpack
import msgpack_numpy as m
import sys
m.patch()  # Patch msgpack to handle numpy arrays efficiently
import gc
from queue import Queue, Empty
import torch
import signal

class DetectionServer:
    def __init__(self, host="localhost", port=5555, model_type="yolov8s.pt", confidence=0.55,
             batch_size=4, batch_timeout=0.05):
        self.host = host
        self.port = port
        self.confidence = confidence

        # Check CUDA availability before loading model
        self.has_cuda = torch.cuda.is_available()

        if self.has_cuda:
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = "0"  # Use first GPU
            # Clean GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("CUDA is not available. Using CPU")
            self.device = "cpu"

        # Load multiple models
        self.model_fast = YOLO("yolov8n.pt")  # Fast model for normal driving
        self.model_accurate = YOLO("yolov8s.pt")  # Accurate model for critical scenarios

        # Preload models to GPU if available
        if self.has_cuda:
            print("Warming up models...")
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model_fast(dummy_input, device=self.device)
            _ = self.model_accurate(dummy_input, device=self.device)

        # Initialize socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Detection server running on {host}:{port}")

        # For warning persistence
        self.warning_timers = {
            'person': 0,
            'car': 0,
            'stop sign': 0,
            'traffic light': 0
        }

        # Warning persistence settings
        self.warning_persistence = {
            'person': 15,       # Critical - persist for 15 frames
            'stop sign': 15,    # Critical
            'traffic light': 10,# Important
            'default': 8        # Standard
        }

        # For audio warnings
        self.last_audio_warning = {}
        self.audio_thread = None

        # COCO class labels
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

        # Define driving-relevant class IDs from COCO dataset
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
            80: 'traffic cone'     # custom class for traffic cones if available
        }

        # Initialize these BEFORE creating any threads
        self.running = True
        self.stop_event = threading.Event()

        # Add these new parameters
        self.batch_size = batch_size          # Max number of images to batch
        self.batch_timeout = batch_timeout    # Max time to wait for batch completion (seconds)
        self.batch_queue = Queue()            # Queue for batch processing
        self.batch_results = {}               # Store results by client ID
        self.batch_lock = threading.Lock()    # Lock for thread safety

        # Now create and start the thread
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()

    def _batch_processor(self):
        """Process images in batches for efficiency"""
        while not self.stop_event.is_set():
            try:
                # Get first item with timeout
                first_item = self.batch_queue.get(timeout=0.1)
                batch_items = [first_item['image']]
                batch_ids = [first_item['id']]

                # Try to get more items without blocking
                timeout = time.time() + self.batch_timeout
                while len(batch_items) < self.batch_size and time.time() < timeout:
                    try:
                        item = self.batch_queue.get(block=False)
                        batch_items.append(item['image'])
                        batch_ids.append(item['id'])
                    except:
                        # Queue is empty, wait a bit
                        time.sleep(0.005)
                        continue

                # Process the batch if we have any items
                if batch_items:
                    # Convert to batch tensor format
                    batch_array = np.stack(batch_items)

                    # Run batch inference
                    start_time = time.time()
                    results = self.model_accurate(batch_array, conf=self.confidence, device=self.device)
                    processing_time = time.time() - start_time

                    # Process each result separately
                    for i, (result, batch_id) in enumerate(zip(results, batch_ids)):
                        # Apply domain-specific filtering
                        filtered_results = self._process_result(result, batch_items[i])

                        # Store results
                        with self.batch_lock:
                            fps = 1.0 / (processing_time / len(batch_items))
                            filtered_results['fps'] = fps
                            filtered_results['processing_time'] = processing_time / len(batch_items)
                            self.batch_results[batch_id] = filtered_results

            except Empty:
                # Queue is empty, just wait quietly without error messages
                time.sleep(0.05)
            except Exception as e:
                if str(e):  # Only log non-empty error messages
                    print(f"Batch processing error: {e}")
                time.sleep(0.05)

    def _process_result(self, result, original_image):
        """Process an individual result from batch detection"""
        height, width = original_image.shape[:2]
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []

        # Extract detections
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Only proceed if this is a driving-relevant class
            if cls_id not in self.driving_relevant_classes:
                continue

            label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"

            # Spatial context validation (adapted from your YOLOv3 implementation)
            if label == 'person':
                # Size validation
                person_area = w * h
                image_area = height * width
                person_ratio = person_area / image_area

                if person_ratio < 0.005 or person_ratio > 0.8:
                    continue

                # Position validation (not floating in the air)
                if y < height / 3:
                    continue
            elif label in ['car', 'truck', 'bus']:
                vehicle_area = w * h
                image_area = height * width
                vehicle_ratio = vehicle_area / image_area

                # Skip unrealistically small or large vehicles
                if vehicle_ratio < 0.003 or vehicle_ratio > 0.9:
                    continue

                # Vehicles should be in the bottom 2/3 of the frame
                if y < height / 3:
                    continue
            elif label == 'traffic light':
                # Traffic lights are typically small and in the upper half
                light_area = w * h
                image_area = height * width
                light_ratio = light_area / image_area

                # Skip unrealistically large traffic lights
                if light_ratio > 0.05:
                    continue

                # Traffic lights should generally be in the upper half
                if y > height * 2/3:
                    continue
            elif label == 'stop sign':
                # Stop signs are typically small to medium
                sign_area = w * h
                image_area = height * width
                sign_ratio = sign_area / image_area

                # Skip unrealistically large stop signs
                if sign_ratio > 0.1:
                    continue

            # Add valid detection
            filtered_boxes.append([x, y, w, h])
            filtered_confidences.append(conf)
            filtered_class_ids.append(cls_id)


        # Apply temporal consistency if we have enough detections
        if len(filtered_boxes) > 0:
            filtered_boxes, filtered_confidences, filtered_class_ids = self.apply_temporal_consistency(
                filtered_boxes, filtered_confidences, filtered_class_ids)

        return {
            'boxes': filtered_boxes,
            'confidences': filtered_confidences,
            'class_ids': filtered_class_ids
        }

    def apply_temporal_consistency(self, boxes, confidences, class_ids):
        """Filter out false positives using temporal consistency"""
        if not hasattr(self, 'detection_history'):
            self.detection_history = {}

        # Current detections
        current_detections = {}
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            # Create a key from class and position
            x, y, w, h = box
            center_x, center_y = x + w//2, y + h//2
            pos_key = f"{center_x//20}_{center_y//20}"
            det_key = f"{cls_id}_{pos_key}"
            current_detections[det_key] = i

            # Update history
            if det_key in self.detection_history:
                self.detection_history[det_key] = min(5, self.detection_history[det_key] + 1)
            else:
                self.detection_history[det_key] = 1

        # Filter based on history
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        for det_key, idx in current_detections.items():
            # Keep if seen multiple times or high confidence
            if self.detection_history[det_key] >= 2 or confidences[idx] > 0.7:
                filtered_boxes.append(boxes[idx])
                filtered_confidences.append(confidences[idx])
                filtered_class_ids.append(class_ids[idx])

        # Clean up old detections
        for key in list(self.detection_history.keys()):
            if key not in current_detections:
                self.detection_history[key] -= 1
                if self.detection_history[key] <= 0:
                    del self.detection_history[key]

        return filtered_boxes, filtered_confidences, filtered_class_ids

    def handle_client(self, client_socket):
        """Handle communication with a single client"""
        try:
            # Add tracking for adaptive frame rate
            last_frame_time = 0
            last_results = None
            frame_interval = 0.05  # Default: 20 FPS max
            complexity_history = []  # Track scene complexity

            while True:
                try:
                    # First receive the size of the incoming data
                    data_size_bytes = client_socket.recv(4)
                    if not data_size_bytes:
                        print("Client disconnected (received empty data)")
                        break

                    # Process data with more debugging
                    data_size = int.from_bytes(data_size_bytes, byteorder='little')
                    print(f"Receiving data of size {data_size} bytes")

                    received_data = b''
                    while len(received_data) < data_size:
                        chunk = client_socket.recv(min(4096, data_size - len(received_data)))
                        if not chunk:
                            print("Connection broken during data transfer")
                            break
                        received_data += chunk

                    if len(received_data) < data_size:
                        print(f"Incomplete data received: {len(received_data)}/{data_size} bytes")
                        continue

                    # Unpack the data with error handling
                    try:
                        unpacked_data = msgpack.unpackb(received_data, raw=False, object_hook=m.decode)
                    except Exception as e:
                        print(f"Error unpacking data: {e}")
                        empty_results = {'boxes': [], 'confidences': [], 'class_ids': [],
                                        'fps': 0, 'processing_time': 0, 'error': str(e)}
                        self.send_data(client_socket, empty_results)
                        continue

                    # Get the image from the unpacked data
                    if 'image_compressed' in unpacked_data:
                        # Handle compressed image
                        image = unpacked_data
                        vehicle_state = unpacked_data.get('vehicle_state', None)
                    elif 'image' in unpacked_data:
                        # Handle uncompressed image
                        image = unpacked_data['image']
                        vehicle_state = unpacked_data.get('vehicle_state', None)
                    else:
                        print(f"No image data found in received message. Keys: {unpacked_data.keys()}")
                        empty_results = {'boxes': [], 'confidences': [], 'class_ids': [],
                                        'fps': 0, 'processing_time': 0, 'error': "No image data"}
                        self.send_data(client_socket, empty_results)
                        continue
                    current_time = time.time()
                    time_since_last_frame = current_time - last_frame_time

                    # Determine if we should process this frame based on:
                    # 1. Adaptive frame interval
                    # 2. Vehicle state (speed, near_intersection)
                    # 3. Previous detection results (if critical objects detected)
                    # 4. Scene complexity

                    # Calculate scene complexity (0-1) based on:
                    # - Number of objects detected previously
                    # - Vehicle speed (faster = need more processing)
                    # - Proximity to intersections

                    vehicle_speed = vehicle_state.get('speed', 0) if vehicle_state else 0
                    near_intersection = vehicle_state.get('near_intersection', False) if vehicle_state else False

                    # Complexity factors
                    num_objects = len(last_results['boxes']) if last_results else 0
                    obj_complexity = min(1.0, num_objects / 10.0)  # Normalize: 10+ objects = max complexity
                    speed_factor = min(1.0, vehicle_speed / 20.0)  # Normalize: 20+ m/s = max complexity
                    intersection_factor = 1.0 if near_intersection else 0.0

                    # Combined complexity (0-1 scale)
                    current_complexity = max(obj_complexity, speed_factor, intersection_factor)

                    # Update history (moving average)
                    complexity_history.append(current_complexity)
                    if len(complexity_history) > 10:
                        complexity_history.pop(0)
                    avg_complexity = sum(complexity_history) / len(complexity_history)

                    # Adjust frame interval based on complexity
                    # More complex scenes = higher frame rate (smaller interval)
                    if avg_complexity > 0.8:
                        frame_interval = 0.03  # ~33 FPS for complex scenes
                    elif avg_complexity > 0.5:
                        frame_interval = 0.05  # ~20 FPS for moderate scenes
                    elif avg_complexity > 0.2:
                        frame_interval = 0.1   # ~10 FPS for simple scenes
                    else:
                        frame_interval = 0.2   # ~5 FPS for very simple scenes

                    # Always process if critical conditions are met
                    critical_scenario = (
                        near_intersection or
                        vehicle_speed > 15.0 or  # High speed
                        (last_results and self.has_critical_objects(last_results))
                    )

                    # Skip frame if interval not met and not critical
                    if time_since_last_frame < frame_interval and not critical_scenario:
                        # Send cached results for non-critical frames
                        if last_results:
                            self.send_data(client_socket, last_results)
                        continue

                    # Process the image (normal flow)
                    results = self.process_image(image, vehicle_state)
                    last_results = results
                    last_frame_time = current_time

                    # Send results
                    self.send_data(client_socket, results)

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    empty_results = {'boxes': [], 'confidences': [], 'class_ids': [],
                                     'fps': 0, 'processing_time': 0, 'error': str(e)}
                    self.send_data(client_socket, empty_results)

        except Exception as e:
            print(f"Client connection error: {e}")
        finally:
            client_socket.close()
            print("Client disconnected")

    def run(self):
        """Run the detection server"""
        # Set up signal handler
        def signal_handler(sig, frame):
            print("\nReceived shutdown signal")
            self.stop()
            # Allow main thread to exit
            sys.exit(0)

        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                try:
                    print("Waiting for client connection...")
                    client_socket, addr = self.server_socket.accept()
                    print(f"Client connected from {addr}")

                    # Handle each client in a separate thread
                    client_thread = threading.Thread(target=self.handle_client,
                                                   args=(client_socket,))
                    client_thread.daemon = True
                    client_thread.start()

                except KeyboardInterrupt:
                    print("Server shutdown requested")
                    break
                except Exception as e:
                    print(f"Server error: {e}")
                    break
        finally:
            self.stop()  # Make sure to call stop() to clean up resources

    def send_data(self, client_socket, data):
        """Send data to the client with message compression"""
        # Pack the data
        packed_data = msgpack.packb(data, default=m.encode)

        # Send the size first, then the data
        data_size = len(packed_data).to_bytes(4, byteorder='little')
        client_socket.sendall(data_size + packed_data)

    def has_critical_objects(self, results):
        """Determine if the detection results contain critical objects"""
        # Check for specific critical objects with position analysis
        for i, cls_id in enumerate(results['class_ids']):
            conf = results['confidences'][i]
            box = results['boxes'][i]
            x, y, w, h = box

            # Person with high confidence (pedestrian)
            if cls_id == 0 and conf > 0.6:
                box_area = w * h
                # Large person (close) or central in image is critical
                if box_area > 5000 or (x > 160 and x < 480):  # Central in 640-width image
                    return True

            # Stop sign with decent confidence
            #elif cls_id == 11 and conf > 0.45:  # Stop sign
            elif cls_id == 11 and conf > 0.5:  # Stop sign
                return True

            # Traffic light
            elif cls_id == 9 and conf > 0.5:  # Traffic light
                return True

            # Vehicles very close (large bounding box)
            elif cls_id in [2, 5, 7] and conf > 0.6:  # Car, bus, truck
                box_area = w * h
                if box_area > 20000:  # Very close vehicle
                    return True

        return False

    def _get_roi_mask(self, height, width, vehicle_speed):
        """Generate a region of interest mask based on speed"""
        mask = np.ones((height, width), dtype=np.uint8)

        # At higher speeds, focus more on the horizon and center
        if vehicle_speed > 10:  # m/s (36 km/h)
            # Create a trapezoid ROI focusing on the road ahead
            points = np.array([[width//4, height],
                              [width*3//4, height],
                              [width*5//6, height//2],
                              [width//6, height//2]])
            cv2.fillConvexPoly(mask, points, 0)  # Mask out irrelevant areas

        return mask

    def detect_traffic_signs(self, image):
        """Specialized detector for traffic signs and lights"""
        # Use an alternative detector optimized for signs
        # Consider using a classifier after detection for better sign recognition
        # ...

    def _multi_scale_detection(self, image, vehicle_speed):
        """Run detection at multiple scales based on vehicle speed"""
        if vehicle_speed < 5:  # Slow speed
            # Higher resolution for nearby objects
            results = self.model(image, size=640)
        else:
            # Process at smaller size for efficiency at speed
            results = self.model(image, size=416)

        return results

    def _manage_gpu_memory(self, force=False):
        """Manage GPU memory with periodic cleanup"""
        if not hasattr(self, 'last_cleanup_time'):
            self.last_cleanup_time = time.time()
            self.processed_frames = 0

        # Update frame counter
        self.processed_frames += 1

        # Check if cleanup is needed
        current_time = time.time()
        time_since_cleanup = current_time - self.last_cleanup_time

        # Cleanup every 5 minutes or every 1000 frames, or when forced
        if force or time_since_cleanup > 300 or self.processed_frames >= 1000:
            if self.has_cuda:
                # Get current GPU memory usage before cleanup
                if hasattr(torch.cuda, 'memory_allocated'):
                    before_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

                # Run Python garbage collector
                gc.collect()

                # Clear CUDA cache
                torch.cuda.empty_cache()

                # Log memory usage difference
                if hasattr(torch.cuda, 'memory_allocated'):
                    after_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    print(f"GPU Memory cleanup: {before_mem:.1f}MB → {after_mem:.1f}MB (freed {before_mem - after_mem:.1f}MB)")

            # Reset counters
            self.last_cleanup_time = current_time
            self.processed_frames = 0
            return True

        return False

    def process_image(self, image_data, vehicle_state=None):
        """Process the image with domain-specific filtering"""
        # Start timing
        start_time = time.time()

        original_vehicle_state = vehicle_state  # Store original vehicle state

        # Check if the image is compressed
        if isinstance(image_data, dict) and 'image_compressed' in image_data:
            # Decompress the image
            compressed_data = image_data['image_compressed']
            shape = image_data['shape']

            # Save vehicle state if present in the dict
            if 'vehicle_state' in image_data:
                vehicle_state = image_data['vehicle_state']

            # Convert compressed bytes to numpy array
            np_arr = np.frombuffer(compressed_data, np.uint8)

            # Decode the JPEG image
            image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Make sure the shape is correct
            if image_data.shape != tuple(shape):
                # Resize if needed
                image_data = cv2.resize(image_data, (shape[1], shape[0]))
        elif isinstance(image_data, dict) and 'image' in image_data:
            # Handle the case where we get {'image': ...} format
            if 'vehicle_state' in image_data:
                vehicle_state = image_data['vehicle_state']
            image_data = image_data['image']

        # Now image_data is always a numpy array, and vehicle_state is either
        # the passed parameter, extracted from dict, or None

        # Add frame counter for handling initial frames differently
        self.frame_counter = getattr(self, 'frame_counter', 0) + 1
        debug_mode = False  # Set to False in production
        debug_frame = self.frame_counter % 30 == 0  # Save every 30th frame

        height, width = image_data.shape[:2]
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []

        # Try fast model first for efficiency
        results_fast = self.model_fast(image_data, conf=self.confidence, device=self.device)

        # Check if we have any results
        has_detections = False
        for r in results_fast:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # Only count driving-relevant detections
                if cls_id in self.driving_relevant_classes:
                    has_detections = True
                    break
            if has_detections:
                break

        # If no detections with fast model or periodically, use accurate model
        if not has_detections or self.frame_counter % 10 == 0:
            print("Using accurate model for this frame")
            results = self.model_accurate(image_data, conf=self.confidence * 0.7, device=self.device)
        else:
            results = results_fast

        # Debug visualization of raw detections (if enabled)
        if debug_mode and debug_frame:
            debug_img = image_data.copy()
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"

                    # Draw raw detection
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(debug_img, f"{label} {conf:.2f}",
                               (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the debug image
            import os
            os.makedirs("debug_frames", exist_ok=True)
            cv2.imwrite(f"debug_frames/raw_frame_{self.frame_counter}.jpg", debug_img)

        # Extract and filter results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Skip non-driving relevant classes
                if cls_id not in self.driving_relevant_classes:
                    continue

                label = self.labels[cls_id] if cls_id < len(self.labels) else "unknown"

                # Spatial context validation with special handling for distant objects
                if label == 'person':
                    # Size validation
                    person_area = w * h
                    image_area = height * width
                    person_ratio = person_area / image_area

                    if person_ratio < 0.003 or person_ratio > 0.8:  # More permissive
                        continue

                    # Position validation (not floating in the air)
                    if y < height / 3:
                        continue
                elif label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_area = w * h
                    image_area = height * width
                    vehicle_ratio = vehicle_area / image_area

                    # HORIZON ZONE: Special case for distant vehicles on the horizon
                    horizon_zone = (height * 0.35 <= y <= height * 0.55)
                    if horizon_zone:
                        # Much more permissive filtering for objects on the horizon
                        if vehicle_ratio < 0.0001:  # Extremely small allowed if on horizon
                            if conf > self.confidence * 1.1:  # But require slightly higher confidence
                                filtered_boxes.append([x, y, w, h])
                                filtered_confidences.append(conf)
                                filtered_class_ids.append(cls_id)
                            continue  # Skip normal filtering checks

                    # Normal filtering for non-horizon objects
                    if vehicle_ratio < 0.0005 or vehicle_ratio > 0.95:  # Even more permissive
                        continue

                    # Allow vehicles higher in the frame for distant detection
                    if y < height / 5:  # More permissive
                        continue
                elif label == 'traffic light':
                    # Traffic lights are typically small and in the upper half
                    light_area = w * h
                    image_area = height * width
                    light_ratio = light_area / image_area

                    # Skip unrealistically large traffic lights
                    if light_ratio > 0.05:
                        continue

                    # Traffic lights should generally be in the upper half
                    if y > height * 2/3:
                        continue
                elif label == 'stop sign':
                    # Stop signs are typically small to medium
                    sign_area = w * h
                    image_area = height * width
                    sign_ratio = sign_area / image_area

                    # Skip unrealistically large stop signs
                    if sign_ratio > 0.1:
                        continue
                # Add valid detection
                filtered_boxes.append([x, y, w, h])
                filtered_confidences.append(conf)
                filtered_class_ids.append(cls_id)

        # Apply temporal consistency if we have enough detections
        if len(filtered_boxes) > 0:
            # Only apply temporal consistency after some frames
            if self.frame_counter > 15:  # Skip first 15 frames
                filtered_boxes, filtered_confidences, filtered_class_ids = self.apply_temporal_consistency(
                    filtered_boxes, filtered_confidences, filtered_class_ids)
            else:
                print(f"Skipping temporal consistency for frame {self.frame_counter}")

        # Call memory management (not forcing cleanup)
        self._manage_gpu_memory(force=False)

        processing_time = time.time() - start_time
        return {
            'boxes': filtered_boxes,
            'confidences': filtered_confidences,
            'class_ids': filtered_class_ids,
            'fps': 1.0 / processing_time,
            'processing_time': processing_time
        }

    def force_cleanup(self):
        """Force GPU memory cleanup"""
        return self._manage_gpu_memory(force=True)

    def stop(self):
        """Gracefully stop the server"""
        print("Stopping server...")
        self.running = False
        self.stop_event.set()

        # Wait for batch thread to finish
        if hasattr(self, 'batch_thread') and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=2.0)

        # Clean up resources
        self._manage_gpu_memory(force=True)

        # Close socket
        if hasattr(self, 'server_socket'):
            self.server_socket.close()

        print("Server shutdown complete")

    def shutdown(self):
        """Clean shutdown of the server"""
        # Force final memory cleanup
        self._manage_gpu_memory(force=True)

        # Close socket
        if hasattr(self, 'server_socket'):
            self.server_socket.close()

        print("Server shutdown complete")

if __name__ == "__main__":
    try:
        # Create and run the detection server
        server = DetectionServer()
        server.run()
    except KeyboardInterrupt:
        print("Server shutdown requested by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server shutdown complete")

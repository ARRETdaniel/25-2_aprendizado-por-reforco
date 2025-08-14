import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch
import json
import urllib.request
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def download_file(url, local_path):
    """Download a file if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"Download completed successfully")
            return True
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    return True

def ensure_model_files():
    """Ensure all required model files are available."""
    model_files = {
        # YOLOv3 and v4 files (original files)
        "yolov3": {
            "cfg": "../yolov3-coco/yolov3.cfg",
            "weights": "../yolov3-coco/yolov3.weights",
            "url_cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "url_weights": "https://pjreddie.com/media/files/yolov3.weights"
        },
        "yolov4": {
            "cfg": "../yolov4-coco/yolov4.cfg",
            "weights": "../yolov4-coco/yolov4.weights",
            "url_cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "url_weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        },
        "yolov3-tiny": {
            "cfg": "../yolov3-coco/yolov3-tiny.cfg",
            "weights": "../yolov3-coco/yolov3-tiny.weights",
            "url_cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
            "url_weights": "https://pjreddie.com/media/files/yolov3-tiny.weights"
        }
    }

    # Ensure directories exist
    os.makedirs("../yolov3-coco", exist_ok=True)
    os.makedirs("../yolov4-coco", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # Ensure COCO labels exist
    if not os.path.exists("../yolov3-coco/coco-labels"):
        print("Downloading COCO labels...")
        labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        with open("../yolov3-coco/coco-labels", "w") as f:
            f.write("\n".join(labels))

    # Download required model files
    for model, files in model_files.items():
        download_file(files["url_cfg"], files["cfg"])
        download_file(files["url_weights"], files["weights"])

    # For YOLOv5-v8, we'll use PyTorch Hub or local installation at runtime

def detect_with_darknet(net, layer_names, image, input_size=(416, 416), conf_threshold=0.5, nms_threshold=0.4):
    """Run detection with Darknet-based models (YOLOv3/v4)."""
    height, width = image.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)

    # Try to use CUDA if available
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except:
        # Fallback to CPU if CUDA fails
        pass

    # Get detections
    start_time = time.time()
    outs = net.forward(layer_names)
    inference_time = time.time() - start_time

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                # Get coordinates (YOLO format: center_x, center_y, width, height)
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")

                # Calculate top-left corner coordinates
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)

                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        indices = indices.flatten() if len(indices) > 0 else []

    return boxes, confidences, class_ids, indices, inference_time

# Modify detect_with_ultralytics function to measure only inference time consistently
# Fix in the detect_with_ultralytics function for YOLOv8
def detect_with_ultralytics(model_name, image, conf_threshold=0.5, device='cuda'):
    """Run detection with Ultralytics-based models (YOLOv5/v8)."""
    if 'yolov8' in model_name.lower():
        # For YOLOv8, handle the model differently
        try:
            from ultralytics import YOLO

            # Load the model
            model = YOLO(model_name)

            # Set the confidence threshold
            model.conf = conf_threshold

            # Convert image if needed
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image.copy()

            # Run inference with timing
            start_time = time.time()
            results = model(img, verbose=False)  # Disable verbose output
            inference_time = time.time() - start_time

            # Process results
            boxes = []
            confidences = []
            class_ids = []

            # Extract detections
            if hasattr(results[0], 'boxes'):
                detections = results[0].boxes
                for i in range(len(detections)):
                    box = detections.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    conf = float(detections.conf[i])
                    cls_id = int(detections.cls[i])

                    width = x2 - x1
                    height = y2 - y1

                    boxes.append([x1, y1, width, height])
                    confidences.append(conf)
                    class_ids.append(cls_id)

            # Generate indices for all detections
            indices = list(range(len(boxes)))

            return boxes, confidences, class_ids, indices, inference_time

        except Exception as e:
            print(f"Error with YOLOv8: {e}")
            # Try alternative implementation for YOLOv8
            try:
                # Load model through torch hub as a backup method
                model = torch.hub.load('ultralytics/yolov5', 'custom',
                                     path=model_name,
                                     device=device)
                model.conf = conf_threshold

                # Rest of the function similar to YOLOv5
                start_time = time.time()
                results = model(image)
                inference_time = time.time() - start_time

                # Extract detections
                detections = results.xyxy[0].cpu().numpy()

                boxes = []
                confidences = []
                class_ids = []

                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    width = x2 - x1
                    height = y2 - y1

                    boxes.append([int(x1), int(y1), int(width), int(height)])
                    confidences.append(float(conf))
                    class_ids.append(int(cls_id))

                indices = list(range(len(boxes)))
                return boxes, confidences, class_ids, indices, inference_time

            except Exception as e:
                print(f"Failed alternative method for YOLOv8: {e}")
                return [], [], [], [], 0
    else:
        # For YOLOv5, use the existing implementation
        try:
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            model.conf = conf_threshold

            # Use GPU if available
            if device == 'cuda' and torch.cuda.is_available():
                model = model.to(device)
            else:
                device = 'cpu'
                model = model.to(device)

            # Run inference
            start_time = time.time()
            results = model(image)
            inference_time = time.time() - start_time

            # Process results
            detections = results.xyxy[0].cpu().numpy()

            boxes = []
            confidences = []
            class_ids = []

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                width = x2 - x1
                height = y2 - y1

                boxes.append([int(x1), int(y1), int(width), int(height)])
                confidences.append(float(conf))
                class_ids.append(int(cls_id))

            indices = list(range(len(boxes)))
            return boxes, confidences, class_ids, indices, inference_time

        except Exception as e:
            print(f"Error with YOLOv5: {e}")
            return [], [], [], [], 0

def draw_detections(image, boxes, confidences, class_ids, indices, labels, colors):
    """Draw detection boxes on an image."""
    image_copy = image.copy()

    if len(indices) > 0:
        for i in indices:
            # Get the bounding box
            x, y, w, h = boxes[i]

            # Make sure class_id is within range of labels
            class_id = class_ids[i]
            if class_id < len(labels):
                label_text = f"{labels[class_id]}: {confidences[i]:.2f}"
            else:
                label_text = f"Class {class_id}: {confidences[i]:.2f}"

            # Get color (ensure it's within range)
            color_index = class_id % len(colors)
            color = [int(c) for c in colors[color_index]]

            # Draw rectangle and label
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_copy, label_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_copy

# Fix for run_multiple_iterations function to handle CUDA backend issues
def run_multiple_iterations(model_info, image, input_size, labels, colors, num_runs=5, conf_threshold=0.5):
    """Run multiple iterations of the model for reliable timing."""
    print(f"Running {model_info['name']} for {num_runs} iterations...")
    inference_times = []
    all_detections = []

    for i in range(num_runs + 1):  # +1 for warmup
        try:
            if model_info['family'] == 'darknet':
                # YOLOv3/v4 models - Use CPU only for darknet models
                net = cv2.dnn.readNetFromDarknet(model_info['cfg'], model_info['weights'])

                # IMPORTANT: DO NOT set CUDA backend for darknet models - use CPU only
                backend = "CPU"

                # Get output layers
                layer_names = net.getLayerNames()
                try:
                    # Different versions of OpenCV have different return types
                    output_indices = net.getUnconnectedOutLayers()
                    if isinstance(output_indices, np.ndarray):
                        if output_indices.ndim > 0 and isinstance(output_indices[0], np.ndarray):
                            output_layers = [layer_names[j[0]-1] for j in output_indices]
                        else:
                            output_layers = [layer_names[j-1] for j in output_indices]
                    else:
                        output_layers = [layer_names[j-1] for j in output_indices]
                except:
                    # Fallback method
                    output_layers = [layer_names[j-1] for j in net.getUnconnectedOutLayers().flatten()]

                # Run detection
                blob = cv2.dnn.blobFromImage(image, 1/255.0, input_size, swapRB=True, crop=False)
                net.setInput(blob)

                start_time = time.time()
                outs = net.forward(output_layers)
                inference_time = time.time() - start_time

                # Process detections
                boxes = []
                confidences = []
                class_ids = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > conf_threshold:
                            # Get coordinates
                            center_x = int(detection[0] * image.shape[1])
                            center_y = int(detection[1] * image.shape[0])
                            w = int(detection[2] * image.shape[1])
                            h = int(detection[3] * image.shape[0])

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # Apply non-maximum suppression
                indices = []
                if len(boxes) > 0:
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
                    if len(indices) > 0:
                        if isinstance(indices, np.ndarray):
                            indices = indices.flatten()
                        else:
                            indices = [int(i) for i in indices.flatten()]

            elif model_info['family'] == 'ultralytics':
                # Use GPU for ultralytics models (YOLOv5/YOLOv8)
                backend = "CUDA" if torch.cuda.is_available() else "CPU"

                # Different handling based on model type
                if 'yolov8' in model_info['model'].lower():
                    from ultralytics import YOLO
                    model = YOLO(model_info['model'])
                    model.conf = conf_threshold

                    # Run inference
                    start_time = time.time()
                    results = model(image, verbose=False)
                    inference_time = time.time() - start_time

                    # Process results
                    boxes = []
                    confidences = []
                    class_ids = []

                    # Extract detections from results
                    if len(results) > 0:
                        boxes_data = results[0].boxes
                        if len(boxes_data) > 0:
                            for j in range(len(boxes_data)):
                                try:
                                    box = boxes_data.xyxy[j].cpu().numpy()
                                    x1, y1, x2, y2 = map(int, box)
                                    width = x2 - x1
                                    height = y2 - y1

                                    boxes.append([x1, y1, width, height])
                                    confidences.append(float(boxes_data.conf[j]))
                                    class_ids.append(int(boxes_data.cls[j]))
                                except:
                                    continue

                    indices = list(range(len(boxes)))

                else:  # YOLOv5 models
                    model = torch.hub.load('ultralytics/yolov5', model_info['model'], pretrained=True)
                    model.conf = conf_threshold

                    if torch.cuda.is_available():
                        model = model.to('cuda')

                    # Run inference
                    start_time = time.time()
                    results = model(image)
                    inference_time = time.time() - start_time

                    # Process results
                    boxes = []
                    confidences = []
                    class_ids = []

                    # Extract detections
                    detections = results.xyxy[0].cpu().numpy()
                    for det in detections:
                        x1, y1, x2, y2, conf, cls_id = det
                        width = x2 - x1
                        height = y2 - y1

                        boxes.append([int(x1), int(y1), int(width), int(height)])
                        confidences.append(float(conf))
                        class_ids.append(int(cls_id))

                    indices = list(range(len(boxes)))

            else:
                # Skip unsupported models
                print(f"Skipping unsupported model family: {model_info['family']}")
                return None

            # Skip warmup run timing
            if i > 0:
                inference_times.append(inference_time)
                print(f"  Run {i}/{num_runs}: {inference_time:.4f}s")
            else:
                print(f"  Warmup run: {inference_time:.4f}s")

            # Save last iteration's detections
            if i == num_runs:
                all_detections = (boxes, confidences, class_ids, indices)

        except Exception as e:
            print(f"Error during run {i}: {e}")
            import traceback
            traceback.print_exc()
            if i == 0:  # If even the warmup fails, return None
                return None

    # Calculate statistics
    if not inference_times:
        print(f"No successful runs for {model_info['name']}")
        return None

    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    std_dev = np.std(inference_times)
    fps = 1.0 / avg_time

    print(f"Statistics for {model_info['name']}:")
    print(f"  Average: {avg_time:.4f}s ({fps:.2f} FPS)")
    print(f"  Min: {min_time:.4f}s, Max: {max_time:.4f}s")
    print(f"  Std Dev: {std_dev:.4f}s")
    print(f"  Detections: {len(all_detections[3])}")

    # Draw detections on image
    result_image = draw_detections(image, *all_detections, labels, colors)

    # Add performance info to image
    cv2.putText(result_image, f"{model_info['name']} - {backend}", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_image, f"FPS: {fps:.2f}", (10, 70),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_image, f"Detections: {len(all_detections[3])}", (10, 110),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Return results
    return {
        'name': model_info['name'],
        'inference_time': avg_time,
        'fps': fps,
        'detections': len(all_detections[3]),
        'min_time': min_time,
        'max_time': max_time,
        'std_dev': std_dev,
        'backend': backend,
        'image': result_image,
        'boxes': all_detections[0],
        'confidences': all_detections[1],
        'class_ids': all_detections[2],
        'indices': all_detections[3]
    }

def benchmark_yolo_models(image_path, output_dir="benchmark_results", conf_threshold=0.5, num_runs=5, selected_models=['all']):
    """
    Benchmark various YOLO models on an image and generate comprehensive performance metrics.
    """
    print(f"Benchmarking YOLO models on image: {image_path}")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Ensure model files are available
    ensure_model_files()

    # Verify image existence
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Get image dimensions
    height, width = img.shape[:2]

    # Load labels
    labels_path = '../yolov3-coco/coco-labels'
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return

    labels = open(labels_path).read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Define models to test - simplified list for compatibility
    models = [
        # Original darknet models
        {
            'name': 'YOLOv3',
            'family': 'darknet',
            'cfg': '../yolov3-coco/yolov3.cfg',
            'weights': '../yolov3-coco/yolov3.weights',
            'color': '#3498db',  # Blue
            'input_size': (416, 416)
        },
        {
            'name': 'YOLOv4',
            'family': 'darknet',
            'cfg': '../yolov4-coco/yolov4.cfg',
            'weights': '../yolov4-coco/yolov4.weights',
            'color': '#e74c3c',  # Red
            'input_size': (416, 416)
        },
        {
            'name': 'YOLOv3-tiny',
            'family': 'darknet',
            'cfg': '../yolov3-coco/yolov3-tiny.cfg',
            'weights': '../yolov3-coco/yolov3-tiny.weights',
            'color': '#2ecc71',  # Green
            'input_size': (416, 416)
        },
        # Newer models with GPU support
        {
            'name': 'YOLOv5s',
            'family': 'ultralytics',
            'model': 'yolov5s',
            'color': '#9b59b6',  # Purple
            'input_size': (640, 640)
        },
        {
            'name': 'YOLOv5n',
            'family': 'ultralytics',
            'model': 'yolov5n',
            'color': '#f39c12',  # Orange
            'input_size': (640, 640)
        },
        # YOLOv8 models - use specific constructor for compatibility
        {
            'name': 'YOLOv8n',
            'family': 'ultralytics',
            'model': 'yolov8n.pt',
            'color': '#2980b9',  # Dark blue
            'input_size': (640, 640),
            'version': 8  # Add version flag to handle differently
        },
        {
            'name': 'YOLOv8s',
            'family': 'ultralytics',
            'model': 'yolov8s.pt',
            'color': '#8e44ad',  # Dark purple
            'input_size': (640, 640),
            'version': 8  # Add version flag to handle differently
        }
    ]

    # Filter models based on selection
    if 'all' not in selected_models:
        models = [m for m in models if m['name'] in selected_models]

    # Results storage
    results = {}
    result_images = {}

    # Warm up GPU with a dummy tensor operation
    if torch.cuda.is_available():
        print("Warming up GPU...")
        dummy = torch.ones(1, 3, 640, 640).cuda()
        for _ in range(10):  # Run 10 iterations to warm up
            _ = dummy * 2
        torch.cuda.synchronize()

    # Run benchmarks on each model
    for model_info in models:
        try:
            print(f"\nTesting model: {model_info['name']}")

            # Check if model files exist for darknet models
            if model_info['family'] == 'darknet':
                if not os.path.exists(model_info['cfg']) or not os.path.exists(model_info['weights']):
                    print(f"Error: Files for model {model_info['name']} not found")
                    continue

            # Run the model with multiple iterations for reliable timing
            model_result = run_multiple_iterations(
                model_info, img, model_info.get('input_size', (416, 416)),
                labels, colors, num_runs, conf_threshold
            )

            # Store results if successful
            if model_result:
                results[model_info['name']] = model_result
                result_images[model_info['name']] = model_result['image']

        except Exception as e:
            print(f"Error testing {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save individual result images
    for model_name, img in result_images.items():
        model_name_safe = model_name.replace('-', '_').lower()
        cv2.imwrite(os.path.join(output_dir, f"{model_name_safe}_result.png"), img)

    # Create comparison visual
    if result_images:
        # Combine images side by side
        img_height = None
        total_width = 0

        # Calculate dimensions
        for model_name, img in result_images.items():
            h, w = img.shape[:2]
            if img_height is None:
                img_height = h
            total_width += w

        # Create combined image
        comparison_img = np.zeros((img_height, total_width, 3), dtype=np.uint8)

        # Place images side by side
        x_offset = 0
        for model_name, img in result_images.items():
            h, w = img.shape[:2]
            comparison_img[0:h, x_offset:x_offset+w] = img
            x_offset += w

        # Save the comparison
        cv2.imwrite(os.path.join(output_dir, "yolo_comparison.png"), comparison_img)

    # Generate performance graphs
    generate_performance_graphs(results, output_dir)

    # Save results as JSON for later analysis
    serializable_results = {k: {kk: vv for kk, vv in v.items() if kk != 'image'} for k, v in results.items()}
    for model, data in serializable_results.items():
        for key in ['boxes', 'confidences', 'class_ids', 'indices']:
            if key in data:
                serializable_results[model][key] = len(data[key])

    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)

    return results

def generate_performance_graphs(results, output_dir):
    """
    Generate graphs comparing the performance of YOLO models.
    Uses the design pattern from the old benchmark for consistency.
    """
    # Extract data
    models = list(results.keys())
    inference_times = [results[model]['inference_time'] for model in models]
    fps_values = [results[model]['fps'] for model in models]
    detection_counts = [results[model]['detections'] for model in models]
    backends = [results[model]['backend'] for model in models]

    # Colors for each model
    model_colors = {
        'YOLOv3': '#3498db',     # Blue
        'YOLOv4': '#e74c3c',     # Red
        'YOLOv3-tiny': '#2ecc71', # Green
        'YOLOv5s': '#9b59b6',    # Purple
        'YOLOv5n': '#f39c12',    # Orange
        'YOLOv8n': '#2980b9',    # Dark blue
        'YOLOv8s': '#8e44ad'     # Dark purple
    }
    colors = [model_colors.get(model, '#95a5a6') for model in models]

    # Precision estimates (based on benchmarks on COCO)
    # mAP (@0.5 IoU) values - for informational purposes
    accuracy = {
        'YOLOv3': 55.3,
        'YOLOv4': 62.8,
        'YOLOv3-tiny': 33.1,
        'YOLOv5s': 56.8,
        'YOLOv5n': 45.7,
        'YOLOv8n': 52.4,
        'YOLOv8s': 60.5
    }

    # Figure 1: Processing Time vs FPS
    plt.figure(figsize=(14, 10))

    # Graph for inference time
    plt.subplot(2, 1, 1)
    bars = plt.bar(models, inference_times, color=colors, alpha=0.8)
    plt.title('Processing Time per Frame', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on bars
    for bar, time_val in zip(bars, inference_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.4f}s', ha='center', fontweight='bold')

    # Add backend info
    for i, backend in enumerate(backends):
        plt.text(i, 0.01, backend, ha='center', color='white',
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

    # Graph for FPS
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, fps_values, color=colors, alpha=0.8)
    plt.title('Frames per Second (FPS)', fontsize=14, fontweight='bold')
    plt.ylabel('FPS', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on bars
    for bar, fps in zip(bars, fps_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{fps:.2f} FPS', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('YOLO Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'yolo_performance_time_fps.png'), format='png', dpi=300, bbox_inches='tight')

    # Figure 2: Accuracy vs Speed vs Detections
    plt.figure(figsize=(14, 10))

    # Subplot for FPS vs mAP
    plt.subplot(2, 1, 1)
    for model in models:
        plt.scatter(accuracy.get(model, 0), results[model]['fps'],
                  s=100, color=model_colors.get(model, '#95a5a6'), label=model, alpha=0.8)
        plt.annotate(model,
                   (accuracy.get(model, 0), results[model]['fps']),
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    plt.title('Precision vs Speed', fontsize=14, fontweight='bold')
    plt.xlabel('Precision (mAP@0.5)', fontsize=12)
    plt.ylabel('Speed (FPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Subplot for detection count
    plt.subplot(2, 1, 2)
    bars = plt.bar(models, detection_counts, color=colors, alpha=0.8)
    plt.title('Objects Detected', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on bars
    for bar, count in zip(bars, detection_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('YOLO Model Effectiveness Analysis', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'yolo_performance_accuracy.png'), format='png', dpi=300, bbox_inches='tight')

    # Create a timing heatmap
    plt.figure(figsize=(10, 6))
    data = np.array([inference_times])
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Inference Time (seconds)')
    plt.xticks(range(len(models)), models, rotation=45)
    plt.yticks([])
    plt.title('Inference Time Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_heatmap.png'), dpi=300)

    # Create a comparison summary table as CSV
    with open(os.path.join(output_dir, 'model_comparison.csv'), 'w') as f:
        f.write("Model,Inference Time (s),FPS,Detections,Min Time (s),Max Time (s),Std Dev,Backend\n")
        for model in models:
            r = results[model]
            f.write(f"{model},{r['inference_time']:.4f},{r['fps']:.2f},{r['detections']},{r['min_time']:.4f},{r['max_time']:.4f},{r['std_dev']:.4f},{r['backend']}\n")

    # Print summary
    print("\nBenchmark Summary:")
    print(f"{'Model':<12} {'Time (s)':<10} {'FPS':<8} {'Detections':<10} {'Backend':<6}")
    print("-" * 50)
    for model in models:
        r = results[model]
        print(f"{model:<12} {r['inference_time']:<10.4f} {r['fps']:<8.2f} {r['detections']:<10} {r['backend']:<6}")

    print(f"\nDetailed results and graphs saved to {output_dir}/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark YOLO models")
    parser.add_argument("-i", "--image", default="../_out/episode_3360/frame_camera/000970.png",
                      help="Path to test image")
    parser.add_argument("-o", "--output", default="benchmark_results",
                      help="Output directory for results")
    parser.add_argument("-c", "--confidence", type=float, default=0.3,  # Changed from 0.5
                      help="Confidence threshold for detection")
    parser.add_argument("-r", "--runs", type=int, default=5,
                      help="Number of timing runs for each model")
    parser.add_argument("--models", nargs='+',
                  choices=['YOLOv3', 'YOLOv4', 'YOLOv3-tiny', 'YOLOv5s', 'YOLOv5n', 'YOLOv8n', 'YOLOv8s', 'all'],
                  default=['all'],
                  help="Specific models to benchmark")

    args = parser.parse_args()

    benchmark_yolo_models(args.image, args.output, args.confidence, args.runs, args.models)

import cv2
import numpy as np
import time
import argparse
from yolo_utils import infer_image, show_image

def compare_yolo_models(image_path):
    """Compare YOLOv3 and YOLOv4 performance on the same image."""

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    height, width = img.shape[:2]

    # Prepare labels and colors (same for both models)
    labels = open('./yolov3-coco/coco-labels').read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load YOLOv3 model
    print("Loading YOLOv3 model...")
    net_v3 = cv2.dnn.readNetFromDarknet('./yolov3-coco/yolov3.cfg',
                                      './yolov3-coco/yolov3.weights')

    # Set preferred backend (uncomment if you have GPU support)
    try:
        net_v3.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net_v3.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA backend for YOLOv3")
    except:
        print("Using CPU backend for YOLOv3")

    layer_names_v3 = net_v3.getLayerNames()
    layer_names_v3 = [layer_names_v3[i-1] for i in net_v3.getUnconnectedOutLayers()]

    # Load YOLOv4 model
    print("Loading YOLOv4 model...")
    try:
        net_v4 = cv2.dnn.readNetFromDarknet('./yolov4-coco/yolov4.cfg',
                                          './yolov4-coco/yolov4.weights')

        # Set preferred backend (uncomment if you have GPU support)
        try:
            net_v4.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net_v4.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using CUDA backend for YOLOv4")
        except:
            print("Using CPU backend for YOLOv4")

        layer_names_v4 = net_v4.getLayerNames()
        layer_names_v4 = [layer_names_v4[i-1] for i in net_v4.getUnconnectedOutLayers()]
    except Exception as e:
        print(f"Error loading YOLOv4: {e}")
        print("Skipping YOLOv4 comparison.")
        net_v4 = None

    # Test YOLOv3
    start_time = time.time()
    img_v3, boxes_v3, confidences_v3, classids_v3, idxs_v3 = infer_image(
        net_v3, layer_names_v3, height, width, img.copy(), colors, labels)
    v3_time = time.time() - start_time

    print(f"YOLOv3 processing time: {v3_time:.4f} seconds")
    print(f"YOLOv3 detected {len(idxs_v3) if isinstance(idxs_v3, np.ndarray) else 0} objects")

    # Test YOLOv4 if available
    if net_v4 is not None:
        start_time = time.time()
        img_v4, boxes_v4, confidences_v4, classids_v4, idxs_v4 = infer_image(
            net_v4, layer_names_v4, height, width, img.copy(), colors, labels)
        v4_time = time.time() - start_time

        print(f"YOLOv4 processing time: {v4_time:.4f} seconds")
        print(f"YOLOv4 detected {len(idxs_v4) if isinstance(idxs_v4, np.ndarray) else 0} objects")

        # Compare speeds
        speed_improvement = (v3_time - v4_time) / v3_time * 100
        if speed_improvement > 0:
            print(f"YOLOv4 is {speed_improvement:.2f}% faster than YOLOv3")
        else:
            print(f"YOLOv3 is {-speed_improvement:.2f}% faster than YOLOv4")

        # Display both images side by side
        combined = np.hstack((img_v3, img_v4))

        # Add text labels
        cv2.putText(combined, "YOLOv3", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "YOLOv4", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv3 vs YOLOv4", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow("YOLOv3", img_v3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare YOLOv3 and YOLOv4")
    parser.add_argument("image", help="Path to the test image")
    args = parser.parse_args()

    compare_yolo_models(args.image)

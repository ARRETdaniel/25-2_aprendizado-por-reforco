# Author: Daniel Terra Gomes
# Date: Jun 30, 2025

import threading
from queue import Queue, Empty
import numpy as np
import time
import cv2

class ThreadedDetector:
    """
    A threaded wrapper for object detection that processes frames
    in a background thread to improve performance and responsiveness.
    """
    def __init__(self, detector, max_queue_size=5):
        """Initialize a threaded detector that processes frames in the background

        Args:
            detector: The actual detector instance (DetectionClient)
            max_queue_size: Maximum number of frames to queue for processing
        """
        self.detector = detector
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        self.running = True
        self.latest_result = None
        self.latest_frame = None
        self.latest_metrics = None

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.worker_thread.start()

        # For FPS calculation
        self.process_times = []
        self.max_times = 30  # Average over last 30 frames

        # Frame skipping control
        self.frame_counter = 0
        self.skip_rate = 0  # Adaptive based on load

        print("ThreadedDetector initialized")

    def _process_frames(self):
        """Worker thread that processes frames from the queue"""
        while self.running:
            try:
                # Get a frame from the queue with a timeout
                frame = self.frame_queue.get(timeout=0.1)

                # Process the frame with better error handling
                try:
                    start_time = time.time()
                    processed_frame, boxes, confidences, classids, idxs = self.detector.detect_objects(frame)
                    process_time = time.time() - start_time

                    # Calculate and track FPS
                    self.process_times.append(process_time)
                    if len(self.process_times) > self.max_times:
                        self.process_times.pop(0)

                    # Put the result in the result queue
                    result = {
                        'frame': processed_frame,
                        'boxes': boxes,
                        'confidences': confidences,
                        'classids': classids,
                        'idxs': idxs,
                        'process_time': process_time
                    }
                    self.result_queue.put(result)
                except Exception as e:
                    print(f"Error processing frame in thread: {e}")
                    # Put a partial result with the original frame to avoid blocking
                    result = {
                        'frame': frame,
                        'boxes': [],
                        'confidences': [],
                        'classids': [],
                        'idxs': [],
                        'process_time': 0.0,
                        'error': str(e)
                    }
                    self.result_queue.put(result)

                # Mark task as done regardless of success/failure
                self.frame_queue.task_done()

                # Update skip rate based on processing time
                # If processing is taking too long, increase skip rate
                avg_time = sum(self.process_times) / max(1, len(self.process_times))
                if avg_time > 0.1:  # > 100ms per frame
                    self.skip_rate = min(3, self.skip_rate + 1)  # Max skip 3 frames
                else:
                    self.skip_rate = max(0, self.skip_rate - 1)  # Min skip 0 frames

            except Empty:
                # No frames to process, just continue
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")
                # Put a None result to indicate an error
                self.result_queue.put(None)
                self.frame_queue.task_done()

    def process_frame(self, frame, metrics=None):
        """Add a frame to the processing queue

        Args:
            frame: The frame to process
            metrics: Optional metrics object to record performance

        Returns:
            Tuple of (processed_frame, boxes, confidences, classids, idxs)
            If no results are available yet, returns the input frame and empty lists
        """
        self.frame_counter += 1

        # Skip frames based on adaptive skip rate
        if self.frame_counter % (self.skip_rate + 1) != 0:
            # If we're skipping this frame but have a previous result, return that
            if self.latest_result is not None:
                return (
                    self.latest_frame,
                    self.latest_result['boxes'],
                    self.latest_result['confidences'],
                    self.latest_result['classids'],
                    self.latest_result['idxs']
                )
            else:
                return frame, [], [], [], []

        # If the queue is full, skip this frame to avoid lagging behind
        if self.frame_queue.full():
            print("Warning: Detection queue full, skipping frame")
            return frame, [], [], [], []

        # Add the frame to the queue
        try:
            self.frame_queue.put(frame, block=False)
        except:
            # Queue is full
            return frame, [], [], [], []

        # Try to get a result, but don't block if none is available
        try:
            result = self.result_queue.get(block=False)
            if result is not None:
                self.latest_result = result
                self.latest_frame = result['frame']

                # Record metrics if provided
                if metrics is not None:
                    metrics.record_detection_metrics(
                        result['process_time'],
                        result['boxes'],
                        result['confidences'],
                        result['classids'],
                        result['idxs']
                    )

                    # Record risk level
                    risk_level = self.detector._calculate_traffic_risk(
                        result['boxes'],
                        result['classids'],
                        result['confidences']
                    )
                    metrics.record_risk_level(risk_level)

                    # Process warning data
                    self._process_warning_data(
                        result['boxes'],
                        result['confidences'],
                        result['classids'],
                        result['idxs'],
                        metrics
                    )

            self.result_queue.task_done()
        except Empty:
            # No result available yet, use the latest if we have one
            pass

        if self.latest_result is None:
            return frame, [], [], [], []
        else:
            return (
                self.latest_frame,
                self.latest_result['boxes'],
                self.latest_result['confidences'],
                self.latest_result['classids'],
                self.latest_result['idxs']
            )

    def _process_warning_data(self, boxes, confidences, classids, idxs, metrics):
        """Process detection results into warning data for metrics"""
        # Initialize warning data structure
        warning_data = {
            'count': 0,
            'types': {},
            'severities': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        }

        if idxs is not None and len(idxs) > 0:
            # Count warnings by type
            for i in idxs:
                class_id = classids[i]
                # Validate confidence before using it
                confidence = confidences[i] if i < len(confidences) else 0

                # Get class name, handling potential index errors
                if class_id < len(self.detector.labels):
                    class_name = self.detector.labels[class_id]
                else:
                    class_name = f"unknown-{class_id}"

                # Skip benches (class 13) to avoid misidentification as traffic signs
                if class_id == 13:  # Bench class
                    continue

                # Update warning counts for relevant classes
                if class_name in ['person', 'car', 'truck', 'bus', 'stop sign', 'traffic light']:
                    warning_data['count'] += 1

                    # Update by type
                    if class_name in warning_data['types']:
                        warning_data['types'][class_name] += 1
                    else:
                        warning_data['types'][class_name] = 1

                    # Assign severity based on class
                    if class_name in ['person', 'stop sign']:
                        warning_data['severities']['HIGH'] += 1
                    elif class_name in ['traffic light']:
                        warning_data['severities']['MEDIUM'] += 1
                    else:
                        warning_data['severities']['LOW'] += 1

        # Record the warning metrics
        metrics.record_warning_metrics(warning_data)

    def get_fps(self):
        """Get the current FPS based on recent processing times"""
        if not self.process_times:
            return 0
        avg_time = sum(self.process_times) / len(self.process_times)
        return 1.0 / avg_time if avg_time > 0 else 0

    def shutdown(self):
        """Stop the worker thread"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        print("ThreadedDetector shutdown complete")

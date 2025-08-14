import unittest
import os
import shutil
import numpy as np
import json
import sys
from datetime import datetime
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from performance_metrics import PerformanceMetrics

class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for the PerformanceMetrics class."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.test_output_dir = "test_metrics_output"
        # Create a fresh test directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        os.makedirs(self.test_output_dir)

        # Initialize the metrics object
        self.metrics = PerformanceMetrics(self.test_output_dir)

        # Set initialization time for consistent testing
        self.metrics.init_time = datetime.now()

        # Mock some weather conditions for testing
        self.metrics.weather_conditions = {
            1: "Clear",
            2: "Cloudy",
            3: "Rain"
        }

        # Initialize weather performance data
        self.metrics.weather_performance = {}

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up test directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_initialization(self):
        """Test proper initialization of the PerformanceMetrics class."""
        self.assertEqual(self.metrics.output_dir, self.test_output_dir)
        self.assertIsInstance(self.metrics.detection_times, list)
        self.assertIsInstance(self.metrics.confidence_scores, list)
        self.assertTrue(os.path.exists(self.test_output_dir))

    def test_update_weather_condition(self):
        """Test updating weather conditions."""
        # Update to a valid weather condition
        self.metrics.update_weather_condition(1)

        # Check if weather was updated correctly
        self.assertEqual(self.metrics.current_weather_id, 1)
        self.assertIn(1, self.metrics.weather_performance)

        # Check weather performance metrics initialization
        self.assertEqual(self.metrics.weather_performance[1]['detections'], 0)
        self.assertEqual(self.metrics.weather_performance[1]['true_positives'], 0)
        self.assertEqual(self.metrics.weather_performance[1]['false_positives'], 0)
        self.assertEqual(self.metrics.weather_performance[1]['confidence_scores'], [])
        self.assertEqual(self.metrics.weather_performance[1]['detection_times'], [])

        # Test with invalid weather ID
        self.metrics.update_weather_condition(999)
        # Should not change the current weather
        self.assertEqual(self.metrics.current_weather_id, 1)

    def test_record_sign_detection(self):
        """Test recording traffic sign detections."""
        timestamp = 10.5
        classids = [9, 11, 2, 5]  # Traffic sign classes: 9, 11
        confidences = [0.8, 0.9, 0.7, 0.6]
        boxes = [(10, 20, 30, 40), (50, 60, 70, 80), (90, 100, 110, 120), (130, 140, 150, 160)]
        idxs = [0, 1, 2, 3]

        # Record the detections
        self.metrics.record_sign_detection(timestamp, classids, confidences, boxes, idxs)

        # Should have recorded 2 sign detections (classes 9 and 11)
        self.assertEqual(len(self.metrics.sign_detections), 2)

        # Check first detection details
        first_detection = self.metrics.sign_detections[0]
        self.assertEqual(first_detection['timestamp'], timestamp)
        self.assertEqual(first_detection['class_id'], 9)
        self.assertEqual(first_detection['confidence'], 0.8)
        self.assertEqual(first_detection['box'], (10, 20, 30, 40))
        self.assertFalse(first_detection['processed'])

        # Check class-specific confidence tracking
        self.assertIn(9, self.metrics.class_confidence_by_id)
        self.assertIn(11, self.metrics.class_confidence_by_id)
        self.assertEqual(self.metrics.class_confidence_by_id[9][0], 0.8)
        self.assertEqual(self.metrics.class_confidence_by_id[11][0], 0.9)

    def test_record_vehicle_response(self):
        """Test recording vehicle responses to detections."""
        # First, add some sign detections
        self.metrics.sign_detections = [
            {
                'id': 'sign_11_0',
                'timestamp': 10.0,
                'class_id': 11,
                'confidence': 0.9,
                'box': (10, 20, 30, 40),
                'processed': False
            }
        ]

        # Now record a vehicle response shortly after
        timestamp = 10.5
        throttle = 0.0
        brake = 0.8
        steer = 0.1
        vehicle_speed = 15.0

        self.metrics.record_vehicle_response(timestamp, throttle, brake, steer, vehicle_speed)

        # Check if response was recorded
        self.assertEqual(len(self.metrics.vehicle_responses), 1)

        # Check if detection was matched with response
        self.assertEqual(len(self.metrics.response_times), 1)
        self.assertEqual(self.metrics.response_times[0]['sign_id'], 'sign_11_0')
        self.assertEqual(self.metrics.response_times[0]['response_delay'], 0.5)  # 10.5 - 10.0
        self.assertEqual(self.metrics.response_times[0]['response_type'], 'braking')

        # Check if sign detection was marked as processed
        self.assertTrue(self.metrics.sign_detections[0]['processed'])

    def test_generate_summary(self):
        """Test generating performance summary."""
        # Add some test data
        self.metrics.detection_times = [0.05, 0.06, 0.04, 0.07, 0.05]
        self.metrics.confidence_scores = [0.8, 0.9, 0.85, 0.75, 0.95]
        self.metrics.class_true_positives = {9: 8, 11: 5, 2: 10}
        self.metrics.class_false_positives = {9: 2, 11: 1, 2: 3}

        # Generate summary
        summary = self.metrics.generate_summary()

        # Check key metrics in summary
        self.assertIn('avg_detection_time', summary)
        self.assertIn('avg_confidence', summary)
        self.assertIn('total_detections', summary)

        # Verify calculations
        self.assertAlmostEqual(summary['avg_detection_time'], 0.054, places=3)
        self.assertAlmostEqual(summary['avg_confidence'], 0.85, places=2)
        self.assertEqual(summary['total_detections'], 5)

        # Check if summary file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'summary_stats.csv')))

    def test_visualize_metrics(self):
        """Test metrics visualization."""
        # Add some test data
        self.metrics.detection_times = [0.05, 0.06, 0.04, 0.07, 0.05]
        self.metrics.confidence_scores = [0.8, 0.9, 0.85, 0.75, 0.95]
        self.metrics.timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Add risk level data
        self.metrics.risk_level_values = ['LOW', 'MEDIUM', 'HIGH', 'LOW', 'LOW']

        # Generate visualizations
        self.metrics.visualize_metrics()

        # Check if visualization files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'performance_metrics.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'detection_times_histogram.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'risk_level_distribution.png')))

    def test_analyze_safety_distance_improvement(self):
        """Test safety distance improvement analysis."""
        # Add some test data
        self.metrics.detection_times = [0.05, 0.06, 0.04, 0.07, 0.05]

        # Run the analysis
        result = self.metrics.analyze_safety_distance_improvement()

        # Check if result contains expected keys
        self.assertIn('speeds', result)
        self.assertIn('human_distances', result)
        self.assertIn('system_distances', result)
        self.assertIn('distance_saved', result)

        # Check if output file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'safety_distance_analysis.png')))

    def test_safe_json_serialization(self):
        """Test that NumPy types can be serialized to JSON."""
        # Create a dictionary with NumPy types
        test_data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_bool': np.bool_(True),
            'numpy_array': np.array([1, 2, 3])
        }

        # Test if the NumpyEncoder class can handle these types
        try:
            # Use the NumpyEncoder from the generate_summary method
            # This requires that the NumpyEncoder is defined in the class
            json_str = json.dumps(test_data, cls=self.metrics.NumpyEncoder)

            # Parse the JSON string back to verify
            parsed_data = json.loads(json_str)

            # Check the values
            self.assertEqual(parsed_data['numpy_int'], 42)
            self.assertEqual(parsed_data['numpy_float'], 3.14)
            self.assertEqual(parsed_data['numpy_bool'], True)
            self.assertEqual(parsed_data['numpy_array'], [1, 2, 3])
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")


if __name__ == '__main__':
    unittest.main()

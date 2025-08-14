import unittest
import os
import shutil
import sys
import numpy as np
from datetime import datetime
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from performance_metrics import PerformanceMetrics
from results_reporter import ResultsReporter

class TestIntegration(unittest.TestCase):
    """Test the integration between PerformanceMetrics and ResultsReporter."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.test_output_dir = "test_integration_output"
        self.test_report_dir = "test_integration_report"

        # Create fresh test directories
        for dir_path in [self.test_output_dir, self.test_report_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        # Initialize the metrics object
        self.metrics = PerformanceMetrics(self.test_output_dir)

        # Set initialization time for consistent testing
        self.metrics.init_time = datetime.now()

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up test directories
        for dir_path in [self.test_output_dir, self.test_report_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    def test_end_to_end_workflow(self):
        """Test the end-to-end workflow from metrics collection to report generation."""
        # 1. Simulate data collection
        # Add detection times
        self.metrics.detection_times = [0.05, 0.06, 0.04, 0.07, 0.05]
        self.metrics.confidence_scores = [0.8, 0.9, 0.85, 0.75, 0.95]
        self.metrics.timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Add this line to match the size of timestamps:
        self.metrics.detection_counts = [1, 2, 2, 1, 3]

        # Add class-specific data
        self.metrics.class_true_positives = {2: 40, 9: 15, 11: 10}
        self.metrics.class_false_positives = {2: 5, 9: 2, 11: 1}

        # Add class confidence data
        self.metrics.class_confidence_by_id = {
            2: [0.85, 0.88, 0.90],  # Car
            9: [0.82, 0.79, 0.83],  # Traffic light
            11: [0.92, 0.94, 0.91]  # Stop sign
        }

        # Add risk level data
        self.metrics.risk_level_values = ['LOW', 'MEDIUM', 'HIGH', 'LOW', 'LOW']

        # 2. Generate metrics summary and visualizations
        self.metrics.generate_summary()
        self.metrics.visualize_metrics()

        # Verify summary and visualization files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'summary_stats.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'performance_metrics.png')))

        # 3. Generate the report using metrics data
        reporter = ResultsReporter(
            metrics_dir=self.test_output_dir,
            report_dir=self.test_report_dir
        )
        reporter.generate_report()

        # Verify report was created
        report_path = os.path.join(self.test_report_dir, 'relatorio_desempenho.html')
        self.assertTrue(os.path.exists(report_path))

        # Check report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()

        # Verify report contains key metrics from our test data
        self.assertIn('0.054', report_content)  # Average detection time
        self.assertIn('0.85', report_content)   # Average confidence

        # Verify report has all major sections
        major_sections = [
            'Resumo de Desempenho',
            'Visualizações de Desempenho',
            'Precisão de Detecção por Classe',
            'Validação de Hipótese',
            'Discussão'
        ]

        for section in major_sections:
            self.assertIn(section, report_content)


if __name__ == '__main__':


    unittest.main()

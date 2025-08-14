import unittest
import os
import shutil
import json
import csv
import sys

# Add the parent directory to sys.path to make FinalProject importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from results_reporter import ResultsReporter

class TestResultsReporter(unittest.TestCase):
    """Test cases for the ResultsReporter class."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.test_metrics_dir = "test_metrics_dir"
        self.test_report_dir = "test_report_dir"

        # Create fresh test directories
        for dir_path in [self.test_metrics_dir, self.test_report_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        # Create mock metrics data files
        self._create_mock_metrics_files()

        # Initialize the reporter
        self.reporter = ResultsReporter(
            metrics_dir=self.test_metrics_dir,
            report_dir=self.test_report_dir
        )

    def _create_mock_metrics_files(self):
        """Create mock metrics files for testing."""
        # Create summary_stats.csv
        with open(os.path.join(self.test_metrics_dir, 'summary_stats.csv'), 'w', encoding='utf-8') as f:
            f.write("avg_detection_time,0.054\n")
            f.write("avg_fps,18.5\n")
            f.write("total_detections,1250\n")
            f.write("avg_confidence,0.85\n")
            f.write("total_warnings,45\n")
            f.write("total_runtime_seconds,300.5\n")
            f.write("class_precision,{\"2\": 0.92, \"9\": 0.88, \"11\": 0.95}\n")
            f.write("class_avg_confidence,{\"2\": 0.87, \"9\": 0.82, \"11\": 0.91}\n")
            f.write("distance_detection_rates,{\"próximo\": 0.95, \"médio\": 0.85, \"distante\": 0.65}\n")

        # Create detection_metrics.csv
        with open(os.path.join(self.test_metrics_dir, 'detection_metrics.csv'), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'class_id', 'confidence', 'detection_time'])
            writer.writerow([10.5, 2, 0.89, 0.05])
            writer.writerow([11.2, 9, 0.82, 0.06])
            writer.writerow([12.1, 11, 0.94, 0.04])

        # Create warning_metrics.csv
        with open(os.path.join(self.test_metrics_dir, 'warning_metrics.csv'), 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'warning_type', 'severity', 'object_id'])
            writer.writerow([10.6, 'vehicle', 'HIGH', 'v_001'])
            writer.writerow([11.3, 'sign', 'MEDIUM', 's_001'])

        # Create necessary images
        for img_name in ['performance_metrics.png', 'detection_times_histogram.png']:
            with open(os.path.join(self.test_metrics_dir, img_name), 'w') as f:
                f.write("Mock image data")

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up test directories
        for dir_path in [self.test_metrics_dir, self.test_report_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    def test_initialization(self):
        """Test proper initialization of the ResultsReporter class."""
        self.assertEqual(self.reporter.metrics_dir, self.test_metrics_dir)
        self.assertEqual(self.reporter.report_dir, self.test_report_dir)
        self.assertTrue(os.path.exists(self.test_report_dir))
        self.assertTrue(os.path.exists(os.path.join(self.test_report_dir, 'style.css')))

    def test_load_metrics_data(self):
        """Test loading metrics data from files."""
        data = self.reporter._load_metrics_data()

        # Check if data contains expected sections
        self.assertIn('summary', data)
        self.assertIn('detections', data)
        self.assertIn('warnings', data)

        # Check summary data
        self.assertEqual(data['summary']['avg_detection_time'], 0.054)
        self.assertEqual(data['summary']['avg_fps'], 18.5)
        self.assertEqual(data['summary']['total_detections'], 1250)

        # Check class precision data
        class_precision = data['summary']['class_precision']
        if isinstance(class_precision, str):
            class_precision = json.loads(class_precision)
        self.assertEqual(class_precision['2'], 0.92)
        self.assertEqual(class_precision['11'], 0.95)

        # Check detections data
        self.assertEqual(len(data['detections']), 3)
        self.assertEqual(float(data['detections'][0]['confidence']), 0.89)

        # Check warnings data
        self.assertEqual(len(data['warnings']), 2)
        self.assertEqual(data['warnings'][0]['severity'], 'HIGH')

    def test_generate_summary_html(self):
        """Test generating summary HTML section."""
        data = self.reporter._load_metrics_data()
        html = self.reporter._generate_summary_html(data['summary'])

        # Check if HTML contains key metrics
        self.assertIn('Tempo Médio de Detecção', html)
        self.assertIn('FPS Médio', html)
        self.assertIn('Total de Detecções', html)
        self.assertIn('Confiança Média', html)

        # Check if values are included
        self.assertIn('0.054', html)
        self.assertIn('18.5', html)
        self.assertIn('1250', html)
        self.assertIn('0.85', html)

    def test_generate_hypothesis_validation_html(self):
        """Test generating hypothesis validation HTML section."""
        data = self.reporter._load_metrics_data()
        html = self.reporter._generate_hypothesis_validation_html(data)

        # Check if HTML contains validation criteria
        self.assertIn('Critérios de Validação', html)
        self.assertIn('Processamento em tempo real', html)
        self.assertIn('Confiança de detecção', html)
        self.assertIn('Detecção de placas de trânsito', html)

        # Check if conclusion is included
        self.assertIn('Conclusão', html)

    def test_generate_class_accuracy_html(self):
        """Test generating class accuracy HTML section."""
        data = self.reporter._load_metrics_data()
        html = self.reporter._generate_class_accuracy_html(data)

        # Check if HTML contains expected elements
        self.assertIn('Precisão de Detecção por Classe', html)
        self.assertIn('Classe', html)
        self.assertIn('Precisão', html)
        self.assertIn('Confiança Média', html)

        # Check if class data is included
        self.assertIn('Carro', html)
        self.assertIn('Semáforo', html)
        self.assertIn('Placa de pare', html)

    def test_generate_distance_metrics_html(self):
        """Test generating distance metrics HTML section."""
        data = self.reporter._load_metrics_data()
        html = self.reporter._generate_distance_metrics_html(data)

        # Check if HTML contains expected elements
        self.assertIn('Taxas de Detecção por Distância', html)
        self.assertIn('Faixa de Distância', html)
        self.assertIn('Taxa de Detecção', html)

        # Check if distance bands are included
        self.assertIn('Próximo', html)
        self.assertIn('Médio', html)
        self.assertIn('Distante', html)

    def test_generate_report(self):
        """Test generating the complete HTML report."""
        self.reporter.generate_report()

        # Check if report file was created
        report_path = os.path.join(self.test_report_dir, 'relatorio_desempenho.html')
        self.assertTrue(os.path.exists(report_path))

        # Read the report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()

        # Check if report contains all major sections
        self.assertIn('<title>Relatório de Desempenho de Detecção de Objetos</title>', report_content)
        self.assertIn('<h2>Resumo de Desempenho</h2>', report_content)
        self.assertIn('<h2>Visualizações de Desempenho</h2>', report_content)
        self.assertIn('<h2>Precisão de Detecção por Classe</h2>', report_content)
        self.assertIn('<h2>Taxas de Detecção por Distância</h2>', report_content)
        self.assertIn('<h2>Validação de Hipótese</h2>', report_content)
        self.assertIn('<h2>Discussão</h2>', report_content)


if __name__ == '__main__':
    unittest.main()

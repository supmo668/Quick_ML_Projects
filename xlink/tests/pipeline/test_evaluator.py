#!/usr/bin/env python3
"""
Unit tests for model evaluator module
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluate.evaluator import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = ModelEvaluator(results_dir=self.temp_dir)
        
        # Create simple model and scaler for testing
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        
        # Create test data
        np.random.seed(42)
        self.X_test = pd.DataFrame(
            np.random.randn(100, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        self.y_test = pd.Series(np.random.choice([0, 1], 100, p=[0.7, 0.3]))
        
        # Fit model and scaler
        X_train = pd.DataFrame(
            np.random.randn(200, 5),
            columns=[f'feature_{i}' for i in range(5)]
        )
        y_train = pd.Series(np.random.choice([0, 1], 200, p=[0.7, 0.3]))
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(self.evaluator.plots_dir.exists())
    
    def test_evaluate_model(self):
        """Test model evaluation metrics"""
        metrics = self.evaluator.evaluate_model(
            self.model, self.scaler, self.X_test, self.y_test, threshold=0.5
        )
        
        # Check all metrics present
        expected_metrics = [
            'roc_auc', 'accuracy', 'precision', 'recall', 'f1_score',
            'brier_score', 'pr_auc', 'confusion_matrix', 'threshold'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['roc_auc'], 0)
        self.assertLessEqual(metrics['roc_auc'], 1)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        
        # Check confusion matrix
        self.assertIn('true_negatives', metrics)
        self.assertIn('false_positives', metrics)
        self.assertIn('false_negatives', metrics)
        self.assertIn('true_positives', metrics)
        
        total = (metrics['true_negatives'] + metrics['false_positives'] + 
                metrics['false_negatives'] + metrics['true_positives'])
        self.assertEqual(total, len(self.y_test))
    
    def test_find_optimal_threshold(self):
        """Test optimal threshold finding"""
        threshold = self.evaluator.find_optimal_threshold(
            self.model, self.scaler, self.X_test, self.y_test, target_metric='f1'
        )
        
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLessEqual(threshold, 0.9)
        
        # Test with different metrics
        threshold_precision = self.evaluator.find_optimal_threshold(
            self.model, self.scaler, self.X_test, self.y_test, target_metric='precision'
        )
        
        threshold_recall = self.evaluator.find_optimal_threshold(
            self.model, self.scaler, self.X_test, self.y_test, target_metric='recall'
        )
        
        # Typically, threshold for high recall should be lower than for high precision
        self.assertLessEqual(threshold_recall, threshold_precision + 0.3)
    
    def test_segment_analysis(self):
        """Test segment analysis functionality"""
        segments = self.evaluator.segment_analysis(
            self.model, self.scaler, self.X_test, self.y_test
        )
        
        self.assertIn('confidence', segments)
        
        # Check confidence segments
        confidence_segments = segments['confidence']
        expected_segments = ['very_low', 'low', 'medium', 'high', 'very_high']
        
        for segment in expected_segments:
            self.assertIn(segment, confidence_segments)
            
            if 'count' in confidence_segments[segment]:
                self.assertGreaterEqual(confidence_segments[segment]['count'], 0)
                self.assertIn('actual_churn_rate', confidence_segments[segment])
                self.assertIn('predicted_churn_rate', confidence_segments[segment])
    
    def test_create_evaluation_plots(self):
        """Test plot creation"""
        self.evaluator.create_evaluation_plots(
            self.model, self.scaler, self.X_test, self.y_test
        )
        
        # Check plot files exist
        expected_plots = [
            'roc_curve.png',
            'pr_curve.png',
            'confusion_matrix.png',
            'prediction_distribution.png'
        ]
        
        for plot_name in expected_plots:
            plot_path = self.evaluator.plots_dir / plot_name
            self.assertTrue(plot_path.exists(), f"Plot {plot_name} not created")
    
    def test_save_results(self):
        """Test saving evaluation results"""
        metrics = {
            'roc_auc': 0.85,
            'accuracy': 0.78,
            'precision': 0.72,
            'recall': 0.68
        }
        
        version = self.evaluator.save_results(metrics, model_version='test_v1')
        
        self.assertIsNotNone(version)
        
        # Check files saved
        results_path = Path(self.temp_dir) / f"evaluation_{version}.json"
        latest_path = Path(self.temp_dir) / "evaluation_latest.json"
        
        self.assertTrue(results_path.exists())
        self.assertTrue(latest_path.exists())
        
        # Load and verify content
        with open(latest_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(loaded_results['metrics']['roc_auc'], 0.85)
        self.assertEqual(loaded_results['version'], version)
    
    def test_print_metrics(self, capsys):
        """Test metrics printing (requires pytest)"""
        metrics = {
            'roc_auc': 0.85,
            'pr_auc': 0.65,
            'accuracy': 0.78,
            'precision': 0.72,
            'recall': 0.68,
            'f1_score': 0.70,
            'brier_score': 0.15,
            'true_negatives': 60,
            'false_positives': 10,
            'false_negatives': 12,
            'true_positives': 18
        }
        
        self.evaluator.print_metrics(metrics)
        
        # Capture printed output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        self.evaluator.print_metrics(metrics)
        
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Check key values appear in output
        self.assertIn('0.85', output)  # ROC-AUC
        self.assertIn('0.78', output)  # Accuracy
        self.assertIn('TN: 60', output)  # Confusion matrix

@pytest.mark.parametrize("threshold,expected_range", [
    (0.3, (0.5, 0.9)),  # Lower threshold -> higher recall
    (0.5, (0.3, 0.7)),  # Default threshold
    (0.7, (0.1, 0.5)),  # Higher threshold -> lower recall
])
def test_threshold_effects(threshold, expected_range):
    """Test that different thresholds affect metrics appropriately"""
    evaluator = ModelEvaluator()
    
    # Create simple test data
    np.random.seed(42)
    model = LogisticRegression()
    scaler = StandardScaler()
    
    X_train = pd.DataFrame(np.random.randn(100, 3))
    y_train = pd.Series(np.random.choice([0, 1], 100))
    X_test = pd.DataFrame(np.random.randn(50, 3))
    y_test = pd.Series(np.random.choice([0, 1], 50))
    
    X_train_scaled = scaler.fit_transform(X_train)
    model.fit(X_train_scaled, y_train)
    
    metrics = evaluator.evaluate_model(model, scaler, X_test, y_test, threshold=threshold)
    
    # Recall should be in expected range
    assert expected_range[0] <= metrics['recall'] <= expected_range[1]

if __name__ == "__main__":
    unittest.main()

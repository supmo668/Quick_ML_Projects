#!/usr/bin/env python3
"""
Unit tests for model trainer module
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import pickle

from train.trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = ModelTrainer(model_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test trainer initialization"""
        self.assertIsNotNone(self.trainer)
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(self.trainer.splits_dir.exists())
    
    def test_create_splits(self, processed_data):
        """Test train-test split creation"""
        # Ensure churned column exists
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        X_train, X_test, y_train, y_test = self.trainer.create_splits(
            processed_data, test_size=0.2
        )
        
        # Check split sizes
        total_size = len(processed_data)
        self.assertAlmostEqual(len(X_test) / total_size, 0.2, places=1)
        self.assertAlmostEqual(len(X_train) / total_size, 0.8, places=1)
        
        # Check stratification maintained
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        original_churn_rate = processed_data['churned'].mean()
        
        self.assertAlmostEqual(train_churn_rate, original_churn_rate, places=1)
        self.assertAlmostEqual(test_churn_rate, original_churn_rate, places=1)
        
        # Check files saved
        latest_train_path = self.trainer.splits_dir / "latest_X_train.pkl"
        self.assertTrue(latest_train_path.exists())
    
    def test_load_latest_splits(self, processed_data):
        """Test loading saved splits"""
        # Create splits first
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = self.trainer.create_splits(
            processed_data
        )
        
        # Load splits
        X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = self.trainer.load_latest_splits()
        
        # Compare shapes
        self.assertEqual(X_train_orig.shape, X_train_loaded.shape)
        self.assertEqual(X_test_orig.shape, X_test_loaded.shape)
        self.assertEqual(len(y_train_orig), len(y_train_loaded))
        self.assertEqual(len(y_test_orig), len(y_test_loaded))
    
    def test_train_model_xgboost(self, processed_data):
        """Test XGBoost model training"""
        # Prepare data
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        X_train, _, y_train, _ = self.trainer.create_splits(processed_data)
        
        # Train model
        model, scaler, version = self.trainer.train_model(
            X_train, y_train, model_type='xgboost'
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertIsNotNone(version)
        self.assertIn('xgboost', version)
        
        # Check model files saved
        model_path = Path(self.temp_dir) / "model_latest.pkl"
        scaler_path = Path(self.temp_dir) / "scaler_latest.pkl"
        self.assertTrue(model_path.exists())
        self.assertTrue(scaler_path.exists())
    
    def test_train_model_logistic(self, processed_data):
        """Test Logistic Regression model training"""
        # Prepare data
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        X_train, _, y_train, _ = self.trainer.create_splits(processed_data)
        
        # Train model
        model, scaler, version = self.trainer.train_model(
            X_train, y_train, model_type='logistic'
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertIn('logistic', version)
    
    def test_load_latest_model(self, processed_data):
        """Test loading latest trained model"""
        # Train a model first
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        X_train, _, y_train, _ = self.trainer.create_splits(processed_data)
        self.trainer.train_model(X_train, y_train, model_type='xgboost')
        
        # Load model
        model, scaler = self.trainer.load_latest_model()
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        
        # Test predictions work
        X_scaled = scaler.transform(X_train)
        predictions = model.predict_proba(X_scaled)
        self.assertEqual(len(predictions), len(X_train))
    
    def test_train_kfold(self, processed_data):
        """Test k-fold cross-validation training"""
        # Prepare data
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        # Run k-fold with small number of folds for speed
        results = self.trainer.train_kfold(
            processed_data, model_type='xgboost', n_folds=3
        )
        
        self.assertIsNotNone(results)
        self.assertIn('fold_results', results)
        self.assertIn('avg_metrics', results)
        self.assertEqual(len(results['fold_results']), 3)
        
        # Check metrics
        self.assertIn('avg_roc_auc', results['avg_metrics'])
        self.assertIn('std_roc_auc', results['avg_metrics'])
        self.assertGreater(results['avg_metrics']['avg_roc_auc'], 0)
        self.assertLess(results['avg_metrics']['avg_roc_auc'], 1)
        
        # Check best model saved
        best_model_path = Path(self.temp_dir) / "model_kfold_best.pkl"
        self.assertTrue(best_model_path.exists())
    
    def test_metadata_tracking(self, processed_data):
        """Test metadata is properly tracked"""
        if 'churned' not in processed_data.columns:
            processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
        
        # Create splits
        self.trainer.create_splits(processed_data)
        
        # Check splits metadata
        self.assertIn('latest_split', self.trainer.metadata)
        self.assertIn('splits', self.trainer.metadata)
        
        # Train model
        X_train, _, y_train, _ = self.trainer.load_latest_splits()
        self.trainer.train_model(X_train, y_train)
        
        # Check model metadata
        self.assertIn('latest_model', self.trainer.metadata)
        self.assertIn('models', self.trainer.metadata)

@pytest.mark.parametrize("model_type", ['xgboost', 'random_forest', 'logistic'])
def test_different_model_types(processed_data, model_type):
    """Test training with different model types"""
    trainer = ModelTrainer()
    
    if 'churned' not in processed_data.columns:
        processed_data['churned'] = np.random.choice([0, 1], len(processed_data))
    
    X_train, _, y_train, _ = trainer.create_splits(processed_data)
    
    model, scaler, version = trainer.train_model(X_train, y_train, model_type=model_type)
    
    assert model is not None
    assert scaler is not None
    assert model_type in version
    
    # Test prediction works
    X_scaled = scaler.transform(X_train)
    predictions = model.predict_proba(X_scaled)
    assert predictions.shape == (len(X_train), 2)

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Integration tests for the complete pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from data.processor import DataProcessor
from train.trainer import ModelTrainer
from evaluate.evaluator import ModelEvaluator

class TestPipelineIntegration:
    """Integration tests for complete pipeline workflow"""
    
    @pytest.fixture(autouse=True)
    def setup(self, temp_data_dir):
        """Set up test environment"""
        self.temp_dir = temp_data_dir
        self.processor = DataProcessor(data_dir=self.temp_dir)
        self.trainer = ModelTrainer(model_dir=self.temp_dir)
        self.evaluator = ModelEvaluator(results_dir=self.temp_dir)
    
    def test_end_to_end_pipeline(self, sample_csv_path):
        """Test complete pipeline from raw data to evaluation"""
        # Step 1: Process data
        processed_df, version = self.processor.process(str(sample_csv_path))
        assert processed_df is not None
        assert 'churned' in processed_df.columns
        
        # Step 2: Create splits
        X_train, X_test, y_train, y_test = self.trainer.create_splits(processed_df)
        assert len(X_train) > 0
        assert len(X_test) > 0
        
        # Step 3: Train model
        model, scaler, model_version = self.trainer.train_model(
            X_train, y_train, model_type='xgboost'
        )
        assert model is not None
        assert scaler is not None
        
        # Step 4: Evaluate model
        metrics = self.evaluator.evaluate_model(
            model, scaler, X_test, y_test
        )
        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] > 0.5  # Better than random
        
        # Step 5: Save results
        eval_version = self.evaluator.save_results(metrics, model_version)
        assert eval_version is not None
        
        # Verify all outputs exist
        assert (self.processor.processed_dir / "processed_latest.pkl").exists()
        assert (self.trainer.model_dir / "model_latest.pkl").exists()
        assert (self.evaluator.results_dir / "evaluation_latest.json").exists()
    
    def test_pipeline_with_missing_data(self, sample_data):
        """Test pipeline handles missing data correctly"""
        # Add significant missing values
        df = sample_data.copy()
        df.loc[0:20, 'months_with_provider'] = np.nan
        df.loc[10:40, 'payment_method'] = np.nan
        df.loc[15:50, 'lifetime_spend'] = np.nan
        
        # Save to CSV
        csv_path = self.temp_dir / "missing_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Process data
        processed_df, _ = self.processor.process(str(csv_path))
        
        # Check no missing values after processing
        assert not processed_df.isna().any().any()
        
        # Continue with training
        X_train, X_test, y_train, y_test = self.trainer.create_splits(processed_df)
        model, scaler, _ = self.trainer.train_model(X_train, y_train)
        
        # Should complete without errors
        metrics = self.evaluator.evaluate_model(model, scaler, X_test, y_test)
        assert metrics is not None
    
    def test_pipeline_with_data_quality_issues(self, sample_data):
        """Test pipeline handles data quality issues"""
        df = sample_data.copy()
        
        # Add various data quality issues
        df.loc[0:5, 'internet_plan'] = 'Fiber optiic'
        df.loc[6:10, 'internet_plan'] = 'FFiber optic'
        df.loc[0:5, 'payment_method'] = 'ELECTRONIC CHECK'
        df.loc[6:10, 'payment_method'] = 'electronic check'
        
        # Save to CSV
        csv_path = self.temp_dir / "messy_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Process data
        processed_df, _ = self.processor.process(str(csv_path))
        
        # Check data cleaned
        # All internet_plan typos should be fixed
        internet_values = processed_df[processed_df.columns[
            processed_df.columns.str.contains('internet_plan')
        ]].values.flatten()
        
        # Continue with full pipeline
        X_train, X_test, y_train, y_test = self.trainer.create_splits(processed_df)
        model, scaler, _ = self.trainer.train_model(X_train, y_train)
        metrics = self.evaluator.evaluate_model(model, scaler, X_test, y_test)
        
        assert metrics['accuracy'] > 0  # Should produce valid results
    
    def test_kfold_integration(self, sample_csv_path):
        """Test k-fold cross-validation integration"""
        # Process data
        processed_df, _ = self.processor.process(str(sample_csv_path))
        
        # Run k-fold
        kfold_results = self.trainer.train_kfold(
            processed_df, model_type='xgboost', n_folds=3
        )
        
        assert 'avg_metrics' in kfold_results
        assert kfold_results['avg_metrics']['avg_roc_auc'] > 0.5
        
        # Best model should be saved
        best_model_path = self.trainer.model_dir / "model_kfold_best.pkl"
        assert best_model_path.exists()
    
    def test_version_consistency(self, sample_csv_path):
        """Test that versioning is consistent across pipeline"""
        # Process data twice
        _, version1 = self.processor.process(str(sample_csv_path))
        _, version2 = self.processor.process(str(sample_csv_path))
        
        # Versions should be different (timestamp-based)
        assert version1 != version2
        
        # Latest should point to most recent
        latest_df = self.processor.load_latest()
        assert latest_df is not None
        assert self.processor.metadata['latest'] == version2
    
    def test_pipeline_reproducibility(self, sample_csv_path):
        """Test that pipeline produces consistent results with same random seed"""
        # Process data
        processed_df, _ = self.processor.process(str(sample_csv_path))
        
        # Train model twice with same seed
        np.random.seed(42)
        X_train1, X_test1, y_train1, y_test1 = self.trainer.create_splits(
            processed_df, random_state=42
        )
        
        np.random.seed(42)
        X_train2, X_test2, y_train2, y_test2 = self.trainer.create_splits(
            processed_df, random_state=42
        )
        
        # Splits should be identical
        assert X_train1.equals(X_train2)
        assert X_test1.equals(X_test2)
        assert y_train1.equals(y_train2)
        assert y_test1.equals(y_test2)
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Try loading non-existent data
        with pytest.raises(FileNotFoundError):
            self.processor.load_latest()
        
        # Try loading non-existent model
        with pytest.raises(FileNotFoundError):
            self.trainer.load_latest_model()
        
        # Try creating splits without churned column
        df_no_target = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        with pytest.raises(ValueError):
            self.trainer.create_splits(df_no_target)

@pytest.mark.slow
def test_large_dataset_handling(temp_data_dir):
    """Test pipeline with larger dataset (marked as slow test)"""
    # Create larger dataset
    np.random.seed(42)
    n_samples = 5000
    
    large_data = pd.DataFrame({
        'account_id': [f'ACC{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'seniorcitizen': np.random.choice([0, 1], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'months_with_provider': np.random.randint(0, 73, n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples),
        'internet_plan': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check'], n_samples),
        'monthly_fee': np.random.uniform(20, 120, n_samples),
        'lifetime_spend': np.random.uniform(100, 5000, n_samples),
        'churned': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    })
    
    # Add other required columns with defaults
    for col in ['extra_lines', 'addon_security', 'addon_backup', 'addon_device_protect',
                'addon_techsupport', 'stream_tv', 'stream_movies', 'paperless_billing',
                'customer_hash', 'marketing_opt_in']:
        if col not in large_data.columns:
            if col == 'marketing_opt_in':
                large_data[col] = np.random.choice([0, 1], n_samples)
            elif col == 'customer_hash':
                large_data[col] = [f'hash_{i}' for i in range(n_samples)]
            else:
                large_data[col] = np.random.choice(['Yes', 'No'], n_samples)
    
    # Save and process
    csv_path = temp_data_dir / "large_data.csv"
    large_data.to_csv(csv_path, index=False)
    
    processor = DataProcessor(data_dir=temp_data_dir)
    trainer = ModelTrainer(model_dir=temp_data_dir)
    
    # Run pipeline
    processed_df, _ = processor.process(str(csv_path))
    X_train, X_test, y_train, y_test = trainer.create_splits(processed_df)
    model, scaler, _ = trainer.train_model(X_train, y_train, model_type='xgboost')
    
    # Should handle large dataset
    assert len(processed_df) == n_samples
    assert model is not None

if __name__ == "__main__":
    pytest.main([__file__, '-v'])

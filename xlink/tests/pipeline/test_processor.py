#!/usr/bin/env python3
"""
Pytest-style tests for data processor module using sample data
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from data.processor import DataProcessor

class TestDataProcessor:
    """Test cases for DataProcessor class using pytest fixtures"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DataProcessor(data_dir=self.temp_dir)
        yield
        # Cleanup after each test
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor is not None
        assert Path(self.temp_dir).exists()
        assert self.processor.processed_dir.exists()
    
    def test_clean_data_internet_plan(self, sample_data):
        """Test cleaning of internet_plan typos"""
        df = sample_data.copy()
        df.loc[0, 'internet_plan'] = 'Fiber optiic'
        if len(df) > 1:
            df.loc[1, 'internet_plan'] = 'FFiber optic'
        
        cleaned = self.processor.clean_data(df)
        
        assert cleaned.loc[0, 'internet_plan'] == 'Fiber optic'
        if len(df) > 1:
            assert cleaned.loc[1, 'internet_plan'] == 'Fiber optic'
    
    def test_clean_data_payment_method(self, sample_data):
        """Test standardization of payment_method"""
        df = sample_data.copy()
        df.loc[0, 'payment_method'] = 'Electronic check'
        if len(df) > 2:
            df.loc[1, 'payment_method'] = 'ELECTRONIC CHECK'
            df.loc[2, 'payment_method'] = 'electronic check'
        
        cleaned = self.processor.clean_data(df)
        
        # All should be standardized to 'electronic_check'
        assert cleaned.loc[0, 'payment_method'] == 'electronic_check'
        if len(df) > 2:
            assert cleaned.loc[1, 'payment_method'] == 'electronic_check'
            assert cleaned.loc[2, 'payment_method'] == 'electronic_check'
    
    def test_impute_missing_values(self, sample_data):
        """Test missing value imputation"""
        df = sample_data.copy()
        
        # Create missing values for testing
        missing_indices = min(5, len(df) - 1)
        df.loc[0:missing_indices-1, 'months_with_provider'] = np.nan
        if len(df) > 10:
            df.loc[5:9, 'payment_method'] = np.nan
            df.loc[10:14, 'lifetime_spend'] = np.nan
        
        imputed = self.processor.impute_missing(df)
        
        # Check imputation
        assert not imputed['months_with_provider'].isna().any()
        assert not imputed['payment_method'].isna().any()
        
        # Check lifetime_spend imputation (should use monthly_fee * months_with_provider)
        if len(df) > 10:
            for idx in range(10, min(15, len(df))):
                if not pd.isna(df.loc[idx, 'monthly_fee']):
                    expected = imputed.loc[idx, 'monthly_fee'] * imputed.loc[idx, 'months_with_provider']
                    assert abs(imputed.loc[idx, 'lifetime_spend'] - expected) < 0.01
    
    def test_engineer_features(self, sample_data):
        """Test feature engineering"""
        df = sample_data.copy()
        
        # Ensure required columns exist
        if 'months_with_provider' not in df.columns:
            df['months_with_provider'] = np.random.randint(0, 73, len(df))
        
        engineered = self.processor.engineer_features(df)
        
        # Check new features exist
        assert 'tenure_group' in engineered.columns
        assert 'high_risk_payment' in engineered.columns
        assert 'high_value_customer' in engineered.columns
        
        # Check dropped columns
        assert 'account_id' not in engineered.columns
        assert 'customer_hash' not in engineered.columns
        if 'lifetime_spend' in sample_data.columns:
            assert 'lifetime_spend' not in engineered.columns
        if 'marketing_opt_in' in sample_data.columns:
            assert 'marketing_opt_in' not in engineered.columns
    
    def test_encode_features(self, sample_data):
        """Test feature encoding"""
        df = sample_data.copy()
        
        # Process data first to get engineered features
        df = self.processor.clean_data(df)
        df = self.processor.impute_missing(df)
        df = self.processor.engineer_features(df)
        
        encoded = self.processor.encode_features(df)
        
        # Check binary encodings
        if 'gender' in df.columns:
            assert encoded['gender'].isin([0, 1]).all()
        
        if 'partner' in df.columns:
            assert encoded['partner'].isin([0, 1]).all()
        
        # Check ordinal encoding for contract_type
        if 'contract_type' in df.columns:
            assert encoded['contract_type'].isin([0, 1, 2]).all()
        
        # Check one-hot encoding created new columns
        assert any('internet_plan_' in col for col in encoded.columns)
    
    def test_full_processing_pipeline(self, sample_csv_path):
        """Test complete processing pipeline"""
        df, version = self.processor.process(str(sample_csv_path))
        
        # Check output
        assert df is not None
        assert version is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check processed files exist
        latest_path = self.processor.processed_dir / "processed_latest.pkl"
        assert latest_path.exists()
        
        # Check metadata updated
        assert 'latest' in self.processor.metadata
        assert version in self.processor.metadata['versions']
    
    def test_load_latest(self, sample_csv_path):
        """Test loading latest processed data"""
        # First process some data
        self.processor.process(str(sample_csv_path))
        
        # Load latest
        df = self.processor.load_latest()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_version_hash_generation(self, sample_data):
        """Test version hash is consistent for same data"""
        hash1 = self.processor._generate_version_hash(sample_data)
        hash2 = self.processor._generate_version_hash(sample_data)
        
        assert hash1 == hash2
        
        # Modify data and check hash changes
        modified_data = sample_data.copy()
        modified_data['new_column'] = 1
        hash3 = self.processor._generate_version_hash(modified_data)
        
        assert hash1 != hash3

@pytest.mark.parametrize("missing_col,expected_strategy", [
    ('months_with_provider', 'median'),
    ('payment_method', 'mode'),
    ('lifetime_spend', 'calculated')
])
def test_imputation_strategies(sample_data, missing_col, expected_strategy):
    """Test different imputation strategies"""
    processor = DataProcessor()
    df = sample_data.copy()
    
    # Create missing values
    missing_count = min(10, len(df) - 1)
    df.loc[0:missing_count-1, missing_col] = np.nan
    
    imputed = processor.impute_missing(df)
    
    # Check no missing values remain
    assert not imputed[missing_col].isna().any()
    
    # Verify imputation logic
    if expected_strategy == 'median' and missing_col == 'months_with_provider':
        # The median might be different after introducing missing values
        expected_val = df[missing_col].median()  # Use the actual median from test data
        assert imputed.loc[0, missing_col] == expected_val
    elif expected_strategy == 'calculated' and missing_col == 'lifetime_spend':
        if 'monthly_fee' in df.columns and 'months_with_provider' in df.columns:
            expected = imputed.loc[0, 'monthly_fee'] * imputed.loc[0, 'months_with_provider']
            assert abs(imputed.loc[0, missing_col] - expected) < 0.01

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
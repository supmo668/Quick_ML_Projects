#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for testing pipeline modules
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add pipeline modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

@pytest.fixture(scope="session")
def sample_data():
    """Load and sample a small subset of real data for testing"""
    data_path = Path(__file__).parent.parent / "customer_churn.csv"
    
    if not data_path.exists():
        # Create minimal synthetic data if real data not available
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'account_id': [f'ACC{str(i).zfill(4)}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'seniorcitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'partner': np.random.choice(['Yes', 'No'], n_samples),
            'dependents': np.random.choice(['Yes', 'No'], n_samples),
            'months_with_provider': np.random.randint(0, 73, n_samples),
            'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'extra_lines': np.random.choice(['No', 'Yes', 'No phone service'], n_samples),
            'internet_plan': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
            'addon_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'addon_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'addon_device_protect': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'addon_techsupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'stream_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'stream_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                              'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'monthly_fee': np.round(np.random.uniform(20, 120, n_samples), 2),
            'lifetime_spend': np.round(np.random.uniform(100, 5000, n_samples), 2),
            'churned': np.random.choice([0, 1], n_samples, p=[0.73, 0.27]),
            'customer_hash': [f'hash_{i}' for i in range(n_samples)],
            'marketing_opt_in': np.random.choice([0, 1], n_samples)
        })
        
        # Add some missing values
        data.loc[np.random.choice(data.index, 5, replace=False), 'months_with_provider'] = np.nan
        data.loc[np.random.choice(data.index, 10, replace=False), 'payment_method'] = np.nan
        data.loc[np.random.choice(data.index, 12, replace=False), 'lifetime_spend'] = np.nan
        
        # Add some data quality issues
        data.loc[0, 'internet_plan'] = 'Fiber optiic'  # Typo
        data.loc[1, 'payment_method'] = 'electronic check'  # Case variation
        
        return data
    
    # Load real data and sample
    try:
        full_data = pd.read_csv(data_path)
        # Sample 200 rows for testing (stratified by churn if possible)
        if 'churned' in full_data.columns:
            sample = full_data.groupby('churned', group_keys=False).apply(
                lambda x: x.sample(min(100, len(x)), random_state=42)
            )
        else:
            sample = full_data.sample(min(200, len(full_data)), random_state=42)
        
        return sample.reset_index(drop=True)
    except Exception as e:
        print(f"Warning: Could not load real data: {e}")
        # Return synthetic data as fallback
        return create_synthetic_data()

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after tests
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_csv_path(sample_data, temp_data_dir):
    """Save sample data to a temporary CSV file"""
    csv_path = temp_data_dir / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def processed_data(sample_data):
    """Create a processed version of sample data"""
    df = sample_data.copy()
    
    # Simple processing for testing
    # Fix known issues
    if 'internet_plan' in df.columns:
        df['internet_plan'] = df['internet_plan'].replace({
            'Fiber optiic': 'Fiber optic',
            'Fiber opttic': 'Fiber optic'
        })
    
    if 'payment_method' in df.columns:
        df['payment_method'] = df['payment_method'].str.lower().str.strip()
        df['payment_method'] = df['payment_method'].replace({
            'electronic check': 'electronic_check',
            'mailed check': 'mailed_check'
        })
    
    # Add engineered features
    df['tenure_group'] = pd.cut(
        df['months_with_provider'].fillna(df['months_with_provider'].median()),
        bins=[-1, 12, 24, 48, 100],
        labels=['0-12', '13-24', '25-48', '49+']
    ).astype(str)
    
    return df

@pytest.fixture
def model_config():
    """Configuration for model testing"""
    return {
        'model_type': 'xgboost',
        'test_size': 0.2,
        'n_folds': 3,  # Smaller for faster tests
        'random_state': 42
    }

def create_synthetic_data():
    """Create synthetic data for testing when real data is not available"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'account_id': [f'ACC{str(i).zfill(4)}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'seniorcitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'months_with_provider': np.random.randint(0, 73, n_samples),
        'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'extra_lines': np.random.choice(['No', 'Yes', 'No phone service'], n_samples),
        'internet_plan': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
        'addon_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'addon_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'addon_device_protect': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'addon_techsupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'stream_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'stream_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'monthly_fee': np.round(np.random.uniform(20, 120, n_samples), 2),
        'lifetime_spend': np.round(np.random.uniform(100, 5000, n_samples), 2),
        'churned': np.random.choice([0, 1], n_samples, p=[0.73, 0.27]),
        'customer_hash': [f'hash_{i}' for i in range(n_samples)],
        'marketing_opt_in': np.random.choice([0, 1], n_samples)
    })
    
    # Add realistic missing values
    data.loc[np.random.choice(data.index, 5, replace=False), 'months_with_provider'] = np.nan
    data.loc[np.random.choice(data.index, 10, replace=False), 'payment_method'] = np.nan
    data.loc[np.random.choice(data.index, 12, replace=False), 'lifetime_spend'] = np.nan
    
    # Add data quality issues to test cleaning
    data.loc[0, 'internet_plan'] = 'Fiber optiic'
    data.loc[1, 'payment_method'] = 'electronic check'
    data.loc[2, 'payment_method'] = 'ELECTRONIC CHECK'
    
    return data

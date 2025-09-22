#!/usr/bin/env python3
"""
Data processing and feature engineering module
Handles data cleaning, imputation, and feature engineering with versioning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime
import pickle
from typing import Dict, Any, Tuple, Optional

class DataProcessor:
    """Main data processing class with versioning support"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir or "data")
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.processed_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Processing parameters
        self.internet_plan_fixes = {
            'Fiber optiic': 'Fiber optic',
            'Fiber opttic': 'Fiber optic',
            'iFber optic': 'Fiber optic',
            'FFiber optic': 'Fiber optic',
            'Fiiber optic': 'Fiber optic',
            'Fibe optic': 'Fiber optic'
        }
        
    def _load_metadata(self) -> Dict:
        """Load processing metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"versions": {}}
    
    def _save_metadata(self):
        """Save processing metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _generate_version_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for data version"""
        data_str = f"{df.shape}_{df.columns.tolist()}_{df.dtypes.tolist()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data inconsistencies"""
        df = df.copy()
        
        # Fix internet_plan typos
        if 'internet_plan' in df.columns:
            df['internet_plan'] = df['internet_plan'].replace(self.internet_plan_fixes)
        
        # Standardize payment_method
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].str.lower().str.strip()
            df['payment_method'] = df['payment_method'].replace({
                'electronic check': 'electronic_check',
                'mailed check': 'mailed_check',
                'bank transfer (automatic)': 'bank_transfer',
                'credit card (automatic)': 'credit_card',
                'nan': np.nan,
                'mailed_check': 'mailed_check',  # Already correct
                'electronic_check': 'electronic_check'  # Already correct
            })
        
        return df
    
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on statistical analysis"""
        df = df.copy()
        
                # Impute months_with_provider with median (4.98% missing)
        if 'months_with_provider' in df.columns:
            median_tenure = df['months_with_provider'].median()
            df['months_with_provider'] = df['months_with_provider'].fillna(median_tenure)
        
        # Impute lifetime_spend using correlation
        if 'lifetime_spend' in df.columns:
            mask = df['lifetime_spend'].isna()
            if mask.any() and 'monthly_fee' in df.columns and 'months_with_provider' in df.columns:
                df.loc[mask, 'lifetime_spend'] = (
                    df.loc[mask, 'monthly_fee'] * df.loc[mask, 'months_with_provider']
                )
        
                # Impute payment_method with mode (9.99% missing)
        if 'payment_method' in df.columns:
            mode_payment = df['payment_method'].mode()[0] if len(df['payment_method'].mode()) > 0 else 'electronic_check'
            df['payment_method'] = df['payment_method'].fillna(mode_payment)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        df = df.copy()
        
        # Tenure groups
        if 'months_with_provider' in df.columns:
            df['tenure_group'] = pd.cut(
                df['months_with_provider'],
                bins=[-1, 12, 24, 48, 100],
                labels=['0-12', '13-24', '25-48', '49+']
            ).astype(str)
        
        # High-risk payment flag
        if 'payment_method' in df.columns:
            df['high_risk_payment'] = (df['payment_method'] == 'electronic_check').astype(int)
        
        # Service bundle indicator
        if 'phone_service' in df.columns and 'internet_plan' in df.columns:
            df['is_multi_service'] = (
                (df['phone_service'] == 'Yes') & 
                (df['internet_plan'] != 'No')
            ).astype(int)
        
        # High-value customer
        if 'monthly_fee' in df.columns:
            df['high_value_customer'] = (df['monthly_fee'] > 70.35).astype(int)
        
        # Senior with dependents
        if 'seniorcitizen' in df.columns and 'dependents' in df.columns:
            df['senior_with_dependents'] = (
                (df['seniorcitizen'] == 1) & 
                (df['dependents'] == 'Yes')
            ).astype(int)
        
        # Drop redundant features
        drop_cols = ['account_id', 'customer_hash', 'lifetime_spend', 'marketing_opt_in']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        # Binary encoding
        binary_mappings = {
            'gender': {'Male': 1, 'Female': 0},
            'partner': {'Yes': 1, 'No': 0},
            'dependents': {'Yes': 1, 'No': 0},
            'phone_service': {'Yes': 1, 'No': 0},
            'paperless_billing': {'Yes': 1, 'No': 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)
        
        # Ordinal encoding for contract_type
        if 'contract_type' in df.columns:
            contract_mapping = {
                'Two year': 0,
                'One year': 1,
                'Month-to-month': 2
            }
            df['contract_type'] = df['contract_type'].map(contract_mapping).fillna(1)
        
        # One-hot encoding for remaining categorical columns
        categorical_cols = ['internet_plan', 'extra_lines', 'payment_method',
                          'addon_security', 'addon_backup', 'addon_device_protect',
                          'addon_techsupport', 'stream_tv', 'stream_movies', 'tenure_group']
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        return df
    
    def process(self, input_path: str, version_name: str = None) -> Tuple[pd.DataFrame, str]:
        """Complete data processing pipeline with versioning"""
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded data: {df.shape}")
        
        # Process data
        df = self.clean_data(df)
        df = self.impute_missing(df)
        df = self.engineer_features(df)
        
        # Separate target if exists
        target = None
        if 'churned' in df.columns:
            target = df['churned']
            df = df.drop(columns=['churned'])
        
        # Encode features
        df = self.encode_features(df)
        
        # Add target back
        if target is not None:
            df['churned'] = target
        
        # Generate version
        version_hash = self._generate_version_hash(df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = version_name or f"v_{timestamp}_{version_hash}"
        
        # Save processed data
        output_path = self.processed_dir / f"processed_{version}.pkl"
        df.to_pickle(output_path)
        
        # Save as latest
        latest_path = self.processed_dir / "processed_latest.pkl"
        df.to_pickle(latest_path)
        
        # Update metadata
        self.metadata["versions"][version] = {
            "timestamp": timestamp,
            "hash": version_hash,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "path": str(output_path)
        }
        self.metadata["latest"] = version
        self._save_metadata()
        
        print(f"Processed data saved: {output_path}")
        print(f"Version: {version}")
        
        return df, version
    
    def load_latest(self) -> pd.DataFrame:
        """Load latest processed data"""
        latest_path = self.processed_dir / "processed_latest.pkl"
        if not latest_path.exists():
            raise FileNotFoundError("No processed data found. Run processing first.")
        return pd.read_pickle(latest_path)
    
    def get_version_info(self) -> Dict:
        """Get information about processed versions"""
        return self.metadata

def main():
    """Main execution for testing"""
    processor = DataProcessor()
    
    # Process data
    input_path = "data/customer_churn.csv"
    if Path(input_path).exists():
        df, version = processor.process(input_path)
        print(f"\nProcessed shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")  # First 10 columns
    else:
        print(f"Input file not found: {input_path}")

if __name__ == "__main__":
    main()

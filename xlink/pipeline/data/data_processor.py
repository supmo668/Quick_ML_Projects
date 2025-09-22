#!/usr/bin/env python3
"""
Data processing module for customer churn prediction
Handles data cleaning, imputation, and feature engineering with versioning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class with versioning support"""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        self.metadata = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data inconsistencies identified in statistical analysis"""
        df = df.copy()
        
        # Fix internet_plan typos
        internet_plan_fixes = {
            'Fiber optiic': 'Fiber optic',
            'Fiber opttic': 'Fiber optic',
            'iFber optic': 'Fiber optic',
            'FFiber optic': 'Fiber optic',
            'Fiiber optic': 'Fiber optic',
            'Fibe optic': 'Fiber optic'
        }
        if 'internet_plan' in df.columns:
            df['internet_plan'] = df['internet_plan'].replace(internet_plan_fixes)
        
        # Standardize payment_method case variations
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].str.lower().str.strip()
            df['payment_method'] = df['payment_method'].replace({
                'electronic check': 'electronic_check',
                'mailed check': 'mailed_check',
                'bank transfer (automatic)': 'bank_transfer',
                'credit card (automatic)': 'credit_card',
                'nan': np.nan,
                'bank transfer': 'bank_transfer',
                'credit card': 'credit_card'
            })
        
        logger.info(f"Data cleaned: {len(df)} rows")
        return df
    
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on correlation analysis"""
        df = df.copy()
        
        # Impute months_with_provider with median
        if 'months_with_provider' in df.columns:
            median_tenure = df['months_with_provider'].median()
            df['months_with_provider'].fillna(median_tenure, inplace=True)
            self.metadata['tenure_median'] = float(median_tenure)
        
        # Impute lifetime_spend using correlation with monthly_fee and tenure
        if all(col in df.columns for col in ['lifetime_spend', 'monthly_fee', 'months_with_provider']):
            mask = df['lifetime_spend'].isna()
            if mask.any():
                df.loc[mask, 'lifetime_spend'] = (
                    df.loc[mask, 'monthly_fee'] * df.loc[mask, 'months_with_provider']
                )
        
        # Impute payment_method with mode
        if 'payment_method' in df.columns:
            mode_payment = df['payment_method'].mode()[0] if len(df['payment_method'].mode()) > 0 else 'electronic_check'
            df['payment_method'].fillna(mode_payment, inplace=True)
            self.metadata['payment_mode'] = mode_payment
        
        logger.info(f"Missing values imputed. Remaining missing: {df.isnull().sum().sum()}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features based on statistical insights"""
        df = df.copy()
        
        # Create tenure bins
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
        if all(col in df.columns for col in ['phone_service', 'internet_plan']):
            df['is_multi_service'] = (
                (df['phone_service'] == 'Yes') & 
                (df['internet_plan'] != 'No')
            ).astype(int)
        
        # High-value customer flag
        if 'monthly_fee' in df.columns:
            monthly_median = df['monthly_fee'].median()
            df['high_value_customer'] = (df['monthly_fee'] > monthly_median).astype(int)
            self.metadata['monthly_fee_median'] = float(monthly_median)
        
        # Total services count
        service_cols = ['phone_service', 'addon_security', 'addon_backup', 
                       'addon_device_protect', 'addon_techsupport', 'stream_tv', 'stream_movies']
        available_services = [col for col in service_cols if col in df.columns]
        if available_services:
            df['total_services'] = (df[available_services] == 'Yes').sum(axis=1)
        
        logger.info(f"Features engineered. Total features: {len(df.columns)}")
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        # Binary encoding for simple binary features
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
        
        # One-hot encoding for multi-category features
        categorical_cols = ['internet_plan', 'extra_lines', 'tenure_group']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        
        # One-hot for addon services (they have 3 categories)
        addon_cols = ['addon_security', 'addon_backup', 'addon_device_protect', 
                     'addon_techsupport', 'stream_tv', 'stream_movies']
        for col in addon_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        # Payment method - create binary flags for high-risk methods
        if 'payment_method' in df.columns:
            df['payment_electronic_check'] = (df['payment_method'] == 'electronic_check').astype(int)
            df['payment_mailed_check'] = (df['payment_method'] == 'mailed_check').astype(int)
            df['payment_bank_transfer'] = (df['payment_method'] == 'bank_transfer').astype(int)
            df = df.drop(columns=['payment_method'])
        
        logger.info(f"Features encoded. Final shape: {df.shape}")
        return df
    
    def remove_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-predictive identifier columns"""
        drop_cols = ['account_id', 'customer_hash', 'lifetime_spend', 'marketing_opt_in']
        drop_cols = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=drop_cols)
        logger.info(f"Removed {len(drop_cols)} identifier columns")
        return df
    
    def process(self, input_path: str, save_version: bool = True) -> Tuple[pd.DataFrame, str]:
        """Complete data processing pipeline with versioning"""
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Store original shape
        self.metadata['original_shape'] = df.shape
        self.metadata['original_columns'] = df.columns.tolist()
        
        # Process data
        df = self.clean_data(df)
        df = self.impute_missing(df)
        df = self.engineer_features(df)
        df = self.encode_features(df)
        df = self.remove_identifiers(df)
        
        # Store processed shape
        self.metadata['processed_shape'] = df.shape
        self.metadata['processed_columns'] = df.columns.tolist()
        self.metadata['processing_timestamp'] = datetime.now().isoformat()
        
        # Generate version hash
        data_hash = hashlib.md5(
            f"{self.metadata['processing_timestamp']}_{df.shape}".encode()
        ).hexdigest()[:8]
        version = f"v_{data_hash}"
        
        if save_version:
            # Save versioned data
            version_dir = self.processed_dir / version
            version_dir.mkdir(exist_ok=True)
            
            df.to_csv(version_dir / 'processed_data.csv', index=False)
            
            with open(version_dir / 'metadata.json', 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Update latest symlink
            latest_path = self.processed_dir / 'latest'
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(version_dir)
            
            logger.info(f"Data processed and saved to version: {version}")
        
        return df, version
    
    def load_latest(self) -> Optional[pd.DataFrame]:
        """Load the latest processed dataset"""
        latest_path = self.processed_dir / 'latest' / 'processed_data.csv'
        if not latest_path.exists():
            logger.error("No processed data found. Run processing first.")
            return None
        
        df = pd.read_csv(latest_path)
        logger.info(f"Loaded latest processed data: {df.shape}")
        return df

def main():
    """Main execution for testing"""
    processor = DataProcessor()
    df, version = processor.process('data/customer_churn.csv')
    print(f"Processing complete. Version: {version}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()

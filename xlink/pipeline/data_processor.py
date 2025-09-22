"""
Data processing and feature engineering module
Handles data cleaning, imputation, and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import joblib
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing pipeline based on statistical analysis from PLAN.md"""
    
    def __init__(self, data_dir: str = "data", processed_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Store preprocessing parameters for consistency
        self.preprocessing_params = {
            'payment_mode': None,
            'tenure_median': None,
            'monthly_fee_median': None,
            'encoding_maps': {}
        }
        
        # Data quality fixes from summary statistics
        self.internet_plan_fixes = {
            'Fiber optiic': 'Fiber optic',
            'Fiber opttic': 'Fiber optic',
            'iFber optic': 'Fiber optic',
            'FFiber optic': 'Fiber optic',
            'Fiiber optic': 'Fiber optic',
            'Fibe optic': 'Fiber optic'
        }
    
    def load_raw_data(self, filename: str = "customer_churn.csv") -> pd.DataFrame:
        """Load raw data from CSV"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data inconsistencies identified in statistical analysis"""
        df = df.copy()
        
        # Fix internet_plan typos
        if 'internet_plan' in df.columns:
            df['internet_plan'] = df['internet_plan'].replace(self.internet_plan_fixes)
            logger.info("Fixed internet_plan inconsistencies")
        
        # Standardize payment_method case variations
        if 'payment_method' in df.columns:
            # Handle NaN strings
            df['payment_method'] = df['payment_method'].replace(['NAN', 'nan'], np.nan)
            # Standardize case
            df['payment_method'] = df['payment_method'].str.lower().str.strip()
            # Consolidate variations
            payment_method_fixes = {
                'electronic check': 'electronic_check',
                'mailed check': 'mailed_check',
                'bank transfer (automatic)': 'bank_transfer',
                'credit card (automatic)': 'credit_card'
            }
            df['payment_method'] = df['payment_method'].replace(payment_method_fixes)
            logger.info("Standardized payment_method values")
        
        return df
    
    def impute_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Impute missing values based on correlation analysis
        Uses correlation insights: lifetime_spend ~ 0.80*tenure + 0.55*monthly_fee
        """
        df = df.copy()
        
        if fit:
            # Calculate and store imputation values
            if 'payment_method' in df.columns:
                self.preprocessing_params['payment_mode'] = df['payment_method'].mode()[0] if len(df['payment_method'].mode()) > 0 else 'electronic_check'
            
            if 'months_with_provider' in df.columns:
                self.preprocessing_params['tenure_median'] = df['months_with_provider'].median()
            
            if 'monthly_fee' in df.columns:
                self.preprocessing_params['monthly_fee_median'] = df['monthly_fee'].median()
        
        # Impute months_with_provider first (needed for lifetime_spend)
        if 'months_with_provider' in df.columns:
            missing_count = df['months_with_provider'].isna().sum()
            df['months_with_provider'].fillna(self.preprocessing_params['tenure_median'], inplace=True)
            logger.info(f"Imputed {missing_count} missing values in months_with_provider")
        
        # Impute lifetime_spend using correlation with monthly_fee and tenure
        if 'lifetime_spend' in df.columns:
            mask = df['lifetime_spend'].isna()
            missing_count = mask.sum()
            if mask.any() and 'monthly_fee' in df.columns and 'months_with_provider' in df.columns:
                df.loc[mask, 'lifetime_spend'] = (
                    df.loc[mask, 'monthly_fee'] * df.loc[mask, 'months_with_provider'] * 1.05  # Small adjustment factor
                )
            logger.info(f"Imputed {missing_count} missing values in lifetime_spend")
        
        # Impute payment_method with mode
        if 'payment_method' in df.columns:
            missing_count = df['payment_method'].isna().sum()
            df['payment_method'].fillna(self.preprocessing_params['payment_mode'], inplace=True)
            logger.info(f"Imputed {missing_count} missing values in payment_method")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features based on statistical insights"""
        df = df.copy()
        
        # Create tenure bins based on churn patterns
        if 'months_with_provider' in df.columns:
            df['tenure_group'] = pd.cut(
                df['months_with_provider'],
                bins=[-1, 12, 24, 48, 100],
                labels=['0-12', '13-24', '25-48', '49+']
            )
            df['is_new_customer'] = (df['months_with_provider'] <= 12).astype(int)
            logger.info("Created tenure-based features")
        
        # High-risk payment flag (electronic check has 44.92% churn)
        if 'payment_method' in df.columns:
            df['high_risk_payment'] = (df['payment_method'] == 'electronic_check').astype(int)
        
        # Service bundle indicators
        if 'phone_service' in df.columns and 'internet_plan' in df.columns:
            df['is_multi_service'] = (
                (df['phone_service'] == 'Yes') & 
                (df['internet_plan'] != 'No')
            ).astype(int)
        
        # High-value customer flag (above median monthly fee: 70.35)
        if 'monthly_fee' in df.columns:
            df['high_value_customer'] = (df['monthly_fee'] > 70.35).astype(int)
        
        # Senior with dependents (vulnerable segment)
        if 'seniorcitizen' in df.columns and 'dependents' in df.columns:
            df['senior_with_dependents'] = (
                (df['seniorcitizen'] == 1) & 
                (df['dependents'] == 'Yes')
            ).astype(int)
        
        # Contract commitment score (ordinal)
        if 'contract_type' in df.columns:
            contract_scores = {
                'Two year': 2,
                'One year': 1,
                'Month-to-month': 0
            }
            df['contract_commitment'] = df['contract_type'].map(contract_scores).fillna(0)
        
        # Internet service quality (Fiber has higher churn)
        if 'internet_plan' in df.columns:
            df['has_fiber'] = (df['internet_plan'] == 'Fiber optic').astype(int)
        
        logger.info("Created 8 engineered features")
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables based on their characteristics"""
        df = df.copy()
        
        # Binary encoding for Yes/No features
        binary_mappings = {'Yes': 1, 'No': 0}
        binary_cols = ['partner', 'dependents', 'phone_service', 'paperless_billing']
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(binary_mappings).fillna(0)
        
        # Gender encoding
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)
        
        # One-hot encode categorical features with multiple categories
        categorical_cols = ['extra_lines', 'internet_plan', 'addon_security', 
                          'addon_backup', 'addon_device_protect', 'addon_techsupport',
                          'stream_tv', 'stream_movies']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
        
        # Handle tenure_group if it exists (from feature engineering)
        if 'tenure_group' in df.columns:
            tenure_dummies = pd.get_dummies(df['tenure_group'], prefix='tenure', drop_first=False)
            df = pd.concat([df, tenure_dummies], axis=1)
            df.drop(columns=['tenure_group'], inplace=True)
        
        # Target encode payment_method based on churn rates
        if 'payment_method' in df.columns and fit:
            # Map to risk scores based on churn analysis
            payment_risk_scores = {
                'electronic_check': 0.449,  # 44.9% churn
                'mailed_check': 0.190,      # 19.0% churn
                'bank_transfer': 0.167,     # 16.7% churn
                'credit_card': 0.150        # 15.0% churn
            }
            self.preprocessing_params['encoding_maps']['payment_risk'] = payment_risk_scores
        
        if 'payment_method' in df.columns:
            df['payment_risk_score'] = df['payment_method'].map(
                self.preprocessing_params['encoding_maps'].get('payment_risk', {})
            ).fillna(0.25)  # Default to average risk
            df.drop(columns=['payment_method'], inplace=True)
        
        # Drop contract_type (already encoded as contract_commitment)
        if 'contract_type' in df.columns:
            df.drop(columns=['contract_type'], inplace=True)
        
        logger.info(f"Encoded features - resulting shape: {df.shape}")
        return df
    
    def process_data(self, df: pd.DataFrame = None, fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete data processing pipeline
        Returns processed dataframe and metadata
        """
        # Load data if not provided
        if df is None:
            df = self.load_raw_data()
        
        initial_shape = df.shape
        
        # Separate target and features
        target = None
        if 'churned' in df.columns:
            target = df['churned'].copy()
            df = df.drop(columns=['churned'])
        
        # Apply processing steps
        df = self.clean_data(df)
        df = self.impute_missing(df, fit=fit)
        df = self.engineer_features(df)
        
        # Drop unnecessary columns
        drop_cols = ['account_id', 'customer_hash', 'lifetime_spend', 'marketing_opt_in']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Encode features
        df = self.encode_features(df, fit=fit)
        
        # Add target back
        if target is not None:
            df['churned'] = target
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'initial_shape': initial_shape,
            'final_shape': df.shape,
            'features': list(df.columns),
            'preprocessing_params': self.preprocessing_params,
            'version': 'latest'
        }
        
        logger.info(f"Processing complete: {initial_shape} -> {df.shape}")
        return df, metadata
    
    def save_processed_data(self, df: pd.DataFrame, metadata: Dict[str, Any], 
                           version: str = None) -> Dict[str, str]:
        """Save processed data with versioning"""
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save versioned data
        versioned_path = self.processed_dir / f"processed_data_{version}.csv"
        df.to_csv(versioned_path, index=False)
        
        # Save as latest
        latest_path = self.processed_dir / "processed_data_latest.csv"
        df.to_csv(latest_path, index=False)
        
        # Save metadata
        metadata['version'] = version
        metadata_path = self.processed_dir / f"metadata_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save latest metadata
        latest_metadata_path = self.processed_dir / "metadata_latest.json"
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save preprocessing parameters for inference
        params_path = self.processed_dir / "preprocessing_params.pkl"
        joblib.dump(self.preprocessing_params, params_path)
        
        logger.info(f"Saved processed data: version={version}, latest updated")
        
        return {
            'versioned_path': str(versioned_path),
            'latest_path': str(latest_path),
            'metadata_path': str(metadata_path),
            'version': version
        }
    
    def load_latest_processed_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load the latest processed dataset"""
        latest_path = self.processed_dir / "processed_data_latest.csv"
        metadata_path = self.processed_dir / "metadata_latest.json"
        
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Latest processed data not found at {latest_path}. "
                "Please run data processing first."
            )
        
        df = pd.read_csv(latest_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load preprocessing parameters
        params_path = self.processed_dir / "preprocessing_params.pkl"
        if params_path.exists():
            self.preprocessing_params = joblib.dump(params_path)
        
        logger.info(f"Loaded latest processed data: {df.shape}")
        return df, metadata

def main():
    """Main execution for data processing"""
    processor = DataProcessor()
    
    # Process data
    df, metadata = processor.process_data()
    
    # Save processed data
    result = processor.save_processed_data(df, metadata)
    
    print(f"✅ Data processing complete!")
    print(f"📁 Saved to: {result['latest_path']}")
    print(f"📊 Shape: {df.shape}")
    print(f"🏷️ Version: {result['version']}")
    
    return result

if __name__ == "__main__":
    main()

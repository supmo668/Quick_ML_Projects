#!/usr/bin/env python3
"""
Model training module for customer churn prediction
Handles train-test split, model training, and k-fold validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main model training class with support for different training strategies"""
    
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.metadata = {}
        self.feature_names = None
        
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Split data into train and test sets"""
        # Separate features and target
        if 'churned' not in df.columns:
            raise ValueError("Target column 'churned' not found in dataset")
        
        X = df.drop(columns=['churned'])
        y = df['churned']
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Store metadata
        self.metadata['split_info'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_ratio': test_size,
            'train_churn_rate': float(y_train.mean()),
            'test_churn_rate': float(y_test.mean())
        }
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"Churn rates: Train={y_train.mean():.2%}, Test={y_test.mean():.2%}")
        
        # Save splits
        splits_dir = self.model_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        X_train.to_csv(splits_dir / 'X_train.csv', index=False)
        X_test.to_csv(splits_dir / 'X_test.csv', index=False)
        y_train.to_csv(splits_dir / 'y_train.csv', index=False)
        y_test.to_csv(splits_dir / 'y_test.csv', index=False)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_type: str = 'xgboost') -> Any:
        """Train a model with calibration"""
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize base model
        if model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=2.76,  # Based on class imbalance
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:  # logistic regression
            base_model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        
        # Apply calibration for better probability estimates
        self.model = CalibratedClassifierCV(base_model, cv=3, method='sigmoid')
        self.model.fit(X_train_scaled, y_train)
        
        # Store model metadata
        self.metadata['model_info'] = {
            'model_type': model_type,
            'n_features': len(self.feature_names),
            'training_samples': len(X_train),
            'training_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model trained: {model_type} with {len(self.feature_names)} features")
        return self.model
    
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, 
                          target_precision: float = 0.8) -> float:
        """Find optimal threshold for target precision"""
        X_val_scaled = self.scaler.transform(X_val)
        y_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
        
        # Find threshold closest to target precision
        idx = np.argmin(np.abs(precision[:-1] - target_precision))
        optimal_threshold = thresholds[idx]
        
        self.metadata['threshold_optimization'] = {
            'target_precision': target_precision,
            'optimal_threshold': float(optimal_threshold),
            'achieved_precision': float(precision[idx]),
            'achieved_recall': float(recall[idx])
        }
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} "
                   f"(Precision={precision[idx]:.3f}, Recall={recall[idx]:.3f})")
        
        return optimal_threshold
    
    def train_kfold(self, df: pd.DataFrame, n_folds: int = 5, 
                   model_type: str = 'xgboost') -> Dict[str, Any]:
        """Train and evaluate model using k-fold cross-validation"""
        if 'churned' not in df.columns:
            raise ValueError("Target column 'churned' not found in dataset")
        
        X = df.drop(columns=['churned'])
        y = df['churned']
        
        self.feature_names = X.columns.tolist()
        
        # Initialize k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Scale features
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)
            
            # Train model
            if model_type == 'xgboost':
                model_fold = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=2.76,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            else:
                model_fold = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )
            
            # Apply calibration
            model_fold = CalibratedClassifierCV(model_fold, cv=3, method='sigmoid')
            model_fold.fit(X_train_scaled, y_train_fold)
            
            # Evaluate
            y_proba = model_fold.predict_proba(X_val_scaled)[:, 1]
            roc_auc = roc_auc_score(y_val_fold, y_proba)
            
            precision, recall, _ = precision_recall_curve(y_val_fold, y_proba)
            pr_auc = auc(recall, precision)
            
            fold_results.append({
                'fold': fold,
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            
            models.append((model_fold, scaler_fold))
            
            logger.info(f"Fold {fold}/{n_folds}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        
        # Calculate aggregate metrics
        avg_roc_auc = np.mean([r['roc_auc'] for r in fold_results])
        std_roc_auc = np.std([r['roc_auc'] for r in fold_results])
        avg_pr_auc = np.mean([r['pr_auc'] for r in fold_results])
        std_pr_auc = np.std([r['pr_auc'] for r in fold_results])
        
        kfold_summary = {
            'n_folds': n_folds,
            'fold_results': fold_results,
            'avg_roc_auc': float(avg_roc_auc),
            'std_roc_auc': float(std_roc_auc),
            'avg_pr_auc': float(avg_pr_auc),
            'std_pr_auc': float(std_pr_auc),
            'model_type': model_type
        }
        
        # Save k-fold results
        kfold_dir = self.model_dir / 'kfold'
        kfold_dir.mkdir(exist_ok=True)
        
        with open(kfold_dir / 'kfold_results.json', 'w') as f:
            json.dump(kfold_summary, f, indent=2)
        
        # Save best model (based on ROC-AUC)
        best_fold_idx = np.argmax([r['roc_auc'] for r in fold_results])
        self.model, self.scaler = models[best_fold_idx]
        
        logger.info(f"K-fold complete: Avg ROC-AUC={avg_roc_auc:.4f} (±{std_roc_auc:.4f})")
        
        return kfold_summary
    
    def save_model(self, version: Optional[str] = None):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f'model_{version}.pkl'
        scaler_path = self.model_dir / f'scaler_{version}.pkl'
        metadata_path = self.model_dir / f'metadata_{version}.json'
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        self.metadata['version'] = version
        self.metadata['feature_names'] = self.feature_names
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Update latest symlinks
        for name, path in [('model_latest.pkl', model_path),
                           ('scaler_latest.pkl', scaler_path),
                           ('metadata_latest.json', metadata_path)]:
            latest_path = self.model_dir / name
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(path.name)
        
        logger.info(f"Model saved: {model_path}")
        
    def load_model(self, version: str = 'latest'):
        """Load a saved model"""
        if version == 'latest':
            model_path = self.model_dir / 'model_latest.pkl'
            scaler_path = self.model_dir / 'scaler_latest.pkl'
            metadata_path = self.model_dir / 'metadata_latest.json'
        else:
            model_path = self.model_dir / f'model_{version}.pkl'
            scaler_path = self.model_dir / f'scaler_{version}.pkl'
            metadata_path = self.model_dir / f'metadata_{version}.json'
        
        if not all(p.exists() for p in [model_path, scaler_path, metadata_path]):
            raise FileNotFoundError(f"Model files not found for version: {version}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata.get('feature_names', [])
        
        logger.info(f"Model loaded: {version}")
        return self.model

def main():
    """Main execution for testing"""
    from pipeline.data.data_processor import DataProcessor
    
    # Process data
    processor = DataProcessor()
    df = processor.load_latest()
    
    if df is None:
        print("Processing data first...")
        df, _ = processor.process('data/customer_churn.csv')
    
    # Train model
    trainer = ModelTrainer()
    splits = trainer.split_data(df)
    trainer.train_model(splits['X_train'], splits['y_train'])
    trainer.optimize_threshold(splits['X_test'], splits['y_test'])
    trainer.save_model()
    
    print("Training complete!")

if __name__ == "__main__":
    main()

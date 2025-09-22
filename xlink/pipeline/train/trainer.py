#!/usr/bin/env python3
"""
Model training module with train-test split and k-fold validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Model training class with versioning and persistence"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = Path(model_dir or "models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir = self.model_dir / "splits"
        self.splits_dir.mkdir(exist_ok=True)
        self.metadata_file = self.model_dir / "training_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load training metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "splits": {}}
    
    def _save_metadata(self):
        """Save training metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def create_splits(self, df: pd.DataFrame, test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        """Create train-test splits"""
        if 'churned' not in df.columns:
            raise ValueError("Target column 'churned' not found in dataframe")
        
        X = df.drop(columns=['churned'])
        y = df['churned']
        
        # Create train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Save splits
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_version = f"split_{timestamp}"
        
        splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        for name, data in splits.items():
            path = self.splits_dir / f"{split_version}_{name}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            # Also save as latest
            latest_path = self.splits_dir / f"latest_{name}.pkl"
            with open(latest_path, 'wb') as f:
                pickle.dump(data, f)
        
        # Update metadata
        self.metadata["splits"][split_version] = {
            "timestamp": timestamp,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": test_size,
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean())
        }
        self.metadata["latest_split"] = split_version
        self._save_metadata()
        
        print(f"Splits created: {split_version}")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def load_latest_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load latest train-test splits"""
        splits = {}
        for name in ['X_train', 'X_test', 'y_train', 'y_test']:
            path = self.splits_dir / f"latest_{name}.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}. Run create_splits first.")
            with open(path, 'rb') as f:
                splits[name] = pickle.load(f)
        
        return splits['X_train'], splits['X_test'], splits['y_train'], splits['y_test']
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_type: str = 'xgboost') -> Tuple[Any, StandardScaler, str]:
        """Train a model with specified algorithm"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize model
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    scale_pos_weight=2.76,  # Based on class imbalance
                    random_state=42,
                    eval_metric='logloss'
                )
            except ImportError:
                # Fallback to RandomForest if XGBoost not available
                print("Warning: XGBoost not available, using RandomForest")
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=42
                )
                model_type = 'random_forest'  # Update model type for versioning
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'logistic':
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_version = f"{model_type}_{timestamp}"
        
        model_path = self.model_dir / f"model_{model_version}.pkl"
        scaler_path = self.model_dir / f"scaler_{model_version}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save as latest
        with open(self.model_dir / f"model_latest.pkl", 'wb') as f:
            pickle.dump(model, f)
        with open(self.model_dir / f"scaler_latest.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Update metadata
        self.metadata["models"][model_version] = {
            "timestamp": timestamp,
            "type": model_type,
            "features": X_train.columns.tolist(),
            "train_size": len(X_train),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path)
        }
        self.metadata["latest_model"] = model_version
        self._save_metadata()
        
        print(f"Model trained: {model_version}")
        
        return model, scaler, model_version
    
    def load_latest_model(self) -> Tuple[Any, StandardScaler]:
        """Load latest trained model and scaler"""
        model_path = self.model_dir / "model_latest.pkl"
        scaler_path = self.model_dir / "scaler_latest.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError("Model or scaler not found. Train a model first.")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    
    def train_kfold(self, df: pd.DataFrame, model_type: str = 'random_forest', 
                   n_folds: int = 5) -> Dict[str, Any]:
        """Train model with k-fold cross-validation"""
        if 'churned' not in df.columns:
            raise ValueError("Target column 'churned' not found")
        
        X = df.drop(columns=['churned'])
        y = df['churned']
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        models = []
        scalers = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            model, scaler, _ = self.train_model(X_train, y_train, model_type)
            
            # Evaluate on validation set
            X_val_scaled = scaler.transform(X_val)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            fold_results.append({
                'fold': fold,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'val_size': len(y_val)
            })
            
            models.append(model)
            scalers.append(scaler)
            
            print(f"Fold {fold}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        
        # Calculate average metrics
        avg_metrics = {
            'avg_roc_auc': np.mean([r['roc_auc'] for r in fold_results]),
            'std_roc_auc': np.std([r['roc_auc'] for r in fold_results]),
            'avg_pr_auc': np.mean([r['pr_auc'] for r in fold_results]),
            'std_pr_auc': np.std([r['pr_auc'] for r in fold_results])
        }
        
        # Save k-fold results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kfold_version = f"kfold_{model_type}_{timestamp}"
        
        kfold_results = {
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'n_folds': n_folds,
            'model_type': model_type
        }
        
        results_path = self.model_dir / f"kfold_{kfold_version}.json"
        with open(results_path, 'w') as f:
            json.dump(kfold_results, f, indent=2)
        
        # Save best model (based on ROC-AUC)
        best_fold_idx = np.argmax([r['roc_auc'] for r in fold_results])
        best_model = models[best_fold_idx]
        best_scaler = scalers[best_fold_idx]
        
        with open(self.model_dir / f"model_kfold_best.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        with open(self.model_dir / f"scaler_kfold_best.pkl", 'wb') as f:
            pickle.dump(best_scaler, f)
        
        print(f"\nK-fold results saved: {kfold_version}")
        print(f"Average ROC-AUC: {avg_metrics['avg_roc_auc']:.4f} ± {avg_metrics['std_roc_auc']:.4f}")
        
        return kfold_results

def main():
    """Main execution for testing"""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data.processor import DataProcessor
    
    # Load processed data
    processor = DataProcessor()
    try:
        df = processor.load_latest()
        print(f"Loaded processed data: {df.shape}")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Create splits
        X_train, X_test, y_train, y_test = trainer.create_splits(df)
        
        # Train model
        model, scaler, version = trainer.train_model(X_train, y_train, model_type='xgboost')
        print(f"Model trained: {version}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data processing first.")

if __name__ == "__main__":
    main()

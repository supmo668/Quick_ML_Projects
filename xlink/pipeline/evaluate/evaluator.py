#!/usr/bin/env python3
"""
Model evaluation module with comprehensive metrics and segment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Model evaluation class with comprehensive metrics"""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = Path(results_dir or "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def evaluate_model(self, model: Any, scaler: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'threshold': threshold
        }
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def find_optimal_threshold(self, model: Any, scaler: Any, 
                             X_val: pd.DataFrame, y_val: pd.Series,
                             target_metric: str = 'f1') -> float:
        """Find optimal threshold based on target metric"""
        X_val_scaled = scaler.transform(X_val)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            if target_metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif target_metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif target_metric == 'recall':
                score = recall_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown target metric: {target_metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        return best_threshold
    
    def segment_analysis(self, model: Any, scaler: Any, 
                        X_test: pd.DataFrame, y_test: pd.Series,
                        original_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze model performance by segments"""
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        segments = {}
        
        # Analyze by prediction confidence
        segments['confidence'] = {
            'very_low': {'range': '0.0-0.2', 'indices': np.where(y_pred_proba < 0.2)[0]},
            'low': {'range': '0.2-0.4', 'indices': np.where((y_pred_proba >= 0.2) & (y_pred_proba < 0.4))[0]},
            'medium': {'range': '0.4-0.6', 'indices': np.where((y_pred_proba >= 0.4) & (y_pred_proba < 0.6))[0]},
            'high': {'range': '0.6-0.8', 'indices': np.where((y_pred_proba >= 0.6) & (y_pred_proba < 0.8))[0]},
            'very_high': {'range': '0.8-1.0', 'indices': np.where(y_pred_proba >= 0.8)[0]}
        }
        
        for segment_name, segment_data in segments['confidence'].items():
            indices = segment_data['indices']
            if len(indices) > 0:
                segment_data['count'] = len(indices)
                segment_data['actual_churn_rate'] = float(y_test.iloc[indices].mean())
                segment_data['predicted_churn_rate'] = float(y_pred_proba[indices].mean())
        
        return segments
    
    def create_evaluation_plots(self, model: Any, scaler: Any,
                              X_test: pd.DataFrame, y_test: pd.Series):
        """Create and save evaluation plots"""
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(self.plots_dir / 'roc_curve.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(self.plots_dir / 'pr_curve.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 4. Prediction Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='No Churn', color='blue')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Churn', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold')
        plt.savefig(self.plots_dir / 'prediction_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {self.plots_dir}")
    
    def save_results(self, metrics: Dict[str, Any], model_version: str = None):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = model_version or f"eval_{timestamp}"
        
        results = {
            'version': version,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        # Save JSON results
        results_path = self.results_dir / f"evaluation_{version}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as latest
        with open(self.results_dir / "evaluation_latest.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved: {results_path}")
        
        return version
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a formatted way"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC Score: {metrics['pr_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
        print(f"  FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")
        print("="*50)

def main():
    """Main execution for testing"""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from train.trainer import ModelTrainer
    
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    try:
        # Load model and splits
        model, scaler = trainer.load_latest_model()
        X_train, X_test, y_train, y_test = trainer.load_latest_splits()
        
        # Find optimal threshold
        optimal_threshold = evaluator.find_optimal_threshold(
            model, scaler, X_train, y_train, target_metric='f1'
        )
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, scaler, X_test, y_test, optimal_threshold)
        
        # Print metrics
        evaluator.print_metrics(metrics)
        
        # Create plots
        evaluator.create_evaluation_plots(model, scaler, X_test, y_test)
        
        # Save results
        evaluator.save_results(metrics)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first.")

if __name__ == "__main__":
    main()

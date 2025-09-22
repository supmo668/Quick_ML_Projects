#!/usr/bin/env python3
"""
Main pipeline orchestrator for customer churn prediction with Click CLI
"""

import click
from pathlib import Path
import sys
import json
from datetime import datetime

# Add pipeline modules to path
sys.path.append(str(Path(__file__).parent))

from data.processor import DataProcessor
from train.trainer import ModelTrainer
from evaluate.evaluator import ModelEvaluator

class ChurnPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
    def run_data_processing(self, input_path: str):
        """Run data processing step"""
        print("\n" + "="*50)
        print("STEP 1: DATA PROCESSING")
        print("="*50)
        
        df, version = self.processor.process(input_path)
        print(f"✓ Data processed successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Version: {version}")
        
        return df, version
    
    def run_train_test_split(self):
        """Run train-test split and model training"""
        print("\n" + "="*50)
        print("STEP 2: TRAIN-TEST SPLIT & MODELING")
        print("="*50)
        
        try:
            # Load latest processed data
            df = self.processor.load_latest()
            print(f"✓ Loaded processed data: {df.shape}")
            
            # Create splits
            X_train, X_test, y_train, y_test = self.trainer.create_splits(df)
            print(f"✓ Splits created")
            print(f"  Train: {len(X_train)} samples")
            print(f"  Test: {len(X_test)} samples")
            
            # Train model
            model, scaler, version = self.trainer.train_model(X_train, y_train, model_type='random_forest')
            print(f"✓ Model trained: {version}")
            
            return model, scaler, version
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print("  Please run data processing first (--process)")
            return None, None, None
    
    def run_evaluation(self):
        """Run model evaluation on test set"""
        print("\n" + "="*50)
        print("STEP 3: MODEL EVALUATION")
        print("="*50)
        
        try:
            # Load model and test data
            model, scaler = self.trainer.load_latest_model()
            X_train, X_test, y_train, y_test = self.trainer.load_latest_splits()
            
            print(f"✓ Model and test data loaded")
            
            # Find optimal threshold
            optimal_threshold = self.evaluator.find_optimal_threshold(
                model, scaler, X_train, y_train, target_metric='f1'
            )
            print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(
                model, scaler, X_test, y_test, optimal_threshold
            )
            
            # Print metrics
            self.evaluator.print_metrics(metrics)
            
            # Create plots
            self.evaluator.create_evaluation_plots(model, scaler, X_test, y_test)
            
            # Segment analysis
            segments = self.evaluator.segment_analysis(model, scaler, X_test, y_test)
            print(f"\n✓ Segment Analysis (Confidence Levels):")
            for level, data in segments['confidence'].items():
                if 'count' in data:
                    print(f"  {level}: n={data['count']}, actual_churn={data['actual_churn_rate']:.2%}")
            
            # Save results
            version = self.evaluator.save_results(metrics)
            print(f"✓ Results saved: {version}")
            
            return metrics
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print("  Please run training first (--train)")
            return None
    
    def run_kfold_evaluation(self):
        """Run k-fold cross-validation"""
        print("\n" + "="*50)
        print("STEP 4: K-FOLD CROSS-VALIDATION")
        print("="*50)
        
        try:
            # Load processed data
            df = self.processor.load_latest()
            print(f"✓ Loaded processed data: {df.shape}")
            
            # Run k-fold training
            results = self.trainer.train_kfold(df, model_type='random_forest', n_folds=5)
            
            print(f"\n✓ K-fold training complete")
            print(f"  Average ROC-AUC: {results['avg_metrics']['avg_roc_auc']:.4f}")
            print(f"  Std ROC-AUC: {results['avg_metrics']['std_roc_auc']:.4f}")
            print(f"  Average PR-AUC: {results['avg_metrics']['avg_pr_auc']:.4f}")
            print(f"  Std PR-AUC: {results['avg_metrics']['std_pr_auc']:.4f}")
            
            return results
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print("  Please run data processing first (--process)")
            return None
    
    def run_full_pipeline(self, input_path: str):
        """Run complete pipeline end-to-end"""
        print("\n" + "="*50)
        print("RUNNING COMPLETE PIPELINE")
        print("="*50)
        
        # Step 1: Process data
        df, data_version = self.run_data_processing(input_path)
        
        # Step 2: Train model
        model, scaler, model_version = self.run_train_test_split()
        if model is None:
            return
        
        # Step 3: Evaluate model
        metrics = self.run_evaluation()
        if metrics is None:
            return
        
        # Step 4: K-fold validation
        kfold_results = self.run_kfold_evaluation()
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETE")
        print("="*50)
        print(f"✓ Data version: {data_version}")
        print(f"✓ Model version: {model_version}")
        print(f"✓ Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"✓ K-fold ROC-AUC: {kfold_results['avg_metrics']['avg_roc_auc']:.4f}")

@click.group()
@click.pass_context
def cli(ctx):
    """Customer Churn Prediction Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['pipeline'] = ChurnPipeline()

@cli.command()
@click.option('--data', default='data/customer_churn.csv', help='Path to input data file')
@click.pass_context
def process(ctx, data):
    """Run data processing with versioning"""
    pipeline = ctx.obj['pipeline']
    df, version = pipeline.run_data_processing(data)

@cli.command()
@click.pass_context
def train(ctx):
    """Run train-test split and model training"""
    pipeline = ctx.obj['pipeline']
    model, scaler, version = pipeline.run_train_test_split()

@cli.command()
@click.pass_context
def evaluate(ctx):
    """Run model evaluation on test set"""
    pipeline = ctx.obj['pipeline']
    metrics = pipeline.run_evaluation()

@cli.command()
@click.option('--model-type', default='random_forest', 
              type=click.Choice(['logistic', 'random_forest']),
              help='Model type for k-fold validation')
@click.option('--n-folds', default=5, help='Number of folds')
@click.pass_context
def kfold(ctx, model_type, n_folds):
    """Run k-fold cross-validation"""
    pipeline = ctx.obj['pipeline']
    
    print("\n" + "="*50)
    print("K-FOLD CROSS-VALIDATION")
    print("="*50)
    
    try:
        # Load processed data
        df = pipeline.processor.load_latest()
        print(f"✓ Loaded processed data: {df.shape}")
        
        # Run k-fold training
        results = pipeline.trainer.train_kfold(df, model_type=model_type, n_folds=n_folds)
        
        print(f"\n✓ K-fold training complete")
        print(f"  Model Type: {model_type}")
        print(f"  Number of Folds: {n_folds}")
        print(f"  Average ROC-AUC: {results['avg_metrics']['avg_roc_auc']:.4f} ± {results['avg_metrics']['std_roc_auc']:.4f}")
        print(f"  Average PR-AUC: {results['avg_metrics']['avg_pr_auc']:.4f} ± {results['avg_metrics']['std_pr_auc']:.4f}")
        
        # Display individual fold results
        print(f"\n📊 Individual Fold Results:")
        for fold_result in results['fold_results']:
            print(f"  Fold {fold_result['fold']}: ROC-AUC={fold_result['roc_auc']:.4f}, PR-AUC={fold_result['pr_auc']:.4f}")
        
        return results
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run data processing first: python main.py process")
        return None

@cli.command()
@click.option('--data', default='data/customer_churn.csv', help='Path to input data file')
@click.pass_context
def all(ctx, data):
    """Run complete pipeline end-to-end"""
    pipeline = ctx.obj['pipeline']
    pipeline.run_full_pipeline(data)

@cli.command()
@click.pass_context
def status(ctx):
    """Show current pipeline status"""
    pipeline = ctx.obj['pipeline']
    
    print("\n📊 PIPELINE STATUS")
    print("="*50)
    
    # Check processed data
    try:
        df = pipeline.processor.load_latest()
        version_info = pipeline.processor.get_version_info()
        print(f"✅ Processed Data Available")
        print(f"   Shape: {df.shape}")
        print(f"   Latest Version: {version_info.get('latest', 'unknown')}")
    except FileNotFoundError:
        print("❌ No processed data found")
    
    # Check trained model
    try:
        model, scaler = pipeline.trainer.load_latest_model()
        metadata = pipeline.trainer.metadata
        print(f"✅ Trained Model Available")
        print(f"   Latest Version: {metadata.get('latest_model', 'unknown')}")
    except FileNotFoundError:
        print("❌ No trained model found")
    
    # Check evaluation results
    results_path = Path("results/evaluation_latest.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            eval_data = json.load(f)
        metrics = eval_data['metrics']
        print(f"✅ Evaluation Results Available")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    else:
        print("❌ No evaluation results found")

def main():
    """Main execution"""
    cli()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FastAPI service for customer churn prediction pipeline
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime
import io

# Add pipeline to path
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))
from data.processor import DataProcessor
from train.trainer import ModelTrainer
from evaluate.evaluator import ModelEvaluator

# Import Pydantic models
from api.models import (
    ProcessingRequest, ProcessingResponse,
    TrainingRequest, TrainingResponse,
    EvaluationResponse,
    KFoldRequest, KFoldResponse,
    PredictionRequest, PredictionResponse
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for data processing, model training, and churn prediction",
    version="1.0.0"
)

# Initialize pipeline components
processor = DataProcessor()
trainer = ModelTrainer()
evaluator = ModelEvaluator()

# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/process",
            "/train",
            "/evaluate",
            "/kfold",
            "/predict",
            "/status"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process", response_model=ProcessingResponse)
async def process_data(request: ProcessingRequest):
    """Process data with versioning"""
    try:
        # Check if input file exists
        if not Path(request.input_file).exists():
            raise HTTPException(status_code=404, detail=f"Input file not found: {request.input_file}")
        
        # Process data
        df, version = processor.process(request.input_file, request.version_name)
        
        return ProcessingResponse(
            status="success",
            version=version,
            shape=df.shape,
            columns=df.columns.tolist()[:20],  # Return first 20 columns
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train model with latest processed data"""
    try:
        # Load latest processed data
        df = processor.load_latest()
        
        # Create train-test splits
        X_train, X_test, y_train, y_test = trainer.create_splits(df, test_size=request.test_size)
        
        # Train model
        model, scaler, version = trainer.train_model(X_train, y_train, model_type=request.model_type)
        
        return TrainingResponse(
            status="success",
            model_version=version,
            train_size=len(X_train),
            test_size=len(X_test),
            timestamp=datetime.now().isoformat()
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No processed data found. Please run /process endpoint first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model():
    """Evaluate model on test set"""
    try:
        # Load model and test data
        model, scaler = trainer.load_latest_model()
        X_train, X_test, y_train, y_test = trainer.load_latest_splits()
        
        # Find optimal threshold
        optimal_threshold = evaluator.find_optimal_threshold(
            model, scaler, X_train, y_train, target_metric='f1'
        )
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, scaler, X_test, y_test, optimal_threshold)
        
        # Save results
        evaluator.save_results(metrics)
        
        return EvaluationResponse(
            status="success",
            metrics={
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "brier_score": metrics["brier_score"],
                "threshold": metrics["threshold"]
            },
            confusion_matrix={
                "true_negatives": metrics["true_negatives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "true_positives": metrics["true_positives"]
            },
            timestamp=datetime.now().isoformat()
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No model or test data found. Please run /train endpoint first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kfold", response_model=KFoldResponse)
async def kfold_evaluation(request: KFoldRequest):
    """Run k-fold cross-validation"""
    try:
        # Load processed data
        df = processor.load_latest()
        
        # Run k-fold training
        results = trainer.train_kfold(df, model_type=request.model_type, n_folds=request.n_folds)
        
        return KFoldResponse(
            status="success",
            avg_roc_auc=results["avg_metrics"]["avg_roc_auc"],
            std_roc_auc=results["avg_metrics"]["std_roc_auc"],
            avg_pr_auc=results["avg_metrics"]["avg_pr_auc"],
            std_pr_auc=results["avg_metrics"]["std_pr_auc"],
            n_folds=results["n_folds"]
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No processed data found. Please run /process endpoint first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Predict churn for a single customer"""
    try:
        # Load model and scaler
        model, scaler = trainer.load_latest_model()
        
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # Process features (same as training pipeline)
        df = processor.clean_data(df)
        df = processor.impute_missing(df)
        df = processor.engineer_features(df)
        df = processor.encode_features(df, fit=False)
        
        # Load training data to get correct column structure
        try:
            training_data = processor.load_latest()
            training_columns = training_data.columns.drop('churned').tolist()
            
            # Add missing columns with default values
            for col in training_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df.reindex(columns=training_columns, fill_value=0)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not align features with training data: {str(e)}")
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Predict
        probability = model.predict_proba(X_scaled)[0, 1]
        threshold = 0.5  # Default threshold
        prediction = int(probability >= threshold)
        
        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction=prediction,
            threshold=threshold,
            timestamp=datetime.now().isoformat()
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No model found. Please train a model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current pipeline status"""
    status = {
        "timestamp": datetime.now().isoformat(),
        "data": {},
        "model": {},
        "evaluation": {}
    }
    
    # Check for processed data
    try:
        df = processor.load_latest()
        version_info = processor.get_version_info()
        status["data"] = {
            "available": True,
            "latest_version": version_info.get("latest", "unknown"),
            "shape": df.shape
        }
    except:
        status["data"]["available"] = False
    
    # Check for trained model
    try:
        model, scaler = trainer.load_latest_model()
        metadata = trainer.metadata
        status["model"] = {
            "available": True,
            "latest_version": metadata.get("latest_model", "unknown")
        }
    except:
        status["model"]["available"] = False
    
    # Check for evaluation results
    results_path = Path("results/evaluation_latest.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            eval_data = json.load(f)
        status["evaluation"] = {
            "available": True,
            "roc_auc": eval_data["metrics"]["roc_auc"],
            "timestamp": eval_data["timestamp"]
        }
    else:
        status["evaluation"]["available"] = False
    
    return status

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file for processing"""
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Save to data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_path, index=False)
        
        # Process the uploaded file
        processed_df, version = processor.process(str(file_path))
        
        return {
            "status": "success",
            "file_path": str(file_path),
            "shape": df.shape,
            "processed_version": version
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

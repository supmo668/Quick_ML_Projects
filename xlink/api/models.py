#!/usr/bin/env python3
"""
Pydantic models for Customer Churn Prediction API
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class ProcessingRequest(BaseModel):
    input_file: str = Field(default="data/customer_churn.csv", description="Path to input CSV file")
    version_name: Optional[str] = Field(default=None, description="Custom version name")


class ProcessingResponse(BaseModel):
    status: str
    version: str
    shape: tuple
    columns: List[str]
    timestamp: str


class TrainingRequest(BaseModel):
    model_type: str = Field(default="xgboost", description="Model type: xgboost, random_forest, or logistic")
    test_size: float = Field(default=0.2, description="Test set size ratio")


class TrainingResponse(BaseModel):
    status: str
    model_version: str
    train_size: int
    test_size: int
    timestamp: str


class EvaluationResponse(BaseModel):
    status: str
    metrics: Dict[str, float]
    confusion_matrix: Dict[str, int]
    timestamp: str


class KFoldRequest(BaseModel):
    model_type: str = Field(default="xgboost", description="Model type")
    n_folds: int = Field(default=5, description="Number of folds")


class KFoldResponse(BaseModel):
    status: str
    avg_roc_auc: float
    std_roc_auc: float
    avg_pr_auc: float
    std_pr_auc: float
    n_folds: int


class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    return_probability: bool = Field(default=True, description="Return probability instead of binary prediction")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold: float
    timestamp: str

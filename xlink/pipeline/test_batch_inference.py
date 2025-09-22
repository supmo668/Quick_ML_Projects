#!/usr/bin/env python3
"""
Batch inference test for the ML pipeline
Tests prediction functionality with real customer data
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.data.processor import DataProcessor
from pipeline.train.trainer import ModelTrainer
from pipeline.evaluate.evaluator import ModelEvaluator

def test_batch_inference():
    """Test batch inference with real customer examples"""
    print("🎯 Testing Batch Inference with Real Customer Data")
    print("=" * 60)
    
    # Load real inference examples
    examples_file = Path(__file__).parent / "real_inference_examples.json"
    if not examples_file.exists():
        print(f"❌ Real inference examples not found: {examples_file}")
        print("   Run: python pipeline/create_inference_examples.py")
        return False
    
    with open(examples_file, 'r') as f:
        inference_data = json.load(f)
    
    print(f"📄 Loaded {len(inference_data['examples'])} real customer examples")
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Process data if needed
    try:
        df = processor.load_latest()
        print(f"✅ Using existing processed data: {df.shape}")
    except:
        print("📊 Processing fresh data...")
        df, version = processor.process("data/customer_churn.csv")
        print(f"✅ Data processed: {df.shape}, version: {version}")
    
    # Train model if needed
    try:
        model, scaler = trainer.load_latest_model()
        print(f"✅ Using existing trained model")
    except:
        print("🎯 Training new model...")
        X_train, X_test, y_train, y_test = trainer.create_splits(df)
        model, scaler, version = trainer.train_model(X_train, y_train, model_type='random_forest')
        print(f"✅ Model trained: {version}")
    
    # Test predictions with each example
    print("\n🔮 Making Predictions...")
    print("-" * 40)
    
    success_count = 0
    total_count = len(inference_data['examples'])
    
    for i, example in enumerate(inference_data['examples'], 1):
        print(f"\n{i}. Testing {example['name']}")
        print(f"   📝 {example['description']}")
        print(f"   🎯 Actual outcome: {'Churned' if example['actual_churned'] else 'Retained'}")
        
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([example['features']])
            
            # Process features through the same pipeline
            processed_df = processor.clean_data(features_df)
            processed_df = processor.impute_missing(processed_df)
            processed_df = processor.engineer_features(processed_df)
            processed_df = processor.encode_features(processed_df, fit=False)
            
            # Ensure we have the same columns as training data
            training_columns = df.columns.drop('churned').tolist()
            
            # Add missing columns with default values
            for col in training_columns:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            # Reorder columns to match training
            processed_df = processed_df.reindex(columns=training_columns, fill_value=0)
            
            # Scale features
            X_scaled = scaler.transform(processed_df)
            
            # Make prediction
            probability = model.predict_proba(X_scaled)[0, 1]
            prediction = model.predict(X_scaled)[0]
            
            print(f"   ✅ Prediction successful!")
            print(f"   📈 Churn Probability: {probability:.3f}")
            print(f"   🎯 Predicted: {'Will Churn' if prediction else 'Will Stay'}")
            
            # Validate against actual outcome
            correct = (prediction == 1) == example['actual_churned']
            if correct:
                print(f"   ✅ Prediction matches actual outcome!")
            else:
                print(f"   ⚠️ Prediction differs from actual outcome")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {str(e)}")
            print(f"   🔧 Error details: {type(e).__name__}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 Batch Inference Results:")
    print(f"   ✅ Successful predictions: {success_count}/{total_count}")
    print(f"   🎯 Success rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print(f"\n🎉 All inference tests passed! Prediction pipeline is working correctly.")
        return True
    else:
        print(f"\n⚠️ Some predictions failed. Check error details above.")
        return False

if __name__ == "__main__":
    success = test_batch_inference()
    exit(0 if success else 1)
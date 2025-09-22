#!/usr/bin/env python3
"""
Comprehensive API testing for Customer Churn Prediction service
Tests both local FastAPI client and external HTTP requests
"""

import json
import subprocess
import time
from pathlib import Path

def run_curl_command(method, endpoint, data=None):
    """Run a curl command and return the result"""
    url = f"http://localhost:8000{endpoint}"
    
    if method == "GET":
        cmd = ["curl", "-s", url]
    elif method == "POST":
        cmd = ["curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json"]
        if data:
            cmd.extend(["-d", json.dumps(data)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            try:
                return True, json.loads(result.stdout)
            except json.JSONDecodeError:
                return True, {"raw_response": result.stdout}
        else:
            return False, {"error": result.stderr}
    except subprocess.TimeoutExpired:
        return False, {"error": "Request timeout"}
    except Exception as e:
        return False, {"error": str(e)}

def test_api_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint via HTTP"""
    print(f"\n🔍 Testing {method} {endpoint} - {description}")
    success, response = run_curl_command(method, endpoint, data)
    
    if success:
        print(f"   ✅ Success")
        if "status" in response:
            print(f"   📊 Status: {response['status']}")
        if "shape" in response:
            print(f"   📏 Data Shape: {response['shape']}")
        if "roc_auc" in response.get("metrics", {}):
            print(f"   📈 ROC-AUC: {response['metrics']['roc_auc']:.4f}")
        return True, response
    else:
        print(f"   ❌ Failed: {response.get('error', 'Unknown error')}")
        return False, response

def test_core_api_functionality():
    """Test core API functionality"""
    print("🌐 Testing Core API Functionality")
    print("=" * 60)
    
    # Test basic endpoints
    tests = [
        ("GET", "/health", None, "Health check"),
        ("GET", "/", None, "Root endpoint"),
        ("GET", "/status", None, "Initial status"),
        ("POST", "/process", {"input_file": "data/customer_churn.csv"}, "Data processing"),
        ("POST", "/train", {"model_type": "random_forest"}, "Model training"),
        ("POST", "/evaluate", None, "Model evaluation"),
        ("POST", "/kfold", {"model_type": "random_forest", "n_folds": 3}, "K-fold validation"),
        ("GET", "/status", None, "Final status"),
    ]
    
    results = []
    
    for method, endpoint, data, description in tests:
        success, response = test_api_endpoint(method, endpoint, data, description)
        results.append((description, success, response))
        time.sleep(1)  # Brief pause between requests
    
    return results

def test_prediction_functionality():
    """Test prediction endpoint with real customer data"""
    print(f"\n🎯 Testing Prediction Functionality")
    print("-" * 50)
    
    # Load real inference examples from pipeline
    inference_file = Path(__file__).parent.parent.parent / "pipeline" / "real_inference_examples.json"
    if not inference_file.exists():
        print(f"   ❌ Real inference examples not found: {inference_file}")
        print(f"   💡 Run: python pipeline/create_inference_examples.py")
        return [False]
    
    with open(inference_file, 'r') as f:
        inference_data = json.load(f)
    
    print(f"📄 Loaded {len(inference_data['examples'])} real customer examples")
    
    # Test predictions with each example
    prediction_results = []
    
    for i, example in enumerate(inference_data["examples"], 1):
        print(f"\n{i}. Testing {example['name']}")
        print(f"   📝 {example['description']}")
        print(f"   🎯 Actual outcome: {'Churned' if example['actual_churned'] else 'Retained'}")
        
        success, response = run_curl_command("POST", "/predict", {
            "features": example["features"],
            "return_probability": True
        })
        
        if success and "churn_probability" in response:
            probability = response["churn_probability"]
            prediction = response["churn_prediction"]
            threshold = response["threshold"]
            
            print(f"   ✅ Prediction successful!")
            print(f"   📈 Churn Probability: {probability:.3f}")
            print(f"   🎯 Predicted: {'Will Churn' if prediction else 'Will Stay'}")
            print(f"   🔍 Threshold: {threshold:.3f}")
            
            # Validate prediction makes sense
            actual_churned = example['actual_churned']
            predicted_churn = prediction == 1
            
            if actual_churned == predicted_churn:
                print(f"   ✅ Prediction matches actual outcome!")
            else:
                print(f"   ⚠️ Prediction differs from actual outcome")
            
            if example.get('expected_churn') is not None:
                if example['expected_churn'] and probability > 0.6:
                    print(f"   ✅ High-risk customer correctly identified")
                elif not example['expected_churn'] and probability < 0.4:
                    print(f"   ✅ Low-risk customer correctly identified")
            
            prediction_results.append(True)
        else:
            print(f"   ❌ Prediction failed")
            error_msg = response.get("detail", response.get("error", "Unknown"))
            print(f"   📝 Error: {str(error_msg)[:100]}...")
            prediction_results.append(False)
    
    return prediction_results

def wait_for_api_ready(timeout=30):
    """Wait for API to be ready"""
    print("⏳ Waiting for API to be ready...")
    
    for i in range(timeout):
        try:
            success, response = run_curl_command("GET", "/health")
            if success and response.get("status") == "healthy":
                print("✅ API is ready!")
                return True
        except:
            pass
        time.sleep(1)
    
    print("❌ API not responding. Make sure the service is running on localhost:8000")
    return False

def main():
    """Run comprehensive API tests"""
    print("🚀 Customer Churn Prediction API Test Suite")
    print("=" * 70)
    
    # Check if API is ready
    if not wait_for_api_ready():
        print("\n💡 To start the API:")
        print("   Local: ./scripts/run_dev.sh")
        print("   Docker: docker run -d -p 8000:8000 --name churn-api churn-prediction-api")
        return False
    
    # Test core functionality
    core_results = test_core_api_functionality()
    
    # Test prediction functionality
    prediction_results = test_prediction_functionality()
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"📊 Test Summary:")
    
    # Core API tests
    successful_core = sum(1 for _, success, _ in core_results if success)
    total_core = len(core_results)
    print(f"   🌐 Core API: {successful_core}/{total_core} tests passed")
    
    # Prediction tests
    successful_pred = sum(prediction_results) if prediction_results else 0
    total_pred = len(prediction_results) if prediction_results else 0
    print(f"   🎯 Predictions: {successful_pred}/{total_pred} successful")
    
    overall_success = (successful_core == total_core)
    
    if overall_success:
        print(f"\n🎉 All core API tests passed! Service is fully functional.")
    else:
        print(f"\n⚠️ Some tests failed. Check output above for details.")
    
    print(f"\n💡 API Information:")
    print(f"   🌐 API URL: http://localhost:8000")
    print(f"   📖 Documentation: http://localhost:8000/docs")
    print(f"   ❤️ Health Check: http://localhost:8000/health")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
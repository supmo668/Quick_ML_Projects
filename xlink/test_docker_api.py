#!/usr/bin/env python3
"""
Test script to verify Docker API functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing {method} {endpoint} - {description}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  ✅ Success")
            return True
        else:
            print(f"  ❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run comprehensive API tests"""
    print("🐳 Testing Docker API Container")
    print("=" * 50)
    
    # Wait for container to be ready
    print("Waiting for container to be ready...")
    time.sleep(5)
    
    tests = [
        ("GET", "/health", None, "Health check"),
        ("GET", "/", None, "Root endpoint"),
        ("GET", "/status", None, "Pipeline status"),
        ("POST", "/train", {"model_type": "random_forest"}, "Model training"),
        ("POST", "/evaluate", None, "Model evaluation"),
        ("POST", "/kfold", {"model_type": "random_forest", "n_folds": 3}, "K-fold validation"),
    ]
    
    passed = 0
    total = len(tests)
    
    for method, endpoint, data, description in tests:
        if test_endpoint(method, endpoint, data, description):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker API is fully functional.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
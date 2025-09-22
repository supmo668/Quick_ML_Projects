#!/bin/bash

# FastAPI Development Server Startup Script
# Customer Churn Prediction API

set -e

echo "🚀 Starting Customer Churn Prediction API Development Server..."

# Change to project root directory
cd "$(dirname "$0")/.."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please create one first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Check if required data file exists
if [ ! -f "data/customer_churn.csv" ]; then
    echo "⚠️  Warning: customer_churn.csv not found in data/ directory"
    echo "   Some API endpoints may not work without the data file"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models results data/processed

# Start the FastAPI server
echo "🌟 Starting FastAPI server on http://localhost:8000"
echo "📖 API Documentation available at http://localhost:8000/docs"
echo "🔄 Server will auto-reload on code changes"
echo ""
echo "Press Ctrl+C to stop the server"
echo "----------------------------------------"

python -m uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir api \
    --reload-dir pipeline

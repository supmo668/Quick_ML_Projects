#!/bin/bash

# Setup script for data exploration environment using uv

echo "🔧 Setting up data exploration environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    pip install uv
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "✨ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies from pyproject.toml..."
uv pip install -e .

# Install development dependencies (optional)
echo "🛠️ Installing development dependencies..."
uv pip install -e ".[dev]"

# Generate sample data
echo "📊 Generating sample data..."
python generate_sample_data.py

# Run initial exploration
echo "🔍 Running initial data exploration..."
python explore_data.py --data sample_data.csv

echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run data exploration:"
echo "  python explore_data.py --data your_data.csv"

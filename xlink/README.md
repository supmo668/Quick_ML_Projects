# Customer Churn Prediction Project

A production-ready machine learning pipeline for predicting customer churn with FastAPI service and Docker deployment.

## 🎯 Project Overview

This project implements a complete end-to-end ML pipeline for customer churn prediction, featuring:

- **Data Processing**: Automated cleaning, feature engineering, and encoding
- **Model Training**: Random Forest with cross-validation and hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, PR-AUC, and confusion matrix
- **API Serving**: FastAPI-based REST API with automatic documentation
- **Pipeline CLI**: Command-line interface for running ML pipeline steps
- **Docker Deployment**: Production-ready containerized deployment

## 📋 Prerequisites

- **Python**: 3.9 or higher
- **Package Manager**: `uv` (recommended) or `pip`
- **Docker**: For containerized deployment (optional)
- **System**: Linux/macOS/Windows with WSL2

## 🛠 Environment Setup

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):
```bash
pip install uv
```

2. **Clone and setup the project**:
```bash
git clone <repository-url>
cd xlink
```

3. **Create virtual environment and install dependencies**:
```bash
# This creates .venv/ and installs all dependencies from pyproject.toml
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

4. **Verify installation**:
```bash
python pipeline/main.py --help
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install -e ".[dev,test]"  # Include dev and test dependencies
```

## 🚀 Quick Start

### Option A: Docker Deployment (Recommended for Production)

1. **Build the Docker image**:
```bash
docker build -t churn-prediction-api .
```

2. **Run the container**:
```bash
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
```

3. **Access the API**:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Option B: Local Development Server

1. **Setup environment** (see Environment Setup above)

2. **Start the development server**:
```bash
./scripts/run_dev.sh
```

3. **Access the API**:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs

## 🔧 CLI Pipeline Usage

The project includes a comprehensive CLI for running the ML pipeline step-by-step or end-to-end.

### Available Commands

```bash
# Show all available commands
python pipeline/main.py --help
```

### Step-by-Step Pipeline Execution

1. **Check pipeline status**:
```bash
python pipeline/main.py status
```

2. **Process data** (cleaning, feature engineering):
```bash
python pipeline/main.py process --data data/customer_churn.csv
```

3. **Train model**:
```bash
python pipeline/main.py train
```

4. **Evaluate model**:
```bash
python pipeline/main.py evaluate
```

5. **Run k-fold cross-validation**:
```bash
python pipeline/main.py kfold --model-type random_forest --n-folds 5
```

### End-to-End Pipeline

Run the complete pipeline in one command:
```bash
python pipeline/main.py all --data data/customer_churn.csv
```

## 📁 Detailed Project Structure

```
xlink/
├── api/                          # FastAPI application
│   ├── app.py                   # Main FastAPI application with endpoints
│   ├── models.py                # Pydantic data models for API
│   └── Dockerfile               # API-specific Dockerfile
├── pipeline/                     # ML pipeline modules
│   ├── data/                    # Data processing components
│   │   ├── __init__.py
│   │   └── processor.py         # Data cleaning and feature engineering
│   ├── train/                   # Model training components  
│   │   ├── __init__.py
│   │   └── trainer.py           # Model training and validation
│   ├── evaluate/                # Model evaluation components
│   │   ├── __init__.py
│   │   └── evaluator.py         # Metrics calculation and visualization
│   └── main.py                  # CLI interface and pipeline orchestrator
├── tests/                       # Test suite
│   ├── api/                     # API endpoint tests
│   ├── pipeline/                # Pipeline component tests
│   └── conftest.py              # Test configuration and fixtures
├── data/                        # Data directory
│   ├── customer_churn.csv       # Raw input data
│   └── processed/               # Processed data versions
├── models/                      # Saved models and metadata
│   └── splits/                  # Train/test split data
├── results/                     # Evaluation results and plots
│   └── plots/                   # Generated visualizations
├── scripts/                     # Utility scripts
│   └── run_dev.sh              # Development server startup script
├── data-exploration/            # Data exploration toolkit
│   ├── explore_data.py         # Data analysis and visualization
│   ├── setup_env.sh            # Environment setup script
│   └── README.md               # Data exploration documentation
├── pyproject.toml              # Project dependencies and configuration
├── Dockerfile                  # Production Docker configuration
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## 🧪 Testing

### Prerequisites for Testing

Ensure your environment is set up:
```bash
# Install test dependencies
uv sync --extra test
# or with pip
pip install -e ".[test]"
```

### Running Tests

1. **Run all tests**:
```bash
pytest tests/ -v
```

2. **Run API tests only**:
```bash
pytest tests/api/ -v
```

3. **Run pipeline tests only**:
```bash
pytest tests/pipeline/ -v
```

4. **Run tests with coverage**:
```bash
pytest tests/ --cov=pipeline --cov=api --cov-report=html
```

5. **Test specific functionality**:
```bash
# Test API endpoints
python tests/api/test_api_external.py

# Test data processing
pytest tests/pipeline/test_processor.py -v
```

## 🔧 Development Workflow

### Local Development Setup

1. **Environment setup**:
```bash
# Clone repository
git clone <repository-url>
cd xlink

# Setup environment with dev dependencies
uv sync --extra dev --extra test
source .venv/bin/activate
```

2. **Start development server**:
```bash
./scripts/run_dev.sh
```

3. **Code formatting and linting** (if dev dependencies installed):
```bash
# Format code
black pipeline/ api/ tests/

# Lint code
pylint pipeline/ api/
```

### Development Environment Variables

The project supports these environment variables:
- `PYTHONPATH`: Automatically set to project root
- `BUILD_ENV`: Set to "development" or "production" 
- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Project Configuration

Key configuration files:
- `pyproject.toml`: Dependencies, project metadata, and tool configurations
- `pytest.ini`: Test configuration and options
- `.gitignore`: Git ignore patterns
- `Dockerfile`: Production container configuration

## 📊 API Endpoints

The FastAPI service provides the following endpoints:

| Endpoint | Method | Description | Example Usage |
|----------|--------|-------------|---------------|
| `/` | GET | API information and available endpoints | `curl http://localhost:8000/` |
| `/health` | GET | Health check for monitoring | `curl http://localhost:8000/health` |
| `/status` | GET | Current pipeline status | `curl http://localhost:8000/status` |
| `/process` | POST | Process uploaded data | `curl -X POST -F "file=@data.csv" http://localhost:8000/process` |
| `/train` | POST | Train ML model | `curl -X POST http://localhost:8000/train` |
| `/evaluate` | POST | Evaluate trained model | `curl -X POST http://localhost:8000/evaluate` |
| `/predict` | POST | Make predictions | `curl -X POST -H "Content-Type: application/json" -d '{"features": [...]}' http://localhost:8000/predict` |
| `/kfold` | POST | Run cross-validation | `curl -X POST -H "Content-Type: application/json" -d '{"model_type": "random_forest", "n_folds": 5}' http://localhost:8000/kfold` |
| `/upload` | POST | Upload CSV file | `curl -X POST -F "file=@data.csv" http://localhost:8000/upload` |

### Interactive API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker Deployment Guide

### Building Docker Images

1. **Production build**:
```bash
docker build -t churn-prediction-api .
```

2. **Development build** (with auto-reload):
```bash
docker build --build-arg BUILD_ENV=development -t churn-api-dev .
```

### Running Containers

1. **Run production container**:
```bash
# Run in detached mode
docker run -d -p 8000:8000 --name churn-api churn-prediction-api

# Run with logs visible
docker run -p 8000:8000 --name churn-api churn-prediction-api
```

2. **Run development container**:
```bash
docker run -d -p 8000:8000 --name churn-api-dev churn-api-dev
```

### Container Management

```bash
# View container logs
docker logs churn-api

# Follow logs in real-time
docker logs -f churn-api

# Stop and remove container
docker stop churn-api && docker rm churn-api

# Execute commands inside container
docker exec -it churn-api /bin/bash

# View container resource usage
docker stats churn-api
```

### Docker Health Checks

The container includes health checks that monitor the API:
```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' churn-api

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' churn-api
```

## 📈 Model Performance

Current model performance metrics:

- **ROC-AUC**: 0.8425 (excellent discrimination)
- **PR-AUC**: 0.6489 (good precision-recall balance)  
- **Accuracy**: 77.8%
- **Dataset**: 7,148 customer records
- **Features**: 46 engineered features
- **Model**: Random Forest with cross-validation

### Performance Characteristics

- **Training time**: ~30 seconds on standard hardware
- **Prediction latency**: <100ms per request
- **Memory usage**: ~200MB for model serving
- **Throughput**: >100 predictions/second

## 🔧 Development

### Project Setup
```bash
# Install dependencies with development tools
uv sync --extra dev

# Run development server with auto-reload
./scripts/run_dev.sh
```

### Environment Variables
- `PYTHONPATH`: Automatically set to project root
- `BUILD_ENV`: Set to "development" or "production"

## 🐳 Docker

### Build Options
```bash
# Production build
docker build -t churn-prediction-api .

# Development build
docker build --build-arg BUILD_ENV=development -t churn-api-dev .
```

### Container Management
```bash
# Run container
docker run -d -p 8000:8000 --name churn-api churn-prediction-api

# View logs
docker logs churn-api

# Stop container
docker stop churn-api && docker rm churn-api
```

## 🚨 Troubleshooting

### Common Issues

1. **Module Import Errors**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Verify PYTHONPATH is set correctly
echo $PYTHONPATH

# Reinstall in development mode
pip install -e .
```

2. **Missing Data File**:
```bash
# Check if customer_churn.csv exists in data/ directory
ls data/customer_churn.csv

# Download sample data or use your own dataset
# Ensure CSV has the expected schema (see data-exploration/README.md)
```

3. **Docker Build Failures**:
```bash
# Clear Docker build cache
docker builder prune

# Rebuild without cache
docker build --no-cache -t churn-prediction-api .

# Check Docker daemon is running
docker info
```

4. **API Server Won't Start**:
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use different port
uvicorn api.app:app --host 0.0.0.0 --port 8001

# Check logs for specific error
python api/app.py
```

5. **Pipeline Status Shows Missing Components**:
```bash
# Run status to see what's missing
python pipeline/main.py status

# Process data first
python pipeline/main.py process --data data/customer_churn.csv

# Train model
python pipeline/main.py train
```

### Getting Help

- Check the interactive API documentation: http://localhost:8000/docs
- Review the RATIONALE.md for architectural decisions
- Check DOCKER_TEST_SUMMARY.md for Docker testing examples
- Examine the data-exploration/ directory for data requirements

## 📚 Documentation

- **[TASK.md](TASK.md)** - Original requirements
- **[RATIONALE.md](RATIONALE.md)** - Design decisions and architecture
- **[DOCKER_TEST_SUMMARY.md](DOCKER_TEST_SUMMARY.md)** - Docker testing results
- **[data-exploration/README.md](data-exploration/README.md)** - Data exploration toolkit

## 🎯 Key Features

- **Production Ready**: Docker deployment, health checks, logging
- **Comprehensive Testing**: Unit, integration, and API tests
- **Model Versioning**: Automatic versioning and metadata tracking
- **Real-time Predictions**: Sub-second response times
- **Interactive Docs**: Auto-generated API documentation
- **Monitoring Ready**: Structured logging and metrics

## 🔬 ML Pipeline

1. **Data Processing**: Cleaning, feature engineering, encoding
2. **Model Training**: Random Forest with cross-validation
3. **Evaluation**: Multiple metrics, calibration analysis
4. **Serving**: FastAPI with async capabilities

## 🛠 Technology Stack

- **ML**: scikit-learn, pandas, numpy, xgboost
- **API**: FastAPI, uvicorn, pydantic  
- **Testing**: pytest, httpx
- **Deployment**: Docker, multi-stage builds
- **Package Management**: uv, pyproject.toml
- **CLI**: Click for command-line interface

---

## 🚀 Quick Reference

### Essential Commands

```bash
# Setup environment
uv sync && source .venv/bin/activate

# Run full pipeline
python pipeline/main.py all --data data/customer_churn.csv

# Start API server
./scripts/run_dev.sh

# Run tests
pytest tests/ -v

# Docker deployment
docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

### API Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Get API info
curl http://localhost:8000/

# Check pipeline status
curl http://localhost:8000/status
```
# Customer Churn Prediction Project

A production-ready machine learning pipeline for predicting customer churn with FastAPI service and Docker deployment.

## 🎯 Project Overview

This project implements a complete ML pipeline for customer churn prediction, including data processing, model training, evaluation, and serving via a REST API.

## 🚀 Quick Start

### Local Development
```bash
# Start the development server
./scripts/run_dev.sh

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Docker Deployment
```bash
# Build and run the Docker container
docker build -t churn-prediction-api .
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
```

## 📁 Repository Structure

```
├── api/                     # FastAPI application
│   ├── app.py              # Main FastAPI application
│   ├── models.py           # Pydantic data models
│   └── Dockerfile          # API-specific Dockerfile
├── pipeline/               # ML pipeline modules
│   ├── data/               # Data processing
│   ├── train/              # Model training
│   ├── evaluate/           # Model evaluation
│   └── main.py            # CLI interface
├── tests/                  # Test suite
│   ├── api/               # API tests
│   ├── pipeline/          # Pipeline tests
│   └── conftest.py        # Test configuration
├── data/                  # Data files
├── models/                # Saved models
├── results/               # Evaluation results
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test API Endpoints
```bash
# Test local FastAPI client
pytest tests/api/test_api.py -v

# Test external HTTP requests
python tests/api/test_api_external.py
```

### Test ML Pipeline
```bash
pytest tests/pipeline/ -v
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Pipeline status |
| `/process` | POST | Process data |
| `/train` | POST | Train model |
| `/evaluate` | POST | Evaluate model |
| `/predict` | POST | Make prediction |
| `/kfold` | POST | Cross-validation |
| `/upload` | POST | Upload data file |

## 🔧 Development

### Project Setup
```bash
# Install dependencies
uv pip install -e .

# Run development server
./scripts/run_dev.sh
```

### Environment Variables
- `PYTHONPATH`: Automatically set to project root
- `BUILD_ENV`: Set to "development" or "production"

## 📈 Model Performance

- **ROC-AUC**: 0.8425 (excellent discrimination)
- **PR-AUC**: 0.6489 (good precision-recall balance)
- **Accuracy**: 77.8%
- **Dataset**: 7,148 customer records

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

## 📚 Documentation

- **[TASK.md](TASK.md)** - Original requirements
- **[RATIONALE.md](RATIONALE.md)** - Design decisions and architecture
- **[DOCKER_TEST_SUMMARY.md](DOCKER_TEST_SUMMARY.md)** - Docker testing results

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

- **ML**: scikit-learn, pandas, numpy
- **API**: FastAPI, uvicorn, pydantic
- **Testing**: pytest, httpx
- **Deployment**: Docker, multi-stage builds
- **Package Management**: uv, pyproject.toml
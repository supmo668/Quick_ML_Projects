# Customer Churn Prediction Project

This repository presents a production-grade machine learning pipeline for customer churn prediction, with a FastAPI service for real-time inference and containerized deployment. The emphasis is on reproducibility, methodological clarity, and operational readiness.

## 🎯 Project Overview

This project implements a comprehensive end-to-end pipeline for customer churn prediction, featuring:

- **Data Processing**: Automated cleaning, feature engineering, and encoding
- **Model Training**: Random Forest with cross-validation and hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, PR-AUC, and confusion matrix
- **API Serving**: FastAPI-based REST API with automatic documentation
- **Pipeline CLI**: Command-line interface for running ML pipeline steps
- **Docker Deployment**: Production-ready containerized deployment

## 📋 Prerequisites

- Python 3.9+
- Package manager: `uv` (recommended) or `pip`
- Docker (optional, for deployment)

## 🛠 Environment Setup

Using uv (recommended):
```bash
pip install uv
git clone <repository-url>
cd xlink
uv sync && source .venv/bin/activate
python pipeline/main.py --help
```

## 🚀 Quick Start

### Option A: Docker (recommended for production)
```bash
docker build -t churn-prediction-api .
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
# API: http://localhost:8000 | Docs: http://localhost:8000/docs | Health: /health
```

### Option B: Local (development)
```bash
./scripts/run_dev.sh
# API: http://localhost:8000 | Docs: http://localhost:8000/docs
```

## 🔧 CLI Pipeline Usage

Key commands:
```bash
python pipeline/main.py status
python pipeline/main.py process --data data/customer_churn.csv
python pipeline/main.py train && python pipeline/main.py evaluate
python pipeline/main.py kfold --model-type random_forest --n-folds 5
python pipeline/main.py all --data data/customer_churn.csv
```

## 📁 Project Structure (abridged)

- api/ (FastAPI app and Dockerfile)
- pipeline/ (data → train → evaluate modules and CLI)
- tests/ (API + pipeline tests)
- data/, models/, results/ (artifacts)
- report/ (final report and curated artifacts)

## 🧪 Testing

### Running Tests
```bash
uv sync --extra test && pytest tests/ -v
pytest tests/api/ -v     # API tests only
pytest tests/pipeline/ -v
```

## 🔧 Development Workflow

### Local Development
```bash
uv sync --extra dev --extra test && source .venv/bin/activate
./scripts/run_dev.sh
# Optional: black pipeline/ api/ tests/ && pylint pipeline/ api/
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

Core endpoints: `/health`, `/status`, `/process`, `/train`, `/evaluate`, `/predict`, `/kfold`.

Interactive docs when running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🐳 Docker Deployment Guide

### Docker Health Checks
```bash
docker inspect --format='{{.State.Health.Status}}' churn-api
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

- Training time: ~30s; latency: <100ms; memory: ~200MB; throughput: >100 req/s

## 📑 Technical Report

For a consolidated, academically written account of the methodology and results, please consult the technical report and curated artifacts:

- Primary report: `report/REPORT.md`
- Data exploration: `report/data-exploration/` (summary tables and figures)
- Modeling artefacts: `report/modeling/` (k-fold summaries and training metadata)
- Evaluation outputs: `report/evaluation/` (hold-out metrics and diagnostic plots)

These documents elaborate the full workflow (EDA → feature engineering → modeling → evaluation), interpret results, discuss modeling risks and trade-offs, and outline deployment/monitoring plans in alignment with `TASK.md`.

## 🔧 Development

Environment variables: `PYTHONPATH` (project root), `BUILD_ENV` (development|production)

## 🐳 Docker

Build: `docker build -t churn-prediction-api .`  Run: `docker run -d -p 8000:8000 --name churn-api churn-prediction-api`

## 🚨 Troubleshooting (concise)
If the API fails to start, ensure the venv is active, port 8000 is free, and dependencies are installed. Rebuild Docker without cache if builds fail.

### Getting Help

- Check the interactive API documentation: http://localhost:8000/docs
- Review the RATIONALE.md for architectural decisions
- Check DOCKER_TEST_SUMMARY.md for Docker testing examples
- Examine the data-exploration/ directory for data requirements

## 📚 Documentation

Curated materials are in `report/`. Additional developer docs are in source folders.

## 🎯 Key Features

- Production-ready serving (Docker + health checks), comprehensive tests, model versioning, interactive docs, and monitoring-ready logging/metrics.

## 🔬 ML Pipeline

Data processing → model training (RF + CV) → evaluation (ROC/PR/CM) → serving (FastAPI).

## 🛠 Technology Stack

ML: scikit-learn, pandas, numpy, xgboost | API: FastAPI, uvicorn, pydantic | Testing: pytest, httpx | Deployment: Docker | Package: uv/pyproject | CLI: Click

---

## 🚀 Quick Reference
```bash
uv sync && source .venv/bin/activate
python pipeline/main.py all --data data/customer_churn.csv
./scripts/run_dev.sh && open http://localhost:8000/docs
```
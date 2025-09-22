# Multi-stage Dockerfile for Customer Churn Prediction API
# Optimized for both development and production deployment

ARG PYTHON_VERSION=3.9
ARG BUILD_ENV=production

# Build stage - Install dependencies and build environment
FROM python:${PYTHON_VERSION}-slim AS builder

# Set build arguments and environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Copy dependency configuration
COPY pyproject.toml ./

# Create virtual environment and install all dependencies
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy \
    fastapi uvicorn pydantic python-multipart joblib click

# Production stage - Minimal runtime environment
FROM python:${PYTHON_VERSION}-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY pipeline/ ./pipeline/
COPY api/ ./api/
COPY data/ ./data/

# Create necessary directories with proper permissions
RUN mkdir -p models results data/processed models/splits results/plots

# Development vs Production command selection
ARG BUILD_ENV
RUN if [ "$BUILD_ENV" = "development" ]; then \
        echo '#!/bin/sh\nuvicorn api.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir api --reload-dir pipeline' > /start.sh; \
    else \
        echo '#!/bin/sh\nuvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1' > /start.sh; \
    fi && chmod +x /start.sh

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/start.sh"]
# Multi-stage Docker build for Autonomous Financial Risk Management System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r fintech && useradd -r -g fintech fintech

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ ./requirements/
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements/production.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -r requirements/development.txt

# Copy source code
COPY . .

# Change ownership to fintech user
RUN chown -R fintech:fintech /app

USER fintech

# Expose port
EXPOSE 8000

# Development command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy source code
COPY . .

# Install production dependencies only
RUN pip install --no-deps -r requirements/production.txt

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R fintech:fintech /app

# Switch to non-root user
USER fintech

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production command
CMD ["gunicorn", "api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Use production stage by default
FROM production

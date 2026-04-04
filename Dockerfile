# ─────────────────────────────────────────────────────────────
# XAI-Based Network Intrusion Detection System
# Docker image — Python 3.11 slim + all dependencies
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Chandra Sekhar Chakraborty <chakrabortychandrasekhar185@gmail.com>"
LABEL description="XAI-Based NIDS — Streamlit dashboard with SHAP explainability"

# System dependencies for scientific Python stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create necessary runtime directories
RUN mkdir -p data/raw data/processed data/samples models docs/screenshots

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command — launch Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

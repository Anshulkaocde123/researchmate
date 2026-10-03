# Multi-stage lightweight Dockerfile for ResearchMate
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml LICENSE README.md ./
RUN pip install --upgrade pip && \
    pip install -e .

# Copy source code and prebuilt indexes
COPY src/ /app/src/
COPY indexes/ /app/indexes/
COPY data/eval/ /app/data/eval/

# Expose Gradio port
EXPOSE 7860

# Run Gradio application
CMD ["python", "-m", "researchmate.ui.app"]

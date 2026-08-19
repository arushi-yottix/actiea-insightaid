# Dockerfile for LLM Service - Remote Mode (No GPU Required)
# Optimized for remote LLM API usage (Ollama, vLLM, etc.)
# This Dockerfile is for deployment when using remote_url (no local model loading)

FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (minimal - no CUDA/GPU libraries needed)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (CPU-only, no torch CUDA needed for remote mode)
# Note: We still install transformers, etc. for compatibility, but they won't be used
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.0.0 \
    httpx \
    pyyaml \
    python-dotenv

# Copy application files
COPY insightaid_llm_service.py .
COPY insightaid_call_llm.py .
COPY config.yaml .
COPY .env* ./

# Expose the service port (default: 8001)
EXPOSE 3003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://192.168.41.49:3003/health || exit 1

# Run the service using uvicorn (consistent with API wrapper)
CMD ["uvicorn", "insightaid_llm_service:app", "--host", "0.0.0.0", "--port", "3003"]







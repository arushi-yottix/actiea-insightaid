# LLM Service

A separate GPU-enabled FastAPI service for LLM inference in the InsightAid RAG system. This service handles all model loading and text generation, running independently on GPU nodes while the API wrapper runs on CPU nodes.

## 📋 Overview

The LLM Service is a dedicated microservice that:
- Loads and manages LLM models (currently Qwen2.5-7B-Instruct)
- Handles GPU memory optimization (quantization, device mapping)
- Provides a REST API for text generation
- Runs separately from the API wrapper for better resource isolation

**Architecture**: The API wrapper (`API_wrapper/`) retrieves relevant chunks from Qdrant, builds the prompt context, and calls this LLM service via HTTP to generate answers.

## 📁 Files

- **`insightaid_llm_service.py`** - FastAPI application with `/generate` and `/health` endpoints
- **`insightaid_call_llm.py`** - Core LLM class with model loading, quantization, and generation logic
- **`config.yaml`** - Service configuration (host, port, model settings)
- **`requirements-llm.txt`** - Python dependencies (PyTorch with CUDA, transformers, etc.)
- **`Dockerfile`** - Docker image definition for GPU-enabled container
- **`docker-compose.yml`** - Docker Compose configuration for local deployment

## 🔧 Configuration

### config.yaml

```yaml
server:
  host: "0.0.0.0"  # Bind to all interfaces (required for server deployment)
  port: 8001       # LLM service port

llm:
  model_id: "Qwen/Qwen2.5-7B-Instruct"  # Single model - 7B optimized for T4 GPU
  model_max_length: 32768  # Qwen2.5-7B-Instruct: 32K context window
```

### Environment Variables

You can override config.yaml settings using environment variables:

- `LLM_SERVICE_HOST` - Overrides `server.host` (default: `0.0.0.0`)
- `LLM_SERVICE_PORT` - Overrides `server.port` (default: `8001`)

## 🚀 Installation

### Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (recommended: T4 16GB, L4 24GB, or A100 40GB+)
- **CUDA**: CUDA 12.1+ (for PyTorch CUDA support)
- **Python**: Python 3.10+
- **Docker**: Docker with NVIDIA Container Toolkit (for containerized deployment)

### Local Installation

1. **Install system dependencies** (Ubuntu/Debian):
   ```bash
   sudo apt-get update && sudo apt-get install -y \
       python3.10 \
       python3-pip \
       python3-dev \
       tesseract-ocr \
       libtesseract-dev \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1
   ```

2. **Install PyTorch with CUDA**:
   ```bash
   pip3 install --no-cache-dir --upgrade pip
   pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Python dependencies**:
   ```bash
   pip3 install --no-cache-dir -r requirements-llm.txt
   ```

4. **Configure**:
   - Copy `config.yaml` and update model settings if needed
   - (Optional) Create `.env` file with environment variables

5. **Run the service**:
   ```bash
   python3 insightaid_llm_service.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn insightaid_llm_service:app --host 0.0.0.0 --port 8001
   ```

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t insightaid-llm-service:latest -f Dockerfile .
```

### Run with Docker

```bash
docker run -d \
  --name insightaid-llm-service \
  --gpus all \
  -p 8001:8001 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/.env:/app/.env:ro \
  insightaid-llm-service:latest
```

### Docker Compose

```bash
docker-compose up -d
```

The `docker-compose.yml` file includes:
- GPU device reservation
- Volume mounts for config files
- Health checks
- Automatic restart policy

## 📡 API Endpoints

### POST `/generate`

Generate an answer using the LLM model.

**Request Body**:
```json
{
  "context": "Document context text here...",
  "query": "What is the repair procedure for damage?"
}
```

**Response**:
```json
{
  "answer": "Based on the provided context...",
  "model": "Qwen2.5-7B-Instruct"
}
```

**Status Codes**:
- `200` - Success
- `500` - Generation error
- `507` - GPU Out of Memory
- `503` - Model not loaded

### GET `/health`

Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen2.5-7B-Instruct",
  "model_max_length": 32768,
  "gpu_available": true,
  "gpu_memory": {
    "total": "16.00 GB",
    "allocated": "7.50 GB",
    "reserved": "8.00 GB",
    "free": "8.00 GB"
  }
}
```

## ⚙️ GPU Optimization

The service automatically optimizes for different GPU types:

- **T4 (16GB)**: Uses 8-bit or 4-bit quantization to fit in memory
- **L4 (24GB)**: Uses bfloat16 without quantization
- **A100+ (40GB+)**: Uses bfloat16, full precision

**Quantization**:
- Automatically detects GPU memory
- Falls back to 4-bit if 8-bit fails
- Falls back to bfloat16 if quantization unavailable

**Memory Management**:
- Dynamic context truncation based on available GPU memory
- Automatic max_new_tokens adjustment for T4 GPUs
- CUDA cache clearing between requests

## 🔍 Model Details

**Current Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Size**: 7B parameters
- **Context Window**: 32K tokens
- **License**: Apache 2.0 (no gating)
- **Optimized for**: T4 GPU (16GB) with quantization

**Model Loading**:
- Loads at service startup (single model instance)
- Uses `device_map="auto"` for optimal GPU placement
- Supports SDPA (Scaled Dot Product Attention) for memory efficiency

## 🛠️ Troubleshooting

### GPU Not Detected

**Issue**: Service runs on CPU instead of GPU

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA availability: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. For Docker: Ensure `--gpus all` flag is used
4. Check NVIDIA Container Toolkit is installed

### Out of Memory (OOM) Errors

**Issue**: `507 Insufficient Storage` errors during generation

**Solutions**:
1. Reduce context size in API wrapper
2. Enable quantization (automatic for T4)
3. Use a smaller model
4. Reduce `max_new_tokens` in config

### Model Loading Fails

**Issue**: Service fails to start, model not loaded

**Solutions**:
1. Check GPU memory: `nvidia-smi`
2. Verify model ID in `config.yaml` is correct
3. Check internet connection (model downloads from HuggingFace)
4. Review startup logs for specific error messages

### CUBLAS Errors

**Issue**: Runtime errors mentioning CUBLAS

**Solutions**:
1. Service automatically falls back to CPU on CUBLAS errors
2. Restart the service to reset CUDA state
3. Check CUDA version compatibility (requires CUDA 12.1+)

## 📊 Performance

**Expected Latency** (T4 GPU, quantized):
- Model loading: ~30-60 seconds
- Generation (500 tokens): ~5-15 seconds
- Context processing: ~1-3 seconds

**Memory Usage**:
- T4 (16GB) with 8-bit quantization: ~4-5GB
- T4 (16GB) with 4-bit quantization: ~2-3GB
- L4 (24GB) without quantization: ~7-8GB

## 🔗 Integration with API Wrapper

The API wrapper calls this service via HTTP:

```python
# In API wrapper
response = httpx.post(
    f"{LLM_SERVICE_URL}/generate",
    json={"context": context, "query": query}
)
```

**Environment Variable**:
- Set `LLM_SERVICE_URL` in API wrapper (e.g., `http://llm-service:8001`)

## 📝 Notes

- **Single Model**: Currently supports one model at a time (configured in `config.yaml`)
- **Startup Time**: First startup takes longer due to model download and loading
- **Health Checks**: Service includes automatic health checks for container orchestration
- **Logging**: All operations are logged with correlation IDs for debugging

## 🔐 Security

- Service binds to `0.0.0.0` by default (accessible from network)
- For production, use firewall rules or network policies to restrict access
- Only the API wrapper should have access to this service

## 📚 Related Documentation

- See main `README.md` for overall system architecture
- See `API_wrapper/README.md` for API wrapper documentation
- See `insightaid_call_llm.py` for detailed LLM implementation and prompt templates

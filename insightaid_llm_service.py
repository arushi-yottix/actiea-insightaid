#!/usr/bin/env python3
"""
LLM Service - Separate GPU service for LLM inference
This service handles all LLM model loading and generation, running on GPU nodes.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import LLM class from insightaid_rag_core
from insightaid_rag_core import LLM, RAGConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("llm_service")

# Load config
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    logger.error("config.yaml not found!")
    raise RuntimeError("Please create config.yaml in the project root")

with open(CONFIG_PATH, "r") as fh:
    cfg_yaml = yaml.safe_load(fh)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Service",
    description="GPU-based LLM inference service for RAG system",
    version="1.0.0"
)

# Global LLM instances
llm_instances: Dict[str, LLM] = {}

# Request/Response models
class GenerateRequest(BaseModel):
    context: str
    query: str
    model: str = "qwen"  # Default model key

class GenerateResponse(BaseModel):
    answer: str
    model: str

@app.on_event("startup")
async def startup_event():
    """Load LLM models at startup"""
    global llm_instances
    
    logger.info("Starting LLM Service - Loading models...")
    
    LLM_MODELS = cfg_yaml["llm"]["models"]
    for key, model_name in LLM_MODELS.items():
        temp_cfg = RAGConfig()
        temp_cfg.model_id = model_name
        logger.info(f"Loading LLM model: {key} ({model_name})")
        print(f"🔄 Loading LLM model: {key} ({model_name})...", flush=True)
        try:
            llm_instances[key] = LLM(temp_cfg)
            print(f"✅ LLM model '{key}' loaded successfully", flush=True)
            logger.info(f"LLM model '{key}' loaded successfully")
        except Exception as e:
            error_msg = f"❌ FAILED to load LLM model '{key}': {e}"
            logger.error(error_msg, exc_info=True)
            print(error_msg, flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            # Don't raise - allow service to start even if one model fails
    
    if not llm_instances:
        error_msg = "❌ CRITICAL: No LLM models were loaded successfully. Service cannot start."
        logger.error(error_msg)
        print(error_msg, flush=True)
        raise RuntimeError("No LLM models available. Check model configuration and GPU availability.")
    
    logger.info(f"LLM Service ready with {len(llm_instances)} model(s): {list(llm_instances.keys())}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "ok",
        "models_loaded": len(llm_instances),
        "available_models": list(llm_instances.keys()),
        "model_max_lengths": {},  # Add model max context lengths
        "gpu_available": False,
        "gpu_memory": {}
    }
    
    # Get model max context lengths
    for model_key, llm in llm_instances.items():
        if hasattr(llm, 'model_max_length'):
            status["model_max_lengths"][model_key] = llm.model_max_length
        else:
            status["model_max_lengths"][model_key] = 32768  # Default fallback
    
    # Check GPU
    try:
        import torch
        if torch and torch.cuda.is_available():
            status["gpu_available"] = True
            status["gpu_memory"] = {
                "total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
                "allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB"
            }
    except Exception as e:
        status["gpu_memory"] = {"error": str(e)}
    
    return status

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate answer using LLM
    
    Args:
        request: GenerateRequest with context, query, and model key
    
    Returns:
        GenerateResponse with answer and model used
    """
    model_key = request.model
    
    # Get LLM instance
    llm = llm_instances.get(model_key)
    if not llm:
        available_models = list(llm_instances.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found. Available models: {available_models}"
        )
    
    # Verify LLM is loaded
    if not hasattr(llm, 'pipe'):
        raise HTTPException(
            status_code=503,
            detail=f"LLM instance '{model_key}' is not properly initialized"
        )
    
    try:
        logger.info(f"Generating answer with {model_key} model (context: {len(request.context)} chars)")
        answer = llm.generate(request.context, request.query)
        logger.info(f"LLM generation successful, answer length: {len(answer)} characters")
        
        return GenerateResponse(answer=answer, model=model_key)
    
    except Exception as e:
        error_str = str(e).lower()
        is_oom = (
            "out of memory" in error_str or 
            "cuda" in error_str or
            "oom" in error_str
        )
        
        if is_oom:
            logger.error(f"GPU OOM error during generation: {e}")
            raise HTTPException(
                status_code=507,  # Insufficient Storage
                detail=f"GPU Out of Memory: {str(e)}"
            )
        else:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"LLM generation error: {str(e)}"
            )

@app.get("/models")
def list_models():
    """List available models"""
    return {
        "available_models": list(llm_instances.keys()),
        "default": cfg_yaml["llm"].get("default", "mistral")
    }

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("LLM_SERVICE_PORT", "8001"))
    host = os.getenv("LLM_SERVICE_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


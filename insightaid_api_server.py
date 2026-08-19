# insightaid_api_server.py
"""
InsightAid API Server - Enhanced Qdrant RAG Server with:
- Logging system
- Performance metrics
- Query caching (Qdrant-based)
- Concurrency control
- File deduplication (Qdrant-based)
- All using Qdrant only (no Redis)
"""
import os
import yaml
import uuid
import json
import shutil
import time
from time import perf_counter
import hashlib
import asyncio
import logging
import socket
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
#from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime, timedelta

# Import your unchanged RAG implementation
# Note: LLM class is not imported - it's only used in insightaid_llm_service.py
from insightaid_rag_core import RAGConfig, PDFIngestor, Embedder, QdrantStore, chunk_document, AircraftRAG

# Import Priority 2 accuracy metrics
try:
    from accuracy_metrics import AnswerQualityEvaluator
    ACCURACY_METRICS_AVAILABLE = True
except ImportError:
    ACCURACY_METRICS_AVAILABLE = False
    logger.warning("accuracy_metrics.py not found. Answer quality evaluation will be disabled.")

# Local modules
from insightaid_prompts import FILE_MODE_SYSTEM, FULL_MODE_SYSTEM, MCP_MODE_SYSTEM

# numpy for similarity
import numpy as np
import re
import requests
import httpx
import gradio as gr
from gradio.routes import mount_gradio_app
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

# For CUDA OOM error handling
try:
    import torch
    # Fix #2: Initialize CUDA once at startup (REQUIRED for cuBLAS)
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            print("✅ CUDA initialized successfully at startup")
        except Exception as e:
            print(f"⚠️  CUDA initialization warning: {e}")
except ImportError:
    torch = None  # torch not available, OOM handling will be limited

# ---------------------------
# LOGGING SETUP
# ---------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure main logger
logger = logging.getLogger("aircraft_rag")
logger.setLevel(logging.INFO)

# File handler with rotation
fh = RotatingFileHandler(
    LOG_DIR / "rag_system.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
fh.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter (includes correlation ID if available)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - [corr_id=%(correlation_id)s] - %(message)s'
)

# Set default correlation_id for records without it
old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    if not hasattr(record, 'correlation_id'):
        record.correlation_id = 'N/A'
    return record
logging.setLogRecordFactory(record_factory)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# Separate loggers for different components
upload_logger = logging.getLogger("aircraft_rag.upload")
query_logger = logging.getLogger("aircraft_rag.query")
session_logger = logging.getLogger("aircraft_rag.session")
performance_logger = logging.getLogger("aircraft_rag.performance")

# ---------------------------
# PERFORMANCE METRICS CLASS
# ---------------------------
class PerformanceMetrics:
    """Track and log performance metrics"""
    
    def __init__(self, operation: str, session_id: str = None):
        self.operation = operation
        self.session_id = session_id
        self.start_time = time.time()
        self.metrics = {}
    
    def __enter__(self):
        performance_logger.info(f"START: {self.operation} | Session: {self.session_id}")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.metrics['total_time'] = elapsed
        performance_logger.info(
            f"END: {self.operation} | Session: {self.session_id} | "
            f"Duration: {elapsed:.3f}s | Metrics: {json.dumps(self.metrics)}"
        )
    
    #def add_metric(self, key: str, value: Any):
        #self.metrics[key] = value

# ---------------------------
# Load config.yaml
# ---------------------------
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    logger.error("config.yaml not found!")
    raise RuntimeError("Please create config.yaml in the project root (see example).")

with open(CONFIG_PATH, "r") as fh:
    cfg_yaml = yaml.safe_load(fh)

# Hard-code host to 0.0.0.0 for Kubernetes compatibility (never use 127.0.0.1)
APP_HOST = "0.0.0.0"
APP_PORT = 3000
# Allow environment variable override for Kubernetes (from ConfigMap)
QDRANT_HOST = os.getenv("QDRANT_HOST", cfg_yaml.get("qdrant", {}).get("host", "localhost"))
QDRANT_PORT = int(os.getenv("QDRANT_PORT", str(cfg_yaml.get("qdrant", {}).get("port", 6333))))
QDRANT_API_KEY = cfg_yaml.get("qdrant", {}).get("api_key", None)
# Allow environment variable override for Kubernetes deployment
# Local: Uses config.yaml paths
# Kubernetes: Overrides with env vars pointing to PVC-mounted directories
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", cfg_yaml["paths"]["uploads_dir"]))
MCP_REPO_DIR = Path(os.getenv("MCP_REPO_DIR", cfg_yaml["paths"]["mcp_repo_dir"]))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

TEMP_TTL = int(cfg_yaml["modes"]["temp_ttl_seconds"])
GRADIO_PATH = cfg_yaml["app"].get("gradio_path", "/ui")

# Concurrency settings
MAX_CONCURRENT_UPLOADS = cfg_yaml.get("concurrency", {}).get("max_uploads", 5)
MAX_CONCURRENT_QUERIES = cfg_yaml.get("concurrency", {}).get("max_queries", 10)

# Rate limiting settings
RATE_LIMIT_ENABLED = cfg_yaml.get("rate_limiting", {}).get("enabled", True)
RATE_LIMIT_PER_MINUTE = cfg_yaml.get("rate_limiting", {}).get("requests_per_minute", 60)
RATE_LIMIT_PER_HOUR = cfg_yaml.get("rate_limiting", {}).get("requests_per_hour", 1000)

# Out-of-context detection threshold (single fixed threshold)
OUT_OF_CONTEXT_THRESHOLD = 0.3  # If top chunk score < 0.3, likely out of context

# Fixed GPU tier context policy (simplified from complex heuristics)
CONTEXT_POLICY = {
    "SMALL": {"max_chunks": 4, "max_chars": 2000},   # T4 / ≤16GB
    "MEDIUM": {"max_chunks": 6, "max_chars": 3000},  # L4 / 24GB
    "LARGE": {"max_chunks": 8, "max_chars": 4000},  # A100+
}

logger.info(f"Configuration loaded - Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
logger.info(f"Concurrency limits: {MAX_CONCURRENT_UPLOADS} concurrent uploads, {MAX_CONCURRENT_QUERIES} concurrent queries (across all clients)")
logger.info(f"Rate limiting: {'enabled' if RATE_LIMIT_ENABLED else 'disabled'} ({RATE_LIMIT_PER_MINUTE} req/min per client, {RATE_LIMIT_PER_HOUR} req/hour per client)")
logger.info(f"Note: Unlimited clients can connect, but only {MAX_CONCURRENT_QUERIES} queries can be processed simultaneously")

# ---------------------------
# Semaphores for concurrency control
# ---------------------------
upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

# ---------------------------
# Correlation ID Middleware
# ---------------------------
class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to extract/generate correlation ID and add to logs"""
    
    async def dispatch(self, request: Request, call_next):
        # Extract correlation ID from header or generate new one
        corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        
        # Add correlation ID to request state for use in endpoints
        request.state.correlation_id = corr_id
        
        # Update logger context with correlation ID
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.correlation_id = corr_id
            return record
        logging.setLogRecordFactory(record_factory)
        
        try:
            response = await call_next(request)
            # Add correlation ID to response header
            response.headers["X-Correlation-ID"] = corr_id
            return response
        finally:
            # Restore original factory
            logging.setLogRecordFactory(old_factory)

# ---------------------------
# Rate Limiting Middleware
# ---------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        # Track requests by client IP
        self.minute_requests = defaultdict(list)  # {ip: [timestamps]}
        self.hour_requests = defaultdict(list)    # {ip: [timestamps]}
        self.cleanup_interval = 60  # Clean up old entries every 60 seconds
        self.last_cleanup = time.time()
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded IP (from proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        # Fallback to direct client
        if request.client:
            return request.client.host
        return "unknown"
    
    def cleanup_old_entries(self):
        """Remove old entries to prevent memory leak"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        # Clean minute requests
        for ip in list(self.minute_requests.keys()):
            self.minute_requests[ip] = [t for t in self.minute_requests[ip] if t > minute_ago]
            if not self.minute_requests[ip]:
                del self.minute_requests[ip]
        
        # Clean hour requests
        for ip in list(self.hour_requests.keys()):
            self.hour_requests[ip] = [t for t in self.hour_requests[ip] if t > hour_ago]
            if not self.hour_requests[ip]:
                del self.hour_requests[ip]
    
    async def dispatch(self, request: Request, call_next):
        # Only rate limit API endpoints (not health checks, docs, etc.)
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        client_ip = self.get_client_ip(request)
        current_time = time.time()
        
        # Cleanup old entries periodically
        self.cleanup_old_entries()
        
        # Check minute limit
        minute_ago = current_time - 60
        recent_minute_requests = [t for t in self.minute_requests[client_ip] if t > minute_ago]
        if len(recent_minute_requests) >= self.requests_per_minute:
            # Calculate actual wait time: oldest request in window expires in (60 - (current_time - oldest_request)) seconds
            oldest_request = min(recent_minute_requests) if recent_minute_requests else current_time
            wait_seconds = max(1, int(60 - (current_time - oldest_request)) + 1)  # Add 1 second buffer, minimum 1 second
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded: {self.requests_per_minute} requests per minute. Wait {wait_seconds} seconds before retrying.",
                    "retry_after": wait_seconds,
                    "limit": self.requests_per_minute,
                    "window": "1 minute",
                    "wait_seconds": wait_seconds
                },
                headers={"Retry-After": str(wait_seconds)}
            )
        
        # Check hour limit
        hour_ago = current_time - 3600
        recent_hour_requests = [t for t in self.hour_requests[client_ip] if t > hour_ago]
        if len(recent_hour_requests) >= self.requests_per_hour:
            # Calculate actual wait time: oldest request in window expires in (3600 - (current_time - oldest_request)) seconds
            oldest_request = min(recent_hour_requests) if recent_hour_requests else current_time
            wait_seconds = max(1, int(3600 - (current_time - oldest_request)) + 1)  # Add 1 second buffer, minimum 1 second
            wait_minutes = wait_seconds // 60
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded: {self.requests_per_hour} requests per hour. Wait {wait_minutes} minutes ({wait_seconds} seconds) before retrying.",
                    "retry_after": wait_seconds,
                    "limit": self.requests_per_hour,
                    "window": "1 hour",
                    "wait_seconds": wait_seconds,
                    "wait_minutes": wait_minutes
                },
                headers={"Retry-After": str(wait_seconds)}
            )
        
        # Record request
        self.minute_requests[client_ip].append(current_time)
        self.hour_requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response

# ---------------------------
# Instantiate base RAG config and components
# ---------------------------
base_cfg = RAGConfig()
base_cfg.pdf_directory = str(MCP_REPO_DIR)
base_cfg.qdrant_host = QDRANT_HOST
base_cfg.qdrant_port = QDRANT_PORT
base_cfg.qdrant_api_key = QDRANT_API_KEY

# Allow environment variable to override embedding model for testing
# Set EMBEDDING_MODEL=all-MiniLM-L6-v2 for fast testing (~80MB instead of 2.27GB)
# Set EMBEDDING_DIM=384 to match all-MiniLM-L6-v2
if os.getenv("EMBEDDING_MODEL"):
    base_cfg.embedding_model = os.getenv("EMBEDDING_MODEL")
    logger.info(f"Using embedding model from environment: {base_cfg.embedding_model}")
if os.getenv("EMBEDDING_DIM"):
    base_cfg.embedding_dim = int(os.getenv("EMBEDDING_DIM"))
    logger.info(f"Using embedding dimension from environment: {base_cfg.embedding_dim}")

# Initialize as None - will be set in startup event
ingestor = None
embedder = None
store = None
initialization_complete = False
initialization_error = None

# LLM Service Architecture: Models are loaded in separate LLM service
# No local LLM instances - all generation is done via HTTP calls to LLM service
DEFAULT_LLM = cfg_yaml["llm"].get("default", "mistral")
# Default to localhost for local testing, override with env var for Kubernetes
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
LLM_SERVICE_TIMEOUT = float(os.getenv("LLM_SERVICE_TIMEOUT", "300.0"))  # 5 minutes

# Get available models from config.yaml for Gradio UI
LLM_MODELS = cfg_yaml["llm"].get("models", {})
if not LLM_MODELS:
    # Fallback if models not in config
    LLM_MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct"}

logger.info(f"Using LLM Service architecture - Default model: {DEFAULT_LLM}")
logger.info(f"LLM models will be loaded in separate LLM service at {LLM_SERVICE_URL}")
logger.info(f"Available models in config: {list(LLM_MODELS.keys())}")

# ---------------------------
# Helper Functions (Simplified Architecture)
# ---------------------------

def detect_gpu_tier() -> str:
    """
    Detect GPU tier: SMALL (T4/≤16GB), MEDIUM (L4/24GB), LARGE (A100+)
    Returns: "SMALL", "MEDIUM", or "LARGE"
    """
    if not torch or not torch.cuda.is_available():
        return "SMALL"
    
    try:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem_gb <= 16:
            return "SMALL"
        elif mem_gb <= 30:
            return "MEDIUM"
        else:
            return "LARGE"
    except:
        return "SMALL"

def get_model_max_tokens(model_key: str) -> int:
    """
    Get model max context length from config.yaml (static, no dynamic probing)
    Returns: max tokens for the model
    """
    config_max_lengths = cfg_yaml.get("llm", {}).get("model_max_lengths", {})
    config_default_max = cfg_yaml.get("llm", {}).get("default_max_length", 32768)
    return config_max_lengths.get(model_key, config_default_max)

def retrieve_candidates(mode: str, qvec_bytes: bytes, session_id: str, top_k: int) -> list:
    """
    Retrieve candidates based on mode (file/repo/full)
    Returns: list of candidate chunks
    """
    candidates = []
    if mode == "file":
        candidates = get_temp_candidates(session_id, qvec_bytes, top_k)
    elif mode == "repo":
        candidates = get_persistent_candidates(qvec_bytes, top_k)
    elif mode == "full":
        persistent_cands = get_persistent_candidates(qvec_bytes, top_k)
        if session_id:
            temp_cands = get_temp_candidates(session_id, qvec_bytes, top_k)
        else:
            temp_cands = []
        # Merge, avoiding duplicates
        seen_texts = set([c.get("text", "") for c in temp_cands])
        candidates = temp_cands.copy()
        for p in persistent_cands:
            if p.get("text", "") and p.get("text", "") not in seen_texts:
                candidates.append(p)
    return candidates

def rerank_candidates(candidates: list, query: str) -> list:
    """
    Rerank candidates using semantic score + keyword overlap + filename/table boosts
    Returns: sorted list of candidates by final_score
    """
    if not candidates:
        return []
    
    q_lower = query.lower()
    qwords = set(re.findall(r'\w+', q_lower))
    
    # Extract filename and table reference from query
    query_filename = None
    for pattern in [r'from\s+file\s+(.+?)(?:\s|$|,|\.|\?|$)', r'in\s+file\s+(.+?)(?:\s|$|,|\.|\?|$)', r'file\s+(.+?)(?:\s|$|,|\.|\?|$)']:
        match = re.search(pattern, q_lower)
        if match:
            query_filename = re.sub(r'\s+(pdf|document|file)$', '', match.group(1).strip().rstrip('.,?'))
            break
    
    query_table_ref = None
    for pattern in [r'table\s*[-\s]*(\d+)', r'table\s*n[°o]\s*(\d+)']:
        match = re.search(pattern, q_lower)
        if match:
            query_table_ref = match.group(1)
            break
    
    # Extract figure and sheet numbers from query (for prompt validation, NOT reranking)
    # Since there's no metadata, we rely on prompt validation, not reranking boosts
    query_figure_ref = None
    query_sheet_ref = None
    # Extract sheet number first (CRITICAL: same figure number can appear in different sheets)
    sheet_match = re.search(r'sheet\s+([0-9/]+)', q_lower)
    if sheet_match:
        query_sheet_ref = sheet_match.group(1).strip()
    
    # Extract figure number (handle various formats)
    figure_patterns = [
        r'figure\s+([0-9a-z\-]+)',  # "Figure 57-41-19-991-022-A"
        r'fig\.\s+([0-9a-z\-]+)',
        r'figure\s+([0-9]+)'
    ]
    for pattern in figure_patterns:
        match = re.search(pattern, q_lower)
        if match:
            query_figure_ref = match.group(1).strip()
            break
    
    # Rerank each candidate
    for c in candidates:
        try:
            text = c.get("text", "")
            if not text:
                c["final_score"] = 0.0
                continue
            
            text_lower = text.lower()
            cwords = set(re.findall(r'\w+', text_lower))
            
            # Keyword overlap
            overlap = len(qwords & cwords) / max(len(qwords), 1)
            
            # Filename boost
            filename_boost = 0.0
            if query_filename and c.get("filename"):
                query_fn_norm = re.sub(r'[^\w\s-]', '', query_filename.lower())
                chunk_fn_norm = re.sub(r'[^\w\s-]', '', c.get("filename", "").lower())
                if query_fn_norm in chunk_fn_norm or chunk_fn_norm in query_fn_norm:
                    filename_boost = 0.3
            
            # Table boost
            table_boost = 0.0
            if query_table_ref:
                for pattern in [rf'\btable\s*[-\s]*{query_table_ref}\b', rf'\btable\s*n[°o]\s*{query_table_ref}\b', rf'\btable\s*#{query_table_ref}\b']:
                    if re.search(pattern, text_lower):
                        table_boost = 0.3
                        break
            
            # Final score: 60% semantic + 40% (keyword + boosts)
            # Note: Figure/Sheet matching is handled by prompt validation, not reranking boost
            # (No metadata stored, so text matching in reranking is unreliable)
            c_score = float(c.get("score", 0.0))
            enhanced_overlap = min(1.0, overlap + filename_boost + table_boost)
            c["final_score"] = 0.6 * c_score + 0.4 * enhanced_overlap
            
        except Exception as e:
            logger.warning(f"Error reranking candidate: {e}")
            c["final_score"] = 0.0
    
    # Sort by final_score
    candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return candidates

def build_context(chunks: list, model_key: str, gpu_tier: str) -> str:
    """
    Build context from chunks using fixed policy based on GPU tier
    Returns: context string
    """
    if not chunks:
        return ""
    
    # Get policy for GPU tier
    policy = CONTEXT_POLICY.get(gpu_tier, CONTEXT_POLICY["SMALL"])
    max_chunks = policy["max_chunks"]
    max_chars = policy["max_chars"]
    
    # Get model max tokens (static from config)
    model_max_tokens = get_model_max_tokens(model_key)
    model_safe_max_tokens = max(1000, model_max_tokens - 1600)  # Reserve 1600 tokens
    model_safe_max_chars = model_safe_max_tokens * 4  # ~4 chars per token
    
    # Cap by model limit
    max_context_chars = min(max_chars * max_chunks, model_safe_max_chars)
    
    context_lines = []
    current_length = 0
    chunks_used = 0
    
    for i, c in enumerate(chunks[:max_chunks], 1):
        if chunks_used >= max_chunks:
            break
        
        try:
            filename = c.get('filename', 'Unknown Document')
            page_num = c.get('page_num', 0)
            text = c.get("text", "")
            
            # Truncate chunk if too long
            if len(text) > max_chars:
                text = text[:max_chars] + "... [truncated]"
            
            chunk_text = f"[{i}] Document: {filename} | Page: {page_num}\nContent from {filename}, Page {page_num}:\n{text}\n" + "-" * 50
            
            # Check if adding would exceed limit
            if current_length + len(chunk_text) > max_context_chars:
                break
            
            context_lines.append(f"[{i}] Document: {filename} | Page: {page_num}")
            context_lines.append(f"Content from {filename}, Page {page_num}:")
            context_lines.append(text)
            context_lines.append("-" * 50)
            current_length += len(chunk_text)
            chunks_used += 1
        except Exception as e:
            logger.warning(f"Error processing chunk {i}: {e}")
            continue
    
    return "\n".join(context_lines)

def detect_out_of_context(top_chunks: list) -> bool:
    """
    Detect if query is out of context (single threshold check)
    Returns: True if out of context
    """
    if not top_chunks:
        return False
    top_score = top_chunks[0].get("final_score", 0.0)
    return top_score < OUT_OF_CONTEXT_THRESHOLD

# ============================================
# Pydantic Models (Must be defined before helper functions)
# ============================================

class QueryRequest(BaseModel):
    """
    Request model for query endpoint.
    
    Attributes:
        query: The user's question or query string
        mode: Search mode - 'file' (session files only), 'repo' (persistent repo only), or 'full' (both)
        session_id: Optional session ID for file mode (required for 'file' mode, optional for 'full' mode)
        model: Optional LLM model override (e.g., 'qwen', 'llama')
        top_k: Optional number of top chunks to retrieve (defaults to config value)
        include_mcp_tools: Optional flag to include MCP tool results (reserved for future MCP integration)
        no_cache: Set to True to bypass cache and get fresh results
    """
    query: str
    mode: str  # 'file' | 'repo' | 'full'
    session_id: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = None
    include_mcp_tools: Optional[bool] = False  # Reserved for future MCP integration
    no_cache: Optional[bool] = False  # Set to True to bypass cache and get fresh results
    
    class Config:
        schema_extra = {
            "example": {
                "query": "what if DAMAGE is WITHIN 325.0 mm (12.795 in) OF INBOARD EDGE and depth is less than 0.4 mm?",
                "mode": "file",
                "session_id": "user1",
                "model": "mistral",
                "top_k": 0,
                "include_mcp_tools": False
            }
        }

# ============================================
# Query Processing Helper Functions (Refactored)
# ============================================

def validate_query_request(req: QueryRequest, mode: str) -> tuple[str, str, str]:
    """
    Validate and normalize query request.
    Returns: (query, session_id, model_key)
    Raises: HTTPException on validation error
    """
    q = req.query.strip()
    if not q:
        raise HTTPException(400, detail={"error": "EMPTY_QUERY", "message": "Query cannot be empty"})
    
    session_id = req.session_id.strip() if req.session_id else None
    if mode == "file" and not session_id:
        raise HTTPException(400, detail={"error": "MISSING_SESSION_ID", "message": "session_id required for file mode"})
    
    model_key = (req.model or cfg_yaml["llm"].get("default", "qwen")).lower()
    return q, session_id or "", model_key

def validate_session_exists(session_id: str, mode: str) -> bool:
    """
    Validate session exists for file mode.
    Returns: True if session exists or mode != file
    """
    if mode != "file":
        return True
    
    try:
        session_check = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="metadata")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]),
            limit=1
        )
        session_exists = len(session_check[0]) > 0 if session_check[0] else False
        
        chunk_check = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="is_temp", match=MatchValue(value=True))
            ]),
            limit=1
        )
        chunks_exist = len(chunk_check[0]) > 0 if chunk_check[0] else False
        
        return session_exists or chunks_exist
    except Exception as e:
        logger.warning(f"Could not verify session existence: {e}")
        return False

def get_history_session(mode: str, session_id: str) -> str:
    """
    Get history session ID based on mode.
    Returns: history session identifier
    """
    if mode == "file":
        return session_id if session_id else "file_default"
    elif mode == "repo":
        return "repo"
    else:  # full mode
        return session_id if session_id else "full_default"

def retrieve_and_rerank(mode: str, query: str, qvec_bytes: bytes, session_id: str, top_k: int, conversation_history: list = None, corr_id: str = 'N/A') -> tuple[list, list, float, float]:
    """
    Retrieve and rerank candidates.
    DISABLED: History-aware retrieval is disabled - uses only current query.
    History is NOT passed to generation (generation is stateless).
    Returns: (all_candidates, top_chunks, retrieve_latency, rerank_latency)
    """
    # DISABLED: History-aware retrieval removed to prevent query mixing
    # Each query is now independent and uses only the current query for retrieval
    # This prevents confusion from mixing information from previous queries
    # conversation_history parameter is kept for compatibility but ignored
    
    retrieve_latency = 0.0
    try:
        retrieve_start = perf_counter()
        candidates = retrieve_candidates(mode, qvec_bytes, session_id, top_k)
        retrieve_latency = perf_counter() - retrieve_start
        logger.info(f"[corr_id={corr_id}] Retrieved {len(candidates)} candidates in {mode} mode (latency: {retrieve_latency:.3f}s)")
    except Exception as e:
        logger.error(f"[corr_id={corr_id}] Error retrieving candidates: {e}", exc_info=True)
        candidates = []
    
    if not candidates:
        return [], [], 0.0, 0.0
    
    rerank_latency = 0.0
    try:
        rerank_start = perf_counter()
        candidates = rerank_candidates(candidates, query)
        rerank_latency = perf_counter() - rerank_start
        final_k = base_cfg.final_top_k
        top_chunks = candidates[:final_k]
        logger.info(f"[corr_id={corr_id}] Reranked to {len(top_chunks)} top chunks (reranking latency: {rerank_latency:.3f}s)")
    except Exception as e:
        logger.error(f"[corr_id={corr_id}] Error during reranking: {e}", exc_info=True)
        top_chunks = candidates[:base_cfg.final_top_k] if candidates else []
    
    return candidates, top_chunks, retrieve_latency, rerank_latency

def build_prompt_context(
    mode: str,
    query: str,
    top_chunks: list,
    model_key: str,
    conversation_history: list,  # Kept for compatibility but NOT used in generation
    gpu_tier: str
) -> str:
    """
    Build complete prompt context: SYSTEM PROMPT → DOCUMENTS → QUESTION
    NOTE: History is NOT included - generation is stateless and deterministic.
    History is only used for retrieval (see retrieve_and_rerank function).
    Returns: Complete context string (hard-capped to prevent CUDA errors)
    """
    # Get model max tokens and hard cap
    model_max_tokens = get_model_max_tokens(model_key)
    model_safe_max_tokens = max(1000, model_max_tokens - 1600)  # Reserve 1600 tokens
    model_safe_max_chars = model_safe_max_tokens * 4  # ~4 chars per token
    
    # Get GPU tier policy
    policy = CONTEXT_POLICY.get(gpu_tier, CONTEXT_POLICY["SMALL"])
    max_chunks = policy["max_chunks"]
    max_chars = policy["max_chars"]
    
    # Hard cap: Never exceed model limit
    max_context_chars = min(max_chars * max_chunks, model_safe_max_chars)
    
    # 1. System prompt
    if mode == "file":
        system_prompt = FILE_MODE_SYSTEM
    elif mode == "repo":
        system_prompt = MCP_MODE_SYSTEM
    else:  # full mode
        system_prompt = FULL_MODE_SYSTEM
    
    # 2. Document context (hard-capped)
    # NOTE: History is NOT included in generation prompt - generation is stateless
    # History is only used for retrieval (see retrieve_and_rerank function)
    doc_context = ""
    if top_chunks:
        context_lines = []
        current_length = 0
        
        for i, c in enumerate(top_chunks[:max_chunks], 1):
            filename = c.get('filename', 'Unknown Document')
            page_num = c.get('page_num', 0)
            text = c.get("text", "")
            
            # Truncate chunk if too long
            if len(text) > max_chars:
                text = text[:max_chars] + "... [truncated]"
            
            chunk_text = f"[{i}] Document: {filename} | Page: {page_num}\nContent from {filename}, Page {page_num}:\n{text}\n" + "-" * 50
            
            # Hard cap: Stop if adding would exceed limit
            if current_length + len(chunk_text) > max_context_chars:
                break
            
            context_lines.append(f"[{i}] Document: {filename} | Page: {page_num}")
            context_lines.append(f"Content from {filename}, Page {page_num}:")
            context_lines.append(text)
            context_lines.append("-" * 50)
            current_length += len(chunk_text)
        
        if context_lines:
            doc_context = "RELEVANT DOCUMENT EXCERPTS:\n" + "=" * 60 + "\n" + "\n".join(context_lines) + "\n"
    
    # 3. User question
    question_text = f"USER QUESTION: {query}\n"
    
    # Combine: SYSTEM → DOCUMENTS → QUESTION (NO HISTORY - stateless generation)
    full_context = f"{system_prompt}\n\n{doc_context}{question_text}"
    
    # Final hard cap check (should never trigger if logic above is correct)
    if len(full_context) > model_safe_max_chars:
        logger.warning(f"Context exceeded safe limit ({len(full_context)} > {model_safe_max_chars}), truncating")
        full_context = full_context[:model_safe_max_chars] + "... [truncated]"
    
    return full_context

async def evaluate_quality_async(
    quality_evaluator,
    answer: str,
    query: str,
    top_chunks: list
):
    """
    Evaluate answer quality in background (fire-and-forget, non-blocking).
    Logs results but doesn't return them.
    """
    if not quality_evaluator:
        return
    
    try:
        # Run in background thread to avoid blocking request
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: quality_evaluator.evaluate_answer(
                predicted_answer=answer,
                ground_truth=None,
                query=query,
                context_chunks=top_chunks
            )
        )
        # Results are logged by evaluator, not returned
    except Exception as e:
        logger.warning(f"Error evaluating answer quality (non-critical): {e}")

# Function to call LLM service
async def call_llm_service(context: str, query: str, model: str = None) -> str:
    """
    Call the LLM service to generate an answer.
    
    Args:
        context: The context string for RAG
        query: The user query
        model: Model key (defaults to DEFAULT_LLM)
    
    Returns:
        Generated answer string
    
    Raises:
        HTTPException: If LLM service call fails
    """
    if model is None:
        model = DEFAULT_LLM
    
    try:
        async with httpx.AsyncClient(timeout=LLM_SERVICE_TIMEOUT) as client:
            response = await client.post(
                f"{LLM_SERVICE_URL}/generate",
                json={
                    "context": context,
                    "query": query,
                    "model": model
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["answer"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 507:  # Insufficient Storage (OOM)
            logger.error(f"LLM Service OOM error: {e.response.text}")
            raise HTTPException(
                status_code=507,
                detail=f"GPU Out of Memory in LLM service: {e.response.text}"
            )
        elif e.response.status_code == 404:
            logger.error(f"LLM model not found: {e.response.text}")
            raise HTTPException(
                status_code=404,
                detail=f"LLM model '{model}' not found in LLM service"
            )
        elif e.response.status_code == 503:
            logger.error(f"LLM service unavailable: {e.response.text}")
            raise HTTPException(
                status_code=503,
                detail=f"LLM service unavailable: {e.response.text}"
            )
        else:
            logger.error(f"LLM service error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"LLM service error: {e.response.text}"
            )
    except httpx.TimeoutException:
        logger.error(f"LLM service timeout after {LLM_SERVICE_TIMEOUT}s")
        raise HTTPException(
            status_code=504,
            detail=f"LLM service timeout after {LLM_SERVICE_TIMEOUT}s"
        )
    except httpx.RequestError as e:
        logger.error(f"LLM service connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LLM service at {LLM_SERVICE_URL}: {str(e)}"
        )

# Session memory helper (using Qdrant)
class QdrantMemory:
    def __init__(self, store: QdrantStore):
        self.store = store
        self.sessions_collection = "sessions"
        self._init_sessions_collection()
    
    def _init_sessions_collection(self):
        from qdrant_client.models import VectorParams, Distance
        
        try:
            self.store.client.get_collection(self.sessions_collection)
        except:
            self.store.client.create_collection(
                collection_name=self.sessions_collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)
            )
    
    def append_user(self, session_id: str, text: str):
        self._append_history(session_id, "user", text)
    
    def append_assistant(self, session_id: str, text: str):
        self._append_history(session_id, "assistant", text)
    
    def _append_history(self, session_id: str, role: str, content: str):
        # Use deterministic ID: hash of session_id + role + content + timestamp
        # This ensures same history entry gets same ID even after restart
        ts = int(time.time())
        point_id = stable_id(f"history_{session_id}_{role}_{content}_{ts}")
        point = PointStruct(
            id=point_id,
            vector=[0.0],
            payload={"type": "history", "session_id": session_id, "role": role, "content": content, "ts": ts}
        )
        self.store.client.upsert(collection_name=self.sessions_collection, points=[point])
    
    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns empty list gracefully on any error."""
        try:
            normalized_session_id = str(session_id).strip() if session_id else None
            if not normalized_session_id:
                return []
            
            results = self.store.client.scroll(
                collection_name=self.sessions_collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="type", match=MatchValue(value="history")),
                    FieldCondition(key="session_id", match=MatchValue(value=normalized_session_id))
                ]),
                limit=limit * 2
            )
            
            if not results or not results[0]:
                return []
            
            points = results[0]
            messages = []
            for point in points:
                try:
                    payload = point.payload if hasattr(point, 'payload') else {}
                    msg = {
                        "role": payload.get("role", ""),
                        "content": payload.get("content", ""),
                        "ts": payload.get("ts", 0)
                    }
                    if msg.get("role") and msg.get("content"):
                        messages.append(msg)
                except:
                    continue
            
            messages.sort(key=lambda x: x.get("ts", 0))
            return messages[-limit:] if len(messages) > limit else messages
        except Exception as e:
            logger.warning(f"Error retrieving history for session {session_id}: {e}")
            return []

# Memory will be initialized in startup event
memory = None

# Initialize answer quality evaluator (Priority 2 metrics)
if ACCURACY_METRICS_AVAILABLE:
    quality_evaluator = AnswerQualityEvaluator(enable_bertscore=True)
    logger.info("Answer quality evaluator initialized (F1, BERTScore)")
else:
    quality_evaluator = None
    logger.warning("Answer quality evaluator not available")

# ---------------------------
# QDRANT-BASED CACHE COLLECTION
# ---------------------------
CACHE_COLLECTION = "query_cache"

# Cache collection initialization moved to startup event

# ---------------------------
# FILE DEDUPLICATION (Qdrant-based)
# ---------------------------
def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_file_already_processed(file_hash: str, session_id: str) -> bool:
    """Check if file hash exists in session (using Qdrant)"""
    try:
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="file_hash")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
            ]),
            limit=1
        )
        return len(results[0]) > 0 if results[0] else False
    except:
        return False

def mark_file_processed(file_hash: str, filename: str, session_id: str):
    """Mark file as processed with hash (using Qdrant)"""
    point_id = stable_id(f"file_hash_{session_id}_{file_hash}")
    point = PointStruct(
        id=point_id,
        vector=[0.0],
        payload={
            "type": "file_hash",
            "session_id": session_id,
            "file_hash": file_hash,
            "filename": filename,
            "ts": int(time.time())
        }
    )
    store.client.upsert(collection_name="sessions", points=[point])
    upload_logger.info(f"Marked file as processed: {filename} (hash: {file_hash[:8]}...) in session {session_id}")

def get_filename_by_hash(file_hash: str, session_id: str) -> Optional[str]:
    """Get original filename by hash (using Qdrant)"""
    try:
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="file_hash")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
            ]),
            limit=1
        )
        if results[0]:
            return results[0][0].payload.get("filename")
    except:
        pass
    return None

# ---------------------------
# PERSISTENT REPO FILE TRACKING (for idempotent ingestion)
# ---------------------------
def is_persistent_file_processed(file_hash: str, file_path: Path) -> bool:
    """Check if persistent repo file has already been ingested (using Qdrant)"""
    try:
        # Use file path as identifier (normalized to relative path from repo root)
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="persistent_file_hash")),
                FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
            ]),
            limit=1
        )
        return len(results[0]) > 0 if results[0] else False
    except:
        return False

def mark_persistent_file_processed(file_hash: str, file_path: Path, repo_root: Path):
    """Mark persistent repo file as processed with hash (using Qdrant)"""
    # Store relative path from repo root for tracking
    try:
        relative_path = str(file_path.relative_to(repo_root))
    except:
        relative_path = str(file_path)
    
    point_id = stable_id(f"persistent_file_hash_{file_hash}")
    point = PointStruct(
        id=point_id,
        vector=[0.0],
        payload={
            "type": "persistent_file_hash",
            "file_hash": file_hash,
            "file_path": relative_path,
            "absolute_path": str(file_path),
            "ts": int(time.time())
        }
    )
    store.client.upsert(collection_name="sessions", points=[point])
    logger.info(f"Marked persistent file as processed: {relative_path} (hash: {file_hash[:8]}...)")

# ---------------------------
# QUERY CACHING (Qdrant-based)
# ---------------------------
def compute_query_hash(query: str, mode: str, session_id: str) -> str:
    """Compute hash for query caching"""
    cache_string = f"{query.lower().strip()}|{mode}|{session_id or 'none'}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def _ensure_cache_collection():
    """Lazy-create cache collection if it doesn't exist (non-blocking)"""
    if not store:
        return  # Store not initialized, skip cache collection creation
    try:
        store.client.get_collection(CACHE_COLLECTION)
    except:
        try:
            from qdrant_client.models import VectorParams, Distance
            store.client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)
            )
            logger.info("Created query cache collection (lazy init)")
        except Exception as e:
            logger.warning(f"Could not create cache collection: {e} (non-critical)")

def get_cached_response(query_hash: str) -> Optional[Dict]:
    """Get cached response if available (using Qdrant)"""
    try:
        _ensure_cache_collection()  # Lazy-create if needed
        results = store.client.scroll(
            collection_name=CACHE_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="query_hash", match=MatchValue(value=query_hash))
            ]),
            limit=1
        )
        if results[0]:
            payload = results[0][0].payload
            # Check if cache is still valid (within TTL)
            cache_ts = payload.get("ts", 0)
            cache_ttl = payload.get("ttl", 3600)
            if time.time() - cache_ts < cache_ttl:
                query_logger.info(f"Cache HIT for query hash: {query_hash}")
                return json.loads(payload.get("response", "{}"))
            else:
                # Cache expired, delete by ID (more efficient than filter)
                point_id = stable_id(f"cache_{query_hash}")
                try:
                    store.client.delete(
                        collection_name=CACHE_COLLECTION,
                        points_selector=[point_id]
                    )
                except Exception as e:
                    logger.warning(f"Error deleting expired cache: {e}")
    except Exception as e:
        logger.warning(f"Error retrieving cache: {e}")
    
    query_logger.info(f"Cache MISS for query hash: {query_hash}")
    return None

def cache_response(query_hash: str, response: Dict, ttl: int = 3600):
    """Cache query response (using Qdrant)"""
    try:
        _ensure_cache_collection()  # Lazy-create if needed
        point_id = stable_id(f"cache_{query_hash}")
        point = PointStruct(
            id=point_id,
            vector=[0.0],
            payload={
                "query_hash": query_hash,
                "response": json.dumps(response),
                "ts": int(time.time()),
                "ttl": ttl
            }
        )
        store.client.upsert(collection_name=CACHE_COLLECTION, points=[point])
        query_logger.info(f"Cached response for query hash: {query_hash}")
    except Exception as e:
        logger.warning(f"Error caching response: {e} (non-critical)")

# ---------------------------
# Helper: Deterministic ID generation (IMPORTANT: Use for persistent IDs)
# ---------------------------
def stable_id(s: str) -> int:
    """
    Generate a deterministic integer ID from a string.
    Uses SHA256 to ensure same input always produces same ID across process restarts.
    CRITICAL: Use this instead of hash() for persistent Qdrant point IDs.
    """
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16) % (2**63)

# ---------------------------
# Helper: get temp/session candidates using Qdrant
# ---------------------------
def get_persistent_candidates(qvec_bytes: bytes, top_k: int) -> List[Dict[str, Any]]:
    """Get persistent repository candidates from Qdrant (is_temp: false)"""
    try:
        filter_conditions = {"is_temp": False}
        logger.info(f"Searching persistent repository with filter: {filter_conditions}")
        results = store.search(qvec_bytes, k=max(top_k * 2, 10), filter_conditions=filter_conditions)
        logger.info(f"Search returned {len(results)} results from persistent repository")
        
        items = []
        for r in results:
            try:
                items.append({
                    "text": r.get("text", ""),
                    "filename": r.get("filename", ""),
                    "page_num": r.get("page_num", 0),
                    "score": r.get("score", 0.0),
                    "sim": r.get("score", 0.0)
                })
            except Exception as e:
                logger.warning(f"Error processing result in get_persistent_candidates: {e}, result: {r}")
                continue
        
        logger.info(f"Processed {len(items)} items, returning top {top_k}")
        return items[:top_k]
    except Exception as e:
        logger.error(f"Error in get_persistent_candidates: {e}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_temp_candidates(session_id: str, qvec_bytes: bytes, top_k: int) -> List[Dict[str, Any]]:
    """Get temporary session-scoped candidates from Qdrant"""
    try:
        # CRITICAL: Normalize exactly as done during upload and upsert
        # Must match: str(session_id).strip() in upload (line 508) and upsert (line 329)
        normalized_session_id = str(session_id).strip() if session_id else None
        if not normalized_session_id:
            logger.warning(f"Empty session_id provided to get_temp_candidates")
            return []
        
        # Log for debugging session_id matching
        logger.info(f"🔍 Searching for session_id='{normalized_session_id}' (type={type(normalized_session_id).__name__}, len={len(normalized_session_id)})")
        
        filter_conditions = {"session_id": normalized_session_id, "is_temp": True}
        logger.info(f"Searching for candidates with filter: {filter_conditions}")
        results = store.search(qvec_bytes, k=max(top_k * 2, 10), filter_conditions=filter_conditions)
        logger.info(f"Search returned {len(results)} results")
        
        # Debug: If no results, check what session_ids actually exist
        if len(results) == 0:
            logger.warning(f"⚠️ No results found for session_id='{normalized_session_id}'. Checking what session_ids exist...")
            try:
                # Quick check: get a few temp chunks to see what session_ids are stored
                debug_results = store.client.scroll(
                    collection_name=store.collection_name,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="is_temp", match=MatchValue(value=True))
                    ]),
                    limit=10,
                    with_payload=True
                )
                if debug_results and debug_results[0]:
                    found_session_ids = set()
                    for point in debug_results[0]:
                        payload = point.payload if hasattr(point, 'payload') else {}
                        stored_session_id = payload.get("session_id")
                        if stored_session_id:
                            found_session_ids.add(str(stored_session_id))
                    logger.warning(f"Found {len(found_session_ids)} different session_ids in temp chunks: {list(found_session_ids)[:5]}")
                    if normalized_session_id not in found_session_ids:
                        logger.error(f"❌ SESSION ID MISMATCH! Looking for '{normalized_session_id}' but found: {list(found_session_ids)[:5]}")
            except Exception as debug_err:
                logger.warning(f"Could not debug session_ids: {debug_err}")
        
        items = []
        for r in results:
            try:
                items.append({
                    "text": r.get("text", ""),
                    "filename": r.get("filename", ""),
                    "page_num": r.get("page_num", 0),
                    "score": r.get("score", 0.0),
                    "sim": r.get("score", 0.0)
                })
            except Exception as e:
                logger.warning(f"Error processing result in get_temp_candidates: {e}, result: {r}")
                continue
        
        logger.info(f"Processed {len(items)} items, returning top {top_k}")
        return items[:top_k]
    except Exception as e:
        logger.error(f"Error in get_temp_candidates: {e}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return []

# ---------------------------
# Helper: Get session file list
# ---------------------------
def get_session_files(session_id: str) -> List[str]:
    """Get list of uploaded filenames for a session"""
    try:
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="metadata")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]),
            limit=1
        )
        if results[0]:
            return results[0][0].payload.get("files", [])
    except:
        pass
    return []

# ---------------------------
# Stub: MCP tools
# ---------------------------
def mcp_tool_stub_search(tool_name: str, query: str) -> Dict[str, Any]:
    return {"tool": tool_name, "query": query, "results": [], "note": "MCP tool stub - no real call performed."}

# ---------------------------
# FastAPI app + endpoints
# ---------------------------
app = FastAPI(
    title="ACTIEA - Chatbot",
    description="RAG system for aircraft technical documentation with file upload, query, and chat capabilities",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # Alternative ReDoc interface
)

# Exception handler for validation errors (handles optional files parameter)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors gracefully, especially for optional files parameter.
    If files parameter has validation error (e.g., string instead of UploadFile),
    treat it as "no files provided" for /api/upload-and-query endpoint.
    """
    # Check if this is the upload-and-query endpoint
    if request.url.path == "/api/upload-and-query":
        errors = exc.errors()
        # Check if the error is about files parameter expecting UploadFile but receiving string
        files_error = None
        for error in errors:
            if error.get("loc") and len(error["loc"]) >= 2 and error["loc"][-1] == "files":
                if "Expected UploadFile" in str(error.get("msg", "")) and "received" in str(error.get("msg", "")):
                    files_error = error
                    break
        
        # If it's a files validation error, treat as no files and continue (files is optional)
        if files_error:
            logger.warning(f"Validation error for files parameter in upload-and-query (treating as no files): {files_error}")
            # Instead of returning error, we'll let the request continue with files=None
            # We need to modify the request to remove the invalid files parameter
            # But since we can't modify the request body, we'll handle this in the endpoint itself
            # For now, just log and let it through - the endpoint will handle it
            pass
    
    # For other endpoints or other validation errors, return standard validation error
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# Add correlation ID middleware (first, so it runs before rate limiting)
app.add_middleware(CorrelationIDMiddleware)
logger.info("Correlation ID middleware enabled")

# Add rate limiting middleware
if RATE_LIMIT_ENABLED:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=RATE_LIMIT_PER_MINUTE,
        requests_per_hour=RATE_LIMIT_PER_HOUR
    )
    logger.info(f"Rate limiting middleware enabled: {RATE_LIMIT_PER_MINUTE} req/min, {RATE_LIMIT_PER_HOUR} req/hour")

# ---------------------------
# Startup event - Initialize heavy components
# ---------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize RAG components at startup"""
    global ingestor, embedder, store, initialization_complete, initialization_error
    
    logger.info("Starting API Service - Initializing components...")
    print("🔄 Initializing RAG components...", flush=True)
    
    try:
        # Initialize PDFIngestor (lightweight)
        print("📄 Initializing PDFIngestor...", flush=True)
        ingestor = PDFIngestor(base_cfg)
        logger.info("PDFIngestor initialized")
        
        # Initialize Embedder (heavy - loads model)
        print("🧠 Initializing Embedder (loading model)...", flush=True)
        embedder = Embedder(base_cfg)
        logger.info("Embedder initialized")
        
        # Initialize QdrantStore (connects to Qdrant)
        print("🗄️  Initializing QdrantStore (connecting to Qdrant)...", flush=True)
        try:
            store = QdrantStore(base_cfg)
            logger.info("QdrantStore initialized")
        except Exception as qdrant_err:
            error_msg = f"Qdrant connection failed: {qdrant_err}. Make sure Qdrant is running on {base_cfg.qdrant_host}:{base_cfg.qdrant_port}"
            logger.error(error_msg)
            print(f"⚠️  {error_msg}", flush=True)
            print(f"   💡 Start Qdrant with: docker run -d -p 6333:6333 --name qdrant qdrant/qdrant", flush=True)
            raise  # Re-raise to mark initialization as failed
        
        # Cache collection will be lazy-created on first use (non-blocking)
        # Initialize memory
        global memory
        memory = QdrantMemory(store)
        
        initialization_complete = True
        print("✅ All components initialized successfully!", flush=True)
        logger.warning("🔥 API STARTUP COMPLETE — SERVICE READY")
        logger.info("API Service ready - all components initialized")
        
    except Exception as e:
        initialization_error = str(e)
        error_msg = f"❌ FAILED to initialize components: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg, flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        # Don't raise - allow server to start but endpoints will return errors

# ---- Persistent Repository ingestion endpoint (idempotent) ----
@app.post(
    "/api/repo/ingest",
    summary="Ingest persistent repository documents (idempotent)",
    description="""
    Ingest PDF files from the configured persistent repository directory into Qdrant.
    
    **Key Feature: Idempotent Ingestion**
    - Only processes NEW or CHANGED files (by file hash)
    - Skips files that haven't changed since last ingestion
    - Never re-embeds unchanged files (saves time and compute)
    
    **What it does:**
    - Scans the persistent repository directory for PDF files
    - Computes SHA256 hash for each file
    - Checks if file was already ingested (by hash)
    - Only processes new or changed files
    - Extracts text, chunks, generates embeddings
    - Stores in Qdrant with persistent storage (not session-scoped)
    
    **Use cases:**
    - Initial setup: Populate the persistent repository
    - Incremental updates: Add new files or update existing ones
    - Re-indexing: Force re-process all files (set reindex=true)
    
    **Note:** 
    - This endpoint processes documents synchronously
    - Only new/changed files are processed (unless reindex=true)
    - Files are stored with `is_temp: false` (persistent, not session-scoped)
    """,
    responses={
        200: {
            "description": "Ingestion successful",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "indexed": 1250,
                        "skipped": 50,
                        "new_files": 10,
                        "updated_files": 5
                    }
                }
            }
        },
        400: {
            "description": "Repository directory not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Repository directory not found: /path/to/repo"
                    }
                }
            }
        }
    }
)
def repo_ingest(
    reindex: bool = Query(False, description="If true, re-indexes all documents even if already indexed"),
    request: Request = None
):
    """Idempotent ingestion of persistent repository documents."""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    logger.info(f"[corr_id={corr_id}] Starting repository ingestion (reindex={reindex})")
    """
    Idempotent ingestion of persistent repository documents.
    
    **Idempotent Behavior:**
    - Only processes NEW files (not seen before)
    - Only processes CHANGED files (hash differs from stored hash)
    - Skips UNCHANGED files (same hash as stored)
    - Use `reindex=true` to force re-processing of all files
    
    **Query Parameters:**
    
    | Parameter | Type | Required | Description |
    |-----------|------|----------|-------------|
    | reindex | boolean | No | If true, re-processes all documents regardless of hash (default: false) |
    
    **Example Request (cURL):**
    ```bash
    # Normal ingestion (only new/changed files)
    curl -X POST "http://localhost:3000/api/repo/ingest?reindex=false"
    
    # Force re-index all files
    curl -X POST "http://localhost:3000/api/repo/ingest?reindex=true"
    ```
    
    **Example Request (Python requests):**
    ```python
    import requests
    
    # Normal ingestion
    response = requests.post('http://localhost:3000/api/repo/ingest', params={'reindex': False})
    
    # Force re-index
    response = requests.post('http://localhost:3000/api/repo/ingest', params={'reindex': True})
    ```
    
    **Success Response (200 OK):**
    ```json
    {
        "status": "ok",
        "indexed": 1250,
        "skipped": 50,
        "new_files": 10,
        "updated_files": 5,
        "total_files_scanned": 60
    }
    ```
    
    **Error Response (400 Bad Request):**
    ```json
    {
        "detail": "Repository directory not found: /path/to/repo"
    }
    ```
    """
    with PerformanceMetrics("repo_ingest"):
        repo_dir = Path(base_cfg.pdf_directory)
        if not repo_dir.exists():
            raise HTTPException(status_code=400, detail=f"Repository directory not found: {repo_dir}")

        pdfs = list(repo_dir.rglob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDF files in repository")
        
        all_chunks = []
        skipped_count = 0
        new_files = 0
        updated_files = 0
        
        for pdf in pdfs:
            # Compute file hash for idempotent ingestion
            file_hash = compute_file_hash(pdf)
            
            # Check if file was already processed (unless reindex=true)
            if not reindex and is_persistent_file_processed(file_hash, pdf):
                skipped_count += 1
                logger.debug(f"Skipping unchanged file: {pdf.name} (hash: {file_hash[:8]}...)")
                continue
            
            # Check if this is a new file or updated file
            was_processed = is_persistent_file_processed(file_hash, pdf)
            if was_processed:
                updated_files += 1
                logger.info(f"Processing updated file: {pdf.name}")
            else:
                new_files += 1
                logger.info(f"Processing new file: {pdf.name}")
            
            try:
                doc = ingestor.extract(pdf)
                chunks = chunk_document(doc, base_cfg)
                
                if chunks:
                    # Mark chunks as persistent (not session-scoped)
                    for c in chunks:
                        # Ensure chunks are marked as persistent (is_temp: false)
                        c["is_temp"] = False
                        # Store original file path for reference
                        c["repo_path"] = str(pdf.relative_to(repo_dir))
                    
                    all_chunks.extend(chunks)
                    # Mark file as processed
                    mark_persistent_file_processed(file_hash, pdf, repo_dir)
                else:
                    logger.warning(f"No chunks generated for {pdf.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf.name}: {e}", exc_info=True)
                continue
        
        if not all_chunks:
            return {
                "status": "ok",
                "indexed": 0,
                "skipped": skipped_count,
                "new_files": new_files,
                "updated_files": updated_files,
                "total_files_scanned": len(pdfs)
            }
        
        embedded = embedder.embed_chunks(all_chunks)
        logger.info(f"Total chunks to index: {len(embedded)}")
        store.upsert(embedded)
        
        logger.info(f"Ingestion complete: {len(embedded)} chunks indexed, {skipped_count} files skipped")
        return {
            "status": "ok",
            "indexed": len(embedded),
            "skipped": skipped_count,
            "new_files": new_files,
            "updated_files": updated_files,
            "total_files_scanned": len(pdfs)
        }

# ---- File upload endpoint with deduplication ----
@app.post(
    "/api/upload",
    summary="Upload PDF files for File Mode or Full Mode",
    description="""
    Upload one or more PDF files to be indexed and made searchable.
    
    **Key Features:**
    - Supports multiple file uploads in a single request
    - Automatic deduplication (same file in same session is skipped)
    - Session-scoped storage (files are tied to a session_id)
    - Works for both File Mode and Full Mode queries
    
    **Request Body:**
    - `files`: One or more PDF files (multipart/form-data)
    - `session_id`: Optional session ID. If not provided, a new session is created.
    
    **Response:**
    - Returns session_id (use this for subsequent queries)
    - Lists uploaded files and ingested chunks
    - Includes skipped files if duplicates were found
    
    **Important Notes:**
    - Only PDF files are processed (other file types are silently skipped)
    - Files are stored with prefix `temp_{session_id}_` in Qdrant
    - Deduplication is session-scoped (same file can exist in different sessions)
    - Files become queryable immediately after upload completes
    """,
    responses={
        200: {
            "description": "Files uploaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "session_id": "abc123def456",
                        "uploaded_files": ["aircraft_manual.pdf", "maintenance_procedures.pdf"],
                        "ingested_chunks": 245,
                        "skipped_files": ["duplicate.pdf (duplicate of aircraft_manual.pdf)"],
                        "warnings": ["Skipped 1 duplicate file(s)"]
                    }
                }
            }
        },
        400: {
            "description": "Bad request - invalid input",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "No PDF files provided"
                    }
                }
            }
        },
        422: {
            "description": "Validation error - missing required field",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "files"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def upload_file(
    files: List[UploadFile] = File(..., description="One or more PDF files to upload"),
    session_id: Optional[str] = Form(None, description="Optional session ID. If not provided, a new session is created automatically."),
    request: Request = None
):
    """Upload multiple files with deduplication and ingest them under session-scoped doc_id prefix."""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    upload_logger.info(f"[corr_id={corr_id}] Starting file upload (files={len(files)}, session_id={session_id})")
    """
    Upload multiple files with deduplication and ingest them under session-scoped doc_id prefix.
    
    **Request Body (multipart/form-data):**
    
    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | files | File[] | Yes | One or more PDF files to upload |
    | session_id | string | No | Session ID. If omitted, a new UUID is generated |
    
    **Example Request (cURL):**
    ```bash
    curl -X POST "http://localhost:3000/api/upload" \\
      -F "files=@aircraft_manual.pdf" \\
      -F "files=@maintenance_procedures.pdf" \\
      -F "session_id=abc123def456"
    ```
    
    **Example Request (Python requests):**
    ```python
    import requests
    
    files = [
        ('files', ('aircraft_manual.pdf', open('aircraft_manual.pdf', 'rb'), 'application/pdf')),
        ('files', ('maintenance_procedures.pdf', open('maintenance_procedures.pdf', 'rb'), 'application/pdf'))
    ]
    data = {'session_id': 'abc123def456'}
    
    response = requests.post('http://localhost:3000/api/upload', files=files, data=data)
    ```
    
    **Success Response (200 OK):**
    ```json
    {
        "status": "ok",
        "session_id": "abc123def456",
        "uploaded_files": ["aircraft_manual.pdf", "maintenance_procedures.pdf"],
        "ingested_chunks": 245
    }
    ```
    
    **Response with Duplicates:**
    ```json
    {
        "status": "ok",
        "session_id": "abc123def456",
        "uploaded_files": ["aircraft_manual.pdf"],
        "ingested_chunks": 120,
        "skipped_files": ["duplicate.pdf (duplicate of aircraft_manual.pdf)"],
        "warnings": ["Skipped 1 duplicate file(s)"]
    }
    ```
    """
    async with upload_semaphore:
        with PerformanceMetrics("file_upload", session_id):
            if session_id is None:
                session_id = uuid.uuid4().hex
            else:
                session_id = str(session_id).strip()
            
            dest_dir = UPLOADS_DIR / session_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            total_chunks = 0
            uploaded_files = []
            skipped_files = []
            
            for file in files:
                if not file.filename.lower().endswith('.pdf'):
                    continue
                
                dest_path = dest_dir / file.filename
                file_content = await file.read()
                dest_path.write_bytes(file_content)
                
                # Check for duplicates
                file_hash = compute_file_hash(dest_path)
                if is_file_already_processed(file_hash, session_id):
                    existing_filename = get_filename_by_hash(file_hash, session_id)
                    skipped_files.append(f"{file.filename} (duplicate of {existing_filename})")
                    upload_logger.info(f"Skipping duplicate file: {file.filename} (hash: {file_hash[:8]}...)")
                    dest_path.unlink()  # Remove duplicate file
                    continue
                
                try:
                    # Track file processing latency
                    file_process_start = perf_counter()
                    
                    extract_start = perf_counter()
                    doc = ingestor.extract(dest_path)
                    extract_latency = perf_counter() - extract_start
                    # Log extraction info for debugging
                    total_pages = len(doc.get("pages", []))
                    upload_logger.info(f"[corr_id={corr_id}] Extracted {total_pages} pages from {file.filename} (extraction latency: {extract_latency:.3f}s)")
                    
                    chunk_start = perf_counter()
                    chunks = chunk_document(doc, base_cfg)
                    chunk_latency = perf_counter() - chunk_start
                    
                    if chunks:
                        # Log session_id for debugging
                        upload_logger.info(f"[corr_id={corr_id}] Processing {len(chunks)} chunks for session_id='{session_id}' (chunking latency: {chunk_latency:.3f}s)")
                        print(f"📝 Processing {len(chunks)} chunks for session_id='{session_id}'")
                        for c in chunks:
                            original_doc_id = c.get('doc_id', '0')
                            c["doc_id"] = f"temp_{session_id}_{original_doc_id}"
                            c["is_temp"] = True  # Mark as session-scoped (temporary)
                            c["session_id"] = session_id  # Store session_id for filtering
                            # Log first chunk's doc_id format for verification
                            if c == chunks[0]:
                                upload_logger.info(f"[corr_id={corr_id}] Sample doc_id format: '{c['doc_id']}'")
                        
                        embed_start = perf_counter()
                        embedded = embedder.embed_chunks(chunks)
                        embed_latency = perf_counter() - embed_start
                        upload_logger.info(f"[corr_id={corr_id}] Embedded {len(embedded)} chunks (embedding latency: {embed_latency:.3f}s)")
                        
                        upsert_start = perf_counter()
                        store.upsert(embedded)
                        upsert_latency = perf_counter() - upsert_start
                        upload_logger.info(f"[corr_id={corr_id}] Upserted {len(embedded)} chunks to Qdrant (upsert latency: {upsert_latency:.3f}s)")
                        
                        total_chunks += len(embedded)
                        uploaded_files.append(file.filename)
                        
                        file_process_latency = perf_counter() - file_process_start
                        upload_logger.info(f"[corr_id={corr_id}] File processing complete for {file.filename}: total latency {file_process_latency:.3f}s (extract: {extract_latency:.3f}s, chunk: {chunk_latency:.3f}s, embed: {embed_latency:.3f}s, upsert: {upsert_latency:.3f}s)")
                        
                        # Mark file as processed
                        mark_file_processed(file_hash, file.filename, session_id)
                    else:
                        # Log warning when no chunks are generated
                        warning_msg = (
                            f"⚠️ No chunks generated for {file.filename}. "
                            f"File had {total_pages} pages. "
                            f"Possible reasons: all pages had text < {base_cfg.min_chunk_size} words, "
                            f"or chunking produced segments below minimum size."
                        )
                        upload_logger.warning(warning_msg)
                        print(warning_msg)
                        # Still add to skipped files with explanation
                        skipped_files.append(f"{file.filename} (no processable content - all pages too short)")
                except Exception as e:
                    error_msg = f"Error processing {file.filename}: {e}"
                    upload_logger.error(error_msg, exc_info=True)
                    import traceback
                    print(f"❌ {error_msg}\n{traceback.format_exc()}")
                    continue
            
            # CRITICAL: Create/update session metadata even if no chunks were generated
            # This ensures the session exists for queries, even if files had no processable content
            try:
                results = store.client.scroll(
                    collection_name="sessions",
                    scroll_filter=Filter(must=[
                        FieldCondition(key="type", match=MatchValue(value="metadata")),
                        FieldCondition(key="session_id", match=MatchValue(value=session_id))
                    ]),
                    limit=1
                )
                files_list = results[0][0].payload.get("files", []) if results[0] else []
                
                # Add uploaded files (even if no chunks were generated)
                for filename in uploaded_files:
                    if filename not in files_list:
                        files_list.append(filename)
                
                # Update or create session metadata
                point_id = stable_id(f"session_meta_{session_id}")
                store.client.upsert(collection_name="sessions", points=[PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload={
                        "type": "metadata",
                        "session_id": session_id,
                        "files": files_list,
                        "file_count": len(files_list),
                        "total_chunks": total_chunks
                    }
                )])
                upload_logger.info(f"Updated session metadata: session_id={session_id}, files={len(files_list)}, chunks={total_chunks}")
            except Exception as e:
                upload_logger.warning(f"Error updating session metadata (non-critical): {e}")
            
            response = {
                "status": "ok",
                "session_id": session_id,
                "uploaded_files": uploaded_files,
                "ingested_chunks": total_chunks
            }
            if skipped_files:
                response["skipped_files"] = skipped_files
                response["warnings"] = [f"Skipped {len(skipped_files)} duplicate file(s)"]
            
            return response

# ---- Get uploaded files for a session ----
@app.get("/api/session/{session_id}/files")
def get_uploaded_files(session_id: str, request: Request = None):
    """Get list of files uploaded in this session"""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    session_logger.info(f"[corr_id={corr_id}] Getting files for session: {session_id}")
    session_id = str(session_id).strip()
    try:
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="metadata")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]),
            limit=1
        )
        files = []
        file_count = 0
        total_chunks = 0
        if results[0]:
            payload = results[0][0].payload
            files = payload.get("files", [])
            file_count = payload.get("file_count", 0)
            total_chunks = payload.get("total_chunks", 0)
        
        return {
            "session_id": session_id,
            "files": files,
            "file_count": file_count,
            "total_chunks": total_chunks
        }
    except Exception as e:
        logger.error(f"Error getting session files: {e}")
        return {"session_id": session_id, "files": [], "file_count": 0, "total_chunks": 0}

# ---- Core query endpoint with caching and performance tracking ----
@app.post(
    "/api/query",
    summary="Query the RAG system",
    description="""
    Query the RAG system to get answers from uploaded documents and/or MCP repository.
    
    **Modes:**
    - **file**: Search only session-scoped uploaded files (requires session_id)
    - **repo**: Search only persistent repository (no session_id needed)
    - **full**: Search both persistent repository AND uploaded session files (session_id optional)
    
    **Features:**
    - Automatic query caching (responses cached for 1 hour)
    - Conversation history support (remembers previous queries in session)
    - Answer quality evaluation (F1, BERTScore)
    - GPU memory management (handles OOM errors gracefully)
    - Reranking (combines vector similarity + keyword overlap)
    
    **Response includes:**
    - LLM-generated answer
    - Source documents with page numbers and scores
    - Quality metrics (if available)
    """,
    responses={
        200: {
            "description": "Query successful",
            "content": {
                "application/json": {
                    "example": {
                        "user_query": "What is the damage limit for overwing panel beams?",
                        "llm_response": "The damage limit for overwing panel beams is 325mm...",
                        "openai_payload": {
                            "model": "qwen",
                            "top_k_candidates": 20,
                            "sources": [
                                {
                                    "filename": "aircraft_manual.pdf",
                                    "page": 45,
                                    "score": 0.92
                                }
                            ]
                        }
                    }
                }
            }
        },
        400: {
            "description": "Bad request - invalid mode or missing required fields",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "session_id required for file mode"
                    }
                }
            }
        },
        404: {
            "description": "Session not found (File Mode)",
            "content": {
                "application/json": {
                    "example": {
                        "user_query": "What is the damage limit?",
                        "llm_response": "⚠️ **No files found for Session ID 'abc123'**...",
                        "error": "SESSION_NOT_FOUND",
                        "session_id": "abc123"
                    }
                }
            }
        },
        507: {
            "description": "Insufficient Storage - GPU Out of Memory",
            "content": {
                "application/json": {
                    "example": {
                        "user_query": "What is the damage limit?",
                        "llm_response": "⚠️ **GPU Memory Error**...",
                        "error": "CUDA_OOM",
                        "mode": "file"
                    }
                }
            }
        },
        503: {
            "description": "Service Unavailable - Qdrant connection error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Qdrant connection error: Connection refused"
                    }
                }
            }
        }
    }
)
async def api_query(req: QueryRequest, request: Request = None):
    """
    Query endpoint - refactored for maintainability.
    Returns machine-readable responses (no UX formatting).
    """
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    query_total_start = perf_counter()
    async with query_semaphore:
        with PerformanceMetrics("api_query", req.session_id):
            try:
                logger.info(f"[corr_id={corr_id}] Processing query: {req.query[:100]}...")
                
                # 0. Check for greetings (skip retrieval for simple greetings)
                q_lower = req.query.strip().lower()
                greeting_patterns = [
                    r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[!?.]*$',
                    r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\s+(there|everyone|all)[!?.]*$',
                    r'^(how\s+can\s+you\s+help|what\s+can\s+you\s+do|help\s+me)[!?.]*$'
                ]
                is_greeting = any(re.match(pattern, q_lower) for pattern in greeting_patterns)
                
                if is_greeting:
                    logger.info(f"[corr_id={corr_id}] Detected greeting - skipping retrieval")
                    greeting_response = "Hello! How can I assist you with aircraft technical documentation today? I can help you with questions about maintenance procedures, damage assessment, repair guidelines, and technical specifications from uploaded documents."
                    return JSONResponse({
                        "user_query": req.query,
                        "llm_response": greeting_response,
                        "openai_payload": {
                            "model": req.model or DEFAULT_LLM,
                            "top_k_candidates": 0,
                            "sources": [],
                            "greeting_detected": True
                        }
                    })
                
                # 1. Validate request
                mode = req.mode.lower()
                q, session_id, model_key = validate_query_request(req, mode)
                top_k = req.top_k or base_cfg.top_k_initial
                
                # 2. Check cache
                if not req.no_cache:
                    query_hash = compute_query_hash(q, mode, session_id)
                    cached_response = get_cached_response(query_hash)
                    if cached_response:
                        query_logger.info(f"Cache HIT for query: {q[:50]}...")
                        return JSONResponse(cached_response)
                else:
                    query_hash = compute_query_hash(q, mode, session_id)
                    query_logger.info(f"Cache bypassed for query: {q[:50]}...")
                
                # 3. Validate session exists (file mode only)
                if mode == "file" and not validate_session_exists(session_id, mode):
                            return JSONResponse(
                                status_code=404,
                                content={
                                    "user_query": q,
                            "llm_response": "",
                                    "error": "SESSION_NOT_FOUND",
                                    "session_id": session_id,
                                    "openai_payload": {"candidates_returned": 0}
                                }
                            )
                
                # 4. Get conversation history (for retrieval only, NOT for generation)
                hist_session = get_history_session(mode, session_id)
                try:
                    conversation_history = memory.get_history(hist_session, limit=10)
                    logger.info(f"Retrieved {len(conversation_history)} history messages for retrieval")
                except Exception as e:
                    logger.warning(f"Error retrieving history: {e}")
                    conversation_history = []
                
                # 5. Embed query
                query_embed_start = perf_counter()
                qvec_bytes = embedder.embed_query(q)
                query_embed_latency = perf_counter() - query_embed_start
                logger.info(f"[corr_id={corr_id}] Query embedding latency: {query_embed_latency:.3f}s")
                
                # 6. Retrieve and rerank (history-aware retrieval DISABLED - uses only current query)
                candidates, top_chunks, retrieve_latency, rerank_latency = retrieve_and_rerank(mode, q, qvec_bytes, session_id, top_k, conversation_history, corr_id)
                
                # If no results found, return error (no LLM call)
                if not candidates:
                    logger.warning(f"[corr_id={corr_id}] No results found for query")
                    return JSONResponse({
                        "user_query": q,
                        "llm_response": "",
                        "error": "NO_RESULTS",
                        "mode": mode,
                        "session_id": session_id if mode == "file" else None,
                        "openai_payload": {
                            "candidates_returned": 0
                        }
                    })

                # 7. Check out-of-context
                if detect_out_of_context(top_chunks):
                    top_score = top_chunks[0].get("final_score", 0.0)
                    logger.warning(f"Out-of-context query: score {top_score:.3f} < {OUT_OF_CONTEXT_THRESHOLD}")
                    return JSONResponse({
                        "user_query": q,
                        "llm_response": "",
                        "error": "OUT_OF_CONTEXT",
                        "mode": mode,
                        "session_id": session_id if mode == "file" else None,
                        "openai_payload": {
                            "top_k_candidates": len(candidates),
                            "top_score": round(top_score, 4),
                            "threshold": OUT_OF_CONTEXT_THRESHOLD
                        }
                    })

                # 8. Build prompt context (stateless - NO history in generation)
                context_start = perf_counter()
                gpu_tier = detect_gpu_tier()
                context = build_prompt_context(mode, q, top_chunks, model_key, conversation_history, gpu_tier)
                context_latency = perf_counter() - context_start
                logger.info(f"[corr_id={corr_id}] Context built: {len(context)} chars ({len(context) // 4} estimated tokens) (latency: {context_latency:.3f}s)")
                
                # 9. Call LLM service
                try:
                    llm_start = perf_counter()
                    answer = await call_llm_service(context, q, model_key)
                    llm_latency = perf_counter() - llm_start
                    logger.info(f"[corr_id={corr_id}] LLM generation latency: {llm_latency:.3f}s, response length: {len(answer)} chars")
                except Exception as e:
                    # CUDA errors should be unreachable (context is hard-capped)
                    # But handle gracefully if they occur
                    error_str = str(e).lower()
                    is_oom = "out of memory" in error_str or "cuda" in error_str
                    is_cublas = "cublas" in error_str
                    
                    if is_oom or is_cublas:
                        logger.error(f"CUDA error (should not happen): {e}")
                        if torch and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return JSONResponse(
                            status_code=507,
                            content={
                                "user_query": q,
                                "llm_response": "",
                                "error": "CUDA_OOM",
                                "mode": mode,
                                "session_id": session_id if mode == "file" else None,
                                "openai_payload": {"candidates_returned": len(candidates), "error_type": "out_of_memory"}
                            }
                        )
                    else:
                        logger.error(f"LLM generation failed: {e}")
                        raise

                # 10. Save to memory
                try:
                    memory.append_user(hist_session, q)
                    memory.append_assistant(hist_session, answer)
                except Exception as e:
                    logger.warning(f"Error saving to memory: {e} (non-critical)")

                # 11. Build response
                query_total_latency = perf_counter() - query_total_start
                openai_payload = {
                    "model": model_key,
                    "top_k_candidates": len(candidates),
                    "top_k_final": len(top_chunks),
                    "latency_breakdown": {
                        "query_embedding": round(query_embed_latency, 3),
                        "retrieval": round(retrieve_latency, 3),
                        "reranking": round(rerank_latency, 3),
                        "context_building": round(context_latency, 3),
                        "llm_generation": round(llm_latency, 3),
                        "total": round(query_total_latency, 3)
                    },
                    "sources": [
                        {
                            "filename": c.get("filename", ""),
                            "page_num": c.get("page_num", 0),
                            "score": round(c.get("final_score", 0.0), 4),
                            "text": c.get("text", "")[:200] + "..." if len(c.get("text", "")) > 200 else c.get("text", "")
                        }
                        for c in top_chunks[:5]
                    ]
                }
                
                logger.info(f"[corr_id={corr_id}] Query processing complete: total latency {query_total_latency:.3f}s (embed: {query_embed_latency:.3f}s, retrieve: {retrieve_latency:.3f}s, rerank: {rerank_latency:.3f}s, context: {context_latency:.3f}s, llm: {llm_latency:.3f}s)")
                response = {"user_query": q, "llm_response": answer, "openai_payload": openai_payload}
                
                # 12. Evaluate quality in background (non-blocking, fire-and-forget)
                if quality_evaluator:
                    import asyncio
                    # Create background task - doesn't block response
                    asyncio.create_task(evaluate_quality_async(quality_evaluator, answer, q, top_chunks))
                
                # 13. Cache and return
                cache_response(query_hash, response, ttl=3600)
                return JSONResponse(response)
            except (ConnectionError, TimeoutError) as conn_err:
                logger.error(f"Qdrant connection error: {conn_err}")
                raise HTTPException(
                    status_code=503,
                    detail={"error": "QDRANT_CONNECTION_ERROR", "message": str(conn_err)}
                )
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Error in /api/query: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail={"error": "INTERNAL_ERROR", "message": str(e)}
                )

# ---- Health check / diagnostic endpoint ----
@app.get(
    "/api/health",
    summary="Check system health and status",
    description="""
    Diagnostic endpoint to check the health and status of the RAG system.
    
    **Returns:**
    - Qdrant connection status
    - LLM model loading status
    - GPU availability and memory usage
    
    **Use cases:**
    - Health checks for monitoring
    - Debugging connection issues
    - Verifying system readiness
    """,
    responses={
        200: {
            "description": "Health check successful - all dependencies ready",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "qdrant": "connected",
                        "llm_models": {
                            "qwen": {
                                "loaded": True,
                                "model_id": "Qwen/Qwen2.5-3B-Instruct"
                            }
                        },
                        "gpu_available": True,
                        "gpu_memory": {
                            "total": "16.00 GB",
                            "allocated": "3.60 GB",
                            "reserved": "4.20 GB",
                            "free": "11.80 GB"
                        }
                    }
                }
            }
        },
        503: {
            "description": "Service Unavailable - dependencies not ready",
            "content": {
                "application/json": {
                    "example": {
                        "status": "initializing",
                        "detail": "Dependencies not ready",
                        "qdrant": "not connected",
                        "llm_service": "not connected"
                    }
                }
            }
        }
    }
)
def health_check(request: Request = None):
    """
    Check system health and LLM status.
    
    **Example Request (cURL):**
    ```bash
    curl -X GET "http://localhost:3000/api/health"
    ```
    
    **Example Request (Python requests):**
    ```python
    import requests
    
    response = requests.get('http://localhost:3000/api/health')
    print(response.json())
    ```
    
    **Success Response (200 OK):**
    ```json
    {
        "status": "ok",
        "qdrant": "connected",
        "llm_models": {
            "qwen": {
                "loaded": True,
                "model_id": "Qwen/Qwen2.5-3B-Instruct"
            }
        },
        "gpu_available": True,
        "gpu_memory": {
            "total": "16.00 GB",
            "allocated": "3.60 GB",
            "reserved": "4.20 GB",
            "free": "11.80 GB"
        }
    }
    ```
    
    **Error Response (Qdrant disconnected):**
    ```json
    {
        "status": "ok",
        "qdrant": "error: Connection refused",
        "llm_models": {
            "qwen": {
                "loaded": True,
                "model_id": "Qwen/Qwen2.5-3B-Instruct"
            }
        },
        "gpu_available": True,
        "gpu_memory": {
            "total": "16.00 GB",
            "allocated": "3.60 GB",
            "reserved": "4.20 GB",
            "free": "11.80 GB"
        }
    }
    ```
    """
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    logger.info(f"[corr_id={corr_id}] Health check requested")
    
    # Initialize health status
    health_status = {
        "status": "ok",
        "qdrant": "unknown",
        "llm_service": "unknown",
        "llm_models": {},
        "gpu_available": False,
        "gpu_memory": {}
    }
    
    # Track readiness with explicit flags (do NOT mutate dependencies_ready until the end)
    qdrant_ready = False
    llm_service_ready = False
    
    # Check if initialization is complete
    if not initialization_complete:
        if initialization_error:
            health_status["status"] = "initializing"
            health_status["error"] = initialization_error
            health_status["message"] = "Service is starting up but encountered an error during initialization"
        else:
            health_status["status"] = "initializing"
            health_status["message"] = "Service is starting up, components are being initialized"
    
    # Check Qdrant (independent of store - check dependency directly)
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qc.get_collections()
        health_status["qdrant"] = "connected"
        qdrant_ready = True
    except Exception as e:
        health_status["qdrant"] = f"error: {str(e)}"
    
    # Check LLM service
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            llm_health = client.get(f"{LLM_SERVICE_URL}/health")
            if llm_health.status_code == 200:
                llm_status = llm_health.json()
                health_status["llm_service"] = "connected"
                health_status["llm_models"] = llm_status.get("available_models", [])
                health_status["llm_service_gpu"] = llm_status.get("gpu_available", False)
                health_status["llm_service_gpu_memory"] = llm_status.get("gpu_memory", {})
                llm_service_ready = True
            else:
                health_status["llm_service"] = f"error: HTTP {llm_health.status_code}"
    except Exception as e:
        health_status["llm_service"] = f"error: {str(e)}"
        health_status["llm_models"] = []
    
    # Note: GPU is now managed by LLM service, not API service
    health_status["gpu_available"] = False
    health_status["gpu_memory"] = {"note": "GPU managed by LLM service"}
    
    # --- Determine readiness AFTER all checks (single source of truth) ---
    dependencies_ready = (
        initialization_complete
        and qdrant_ready
        and llm_service_ready
    )
    
    # Return 503 if dependencies are not ready (for readiness probe)
    # Return 200 if all dependencies are ready (for liveness probe)
    if not dependencies_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "initializing",
                "message": "Dependencies not ready",
                "initialization_complete": initialization_complete,
                "qdrant": health_status["qdrant"],
                "llm_service": health_status["llm_service"]
            }
        )
    
    return health_status

# ---- Liveness probe endpoint (always returns 200) ----
@app.get(
    "/api/live",
    summary="Liveness probe",
    description="Simple liveness check that always returns 200 to indicate the container is alive.",
    responses={
        200: {
            "description": "Container is alive",
            "content": {
                "application/json": {
                    "example": {"status": "alive"}
                }
            }
        }
    }
)
def liveness_check():
    """Liveness probe - always returns 200 (container is alive)"""
    return {"status": "alive"}

# ---- Combined Upload + Query Endpoint ----
@app.post(
    "/api/upload-and-query",
    summary="Upload files and query in one request",
    description="""
    Combined endpoint that uploads files (optional) and immediately queries them.
    Useful for testing and quick workflows.
    
    **Request (multipart/form-data):**
    - `files`: Optional PDF files to upload (if not provided, session_id is required)
    - `query`: The question to ask about the uploaded files (required)
    - `session_id`: Optional session ID (required if files not provided, auto-generated if files provided)
    - `model`: Optional LLM model (default: qwen)
    - `mode`: Optional mode - 'file' (default) or 'full'
    
    **Response:**
    - Upload results (session_id, uploaded_files, ingested_chunks) - if files were uploaded
    - Query results (llm_response, sources, etc.)
    
    **Usage:**
    - With files: Upload files and query them immediately
    - Without files: Query existing session (provide session_id)
    """,
    responses={
        200: {
            "description": "Upload and query successful",
            "content": {
                "application/json": {
                    "example": {
                        "upload": {
                            "status": "ok",
                            "session_id": "abc123",
                            "uploaded_files": ["file1.pdf"],
                            "ingested_chunks": 50
                        },
                        "query": {
                            "user_query": "What is the damage limit?",
                            "llm_response": "Based on the documents...",
                            "openai_payload": {
                                "model": "qwen",
                                "sources": [
                                    {
                                        "filename": "document.pdf",
                                        "page": 1,
                                        "score": 0.95
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
    }
)
async def upload_and_query(
    files: Optional[List[UploadFile]] = File(None, description="Optional PDF files to upload"),
    query: str = Form(..., description="Question to ask about the uploaded files"),
    session_id: Optional[str] = Form(None, description="Optional session ID (required if files not provided)"),
    model: Optional[str] = Form(None, description="Optional LLM model"),
    mode: Optional[str] = Form("file", description="Query mode: 'file' or 'full'"),
    request: Request = None
):
    """
    Combined endpoint: Upload files (optional) and query them immediately.
    Perfect for testing and quick workflows.
    
    - If files are provided: uploads them and queries
    - If files are not provided: queries existing session (session_id required)
    """
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    
    # Handle files parameter - FastAPI now validates it as Optional[List[UploadFile]]
    # So files will be either None or a list of UploadFile objects
    if files is None:
        files = []
    files_count = len(files)
    logger.info(f"[corr_id={corr_id}] Upload and query - files={files_count}, query={query[:50]}...")
    
    # Check initialization
    if not initialization_complete:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service initializing",
                "message": "Service is still starting up. Please wait and try again.",
                "status": "initializing"
            }
        )
    
    try:
        upload_data = None
        
        # Step 1: Upload files if provided (reuse existing upload logic)
        if files_count > 0:
            upload_result = await upload_file(files=files, session_id=session_id, request=request)
            
            # Handle upload result (could be JSONResponse or dict)
            if isinstance(upload_result, JSONResponse):
                import json
                upload_body = upload_result.body
                if isinstance(upload_body, bytes):
                    upload_data = json.loads(upload_body.decode('utf-8'))
                else:
                    upload_data = upload_body
            else:
                upload_data = upload_result
            
            # Check for upload errors
            if isinstance(upload_data, dict) and upload_data.get("status") != "ok":
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Upload failed",
                        "upload_result": upload_data
                    }
                )
            
            # Extract session_id from upload result
            session_id = upload_data.get("session_id")
            if not session_id:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Upload failed",
                        "message": "File upload completed but no session_id was returned",
                        "upload_result": upload_data
                    }
                )
            
            # Small delay to ensure files are fully ingested
            await asyncio.sleep(0.5)
        else:
            # No files provided - use existing session_id
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Missing required parameter",
                        "message": "Either 'files' or 'session_id' must be provided"
                    }
                )
            upload_data = {
                "status": "ok",
                "session_id": session_id,
                "uploaded_files": [],
                "ingested_chunks": 0,
                "note": "No files uploaded - querying existing session"
            }
        
        # Step 2: Query immediately (reuse existing query logic)
        query_request = QueryRequest(
            query=query,
            mode=mode or "file",
            session_id=session_id,
            model=model,
            no_cache=False
        )
        
        query_result = await api_query(query_request, request)
        
        # Handle query result (could be JSONResponse or dict)
        if isinstance(query_result, JSONResponse):
            import json
            query_body = query_result.body
            if isinstance(query_body, bytes):
                query_data = json.loads(query_body.decode('utf-8'))
            else:
                query_data = query_body if isinstance(query_body, dict) else json.loads(query_body)
        else:
            query_data = query_result
        
        # Combine results
        return JSONResponse({
            "upload": upload_data,
            "query": query_data,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"[corr_id={corr_id}] Error in upload_and_query: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Upload and query failed",
                "message": str(e),
                "session_id": session_id if 'session_id' in locals() else None
            }
        )

# ---- Session history endpoint ----
@app.get("/api/session/{session_id}/history")
def get_session_history(session_id: str, limit: int = 100, request: Request = None):
    """Get conversation history for a session"""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    session_logger.info(f"[corr_id={corr_id}] Getting history for session: {session_id}")
    hist = memory.get_history(session_id, limit=limit)
    return {"session_id": session_id, "history": hist}

# ---- Restore session endpoint ----
@app.get("/api/session/{session_id}/restore")
def restore_session(session_id: str, request: Request = None):
    """Restore a session's chat history and file information"""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    session_logger.info(f"[corr_id={corr_id}] Restoring session: {session_id}")
    
    # Check if session exists (has metadata or chunks)
    try:
        session_check = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="metadata")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]),
            limit=1
        )
        session_exists = len(session_check[0]) > 0 if session_check[0] else False
        
        chunk_check = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="is_temp", match=MatchValue(value=True))
            ]),
            limit=1
        )
        chunks_exist = len(chunk_check[0]) > 0 if chunk_check[0] else False
        
        if not session_exists and not chunks_exist:
            session_logger.warning(f"[corr_id={corr_id}] Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found or expired")
    except HTTPException:
        raise
    except Exception as e:
        session_logger.warning(f"[corr_id={corr_id}] Could not verify session existence: {e}")
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    history = memory.get_history(session_id, limit=100)
    files = get_session_files(session_id)
    
    # Get session metadata
    try:
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="metadata")),
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ]),
            limit=1
        )
        if results[0]:
            payload = results[0][0].payload
            file_count = payload.get("file_count", 0)
            total_chunks = payload.get("total_chunks", 0)
            # Try to get last_updated from metadata (if stored)
            last_updated = payload.get("last_updated", int(time.time()))
        else:
            file_count = len(files)
            total_chunks = 0
            last_updated = int(time.time())
    except Exception as e:
        session_logger.warning(f"[corr_id={corr_id}] Could not get session metadata: {e}")
        file_count = len(files)
        total_chunks = 0
        last_updated = int(time.time())
    
    session_logger.info(f"[corr_id={corr_id}] Session restored: {session_id} ({len(files)} files, {len(history)} messages)")
    
    return {
        "session_id": session_id,
        "history": history,
        "files": files,
        "file_count": file_count,
        "total_chunks": total_chunks,
        "last_updated": last_updated
    }

# ---- List all sessions endpoint ----
@app.get("/api/sessions")
def list_sessions(request: Request = None):
    """List all active sessions"""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    session_logger.info(f"[corr_id={corr_id}] Listing all sessions")
    
    sessions = []
    try:
        # Get all session metadata
        results = store.client.scroll(
            collection_name="sessions",
            scroll_filter=Filter(must=[
                FieldCondition(key="type", match=MatchValue(value="metadata"))
            ]),
            limit=1000  # Adjust if you have more sessions
        )
        
        if results[0]:
            for point in results[0]:
                try:
                    payload = point.payload if hasattr(point, 'payload') else {}
                    session_id = payload.get("session_id")
                    if session_id:
                        file_count = payload.get("file_count", 0)
                        # Try to get last_updated, default to current time if not available
                        last_updated = payload.get("last_updated", int(time.time()))
                        
                        sessions.append({
                            "session_id": session_id,
                            "file_count": file_count,
                            "last_updated": last_updated
                        })
                except Exception as e:
                    session_logger.warning(f"[corr_id={corr_id}] Error processing session point: {e}")
                    continue
    except Exception as e:
        session_logger.error(f"[corr_id={corr_id}] Error listing sessions: {e}", exc_info=True)
        return {"sessions": []}
    
    # Sort by last_updated (most recent first)
    sessions.sort(key=lambda x: x.get("last_updated", 0), reverse=True)
    session_logger.info(f"[corr_id={corr_id}] Found {len(sessions)} active sessions")
    return {"sessions": sessions}

# ---- End session cleanup ----
@app.post("/api/session/end")
def end_session(session_id: str = Form(...), request: Request = None):
    """End session and clean up all session data"""
    corr_id = getattr(request.state, 'correlation_id', 'N/A') if request else 'N/A'
    session_logger.info(f"[corr_id={corr_id}] Ending session: {session_id}")
    try:
        store.client.delete(
            collection_name=store.collection_name,
            points_selector=Filter(must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="is_temp", match=MatchValue(value=True))
            ])
        )
    except Exception as e:
        logger.warning(f"Could not delete chunks for session {session_id}: {e}")
    
    try:
        store.client.delete(
            collection_name="sessions",
            points_selector=Filter(must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            ])
        )
    except Exception as e:
        logger.warning(f"Could not delete session metadata for {session_id}: {e}")
    
    p = UPLOADS_DIR / session_id
    if p.exists():
        shutil.rmtree(p)
    return {"status": "ok", "deleted": "session cleared"}

# ---------------------------
# Gradio UI (same as original)
# ---------------------------
def gr_upload_fn(file_objs, session_id):
    """Handle multiple file uploads"""
    if not file_objs or len(file_objs) == 0:
        error_msg = "❌ No files selected. Please select at least one PDF file."
        return session_id or "", error_msg, session_id or ""
    
    files = []
    file_handles = []
    try:
        for file_obj in file_objs:
            if file_obj is not None:
                fh = open(file_obj.name, "rb")
                file_handles.append(fh)
                files.append(("files", (Path(file_obj.name).name, fh, "application/pdf")))
        
        if not files:
            error_msg = "❌ No valid PDF files found."
            return session_id or "", error_msg, session_id or ""
        
        data_dict = {}
        session_id_to_send = str(session_id).strip() if session_id else None
        if session_id_to_send:
            data_dict["session_id"] = session_id_to_send
        
        r = requests.post(
            f"http://localhost:{APP_PORT}/api/upload", 
            files=files,
            data=data_dict,
            timeout=300
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Connection error: {str(e)}"
        return session_id or "", error_msg, session_id or ""
    except Exception as e:
        error_msg = f"❌ Upload failed: {str(e)}"
        return session_id or "", error_msg, session_id or ""
    finally:
        for fh in file_handles:
            try:
                fh.close()
            except:
                pass
    
    new_session_id = data.get("session_id", session_id)
    uploaded_files = data.get("uploaded_files", [])
    skipped_files = data.get("skipped_files", [])
    chunks = data.get("ingested_chunks", 0)
    
    if uploaded_files:
        status_msg = f"✅ Uploaded {len(uploaded_files)} file(s)\n"
        status_msg += f"📄 Files: {', '.join(uploaded_files)}\n"
        status_msg += f"📊 Total chunks indexed: {chunks}\n"
        status_msg += f"🔑 Session ID: {new_session_id}\n"
        if skipped_files:
            status_msg += f"\n⚠️ Skipped {len(skipped_files)} file(s):\n"
            for skipped in skipped_files:
                status_msg += f"  • {skipped}\n"
        return new_session_id, status_msg, new_session_id
    else:
        # No files were processed - provide detailed explanation
        if skipped_files:
            error_msg = f"⚠️ **All files were skipped**\n\n"
            error_msg += f"**Reason:** All {len(skipped_files)} file(s) were already processed in this session.\n\n"
            error_msg += f"**Skipped files:**\n"
            for skipped in skipped_files:
                error_msg += f"  • {skipped}\n"
            error_msg += f"\n💡 **Tip:** These files are already indexed and available for querying. "
            error_msg += f"Upload different files or use a new Session ID to process them again."
        else:
            error_msg = "⚠️ **No files were processed**\n\n"
            error_msg += "**Possible reasons:**\n"
            error_msg += "  • No PDF files were provided\n"
            error_msg += "  • Files contained no processable content\n"
            error_msg += "  • All files were below the minimum size threshold\n\n"
            error_msg += "💡 **Tip:** Ensure you're uploading valid PDF files with readable text content."
        return new_session_id or session_id or "", error_msg, new_session_id or session_id or ""

def gr_ask_fn(message, history, mode, session_id_input, model_choice):
    """Gradio query handler"""
    session_id = str(session_id_input).strip() if session_id_input else None
    
    if history is None:
        history = []
    
    history_lists = []
    if history:
        if isinstance(history[0], dict):
            user_msg = ""
            for h in history:
                if h.get("role") == "user":
                    user_msg = h.get("content", "")
                elif h.get("role") == "assistant":
                    assistant_msg = h.get("content", "")
                    history_lists.append([user_msg, assistant_msg])
                    user_msg = ""
        elif isinstance(history[0], (tuple, list)) and len(history[0]) == 2:
            history_lists = [[h[0], h[1]] if isinstance(h, tuple) else h for h in history]
    
    history = history_lists
    
    if mode == "file":
        if not session_id:
            history.append(["", (
                "⚠️ **Session ID Required for File Mode**\n\n"
                "To query uploaded files, you need a Session ID.\n\n"
                "**Please:**\n"
                "1. Upload PDF files using the 'Upload Files' button\n"
                "2. A Session ID will be generated automatically\n"
                "3. Use that Session ID for your queries\n\n"
                "💡 **Tip:** You can leave the Session ID field empty when uploading to create a new session."
            )])
            return history, session_id_input
    
    payload = {"query": message, "mode": mode, "session_id": session_id, "model": model_choice}
    try:
        r = requests.post(f"http://localhost:{APP_PORT}/api/query", json=payload, timeout=1200)
        if r.status_code != 200:
            error_detail = r.text
            try:
                error_json = r.json()
                # Prioritize user-friendly llm_response message, fallback to detail or raw text
                error_detail = error_json.get("llm_response") or error_json.get("detail") or error_detail
            except:
                pass
            # Format error message nicely - remove technical JSON if it's just the raw response
            if error_detail.startswith("{") and "llm_response" in error_detail:
                try:
                    parsed = r.json()
                    error_detail = parsed.get("llm_response", error_detail)
                except:
                    pass
            history.append(["", error_detail if error_detail else f"❌ Error {r.status_code}: An error occurred. Please check server logs."])
            return history, session_id_input
    except requests.exceptions.RequestException as e:
        history.append(["", f"❌ Connection error: {str(e)}"])
        return history, session_id_input
    
    data = r.json()
    answer = data.get("llm_response", "No answer returned.")
    
    sources = data.get("openai_payload", {}).get("sources", [])
    if sources and mode == "file":
        unique_files = list(set([s.get("filename", "") for s in sources if s.get("filename")]))
        if unique_files:
            answer += f"\n\n📎 Sources: {', '.join(unique_files)}"
    
    history.append([message, answer])
    return history, session_id_input

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ✈️ ACTIEA - ChatBot")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot", height=500, allow_tags=False)
            txt = gr.Textbox(placeholder="Ask a question...", lines=2)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=1):
            mode_radio = gr.Radio(["file", "repo", "full"], value="repo", label="Mode")
            # Get available models from config.yaml
            available_models = list(cfg_yaml["llm"].get("models", {}).keys())
            if not available_models:
                available_models = ["qwen"]  # Fallback
            model_choice = gr.Radio(available_models, value=DEFAULT_LLM, label="LLM")
            
            gr.Markdown("### 📁 File Upload (File Mode)")
            gr.Markdown("**Session ID:** (Leave empty for new session)")
            session_id_input = gr.Textbox(label="Session ID (Optional)", placeholder="Leave empty for new session", value="", interactive=True)
            file_in = gr.File(label="Upload PDF(s)", file_count="multiple", file_types=[".pdf"])
            upload_btn = gr.Button("Upload Files", variant="secondary")
            upload_out = gr.Textbox(label="Upload Status", interactive=False, lines=8)
            
            session_state = gr.State("")

    upload_btn.click(
        fn=gr_upload_fn, 
        inputs=[file_in, session_id_input],
        outputs=[session_state, upload_out, session_id_input]
    )
    send.click(
        fn=gr_ask_fn, 
        inputs=[txt, chatbot, mode_radio, session_id_input, model_choice],
        outputs=[chatbot, session_id_input]
    )
    txt.submit(
        fn=gr_ask_fn, 
        inputs=[txt, chatbot, mode_radio, session_id_input, model_choice],
        outputs=[chatbot, session_id_input]
    )
    clear.click(lambda: ([], ""), outputs=[chatbot, session_state])

app = mount_gradio_app(app, demo, path=GRADIO_PATH)

# ---------------------------
# Helper function to get accessible IP addresses
# ---------------------------
def get_accessible_ips():
    """Get all accessible IP addresses for the server"""
    ips = []
    hostname = socket.gethostname()
    
    # Get localhost
    ips.append(("localhost", "127.0.0.1"))
    
    # Get hostname IP
    try:
        hostname_ip = socket.gethostbyname(hostname)
        if hostname_ip not in ["127.0.0.1", "127.0.1.1"]:
            ips.append((hostname, hostname_ip))
    except:
        pass
    
    # Get all network interface IPs
    try:
        import subprocess
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            for ip in result.stdout.strip().split():
                ip = ip.strip()
                if ip and ip not in ["127.0.0.1", "::1"]:
                    # Check if we already have this IP
                    if not any(existing_ip == ip for _, existing_ip in ips):
                        ips.append((f"Network interface", ip))
    except:
        pass
    
    # Fallback: try socket connection method
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        if local_ip not in ["127.0.0.1", "127.0.1.1"]:
            if not any(existing_ip == local_ip for _, existing_ip in ips):
                ips.append(("Primary network", local_ip))
    except:
        pass
    
    return ips

# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Display accessible URLs
    print("\n" + "=" * 80)
    print("🚀 SERVER STARTING")
    print("=" * 80)
    print(f"📡 Server bound to: {APP_HOST}:{APP_PORT}")
    print(f"📱 Gradio UI path: {GRADIO_PATH}")
    print("\n🌐 ACCESSIBLE ENDPOINTS (use these URLs to connect):")
    print("-" * 80)
    
    accessible_ips = get_accessible_ips()
    if accessible_ips:
        for label, ip in accessible_ips:
            print(f"\n  📍 {label}:")
            print(f"     🎨 Gradio UI:     http://{ip}:{APP_PORT}{GRADIO_PATH}")
            print(f"     📚 Swagger Docs:  http://{ip}:{APP_PORT}/docs")
            print(f"     📖 ReDoc:         http://{ip}:{APP_PORT}/redoc")
            print(f"     🔌 API Base:      http://{ip}:{APP_PORT}/api/")
    else:
        print("  ⚠️  Could not detect IP addresses automatically")
        print(f"  💡 Try: http://127.0.0.1:{APP_PORT}{GRADIO_PATH} (local access)")
        print(f"  💡 Or find your IP with: hostname -I")
    
    logger.info(f"🚀 Starting enhanced server on {APP_HOST}:{APP_PORT}")
    logger.info(f"📱 Gradio UI will be at: http://{APP_HOST}:{APP_PORT}{GRADIO_PATH}")
    
    # Log all accessible URLs
    for label, ip in accessible_ips:
        logger.info(f"🌐 Accessible URL ({label}): http://{ip}:{APP_PORT}{GRADIO_PATH}")
    
    #uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
    print("⚠️ Do not run this file directly. Use Uvicorn CLI.")


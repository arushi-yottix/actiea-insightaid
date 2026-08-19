#!/usr/bin/env python3
"""
LLM Service - LLM Inference Module
Dedicated module for LLM model loading and text generation.
Supports both local model loading (with GPU) and remote API calls (Ollama, vLLM, etc.).

Current model: Qwen/Qwen2.5-7B-Instruct (7B - optimized for T4 GPU, uses ~7-8GB)
Previous models tested:
- mlx-community/Llama-3.1-8B-Instruct - 8B model, too large for T4 (uses ~12.7GB)
- meta-llama/Llama-3.1-8B-Instruct - could not use (gated)

Note: RAG components (PDF ingestion, embeddings, Qdrant) are handled by API_wrapper service.
This module focuses solely on LLM inference.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Conditional imports for local model mode (not needed for remote mode)
try:
    import torch
    from transformers import pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    pipeline = None

# HTTP client for remote LLM API
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  httpx not available - remote LLM API will not work. Install with: pip install httpx", flush=True)

# Load config.yaml
import yaml
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    raise RuntimeError("Please create config.yaml in the project root (see example).")

with open(CONFIG_PATH, "r") as fh:
    cfg_yaml = yaml.safe_load(fh) 
    
# ----------------------------
# CONFIGURATION
# ----------------------------
@dataclass
class RAGConfig:
    # Note: This class is named RAGConfig for compatibility, but in LLM Service context,
    # it only contains LLM-related configurations. RAG components (PDF, embeddings, Qdrant)
    # are handled by the API_wrapper service, not this LLM Service.

    # LLM
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"  # Mistral-7B-Instruct-v0.3 - Apache 2.0, no gating
    # Options: Mistral-7B-Instruct-v0.3 (recommended, no license needed), Llama-3.1-8B (requires license), Qwen2.5-7B
    # Note: Mistral uses Apache 2.0 license - no HuggingFace license acceptance required
    max_new_tokens: int = 500  # Increased from 400 to 500 for more detailed responses
    temperature: float = 0.1  # Lower temperature for more deterministic, consistent outputs
    
    # Remote LLM API (Ollama or compatible API)
    # If set, the LLM will use HTTP API calls instead of loading model locally
    remote_llm_url: str = os.getenv("REMOTE_LLM_URL", cfg_yaml.get("llm", {}).get("remote_url", ""))
    # Model name for remote API (e.g., "qwen2.5:7b" for Ollama)
    remote_model_name: str = os.getenv("REMOTE_MODEL_NAME", cfg_yaml.get("llm", {}).get("remote_model_name", "Qwen/Qwen2.5-7B-Instruct"))
    # API format: "ollama" (default), "openai" (for OpenAI-compatible APIs like vLLM), or "custom" (for other APIs)
    remote_api_format: str = os.getenv("REMOTE_API_FORMAT", cfg_yaml.get("llm", {}).get("remote_api_format", "openai"))

    def __post_init__(self):
        # No initialization needed for LLM Service
        # (RAG components like image_dir are handled by API_wrapper)
        pass

# ----------------------------
# LLM GENERATOR
# ----------------------------
class LLM:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.use_remote = bool(cfg.remote_llm_url and cfg.remote_llm_url.strip())
        
        if self.use_remote:
            # Remote LLM API mode (Ollama or compatible)
            print(f"\n🌐 Using Remote LLM API: {cfg.remote_llm_url}", flush=True)
            print(f"   Model: {cfg.remote_model_name}", flush=True)
            
            if not HTTPX_AVAILABLE:
                raise RuntimeError("httpx is required for remote LLM API. Install with: pip install httpx")
            
            # Validate URL format
            if not cfg.remote_llm_url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid remote_llm_url: {cfg.remote_llm_url}. Must start with http:// or https://")
            
            # Remove trailing slash
            self.remote_url = cfg.remote_llm_url.rstrip("/")
            self.remote_model = cfg.remote_model_name
            self.remote_api_format = cfg.remote_api_format.lower()  # Normalize to lowercase
            
            # Test connection
            connection_verified = False
            try:
                import httpx
                with httpx.Client(timeout=10.0) as client:
                    test_endpoint = ""
                    if self.remote_api_format == "ollama":
                        test_endpoint = "/api/tags"
                    elif self.remote_api_format == "openai":
                        test_endpoint = "/v1/models"  # OpenAI-compatible endpoint
                    elif self.remote_api_format == "custom":
                        test_endpoint = "/health"  # Common health check endpoint
                    
                    if test_endpoint:
                        test_url = f"{self.remote_url}{test_endpoint}"
                        response = client.get(test_url)
                        if response.status_code == 200:
                            print(f"   ✅ Connected to remote LLM server", flush=True)
                            connection_verified = True
                        else:
                            print(f"   ⚠️  Warning: Server responded with status {response.status_code}", flush=True)
                    else:
                        print(f"   ⚠️  Warning: Unknown remote_api_format '{self.remote_api_format}' - skipping connection test", flush=True)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not verify connection to remote server: {e}", flush=True)
                print(f"   💡 Will attempt to use it anyway during generation", flush=True)
            
            # Set default model_max_length for remote (Qwen2.5-7B supports 32K)
            self.model_max_length = 32768
            self.pipe = None  # Not used in remote mode
            
            if connection_verified:
                print("✅ Remote LLM configured successfully!\n", flush=True)
            else:
                print("⚠️  Remote LLM configured (connection test failed, but will attempt to use it)\n", flush=True)
            return
        
        # Local model loading mode (original code)
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch and transformers are required for local model mode. Install with: pip install torch transformers")
        
        print(f"\nLoading LLM: {cfg.model_id}", flush=True)
        
        try:
            # Detect GPU type and adjust settings accordingly
            gpu_name = ""
            gpu_memory_gb = 0
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🔍 Detected GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)", flush=True)
                print(f"   CUDA Device: {torch.cuda.current_device()}", flush=True)
            else:
                print(f"⚠️  WARNING: CUDA not available - model will run on CPU (slow!)", flush=True)
            
            # GPU tier detection for Qwen2.5-7B-Instruct optimization
            # T4 (16GB): May need quantization for memory efficiency
            # L4 (24GB): Sufficient for bfloat16 without quantization
            # A100+ (40GB+): Plenty of memory, use bfloat16
            is_t4 = "T4" in gpu_name or (gpu_memory_gb >= 14 and gpu_memory_gb < 20)
            is_l4 = "L4" in gpu_name or (gpu_memory_gb >= 20 and gpu_memory_gb < 30)
            is_large_gpu = gpu_memory_gb >= 30  # A100, H100, etc.
            
            # Check if running in Colab (VLM may also be loaded, so use quantization even on larger GPUs)
            is_colab = os.getenv("COLAB_GPU") is not None or "colab" in str(Path.cwd()).lower()
            force_quantization = is_colab  # Force quantization in Colab to save memory for VLM
            
            model_kwargs = {"low_cpu_mem_usage": True}
            quantization_enabled = False  # Track if quantization is actually enabled
            quantization_bits = 0  # Track quantization bits (0 = none, 4 = 4-bit, 8 = 8-bit)
            
            # Default: bfloat16 for GPU (better compatibility with Qwen's RoPE), float32 for CPU
            pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # L4 and large GPUs: Use bfloat16 without quantization (sufficient memory)
            # BUT: In Colab, force quantization to save memory for VLM
            if (is_l4 or is_large_gpu) and not force_quantization:
                if is_l4:
                    print(f"⚙️  L4 GPU detected (24GB) - using bfloat16 without quantization", flush=True)
                else:
                    print(f"⚙️  Large GPU detected ({gpu_memory_gb:.1f}GB) - using bfloat16 without quantization", flush=True)
                quantization_enabled = False
                quantization_bits = 0
                pipeline_dtype = torch.bfloat16
            
            # T4, small GPUs, OR Colab (even on larger GPUs if VLM is also loaded): Try quantization for memory efficiency
            elif is_t4 or (cuda_available and gpu_memory_gb < 16) or force_quantization:
                if force_quantization:
                    print(f"⚙️  Colab environment detected - forcing quantization to save memory for VLM", flush=True)
                elif is_t4:
                    print(f"⚙️  T4 GPU detected (16GB) - attempting quantization for memory efficiency", flush=True)
                else:
                    print(f"⚙️  Small GPU detected ({gpu_memory_gb:.1f}GB) - attempting quantization for memory efficiency", flush=True)
                if is_t4:
                    print(f"⚙️  T4 GPU detected (16GB) - attempting quantization for memory efficiency", flush=True)
                else:
                    print(f"⚙️  Small GPU detected ({gpu_memory_gb:.1f}GB) - attempting quantization for memory efficiency", flush=True)
                
                # T4: Try 8-bit quantization first, then fall back to 4-bit if needed
                # 8-bit reduces memory from ~7-8GB to ~4-5GB
                # 4-bit reduces memory to ~2-3GB (more compatible with Colab)
                quantization_enabled = False
                quantization_bits = 0
                
                # Check if bitsandbytes is available and properly installed
                bitsandbytes_available = False
                try:
                    import bitsandbytes
                    # Try to import the actual quantization modules
                    from transformers import BitsAndBytesConfig
                    # Additional check: try to create a dummy config to verify it works
                    try:
                        _ = BitsAndBytesConfig(load_in_8bit=True)
                        bitsandbytes_available = True
                        print(f"   ✅ bitsandbytes is available", flush=True)
                    except Exception as check_err:
                        error_msg = str(check_err).lower()
                        if "requires the latest version" in error_msg or "bitsandbytes" in error_msg:
                            print(f"   ⚠️  bitsandbytes is installed but outdated or incompatible", flush=True)
                            print(f"   💡 Update with: pip install -U bitsandbytes", flush=True)
                            bitsandbytes_available = False
                        else:
                            # Other error, might still work
                            bitsandbytes_available = True
                except ImportError:
                    print(f"   ⚠️  bitsandbytes not installed - skipping quantization", flush=True)
                    print(f"   💡 Install with: pip install bitsandbytes", flush=True)
                    bitsandbytes_available = False
                
                if bitsandbytes_available:
                    # Try 8-bit quantization first
                    try:
                        quantization_config_8bit = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False
                        )
                        print(f"   🔄 Attempting 8-bit quantization...", flush=True)
                        # Store 8-bit config for pipeline creation
                        model_kwargs["quantization_config"] = quantization_config_8bit
                        pipeline_dtype = None
                        quantization_enabled = True
                        quantization_bits = 8
                    except Exception as e8:
                        print(f"   ⚠️  8-bit quantization config failed: {e8}", flush=True)
                        # Try 4-bit as fallback
                        print(f"   💡 Trying 4-bit quantization as fallback...", flush=True)
                        try:
                            quantization_config_4bit = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            model_kwargs["quantization_config"] = quantization_config_4bit
                            print(f"   ✅ Using 4-bit quantization (reduces model memory to ~2-3GB)", flush=True)
                            pipeline_dtype = None
                            quantization_enabled = True
                            quantization_bits = 4
                        except Exception as e4:
                            print(f"   ❌ 4-bit quantization also failed: {e4}", flush=True)
                            print(f"   💡 Falling back to bfloat16 (may use more memory)", flush=True)
                            quantization_enabled = False
                            quantization_bits = 0
                            if "quantization_config" in model_kwargs:
                                del model_kwargs["quantization_config"]
                            pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                else:
                    # bitsandbytes not available - skip quantization
                    print(f"   💡 Skipping quantization - using bfloat16 instead", flush=True)
                    quantization_enabled = False
                    quantization_bits = 0
                    pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Attention implementation: Use SDPA for better memory efficiency (Qwen2.5 supports it well)
            # Only set attention if not using quantization (quantization has its own path)
            if not quantization_enabled:
                try:
                    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                        # Use SDPA for all GPUs - Qwen2.5-7B works well with SDPA
                        model_kwargs["attn_implementation"] = "sdpa"
                        print(f"   ✅ Using SDPA (Scaled Dot Product Attention) for memory efficiency", flush=True)
                    else:
                        print(f"   Using default attention implementation", flush=True)
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not set attention implementation: {e}", flush=True)
            
            # Build pipeline - dtype only if not using quantization
            # If 8-bit fails during pipeline creation, retry with 4-bit
            # Use device_map="auto" for GPU placement (best practice - accelerate optimizes automatically)
            # Note: When using device_map, we should NOT use device parameter
            # Using "auto" lets accelerate optimize memory usage:
            # - For T4 (16GB): accelerate will use quantization or CPU offloading if needed
            # - For L4 (24GB): accelerate will place model on GPU efficiently
            # - For A100+ (40GB+): accelerate will place everything on GPU
            if torch.cuda.is_available():
                device_map_setting = "auto"  # Let accelerate optimize (recommended for all GPU sizes)
                print(f"   🎯 Using device_map='auto' (accelerate will optimize for {gpu_memory_gb:.1f}GB GPU)", flush=True)
                if is_t4 or gpu_memory_gb < 16:
                    print(f"   💡 Small GPU detected - quantization/CPU offloading may be used if needed", flush=True)
            else:
                device_map_setting = "cpu"
                print(f"   ⚠️  CUDA not available - using CPU", flush=True)
            
            pipeline_created = False
            if pipeline_dtype is not None:
                # Not using quantization - include dtype (torch_dtype is deprecated)
                self.pipe = pipeline(
                    "text-generation",
                    model=cfg.model_id,
                    dtype=pipeline_dtype,  # Use dtype instead of torch_dtype
                    device_map=device_map_setting,  # Use explicit GPU mapping (no device param)
                    model_kwargs=model_kwargs
                )
                pipeline_created = True
            else:
                # Using quantization - try to create pipeline
                # If quantization fails (bitsandbytes missing/outdated, OOM, etc.), retry without quantization
                try:
                    self.pipe = pipeline(
                        "text-generation",
                        model=cfg.model_id,
                        device_map=device_map_setting,  # Use explicit GPU mapping (no device param)
                        model_kwargs=model_kwargs
                    )
                    pipeline_created = True
                except (ImportError, Exception) as pipeline_error:
                    error_str = str(pipeline_error).lower()
                    error_msg = str(pipeline_error)
                    
                    # Check if it's a bitsandbytes-related error
                    is_bitsandbytes_error = (
                        "bitsandbytes" in error_str or 
                        "requires the latest version" in error_msg or
                        isinstance(pipeline_error, ImportError)
                    )
                    
                    # Check if it's the CPU/disk offloading error for 8-bit
                    is_oom_error = quantization_bits == 8 and ("cpu" in error_str or "disk" in error_str or "dispatch" in error_str)
                    
                    if is_bitsandbytes_error:
                        print(f"   ⚠️  Quantization failed: {pipeline_error}", flush=True)
                        if "requires the latest version" in error_msg:
                            print(f"   💡 bitsandbytes is outdated or incompatible", flush=True)
                            print(f"   💡 Update with: pip install -U bitsandbytes", flush=True)
                        else:
                            print(f"   💡 bitsandbytes may not be properly installed", flush=True)
                            print(f"   💡 Install with: pip install bitsandbytes", flush=True)
                        
                        # If 8-bit failed, try 4-bit as fallback
                        if quantization_bits == 8:
                            print(f"   🔄 Trying 4-bit quantization as fallback...", flush=True)
                            if "quantization_config" in model_kwargs:
                                del model_kwargs["quantization_config"]
                            
                            try:
                                from transformers import BitsAndBytesConfig
                                quantization_config_4bit = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4"
                                )
                                model_kwargs["quantization_config"] = quantization_config_4bit
                                
                                self.pipe = pipeline(
                                    "text-generation",
                                    model=cfg.model_id,
                                    device_map=device_map_setting,
                                    model_kwargs=model_kwargs
                                )
                                quantization_bits = 4
                                quantization_enabled = True
                                pipeline_created = True
                                print(f"   ✅ 4-bit quantization works!", flush=True)
                            except Exception as e4:
                                # 4-bit also failed, fallback to bfloat16
                                print(f"   ❌ 4-bit quantization also failed: {e4}", flush=True)
                                print(f"   💡 Falling back to bfloat16 (may use more memory)", flush=True)
                                if "quantization_config" in model_kwargs:
                                    del model_kwargs["quantization_config"]
                                pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                                quantization_enabled = False
                                quantization_bits = 0
                                
                                self.pipe = pipeline(
                                    "text-generation",
                                    model=cfg.model_id,
                                    dtype=pipeline_dtype,
                                    device_map=device_map_setting,
                                    model_kwargs=model_kwargs
                                )
                                pipeline_created = True
                        else:
                            # 4-bit or other quantization failed, fallback to bfloat16
                            print(f"   ❌ Quantization failed: {pipeline_error}", flush=True)
                            print(f"   💡 Falling back to bfloat16 (may use more memory)", flush=True)
                            if "quantization_config" in model_kwargs:
                                del model_kwargs["quantization_config"]
                            pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                            quantization_enabled = False
                            quantization_bits = 0
                            
                            self.pipe = pipeline(
                                "text-generation",
                                model=cfg.model_id,
                                dtype=pipeline_dtype,
                                device_map=device_map_setting,
                                model_kwargs=model_kwargs
                            )
                            pipeline_created = True
                    elif is_oom_error:
                        # 8-bit OOM error - try 4-bit
                        print(f"   ⚠️  8-bit quantization failed during pipeline creation: {pipeline_error}", flush=True)
                        print(f"   💡 This usually means not enough GPU RAM for 8-bit. Trying 4-bit...", flush=True)
                        
                        if "quantization_config" in model_kwargs:
                            del model_kwargs["quantization_config"]
                        
                        try:
                            from transformers import BitsAndBytesConfig
                            quantization_config_4bit = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            model_kwargs["quantization_config"] = quantization_config_4bit
                            print(f"   🔄 Retrying with 4-bit quantization...", flush=True)
                            
                            self.pipe = pipeline(
                                "text-generation",
                                model=cfg.model_id,
                                device_map=device_map_setting,
                                model_kwargs=model_kwargs
                            )
                            quantization_bits = 4
                            quantization_enabled = True
                            pipeline_created = True
                            print(f"   ✅ 4-bit quantization works!", flush=True)
                        except Exception as e4:
                            print(f"   ❌ 4-bit quantization also failed: {e4}", flush=True)
                            print(f"   💡 Falling back to bfloat16 (may use more memory)", flush=True)
                            if "quantization_config" in model_kwargs:
                                del model_kwargs["quantization_config"]
                            pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                            quantization_enabled = False
                            quantization_bits = 0
                            
                            self.pipe = pipeline(
                                "text-generation",
                                model=cfg.model_id,
                                dtype=pipeline_dtype,
                                device_map=device_map_setting,
                                model_kwargs=model_kwargs
                            )
                            pipeline_created = True
                    else:
                        # Other error during quantization - fallback to bfloat16
                        print(f"   ⚠️  Quantization pipeline creation failed: {pipeline_error}", flush=True)
                        print(f"   💡 Falling back to bfloat16", flush=True)
                        if "quantization_config" in model_kwargs:
                            del model_kwargs["quantization_config"]
                        pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                        quantization_enabled = False
                        quantization_bits = 0
                        
                        self.pipe = pipeline(
                            "text-generation",
                            model=cfg.model_id,
                            dtype=pipeline_dtype,
                            device_map=device_map_setting,
                            model_kwargs=model_kwargs
                        )
                        pipeline_created = True
            # Verify quantization is actually applied
            if quantization_enabled and torch.cuda.is_available():
                try:
                    # Check if model is actually quantized
                    model = self.pipe.model
                    is_quantized = False
                    quant_type = "unknown"
                    
                    # Check for quantization indicators
                    if hasattr(model, 'hf_quantizer'):
                        is_quantized = True
                        quant_type = "8-bit (hf_quantizer)"
                    elif hasattr(model, 'quantization_config'):
                        config = model.quantization_config
                        if hasattr(config, 'load_in_4bit') and config.load_in_4bit:
                            is_quantized = True
                            quant_type = "4-bit"
                        elif hasattr(config, 'load_in_8bit') and config.load_in_8bit:
                            is_quantized = True
                            quant_type = "8-bit"
                    
                    if is_quantized:
                        print(f"   ✅ Quantization verified: Model is using {quant_type} quantization", flush=True)
                    else:
                        print(f"   ⚠️  WARNING: Quantization config provided but model may not be quantized!", flush=True)
                        print(f"   ⚠️  Model memory usage may be higher than expected", flush=True)
                    
                    # Check actual memory usage after loading
                    allocated_gb = torch.cuda.memory_allocated(0) / 1e9
                    reserved_gb = torch.cuda.memory_reserved(0) / 1e9
                    print(f"   📊 Model memory after load: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved", flush=True)
                    
                    # Check if quantization is working based on memory usage
                    if quantization_bits == 8:
                        if reserved_gb > 6.0:
                            print(f"   ⚠️  WARNING: Model using {reserved_gb:.2f} GB - 8-bit quantization may not be working!", flush=True)
                            print(f"   💡 Expected: ~4-5GB with 8-bit, ~6-7GB without", flush=True)
                            print(f"   💡 Try: Restart server after installing bitsandbytes", flush=True)
                        else:
                            print(f"   ✅ GOOD: Memory usage ({reserved_gb:.2f} GB) indicates 8-bit quantization is active", flush=True)
                    elif quantization_bits == 4:
                        if reserved_gb > 4.0:
                            print(f"   ⚠️  WARNING: Model using {reserved_gb:.2f} GB - 4-bit quantization may not be working!", flush=True)
                            print(f"   💡 Expected: ~2-3GB with 4-bit", flush=True)
                        else:
                            print(f"   ✅ GOOD: Memory usage ({reserved_gb:.2f} GB) indicates 4-bit quantization is active", flush=True)
                except Exception as e:
                    print(f"   ⚠️  Could not verify quantization: {e}", flush=True)
            
            # Store model's max context length for validation
            try:
                if hasattr(self.pipe.model, 'config'):
                    self.model_max_length = getattr(self.pipe.model.config, 'max_position_embeddings', None) or getattr(self.pipe.model.config, 'model_max_length', None)
                    if self.model_max_length:
                        print(f"   📏 Model max context length: {self.model_max_length} tokens", flush=True)
                    else:
                        # Qwen2.5-7B-Instruct typically supports 32k tokens
                        self.model_max_length = 32768
                        print(f"   📏 Using default max context length: {self.model_max_length} tokens", flush=True)
                else:
                    self.model_max_length = 32768  # Safe default for Qwen2.5
            except Exception as e:
                print(f"   ⚠️  Could not determine model max length: {e}", flush=True)
                self.model_max_length = 32768  # Safe default
            
            # Store config for potential CPU fallback
            self.cfg = cfg
            self.cpu_pipe = None  # Will be created on-demand if CUBLAS errors occur
            self.use_cpu = False  # Track if we're using CPU fallback
            
            print("✅ LLM loaded successfully!\n", flush=True)
        except Exception as e:
            print(f"❌ FAILED to load LLM: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            raise

    def generate(self, context: str, query: str) -> str:  # Added context and query parameters
        # ----------------------------
        # PROMPT TEMPLATE
        # ----------------------------
        A320_WING_REPAIR_PROMPT = (
            "⚠️ SAFETY NOTICE: This is an AIRCRAFT MAINTENANCE SYSTEM. Incorrect information "
            "can lead to catastrophic failures. Base your answer EXCLUSIVELY on the provided context. "
            "NEVER guess, infer, or provide information that is not explicitly stated in the context.\n\n"
            
            "You are given extracted text from aircraft technical documentation (manuals, repair procedures, "
            "technical dispositions, SRM, AMM, NTM, etc.).\n\n"
            
            "Context:\n<<<CONTEXT>>>\n\n"
            "Question:\n<<<QUERY>>>\n\n"
            
            "INSTRUCTIONS:\n"
            
            "0. RESPONSE FORMAT:\n"
            "   - Start directly with your answer - DO NOT include the query text or format as \"Query: ... Answer: ...\"\n"
            "   - Provide clear, direct answers without unnecessary repetition\n"
            "   - Cite sources using format: [Document: Filename, Page X]\n\n"
            
            "1. CORE PRINCIPLES:\n"
            "   - Use EXACT text from context/JSON - do NOT modify, paraphrase, or add identifiers\n"
            "   - Preserve ALL technical data, measurements, tolerances, specifications, dimensions, and values EXACTLY as written\n"
            "   - Do not round, approximate, or modify any numerical values\n"
            "   - CONTEXT AVAILABILITY RULE:\n"
            "     * If context contains ANY relevant information (even just a reference), provide it\n"
            "     * DO NOT say \"INSUFFICIENT INFORMATION\" unless context has ZERO relevant information\n"
            "     * CRITICAL FOR FLOWCHART QUERIES: If context contains \"FLOWCHART STRUCTURE (JSON - LLM-Optimized)\", "
            "       you MUST attempt to match and traverse - DO NOT say \"context does not contain\" without first checking ALL nodes\n"
            "     * For flowchart decision queries (\"what if\", \"what needs to be done\"): "
            "       You MUST search ALL decision nodes in the JSON before concluding \"context does not contain\"\n"
            "     * If context has ZERO relevant information, state: \"The requested information is not available in the provided context. "
            "       The context does not contain details about [specific topic requested].\"\n"
            "     * If context has partial information, provide what IS available and indicate if incomplete\n"
            "   - ANTI-HALLUCINATION RULES:\n"
            "     * DO NOT add \"Box\" numbers (e.g., \"Box 11\", \"Box 12\") - these do NOT exist unless explicitly in context/JSON\n"
            "     * DO NOT invent sheet numbers, figure references, or node numbers not in context/JSON\n"
            "     * DO NOT misread references (e.g., \"SHEET 1 AND 2\" is NOT \"Sheet 202\" - it means \"Sheet 1 AND Sheet 2\")\n"
            "     * DO NOT paraphrase references - use EXACT text (e.g., \"REFER TO FIGURE 002 SHEET 1 AND 2\" → use EXACTLY that phrase)\n"
            "     * DO NOT infer structure patterns from training data - use ONLY what's explicitly in context/JSON\n"
            "     * VALIDATION: Before using any identifier, check if it exists in context/JSON - if not, do NOT add it\n\n"
            
            "2. ANSWER FROM CONTEXT:\n"
            "   - If the context contains sufficient information, provide a complete answer based on that information\n"
            "   - For TEXT-BASED QUERIES (e.g., \"What are the cautions?\", \"What materials are required?\"):\n"
            "     * Extract ALL relevant information from text chunks in context\n"
            "     * If context contains text about the topic, provide it - DO NOT say \"Not available\"\n"
            "     * Scan ALL chunks for relevant keywords (e.g., \"CAUTION\", \"WARNING\", \"MATERIALS\")\n"
            "     * Provide comprehensive answers combining information from multiple chunks if needed\n"
            "   - For KEY-VALUE FIELD queries (e.g., \"what are the Drawing Numbers?\", \"Operator's Damage Report\", \"References\"):\n"
            "     * Search for explicit field labels in context such as \"Drawing Numbers:\", \"References:\", \"Operator's Damage Report:\"\n"
            "     * If a matching label is found, return the value that follows it EXACTLY as written\n"
            "     * Prefer direct extraction over generic summarization for these queries\n"
            "     * If at least one matching field label is present in any chunk, DO NOT answer \"Not available\"\n"
            "   - Apply LOGICAL REASONING to numerical comparisons:\n"
            "     * If context states a threshold (e.g., \"depth ≤ 0.4mm\"), then:\n"
            "       - Values EQUAL to or LESS than threshold → follow \"YES/TRUE\" branch\n"
            "       - Values GREATER than threshold → follow \"NO/FALSE\" branch\n"
            "     * Example: If flowchart says \"IS DEPTH ≤ 0.4mm?\" and query asks about \"depth > 0.4mm\" or \"depth = 0.6mm\":\n"
            "       - All these values are > 0.4mm → follow NO branch (CONTACT AIRBUS)\n"
            "     * DO NOT say \"INSUFFICIENT INFORMATION\" - compare query value against threshold and determine branch\n"
            "     * If query value > threshold → follow NO branch; If query value ≤ threshold → follow YES branch\n\n"
            
            "3. REPAIR PROCEDURES:\n"
            "   - When asked for complete repair procedures, include:\n"
            "     * All warnings, cautions, and notes from the context\n"
            "     * All required steps in sequence as stated\n"
            "     * All required materials, tools, and equipment mentioned\n"
            "     * All inspection requirements\n"
            "   - If the context mentions additional information exists elsewhere, note this\n"
            "   - Include safety warnings prominently at the beginning if present\n\n"
            
            "4. FLOWCHART AND DECISION TREE QUERIES:\n"
            "   - CRITICAL: Flowchart information is provided in TWO formats:\n"
            "     * FLOWCHART STRUCTURE (JSON - LLM-Optimized): This is the ONLY SOURCE OF TRUTH - you MUST use this for all traversal\n"
            "     * FLOWCHART TEXT (for reference only): DO NOT use this for node labels, references, or traversal - it is ONLY for understanding structure\n"
            "   - ABSOLUTE RULE: If context contains \"FLOWCHART STRUCTURE (JSON - LLM-Optimized)\", you MUST:\n"
            "     * IGNORE the FLOWCHART TEXT section completely for traversal\n"
            "     * Use ONLY the JSON structure for all node references, conditions, and traversal\n"
            "     * DO NOT invent or hallucinate nodes that are not in the JSON\n"
            "     * DO NOT use text patterns to infer structure - use ONLY what's in the JSON\n"
            "     * DO NOT generalize or infer aircraft damage workflows from training data - use ONLY the specific JSON structure provided\n"
            "     * If JSON shows specific nodes (e.g., Node 4: \"BETWEEN RIB 1 AND RIB 3 OR BETWEEN RIB 6 AND RIB 27\"), use EXACTLY that - do NOT invent other conditions\n"
            "   - FLOWCHART REASONING STEPS (for decision queries like \"what if\", \"what needs to be done\"):\n"
            "     1. Locate the JSON section: Find \"FLOWCHART STRUCTURE (JSON - LLM-Optimized)\" in context\n"
            "     2. Identify the starting node: Use \"start_node\" field (usually \"1\")\n"
            "     3. Find matching decision node: Scan nodes dict to find decision node whose \"text\" matches query condition:\n"
            "        - For \"between Rib1 and Rib2\": Look for node with text containing \"BETWEEN RIB 1\" or \"BETWEEN RIB 2\" or similar\n"
            "        - For \"between Rib7 and Rib20\": Look for node with text containing \"BETWEEN RIB 7\" or \"BETWEEN RIB 20\" or similar\n"
            "        - Match query text to node[\"text\"] field (case-insensitive, partial match OK)\n"
            "     4. Evaluate condition: If node has \"parameter\", \"operator\", \"value\" fields, use them for matching\n"
            "        - For range queries: Check if query value falls within node's \"value_min\" and \"value_max\" (if present)\n"
            "        - If no parameter fields, match based on node text content\n"
            "     5. Follow branch: Use node[\"branches\"][\"YES\"] or node[\"branches\"][\"NO\"] to get next node ID\n"
            "     6. Continue traversal: Follow node[\"next\"] for action nodes, node[\"branches\"] for decision nodes\n"
            "     7. Stop at terminal: Check if current node ID is in \"terminal_nodes\" dict\n"
            "     8. Return result: Use terminal_nodes[node_id] for final action, include ALL details from node[\"action\"] or node[\"text\"]\n"
            "   - EXAMPLE TRAVERSAL:\n"
            "     Query: \"What needs to be done if damage is between Rib1 and Rib2?\"\n"
            "     Steps:\n"
            "     1. Find JSON: Locate \"FLOWCHART STRUCTURE (JSON - LLM-Optimized)\" section in context\n"
            "     2. Start node: Use start_node = \"1\" from JSON\n"
            "     3. Traverse from start: Follow nodes[\"1\"][\"next\"] → \"2\", then nodes[\"2\"][\"next\"] → \"3\"\n"
            "     4. Find matching decision node: Search ALL nodes in nodes dict for text containing \"BETWEEN RIB\"\n"
            "        - Scan each node: Check if node[\"text\"] contains \"BETWEEN RIB 1\" or \"BETWEEN RIB 2\" or \"RIB 1 AND RIB 3\"\n"
            "        - Example match: nodes[\"4\"][\"text\"] = \"IS THE DAMAGE LOCATED BETWEEN RIB 1 AND RIB 3\"\n"
            "        - Query \"between Rib1 and Rib2\" matches because Rib1 and Rib2 are WITHIN the range \"RIB 1 AND RIB 3\"\n"
            "     5. Match condition: Query \"between Rib1 and Rib2\" matches node 4 (Rib1 and Rib2 fall within RIB 1-3 range) → Answer YES\n"
            "     6. Follow YES branch: Use nodes[\"4\"][\"branches\"][\"YES\"] → get next node ID (e.g., \"6\")\n"
            "     7. Continue traversal: Follow nodes[\"6\"][\"branches\"] or nodes[\"6\"][\"next\"] until reaching terminal node\n"
            "     8. Check terminal: If node ID is in terminal_nodes dict, use terminal_nodes[node_id] for final action\n"
            "     9. Return result: Provide the EXACT text from terminal_nodes[node_id] or node[\"action\"]\n"
            "   - TEXT MATCHING RULES:\n"
            "     * For \"between Rib1 and Rib2\": Match to node with text containing \"BETWEEN RIB 1\" or \"RIB 1 AND RIB 3\" or similar\n"
            "     * For \"between Rib7 and Rib20\": Match to node with text containing \"BETWEEN RIB 7\" or \"BETWEEN RIB 20\" or \"RIB 6 AND RIB 27\" (if Rib7-20 falls within that range)\n"
            "     * Use case-insensitive, partial matching: \"Rib1\" matches \"RIB 1\", \"between\" matches \"BETWEEN\"\n"
            "     * Range logic: If query says \"between X and Y\" and node says \"BETWEEN A AND B\", check if X >= A AND Y <= B (query range is within node range)\n"
            "     * OR CONDITIONS: If node text contains \"OR\" (e.g., \"BETWEEN RIB 1 AND RIB 3 OR BETWEEN RIB 6 AND RIB 27\"):\n"
            "       - Parse BOTH conditions separated by \"OR\"\n"
            "       - Check if query matches EITHER condition\n"
            "       - Example: Query \"between Rib7 and Rib20\" matches \"BETWEEN RIB 6 AND RIB 27\" (second part of OR) because 7 >= 6 AND 20 <= 27\n"
            "       - If query matches ANY part of the OR condition, the node matches → Follow YES branch\n"
            "   - JSON STRUCTURE (simplified):\n"
            "     * \"start_node\": Starting node ID (usually \"1\")\n"
            "     * \"nodes\": Dict where each key is node ID, value contains:\n"
            "       - \"type\": start/decision/action/terminal\n"
            "       - \"text\": Node label/text (EXACT text from flowchart)\n"
            "       - \"next\": For start/action nodes - single node ID or list of node IDs\n"
            "       - \"branches\": For decision nodes - dict with \"YES\"/\"NO\" keys mapping to next node IDs\n"
            "       - \"parameter\", \"operator\", \"value\": For decision nodes with conditions\n"
            "       - \"action\": For action/terminal nodes - the action text\n"
            "     * \"terminal_nodes\": Dict mapping terminal node IDs to their action text\n"
            "     * \"paths\": Optional precomputed paths (useful for summarization)\n"
            "   - TRAVERSAL:\n"
            "     * Start from \"start_node\" (e.g., \"1\")\n"
            "     * For decision nodes: Check \"branches\" dict - if query condition matches, follow \"YES\" or \"NO\" branch\n"
            "       Example: node[\"3\"][\"branches\"][\"YES\"] directly gives next node ID\n"
            "     * For action/start nodes: Follow \"next\" (single ID or list)\n"
            "       Example: node[\"1\"][\"next\"] → \"2\"\n"
            "     * Continue until reaching a node ID in \"terminal_nodes\"\n"
            "   - CONDITION MATCHING:\n"
            "     * CRITICAL: Match query text to node[\"text\"] field in JSON - search for keywords in node text\n"
            "     * PREFERRED METHOD: Use structured \"ranges\" field if available (extracted from node text)\n"
            "     * STEP-BY-STEP MATCHING PROCESS:\n"
            "       1. Extract query range: If query says \"between Rib7 and Rib20\", extract min=7, max=20\n"
            "       2. Search ALL nodes in nodes dict for decision nodes (type=\"decision\")\n"
            "       3. For each decision node, check in this order:\n"
            "          a. FIRST: Check if node has \"ranges\" field (array of {min, max} objects)\n"
            "             - For each range in node[\"ranges\"]: Check if query_min >= range[\"min\"] AND query_max <= range[\"max\"]\n"
            "             - If ANY range matches, the node matches → Follow YES branch\n"
            "          b. SECOND: Check if node has \"value_min\" and \"value_max\" fields\n"
            "             - Check if query_min >= value_min AND query_max <= value_max\n"
            "          c. THIRD: If no structured fields, parse node[\"text\"]:\n"
            "             - If node text contains \"OR\": Split by \"OR\" and check EACH part separately\n"
            "             - For each part, extract range (e.g., \"BETWEEN RIB 6 AND RIB 27\" → min=6, max=27)\n"
            "             - Check if query range falls within node range: query_min >= node_min AND query_max <= node_max\n"
            "             - If ANY part matches, the node matches\n"
            "       4. Example: Query \"between Rib7 and Rib20\" vs Node 4 with ranges=[{min:1, max:3}, {min:6, max:27}]:\n"
            "          - Range 1: 7 >= 1? YES, but 20 <= 3? NO → Range 1 doesn't match\n"
            "          - Range 2: 7 >= 6? YES, 20 <= 27? YES → Range 2 MATCHES\n"
            "          - Since Range 2 matches, Node 4 MATCHES → Follow YES branch\n"
            "     * Decision nodes may have \"parameter\", \"operator\", \"ranges\" fields for structured matching\n"
            "     * For range queries: ALWAYS check \"ranges\" field first (if present), then \"value_min\"/\"value_max\", then parse text\n"
            "     * TRAVERSAL MUST CONTINUE: After matching a decision node, you MUST follow the branch and continue to the next node\n"
            "       - DO NOT stop after the first match\n"
            "       - Continue following nodes[\"next\"] or nodes[\"branches\"][\"YES\"]/[\"NO\"] until reaching a terminal node\n"
            "       - Only terminal nodes provide the final answer\n"
            "   - When query asks \"what needs to be done if damage is between X and Y mm\":\n"
            "     1. Find decision node checking this range\n"
            "     2. Answer YES to that node\n"
            "     3. Continue to NEXT decision node (usually depth check)\n"
            "     4. If depth not specified, explain depth check required and provide both outcomes\n"
            "     5. If depth IS specified, follow appropriate branch based on depth value\n"
            "   - When reaching terminal action, check if context contains EXPANDED DETAILS:\n"
            "     * Terminal actions may have additional details in same/adjacent chunks\n"
            "     * Look for complete procedure text (e.g., \"FILL DENT, REFER TO CHAPTER 51-73-00, ...\")\n"
            "     * Include ALL details, not just action name\n"
            "   - IMPORTANT: For flowchart queries (decision queries, conditional queries), DO NOT paraphrase or summarize across multiple pages. "
            "     Use EXACT text from the specific flowchart node/action that matches the query condition. "
            "     Do NOT combine or mix contents from multiple flowchart pages unless the query explicitly asks for a summary.\n"
            "   - FALLBACK: If only text is available (NO JSON), reconstruct flowchart structure:\n"
            "     * Each chunk may contain MULTIPLE decision nodes\n"
            "     * Scan ALL chunks to identify ALL decision nodes by CONTENT and STRUCTURE (not just punctuation)\n"
            "     * Decision nodes are questions/conditions checking values (e.g., \"IS DAMAGE WITHIN X mm\", \"IS DEPTH > Y mm\")\n"
            "     * Identify by patterns: \"IS [condition]\", comparison operators (>, <, ≤, ≥, BETWEEN, WITHIN)\n"
            "     * Map connections between nodes to understand flowchart topology\n"
            "     * Identify terminal nodes (final actions like \"BLEND DAMAGE\", \"CONTACT AIRBUS\")\n"
            "   - Provide DIRECT, CONCISE answers with specific outcome/action and ALL associated details from context\n"
            "   - Handling ambiguous/incomplete queries:\n"
            "     * If query asks about damage depth without location: Explain depth limits vary by location\n"
            "     * If query asks about location without depth: Provide depth check requirements for that location\n"
            "     * If query is incomplete: Provide relevant decision question and outcomes\n"
            "   - Use ONLY explicitly stated decision conditions, branches, and outcomes from context\n"
            "   - DO NOT infer logical steps not explicitly stated\n"
            "   - DO NOT list generic step-by-step procedures - provide direct outcome for specific scenario\n\n"
            
            "5. SUMMARIZATION, EXPLANATION, AND DETAIL QUERIES:\n"
            "   - For FLOWCHART SUMMARIES: Use ONLY the JSON structure - list EXACT nodes from JSON, do NOT reconstruct or invent nodes\n"
            "     * Start from \"start_node\" field\n"
            "     * List ALL nodes in \"nodes\" dict with their EXACT \"text\" field\n"
            "     * Show connections using \"next\" and \"branches\" fields\n"
            "     * List terminal nodes from \"terminal_nodes\" dict\n"
            "     * DO NOT invent nodes, conditions, or connections not in the JSON\n"
            "     * DO NOT repeat the same node multiple times - each node ID appears ONCE in the JSON\n"
            "     * DO NOT generalize aircraft damage workflows - use ONLY the specific flowchart structure provided\n"
            "   - This section applies to: \"summarize\", \"describe\", \"explain\", \"detail\", \"what is\", \"how does\"\n"
            "   - FIRST: Check if query is related to aircraft technical documentation\n"
            "   - Queries about task numbers, figure numbers, sheet numbers, or flowchart procedures ARE aircraft documentation queries, NOT general knowledge\n"
            "   - If the query is about something completely unrelated to aircraft documentation (e.g., \"capital of India\", \"summarize the history of France\", \"explain quantum physics\", \"what is the weather\"):\n"
            "     → Provide a helpful, friendly response: \"I'm an aircraft technical documentation assistant specialized in aircraft maintenance, repair procedures, and technical documentation (SRM, AMM, NTM). I can help you with questions about aircraft systems, damage assessment, repair procedures, and related technical information. For general knowledge questions like '{query}', please consult a general knowledge source or search engine.\"\n"
            "     → Be polite and helpful, but clearly indicate the system's scope\n"
            "   - DO NOT treat queries about task numbers, figure numbers, or aircraft procedures as \"general knowledge\" - these ARE within the system's scope\n\n"
            "   - HANDLING FIGURE/SHEET/TABLE QUERIES:\n"
            "     * When query mentions specific figure/sheet (e.g., \"Figure 001 / 57-51-11-283-010 (SHEET 2/4)\"):\n"
            "       → Find chunks that contain EXACTLY this figure and sheet reference\n"
            "       → If query says \"SHEET 2/4\", look for chunks with \"sheet 2\" or \"sheet 02\" or \"sheet 2/4\" (normalize leading zeros)\n"
            "       → If query says \"Figure 001\", look for chunks with \"figure 001\" or \"figure 1\" (normalize leading zeros)\n"
            "       → Include ALL decision nodes, branches, and outcomes from that specific sheet\n"
            "       → If multiple sheets are mentioned in context, prioritize the one matching the query\n"
            "     * When query mentions a table (e.g., \"Table 1\", \"table-1\", \"Table n° 1\"):\n"
            "       → Search for table content in ALL chunks, regardless of filename specification\n"
            "       → Tables may be formatted as: \"Table 1\", \"Table n° 1\", \"Table n°1\", \"table-1\", \"table 1\"\n"
            "       → Include the complete table structure: headers, rows, columns, and all data\n"
            "       → If filename is specified, still search all chunks (table might be in different file or page)\n"
            "       → Provide the full table content, not just a reference to it\n\n"
            "   - HANDLING BASED ON QUERY TYPE:\n"
            "     * For \"summarize\" or \"describe\" queries:\n"
            "       → CRITICAL: Even if the context only contains REFERENCES to the requested figure/table/sheet, provide that information\n"
            "       → If context mentions the figure/table exists or references it, say: \"The context indicates that [figure/table] exists and [describe what the reference says]\". DO NOT say \"INSUFFICIENT INFORMATION\"\n"
            "       → Provide a concise overview of the key points from the context\n"
            "       → Include main decision nodes, conditions, outcomes, measurements, and references\n"
            "       → Focus on high-level structure and key information\n"
            "       → For flowcharts: Include ALL decision nodes and their branches from the specified sheet\n"
            "       → CRITICAL FOR FLOWCHART SUMMARIES: Use EXACT node text from JSON - do NOT add \"Box\" numbers or other identifiers\n"
            "       → CRITICAL FOR FLOWCHART SUMMARIES: Use EXACT references from JSON - if JSON says \"REFER TO FIGURE 002 SHEET 1 AND 2\", use EXACTLY that phrase\n"
            "       → CRITICAL FOR FLOWCHART SUMMARIES: If summarizing a flowchart, extract node text and references EXACTLY from the JSON structure\n"
            "       → OPTIMIZATION FOR FLOWCHART SUMMARIES: If JSON contains \"paths\" field, use it for summarization - it provides precomputed decision paths from start to terminal nodes\n"
            "       → When using paths: Describe each path using the \"description\" field (e.g., \"1 → 2 → 3 → 5\") and \"terminal_text\" (final action) - this is more accurate than manual traversal\n"
            "       → For tables: Include the complete table with all rows and columns\n"
            "       → If only a reference exists (e.g., \"See Figure X\" or \"Refer to Table Y\"): Include that reference and explain what it means\n"
            "     * For \"explain\" or \"how does\" queries:\n"
            "       → Provide a detailed explanation with context and reasoning\n"
            "       → Explain the logic, flow, and relationships between elements\n"
            "       → Include step-by-step processes if applicable\n"
            "     * For \"detail\" or \"what are the details\" queries:\n"
            "       → Provide comprehensive, detailed information from the context\n"
            "       → Include all measurements, thresholds, specifications, and procedures\n"
            "       → Provide complete step-by-step information when available\n\n"
            "   - CONTEXT AVAILABILITY HANDLING:\n"
            "     * If the query IS related to aircraft documentation AND the context contains relevant information (even if partial or just references):\n"
            "       → Provide the requested information (summary/explanation/detail) based on what IS in the context\n"
            "       → Include ALL available information from the context:\n"
            "         - Decision nodes, conditions, and branches found in the context\n"
            "         - References to other sheets/figures (e.g., \"REFER TO FIGURE 1, SHEET 3\")\n"
            "         - Terminal actions and outcomes\n"
            "         - Measurements, thresholds, and technical specifications\n"
            "       → If the context contains a reference node (e.g., \"REFER TO FIGURE 1, SHEET 3\"):\n"
            "         - Include the reference in your response and explain what it means\n"
            "         - Example: \"For damage on mid beam, the flowchart directs to Figure 1, Sheet 3 for detailed assessment procedures\"\n"
            "         - DO NOT say \"INSUFFICIENT INFORMATION\" - a reference IS information that should be included\n"
            "       → If the context shows partial information (e.g., \"truncated\" or incomplete chunks):\n"
            "         - Provide information based on what IS available\n"
            "         - Clearly indicate if information appears incomplete (e.g., \"Based on the available context...\" or \"The provided information shows...\")\n"
            "       → DO NOT say \"INSUFFICIENT INFORMATION\" or \"information is insufficient\" when relevant context exists - provide the requested information based on what is in the context\n"
            "       → CRITICAL: If you provide ANY information in your response, DO NOT end with \"Not available in the current document\" or similar phrases - that contradicts the information you just provided\n"
            "       → CRITICAL FOR \"SUMMARIZE\" QUERIES: If the context mentions the figure/table/sheet (even just a reference), provide that information. A reference IS information.\n"
            "       → Example: If query is \"Summarize Figure X\" and context says \"See Figure X for details\", respond: \"The context references Figure X and indicates it contains [describe what the reference says]. For complete details, refer to Figure X.\"\n"
            "       → CRITICAL ANTI-HALLUCINATION RULES:\n"
            "         * Use EXACTLY the text from the context - do NOT paraphrase sheet/figure references in ways that change their meaning\n"
            "         * If context says \"REFER TO FIGURE 002 SHEET 1 AND 2\", use EXACTLY that phrase - do NOT change to \"Figure 002, Sheet 202\" or \"Figure 002, Sheet 1, Sheet 2\"\n"
            "         * If context says \"FROM SHEET 1\", use EXACTLY that - do NOT add \"Box 11\" or other identifiers not in the context\n"
            "         * If context says \"See Sheet 2\", use EXACTLY that - do NOT add \"Box 12\" or other identifiers not in the context\n"
            "         * DO NOT infer or invent node numbers, box numbers, or other identifiers that are not explicitly in the context\n"
            "       → ALWAYS include the filename when citing information from chunks - the context includes [Document: filename, Page X] for each chunk\n"
            "       → If multiple chunks contain related information, combine them into a coherent response\n"
            "       → For figures/flowcharts: Include ALL decision nodes, branches, outcomes, and references that are present in the context\n"
            "     * If the query IS related to aircraft documentation but the context contains NO relevant information at all (completely unrelated topic):\n"
            "       → State: \"The requested information is not available in the provided context. The context does not contain details about [specific topic requested].\"\n"
            "       → This is different from partial information - use this only when context has ZERO relevant information\n\n"
            "   - EXAMPLES:\n"
            "     * \"Summarize Flowchart - Damage on Mid Beam\" (context has: \"IS DAMAGE ON MID BEAM? YES → REFER TO FIGURE 1, SHEET 3\"):\n"
            "       → Response: \"The flowchart for damage on mid beam shows a decision node asking 'IS DAMAGE ON MID BEAM?'. If the answer is YES, the flowchart directs to Figure 1, Sheet 3 for detailed assessment procedures. The context does not contain the detailed flowchart from Sheet 3, only this routing reference.\"\n"
            "       → CRITICAL: Use EXACTLY what is in the context - do NOT add \"Box\" numbers, do NOT change \"SHEET 3\" to \"Sheet 203\" or other variations\n"
            "     * \"Explain how damage assessment works\" (context has flowchart nodes):\n"
            "       → Response: Provide detailed explanation of the flowchart logic, decision nodes, and how the assessment process flows\n"
            "     * \"Detail the repair procedure\" (context has procedure steps):\n"
            "       → Response: Provide comprehensive step-by-step details of all procedure steps found in the context\n"
            "     * \"Summarize Figure 001\" (context has complete flowchart):\n"
            "       → Response: Provide a summary of all flowchart nodes, conditions, branches, and outcomes found in the context chunks\n"
            "       → CRITICAL: Extract node labels and references EXACTLY from the JSON - do NOT add \"Box\" numbers or modify references\n"
            "       → Example: If JSON node label is \"FROM SHEET 1\", say \"FROM SHEET 1\" - do NOT say \"From Sheet 1 (Box 11)\"\n"
            "       → Example: If JSON reference is \"REFER TO FIGURE 002 SHEET 1 AND 2\", say exactly that - do NOT say \"Refer to Figure 002, Sheet 202\"\n"
            "     * \"Summarize Figure 57-41-19-991-022-A (SHEET 01/4)\" (context only has reference to this figure):\n"
            "       → Response: \"The context references Figure 57-41-19-991-022-A (SHEET 01/4) and indicates it is related to [describe what the reference says]. The context shows that this figure exists and is referenced in the document for [topic]. For complete details, refer to the actual figure.\"\n"
            "       → DO NOT say \"INSUFFICIENT INFORMATION\" - a reference IS information\n"
            "     * \"Summarize table-1 from file [filename]\" (context has table content):\n"
            "       → Response: Provide the complete table structure with all headers, rows, columns, and data from the context\n"
            "     * \"Capital of India\" or \"Summarize the history of France\" (unrelated to aircraft):\n"
            "       → Response: \"I'm an aircraft technical documentation assistant specialized in aircraft maintenance, repair procedures, and technical documentation. I can help you with questions about aircraft systems, damage assessment, repair procedures, and related technical information. For general knowledge questions, please consult a general knowledge source or search engine.\"\n"
            "   - If the context contains multiple pages/sheets, include information from all available pages\n\n"
            
            "5. MULTI-SOURCE INFORMATION:\n"
            "   - If information spans multiple sections/documents, cite ALL relevant sources\n"
            "   - Clearly indicate which information comes from which source\n"
            "   - If flowchart information conflicts with textual procedures, the RAW TEXTUAL PROCEDURE takes precedence unless context states otherwise\n\n"
            
            "6. AIRCRAFT-SPECIFIC QUESTIONS:\n"
            "   - For questions about specific aircraft (MSN/registration), use information from the context that applies to that aircraft\n"
            "   - If context doesn't specify aircraft-specific information, use general information provided\n\n"
        )

        # Remote LLM API mode (Ollama or compatible)
        if self.use_remote:
            try:
                import httpx
                
                # Build prompt using .replace() with unique tokens (safer than .format() - avoids brace escaping issues)
                # Using <<<TOKEN>>> format to ensure zero collision probability with context/query content
                full_prompt = A320_WING_REPAIR_PROMPT.replace("<<<CONTEXT>>>", context).replace("<<<QUERY>>>", query)
                
                # Determine API format and construct request
                if self.remote_api_format == "ollama":
                    # Ollama API format (default)
                    # Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
                    api_url = f"{self.remote_url}/api/generate"
                    payload = {
                        "model": self.remote_model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.cfg.temperature,
                            "num_predict": self.cfg.max_new_tokens,
                        }
                    }
                elif self.remote_api_format == "openai":
                    # OpenAI-compatible API format (vLLM, TGI, etc.)
                    # Documentation: https://platform.openai.com/docs/api-reference/chat/create
                    api_url = f"{self.remote_url}/v1/chat/completions"
                    payload = {
                        "model": self.remote_model,
                        "messages": [
                            {"role": "user", "content": full_prompt}
                        ],
                        "temperature": self.cfg.temperature,
                        "max_tokens": self.cfg.max_new_tokens,
                        "stream": False
                    }
                elif self.remote_api_format == "custom":
                    # Custom API format (other custom FastAPI servers)
                    # Expected format: POST /generate with {"prompt": "...", "temperature": ..., "max_tokens": ...}
                    api_url = f"{self.remote_url}/generate"
                    payload = {
                        "prompt": full_prompt,
                        "temperature": self.cfg.temperature,
                        "max_tokens": self.cfg.max_new_tokens,  # Note: custom APIs may use "max_tokens" instead of "num_predict"
                    }
                else:
                    raise ValueError(f"Unknown remote_api_format: {self.remote_api_format}. Supported: 'ollama', 'openai', 'custom'")
                
                print(f"🌐 Calling remote LLM API: {api_url}", flush=True)
                print(f"   Model: {self.remote_model}, Context: {len(context)} chars, Query: {len(query)} chars", flush=True)
                
                # Make HTTP request
                with httpx.Client(timeout=120.0) as client:  # 2 minute timeout for large contexts
                    response = client.post(api_url, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Extract response text based on API format
                    if self.remote_api_format == "ollama":
                        # Ollama format: {"response": "..."}
                        if "response" in result:
                            answer = result["response"].strip()
                        elif "text" in result:
                            answer = result["text"].strip()
                        else:
                            # Fallback: try to get any text field
                            answer = str(result.get("content", result.get("output", ""))).strip()
                    elif self.remote_api_format == "openai":
                        # OpenAI-compatible format: {"choices": [{"message": {"content": "..."}}]}
                        if "choices" in result and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                answer = choice["message"]["content"].strip()
                            elif "text" in choice:
                                answer = choice["text"].strip()
                            else:
                                answer = str(choice.get("content", choice.get("text", ""))).strip()
                        elif "content" in result:
                            answer = result["content"].strip()
                        else:
                            # Fallback: try common field names
                            answer = str(result.get("response", result.get("text", result.get("output", "")))).strip()
                    elif self.remote_api_format == "custom":
                        # Custom format (other APIs): {"response": "..."} or {"text": "..."}
                        if "response" in result:
                            answer = result["response"].strip()
                        elif "text" in result:
                            answer = result["text"].strip()
                        elif "content" in result:
                            answer = result["content"].strip()
                        else:
                            # Fallback: try common field names
                            answer = str(result.get("output", result.get("generated_text", ""))).strip()
                    
                    if not answer:
                        raise ValueError(f"Empty response from remote LLM API: {result}")
                    
                    print(f"✅ Remote LLM generation successful, answer length: {len(answer)} characters", flush=True)
                    return answer
                    
            except httpx.TimeoutException:
                error_msg = "Remote LLM API request timed out (120s). The server may be overloaded or the context is too large."
                print(f"❌ {error_msg}", flush=True)
                raise RuntimeError(error_msg)
            except httpx.HTTPStatusError as e:
                error_msg = f"Remote LLM API returned error {e.response.status_code}: {e.response.text}"
                print(f"❌ {error_msg}", flush=True)
                raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"Remote LLM API error: {e}"
                print(f"❌ {error_msg}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                raise RuntimeError(error_msg) from e
        
        # Local model mode (original code)
        try:
            # CRITICAL: Truncate context BEFORE creating messages to prevent OOM errors
            # This ensures we only create messages once with the correct context size
            # NOTE: Qwen2.5-7B-Instruct supports 32K+ tokens context window
            # BUT: GPU memory limits how much we can actually process
            # With quantization (~3-4GB model), we can use more context
            # Without quantization (~11GB model), we need less context
            
            # Dynamic context limit based on available memory
            # Use actual tokenizer for accurate token counting (important for OCR'd text)
            # OCR text can have different tokenization characteristics than normal text
            try:
                if hasattr(self, 'pipe') and hasattr(self.pipe, 'tokenizer'):
                    # Accurate token count using actual tokenizer (better for OCR text)
                    context_token_estimate = len(self.pipe.tokenizer.encode(context, add_special_tokens=False))
                else:
                    # Fallback: conservative estimate for OCR text (3.5 chars/token instead of 4)
                    context_token_estimate = int(len(context) / 3.5)
            except Exception as e:
                # Fallback: conservative estimate for OCR text
                context_token_estimate = int(len(context) / 3.5)
            
            # Get model's actual max context length
            model_max_tokens = getattr(self, 'model_max_length', 32768)  # Default to 32k for Qwen2.5
            # Reserve tokens for: system prompt (~200), user query (~100), generation (~400), safety margin (~500)
            # Total reserved: ~1200 tokens
            model_safe_max = model_max_tokens - 1200
            
            # Calculate safe context limit based on free memory
            if torch.cuda.is_available():
                try:
                    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
                    allocated_memory_gb = torch.cuda.memory_reserved(0) / 1e9
                    
                    # Check if quantization is active for 7B model
                    # 7B model: ~11GB non-quantized, ~3-4GB quantized
                    # If allocated memory < 6GB, model is likely quantized
                    is_quantized = allocated_memory_gb < 6.0
                    
                    # Adjust context limit based on available memory AND quantization status
                    if not is_quantized:
                        # 7B non-quantized - conservative limits due to high memory usage
                        if free_memory_gb >= 6.0:
                            MAX_SAFE_CONTEXT_TOKENS = 800  # Very conservative for non-quantized 7B
                        elif free_memory_gb >= 4.0:
                            MAX_SAFE_CONTEXT_TOKENS = 600
                        else:
                            MAX_SAFE_CONTEXT_TOKENS = 400  # Extremely conservative
                        print(f"⚠️  7B model NOT quantized ({allocated_memory_gb:.2f} GB allocated) - using conservative context limits", flush=True)
                    else:
                        # 7B quantized - can use more context
                        if free_memory_gb >= 10.0:
                            MAX_SAFE_CONTEXT_TOKENS = 4000  # With quantization: can handle more
                        elif free_memory_gb >= 8.0:
                            MAX_SAFE_CONTEXT_TOKENS = 3000
                        elif free_memory_gb >= 6.0:
                            MAX_SAFE_CONTEXT_TOKENS = 2000
                        elif free_memory_gb >= 4.0:
                            MAX_SAFE_CONTEXT_TOKENS = 1200
                        else:
                            MAX_SAFE_CONTEXT_TOKENS = 800
                        print(f"✅ 7B model quantized ({allocated_memory_gb:.2f} GB allocated) - using optimized context limits", flush=True)
                    
                    # Ensure we don't exceed model's max_position_embeddings
                    MAX_SAFE_CONTEXT_TOKENS = min(MAX_SAFE_CONTEXT_TOKENS, model_safe_max)
                    print(f"📊 Context limit adjusted based on free memory ({free_memory_gb:.2f} GB): {MAX_SAFE_CONTEXT_TOKENS} tokens (model max: {model_max_tokens})", flush=True)
                except:
                    MAX_SAFE_CONTEXT_TOKENS = min(1200, model_safe_max)  # Fallback to conservative limit, but respect model max
            else:
                MAX_SAFE_CONTEXT_TOKENS = min(1200, model_safe_max)  # Conservative default, but respect model max
            
            # Truncate context if too large
            # CRITICAL: Prioritize flowchart JSON - it must NEVER be truncated
            if context_token_estimate > MAX_SAFE_CONTEXT_TOKENS:
                # Check if context contains flowchart JSON
                flowchart_json_start = context.find("FLOWCHART STRUCTURE (JSON - LLM-Optimized):")
                flowchart_json_end = -1
                if flowchart_json_start != -1:
                    # Find the end of flowchart JSON section
                    flowchart_text_start = context.find("FLOWCHART TEXT (for reference only", flowchart_json_start)
                    if flowchart_text_start != -1:
                        flowchart_json_end = flowchart_text_start
                    else:
                        # Fallback: find next major section marker
                        next_section = context.find("\n" + "=" * 60, flowchart_json_start + 100)
                        if next_section != -1:
                            flowchart_json_end = next_section
                
                # Use conservative ratio for OCR text (3.5 chars/token) when calculating char limits
                max_context_chars = int(MAX_SAFE_CONTEXT_TOKENS * 3.5)
                original_length = len(context)
                
                if flowchart_json_start != -1 and flowchart_json_end != -1:
                    # Flowchart JSON found - prioritize it
                    flowchart_json_size = flowchart_json_end - flowchart_json_start
                    remaining_chars = max_context_chars - flowchart_json_size - 500  # Reserve 500 for system prompt/query
                    
                    # Extract flowchart JSON (always preserve it)
                    flowchart_json_section = context[flowchart_json_start:flowchart_json_end]
                    
                    if remaining_chars > 0:
                        # Get text before and after flowchart JSON
                        before_flowchart = context[:flowchart_json_start]
                        after_flowchart = context[flowchart_json_end:]
                        
                        # Truncate before/after sections if needed, but keep flowchart JSON intact
                        if len(before_flowchart) + len(after_flowchart) > remaining_chars:
                            # Keep more of "before" (system prompt, early chunks) and less of "after"
                            before_limit = int(remaining_chars * 0.6)
                            after_limit = remaining_chars - before_limit
                            
                            if len(before_flowchart) > before_limit:
                                before_flowchart = before_flowchart[:before_limit] + "\n\n[... earlier context truncated ...]\n\n"
                            if len(after_flowchart) > after_limit:
                                after_flowchart = "\n\n[... later context truncated ...]\n\n" + after_flowchart[-after_limit:]
                        
                        # Reconstruct context with flowchart JSON preserved
                        context = before_flowchart + flowchart_json_section + after_flowchart
                        print(f"⚠️  Context truncated ({context_token_estimate} → {MAX_SAFE_CONTEXT_TOKENS} tokens) - FLOWCHART JSON PRESERVED ({flowchart_json_size} chars)", flush=True)
                    else:
                        # Flowchart JSON is too large - this shouldn't happen, but handle gracefully
                        print(f"⚠️  WARNING: Flowchart JSON ({flowchart_json_size} chars) exceeds context limit ({max_context_chars} chars) - keeping JSON only", flush=True)
                        # Keep only flowchart JSON + minimal context
                        context = context[:flowchart_json_start] + flowchart_json_section + "\n\n[Other context truncated to preserve flowchart JSON]"
                else:
                    # No flowchart JSON - use standard smart truncation
                    if original_length > max_context_chars:
                        # Keep first 70% and last 30% of allowed size
                        first_part = int(max_context_chars * 0.7)
                        last_part = max_context_chars - first_part
                        context = context[:first_part] + "\n\n[... context truncated due to memory constraints ...]\n\n" + context[-last_part:]
                        print(f"⚠️  Context truncated ({context_token_estimate} → {MAX_SAFE_CONTEXT_TOKENS} tokens) using smart truncation", flush=True)
                    else:
                        context = context[:max_context_chars] + "\n\n[Context truncated due to memory constraints]"
                        print(f"⚠️  Context truncated ({context_token_estimate} → {MAX_SAFE_CONTEXT_TOKENS} tokens)", flush=True)
            
            # Final validation: ensure context doesn't exceed model limits
            # Use actual tokenizer for accurate final count
            try:
                if hasattr(self, 'pipe') and hasattr(self.pipe, 'tokenizer'):
                    final_context_tokens = len(self.pipe.tokenizer.encode(context, add_special_tokens=False))
                else:
                    final_context_tokens = int(len(context) / 3.5)
            except Exception:
                final_context_tokens = int(len(context) / 3.5)
            
            model_max = getattr(self, 'model_max_length', 32768)
            if final_context_tokens > model_max - 500:  # Reserve 500 tokens for generation
                print(f"⚠️  Context too long ({final_context_tokens} tokens) for model max ({model_max}), truncating...", flush=True)
                # Use conservative ratio for OCR text
                max_allowed_chars = int((model_max - 500) * 3.5)
                context = context[:max_allowed_chars] + "\n\n[Context truncated to fit model limits]"
            
            # Create messages AFTER truncation (only once, with correct context size)
            messages = [
                {"role": "system", "content": (
                    "You are a SAFETY-CRITICAL aircraft maintenance documentation expert. "
                    "Your responses will be used for aircraft maintenance and repair operations where errors can result in catastrophic failures. "
                    "Follow the detailed instructions in the user message below."
                )},
                {"role": "user", "content": A320_WING_REPAIR_PROMPT.replace("<<<CONTEXT>>>", context).replace("<<<QUERY>>>", query)}
            ]
            
            # Check GPU memory and adjust max_new_tokens for T4
            max_tokens = self.cfg.max_new_tokens
            if torch.cuda.is_available():
                try:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_name = torch.cuda.get_device_name(0)
                    is_t4 = "T4" in gpu_name or gpu_memory_gb < 20
                    
                    if is_t4:
                        # T4: Check actual free memory AND context size, adjust dynamically
                        try:
                            free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
                            allocated_memory_gb = torch.cuda.memory_reserved(0) / 1e9

                            # Faster but approximate (current approach):
                            # Estimate context tokens (rough: 4 chars per token)
                            context_tokens = len(context) // 4
                            
                            print(f"📊 T4 GPU Memory: {allocated_memory_gb:.2f} GB allocated, {free_memory_gb:.2f} GB free", flush=True)
                            print(f"📊 Context: {len(context)} chars (~{context_tokens} tokens)", flush=True)
                            
                            # Adjust max_new_tokens based on BOTH free memory AND context size
                            # Larger context = less room for generation tokens
                            # Rule of thumb: ~1GB needed per 500 tokens of generation with context
                            
                            if free_memory_gb < 2.0:
                                # Very low memory - use minimal tokens regardless of context
                                max_tokens = min(200, self.cfg.max_new_tokens)
                                print(f"⚠️  Very low GPU memory ({free_memory_gb:.2f} GB free) - reducing to {max_tokens} tokens", flush=True)
                            elif free_memory_gb < 3.0:
                                # Low memory - moderate tokens
                                max_tokens = min(250, self.cfg.max_new_tokens)
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (low memory: {free_memory_gb:.2f} GB free)", flush=True)
                            elif free_memory_gb < 4.0:
                                # Moderate memory - adjust based on context size
                                if context_tokens > 1000:
                                    # Large context - use fewer generation tokens
                                    max_tokens = min(200, self.cfg.max_new_tokens)
                                elif context_tokens > 1500:
                                    # Very large context - minimal tokens
                                    max_tokens = min(150, self.cfg.max_new_tokens)
                                else:
                                    max_tokens = min(250, self.cfg.max_new_tokens)
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (moderate memory: {free_memory_gb:.2f} GB free, context: ~{context_tokens} tokens)", flush=True)
                            elif free_memory_gb < 5.0:
                                # Good memory - but still consider context
                                if context_tokens > 1500:
                                    # Large context - reduce generation tokens
                                    max_tokens = min(200, self.cfg.max_new_tokens)
                                elif context_tokens > 1000:
                                    max_tokens = min(250, self.cfg.max_new_tokens)
                                else:
                                    max_tokens = min(300, self.cfg.max_new_tokens)
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (good memory: {free_memory_gb:.2f} GB free, context: ~{context_tokens} tokens)", flush=True)
                            else:
                                # Excellent memory - can use more for detailed responses
                                if context_tokens > 2000:
                                    max_tokens = min(450, self.cfg.max_new_tokens)  # Increased from 350
                                elif context_tokens > 1500:
                                    max_tokens = min(500, self.cfg.max_new_tokens)  # Increased from 350
                                else:
                                    max_tokens = min(500, self.cfg.max_new_tokens)  # Increased from 400 for better accuracy
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (excellent memory: {free_memory_gb:.2f} GB free, context: ~{context_tokens} tokens)", flush=True)
                        except Exception as e:
                            # Fallback if memory check fails
                            max_tokens = min(250, self.cfg.max_new_tokens)  # Conservative default
                            print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (could not check memory: {e})", flush=True)
                except:
                    pass
            
            # Clear cache aggressively before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            # Check memory before generation
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0) / 1e9
                reserved_before = torch.cuda.memory_reserved(0) / 1e9
                free_before = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
                print(f"📊 Memory before generation: {allocated_before:.2f} GB allocated, {reserved_before:.2f} GB reserved, {free_before:.2f} GB free", flush=True)
                if free_before < 3.0:
                    print(f"⚠️  WARNING: Very low free memory ({free_before:.2f} GB) - generation may fail!", flush=True)
                    # Reduce generation tokens even more if memory is critically low
                    if free_before < 2.0:
                        max_tokens = min(100, max_tokens)
                        print(f"⚠️  CRITICAL: Reducing max_new_tokens to {max_tokens} due to low memory", flush=True)
            
            print(f"🔄 Generating with context length: {len(context)} chars ({len(context) // 4} est. tokens), max_new_tokens: {max_tokens}", flush=True)
            
            # Clear CUDA cache and synchronize before generation to avoid CUBLAS errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations complete
                # Reset peak memory stats to avoid state issues
                torch.cuda.reset_peak_memory_stats(0)
                # Ensure we're on the correct device
                with torch.cuda.device(0):
                    pass  # Just ensure device context is set
            
            # Retry mechanism for CUBLAS errors
            max_retries = 2
            retry_count = 0
            outputs = None
            
            while retry_count <= max_retries:
                try:
                    outputs = self.pipe(
                        messages,
                        max_new_tokens=max_tokens,
                        temperature=self.cfg.temperature,
                        do_sample=False,  # Use greedy decoding for deterministic outputs in safety-critical system
                        pad_token_id=self.pipe.tokenizer.eos_token_id
                    )
                    break  # Success, exit retry loop
                except RuntimeError as e:
                    error_str = str(e)
                    if ("CUBLAS" in error_str or "cublas" in error_str.lower()):
                        if retry_count == 0:
                            # First CUBLAS error: Try CPU fallback
                            retry_count += 1
                            print(f"⚠️  CUBLAS error detected - switching to CPU inference (slower but stable)", flush=True)
                            
                            # Create CPU pipeline if not already created
                            if self.cpu_pipe is None:
                                print(f"   🔄 Loading model on CPU for fallback...", flush=True)
                                try:
                                    # Load model on CPU
                                    self.cpu_pipe = pipeline(
                                        "text-generation",
                                        model=self.cfg.model_id,
                                        device_map="cpu",
                                        dtype=torch.float32,  # Use dtype instead of torch_dtype
                                        model_kwargs={"low_cpu_mem_usage": True}
                                    )
                                    print(f"   ✅ CPU pipeline loaded successfully", flush=True)
                                except Exception as cpu_error:
                                    print(f"   ❌ Failed to load CPU pipeline: {cpu_error}", flush=True)
                                    raise RuntimeError(f"CUBLAS error and CPU fallback failed: {cpu_error}")
                            
                            # Switch to CPU pipeline
                            original_pipe = self.pipe
                            self.pipe = self.cpu_pipe
                            self.use_cpu = True
                            print(f"   ✅ Switched to CPU inference - will use CPU for all future requests", flush=True)
                            
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                            
                            continue  # Retry with CPU
                        elif retry_count < max_retries:
                            # Additional retries with CPU
                            retry_count += 1
                            print(f"⚠️  CUBLAS error on CPU (attempt {retry_count}/{max_retries}), retrying...", flush=True)
                            import time
                            time.sleep(0.5)
                            continue
                        else:
                            # Max retries reached
                            raise
                    else:
                        # Not a CUBLAS error, re-raise
                        raise
            
            if outputs is None:
                raise RuntimeError("Failed to generate after retries")
            
            if retry_count > 0:
                print(f"✅ Generation completed after {retry_count} retry(ies)", flush=True)
            else:
                print(f"✅ Generation completed", flush=True)
            
            # Clear GPU cache after successful generation to prevent memory accumulation
            # Colab-specific: Also force garbage collection to free Python objects
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Force Python garbage collection (helps in Colab to free Python objects)
                import gc
                gc.collect()
                # Double-check: clear cache again after GC
                torch.cuda.empty_cache()
            
            # Extract assistant's reply
            # Error Handling : Safer version
            try:
                result = outputs[0]["generated_text"][-1]["content"].strip()
                return result
            except (KeyError, IndexError, TypeError) as e:
                # Fallback: try alternative structure
                if isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                    # Handle different structure
                    return str(outputs[0].get("generated_text", "")).strip()
                raise ValueError(f"Unexpected pipeline output structure: {outputs}") from e
            except torch.cuda.OutOfMemoryError as e:
                # Clear cache and re-raise with context
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Also force GC on error
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                print(f"❌ CUDA OOM during generation. Context length: {len(context)} chars, Query: {query[:100]}...", flush=True)
                raise
            except RuntimeError as e:
                error_str = str(e)
                # Handle CUBLAS errors specifically (only if not already handled by retry mechanism)
                if "CUBLAS" in error_str or "cublas" in error_str.lower():
                    print(f"❌ CUBLAS error during generation after retries (likely CUDA state or tensor shape issue)", flush=True)
                    print(f"   Context length: {len(context)} chars (~{len(context) // 4} tokens)", flush=True)
                    print(f"   Model max length: {getattr(self, 'model_max_length', 'unknown')} tokens", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    # Re-raise as a more informative error
                    raise RuntimeError(f"CUBLAS error - CUDA state may be corrupted or tensor shapes invalid. Context: {len(context)} chars, Model max: {getattr(self, 'model_max_length', 'unknown')} tokens. Error: {error_str[:200]}")
                else:
                    print(f"❌ RuntimeError during LLM generation: {e}", flush=True)
                    import traceback
                    print(traceback.format_exc(), flush=True)
                    raise
            except Exception as e:
                print(f"❌ Error during LLM generation: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                raise
        except Exception as e:
            # Catch any errors from the outer try block (shouldn't happen if inner except blocks work)
            print(f"❌ Unexpected error in generate method: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            raise

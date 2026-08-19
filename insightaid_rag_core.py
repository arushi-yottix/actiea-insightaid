#!/usr/bin/env python3
"""
Aircraft Technical Documents RAG System
Single-file, simplified version using only transformers.pipeline
Optimized for: openai-community/gpt-oss-20b- cuda out of memory

Current model: Qwen/Qwen2.5-7B-Instruct (7B - optimized for T4 GPU, uses ~7-8GB)
Previous models tested:
- mlx-community/Llama-3.1-8B-Instruct - 8B model, too large for T4 (uses ~12.7GB)
- meta-llama/Llama-3.1-8B-Instruct - could not use (gated)

Qdrant RAG System
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import time
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
#, Query
from qdrant_client.http import models

import torch
from transformers import pipeline

# ----------------------------
# CONFIGURATION
# ----------------------------
@dataclass
class RAGConfig:
    pdf_directory: str = "/home/mirawo_dgxspark/VanathiS/documents"
    image_dir: str = "./assets/images"
    
    # Native text chunking (sentence-based)
    native_chunk_size: int = 400  # words
    native_overlap: int = 80
    
    # OCR text chunking (word-window)
    # Optimized for bge-m3 and procedural manuals/flowcharts
    ocr_chunk_size: int = 384  # words (sweet spot for bge-m3)
    ocr_overlap: int = 96  # 25% overlap
    
    min_chunk_size: int = 80  # Minimum chunk size for embeddings (lowered to capture short but valuable pages like flowcharts)

    #embedding_model: str = "BAAI/bge-m3"
    #embedding_model: str = "BAAI/bge-base-en-v1.5"  # 768 dim, smaller model
    #embedding_model: str = "all-MiniLM-L6-v2"  # 384 dim, ~80MB - good for testing
    #embedding_dim: int = 1024  # bge-m3 produces 1024-dimensional vectors
    # Set to 768 for bge-base-en-v1.5, 384 for all-MiniLM-L6-v2

    embedding_model: str = "all-MiniLM-L6-v2"  # ~80MB, much faster
    embedding_dim: int = 384

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = None
    qdrant_collection: str = "aircraft_docs"

    top_k_initial: int = 20
    final_top_k: int = 7

    # LLM
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"  # Mistral-7B-Instruct-v0.3 - Apache 2.0, no gating
    # Options: Mistral-7B-Instruct-v0.3 (recommended, no license needed), Llama-3.1-8B (requires license), Qwen2.5-7B
    # Note: Mistral uses Apache 2.0 license - no HuggingFace license acceptance required
    max_new_tokens: int = 500  # Increased from 400 to 500 for more detailed responses (especially for Qwen 3B)
    temperature: float = 0.1  # Lower temperature for more deterministic, consistent outputs

    def __post_init__(self):
        Path(self.image_dir).mkdir(parents=True, exist_ok=True)

config = RAGConfig()

# ----------------------------
# TIMER UTILITY
# ----------------------------
class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"Started: {self.name}")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"Completed: {self.name} in {elapsed:.2f}s")


# ----------------------------
# 1. PDF INGESTION + IMAGES
# ----------------------------
class PDFIngestor:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg

    def extract(self, pdf_path: Path) -> Dict:
        """
        Extract PDF pages - ALWAYS render every page to PNG.
        No conditional logic - deterministic pipeline.
        """
        doc = fitz.open(pdf_path)
        doc_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]

        pages = []
        images_saved = []

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
               
                # Save embedded images (if any)
                image_ids = []
                try:
                    for img_index, img in enumerate(page.get_images(full=True)):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            ext = base_image["ext"]
                            img_hash = hashlib.md5(image_bytes).hexdigest()
                            img_id = img_hash
                            img_path = Path(self.cfg.image_dir) / f"{img_id}.{ext}"
                            if not img_path.exists():
                                img_path.write_bytes(image_bytes)
                            image_ids.append(img_id)
                            images_saved.append({"id": img_id, "path": str(img_path)})
                        except Exception as e:
                            print(f"Warning: could not extract image {img_index} from page {page_num+1}: {e}")
                            continue
                except Exception as e:
                    print(f"Warning: could not process images for page {page_num+1}: {e}")

                # CRITICAL: Always render every page to PNG (no conditions)
                # This ensures deterministic OCR pipeline
                # Use dpi only (not both dpi and matrix) to avoid double-scaling
                try:
                    render_start = perf_counter()
                    pix = page.get_pixmap(dpi=400)
                    render_bytes = pix.tobytes()
                    render_hash = hashlib.md5(render_bytes).hexdigest()
                    render_id = render_hash
                    render_path = Path(self.cfg.image_dir) / f"{render_id}.png"
                    
                    if not render_path.exists():
                        pix.save(str(render_path))
                    
                    image_ids.append(render_id)
                    images_saved.append({"id": render_id, "path": str(render_path)})
                    
                    render_latency = perf_counter() - render_start
                    print(f"✅ Rendered page {page_num+1} to PNG: {render_path.name} (latency: {render_latency:.3f}s)")
                except Exception as e:
                    print(f"❌ Error rendering page {page_num+1}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    # If rendering fails, skip this page (cannot proceed without PNG)
                    continue
 
                # Store page metadata - NO native text (OCR will extract it)
                pages.append({
                    "page_num": page_num + 1,
                    "image_ids": image_ids,
                    "render_path": str(render_path)  # Always present now
                })
            except Exception as e:
                # If entire page processing fails, log and continue with next page
                print(f"❌ Error processing page {page_num+1}: {e}")
                import traceback
                print(traceback.format_exc())
                continue

        doc.close()
        print(f"📄 Extracted {len(pages)} pages from {pdf_path.name}")
        return {
            "doc_id": doc_id,
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "pages": pages,
            "images": images_saved
        }

# ----------------------------
# 2. TEXT NORMALIZATION
# ----------------------------
def normalize_text(text: str) -> str:
    """
    Normalize OCR text: clean up whitespace, remove artifacts.
    """
    if not text:
        return ""
   
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
   
    # Remove non-printable ASCII characters (keep symbols like ≤ ≥ ± → for flowcharts)
    # This preserves flowchart semantics while removing OCR junk
    text = re.sub(r'[^\x20-\x7E]', '', text)  # Keep all printable ASCII (0x20-0x7E)
   
    return text.strip()
 
 
# ----------------------------
# 3. WORD-WINDOW CHUNKING (for OCR text)
# ----------------------------
def chunk_ocr_text(text: str, cfg: RAGConfig) -> List[str]:
    """
    Chunk OCR text using word-window approach (not sentence-based).
    This preserves YES/NO adjacency and decision grouping in flowcharts.
    """
    words = text.split()
    if not words:
        return []
   
    chunks = []
    i = 0
   
    # Validate chunk parameters to prevent infinite loop
    step = cfg.ocr_chunk_size - cfg.ocr_overlap
    if step <= 0:
        raise ValueError(f"ocr_overlap ({cfg.ocr_overlap}) must be smaller than ocr_chunk_size ({cfg.ocr_chunk_size})")
    
    while i < len(words):
        # Take chunk_size words
        chunk_words = words[i:i + cfg.ocr_chunk_size]
       
        if len(chunk_words) >= cfg.min_chunk_size:
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
       
        # Move forward with overlap (validated step size)
        i += step
       
        # Prevent infinite loop (safety check)
        if i >= len(words):
            break
   
    return chunks
 
 
# ----------------------------
# 4. CHUNKING + DEDUPLICATION
# ----------------------------
def _process_page_ocr(page: Dict, doc: Dict, cfg: RAGConfig) -> List[Dict]:
    """
    Process a single page: OCR + chunking.
    Returns list of chunk dicts, or empty list if page should be skipped.
    """
    # Skip if no render_path (should not happen, but safety check)
    render_path = page.get("render_path")
    if not render_path or not Path(render_path).exists():
        print(f"⚠️ Skipping page {page.get('page_num')}: no render_path")
        return []
   
    # CRITICAL: Always OCR every page (no conditions)
    # PSM 11 is better for flowcharts with sparse text (boxes, arrows, labels)
    try:
        ocr_start = perf_counter()
        ocr_config = r'--oem 3 --psm 11'
        ocr_text = pytesseract.image_to_string(
            Image.open(render_path),
            lang="eng",
            config=ocr_config
        )
        ocr_text = ocr_text.strip()
        ocr_latency = perf_counter() - ocr_start

        if not ocr_text:
            print(f"⚠️ No OCR text extracted from page {page.get('page_num')} (latency: {ocr_latency:.3f}s)")
            return []

        # Normalize OCR text
        text = normalize_text(ocr_text)
       
        # Check for minimum meaningful content (use min_chunk_size for consistency)
        # Pages with flowcharts/tables may have less text but still be valuable
        if not text or len(text.split()) < cfg.min_chunk_size:
            print(f"⚠️ Skipping page {page.get('page_num')}: OCR text too short ({len(text.split())} words, need {cfg.min_chunk_size})")
            return []
       
        # Chunk using word-window approach (NOT sentence-based)
        chunk_start = perf_counter()
        page_chunks = chunk_ocr_text(text, cfg)
        chunk_latency = perf_counter() - chunk_start
        
        if not page_chunks:
            print(f"⚠️ No chunks created from page {page.get('page_num')}")
            return []
        
        print(f"✅ Page {page.get('page_num')}: OCR extracted {len(text.split())} words, created {len(page_chunks)} chunks (OCR: {ocr_latency:.3f}s, chunking: {chunk_latency:.3f}s)")
       
        # Create chunk objects
        page_chunk_objects = []
        for chunk_text in page_chunks:
            page_chunk_objects.append({
                "text": chunk_text,
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "page_num": page["page_num"],
                "image_ids": ",".join(page.get("image_ids", [])),
                "is_ocr": True,  # All chunks are OCR now
                "_page_hash": hashlib.md5(text.encode()).hexdigest()  # For deduplication
            })
        
        return page_chunk_objects
           
    except Exception as e:
        print(f"❌ OCR failed for page {page.get('page_num')}: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def chunk_document(doc: Dict, cfg: RAGConfig) -> List[Dict]:
    """
    Chunk document using OCR-only pipeline with PARALLEL OCR processing.
    - Always OCR every page (no conditions)
    - Use word-window chunking (not sentence-based)
    - Page-level and chunk-level deduplication
    - Parallel OCR processing for faster ingestion
    """
    chunks = []
    seen_pages = set()  # Page-level deduplication
    
    # Filter pages with valid render_path
    valid_pages = [p for p in doc["pages"] if p.get("render_path") and Path(p.get("render_path")).exists()]
    
    if not valid_pages:
        print("⚠️ No valid pages to process")
        return []
    
    # Determine max workers (use CPU count, but cap at reasonable limit)
    max_workers = min(len(valid_pages), os.cpu_count() or 4, 8)  # Cap at 8 to avoid overwhelming system
    
    print(f"🔄 Processing {len(valid_pages)} pages with parallel OCR (max_workers={max_workers})...")
    parallel_start = perf_counter()
    
    # Process pages in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pages for processing
        future_to_page = {
            executor.submit(_process_page_ocr, page, doc, cfg): page 
            for page in valid_pages
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page = future_to_page[future]
            try:
                page_chunks = future.result()
                
                # Apply page-level deduplication
                for chunk in page_chunks:
                    page_hash = chunk.get("_page_hash")
                    if page_hash and page_hash not in seen_pages:
                        seen_pages.add(page_hash)
                        # Remove internal hash before adding to chunks
                        chunk.pop("_page_hash", None)
                        chunks.append(chunk)
                    elif page_hash and page_hash in seen_pages:
                        print(f"⚠️ Skipping page {page.get('page_num')}: duplicate page content")
            except Exception as e:
                print(f"❌ Error processing page {page.get('page_num')}: {e}")
                import traceback
                print(traceback.format_exc())
                continue
    
    parallel_latency = perf_counter() - parallel_start
    print(f"📊 Parallel OCR processing complete: {len(chunks)} chunks from {len(valid_pages)} pages in {parallel_latency:.3f}s")
    
    # Chunk-level deduplication
    seen_chunks = set()
    unique = []
    for c in chunks:
        chunk_hash = hashlib.md5(c["text"].encode()).hexdigest()
        if chunk_hash not in seen_chunks:
            seen_chunks.add(chunk_hash)
            unique.append(c)
    
    print(f"📊 Total chunks: {len(chunks)} → {len(unique)} after deduplication")
    return unique


# ----------------------------
# 3. EMBEDDINGS
# ----------------------------
class Embedder:
    def __init__(self, cfg: RAGConfig):
        print(f"Loading embedding model: {cfg.embedding_model}")
        # Force safetensors usage to avoid PyTorch 2.6 requirement
        # SentenceTransformer automatically prefers safetensors if available
        # FOR PRODUCTION: Use CPU for embeddings (recommended for stability)
        # - BAAI/bge-m3 (~560M params, 1024 dim) works well on CPU
        # - Typical performance: 8-40 sentences/sec on CPU (more than enough for RAG)
        # - Single query: ~15-60ms, batch of 4: ~150-600ms (acceptable)
        # - Avoids CUDA conflicts with LLM model
        # - More stable, especially with GB10 compatibility issues
        # - LLM model uses GPU (which is what really needs it)
        # NOTE: If you need GPU embeddings (high-throughput scenarios), change to "cuda"
        #       but ensure CUDA initialization is done (see insightaid_api_server.py)
        # FOR TESTING: Use "all-MiniLM-L6-v2" (~80MB, 384 dim) - much faster to download
        self.device = "cpu"  # Force CPU for production stability
        
        # Model size info
        model_info = {
            "BAAI/bge-m3": "~560M params, 1024 dim, ~2.27GB",
            "BAAI/bge-base-en-v1.5": "~110M params, 768 dim, ~420MB",
            "all-MiniLM-L6-v2": "~22M params, 384 dim, ~80MB (good for testing)"
        }
        info = model_info.get(cfg.embedding_model, f"{cfg.embedding_dim} dim")
        print(f"   ℹ️  Using CPU for embeddings ({cfg.embedding_model} - {info})")
        print(f"   📊 Performance: ~8-40 sentences/sec - sufficient for typical RAG workloads")
        print(f"   💡 LLM model uses GPU separately - this avoids CUDA conflicts")
        
        print(f"   Device: {self.device} (safetensors format will be preferred automatically)")
        self.model = SentenceTransformer(
            cfg.embedding_model,
            device=self.device
        )

    def embed_chunks(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        # Using CPU - no CUDA initialization needed
        # This is fast enough for production and avoids CUDA conflicts
        # Using batch_size=4 for memory efficiency
        embed_start = perf_counter()
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=4, show_progress_bar=True)
        embed_latency = perf_counter() - embed_start
        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb.astype(np.float32).tobytes()
        print(f"📊 Embedded {len(chunks)} chunks in {embed_latency:.3f}s ({len(chunks)/embed_latency:.1f} chunks/sec)")
        return chunks

    def embed_query(self, query: str) -> bytes:
        # Using CPU - no CUDA initialization needed
        # This is fast enough for production and avoids CUDA conflicts
        embed_start = perf_counter()
        emb = self.model.encode([query], normalize_embeddings=True)[0]
        embed_latency = perf_counter() - embed_start
        print(f"📊 Query embedding latency: {embed_latency:.3f}s")
        return emb.astype(np.float32).tobytes()


# ----------------------------
# 4. QDRANT VECTOR STORE
# ----------------------------
class QdrantStore:
    def __init__(self, cfg: RAGConfig):
        self.collection_name = cfg.qdrant_collection
        self.embedding_dim = cfg.embedding_dim
        
        if cfg.qdrant_api_key:
            self.client = QdrantClient(url=f"http://{cfg.qdrant_host}:{cfg.qdrant_port}", api_key=cfg.qdrant_api_key, check_compatibility=False)
        else:
            self.client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
            #, check_compatibility=False)
        
        self._create_collection(cfg)
    
    def _create_collection(self, cfg: RAGConfig):
        try:
            collection_info = self.client.get_collection(self.collection_name)
            # Check if vector size matches expected dimension
            existing_dim = None
            
            # Get vector size from collection config
            try:
                # Try to get vector config - Qdrant API structure
                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                    params = collection_info.config.params
                    if hasattr(params, 'vectors'):
                        vectors = params.vectors
                        # Handle both named vectors dict and single VectorParams
                        if isinstance(vectors, dict):
                            # Named vectors - check default or first
                            if 'default' in vectors:
                                existing_dim = vectors['default'].size
                            elif len(vectors) > 0:
                                first_key = next(iter(vectors.keys()))
                                existing_dim = vectors[first_key].size
                        elif hasattr(vectors, 'size'):
                            # Single VectorParams object
                            existing_dim = vectors.size
            except Exception as e:
                print(f"⚠️  Could not determine existing vector size: {e}")
            
            if existing_dim and existing_dim != cfg.embedding_dim:
                error_msg = (
                    f"\n❌ Vector dimension mismatch detected!\n"
                    f"   Collection: '{self.collection_name}'\n"
                    f"   Existing vector size: {existing_dim}\n"
                    f"   Required vector size: {cfg.embedding_dim} (from {cfg.embedding_model})\n\n"
                    f"   To fix this, you have two options:\n\n"
                    f"   Option 1: Delete and recreate collection (RECOMMENDED)\n"
                    f"   ----------------------------------------\n"
                    f"   1. Delete the collection using Qdrant API or UI:\n"
                    f"      curl -X DELETE http://{cfg.qdrant_host}:{cfg.qdrant_port}/collections/{self.collection_name}\n"
                    f"   2. Restart the server - it will auto-create with correct dimensions\n"
                    f"   3. Re-upload your documents\n\n"
                    f"   Option 2: Change embedding model to match collection\n"
                    f"   ----------------------------------------\n"
                    f"   In insightaid_rag_core.py, change:\n"
                    f"   - If collection is 1024: embedding_model = 'BAAI/bge-m3', embedding_dim = 1024\n"
                    f"   - If collection is 768: embedding_model = 'BAAI/bge-base-en-v1.5', embedding_dim = 768\n"
                )
                print(error_msg)
                raise ValueError(f"Vector dimension mismatch: collection has {existing_dim}, but model produces {cfg.embedding_dim}")
            
            if existing_dim:
                print(f"✅ Collection '{self.collection_name}' exists with correct dimensions ({existing_dim}).")
            else:
                print(f"✅ Collection '{self.collection_name}' exists.")
        except ValueError:
            # Re-raise dimension mismatch errors
            raise
        except Exception as e:
            # Collection doesn't exist, create it
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=cfg.embedding_dim, distance=Distance.COSINE)
                )
                print(f"✅ Created new collection '{self.collection_name}' with vector size {cfg.embedding_dim}.")
            except Exception as create_err:
                print(f"❌ Error creating collection: {create_err}")
                raise
    
    def delete_collection(self, confirm: bool = False):
        """Delete the collection (use with caution - deletes all data)"""
        if not confirm:
            raise ValueError("Must set confirm=True to delete collection")
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✅ Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False
    
    def _bytes_to_np(self, vec_bytes: bytes) -> np.ndarray:
        return np.frombuffer(vec_bytes, dtype=np.float32)

    def upsert(self, chunks: List[Dict]):
        points = []
        for i, c in enumerate(chunks):
            embedding = self._bytes_to_np(c["embedding"])
            point_id = abs(hash(f"{c['doc_id']}_{i}")) % (2**63)
            
            payload = {
                "text": c["text"],
                "filename": c["filename"],
                "page_num": int(c.get("page_num", 0)),
                "doc_id": c["doc_id"],
                "image_ids": c.get("image_ids", ""),
            }
            
            if c["doc_id"].startswith("temp_"):
                # Extract session_id from doc_id format: temp_{session_id}_{rest}
                # IMPORTANT: Use maxsplit=2 to handle doc_ids that contain underscores
                # However, session_id should NOT contain underscores (use UUID or simple strings)
                parts = c["doc_id"].split("_", 2)  # maxsplit=2: ["temp", "session_id", "rest..."]
                if len(parts) >= 2:  # At least temp and session_id
                    session_id = parts[1]  # Get session_id (second part)
                    # CRITICAL: Normalize session_id to ensure consistent string format
                    # This must match the normalization in insightaid_api_server.py
                    normalized_session_id = str(session_id).strip()
                    if normalized_session_id:  # Only set if not empty
                        payload["session_id"] = normalized_session_id
                        payload["is_temp"] = True
                        # Only log first chunk to avoid spam
                        if i == 0:
                            print(f"📝 Storing chunk with session_id='{normalized_session_id}' (type={type(normalized_session_id).__name__}, len={len(normalized_session_id)}), doc_id={c['doc_id']}")
                    else:
                        print(f"⚠️  Warning: Extracted session_id is empty for doc_id: {c['doc_id']}")
                else:
                    print(f"⚠️  Warning: Invalid temp doc_id format: {c['doc_id']} (expected: temp_<session_id>_<rest>)")
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            ))
        
        if points:
            upsert_start = perf_counter()
            self.client.upsert(collection_name=self.collection_name, points=points)
            upsert_latency = perf_counter() - upsert_start
            print(f"📊 Upserted {len(points)} points to Qdrant in {upsert_latency:.3f}s ({len(points)/upsert_latency:.1f} points/sec)")
    
    def search(self, query_vec: bytes, k: int = 10, filter_conditions: Dict = None) -> List[Dict]:
        """
        Search Qdrant collection with optional filters.
        Replaces Redis search functionality.
        """
        query_np = self._bytes_to_np(query_vec)
        query_filter = None
        
        if filter_conditions:
            # Build filter conditions - Qdrant uses AND logic for multiple conditions
            conditions = []
            for key, value in filter_conditions.items():
                # Handle boolean values specially
                if isinstance(value, bool):
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            
            if conditions:
                query_filter = Filter(must=conditions)
                print(f"   🔍 Using filter: {filter_conditions}")
            else:
                query_filter = None
        
        try:
            # Try search() method first (standard Qdrant API)
            if hasattr(self.client, 'search'):
                results = self.client.search(
                    collection_name=self.collection_name,
                    #query_vector=query_np.tolist(),
                    query_vector=query_np,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True
                )
            else:
                # If search() doesn't exist, skip query_points() and go directly to fallback
                # query_points() API varies by version and doesn't reliably support filters
                # So we'll use the scroll + manual similarity fallback instead
                raise AttributeError("search() method not available, using fallback")
        except (ConnectionError, TimeoutError) as conn_err:
            # Handle Qdrant connection failures specifically
            error_type = type(conn_err).__name__
            print(f"❌ Qdrant connection error during search: {conn_err}", flush=True)
            print(f"   Error type: {error_type}", flush=True)
            print(f"   Qdrant may be down or unreachable. Check Qdrant service status.", flush=True)
            return []  # Return empty list - let caller handle the error
        except Exception as e:
            print(f"❌ Qdrant search error: {e}")
            import traceback
            print(traceback.format_exc())
            # Try alternative: use scroll with filter and compute similarity manually
            print(f"   ⚠️  Attempting fallback: scroll + manual similarity...")
            try:
                # Fallback: get all points with filter and compute similarity manually
                scroll_results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=min(k * 10, 1000),  # Get more points to compute similarity
                    with_payload=True,
                    with_vectors=True
                )
                points = scroll_results[0] if scroll_results and len(scroll_results) > 0 else []
                
                # Compute cosine similarity manually
                scored_points = []
                query_vec_norm = np.linalg.norm(query_np)
                for point in points:
                    if hasattr(point, 'vector') and point.vector:
                        vec = np.array(point.vector, dtype=np.float32)
                        vec_norm = np.linalg.norm(vec)
                        if vec_norm > 0 and query_vec_norm > 0:
                            similarity = float(np.dot(query_np, vec) / (query_vec_norm * vec_norm))
                            scored_points.append((point, similarity))
                
                # Sort by similarity and take top k
                scored_points.sort(key=lambda x: x[1], reverse=True)
                # Create result objects with score attribute for consistent processing
                class ScoredPoint:
                    def __init__(self, point, score):
                        self.payload = point.payload if hasattr(point, 'payload') else {}
                        self.score = score
                
                results = [ScoredPoint(sp[0], sp[1]) for sp in scored_points[:k]]
                print(f"   ✅ Fallback search found {len(results)} results")
            except (ConnectionError, TimeoutError) as conn_err2:
                # Handle connection errors in fallback too
                print(f"   ❌ Fallback search failed due to Qdrant connection error: {conn_err2}", flush=True)
                print(f"   Qdrant service appears to be unavailable.", flush=True)
                return []
            except Exception as e2:
                print(f"   ❌ Fallback also failed: {e2}")
                return []
        
        hits = []
        # Safety check: ensure results is iterable
        if not results:
            print(f"   ⚠️  Search returned no results")
            return []
        
        # Process results (can be list of ScoredPoint or QueryResponse)
        for result in results:
            # Handle ScoredPoint structure (from search())
            if hasattr(result, 'payload') and hasattr(result, 'score'):
                payload = result.payload
                score = result.score
            # Handle Point structure (from query_points)
            elif hasattr(result, 'payload'):
                payload = result.payload
                score = getattr(result, 'score', 1.0)
            else:
                # Fallback for dict-like structures
                payload = getattr(result, 'payload', {})
                score = getattr(result, 'score', 1.0)
            
            hits.append({
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "page_num": int(payload.get("page_num", 0)),
                "image_ids": payload.get("image_ids", "").split(",") if payload.get("image_ids") else [],
                "score": float(score),
                "doc_id": payload.get("doc_id", ""),
                # Include session_id and is_temp for debugging
                "session_id": payload.get("session_id"),
                "is_temp": payload.get("is_temp", False)
            })
        return hits


# ----------------------------
# 5. RETRIEVER + RERANK
# ----------------------------
def retrieve(query: str, embedder: Embedder, store: QdrantStore, cfg: RAGConfig):
    qvec = embedder.embed_query(query)
    
    # Simple approach: Use standard semantic search + keyword overlap
    # No complex figure/sheet/task matching since those identifiers may not be in chunk text
    # (Flowcharts are OCR'd and may not contain exact figure numbers)
    q_lower = query.lower()
    candidates = store.search(qvec, cfg.top_k_initial)
    
    print(f"🔍 Retrieved {len(candidates)} initial candidates from vector search (k={cfg.top_k_initial})")

    # Extract filename from query if specified (e.g., "from file X" or "in file X")
    query_filename = None
    filename_patterns = [
        r'from\s+file\s+(.+?)(?:\s|$|,|\.|\?|$)',  # "from file X"
        r'in\s+file\s+(.+?)(?:\s|$|,|\.|\?|$)',    # "in file X"
        r'file\s+(.+?)(?:\s|$|,|\.|\?|$)',          # "file X"
    ]
    for pattern in filename_patterns:
        match = re.search(pattern, q_lower)
        if match:
            query_filename = match.group(1).strip().rstrip('.,?')
            # Remove common trailing words
            query_filename = re.sub(r'\s+(pdf|document|file)$', '', query_filename)
            print(f"📄 Detected filename in query: '{query_filename}'")
            break
    
    # Extract table reference from query (e.g., "table-1", "Table 1", "table 1")
    query_table_ref = None
    table_patterns = [
        r'table\s*[-\s]*(\d+)',  # "table-1", "table 1", "Table 1"
        r'table\s*n[°o]\s*(\d+)',  # "Table n° 1", "Table no 1"
    ]
    for pattern in table_patterns:
        match = re.search(pattern, q_lower)
        if match:
            query_table_ref = match.group(1)
            print(f"📊 Detected table reference in query: 'Table {query_table_ref}'")
            break
    
    qwords = set(re.findall(r'\w+', q_lower))
    
    # Simple reranking: keyword overlap + filename/table matching only
    # No complex figure/sheet/task matching since those may not be in OCR'd text
    for c in candidates:
        try:
            text = c.get("text", "")
            if not text:
                c["final_score"] = 0.0
                continue
            
            text_lower = text.lower()
            cwords = set(re.findall(r'\w+', text_lower))
            
            # Basic keyword overlap
            overlap = len(qwords & cwords) / max(len(qwords), 1)
            
            # Simple boosts: filename and table matching only
            filename_match_boost = 0.0
            table_match_boost = 0.0
            
            # Filename matching boost (if filename specified in query)
            chunk_filename = c.get("filename", "").lower()
            if chunk_filename and query_filename:
                # Normalize both filenames for comparison
                query_fn_normalized = re.sub(r'[^\w\s-]', '', query_filename.lower())
                chunk_fn_normalized = re.sub(r'[^\w\s-]', '', chunk_filename.lower())
                
                # Check if query filename is contained in chunk filename or vice versa
                if query_fn_normalized in chunk_fn_normalized or chunk_fn_normalized in query_fn_normalized:
                    filename_match_boost = 0.3  # Moderate boost for filename match
                    print(f"   ✅ Filename match: query '{query_filename}' matches chunk '{chunk_filename}'")
            
            # Table matching boost (if table reference specified in query)
            if query_table_ref:
                # Check for table references in chunk text
                table_patterns_in_chunk = [
                    rf'\btable\s*[-\s]*{query_table_ref}\b',  # "table-1", "table 1", "Table 1"
                    rf'\btable\s*n[°o]\s*{query_table_ref}\b',  # "Table n° 1"
                    rf'\btable\s*#{query_table_ref}\b',  # "Table #1"
                ]
                for pattern in table_patterns_in_chunk:
                    if re.search(pattern, text_lower):
                        table_match_boost = 0.3  # Moderate boost for table match
                        print(f"   ✅ Table match: query 'Table {query_table_ref}' found in chunk")
                        break
            
            # Final score: standard reranking with simple boosts
            c_score = float(c.get("score", 0.0))
            enhanced_overlap = min(1.0, overlap + filename_match_boost + table_match_boost)
            c["final_score"] = 0.6 * c_score + 0.4 * enhanced_overlap
            
        except Exception as e:
            print(f"Warning: Error reranking candidate: {e}")
            c["final_score"] = 0.0

    candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    # Use standard final_top_k (no multipliers)
    final_k = cfg.final_top_k
    top_chunks = candidates[:final_k]
    
    # Debug: Log top chunks after reranking
    print(f"📊 Top {len(top_chunks)} chunks after reranking:")
    for i, c in enumerate(top_chunks[:10], 1):  # Show top 10
        filename = c.get("filename", "unknown")
        page = c.get("page_num", 0)
        score = c.get("final_score", 0.0)
        # Check if chunk contains figure/sheet references
        text_lower = c.get("text", "").lower()
        has_figure = "figure" in text_lower
        has_sheet = "sheet" in text_lower
        refs = []
        if has_figure:
            fig_match = re.search(r'figure\s+[0-9a-z\-]+', text_lower)
            if fig_match:
                refs.append(f"fig:{fig_match.group(0)[:30]}")
        if has_sheet:
            sheet_match = re.search(r'sheet\s+[0-9/]+', text_lower)
            if sheet_match:
                refs.append(f"sheet:{sheet_match.group(0)[:20]}")
        ref_str = " | " + ", ".join(refs) if refs else ""
        print(f"   [{i}] {filename} p.{page} | score={score:.3f}{ref_str}")
    
    return top_chunks


# ----------------------------
# 6. LLM GENERATOR (Only pipeline!)
# ----------------------------
# class LLM:
#     def __init__(self, cfg: RAGConfig):
#         print(f"\nLoading LLM: {cfg.model_id}")
#         self.pipe = pipeline(
#             "text-generation",
#             model=cfg.model_id,
#             torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto",
#             model_kwargs={"low_cpu_mem_usage": True}
#         )
#         print("LLM loaded successfully!\n")

class LLM:
    def __init__(self, cfg: RAGConfig):
        print(f"\nLoading LLM: {cfg.model_id}", flush=True)
        
        # Login to HuggingFace (if needed) - move this BEFORE pipeline
        huggingface_key = os.getenv("HUGGINGFACE_TOKEN")
        if huggingface_key:
            from huggingface_hub import login
            login(token=huggingface_key)
        
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
            
            # Optimize for T4 (16GB) vs A100 (40GB+)
            # T4 needs more aggressive memory management
            is_t4 = "T4" in gpu_name or gpu_memory_gb < 20
            is_gb10 = "GB10" in gpu_name or "GB200" in gpu_name
            model_kwargs = {"low_cpu_mem_usage": True}
            quantization_enabled = False  # Track if quantization is actually enabled
            quantization_bits = 0  # Track quantization bits (0 = none, 4 = 4-bit, 8 = 8-bit)
            
            # Always use direct torch_dtype argument (never in model_kwargs to avoid conflict)
            # bfloat16 has better compatibility with RoPE and modern GPUs (GB10, A100, etc.)
            pipeline_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # CPU fallback is the reliable solution for GB10
            if is_gb10:
                print(f"⚙️  GB10 GPU detected - CUBLAS compatibility note", flush=True)
                is_mistral = "mistral" in cfg.model_id.lower() or "Mistral" in cfg.model_id
                is_qwen = "qwen" in cfg.model_id.lower() or "Qwen" in cfg.model_id
                is_llama = "llama" in cfg.model_id.lower() or "Llama" in cfg.model_id
                
                if is_llama:
                    print(f"   ✅ Using Llama model - may have better GPU compatibility", flush=True)
                elif is_mistral:
                    print(f"   ⚠️  GB10 has known CUBLAS issues with Mistral models", flush=True)
                    print(f"   💡 System will automatically fallback to CPU if CUBLAS errors occur", flush=True)
                elif is_qwen:
                    print(f"   ⚠️  GB10 has known CUBLAS issues with Qwen2.5's RoPE computation", flush=True)
                    print(f"   💡 System will automatically fallback to CPU if CUBLAS errors occur", flush=True)
                else:
                    print(f"   ⚠️  GB10 may have CUBLAS compatibility issues with this model", flush=True)
                    print(f"   💡 System will automatically fallback to CPU if CUBLAS errors occur", flush=True)
                
                print(f"   📊 Using bfloat16 for best compatibility", flush=True)
                quantization_enabled = False
                quantization_bits = 0
                pipeline_dtype = torch.bfloat16
            
            if is_t4 and not is_gb10:  # Skip T4 logic if GB10 was already handled
                print(f"⚙️  T4 GPU detected - using optimized settings for 16GB memory", flush=True)
                
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
            
            # Attention implementation: SDPA causes CUBLAS errors on GB10
            # Use default/eager attention for GB10, SDPA for other GPUs
            # This check applies to all GPUs (not just T4)
            if not quantization_enabled:  # Only set attention if not using quantization (quantization has its own path)
                try:
                    if is_gb10:
                        # GB10 has CUBLAS compatibility issues with SDPA - use default attention
                        print(f"   ⚠️  GB10 detected - using default attention (SDPA causes CUBLAS errors)", flush=True)
                        # Don't set attn_implementation - use default/eager
                    elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                        # Use SDPA for other GPUs (T4, A100, etc.) - it's more efficient
                        model_kwargs["attn_implementation"] = "sdpa"
                        print(f"   Using SDPA (Scaled Dot Product Attention) for memory efficiency", flush=True)
                    else:
                        print(f"   Using default attention implementation", flush=True)
                except Exception as e:
                    print(f"   Warning: Could not set attention implementation: {e}", flush=True)
            
            # Build pipeline - dtype only if not using quantization
            # If 8-bit fails during pipeline creation, retry with 4-bit
            # Use device_map for GPU placement (best practice for all GPU sizes)
            # Note: When using device_map, we should NOT use device parameter
            # Using "auto" lets accelerate optimize memory usage (works for T4 16GB, A10 24GB, A100 40GB+, GB10 128GB)
            # - For small GPUs (T4): accelerate will use quantization or CPU offloading if needed
            # - For large GPUs: accelerate will place everything on GPU efficiently
            # Determine device_map setting
            if torch.cuda.is_available():
                device_map_setting = "auto"  # Let accelerate optimize (recommended for all GPU sizes)
                print(f"   🎯 Using device_map='auto' (accelerate will optimize for {gpu_memory_gb:.1f}GB GPU)", flush=True)
                if gpu_memory_gb < 20:
                    print(f"   💡 Small GPU detected - quantization/CPU offloading may be used if needed", flush=True)
            else:
                device_map_setting = "cpu"
                print(f"   ⚠️  CUDA not available - using CPU", flush=True)
            
            # Pass token explicitly to pipeline for gated models
            pipeline_token = huggingface_key if huggingface_key else None
            
            pipeline_created = False
            if pipeline_dtype is not None:
                # Not using quantization - include dtype (torch_dtype is deprecated)
                self.pipe = pipeline(
                    "text-generation",
                    model=cfg.model_id,
                    token=pipeline_token,  # Explicitly pass token for gated models
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
                        token=pipeline_token,  # Explicitly pass token for gated models
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
                                    token=pipeline_token,
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
                                    token=pipeline_token,
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
                                token=pipeline_token,
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
                                token=pipeline_token,
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
                                token=pipeline_token,
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
                            token=pipeline_token,
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
            "⚠️ CRITICAL SAFETY NOTICE: This is an AIRCRAFT MAINTENANCE SYSTEM. Incorrect information "
            "can lead to catastrophic failures. Base your answer EXCLUSIVELY on the provided context. "
            "NEVER guess, infer, or provide information that is not explicitly stated in the context.\n\n"
            
            "You are given extracted text from aircraft technical documentation (manuals, repair procedures, "
            "technical dispositions, SRM, AMM, NTM, etc.).\n\n"
            
            "Context:\n{context}\n\n"
            "Question:\n{query}\n\n"
            
            "INSTRUCTIONS:\n"
            "0. RESPONSE FORMAT:\n"
            "   - Start directly with the answer - DO NOT include the query text or format as \"Query: ... Answer: ...\"\n"
            "   - Provide COMPLETE, DETAILED answers that fully address the query\n"
            "   - Include relevant context, conditions, thresholds, and all necessary information from the chunks\n"
            "   - Be thorough: explain the decision logic, include all relevant details, and provide comprehensive information\n"
            "   - DO NOT be overly brief - ensure the answer fully addresses the question with sufficient detail\n"
            "   - ALWAYS cite sources using format: [Document: Filename, Page X]\n\n"
            
            "1. ANSWER FROM CONTEXT:\n"
            "   - Base the answer EXCLUSIVELY on the provided context. NEVER guess, infer, or provide information not explicitly stated.\n"
            "   - CRITICAL: DO NOT mix information from different documents or flowcharts\n"
            "   - CRITICAL: Each flowchart is INDEPENDENT - do NOT use decision questions or actions from one flowchart to answer questions about another\n"
            "   - If multiple documents are in the context, identify which document/flowchart matches the query and use ONLY that document's information\n"
            "   - Verify that all decision questions, thresholds, and actions come from the SAME flowchart/document\n"
            "   - CRITICAL: Check filename and page numbers - if query mentions specific values (e.g., \"75mm\", \"depth after blending\"), find the EXACT flowchart that contains those values\n"
            "   - CRITICAL: Verify terminology matches - if query says \"depth\" but flowchart says \"length\", they are DIFFERENT - do NOT mix them\n"
            "   - CRITICAL: If query asks about \"maximum depth after blending\" but flowchart asks about \"maximum length after blending\", they are DIFFERENT questions - find the correct flowchart\n"
            "   - If the context contains relevant information (even partial or just references), provide a complete answer based on what IS available.\n"
            "   - CRITICAL: DO NOT say 'INSUFFICIENT INFORMATION' or 'Not available' unless the context has ZERO relevant information about the query topic.\n"
            "   - A reference (e.g., 'REFER TO FIGURE 1') IS information - include it in the response.\n"
            "   - CRITICAL FOR FLOWCHARTS: Decision questions and their YES/NO branches ARE information - they tell you what happens next\n"
            "     * If context contains a decision question like \"IS THE SKIN FREE FROM CRACKS?\" with branches, this IS information\n"
            "     * You MUST follow the branches and explain what happens for YES and NO paths\n"
            "     * Example: If query asks \"What if the skin is free from cracks?\" and context shows \"IS THE SKIN FREE FROM CRACKS? YES → DO A HFEC INSPECTION\", provide: \"If the skin is free from cracks, the next step is: DO A HFEC INSPECTION. REFER TO NTM 51-10-08-250-802 OR 803.\"\n"
            "     * DO NOT say \"Not available\" when flowchart decision logic is present in the context\n"
            "   - Preserve ALL technical data, measurements, tolerances, specifications, dimensions, and values EXACTLY as written\n"
            "   - Do not round, approximate, or modify any numerical values\n"
            "   - CRITICAL: Apply LOGICAL REASONING to numerical comparisons:\n"
            "     * For \"≤\" or \"less than or equal to\" thresholds:\n"
            "       - Values EQUAL to or LESS than the threshold → YES branch\n"
            "       - Values GREATER than the threshold → NO branch\n"
            "       - Example: \"IS DEPTH ≤ 0.4mm?\" with depth = 0.3mm → YES; depth = 0.5mm → NO\n"
            "     * For \"LESS THAN\" (without \"equal to\"):\n"
            "       - Values STRICTLY LESS than the threshold → YES branch\n"
            "       - Values EQUAL to or GREATER than the threshold → NO branch\n"
            "       - CRITICAL: If question says \"IS X LESS THAN 75mm?\" and query value is exactly 75mm:\n"
            "         → 75mm is NOT less than 75mm (it's equal), so answer is NO\n"
            "       - Example: \"IS THE MAXIMUM LENGTH AFTER BLENDING LESS THAN 75 mm?\" with length = 75mm → NO (not less than)\n"
            "       - Example: \"IS THE MAXIMUM LENGTH AFTER BLENDING LESS THAN 75 mm?\" with length = 74mm → YES (is less than)\n"
            "     * For \"GREATER THAN\" (without \"equal to\"):\n"
            "       - Values STRICTLY GREATER than the threshold → YES branch\n"
            "       - Values EQUAL to or LESS than the threshold → NO branch\n"
            "     * DO NOT say \"INSUFFICIENT INFORMATION\" - compare the query value against the threshold and determine which branch it follows.\n"
            "     * Always evaluate: query value vs threshold → determine YES or NO → follow that branch to NEXT question or terminal action.\n\n"
            "   - For TABLE-BASED queries:\n"
            "     * Tables are OCR text - identify from column headers and row structure\n"
            "     * Match query values to table ranges correctly:\n"
            "       - \"Up to X\" or \"≤X\" or \"Less than or equal to X\": value must be ≤ X\n"
            "       - \"Between X and Y\" or \"X to Y\": value must be > X AND < Y (or ≤ Y if inclusive)\n"
            "       - \"Greater than X\" or \">X\": value must be > X\n"
            "     * CRITICAL: When query provides a SINGLE value (e.g., \"4 mm\", \"0.197 in\"), match it to the CORRECT range:\n"
            "       - Example: Query says \"depth is 4 mm\" → Match to \"Between 3 mm and 5 mm\" row (NOT \"Greater than 5 mm\")\n"
            "       - Example: Query says \"0.197 in\" → Match to \"Between 0.118 in and 0.197 in\" row if that range exists\n"
            "       - Compare the query value against ALL range boundaries to find the correct match\n"
            "       - For \"Between X and Y\" ranges: value must fall within the range (X < value < Y, or X ≤ value ≤ Y if inclusive)\n"
            "     * Read the complete repair action from the matching row\n\n"
            
            "2. REPAIR PROCEDURES AND TERMINAL ACTIONS (for detailed procedural questions):\n"
            "   - CRITICAL: When the context contains terminal actions (e.g., \"BLEND DAMAGE\", \"CONTACT AIRBUS\"), you MUST:\n"
            "     * Check if the context provides the FULL procedure details (not just the action name)\n"
            "     * If the context shows complete procedure steps, references, and requirements, provide ALL of them\n"
            "     * Example: If context says \"BLEND DAMAGE IMMEDIATELY (REFER TO 51-73-00 OR 51-74-00), APPLY PROTECTIVE TREATMENT (REFER TO 51-21-11) AND RESTORE ORIGINAL STANDARD PAINT SCHEME (REFER TO 51-75-12)\", provide the ENTIRE procedure, not just \"BLEND DAMAGE\"\n"
            "     * DO NOT abbreviate or summarize terminal actions - provide the complete details as stated in the context\n"
            "     * If multiple steps are required, list ALL of them in sequence\n"
            "   - For repair procedures:\n"
            "   - When asked for complete repair procedures, include:\n"
            "     * All warnings, cautions, and notes from the context\n"
            "     * All required steps in sequence as stated\n"
            "     * All required materials, tools, and equipment mentioned\n"
            "     * All inspection requirements\n"
            "   - If the context mentions additional information exists elsewhere, note this\n"
            "   - Include safety warnings prominently at the beginning if present\n\n"
            
            "3. FLOWCHART AND DECISION TREE QUERIES:\n"
            "   - IMPORTANT: All text is from OCR (Optical Character Recognition) of rendered PDF pages\n"
            "     * Flowcharts are stored as OCR'd text chunks (word-window chunking preserves YES/NO adjacency)\n"
            "     * Decision questions may appear across multiple chunks - read ALL chunks from the SAME document\n"
            "     * OCR text may have variations: \"IS DEPTH ≤ 0.4mm?\" vs \"IS DEPTH <= 0.4 mm?\" - treat as equivalent\n"
            "     * Symbols like ≤ ≥ ± → are preserved in OCR text - use them for comparisons\n\n"
            "   - DOCUMENT IDENTIFICATION (CRITICAL - DO THIS FIRST):\n"
            "     * STEP 1: Identify the EXACT document/flowchart that matches the query\n"
            "       - Check filename and page numbers in chunks\n"
            "       - Look for decision questions that contain the EXACT values mentioned in the query (e.g., if query says \"75mm\", find flowchart with \"75 mm\")\n"
            "       - Look for decision questions that match the EXACT terminology (e.g., if query says \"depth after blending\", find flowchart with \"depth after blending\", NOT \"length after blending\")\n"
            "     * STEP 2: Verify terminology matches EXACTLY\n"
            "       - \"depth\" ≠ \"length\" - these are DIFFERENT measurements\n"
            "       - \"maximum depth after blending\" ≠ \"maximum length after blending\" - these are DIFFERENT questions\n"
            "       - \"damage located between Rib X and Rib Y\" - must match EXACTLY (Rib 1-3 ≠ Rib 1-2)\n"
            "       - If query terminology does NOT match flowchart terminology, the WRONG flowchart is being used - find the correct one\n"
            "     * STEP 3: Use ONLY the identified flowchart\n"
            "       - Use ONLY decision questions, thresholds, and actions from the SAME document/flowchart\n"
            "       - DO NOT use actions from a different flowchart even if they seem similar\n"
            "       - Each flowchart is INDEPENDENT - do NOT mix them\n"
            "     * STEP 4: Scan chunks from the SAME document to find decision questions\n"
            "       - Patterns: \"IS DAMAGE WITHIN X mm?\", \"IS DEPTH ≤ Y mm?\", \"IS THE DAMAGE...?\", \"IS THE MAXIMUM LENGTH...?\", \"IS THE MAXIMUM DEPTH...?\"\n"
            "       - Decision questions and YES/NO branches ARE information - provide answers when found, do NOT say \"Not available\"\n\n"
            "   - SEQUENTIAL FLOWCHART FOLLOWING (CRITICAL - FOLLOW EXACT ORDER):\n"
            "     * Flowcharts follow SEQUENTIAL ORDER - follow decision nodes in the EXACT order they appear in OCR chunks\n"
            "     * Process: (1) Find first matching decision question in chunks, (2) Evaluate using query parameters, (3) Follow branch (YES/NO), (4) Find IMMEDIATE next question in chunks, (5) Repeat until terminal action\n"
            "     * CRITICAL: When query matches a condition (e.g., \"damage between Rib 1 and Rib 3\"):\n"
            "       - Evaluate the matching question: Does query match? → YES or NO\n"
            "       - Follow the YES/NO branch to find the IMMEDIATE NEXT question or terminal action\n"
            "       - DO NOT jump to actions from later questions - follow the sequential flow step-by-step\n"
            "       - Example: If query says \"damage between Rib 1 and Rib 3\" and question is \"IS THE DAMAGE LOCATED BETWEEN RIB 1 AND RIB 3 OR BETWEEN RIB 6 AND RIB 27?\":\n"
            "         → Answer is YES (matches), so follow YES branch to find the NEXT question (e.g., \"IS THE DAMAGE LOCATED BETWEEN RIB 1 AND RIB 2?\")\n"
            "         → DO NOT jump to terminal actions from the NO branch of this question\n"
            "     * DO NOT skip intermediate questions - follow step-by-step from START to END\n"
            "     * For location queries (Rib-based, mm ranges): Find the FIRST location check question in chunks, evaluate it, then follow to the NEXT question\n"
            "     * For depth/threshold comparisons: Apply numerical logic correctly (see comparison rules above)\n"
            "     * Pay attention to units (depth typically 0.1-0.5mm, length typically 50-200mm) - verify terminology matches\n\n"
            "   - ANSWER FORMAT (based on query completeness):\n"
            "     * CRITICAL: When presenting decision questions, ALWAYS show BOTH branches (YES and NO) explicitly\n"
            "     * Format: \"The next step is: [EXACT QUESTION TEXT FROM CHUNKS]. If YES: [YES branch action]. If NO: [NO branch action].\"\n"
            "     * DO NOT embed branch actions in the question text - separate them clearly\n"
            "     * Example CORRECT: \"The next step is: IS THE DAMAGE DUE TO SCRATCHES, GOUGES, ABRASIONS OR CORROSION? If YES: BLEND DAMAGE TO A SMOOTH POLISHED CONTOUR... If NO: REFER TO SRM FOR OTHER DAMAGE TYPES.\"\n"
            "     * Example WRONG: \"The next step is: IS THE DAMAGE DUE TO SCRATCHES, GOUGES, ABRASIONS OR CORROSION? (REFER TO SRM FOR OTHER DAMAGE TYPES)\" - this mixes the question with the NO branch\n"
            "     * CRITICAL: Before providing the answer, verify the CORRECT flowchart is being used:\n"
            "       - If query says \"depth after blending\" but flowchart says \"length after blending\", STOP - find the correct flowchart\n"
            "       - If query says \"75mm\" but flowchart has different values, verify it's the same context (e.g., both about \"after blending\")\n"
            "       - If query mentions specific Rib numbers, verify they match the flowchart's Rib ranges\n"
            "       - DO NOT use actions from a different flowchart even if the question seems similar\n\n"
            "     * If query specifies ALL required values (e.g., location AND damage type, OR length/depth after blending):\n"
            "       - FIRST: Verify the CORRECT flowchart is being used (check terminology: depth vs length, exact values, context)\n"
            "       - CRITICAL: If query says \"depth\" but flowchart says \"length\", STOP - these are DIFFERENT measurements - find the flowchart with matching terminology\n"
            "       - CRITICAL: If query says \"maximum depth after blending is 75mm\" but flowchart says \"maximum LENGTH after blending\", they are DIFFERENT - do NOT use this flowchart\n"
            "       - CRITICAL: If query mentions \"75mm\" but flowchart has different values or different context, verify it's the same flowchart\n"
            "       - Follow complete path sequentially through all decision questions in the CORRECT flowchart ONLY\n"
            "       - Evaluate each question using query parameters:\n"
            "         * For location: Does query location match the question's location range? → YES or NO\n"
            "         * For thresholds: Compare query value to threshold using correct logic (see comparison rules)\n"
            "       - After evaluating a question, follow the YES/NO branch to the IMMEDIATE NEXT question (not a later question or wrong action)\n"
            "       - Continue following sequentially until reaching a terminal action\n"
            "       - DO NOT use actions from a different flowchart even if they seem similar\n"
            "       - Provide FULL terminal action with ALL procedure details from context (not just action name)\n"
            "       - Include reasoning: condition matched, evaluation, branch followed, and complete procedure steps\n"
            "     * If query specifies PARTIAL information (e.g., location but NOT damage type):\n"
            "       - Follow path until reaching a question requiring missing information\n"
            "       - Provide EXACT question text from chunks and BOTH branches explicitly: \"The next step is: [EXACT QUESTION]. If YES: [YES action]. If NO: [NO action].\"\n"
            "       - DO NOT provide just one action or embed actions in question text - show the question and both outcomes separately\n"
            "     * If query asks about one parameter without another:\n"
            "       - Provide the relevant decision question and outcomes for EACH applicable scenario found in context\n"
            "       - Example: \"For damage within 325mm: IS DEPTH ≤ 0.4 mm? If YES: BLEND DAMAGE. If NO: CONTACT AIRBUS. For damage between 325-444mm: IS DEPTH ≤ 0.2 mm? If YES: BLEND DAMAGE. If NO: CONTACT AIRBUS.\"\n\n"
            "   - TERMINAL ACTIONS:\n"
            "     * When you reach a terminal action (e.g., \"BLEND DAMAGE\", \"CONTACT AIRBUS\"):\n"
            "       - Check if context contains FULL procedure details (not just action name)\n"
            "       - If complete procedure exists: Provide ALL steps, references, and requirements\n"
            "       - If only action name exists: Provide reasoning and action name with citation\n"
            "       - DO NOT abbreviate - expand to include all available details from context\n\n"
            "   - GENERAL RULES:\n"
            "     * Use ONLY explicitly stated conditions, branches, and outcomes from context\n"
            "     * DO NOT infer logical steps not explicitly stated\n"
            "     * Provide COMPLETE answers: condition evaluation, decision question, both branches, reasoning\n"
            "     * If information is incomplete: Describe what IS available, do NOT say \"INSUFFICIENT INFORMATION\"\n"
            "     * Always cite source: [Document: filename, Page X]\n\n"
            
            "4. TECHNICAL DOCUMENT SUMMARIZATION (for queries: \"summarize\", \"describe\", \"explain\", \"detail\", \"what is\", \"how does\"):\n"
            "   - IMPORTANT: All text is from OCR (Optical Character Recognition) - chunks contain OCR'd text from rendered PDF pages\n"
            "     * Text may have OCR artifacts but symbols (≤ ≥ ± →) are preserved\n"
            "     * Word-window chunking preserves decision logic and YES/NO adjacency\n"
            "     * Read ALL chunks from the SAME document to get complete information\n\n"
            "   - SCOPE: Handle aircraft technical documentation (SRM, AMM, NTM, repair procedures, damage assessment, flowcharts, tables, figures)\n"
            "   - For unrelated queries: State: \"This system provides information from aircraft technical documentation (SRM, AMM, NTM, repair procedures). For general knowledge or unrelated topics, consult other sources.\"\n\n"
            "   - SUMMARIZATION PROCESS (ALWAYS provide summaries):\n"
            "     * Read ALL context chunks from the SAME document\n"
            "     * Extract information: decision questions, procedures, specifications, measurements, references, terminal actions\n"
            "     * Organize by query type:\n"
            "       - \"summarize\": Provide a CONCISE but COMPLETE summary (main topics, key decisions, important thresholds, page references, all decision questions found)\n"
            "       - \"explain\": Provide DETAILED explanation (step-by-step flow, logical relationships, technical reasoning, how decisions connect)\n"
            "       - \"detail\": Provide COMPREHENSIVE information (all measurements, tolerances, complete procedures, all references, all decision paths)\n"
            "     * CRITICAL: For summarization queries, ALWAYS provide a summary - do NOT just list chunks or say \"see context\"\n"
            "     * Synthesize information from chunks into a coherent, organized summary\n"
            "     * Group related information together (e.g., all location-based thresholds, all depth checks, all terminal actions)\n\n"
            "   - FIGURE/SHEET/TABLE:\n"
            "     * CRITICAL: If query specifies a figure number (e.g., \"Figure 57-41-19-991-022-A\", \"Figure 001\"):\n"
            "       - ONLY use chunks that contain that EXACT figure number (with OCR variations allowed)\n"
            "       - CRITICAL: If query also specifies a sheet number (e.g., \"SHEET 04/4\", \"sheet 2/4\"):\n"
            "         * The SAME figure number can appear in MULTIPLE sheets (e.g., Figure 57-41-19-991-022-A in SHEET 02/4, 03/4, 04/4)\n"
            "         * You MUST match BOTH the figure number AND the sheet number\n"
            "         * DO NOT use content from a different sheet, even if the figure number matches\n"
            "         * Example: If query says \"Figure 57-41-19-991-022-A (SHEET 04/4)\", only use chunks with BOTH \"57-41-19-991-022-A\" AND \"04/4\" or \"SHEET 04/4\"\n"
            "       - DO NOT use information from other figures or documents, even if content seems similar\n"
            "       - Verify the figure number AND sheet number (if specified) in chunks match the query before using the content\n"
            "       - If no chunks contain the specified figure number (and sheet number if specified), state that the figure/sheet was not found in the context\n"
            "     * Find references in OCR text (variations: \"Figure 001\", \"FIGURE 1\", \"Fig. 1\", \"sheet 2/4\")\n"
            "     * OCR may have variations - normalize when searching (e.g., \"figure 001\" = \"FIGURE 1\")\n"
            "     * If query specifies figure/sheet/table: Include ALL information from that reference in context\n"
            "     * Include page number: [Document: filename, Page X]\n"
            "     * If only reference exists (e.g., \"REFER TO FIGURE 1\"): State the reference and its meaning\n"
            "     * For tables: CRITICAL - Tables appear as structured text in OCR chunks (NO separate table metadata or table names)\n"
            "       - Identify tables from text structure: look for column headers, rows with consistent formatting, tabular data patterns\n"
            "       - Extract table structure (headers, rows, columns, all data) directly from the OCR text\n"
            "       - Tables are embedded within document chunks - parse them from the text patterns, not from metadata\n\n"
            "   - INFORMATION RULES:\n"
            "     * If context has relevant information: Provide it from context as a SUMMARY (not just raw chunks)\n"
            "     * References and partial information ARE information - include them in summary\n"
            "     * Combine multiple related chunks into coherent, organized response\n"
            "     * Always cite: [Document: filename, Page X]\n"
            "     * DO NOT say \"Not available\" if context contains ANY relevant information\n"
            "     * Only say \"not available\" if context has ZERO information about the query topic\n\n"
        )

        messages = [
            {"role": "system", "content": (
                "You are a SAFETY-CRITICAL aircraft maintenance documentation expert. "
                "Your responses will be used for aircraft maintenance and repair operations where errors can result in catastrophic failures. "
                "Follow the detailed instructions in the user message below."
            )},
            {"role": "user", "content": A320_WING_REPAIR_PROMPT.format(context=context, query=query)}
        ]

        try:
            # Check GPU memory and adjust max_new_tokens for T4
            max_tokens = config.max_new_tokens
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
                                max_tokens = min(200, config.max_new_tokens)
                                print(f"⚠️  Very low GPU memory ({free_memory_gb:.2f} GB free) - reducing to {max_tokens} tokens", flush=True)
                            elif free_memory_gb < 3.0:
                                # Low memory - moderate tokens
                                max_tokens = min(250, config.max_new_tokens)
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (low memory: {free_memory_gb:.2f} GB free)", flush=True)
                            elif free_memory_gb < 4.0:
                                # Moderate memory - adjust based on context size
                                if context_tokens > 1000:
                                    # Large context - use fewer generation tokens
                                    max_tokens = min(200, config.max_new_tokens)
                                elif context_tokens > 1500:
                                    # Very large context - minimal tokens
                                    max_tokens = min(150, config.max_new_tokens)
                                else:
                                    max_tokens = min(250, config.max_new_tokens)
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (moderate memory: {free_memory_gb:.2f} GB free, context: ~{context_tokens} tokens)", flush=True)
                            elif free_memory_gb < 5.0:
                                # Good memory - but still consider context
                                if context_tokens > 1500:
                                    # Large context - reduce generation tokens
                                    max_tokens = min(200, config.max_new_tokens)
                                elif context_tokens > 1000:
                                    max_tokens = min(250, config.max_new_tokens)
                                else:
                                    max_tokens = min(300, config.max_new_tokens)
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (good memory: {free_memory_gb:.2f} GB free, context: ~{context_tokens} tokens)", flush=True)
                            else:
                                # Excellent memory - can use more for detailed responses
                                if context_tokens > 2000:
                                    max_tokens = min(450, config.max_new_tokens)  # Increased from 350
                                elif context_tokens > 1500:
                                    max_tokens = min(500, config.max_new_tokens)  # Increased from 350
                                else:
                                    max_tokens = min(500, config.max_new_tokens)  # Increased from 400 for better accuracy
                                print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (excellent memory: {free_memory_gb:.2f} GB free, context: ~{context_tokens} tokens)", flush=True)
                        except Exception as e:
                            # Fallback if memory check fails
                            max_tokens = min(250, config.max_new_tokens)  # Conservative default
                            print(f"⚙️  T4 GPU: Using max_new_tokens={max_tokens} (could not check memory: {e})", flush=True)
                except:
                    pass
            
            # CRITICAL: Additional safety check - truncate context if still too large
            # NOTE: Qwen2.5-3B-Instruct supports 32K+ tokens context window
            # BUT: GPU memory limits how much we can actually process
            # With quantization (3.6GB model), we can use more context
            # Without quantization (11GB model), we need less context
            
            # Dynamic context limit based on available memory
            # Model CAN handle 32K tokens, but GPU memory is the constraint
            context_token_estimate = len(context) // 4
            
            # Get model's actual max context length
            model_max_tokens = getattr(self, 'model_max_length', 32768)  # Default to 32k for Qwen2.5
            # Reserve tokens for: system prompt (~200), user query (~100), generation (~400), safety margin (~500)
            # Total reserved: ~1200 tokens
            model_safe_max = model_max_tokens - 1200
            
            # Calculate safe context limit based on free memory
            # More free memory = can process more context
            # BUT: Never exceed model's max_position_embeddings
            if torch.cuda.is_available():
                try:
                    free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
                    allocated_memory_gb = torch.cuda.memory_reserved(0) / 1e9
                    
                    # Check if quantization is active
                    # For 3B model: ~6GB non-quantized, ~2-3GB quantized
                    # For 7B model: ~11GB non-quantized, ~3-4GB quantized
                    # Detect model size from model_id
                    is_3b_model = "3B" in self.cfg.model_id or "3b" in self.cfg.model_id.lower()
                    if is_3b_model:
                        # 3B model: ~6GB is normal for non-quantized, ~2-3GB for quantized
                        is_quantized = allocated_memory_gb < 4.0
                    else:
                        # 7B or larger: ~11GB+ is normal for non-quantized
                        is_quantized = allocated_memory_gb < 6.0
                    
                    # Adjust context limit based on available memory AND quantization status
                    # For 3B models, even non-quantized can handle more context with plenty of free memory
                    if not is_quantized:
                        if is_3b_model:
                            # 3B non-quantized: Can handle more context, especially with lots of free memory
                            if free_memory_gb >= 100.0:
                                MAX_SAFE_CONTEXT_TOKENS = 8000  # Large GPU, can use more context
                            elif free_memory_gb >= 50.0:
                                MAX_SAFE_CONTEXT_TOKENS = 6000
                            elif free_memory_gb >= 20.0:
                                MAX_SAFE_CONTEXT_TOKENS = 4000
                            elif free_memory_gb >= 10.0:
                                MAX_SAFE_CONTEXT_TOKENS = 2000
                            else:
                                MAX_SAFE_CONTEXT_TOKENS = 1200
                            print(f"ℹ️  3B model (non-quantized, {allocated_memory_gb:.2f} GB) - using context limits based on free memory", flush=True)
                        else:
                            # 7B+ non-quantized - more conservative
                            if free_memory_gb >= 6.0:
                                MAX_SAFE_CONTEXT_TOKENS = 800  # Very conservative for non-quantized 7B
                            elif free_memory_gb >= 4.0:
                                MAX_SAFE_CONTEXT_TOKENS = 600
                            else:
                                MAX_SAFE_CONTEXT_TOKENS = 400  # Extremely conservative
                            print(f"⚠️  Model appears NOT quantized ({allocated_memory_gb:.2f} GB allocated) - using very conservative context limits", flush=True)
                    else:
                        # Model is quantized - can use more context
                        if free_memory_gb >= 10.0:
                            MAX_SAFE_CONTEXT_TOKENS = 4000  # With quantization: can handle more
                        elif free_memory_gb >= 8.0:
                            MAX_SAFE_CONTEXT_TOKENS = 3000
                        elif free_memory_gb >= 6.0:
                            MAX_SAFE_CONTEXT_TOKENS = 2000
                        elif free_memory_gb >= 4.0:
                            MAX_SAFE_CONTEXT_TOKENS = 1200  # Reduced from 1500
                        else:
                            MAX_SAFE_CONTEXT_TOKENS = 800  # Reduced from 1200
                    
                    # Ensure we don't exceed model's max_position_embeddings
                    MAX_SAFE_CONTEXT_TOKENS = min(MAX_SAFE_CONTEXT_TOKENS, model_safe_max)
                    print(f"📊 Context limit adjusted based on free memory ({free_memory_gb:.2f} GB): {MAX_SAFE_CONTEXT_TOKENS} tokens (model max: {model_max_tokens})", flush=True)
                except:
                    MAX_SAFE_CONTEXT_TOKENS = min(1200, model_safe_max)  # Fallback to conservative limit, but respect model max
            else:
                MAX_SAFE_CONTEXT_TOKENS = min(1200, model_safe_max)  # Conservative default, but respect model max
            
            if context_token_estimate > MAX_SAFE_CONTEXT_TOKENS:
                max_context_chars = MAX_SAFE_CONTEXT_TOKENS * 4
                original_length = len(context)
                # Smart truncation: Keep beginning (most relevant) and end (document structure)
                if original_length > max_context_chars:
                    # Keep first 70% and last 30% of allowed size
                    first_part = int(max_context_chars * 0.7)
                    last_part = max_context_chars - first_part
                    context = context[:first_part] + "\n\n[... context truncated due to memory constraints ...]\n\n" + context[-last_part:]
                    print(f"⚠️  Context truncated ({context_token_estimate} → {MAX_SAFE_CONTEXT_TOKENS} tokens) using smart truncation", flush=True)
                else:
                    context = context[:max_context_chars] + "\n\n[Context truncated due to memory constraints]"
                    print(f"⚠️  Context truncated ({context_token_estimate} → {MAX_SAFE_CONTEXT_TOKENS} tokens)", flush=True)
                # Rebuild messages with truncated context
                messages = [
                    {"role": "system", "content": (
                        "You are a technical documentation assistant for aircraft maintenance manuals. "
                        "Answer with precision using ONLY the provided context. "
                        "NEVER guess, infer, or provide information not explicitly stated in the context."
                    )},
                    {"role": "user", "content": A320_WING_REPAIR_PROMPT.format(context=context, query=query)}
                ]
            
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
            
            # Final validation: ensure context doesn't exceed model limits
            final_context_tokens = len(context) // 4
            model_max = getattr(self, 'model_max_length', 32768)
            if final_context_tokens > model_max - 500:  # Reserve 500 tokens for generation
                print(f"⚠️  Context too long ({final_context_tokens} tokens) for model max ({model_max}), truncating...", flush=True)
                max_allowed_chars = (model_max - 500) * 4
                context = context[:max_allowed_chars] + "\n\n[Context truncated to fit model limits]"
                # Rebuild messages with truncated context
                messages = [
                    {"role": "system", "content": (
                        "You are a technical documentation assistant for aircraft maintenance manuals. "
                        "Answer with precision using ONLY the provided context. "
                        "NEVER guess, infer, or provide information not explicitly stated in the context."
                    )},
                    {"role": "user", "content": A320_WING_REPAIR_PROMPT.format(context=context, query=query)}
                ]
            
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
                        temperature=config.temperature,
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
                                    cpu_token = os.getenv("HUGGINGFACE_TOKEN")
                                    self.cpu_pipe = pipeline(
                                        "text-generation",
                                        model=self.cfg.model_id,
                                        token=cpu_token,  # Explicitly pass token for gated models
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
# ----------------------------
# MAIN RAG SYSTEM
# ----------------------------
def collect_related_images(chunks, image_dir, max_images=3):
        seen = set()
        images = []

        for c in chunks:
            for img_id in c.get("image_ids", []):
                if not img_id or img_id in seen:
                    continue

                # find actual file
                for ext in ("png", "jpg", "jpeg", "svg"):
                    path = Path(image_dir) / f"{img_id}.{ext}"
                    if path.exists():
                        images.append({
                            "image_id": img_id,
                            "path": str(path),
                            "source": f"{c['filename']} | Page {c['page_num']}",
                            "score": c.get("final_score", 0)
                        })
                        seen.add(img_id)
                        break

            if len(images) >= max_images:
                break

        return images

class AircraftRAG:
    def __init__(self):
        self.cfg = config
        self.ingestor = PDFIngestor(self.cfg)
        self.embedder = Embedder(self.cfg)
        self.store = QdrantStore(self.cfg)
        self.llm = LLM(self.cfg)

    def ingest(self):
        pdfs = list(Path(self.cfg.pdf_directory).rglob("*.pdf"))
        print(f"Found {len(pdfs)} PDFs")

        all_chunks = []
        for pdf in pdfs:
            print(f"Processing {pdf.name}...")
            doc = self.ingestor.extract(pdf)
            chunks = chunk_document(doc, self.cfg)
            all_chunks.extend(chunks)

        print(f"Total unique chunks: {len(all_chunks)}")
        embedded = self.embedder.embed_chunks(all_chunks)
        self.store.upsert(embedded)
        print("Ingestion complete!\n")
    

    def ask(self, question: str):
        print(f"\nQuestion: {question}\n")
        chunks = retrieve(question, self.embedder, self.store, self.cfg)

        if not chunks:
            print("⚠️  No chunks retrieved from vector search. Check if documents are ingested.")
            return
        
        top_score = chunks[0].get("final_score", 0.0)
        if top_score < 0.2:  # Lowered threshold from 0.3 to 0.2 for better recall
            print(f"⚠️  Low relevance score ({top_score:.3f}) - proceeding anyway to check context quality")
            print(f"   Top chunk: {chunks[0].get('filename', 'unknown')} p.{chunks[0].get('page_num', 0)}")
            # Don't return - let it proceed to see if context is actually useful

        # 🟢 NEW: collect images
        images = collect_related_images(
            chunks,
            image_dir=self.cfg.image_dir,
            max_images=2
        )

        # build context with debug info
        print(f"\n📚 Building context from {len(chunks)} chunks:")
        context_lines = []
        context_lines.append("RELEVANT DOCUMENT EXCERPTS:")
        context_lines.append("=" * 60)
        for i, c in enumerate(chunks, 1):
            filename = c.get('filename', 'unknown')
            page = c.get('page_num', 0)
            score = c.get('final_score', 0.0)
            text_preview = c.get('text', '')[:100].replace('\n', ' ')
            
            # Check for figure/sheet references in chunk
            text_lower = c.get('text', '').lower()
            has_figure = "figure" in text_lower
            has_sheet = "sheet" in text_lower
            
            ref_info = ""
            if has_figure or has_sheet:
                refs = []
                if has_figure:
                    fig_match = re.search(r'figure\s+[0-9a-z\-/]+', text_lower)
                    if fig_match:
                        refs.append(f"Figure: {fig_match.group(0)[:40]}")
                if has_sheet:
                    sheet_match = re.search(r'sheet\s+[0-9/]+', text_lower)
                    if sheet_match:
                        refs.append(f"Sheet: {sheet_match.group(0)[:20]}")
                if refs:
                    ref_info = f" | {' | '.join(refs)}"
            
            print(f"   [{i}] {filename} p.{page} | score={score:.3f}{ref_info}")
            print(f"       Preview: {text_preview}...")
            
            context_lines.append(f"[{i}] Document: {filename} | Page {page} | Score: {score:.3f}")
            context_lines.append(c['text'])
            context_lines.append("-" * 50)

        context = "\n".join(context_lines)

        print(f"\n📝 Context built: {len(context)} characters, {len(chunks)} chunks")
        print("Generating answer...")
        answer = self.llm.generate(context, question)

        # output
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(answer)

        # 🟢 NEW: show images
        if images:
            print("\nRELATED FIGURES:")
            for img in images:
                print(f"• {img['path']}  ({img['source']})")
        else:
            print("\n(No related images found)")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    rag = AircraftRAG()
    
    print("Do you want to re-ingest documents? (y/n)")
    if input().lower().startswith('y'):
        with Timer("Full ingestion"):
            rag.ingest()
    print("\nRAG System Ready! Ask questions (or type 'quit')")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            rag.ask(q)

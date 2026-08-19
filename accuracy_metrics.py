"""
Priority 2 Accuracy Metrics Implementation
- F1 Score (Token Overlap)
- BERTScore (Semantic Similarity)
- Procedure Completeness

Integrates with the RAG system to measure answer quality.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("aircraft_rag.accuracy")

# Try to import optional dependencies
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert_score not installed. BERTScore will be disabled. Install with: pip install bert-score")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not installed. ROUGE will be disabled. Install with: pip install rouge-score")


# ---------------------------
# 1. F1 Score (Token Overlap)
# ---------------------------
def f1_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score based on token overlap.
    
    Measures: Token-level overlap (good for partial matches)
    Use for: Procedure steps, multi-part answers
    
    Args:
        predicted: Generated answer text
        ground_truth: Expected answer text
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not predicted or not ground_truth:
        return 0.0
    
    pred_tokens = set(predicted.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # Calculate precision and recall
    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(intersection) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    
    # F1 score (harmonic mean)
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def f1_score_detailed(predicted: str, ground_truth: str) -> Dict[str, Any]:
    """
    Calculate F1 score with detailed breakdown.
    
    Returns:
        {
            "f1_score": float,
            "precision": float,
            "recall": float,
            "predicted_tokens": int,
            "ground_truth_tokens": int,
            "overlapping_tokens": int
        }
    """
    if not predicted or not ground_truth:
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "predicted_tokens": 0,
            "ground_truth_tokens": 0,
            "overlapping_tokens": 0
        }
    
    pred_tokens = set(predicted.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    
    intersection = pred_tokens & gt_tokens
    
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(intersection) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "predicted_tokens": len(pred_tokens),
        "ground_truth_tokens": len(gt_tokens),
        "overlapping_tokens": len(intersection)
    }


# ---------------------------
# 2. BERTScore (Semantic Similarity)
# ---------------------------
def bertscore_accuracy(predicted: str, ground_truth: str) -> Optional[float]:
    """
    Calculate BERTScore for semantic similarity.
    
    Measures: Semantic similarity using BERT embeddings
    Better than cosine similarity because:
    - Context-aware
    - Handles paraphrasing
    - More accurate for technical terms
    
    Args:
        predicted: Generated answer text
        ground_truth: Expected answer text
    
    Returns:
        BERTScore F1 (0.0 to 1.0) or None if bert_score not available
    """
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore not available - install with: pip install bert-score")
        return None
    
    if not predicted or not ground_truth:
        return 0.0
    
    try:
        P, R, F1 = bert_score_func([predicted], [ground_truth], lang='en', verbose=False)
        return F1.item()  # Returns F1 score (harmonic mean of precision and recall)
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        return None


def bertscore_accuracy_detailed(predicted: str, ground_truth: str) -> Optional[Dict[str, Any]]:
    """
    Calculate BERTScore with detailed breakdown.
    
    Returns:
        {
            "bertscore_f1": float,
            "bertscore_precision": float,
            "bertscore_recall": float
        } or None if not available
    """
    if not BERTSCORE_AVAILABLE:
        return None
    
    if not predicted or not ground_truth:
        return {
            "bertscore_f1": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0
        }
    
    try:
        P, R, F1 = bert_score_func([predicted], [ground_truth], lang='en', verbose=False)
        return {
            "bertscore_f1": F1.item(),
            "bertscore_precision": P.item(),
            "bertscore_recall": R.item()
        }
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        return None


# ---------------------------
# 2.5. BERTScore Against Context (No Ground Truth Needed)
# ---------------------------
def bertscore_against_context(
    predicted: str, 
    context_chunks: List[Dict],
    method: str = "max"
) -> Optional[Dict[str, Any]]:
    """
    Calculate BERTScore comparing predicted answer against retrieved context documents.
    
    This is useful when you don't have ground truth but want to measure:
    - Answer-source alignment (how well answer matches retrieved docs)
    - Hallucination detection (low score = answer not in context)
    - Source faithfulness
    
    Args:
        predicted: Generated answer text
        context_chunks: List of retrieved chunks with 'text' field
        method: How to combine scores from multiple chunks
            - "max": Use maximum score (best match)
            - "mean": Use average score across all chunks
            - "weighted": Weight by chunk relevance score if available
    
    Returns:
        {
            "bertscore_f1": float,
            "bertscore_precision": float,
            "bertscore_recall": float,
            "best_matching_chunk": int,  # Index of chunk with highest score
            "method": str
        } or None if not available
    """
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore not available - install with: pip install bert-score")
        return None
    
    if not predicted or not context_chunks:
        return {
            "bertscore_f1": 0.0,
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "best_matching_chunk": -1,
            "method": method
        }
    
    try:
        # Extract text from chunks
        context_texts = [chunk.get("text", "") for chunk in context_chunks if chunk.get("text")]
        
        if not context_texts:
            return {
                "bertscore_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "best_matching_chunk": -1,
                "method": method
            }
        
        # Calculate BERTScore for each context chunk
        scores = []
        for i, context_text in enumerate(context_texts):
            if not context_text.strip():
                continue
            
            try:
                P, R, F1 = bert_score_func([predicted], [context_text], lang='en', verbose=False)
                scores.append({
                    "chunk_index": i,
                    "precision": P.item(),
                    "recall": R.item(),
                    "f1": F1.item(),
                    "context_text": context_text[:200]  # First 200 chars for reference
                })
            except Exception as e:
                logger.warning(f"Error calculating BERTScore for chunk {i}: {e}")
                continue
        
        if not scores:
            return {
                "bertscore_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "best_matching_chunk": -1,
                "method": method
            }
        
        # Combine scores based on method
        if method == "max":
            # Use the chunk with highest F1 score
            best_score = max(scores, key=lambda x: x["f1"])
            return {
                "bertscore_f1": best_score["f1"],
                "bertscore_precision": best_score["precision"],
                "bertscore_recall": best_score["recall"],
                "best_matching_chunk": best_score["chunk_index"],
                "method": method,
                "all_chunk_scores": scores  # Include all for analysis
            }
        
        elif method == "mean":
            # Average across all chunks
            avg_precision = sum(s["precision"] for s in scores) / len(scores)
            avg_recall = sum(s["recall"] for s in scores) / len(scores)
            avg_f1 = sum(s["f1"] for s in scores) / len(scores)
            best_idx = max(scores, key=lambda x: x["f1"])["chunk_index"]
            
            return {
                "bertscore_f1": avg_f1,
                "bertscore_precision": avg_precision,
                "bertscore_recall": avg_recall,
                "best_matching_chunk": best_idx,
                "method": method,
                "all_chunk_scores": scores
            }
        
        elif method == "weighted":
            # Weight by chunk score if available, otherwise use mean
            weights = []
            for i, chunk in enumerate(context_chunks):
                # Try to get relevance score from chunk
                score = chunk.get("score", chunk.get("final_score", 1.0))
                weights.append(score)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(scores)] * len(scores)
            
            # Weighted average
            weighted_precision = sum(s["precision"] * weights[s["chunk_index"]] for s in scores)
            weighted_recall = sum(s["recall"] * weights[s["chunk_index"]] for s in scores)
            weighted_f1 = sum(s["f1"] * weights[s["chunk_index"]] for s in scores)
            best_idx = max(scores, key=lambda x: x["f1"])["chunk_index"]
            
            return {
                "bertscore_f1": weighted_f1,
                "bertscore_precision": weighted_precision,
                "bertscore_recall": weighted_recall,
                "best_matching_chunk": best_idx,
                "method": method,
                "all_chunk_scores": scores
            }
        
        else:
            # Default to max
            best_score = max(scores, key=lambda x: x["f1"])
            return {
                "bertscore_f1": best_score["f1"],
                "bertscore_precision": best_score["precision"],
                "bertscore_recall": best_score["recall"],
                "best_matching_chunk": best_score["chunk_index"],
                "method": "max",
                "all_chunk_scores": scores
            }
            
    except Exception as e:
        logger.error(f"Error calculating BERTScore against context: {e}")
        return None


def f1_score_against_context(
    predicted: str,
    context_chunks: List[Dict],
    method: str = "max"
) -> Dict[str, Any]:
    """
    Calculate F1 score comparing predicted answer against retrieved context documents.
    
    Simpler alternative to BERTScore - uses token overlap instead of semantic similarity.
    Faster but less accurate for paraphrasing.
    
    Args:
        predicted: Generated answer text
        context_chunks: List of retrieved chunks with 'text' field
        method: How to combine scores ("max", "mean", "weighted")
    
    Returns:
        {
            "f1_score": float,
            "precision": float,
            "recall": float,
            "best_matching_chunk": int
        }
    """
    if not predicted or not context_chunks:
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "best_matching_chunk": -1
        }
    
    pred_tokens = set(predicted.lower().split())
    
    if not pred_tokens:
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "best_matching_chunk": -1
        }
    
    scores = []
    for i, chunk in enumerate(context_chunks):
        context_text = chunk.get("text", "")
        if not context_text:
            continue
        
        context_tokens = set(context_text.lower().split())
        if not context_tokens:
            continue
        
        intersection = pred_tokens & context_tokens
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(context_tokens) if context_tokens else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        scores.append({
            "chunk_index": i,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    if not scores:
        return {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "best_matching_chunk": -1
        }
    
    # Combine based on method
    if method == "max":
        best = max(scores, key=lambda x: x["f1"])
        return {
            "f1_score": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
            "best_matching_chunk": best["chunk_index"]
        }
    elif method == "mean":
        return {
            "f1_score": sum(s["f1"] for s in scores) / len(scores),
            "precision": sum(s["precision"] for s in scores) / len(scores),
            "recall": sum(s["recall"] for s in scores) / len(scores),
            "best_matching_chunk": max(scores, key=lambda x: x["f1"])["chunk_index"]
        }
    else:  # weighted or default
        weights = []
        for i, chunk in enumerate(context_chunks):
            score = chunk.get("score", chunk.get("final_score", 1.0))
            weights.append(score)
        total_weight = sum(weights) if weights else len(scores)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(scores)] * len(scores)
        
        return {
            "f1_score": sum(s["f1"] * weights[s["chunk_index"]] for s in scores),
            "precision": sum(s["precision"] * weights[s["chunk_index"]] for s in scores),
            "recall": sum(s["recall"] * weights[s["chunk_index"]] for s in scores),
            "best_matching_chunk": max(scores, key=lambda x: x["f1"])["chunk_index"]
        }


# ---------------------------
# 3. Procedure Completeness
# ---------------------------
# ---------------------------
# Combined Priority 2 Metrics
# ---------------------------
def calculate_priority2_metrics(
    predicted: str,
    ground_truth: Optional[str] = None,
    required_steps: Optional[List[str]] = None,
    context_chunks: Optional[List[Dict]] = None,
    context_method: str = "max"
) -> Dict[str, Any]:
    """
    Calculate all Priority 2 metrics (Answer Quality).
    
    Priority 2 Metrics:
    4. F1 Score - Token overlap
    5. BERTScore - Semantic similarity
    
    NEW: If ground_truth is not provided, uses context_chunks for evaluation.
    This allows evaluation without manual ground truth creation!
    
    Args:
        predicted: Generated answer text
        ground_truth: Expected answer text (optional, for F1 and BERTScore)
        required_steps: List of required procedure steps (optional, for completeness)
        context_chunks: List of retrieved context chunks (optional, used if ground_truth not provided)
        context_method: How to combine scores from multiple chunks ("max", "mean", "weighted")
    
    Returns:
        {
            "f1_score": float,
            "f1_details": Dict,
            "bertscore_f1": Optional[float],
            "bertscore_details": Optional[Dict],
            "overall_quality_score": float  # Weighted average
        }
    """
    results = {
        "f1_score": None,
        "f1_details": None,
        "bertscore_f1": None,
        "bertscore_details": None,
        "overall_quality_score": 0.0
    }
    
    # 1. F1 Score - Use ground_truth if available, otherwise use context_chunks
    if ground_truth:
        results["f1_score"] = f1_score(predicted, ground_truth)
        results["f1_details"] = f1_score_detailed(predicted, ground_truth)
    elif context_chunks:
        # Use context chunks for evaluation (no ground truth needed!)
        f1_context_result = f1_score_against_context(predicted, context_chunks, method=context_method)
        results["f1_score"] = f1_context_result["f1_score"]
        results["f1_details"] = {
            "f1_score": f1_context_result["f1_score"],
            "precision": f1_context_result["precision"],
            "recall": f1_context_result["recall"],
            "best_matching_chunk": f1_context_result["best_matching_chunk"],
            "method": "context_based"
        }
    
    # 2. BERTScore - Use ground_truth if available, otherwise use context_chunks
    if ground_truth:
        results["bertscore_f1"] = bertscore_accuracy(predicted, ground_truth)
        results["bertscore_details"] = bertscore_accuracy_detailed(predicted, ground_truth)
    elif context_chunks:
        # Use context chunks for evaluation (no ground truth needed!)
        bertscore_context_result = bertscore_against_context(predicted, context_chunks, method=context_method)
        if bertscore_context_result:
            results["bertscore_f1"] = bertscore_context_result["bertscore_f1"]
            results["bertscore_details"] = {
                "bertscore_f1": bertscore_context_result["bertscore_f1"],
                "bertscore_precision": bertscore_context_result["bertscore_precision"],
                "bertscore_recall": bertscore_context_result["bertscore_recall"],
                "best_matching_chunk": bertscore_context_result["best_matching_chunk"],
                "method": "context_based"
            }
    
    # Calculate overall quality score (weighted average)
    weights = {
        "f1_score": 0.5,  # 50% weight
        "bertscore_f1": 0.5,  # 50% weight
    }
    
    scores = []
    total_weight = 0.0
    
    if results["f1_score"] is not None:
        scores.append(results["f1_score"] * weights["f1_score"])
        total_weight += weights["f1_score"]
    
    if results["bertscore_f1"] is not None:
        scores.append(results["bertscore_f1"] * weights["bertscore_f1"])
        total_weight += weights["bertscore_f1"]
    
    if total_weight > 0:
        results["overall_quality_score"] = sum(scores) / total_weight
    else:
        results["overall_quality_score"] = 0.0
    
    return results


# ---------------------------
# Integration with RAG System
# ---------------------------
class AnswerQualityEvaluator:
    """
    Evaluator class for measuring answer quality in RAG system.
    Integrates Priority 2 metrics into query processing.
    """
    
    def __init__(self, enable_bertscore: bool = True):
        """
        Initialize evaluator.
        
        Args:
            enable_bertscore: Whether to enable BERTScore (requires bert_score package)
        """
        self.enable_bertscore = enable_bertscore and BERTSCORE_AVAILABLE
        if enable_bertscore and not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore requested but not available. Install with: pip install bert-score")
    
    def evaluate_answer(
        self,
        predicted_answer: str,
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        context_chunks: Optional[List[Dict]] = None,
        context_method: str = "max"
    ) -> Dict[str, Any]:
        """
        Evaluate answer quality using Priority 2 metrics.
        
        NEW: Can use context_chunks instead of ground_truth for evaluation!
        This allows evaluation without manual ground truth creation.
        
        Args:
            predicted_answer: Generated answer from RAG system
            ground_truth: Expected answer (optional, for F1 and BERTScore)
            query: Original query (for logging)
            context_chunks: Retrieved context chunks (optional, used if ground_truth not provided)
            context_method: How to combine scores from multiple chunks ("max", "mean", "weighted")
        
        Returns:
            Dictionary with all Priority 2 metrics
        """
        if query:
            logger.info(f"Evaluating answer quality for query: {query[:100]}...")
        
        # Use context_chunks if ground_truth not provided
        if not ground_truth and context_chunks:
            logger.info(f"Using context-based evaluation (no ground truth provided)")
        
        results = calculate_priority2_metrics(
            predicted=predicted_answer,
            ground_truth=ground_truth,
            context_chunks=context_chunks,
            context_method=context_method
        )
        
        # Log results
        if results["f1_score"] is not None:
            logger.info(f"F1 Score: {results['f1_score']:.3f}")
        
        if results["bertscore_f1"] is not None:
            logger.info(f"BERTScore F1: {results['bertscore_f1']:.3f}")
        
        logger.info(f"Overall Quality Score: {results['overall_quality_score']:.3f}")
        
        return results
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: Optional[List[str]] = None,
        queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple answers in batch.
        
        Args:
            predictions: List of generated answers
            ground_truths: List of expected answers (optional)
            queries: List of original queries (optional)
        
        Returns:
            Dictionary with aggregated metrics
        """
        results_list = []
        
        for i, predicted in enumerate(predictions):
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            query = queries[i] if queries and i < len(queries) else None
            
            result = self.evaluate_answer(
                predicted_answer=predicted,
                ground_truth=ground_truth,
                query=query
            )
            results_list.append(result)
        
        # Aggregate metrics
        f1_scores = [r["f1_score"] for r in results_list if r["f1_score"] is not None]
        bertscore_scores = [r["bertscore_f1"] for r in results_list if r["bertscore_f1"] is not None]
        overall_scores = [r["overall_quality_score"] for r in results_list]
        
        return {
            "individual_results": results_list,
            "aggregate_metrics": {
                "avg_f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else None,
                "avg_bertscore": sum(bertscore_scores) / len(bertscore_scores) if bertscore_scores else None,
                "avg_overall_quality": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            },
            "total_evaluated": len(results_list)
        }


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Example 1: Evaluate single answer
    evaluator = AnswerQualityEvaluator()
    
    predicted = "The damage limit for overwing panel beams is 325mm. If damage exceeds this limit, contact Airbus."
    ground_truth = "The damage limit for overwing panel beams is 325mm. Contact Airbus if exceeded."
    
    result = evaluator.evaluate_answer(
        predicted_answer=predicted,
        ground_truth=ground_truth,
        query="What is the damage limit?"
    )
    
    print("\n=== Single Answer Evaluation ===")
    print(f"F1 Score: {result['f1_score']:.3f}")
    print(f"BERTScore: {result['bertscore_f1']:.3f if result['bertscore_f1'] else 'N/A'}")
    print(f"Overall Quality: {result['overall_quality_score']:.3f}")
    
    # Example 2: Batch evaluation
    predictions = [
        "The damage limit is 325mm.",
        "First inspect the area, then measure depth, finally contact Airbus if needed."
    ]
    ground_truths = [
        "The damage limit for overwing panels is 325mm.",
        "Inspect damage area. Measure depth. Contact Airbus if depth > 0.2mm."
    ]
    
    batch_result = evaluator.evaluate_batch(
        predictions=predictions,
        ground_truths=ground_truths
    )
    
    print("\n=== Batch Evaluation ===")
    print(f"Average F1: {batch_result['aggregate_metrics']['avg_f1_score']:.3f}")
    print(f"Average BERTScore: {batch_result['aggregate_metrics']['avg_bertscore']:.3f if batch_result['aggregate_metrics']['avg_bertscore'] else 'N/A'}")
    print(f"Average Overall Quality: {batch_result['aggregate_metrics']['avg_overall_quality']:.3f}")


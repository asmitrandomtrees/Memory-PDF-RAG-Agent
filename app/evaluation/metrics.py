from dataclasses import dataclass
import math
from typing import Any, Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int | None = None,
) -> float:
    """Calculate Recall@K: fraction of relevant ground truth items retrieved in top-k."""
    if not ground_truth_ids:
        return 1.0 if not retrieved_ids else 0.0
    cutoff = len(retrieved_ids) if k is None else min(k, len(retrieved_ids))
    retrieved_set = set(retrieved_ids[:cutoff])
    ground_truth_set = set(ground_truth_ids)
    hits = len(retrieved_set.intersection(ground_truth_set))
    return hits / len(ground_truth_set)


def precision_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int | None = None,
) -> float:
    """Calculate Precision@K: fraction of retrieved top-k items that are relevant."""
    cutoff = len(retrieved_ids) if k is None else min(k, len(retrieved_ids))
    if cutoff == 0:
        return 1.0 if not ground_truth_ids else 0.0
    retrieved_set = set(retrieved_ids[:cutoff])
    ground_truth_set = set(ground_truth_ids)
    hits = len(retrieved_set.intersection(ground_truth_set))
    return hits / cutoff


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant item."""
    if not ground_truth_ids:
        return 1.0 if not retrieved_ids else 0.0
    ground_truth_set = set(ground_truth_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in ground_truth_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
    k: int | None = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance."""
    if not ground_truth_ids:
        return 1.0 if not retrieved_ids else 0.0
    cutoff = len(retrieved_ids) if k is None else min(k, len(retrieved_ids))
    ground_truth_set = set(ground_truth_ids)
    
    # DCG
    dcg = 0.0
    for i in range(cutoff):
        if retrieved_ids[i] in ground_truth_set:
            dcg += 1.0 / math.log2(i + 2)
            
    # IDCG
    ideal_hits = min(len(ground_truth_set), cutoff)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    
    return dcg / idcg if idcg > 0 else 0.0


def routing_accuracy(
    predicted_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    """Calculate Jaccard similarity between predicted and expected sources."""
    pred_set = set(predicted_sources)
    exp_set = set(expected_sources)
    if not pred_set and not exp_set:
        return 1.0
    union = pred_set.union(exp_set)
    if not union:
        return 1.0
    return len(pred_set.intersection(exp_set)) / len(union)


def citation_groundedness(
    cited_ids: Sequence[str],
    available_context_ids: Sequence[str],
) -> float:
    """Calculate citation precision: fraction of cited IDs that genuinely exist in retrieved context."""
    if not cited_ids:
        return 1.0
    context_set = set(available_context_ids)
    valid_citations = sum(1 for cid in cited_ids if cid in context_set)
    return valid_citations / len(cited_ids)


def memory_consolidation_accuracy(
    predicted_action: str,
    expected_action: str,
) -> float:
    """Check if memory consolidation action matches expected action."""
    return 1.0 if predicted_action.strip().lower() == expected_action.strip().lower() else 0.0


@dataclass
class EvaluationScores:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    routing_score: float = 0.0
    groundedness_score: float = 0.0
    consolidation_score: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "routing_score": round(self.routing_score, 4),
            "groundedness_score": round(self.groundedness_score, 4),
            "consolidation_score": round(self.consolidation_score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "retries": self.retries,
        }

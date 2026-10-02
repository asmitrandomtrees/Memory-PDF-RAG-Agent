from app.evaluation.metrics import (
    citation_groundedness,
    mean_reciprocal_rank,
    memory_consolidation_accuracy,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    routing_accuracy,
)


def test_recall_and_precision_at_k() -> None:
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
    ground_truth = ["doc_2", "doc_4", "doc_5"]

    # Top-2
    assert recall_at_k(retrieved, ground_truth, k=2) == 1 / 3
    assert precision_at_k(retrieved, ground_truth, k=2) == 1 / 2

    # Top-4
    assert recall_at_k(retrieved, ground_truth, k=4) == 2 / 3
    assert precision_at_k(retrieved, ground_truth, k=4) == 2 / 4


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank(["d1", "d2", "d3"], ["d2"]) == 0.5
    assert mean_reciprocal_rank(["d1", "d2", "d3"], ["d1"]) == 1.0
    assert mean_reciprocal_rank(["d1", "d2", "d3"], ["d9"]) == 0.0


def test_ndcg_at_k() -> None:
    retrieved = ["d1", "d2", "d3"]
    ground_truth = ["d1", "d2"]
    # Perfect ranking at top 2
    assert round(ndcg_at_k(retrieved, ground_truth, k=2), 4) == 1.0


def test_routing_accuracy() -> None:
    assert routing_accuracy(["stm", "ltm"], ["stm", "ltm"]) == 1.0
    assert routing_accuracy(["stm"], ["stm", "ltm"]) == 0.5
    assert routing_accuracy(["pdf"], ["stm"]) == 0.0


def test_citation_groundedness() -> None:
    assert citation_groundedness(["c1", "c2"], ["c1", "c2", "c3"]) == 1.0
    assert citation_groundedness(["c1", "c99"], ["c1", "c2"]) == 0.5
    assert citation_groundedness([], ["c1"]) == 1.0


def test_memory_consolidation_accuracy() -> None:
    assert memory_consolidation_accuracy("supersede", "supersede") == 1.0
    assert memory_consolidation_accuracy("update", "new") == 0.0

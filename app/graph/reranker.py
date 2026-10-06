from typing import Protocol

from app.contracts.retrieval import RetrievedItem


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        ...


class ReciprocalRankFusionReranker:
    def __init__(self, *, rank_constant: int = 60) -> None:
        if rank_constant < 1:
            raise ValueError("rank_constant must be at least one")
        self.rank_constant = rank_constant

    def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        fused: dict[tuple[str, str], tuple[float, RetrievedItem]] = {}

        by_source: dict[str, list[RetrievedItem]] = {}
        for item in items:
            by_source.setdefault(item.source, []).append(item)

        for source_items in by_source.values():
            ordered = sorted(
                enumerate(source_items, start=1),
                key=lambda pair: (
                    pair[1].rank if pair[1].rank is not None else pair[0]
                ),
            )
            for fallback_rank, item in ordered:
                source_rank = item.rank or fallback_rank
                key = (item.source, item.item_id)
                if key in fused:
                    continue

                fusion_score = 1.0 / (self.rank_constant + source_rank)

                metadata = {
                    **item.metadata,
                    "source_rank": source_rank,
                    "source_score": item.score,
                    "fusion_score": fusion_score,
                }
                fused[key] = (
                    fusion_score,
                    item.model_copy(
                        update={"score": fusion_score, "metadata": metadata}
                    ),
                )

        ranked = sorted(
            fused.values(),
            key=lambda pair: (-pair[0], pair[1].source, pair[1].item_id),
        )
        return [
            item.model_copy(update={"rank": rank})
            for rank, (_, item) in enumerate(ranked, start=1)
        ]

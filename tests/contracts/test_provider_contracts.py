from app.contracts.retrieval import RetrievedItem
from app.graph.reranker import Reranker


class FakeReranker:
    def rerank(
        self,
        query: str,
        items: list[RetrievedItem],
    ) -> list[RetrievedItem]:
        return items


def test_reranker_contract() -> None:
    reranker: Reranker = FakeReranker()

    items = [
        RetrievedItem(
            item_id="item_1",
            source="stm",
            content="test",
            score=0.9,
            rank=1,
        )
    ]

    result = reranker.rerank(
        "test query",
        items,
    )

    assert len(result) == 1
    assert result[0].item_id == "item_1"

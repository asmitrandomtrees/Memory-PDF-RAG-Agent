from collections import OrderedDict
from threading import Lock

from sentence_transformers import SentenceTransformer

from app.contracts.errors import EmbeddingError


class SentenceTransformerEmbeddings:
    def __init__(
        self,
        model_name: str,
        *,
        query_cache_size: int = 256,
    ) -> None:
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model: {model_name}"
            ) from exc

        # STM, LTM and PDF retrieval run in parallel and each embed the same
        # query. The lock makes concurrent callers wait for the first encode
        # instead of each running the model, and the cache serves the rest.
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._query_cache_size = query_cache_size
        self._query_lock = Lock()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                "Failed to create document embeddings"
            ) from exc

        vectors = embeddings.tolist()
        # A single document is usually a chat message that is about to be
        # used as a retrieval query; the model uses no query prefix, so the
        # vector can be reused.
        if len(texts) == 1:
            with self._query_lock:
                self._remember(texts[0], vectors[0])
        return vectors

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        if not text:
            return []

        with self._query_lock:
            cached = self._query_cache.get(text)
            if cached is not None:
                self._query_cache.move_to_end(text)
                return list(cached)

            try:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).tolist()
            except Exception as exc:
                raise EmbeddingError(
                    "Failed to create query embedding"
                ) from exc

            self._remember(text, embedding)
            return embedding

    def _remember(self, text: str, embedding: list[float]) -> None:
        if self._query_cache_size <= 0:
            return
        self._query_cache[text] = tuple(embedding)
        self._query_cache.move_to_end(text)
        while len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)

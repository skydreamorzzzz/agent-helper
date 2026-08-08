from __future__ import annotations

from src.planner.embedding_router import EmbeddingRouter
from src.planner.models import Route


class FakeEmbedder:
    model_name = "fake-embedder"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        vectors: list[list[float]] = []
        for text in texts:
            if text == "direct prototype" or text == "direct query":
                vectors.append([1.0, 0.0])
            elif text == "web prototype" or text == "web query":
                vectors.append([0.0, 1.0])
            elif text == "ambiguous query":
                vectors.append([0.71, 0.70])
            else:
                vectors.append([0.1, 0.1])
        return vectors


def test_embedding_router_routes_to_nearest_prototype() -> None:
    router = EmbeddingRouter(
        embedder=FakeEmbedder(),
        prototypes={
            Route.DIRECT_ANSWER: ("direct prototype",),
            Route.WEB_LOOKUP: ("web prototype",),
        },
        similarity_threshold=0.5,
        margin_threshold=0.1,
    )

    candidate = router.route("web query")

    assert candidate is not None
    assert candidate.decision.route == Route.WEB_LOOKUP
    assert candidate.source == "embedding"


def test_embedding_router_returns_uncertain_below_similarity_threshold() -> None:
    router = EmbeddingRouter(
        embedder=FakeEmbedder(),
        prototypes={
            Route.DIRECT_ANSWER: ("direct prototype",),
            Route.WEB_LOOKUP: ("web prototype",),
        },
        similarity_threshold=0.5,
        margin_threshold=0.1,
    )

    assert router.route("unknown query") is None


def test_embedding_router_returns_uncertain_when_margin_is_too_small() -> None:
    router = EmbeddingRouter(
        embedder=FakeEmbedder(),
        prototypes={
            Route.DIRECT_ANSWER: ("direct prototype",),
            Route.WEB_LOOKUP: ("web prototype",),
        },
        similarity_threshold=0.5,
        margin_threshold=0.1,
    )

    assert router.route("ambiguous query") is None


def test_embedding_router_caches_prototype_embeddings() -> None:
    embedder = FakeEmbedder()
    router = EmbeddingRouter(
        embedder=embedder,
        prototypes={
            Route.DIRECT_ANSWER: ("direct prototype",),
            Route.WEB_LOOKUP: ("web prototype",),
        },
        similarity_threshold=0.5,
        margin_threshold=0.1,
    )

    router.route("direct query")
    router.route("web query")

    assert embedder.batch_sizes == [2, 1, 1]

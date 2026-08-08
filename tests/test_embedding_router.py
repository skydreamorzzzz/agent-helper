from __future__ import annotations

import pytest

from src.planner.embedding_router import EmbeddingRouter, SentenceTransformerEmbedder
from src.planner.models import Route


class FakeEmbedder:
    provider = "fake"
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
            elif text == "long direct prototype" or text == "long direct query":
                vectors.append([2.0, 0.0])
            elif text == "long web prototype":
                vectors.append([0.0, 3.0])
            elif text == "zero query":
                vectors.append([0.0, 0.0])
            else:
                vectors.append([0.0, 0.0])
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


def test_embedding_router_cosine_handles_non_normalized_vectors() -> None:
    router = EmbeddingRouter(
        embedder=FakeEmbedder(),
        prototypes={
            Route.DIRECT_ANSWER: ("long direct prototype",),
            Route.WEB_LOOKUP: ("long web prototype",),
        },
        similarity_threshold=0.9,
        margin_threshold=0.1,
    )

    candidate = router.route("long direct query")

    assert candidate is not None
    assert candidate.decision.route == Route.DIRECT_ANSWER


def test_embedding_router_handles_zero_vector_without_crashing() -> None:
    router = EmbeddingRouter(
        embedder=FakeEmbedder(),
        prototypes={
            Route.DIRECT_ANSWER: ("long direct prototype",),
            Route.WEB_LOOKUP: ("long web prototype",),
        },
        similarity_threshold=0.1,
        margin_threshold=0.1,
    )

    assert router.route("zero query") is None


class FakeSentenceTransformerModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(self, texts: list[str]):
        self.inputs.append(texts)
        return [[float(len(text)), 1.0] for text in texts]


def test_sentence_transformer_embedder_uses_injected_model() -> None:
    model = FakeSentenceTransformerModel()
    embedder = SentenceTransformerEmbedder(model_name="fake-sentence-model", model=model)

    vectors = embedder.encode(["hello", "你好"])

    assert embedder.provider == "sentence_transformers"
    assert embedder.model_name == "fake-sentence-model"
    assert vectors == [[5.0, 1.0], [2.0, 1.0]]
    assert model.inputs == [["hello", "你好"]]


def test_sentence_transformer_embedder_reports_missing_dependency(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="sentence-transformers is required"):
        SentenceTransformerEmbedder(model_name="some-model")

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from src.planner.models import Route, RouteDecision
from src.planner.router import RouteCandidate


class Embedder(Protocol):
    provider: str
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    """Dependency-free hashed lexical vector baseline for router experiments."""

    provider = "hashing"
    model_name = "hashing-multilingual-v1"

    def __init__(self, *, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 63) else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9]+", lowered)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
        features: list[str] = []
        features.extend(f"w:{word}" for word in words)
        features.extend(f"w2:{words[index]}_{words[index + 1]}" for index in range(len(words) - 1))
        features.extend(f"zh:{char}" for char in chinese_chars)
        features.extend(
            f"zh2:{chinese_chars[index]}{chinese_chars[index + 1]}"
            for index in range(len(chinese_chars) - 1)
        )
        compact = re.sub(r"\s+", " ", lowered).strip()
        features.extend(f"c3:{compact[index:index + 3]}" for index in range(max(0, len(compact) - 2)))
        return features


class SentenceTransformerEmbedder:
    provider = "sentence_transformers"

    def __init__(self, *, model_name: str, model: Any | None = None, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for provider='sentence_transformers'. "
                    "Install the optional embedding dependencies first."
                ) from exc
            self._model = SentenceTransformer(model_name, local_files_only=local_files_only)
        else:
            self._model = model

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [[float(value) for value in vector] for vector in vectors]


DEFAULT_ROUTE_PROTOTYPES: dict[Route, tuple[str, ...]] = {
    Route.DIRECT_ANSWER: (
        "hello casual greeting chat",
        "explain a stable technical concept from general knowledge",
        "解释 一个 稳定 知识 概念",
        "explain software version numbers",
        "explain url structure and slashes",
        "research methods writing advice",
    ),
    Route.SINGLE_TOOL: (
        "calculate arithmetic expression",
        "compute a math expression",
        "读取 一个 txt 文件",
        "read one workspace file",
        "write text to one file",
        "把 内容 写入 单个 文件",
    ),
    Route.WEB_LOOKUP: (
        "latest api pricing simple lookup",
        "current ceo current president office holder",
        "current stock price share price bitcoin price",
        "最新 价格 当前 股价",
        "latest release version status",
        "现在 是 谁 当前 负责人",
    ),
    Route.PLANNED_TASK: (
        "read a file summarize it and save to another file",
        "read polish revise and save final version",
        "extract errors from log and write markdown",
        "calculate answer then write to file",
        "读取 总结 整理 保存",
        "合并 多个 文件 摘要 写入",
    ),
    Route.DEEP_RESEARCH: (
        "research compare pros and cons with multiple sources",
        "deep dive current trends cite sources",
        "compare latest model pricing and cite sources",
        "write a research report",
        "调研 比较 优缺点 多来源",
        "研究报告 竞品 差异化",
    ),
    Route.CLARIFICATION: (
        "missing file path or target",
        "save the result somewhere missing destination",
        "summarize it missing source content",
        "analyze this missing object",
        "读取 文件 但 没有 文件名",
        "分析 一下 缺少 对象",
    ),
}


@dataclass(frozen=True)
class EmbeddingRouteScore:
    route: Route
    similarity: float


class EmbeddingRouter:
    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        prototypes: dict[Route, tuple[str, ...]] | None = None,
        similarity_threshold: float = 0.32,
        margin_threshold: float = 0.04,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.prototypes = prototypes or DEFAULT_ROUTE_PROTOTYPES
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self._prototype_vectors = self._embed_prototypes()

    @property
    def provider(self) -> str:
        return self.embedder.provider

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def route(self, user_input: str) -> RouteCandidate | None:
        query_vector = self._normalize(self.embedder.encode([user_input])[0])
        scores = sorted(
            (
                EmbeddingRouteScore(route=route, similarity=self._best_similarity(query_vector, vectors))
                for route, vectors in self._prototype_vectors.items()
            ),
            key=lambda item: item.similarity,
            reverse=True,
        )
        if not scores:
            return None
        best = scores[0]
        second_similarity = scores[1].similarity if len(scores) > 1 else 0.0
        margin = best.similarity - second_similarity
        if best.similarity < self.similarity_threshold or margin < self.margin_threshold:
            return None
        return RouteCandidate(
            RouteDecision(
                route=best.route,
                reason=(
                    f"embedding route: best={best.route} similarity={best.similarity:.3f} "
                    f"second={second_similarity:.3f} margin={margin:.3f}"
                ),
            ),
            confidence=best.similarity,
            final=False,
            source="embedding",
        )

    def _embed_prototypes(self) -> dict[Route, list[list[float]]]:
        flattened: list[tuple[Route, str]] = [
            (route, text)
            for route, texts in self.prototypes.items()
            for text in texts
        ]
        vectors = [self._normalize(vector) for vector in self.embedder.encode([text for _, text in flattened])]
        by_route: dict[Route, list[list[float]]] = {route: [] for route in self.prototypes}
        for (route, _), vector in zip(flattened, vectors):
            by_route[route].append(vector)
        return by_route

    def _best_similarity(self, query_vector: list[float], prototype_vectors: list[list[float]]) -> float:
        if not prototype_vectors:
            return 0.0
        return max(self._cosine(query_vector, vector) for vector in prototype_vectors)

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        return sum(lv * rv for lv, rv in zip(left, right))

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SummarizerConfig:
    """Tuning knobs for cheap, agent-oriented extractive summarization."""

    max_sentences: int = 12
    min_sentences: int = 3
    similarity_threshold: float = 0.08
    graph_top_k: int = 8
    protected_budget_ratio: float = 0.65
    centrality_weight: float = 0.34
    query_weight: float = 0.18
    rarity_weight: float = 0.10
    anchor_weight: float = 0.25
    recency_weight: float = 0.08
    structure_weight: float = 0.05
    preserve_order: bool = True
    include_agent_state: bool = True
    max_memory_units: int = 24


@dataclass(frozen=True)
class SentenceCandidate:
    index: int
    text: str
    final_score: float
    centrality_score: float
    query_score: float
    rarity_score: float
    anchor_score: float
    recency_score: float
    structure_score: float
    labels: tuple[str, ...] = field(default_factory=tuple)
    protected: bool = False


@dataclass(frozen=True)
class MemoryUnit:
    type: str
    key: str
    value: str
    source: str = ""

    @property
    def text(self) -> str:
        return f"{self.type}.{self.key}: {self.value}"


@dataclass(frozen=True)
class SummaryResult:
    text: str
    sentences: tuple[SentenceCandidate, ...]
    all_candidates: tuple[SentenceCandidate, ...]
    compression_ratio: float
    memory_units: tuple[MemoryUnit, ...] = field(default_factory=tuple)

    @property
    def labels(self) -> tuple[str, ...]:
        seen: set[str] = set()
        labels: list[str] = []
        for candidate in self.sentences:
            for label in candidate.labels:
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        return tuple(labels)

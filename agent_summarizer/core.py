from __future__ import annotations

from collections.abc import Callable, Sequence

from .features import cosine_counts, detect_labels, normalize, structure_score, token_counter
from .models import MemoryUnit, SentenceCandidate, SummaryResult, SummarizerConfig
from .splitting import split_sentences
from .state import extract_agent_state, render_agent_state

Vector = Sequence[float]
Embedder = Callable[[Sequence[str]], Sequence[Vector]]


class AgentContextSummarizer:
    """Cheap extractive summarizer tuned for agent memory and tool traces."""

    def __init__(self, config: SummarizerConfig | None = None, embedder: Embedder | None = None):
        self.config = config or SummarizerConfig()
        self.embedder = embedder

    def summarize(
        self,
        text: str,
        query: str = "",
        max_sentences: int | None = None,
        min_sentences: int | None = None,
        max_output_chars: int | None = None,
    ) -> SummaryResult:
        output_char_budget = max_output_chars if max_output_chars is not None else self.config.max_output_chars
        sentences = split_sentences(text)
        if not sentences:
            return SummaryResult("", (), (), 0.0, output_char_budget=output_char_budget)

        memory_units = (
            extract_agent_state(text, query, self.config.max_memory_units)
            if self.config.include_agent_state
            else ()
        )
        limit = self._resolve_limit(len(sentences), max_sentences, min_sentences)
        similarity_matrix = self._similarity_matrix(sentences)
        centrality = _pagerank(similarity_matrix)
        query_scores = self._query_scores(sentences, query)
        rarity_scores = _rarity_scores(similarity_matrix)

        labels_and_scores = [detect_labels(sentence) for sentence in sentences]
        anchor_scores = [item[1] for item in labels_and_scores]
        recency_scores = [index / max(1, len(sentences) - 1) for index in range(len(sentences))]
        structure_scores = [structure_score(sentence) for sentence in sentences]

        centrality_norm = normalize(centrality)
        query_norm = normalize(query_scores)
        rarity_norm = normalize(rarity_scores)

        candidates: list[SentenceCandidate] = []
        for index, sentence in enumerate(sentences):
            labels, anchor_score, protected = labels_and_scores[index]
            final_score = (
                self.config.centrality_weight * centrality_norm[index]
                + self.config.query_weight * query_norm[index]
                + self.config.rarity_weight * rarity_norm[index]
                + self.config.anchor_weight * anchor_score
                + self.config.recency_weight * recency_scores[index]
                + self.config.structure_weight * structure_scores[index]
            )
            candidates.append(
                SentenceCandidate(
                    index=index,
                    text=sentence,
                    final_score=final_score,
                    centrality_score=centrality_norm[index],
                    query_score=query_norm[index],
                    rarity_score=rarity_norm[index],
                    anchor_score=anchor_score,
                    recency_score=recency_scores[index],
                    structure_score=structure_scores[index],
                    labels=labels,
                    protected=protected,
                )
            )

        selected = self._select(candidates, limit)
        selected = tuple(sorted(selected, key=lambda item: item.index if self.config.preserve_order else -item.final_score))
        summary, selected, memory_units = _fit_output_budget(
            memory_units,
            selected,
            output_char_budget,
            self.config.preserve_order,
        )
        compression_ratio = len(summary) / len(text) if text else 0.0
        return SummaryResult(
            summary,
            selected,
            tuple(candidates),
            compression_ratio,
            memory_units,
            output_char_budget,
        )

    def _resolve_limit(self, sentence_count: int, max_sentences: int | None, min_sentences: int | None) -> int:
        max_limit = max_sentences if max_sentences is not None else self.config.max_sentences
        # min_sentences is a preference only. The maximum is always a hard
        # upper bound, including when callers explicitly pass a value below it.
        _ = min_sentences if min_sentences is not None else self.config.min_sentences
        return min(sentence_count, max(0, max_limit))

    def _select(self, candidates: list[SentenceCandidate], limit: int) -> list[SentenceCandidate]:
        if limit <= 0:
            return []

        protected_limit = min(limit, max(0, int(limit * self.config.protected_budget_ratio)))
        protected = sorted(
            (candidate for candidate in candidates if candidate.protected),
            key=lambda item: (item.anchor_score, item.final_score),
            reverse=True,
        )

        selected_by_index: dict[int, SentenceCandidate] = {}
        query_matches = sorted(
            (candidate for candidate in candidates if candidate.query_score >= 0.82),
            key=lambda item: (item.query_score, item.anchor_score, item.final_score),
            reverse=True,
        )[:min(4, limit)]
        for candidate in query_matches:
            if len(selected_by_index) >= limit:
                break
            selected_by_index[candidate.index] = candidate

        priority_labels = ("task", "test", "constraint", "policy", "error", "decision", "user_preference", "security", "todo")
        if protected_limit:
            for label in priority_labels:
                if len(selected_by_index) >= limit:
                    break
                matches = [candidate for candidate in protected if label in candidate.labels]
                if label == "constraint":
                    matches = [candidate for candidate in matches if "security" not in candidate.labels] or matches
                if matches:
                    best = max(matches, key=lambda item: (item.query_score, item.anchor_score, item.final_score))
                    selected_by_index[best.index] = best
                if len(selected_by_index) >= protected_limit or len(selected_by_index) >= limit:
                    break

            for candidate in protected:
                if len(selected_by_index) >= protected_limit or len(selected_by_index) >= limit:
                    break
                selected_by_index.setdefault(candidate.index, candidate)

        remaining_slots = limit - len(selected_by_index)
        if remaining_slots > 0:
            for candidate in sorted(candidates, key=lambda item: item.final_score, reverse=True):
                if candidate.index not in selected_by_index:
                    if _is_redundant(candidate, selected_by_index.values()):
                        continue
                    selected_by_index[candidate.index] = candidate
                    remaining_slots -= 1
                if remaining_slots == 0:
                    break

        return list(selected_by_index.values())[:limit]

    def _similarity_matrix(self, sentences: list[str]) -> list[list[float]]:
        if self.embedder:
            vectors = self.embedder(sentences)
            matrix = [
                [_cosine_vectors(left, right) for right in vectors]
                for left in vectors
            ]
        else:
            counters = [token_counter(sentence) for sentence in sentences]
            matrix = [
                [cosine_counts(left, right) for right in counters]
                for left in counters
            ]

        for row_index, row in enumerate(matrix):
            row[row_index] = 0.0
            for column_index, value in enumerate(row):
                row[column_index] = max(0.0, value)

        return _prune_matrix(matrix, self.config.similarity_threshold, self.config.graph_top_k)

    def _query_scores(self, sentences: list[str], query: str) -> list[float]:
        if not query:
            return [0.0 for _ in sentences]

        if self.embedder:
            vectors = self.embedder([query, *sentences])
            query_vector = vectors[0]
            return [_cosine_vectors(query_vector, vector) for vector in vectors[1:]]

        query_counter = token_counter(query)
        query_anchor_tokens = {
            token
            for token in query_counter
            if len(token) >= 6 and any(char.isdigit() or char in "_:/#-" for char in token)
        }
        scores: list[float] = []
        for sentence in sentences:
            sentence_counter = token_counter(sentence)
            score = cosine_counts(query_counter, sentence_counter)
            if query_anchor_tokens:
                overlap = len(query_anchor_tokens & set(sentence_counter)) / len(query_anchor_tokens)
                score += 0.8 * overlap
            scores.append(score)
        return scores


def summarize_agent_context(text: str, query: str = "", max_sentences: int = 12) -> str:
    return AgentContextSummarizer(SummarizerConfig(max_sentences=max_sentences)).summarize(text, query).text


def summarize_text(text: str, title: str = "", num_sentences: int | None = None) -> str:
    max_sentences = num_sentences if num_sentences is not None else SummarizerConfig().max_sentences
    return summarize_agent_context(text, title, max_sentences=max_sentences)


def _pagerank(matrix: list[list[float]], damping: float = 0.85, iterations: int = 50) -> list[float]:
    n = len(matrix)
    if n == 0:
        return []

    ranks = [1.0 / n for _ in range(n)]
    outbound = [sum(row) for row in matrix]

    for _ in range(iterations):
        next_ranks = [(1.0 - damping) / n for _ in range(n)]
        for source, row in enumerate(matrix):
            if outbound[source] == 0:
                share = damping * ranks[source] / n
                for target in range(n):
                    next_ranks[target] += share
                continue
            for target, weight in enumerate(row):
                if weight:
                    next_ranks[target] += damping * ranks[source] * weight / outbound[source]
        ranks = next_ranks
    return ranks


def _rarity_scores(matrix: list[list[float]]) -> list[float]:
    scores: list[float] = []
    for row in matrix:
        positives = sorted((value for value in row if value > 0), reverse=True)[:5]
        average_similarity = sum(positives) / len(positives) if positives else 0.0
        scores.append(1.0 - average_similarity)
    return scores


def _prune_matrix(matrix: list[list[float]], threshold: float, top_k: int) -> list[list[float]]:
    pruned: list[list[float]] = []
    for row in matrix:
        keep = set(
            index
            for index, value in sorted(enumerate(row), key=lambda item: item[1], reverse=True)[:top_k]
            if value >= threshold
        )
        pruned.append([value if index in keep else 0.0 for index, value in enumerate(row)])
    return pruned


def _cosine_vectors(left: Vector, right: Vector) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _join_sentences(sentences: list[str]) -> str:
    if not sentences:
        return ""
    result: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        result.append(sentence)
    return "\n".join(result)


def _fit_output_budget(
    memory_units: tuple[MemoryUnit, ...],
    selected: tuple[SentenceCandidate, ...],
    max_chars: int | None,
    preserve_order: bool,
) -> tuple[str, tuple[SentenceCandidate, ...], tuple[MemoryUnit, ...]]:
    """Render state and sentences inside one shared character budget.

    Structured state is part of the same budget as extractive sentences. When
    the complete result is too large, low-priority state units and low-scoring
    sentences are removed until the result fits. A single oversized component
    is clipped as a final safeguard so the bound is unconditional.
    """

    kept_units = list(memory_units)
    kept_sentences = list(selected)
    if max_chars is not None and max_chars <= 0:
        return "", (), ()

    def ordered_sentences() -> tuple[SentenceCandidate, ...]:
        key = (lambda item: item.index) if preserve_order else (lambda item: -item.final_score)
        return tuple(sorted(kept_sentences, key=key))

    def render() -> str:
        parts = [
            part
            for part in (
                render_agent_state(tuple(kept_units)),
                _join_sentences([candidate.text for candidate in ordered_sentences()]),
            )
            if part
        ]
        return "\n\n".join(parts)

    summary = render()
    if max_chars is None:
        return summary, ordered_sentences(), tuple(kept_units)

    while len(summary) > max_chars and len(kept_units) + len(kept_sentences) > 1:
        state_length = len(render_agent_state(tuple(kept_units)))
        sentence_length = len(_join_sentences([candidate.text for candidate in ordered_sentences()]))
        if kept_units and (not kept_sentences or state_length >= sentence_length):
            kept_units.pop()
        elif kept_sentences:
            lowest = min(kept_sentences, key=lambda item: item.final_score)
            kept_sentences.remove(lowest)
        summary = render()

    if len(summary) > max_chars:
        summary = _clip_to_budget(summary, max_chars)
    return summary, ordered_sentences(), tuple(kept_units)


def _clip_to_budget(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[: max_chars - 1].rstrip() + "…"


def _is_redundant(candidate: SentenceCandidate, selected: Sequence[SentenceCandidate]) -> bool:
    candidate_tokens = set(candidate.text.lower().split())
    if len(candidate_tokens) < 4:
        return False
    for existing in selected:
        existing_tokens = set(existing.text.lower().split())
        overlap = len(candidate_tokens & existing_tokens) / max(1, min(len(candidate_tokens), len(existing_tokens)))
        if overlap >= 0.82:
            return True
    return False

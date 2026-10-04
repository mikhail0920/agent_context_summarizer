from __future__ import annotations

import difflib
import re
from typing import Union

import numpy as np

from agent_summarizer import AgentContextSummarizer, SummarizerConfig, summarize_text

_DEFAULT_SUMMARIZER = AgentContextSummarizer()


def find_most_similar(text: str, pattern_or_flags: dict[str, bool] | str, coeff: float = 0.6):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    def best_match(pattern: str) -> str | None:
        best_sentence = None
        best_ratio = 0.0
        for sentence in sentences:
            ratio = difflib.SequenceMatcher(None, sentence.lower(), pattern.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_sentence = sentence
        return best_sentence if best_sentence and best_ratio > coeff else None

    if isinstance(pattern_or_flags, str):
        return best_match(pattern_or_flags)

    highlighted = text
    for pattern, is_positive in pattern_or_flags.items():
        match = best_match(pattern)
        if not match:
            continue
        prefix = "\033[42m" if is_positive else "\033[41m"
        highlighted = highlighted.replace(match, f"{prefix}{match}\033[0m")
    return highlighted


def summarize(text: str, title: str = "", num_sentences: int | None = None) -> str:
    return summarize_text(text, title, num_sentences)


def summarize_with_graph(
    text: str,
    title: str = "",
    num_sentences: int | None = None,
    show_graph: bool = True,
) -> str:
    if show_graph:
        print("Graph rendering was removed from utils.py; use AgentContextSummarizer().summarize(...) for scored candidates.")
    return summarize(text, title, num_sentences)


def summarize_agent_context(text: str, query: str = "", max_sentences: int = 12) -> str:
    return AgentContextSummarizer(SummarizerConfig(max_sentences=max_sentences)).summarize(text, query).text


def count_excellent(text: str) -> int:
    excellent_words = ("excellent", "great", "важно", "отлично", "critical", "must")
    lowered = text.lower()
    return sum(lowered.count(word) for word in excellent_words)


def get_vec(text: str) -> np.ndarray:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text.lower())
    buckets = np.zeros(128, dtype=float)
    for token in tokens:
        buckets[hash(token) % len(buckets)] += 1.0
    norm = np.linalg.norm(buckets)
    return buckets / norm if norm else buckets


def sim_two_sentences(
    sent1: Union[str, np.ndarray],
    sent2: Union[str, np.ndarray],
    return_vec: bool = False,
) -> Union[float, tuple[float, np.ndarray, np.ndarray]]:
    vec1 = get_vec(sent1) if isinstance(sent1, str) else sent1
    vec2 = get_vec(sent2) if isinstance(sent2, str) else sent2
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    score = float(np.dot(vec1, vec2) / denom) if denom else 0.0
    if return_vec:
        return score, vec1, vec2
    return score


if __name__ == "__main__":
    demo = """
    User: Нужно сжать длинный агентский контекст.
    Tool: pytest tests/test_memory.py failed with AssertionError in tests/test_memory.py.
    Decision: используем дешевый extractive compressor и сохраняем protected facts.
    Constraint: нельзя удалять пользовательские изменения без разрешения.
    Assistant: Я добавил паттерны риска и ранжирование.
    """
    print(_DEFAULT_SUMMARIZER.summarize(demo, "agent context compression").text)

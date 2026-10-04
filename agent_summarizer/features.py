from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_./\\:-]+")

RISK_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("task", r"\b(user query|user task|task description|user goal|issue title|instance_id|query_id|expected_next_action|id:)\b", 0.8),
    ("test", r"\b(FAIL_TO_PASS|PASS_TO_PASS|pytest|test_[A-Za-z0-9_:-]+|::test|failed test|failing test)\b", 0.82),
    ("policy", r"\b(policy|rule|allowed only|not allowed|eligible|eligibility|fare difference|refund policy|rebook|политика|правило)\b", 0.88),
    ("constraint", r"\b(must not|do not|don't|never|forbidden|without approval|нельзя|запрещено|не трогать|без разрешения|строго)\b", 1.0),
    ("requirement", r"\b(must|required|requirement|обязательно|требование|нужно|надо|должен|должна)\b", 0.72),
    ("decision", r"\b(decision|decided|agreed|chosen|use sqlite|use postgres|решили|выбрали|договорились|оставляем|используем)\b", 0.88),
    ("error", r"\b(traceback|exception|error|failed|failure|crash|timeout|ошибка|падает|упал|исключение|не проходит)\b", 0.95),
    ("todo", r"\b(todo|fixme|next step|open issue|осталось|сделать|доделать|следующий шаг)\b", 0.7),
    ("user_preference", r"\b(user asked|user requested|user prefers|пользователь просил|пользователь попросил|предпочитает|просит)\b", 0.86),
    ("security", r"\b(secret|token|api key|api_key|credential|password|pii|[A-Z0-9_]*API_KEY|секрет|токен|пароль|ключ)\b", 0.9),
)

ANCHOR_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("file", r"([A-Za-z]:\\|/[\w.-]+/|[\w./\\-]+\.(py|js|ts|tsx|json|md|yaml|yml|toml|sql|env))", 0.86),
    ("command", r"(`[^`]+`|\b(pytest|python|pip|npm|pnpm|git|curl|docker|kubectl)\s+[-\w./\\:=]+)", 0.74),
    ("symbol", r"\b[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)|\b[A-Z_]{3,}\b", 0.55),
    ("number", r"\b(v?\d+(\.\d+){1,3}|\d{4}-\d{2}-\d{2}|\d+%|\d+ms|\d+s|\d+)\b", 0.3),
    ("quote", r"['\"`][^'\"`]{3,}['\"`]", 0.45),
)

STOPWORDS = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "was",
    "were",
    "и",
    "в",
    "во",
    "на",
    "не",
    "что",
    "это",
    "как",
    "для",
    "с",
    "по",
    "из",
    "к",
    "а",
    "но",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def token_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_counts(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def detect_labels(text: str) -> tuple[tuple[str, ...], float, bool]:
    labels: list[str] = []
    score = 0.0

    for label, pattern, weight in RISK_PATTERNS:
        if re.search(pattern, text, re.I):
            labels.append(label)
            score += weight

    for label, pattern, weight in ANCHOR_PATTERNS:
        flags = 0 if label == "symbol" else re.I
        if re.search(pattern, text, flags):
            labels.append(label)
            score += weight

    normalized = min(1.0, score / 2.8)
    protected = any(label in labels for label in ("task", "test", "policy", "constraint", "decision", "error", "user_preference", "security"))
    protected = protected or ("file" in labels and ("command" in labels or "error" in labels))
    return tuple(dict.fromkeys(labels)), normalized, protected


def structure_score(text: str) -> float:
    score = 0.0
    if re.match(r"^(\[[^\]]+\]|(user|assistant|system|tool|error|warning)\s*:)", text, re.I):
        score += 0.45
    if re.match(r"^[-*]\s+", text):
        score += 0.2
    if ":" in text[:40]:
        score += 0.15
    if re.search(r"\b(step|phase|milestone|этап|шаг)\b", text, re.I):
        score += 0.2
    return min(1.0, score)

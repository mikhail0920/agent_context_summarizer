from __future__ import annotations

import json
import re

from .features import tokenize
from .models import MemoryUnit


ID_PATTERNS: tuple[tuple[str, str], ...] = (
    ("instance_id", r"\b[A-Za-z]+__[A-Za-z0-9_.-]+-\d+\b"),
    ("test", r"[\w./-]+\.py::[\w\[\].:-]+"),
    ("order_id", r"#\w{6,12}\b"),
    ("reservation_id", r"\b(?=[A-Z0-9]{6,10}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]+\b"),
    ("user_id", r"\b[a-z]+_[a-z]+_\d{4}\b"),
    ("item_id", r"\b\d{10}\b"),
    ("payment_id", r"\b(?:credit_card|gift_card|paypal)_\d+\b"),
)


def extract_agent_state(text: str, query: str = "", max_units: int = 24) -> tuple[MemoryUnit, ...]:
    units: list[MemoryUnit] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        _extract_metadata_line(line, units)
        _extract_role_line(line, units)
        _extract_tool_call_line(line, units)
        _extract_expected_next_action(line, units)
        _extract_relevant_apis(line, units)
        _extract_tests(line, units)

    # The query may guide selection of excerpts, but it is never treated as a
    # source of facts. Every emitted value must occur in the source context.
    _extract_ids(text, units)
    _extract_query_anchors(text, query, units)

    deduped = _dedupe(units)
    return tuple(deduped[:max_units])


def render_agent_state(units: tuple[MemoryUnit, ...]) -> str:
    if not units:
        return ""
    groups: dict[str, list[MemoryUnit]] = {}
    for unit in units:
        groups.setdefault(unit.type, []).append(unit)

    lines = ["Agent State:"]
    for unit_type in ("task", "entity", "test", "tool_call", "api", "constraint", "excerpt"):
        items = groups.get(unit_type, [])
        if not items:
            continue
        lines.append(f"{unit_type}:")
        for item in items:
            lines.append(f"- {item.key}: {item.value}")
    return "\n".join(lines)


def _extract_metadata_line(line: str, units: list[MemoryUnit]) -> None:
    for key in ("Dataset", "repo", "instance_id", "base_commit", "query_id", "id", "expected_next_action"):
        prefix = f"{key}:"
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            unit_type = "task" if key in {"Dataset", "query_id", "id", "expected_next_action"} else "entity"
            units.append(MemoryUnit(unit_type, key.lower(), _clip(value)))

    if line.startswith("user query:"):
        units.append(MemoryUnit("task", "user_query", _clip(line.split(":", 1)[1].strip(), 220)))


def _extract_role_line(line: str, units: list[MemoryUnit]) -> None:
    if line.startswith("user:"):
        value = line.split(":", 1)[1].strip()
        if value:
            units.append(MemoryUnit("task", "user_goal", _clip(value, 220)))
    if line.startswith("system:") and "Task description:" in line:
        value = line.split("Task description:", 1)[1].strip()
        units.append(MemoryUnit("task", "task_description", _clip(value, 220)))
    if line.startswith("assistant:") and re.search(r"\b(Decision|Action|Thought)\b", line):
        units.append(MemoryUnit("excerpt", "assistant_step", _clip(line, 220)))


def _extract_tool_call_line(line: str, units: list[MemoryUnit]) -> None:
    if "tool_call:" not in line:
        return
    call = line.split("tool_call:", 1)[1].strip()
    name = call.split("(", 1)[0].strip()
    units.append(MemoryUnit("tool_call", name or "call", _clip(call, 240)))
    try:
        args_text = call.split("(", 1)[1].rsplit(")", 1)[0]
        args = json.loads(args_text)
        for key, value in args.items():
            units.append(MemoryUnit("entity", key, _clip(str(value))))
    except Exception:
        return


def _extract_expected_next_action(line: str, units: list[MemoryUnit]) -> None:
    if not line.startswith("expected_next_action:"):
        return
    value = line.split(":", 1)[1].strip()
    units.append(MemoryUnit("task", "expected_next_action", _clip(value, 120)))
    for name in re.findall(r"'name': '([^']+)'|\"name\": \"([^\"]+)\"", value):
        tool_name = name[0] or name[1]
        if tool_name:
            units.append(MemoryUnit("tool_call", tool_name, tool_name))


def _extract_relevant_apis(line: str, units: list[MemoryUnit]) -> None:
    if "relevant_apis:" not in line:
        return
    value = line.split("relevant_apis:", 1)[1].strip()
    units.append(MemoryUnit("api", "relevant_apis", _clip(value, 260)))
    for api in re.findall(r"'([^']+)'|\"([^\"]+)\"", value):
        token = api[0] or api[1]
        if len(token) > 2:
            units.append(MemoryUnit("api", "name", token))


def _extract_tests(line: str, units: list[MemoryUnit]) -> None:
    if "FAIL_TO_PASS" in line or "::test" in line or line.startswith("- ") and ".py::" in line:
        for match in re.findall(r"[\w./-]+\.py::[\w\[\].:-]+", line):
            units.append(MemoryUnit("test", "failing_or_relevant", match))


def _extract_ids(text: str, units: list[MemoryUnit]) -> None:
    for key, pattern in ID_PATTERNS:
        for match in re.findall(pattern, text):
            units.append(MemoryUnit("entity" if key != "test" else "test", key, match))


def _extract_query_anchors(text: str, query: str, units: list[MemoryUnit]) -> None:
    if not query:
        return
    query_tokens = set(tokenize(query))
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        line_tokens = set(tokenize(stripped))
        if not line_tokens:
            continue
        overlap = len(query_tokens & line_tokens) / max(1, len(query_tokens))
        if overlap >= 0.35 and len(stripped) <= 500:
            units.append(MemoryUnit("excerpt", "query_match", _clip(stripped, 240)))


def _dedupe(units: list[MemoryUnit]) -> list[MemoryUnit]:
    seen: set[tuple[str, str, str]] = set()
    key_counts: dict[tuple[str, str], int] = {}
    result: list[tuple[int, MemoryUnit]] = []
    for index, unit in enumerate(units):
        identity = (unit.type, unit.key, unit.value)
        if identity in seen:
            continue
        key_identity = (unit.type, unit.key)
        if unit.key in {"user_goal", "task_description", "assistant_step"} and key_counts.get(key_identity, 0) >= 1:
            continue
        key_counts[key_identity] = key_counts.get(key_identity, 0) + 1
        seen.add(identity)
        result.append((index, unit))
    result.sort(key=lambda item: (_unit_priority(item[1]), item[0]))
    return [unit for _, unit in result]


def _clip(value: str, limit: int = 160) -> str:
    value = " ".join(str(value).split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _unit_priority(unit: MemoryUnit) -> int:
    if unit.key in {"user_goal", "user_query"}:
        return 0
    if unit.key in {"query_id", "id", "instance_id", "repo"}:
        return 1
    if unit.key == "expected_next_action":
        return 2
    if unit.type == "tool_call":
        return 3
    if unit.type == "test":
        return 4
    if unit.key in {"order_id", "reservation_id", "item_id", "user_id"}:
        return 5
    if unit.type == "api":
        return 6
    if unit.type == "entity":
        return 7
    if unit.type == "task":
        return 8
    return 9

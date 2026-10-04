from __future__ import annotations

import re

_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "e.g.",
    "i.e.",
    "т.е.",
    "т.к.",
    "т.д.",
    "т.п.",
}


def split_sentences(text: str, max_chars: int = 260) -> list[str]:
    """Split mixed prose, chat logs, bullets, and tool output into useful units."""

    if not text or not text.strip():
        return []

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue

        if _looks_like_log_block(lines):
            units.extend(_split_log_lines(lines, max_chars))
            continue

        paragraph_text = " ".join(lines)
        units.extend(_split_prose(paragraph_text, max_chars))

    return [unit for unit in (_clean_unit(unit) for unit in units) if len(unit) > 2]


def _looks_like_log_block(lines: list[str]) -> bool:
    if len(lines) == 1:
        return _is_structured_line(lines[0])
    structured = sum(1 for line in lines if _is_structured_line(line))
    return structured >= max(2, len(lines) // 2)


def _is_structured_line(line: str) -> bool:
    return bool(
        re.match(r"^(\[[^\]]+\]|(user|assistant|system|tool|error|warning)\s*:)", line, re.I)
        or re.match(r"^[-*]\s+", line)
        or re.search(r"([A-Za-z]:\\|/[\w.-]+/|[\w./\\-]+\.(py|js|ts|tsx|json|md|yaml|yml|toml))", line)
        or re.search(r"\b(traceback|exception|error|failed|pytest|npm|git|curl|python)\b", line, re.I)
    )


def _split_log_lines(lines: list[str], max_chars: int) -> list[str]:
    units: list[str] = []
    buffer: list[str] = []

    for line in lines:
        if _is_structured_line(line):
            if buffer:
                units.extend(_split_prose(" ".join(buffer), max_chars))
                buffer = []
            units.extend(_split_long_unit(line, max_chars))
        else:
            buffer.append(line)

    if buffer:
        units.extend(_split_prose(" ".join(buffer), max_chars))
    return units


def _split_prose(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9\"'`(\[])|(?<=[.!?])\s+$", text):
        end = match.start()
        piece = text[start:end].strip()
        if piece and not _ends_with_abbreviation(piece):
            pieces.extend(_split_long_unit(piece, max_chars))
            start = match.end()

    tail = text[start:].strip()
    if tail:
        pieces.extend(_split_long_unit(tail, max_chars))
    return pieces


def _split_long_unit(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for part in re.split(r"([,;]\s+|\s+-\s+)", text):
        if not part:
            continue
        if size + len(part) > max_chars and current:
            chunks.append("".join(current).strip(" ,;-"))
            current = [part]
            size = len(part)
        else:
            current.append(part)
            size += len(part)
    if current:
        chunks.append("".join(current).strip(" ,;-"))
    return chunks


def _ends_with_abbreviation(text: str) -> bool:
    tail = text.lower().split()[-1] if text.split() else ""
    return tail in _ABBREVIATIONS


def _clean_unit(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text.strip()

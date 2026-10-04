from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from huggingface_hub import hf_hub_download

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_summarizer import AgentContextSummarizer, SummarizerConfig
from agent_summarizer.evaluation import BenchmarkScenario, OracleFact, evaluate_scenarios, format_report

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "hf_rows"


def fetch_live_scenarios() -> list[BenchmarkScenario]:
    scenarios: list[BenchmarkScenario] = []
    scenarios.extend(_swe_bench_scenario(offset) for offset in range(5))
    scenarios.extend(_toolbench_conversation_scenario(offset) for offset in range(5))
    scenarios.extend(_toolbench_instruction_scenario(offset) for offset in range(5))
    scenarios.extend(_tau_airline_trace_scenario(offset) for offset in range(5))
    scenarios.extend(_tau_retail_trace_scenario(offset) for offset in range(5))
    return scenarios


def run() -> str:
    scenarios = fetch_live_scenarios()
    agent = AgentContextSummarizer(
        SummarizerConfig(max_sentences=8, min_sentences=1, max_memory_units=8)
    )
    baseline = AgentContextSummarizer(
        SummarizerConfig(
            max_sentences=8,
            min_sentences=6,
            include_agent_state=False,
            protected_budget_ratio=0.0,
            centrality_weight=1.0,
            query_weight=0.08,
            rarity_weight=0.0,
            anchor_weight=0.0,
            recency_weight=0.0,
            structure_weight=0.0,
        )
    )
    agent_report = evaluate_scenarios(scenarios, agent)
    baseline_report = evaluate_scenarios(scenarios, baseline)
    return "\n\n".join(
        [
            "LIVE HUGGING FACE AGENT PROFILE",
            format_report(agent_report),
            "LIVE HUGGING FACE CENTRALITY BASELINE",
            format_report(baseline_report),
        ]
    )


def _swe_bench_scenario(offset: int) -> BenchmarkScenario:
    row = _fetch_row("SWE-bench/SWE-bench_Lite", "default", "test", offset)
    fail_to_pass = _parse_jsonish_list(row.get("FAIL_TO_PASS", []))
    pass_to_pass = _parse_jsonish_list(row.get("PASS_TO_PASS", []))
    problem_statement = _normalize(row["problem_statement"])
    context = "\n".join(
        [
            "Dataset: SWE-bench/SWE-bench_Lite",
            f"repo: {row['repo']}",
            f"instance_id: {row['instance_id']}",
            f"base_commit: {row['base_commit']}",
            "problem_statement:",
            problem_statement,
            f"hints_text: {_normalize(row.get('hints_text') or '')}",
            "FAIL_TO_PASS tests:",
            *[f"- {test}" for test in fail_to_pass],
            "PASS_TO_PASS tests:",
            *[f"- {test}" for test in pass_to_pass[:6]],
            "test_patch excerpt:",
            _normalize(row.get("test_patch", ""))[:2500],
            "other real SWE-bench rows as long-context noise:",
            *_swe_noise_rows(exclude_offset=offset),
        ]
    )
    oracle = [
        OracleFact("repo", row["repo"]),
        OracleFact("instance_id", row["instance_id"]),
    ]
    if fail_to_pass:
        oracle.append(OracleFact("fail_to_pass", fail_to_pass[0]))
    identifier = _first_code_identifier(problem_statement)
    if identifier:
        oracle.append(OracleFact("problem_identifier", identifier))

    query_parts = [
        f"compress SWE-bench issue {row['instance_id']}",
        f"repo {row['repo']}",
        "keep failing tests and issue identifiers",
    ]
    if fail_to_pass:
        query_parts.append(fail_to_pass[0])

    return BenchmarkScenario(
        name=f"live SWE-bench Lite row {offset}: {row['instance_id']}",
        source="https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite",
        query=" ".join(query_parts),
        context=context,
        oracle_facts=tuple(oracle),
        max_sentences=1,
        max_compression_ratio=0.24,
    )


def _toolbench_conversation_scenario(offset: int) -> BenchmarkScenario:
    row = _fetch_row("tuandunghcmut/toolbench-v1", "default", "validation", offset)
    messages = _conversation_messages(row["conversations"])
    context = "\n".join(
        [
            "Dataset: tuandunghcmut/toolbench-v1 default validation",
            f"id: {row['id']}",
            *_format_messages(messages),
            "other real ToolBench conversations as long-context noise:",
            *_toolbench_conversation_noise(exclude_offset=offset),
        ]
    )
    tool_names = _tool_call_names(messages)
    first_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
    oracle = [OracleFact("task_id", row["id"][:90])]
    if first_user:
        oracle.append(OracleFact("user_query", first_user[:90]))
    for index, name in enumerate(tool_names[:3], start=1):
        oracle.append(OracleFact(f"tool_call_{index}", name))

    return BenchmarkScenario(
        name=f"live ToolBench conversation row {offset}",
        source="https://huggingface.co/datasets/tuandunghcmut/toolbench-v1",
        query=f"compress ToolBench conversation preserving task, API calls, and parameters: {row['id']} {first_user[:160]}",
        context=context,
        oracle_facts=tuple(oracle),
        max_sentences=1,
        max_compression_ratio=0.22,
    )


def _toolbench_instruction_scenario(offset: int) -> BenchmarkScenario:
    row = _fetch_row("tuandunghcmut/toolbench-v1", "benchmark", "g2_instruction", offset)
    api_list = json.loads(row["api_list"])
    relevant = json.loads(row["relevant_apis"]) if isinstance(row["relevant_apis"], str) else row["relevant_apis"]
    context = "\n".join(
        [
            "Dataset: tuandunghcmut/toolbench-v1 benchmark g2_instruction",
            f"query_id: {row['query_id']}",
            f"user query: {row['query']}",
            "available APIs:",
            *[
                f"- tool={api.get('tool_name')} api={api.get('api_name')} required={api.get('required_parameters')}"
                for api in api_list
            ],
            f"relevant_apis: {relevant}",
            "other real ToolBench instructions as long-context noise:",
            *_toolbench_instruction_noise(exclude_offset=offset),
        ]
    )
    oracle = [
        OracleFact("query_id", row["query_id"]),
        OracleFact("query", row["query"][:90]),
    ]
    for index, api in enumerate(relevant[:3], start=1):
        if isinstance(api, dict):
            text = api.get("api_name") or api.get("tool_name") or str(api)
        elif isinstance(api, list) and api:
            text = str(api[-1])
        else:
            text = str(api)
        oracle.append(OracleFact(f"relevant_api_{index}", text))

    return BenchmarkScenario(
        name=f"live ToolBench instruction row {offset}: {row['query_id']}",
        source="https://huggingface.co/datasets/tuandunghcmut/toolbench-v1",
        query=f"compress ToolBench instruction {row['query_id']} preserving query and relevant APIs: {row['query']} {relevant}",
        context=context,
        oracle_facts=tuple(oracle),
        max_sentences=1,
        max_compression_ratio=0.20,
    )


def _tau_airline_trace_scenario(offset: int) -> BenchmarkScenario:
    row = _fetch_row("jkazdan/taubench_traces_training_data", "default", "train", offset)
    messages = row["messages"]
    context = "\n".join(
        [
            "Dataset: jkazdan/taubench_traces_training_data",
            *_format_messages(messages),
            "other real tau-bench airline traces as long-context noise:",
            *_tau_airline_noise(exclude_offset=offset),
        ]
    )
    ids = _extract_ids(context)
    tool_names = _tool_call_names(messages)
    first_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
    oracle = []
    if first_user:
        oracle.append(OracleFact("user_goal", first_user[:80]))
    for index, value in enumerate(ids[:3], start=1):
        oracle.append(OracleFact(f"id_{index}", value))
    for index, name in enumerate(tool_names[:3], start=1):
        oracle.append(OracleFact(f"tool_call_{index}", name))

    return BenchmarkScenario(
        name=f"live tau-bench airline trace row {offset}",
        source="https://huggingface.co/datasets/jkazdan/taubench_traces_training_data",
        query=f"compress tau-bench airline trace preserving user goal, ids, and tool calls: {first_user} {ids[:3]} {tool_names[:3]}",
        context=context,
        oracle_facts=tuple(oracle[:7]),
        max_sentences=1,
        max_compression_ratio=0.20,
    )


def _tau_retail_trace_scenario(offset: int) -> BenchmarkScenario:
    row = _fetch_row("amityco/tau-bench-retail-train-next-action", "default", "train", offset)
    messages = row["conversations"]
    context = "\n".join(
        [
            "Dataset: amityco/tau-bench-retail-train-next-action",
            *_format_messages(messages),
            f"expected_next_action: {row.get('answer')}",
            "other real tau-bench retail traces as long-context noise:",
            *_tau_retail_noise(exclude_offset=offset),
        ]
    )
    ids = _extract_ids(context)
    tool_names = _tool_call_names(messages)
    first_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
    answer_ids = _extract_ids(json.dumps(row.get("answer"), ensure_ascii=False))
    answer_tools = _answer_tool_names(row.get("answer"))
    oracle = []
    if first_user:
        oracle.append(OracleFact("user_goal", first_user[:80]))
    for index, value in enumerate(answer_ids[:4], start=1):
        oracle.append(OracleFact(f"id_{index}", value))
    for index, name in enumerate((answer_tools or tool_names)[:3], start=1):
        oracle.append(OracleFact(f"tool_call_{index}", name))

    return BenchmarkScenario(
        name=f"live tau-bench retail next-action row {offset}",
        source="https://huggingface.co/datasets/amityco/tau-bench-retail-train-next-action",
        query=f"compress tau-bench retail trace preserving next action, order ids, item ids, user ids, and tool calls: {first_user} {answer_ids[:4]} {(answer_tools or tool_names)[:3]}",
        context=context,
        oracle_facts=tuple(oracle[:8]),
        max_sentences=1,
        max_compression_ratio=0.20,
    )


def _fetch_row(dataset: str, config: str, split: str, offset: int) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / _cache_name(dataset, config, split, offset)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = (
        f"{DATASETS_SERVER}?dataset={quote(dataset, safe='')}"
        f"&config={quote(config)}&split={quote(split)}&offset={offset}&length=1"
    )
    try:
        response = None
        for attempt in range(4):
            response = requests.get(url, timeout=30)
            if response.status_code != 429:
                break
            time.sleep(2 ** attempt)
        assert response is not None
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError(f"No rows returned for {dataset}/{config}/{split} offset={offset}")
        row = rows[0]["row"]
    except requests.HTTPError:
        row = _fetch_row_from_parquet(dataset, config, split, offset)
    cache_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def _cache_name(dataset: str, config: str, split: str, offset: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", f"{dataset}__{config}__{split}__{offset}")
    return f"{safe}.json"


def _fetch_row_from_parquet(dataset: str, config: str, split: str, offset: int) -> dict[str, Any]:
    import pandas as pd

    path = _parquet_path(dataset, config, split)
    local_path = hf_hub_download(repo_id=dataset, repo_type="dataset", filename=path)
    frame = pd.read_parquet(local_path)
    if offset >= len(frame):
        raise RuntimeError(f"Offset {offset} is outside {dataset}/{config}/{split} with {len(frame)} rows")
    value = frame.iloc[offset].to_dict()
    return _json_ready(value)


def _parquet_path(dataset: str, config: str, split: str) -> str:
    if dataset == "SWE-bench/SWE-bench_Lite":
        return f"data/{split}-00000-of-00001.parquet"
    if dataset == "tuandunghcmut/toolbench-v1" and config == "benchmark":
        return f"benchmark/{split}-00000-of-00001.parquet"
    if dataset == "tuandunghcmut/toolbench-v1":
        if split == "validation":
            return "data/validation-00000-of-00001.parquet"
        return "data/train-00000-of-00004.parquet"
    if dataset in {
        "jkazdan/taubench_traces_training_data",
        "amityco/tau-bench-retail-train-next-action",
    }:
        return "data/train-00000-of-00001.parquet"
    raise RuntimeError(f"No parquet mapping for {dataset}/{config}/{split}")


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _swe_noise_rows(exclude_offset: int) -> list[str]:
    lines: list[str] = []
    for offset in range(0, 8):
        if offset == exclude_offset:
            continue
        row = _fetch_row("SWE-bench/SWE-bench_Lite", "default", "test", offset)
        lines.append(
            "noise SWE row: "
            f"instance_id={row['instance_id']} repo={row['repo']} "
            f"problem_statement={_normalize(row['problem_statement'])[:1000]} "
            f"FAIL_TO_PASS={_parse_jsonish_list(row.get('FAIL_TO_PASS', []))[:2]}"
        )
    return lines


def _toolbench_conversation_noise(exclude_offset: int) -> list[str]:
    lines: list[str] = []
    for offset in range(0, 5):
        if offset == exclude_offset:
            continue
        row = _fetch_row("tuandunghcmut/toolbench-v1", "default", "validation", offset)
        messages = _conversation_messages(row["conversations"])
        first_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
        lines.append(
            "noise ToolBench conversation: "
            f"id={row['id'][:160]} first_user={_normalize(first_user)[:500]} "
            f"tools={_tool_call_names(messages)[:4]}"
        )
    return lines


def _toolbench_instruction_noise(exclude_offset: int) -> list[str]:
    lines: list[str] = []
    for offset in range(0, 6):
        if offset == exclude_offset:
            continue
        row = _fetch_row("tuandunghcmut/toolbench-v1", "benchmark", "g2_instruction", offset)
        lines.append(
            "noise ToolBench instruction: "
            f"query_id={row['query_id']} query={_normalize(row['query'])[:700]} relevant_apis={row['relevant_apis']}"
        )
    return lines


def _tau_airline_noise(exclude_offset: int) -> list[str]:
    lines: list[str] = []
    for offset in range(0, 5):
        if offset == exclude_offset:
            continue
        row = _fetch_row("jkazdan/taubench_traces_training_data", "default", "train", offset)
        messages = row["messages"]
        first_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
        lines.append(
            "noise tau airline trace: "
            f"first_user={_normalize(first_user)[:500]} ids={_extract_ids(json.dumps(messages))[:5]} "
            f"tools={_tool_call_names(messages)[:4]}"
        )
    return lines


def _tau_retail_noise(exclude_offset: int) -> list[str]:
    lines: list[str] = []
    for offset in range(0, 5):
        if offset == exclude_offset:
            continue
        row = _fetch_row("amityco/tau-bench-retail-train-next-action", "default", "train", offset)
        messages = row["conversations"]
        first_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
        lines.append(
            "noise tau retail trace: "
            f"first_user={_normalize(first_user)[:500]} ids={_extract_ids(json.dumps(messages))[:5]} "
            f"tools={_tool_call_names(messages)[:4]} answer={row.get('answer')}"
        )
    return lines


def _conversation_messages(conversations: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [
        {"role": role, "content": content}
        for role, content in zip(conversations["from"], conversations["value"])
    ]


def _format_messages(messages: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        role = message.get("role") or message.get("from") or "unknown"
        content = message.get("content") or message.get("value")
        tool_calls = message.get("tool_calls") or []
        name = message.get("name")
        if content:
            prefix = f"{role}"
            if name:
                prefix += f"[{name}]"
            lines.append(f"{prefix}: {_normalize(str(content))}")
        for call in tool_calls:
            function = call.get("function", {})
            lines.append(f"{role} tool_call: {function.get('name')}({function.get('arguments')})")
    return lines


def _tool_call_names(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if name and name not in names:
                names.append(name)
    return names


def _answer_tool_names(answer: Any) -> list[str]:
    names: list[str] = []
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except Exception:
            return names
    if not isinstance(answer, list):
        return names
    for message in answer:
        for call in message.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if name and name not in names:
                names.append(name)
    return names


def _parse_jsonish_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    return [str(item) for item in json.loads(value)]


def _first_code_identifier(text: str) -> str | None:
    matches = re.findall(r"`([^`]{4,80})`|\b([A-Za-z_][A-Za-z0-9_]{4,})\b", text)
    for left, right in matches:
        value = left or right
        if value not in {"python", "array", "False", "True"}:
            return value
    return None


def _extract_ids(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z0-9]{6,10}\b",
        r"\b[a-z]+_[a-z]+_\d{4}\b",
        r"#\w{6,12}\b",
        r"\b\d{10}\b",
        r"\b(?:credit_card|gift_card|paypal)_\d+\b",
    ]
    ids: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match not in ids:
                ids.append(match)
    return ids


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    print(run())

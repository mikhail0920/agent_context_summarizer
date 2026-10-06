from __future__ import annotations

from pathlib import Path

import pytest

from agent_summarizer import AgentContextSummarizer, SummarizerConfig
from agent_summarizer.evaluation import (
    BenchmarkScenario,
    OracleFact,
    evaluate_scenario,
    oracle_facts_in_query,
)
from benchmarks.live_hf_agent_benchmark import BENCHMARK_QUERIES
from benchmarks.real_agent_scenarios import real_agent_scenarios


def test_benchmark_queries_do_not_contain_oracle_values():
    assert all(not oracle_facts_in_query(scenario) for scenario in real_agent_scenarios())
    combined_queries = " ".join(BENCHMARK_QUERIES.values()).lower()
    assert "expected_next_action" not in combined_queries
    assert "query_id" not in combined_queries
    assert "instance_id" not in combined_queries


def test_evaluation_rejects_oracle_leakage_from_query():
    scenario = BenchmarkScenario(
        name="leaky",
        source="fixture",
        query="Summarize order #W6067464",
        context="User: Please return order #W6067464.",
        oracle_facts=(OracleFact("order_id", "#W6067464"),),
        max_sentences=1,
        max_compression_ratio=1.0,
    )

    with pytest.raises(ValueError, match="order_id"):
        evaluate_scenario(scenario, AgentContextSummarizer())


def test_live_benchmark_has_no_special_noise_marker_lines():
    source = (Path(__file__).parents[1] / "benchmarks" / "live_hf_agent_benchmark.py").read_text(
        encoding="utf-8"
    )
    marker_prefix = "other real "
    marker_suffix = " as long-context noise:"

    assert marker_prefix + "SWE-bench rows" + marker_suffix not in source
    assert marker_prefix + "ToolBench conversations" + marker_suffix not in source
    assert marker_prefix + "tau-bench retail traces" + marker_suffix not in source


def test_agent_and_baseline_use_the_same_total_output_budget():
    scenario = BenchmarkScenario(
        name="shared-budget",
        source="fixture",
        query="Summarize the primary task and operational details.",
        context="\n".join(
            [
                "user: Return my order because it arrived damaged.",
                "assistant tool_call: lookup_order({\"order_id\":\"#W6067464\"})",
                "Decision: refund the damaged item after validation.",
                *[f"Background sentence {index} about ordinary catalog data." for index in range(20)],
            ]
        ),
        oracle_facts=(),
        max_sentences=8,
        max_compression_ratio=1.0,
        max_output_chars=180,
    )
    agent = AgentContextSummarizer(SummarizerConfig(include_agent_state=True))
    baseline = AgentContextSummarizer(
        SummarizerConfig(
            include_agent_state=False,
            protected_budget_ratio=0.0,
            centrality_weight=1.0,
            query_weight=0.0,
            rarity_weight=0.0,
            anchor_weight=0.0,
            recency_weight=0.0,
            structure_weight=0.0,
        )
    )

    agent_score = evaluate_scenario(scenario, agent)
    baseline_score = evaluate_scenario(scenario, baseline)

    assert agent_score.result.output_char_budget == 180
    assert baseline_score.result.output_char_budget == 180
    assert len(agent_score.result.text) <= 180
    assert len(baseline_score.result.text) <= 180

from __future__ import annotations

from agent_summarizer import AgentContextSummarizer, SummarizerConfig
from agent_summarizer.evaluation import evaluate_scenarios
from benchmarks.real_agent_scenarios import real_agent_scenarios


def test_curated_agent_scenarios_are_stress_fixtures_not_live_proof():
    summarizer = AgentContextSummarizer(SummarizerConfig(max_sentences=12, min_sentences=5))
    report = evaluate_scenarios(real_agent_scenarios(), summarizer)

    assert report.pass_rate >= 0.5
    assert report.average_fact_recall >= 0.85
    assert report.average_compression_ratio < 0.31


def test_real_agent_scenarios_are_not_tiny_toy_contexts():
    scenarios = real_agent_scenarios()

    assert len(scenarios) >= 4
    assert all(len(scenario.context) > 5000 for scenario in scenarios)
    assert all(len(scenario.oracle_facts) >= 5 for scenario in scenarios)


def test_agent_profile_is_at_least_as_good_as_centrality_on_curated_fixtures():
    scenarios = real_agent_scenarios()
    agent_report = evaluate_scenarios(
        scenarios,
        AgentContextSummarizer(SummarizerConfig(max_sentences=12, min_sentences=5)),
    )
    baseline_report = evaluate_scenarios(
        scenarios,
        AgentContextSummarizer(
            SummarizerConfig(
                max_sentences=12,
                min_sentences=5,
                include_agent_state=False,
                protected_budget_ratio=0.0,
                centrality_weight=1.0,
                query_weight=0.0,
                rarity_weight=0.0,
                anchor_weight=0.0,
                recency_weight=0.0,
                structure_weight=0.0,
            )
        ),
    )

    assert agent_report.average_fact_recall >= baseline_report.average_fact_recall

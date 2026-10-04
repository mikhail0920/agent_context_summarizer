from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_summarizer import AgentContextSummarizer, SummarizerConfig
from agent_summarizer.evaluation import BenchmarkScenario, OracleFact, evaluate_scenarios, format_report


SOURCES = {
    "swe_bench": "https://www.swebench.com/SWE-bench/guides/datasets/",
    "swe_agent": "https://swe-agent.com/latest/usage/trajectories/",
    "tau_bench": "https://sierra.ai/blog/benchmarking-ai-agents",
    "toolbench": "https://mcpbr.org/toolbench",
}


def real_agent_scenarios() -> list[BenchmarkScenario]:
    return [
        _swe_bench_issue_context(),
        _swe_agent_trajectory_context(),
        _tau_airline_policy_context(),
        _toolbench_orchestration_context(),
    ]


def run() -> str:
    summarizer = AgentContextSummarizer(SummarizerConfig(max_sentences=12, min_sentences=5))
    report = evaluate_scenarios(real_agent_scenarios(), summarizer)
    baseline = AgentContextSummarizer(
        SummarizerConfig(
            max_sentences=12,
            min_sentences=5,
            protected_budget_ratio=0.0,
            centrality_weight=1.0,
            query_weight=0.08,
            rarity_weight=0.0,
            anchor_weight=0.0,
            recency_weight=0.0,
            structure_weight=0.0,
        )
    )
    baseline_report = evaluate_scenarios(real_agent_scenarios(), baseline)
    return "\n\n".join(
        [
            "PROTECTED AGENT PROFILE",
            format_report(report),
            "CENTRALITY BASELINE",
            format_report(baseline_report),
        ]
    )


def _swe_bench_issue_context() -> BenchmarkScenario:
    filler = "\n".join(
        f"Repository scan note {i}: astropy has many modeling modules, many examples, and repeated docs references."
        for i in range(70)
    )
    context = f"""
    Source family: SWE-bench uses real GitHub issues with repo, problem_statement, test_patch, FAIL_TO_PASS, and PASS_TO_PASS fields.
    User task: Fix astropy__astropy-12907 without reading the gold patch.
    Issue title: Modeling's separability_matrix does not compute separability correctly for nested CompoundModels.
    Problem detail: A nested compound model like m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5) produces a wrong separability matrix.
    Expected behavior: separability_matrix should preserve the diagonal separability of the independent Linear1D branches.
    {filler}
    Constraint: do not inspect the gold solution patch before proposing the fix.
    Tool: python -m pytest astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6] failed.
    Tool: python -m pytest astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9] failed.
    File focus: astropy/modeling/separable.py is the likely implementation target.
    Decision: start by reproducing the nested CompoundModel behavior and add regression coverage before editing the algorithm.
    """
    return BenchmarkScenario(
        name="swe-bench nested CompoundModel issue",
        source=SOURCES["swe_bench"],
        query="fix real SWE-bench astropy separability issue and keep tests/constraints",
        context=context,
        max_sentences=10,
        max_compression_ratio=0.28,
        oracle_facts=(
            OracleFact("issue", "separability_matrix does not compute separability correctly"),
            OracleFact("constraint", "do not inspect the gold solution patch"),
            OracleFact("failing_test_1", "test_separable[compound_model6-result6]"),
            OracleFact("failing_test_2", "test_separable[compound_model9-result9]"),
            OracleFact("file_focus", "astropy/modeling/separable.py"),
            OracleFact("decision", "add regression coverage before editing the algorithm"),
        ),
    )


def _swe_agent_trajectory_context() -> BenchmarkScenario:
    filler = "\n".join(
        f"Trajectory observation {i}: package installation, environment setup, and ordinary repository listing completed."
        for i in range(55)
    )
    context = f"""
    Source family: SWE-agent trajectories contain thoughts, actions, observations, open_file state, submissions, and logs.
    User task: Resolve a marshmallow TimeDelta precision bug from a trajectory-style coding run.
    Observation: open_file is /marshmallow-code__marshmallow/src/marshmallow/fields.py.
    Issue detail: TimeDelta serialization truncates fractional units and should round to the nearest integer.
    {filler}
    Tool: python -m pytest tests/test_serialization.py::test_timedelta_rounding failed with AssertionError: expected 12, got 11.
    Action taken: edit src/marshmallow/fields.py around class TimeDelta.
    Decision: replace int(value.total_seconds() / base_unit.total_seconds()) with int(round(value.total_seconds() / base_unit.total_seconds())).
    Constraint: submit only after rerunning the focused TimeDelta tests.
    Tool: python -m pytest tests/test_serialization.py::test_timedelta_rounding passed.
    """
    return BenchmarkScenario(
        name="swe-agent trajectory with patch decision",
        source=SOURCES["swe_agent"],
        query="compress coding trajectory while keeping file, failing test, patch decision, and submit constraint",
        context=context,
        max_sentences=9,
        max_compression_ratio=0.30,
        oracle_facts=(
            OracleFact("open_file", "src/marshmallow/fields.py"),
            OracleFact("failure", "test_timedelta_rounding failed"),
            OracleFact("decision", "int(round(value.total_seconds() / base_unit.total_seconds()))"),
            OracleFact("constraint", "submit only after rerunning the focused TimeDelta tests"),
            OracleFact("pass", "test_timedelta_rounding passed"),
        ),
    )


def _tau_airline_policy_context() -> BenchmarkScenario:
    filler = "\n".join(
        f"Conversation turn {i}: user and assistant discuss seat preferences, baggage, loyalty number, and ordinary travel details."
        for i in range(65)
    )
    context = f"""
    Source family: tau-bench style customer-service agents must remember long multi-turn conversations, policies, tools, and target database state.
    User goal: change reservation HZ4921 from SFO to JFK and keep the passenger on the same travel date.
    Policy: destination changes are allowed only before check-in and only if the fare difference is collected.
    Policy: basic economy tickets cannot be changed after check-in.
    Tool: get_reservation(record_locator="HZ4921") returned fare_class="standard", checked_in=false, passenger="Mira Chen".
    {filler}
    User preference: user refuses red-eye flights and accepts a fare difference up to 120 USD.
    Tool: search_flights(origin="SFO", destination="JFK", date="2026-05-18") returned flight UA218 at 14:20 with fare_difference=87.
    Decision: rebook reservation HZ4921 to flight UA218 and charge 87 USD.
    Constraint: do not offer basic economy downgrade because the user paid for standard fare.
    Final state target: reservation destination JFK, flight UA218, charged_amount=87, checked_in=false.
    """
    return BenchmarkScenario(
        name="tau-bench airline long-horizon policy memory",
        source=SOURCES["tau_bench"],
        query="compress airline rebooking context with policy constraints and target database state",
        context=context,
        max_sentences=10,
        max_compression_ratio=0.30,
        oracle_facts=(
            OracleFact("reservation", "HZ4921"),
            OracleFact("policy", "destination changes are allowed only before check-in"),
            OracleFact("user_preference", "refuses red-eye flights"),
            OracleFact("tool_result", "UA218 at 14:20 with fare_difference=87"),
            OracleFact("decision", "rebook reservation HZ4921 to flight UA218"),
            OracleFact("final_state", "charged_amount=87"),
        ),
    )


def _toolbench_orchestration_context() -> BenchmarkScenario:
    filler = "\n".join(
        f"Tool catalog note {i}: unrelated APIs include news search, image upload, calendar lookup, and unit conversion."
        for i in range(75)
    )
    context = f"""
    Source family: ToolBench evaluates real-world API tool use, correct parameters, and multi-step sequence matching.
    User query: Find restaurants near Central Park in New York, then check the weather to decide if outdoor dining is feasible.
    Available tool: search_restaurants(location, type, open_now)
    Available tool: get_weather(location, units)
    Available tool: get_timezone(location)
    {filler}
    Required sequence step 1: search_restaurants(location="Central Park, New York", type="restaurant", open_now=true).
    Required sequence step 2: get_weather(location="New York", units="imperial").
    Constraint: call search_restaurants before get_weather because the restaurant shortlist is the primary user request.
    Decision: outdoor dining is feasible only if weather.conditions is not rain and wind_mph < 20.
    Schema warning: parameter must be named units, not unit.
    """
    return BenchmarkScenario(
        name="toolbench multi-tool sequence and schema",
        source=SOURCES["toolbench"],
        query="compress tool orchestration context and preserve exact tool sequence and parameters",
        context=context,
        max_sentences=9,
        max_compression_ratio=0.28,
        oracle_facts=(
            OracleFact("query", "Find restaurants near Central Park"),
            OracleFact("step_1", "search_restaurants(location=\"Central Park, New York\", type=\"restaurant\", open_now=true)"),
            OracleFact("step_2", "get_weather(location=\"New York\", units=\"imperial\")"),
            OracleFact("sequence_constraint", "call search_restaurants before get_weather"),
            OracleFact("schema_warning", "parameter must be named units"),
        ),
    )


if __name__ == "__main__":
    print(run())

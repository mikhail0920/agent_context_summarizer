from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .core import AgentContextSummarizer
from .models import SummaryResult


@dataclass(frozen=True)
class OracleFact:
    name: str
    must_contain: str
    weight: float = 1.0


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    source: str
    query: str
    context: str
    oracle_facts: tuple[OracleFact, ...]
    max_sentences: int
    max_compression_ratio: float
    max_output_chars: int | None = None


@dataclass(frozen=True)
class ScenarioScore:
    scenario: BenchmarkScenario
    result: SummaryResult
    fact_recall: float
    compression_ok: bool
    missing_facts: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.fact_recall >= 1.0 and self.compression_ok


@dataclass(frozen=True)
class BenchmarkReport:
    scores: tuple[ScenarioScore, ...]

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(score.passed for score in self.scores) / len(self.scores)

    @property
    def average_fact_recall(self) -> float:
        return mean(score.fact_recall for score in self.scores) if self.scores else 0.0

    @property
    def average_compression_ratio(self) -> float:
        return mean(score.result.compression_ratio for score in self.scores) if self.scores else 0.0

    @property
    def average_output_chars(self) -> float:
        return mean(len(score.result.text) for score in self.scores) if self.scores else 0.0


def evaluate_scenarios(
    scenarios: list[BenchmarkScenario],
    summarizer: AgentContextSummarizer | None = None,
) -> BenchmarkReport:
    summarizer = summarizer or AgentContextSummarizer()
    scores = [
        evaluate_scenario(scenario, summarizer)
        for scenario in scenarios
    ]
    return BenchmarkReport(tuple(scores))


def evaluate_scenario(scenario: BenchmarkScenario, summarizer: AgentContextSummarizer) -> ScenarioScore:
    leaked_facts = oracle_facts_in_query(scenario)
    if leaked_facts:
        names = ", ".join(leaked_facts)
        raise ValueError(f"Benchmark query contains oracle facts: {names}")

    result = summarizer.summarize(
        scenario.context,
        query=scenario.query,
        max_sentences=scenario.max_sentences,
        max_output_chars=scenario.max_output_chars,
    )
    summary_lower = _normalize_for_match(result.text)
    total_weight = sum(fact.weight for fact in scenario.oracle_facts)
    found_weight = 0.0
    missing: list[str] = []

    for fact in scenario.oracle_facts:
        if _normalize_for_match(fact.must_contain) in summary_lower:
            found_weight += fact.weight
        else:
            missing.append(fact.name)

    fact_recall = found_weight / total_weight if total_weight else 1.0
    return ScenarioScore(
        scenario=scenario,
        result=result,
        fact_recall=fact_recall,
        compression_ok=result.compression_ratio <= scenario.max_compression_ratio,
        missing_facts=tuple(missing),
    )


def format_report(report: BenchmarkReport) -> str:
    lines = [
        "Agent context summarization benchmark",
        f"pass_rate={report.pass_rate:.2%}",
        f"average_fact_recall={report.average_fact_recall:.2%}",
        f"average_compression_ratio={report.average_compression_ratio:.2%}",
        f"average_output_chars={report.average_output_chars:.2f}",
        "",
    ]
    for score in report.scores:
        budget = score.result.output_char_budget
        budget_text = str(budget) if budget is not None else "-"
        lines.extend(
            [
                f"- {score.scenario.name}",
                f"  source: {score.scenario.source}",
                f"  fact_recall={score.fact_recall:.2%}",
                f"  compression_ratio={score.result.compression_ratio:.2%}",
                f"  output_chars={len(score.result.text)}/{budget_text}",
                f"  passed={score.passed}",
                f"  missing={', '.join(score.missing_facts) if score.missing_facts else '-'}",
            ]
        )
    return "\n".join(lines)


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def oracle_facts_in_query(scenario: BenchmarkScenario) -> tuple[str, ...]:
    """Return oracle fact names copied verbatim into the benchmark query."""

    query = _normalize_for_match(scenario.query)
    if not query:
        return ()
    return tuple(
        fact.name
        for fact in scenario.oracle_facts
        if (value := _normalize_for_match(fact.must_contain)) and value in query
    )

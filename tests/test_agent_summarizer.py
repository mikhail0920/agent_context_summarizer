from __future__ import annotations

from agent_summarizer import AgentContextSummarizer, SummarizerConfig
from agent_summarizer.splitting import split_sentences
from agent_summarizer.state import extract_agent_state


def test_preserves_rare_operational_facts_in_long_noisy_context():
    context = _long_agent_context()
    summarizer = AgentContextSummarizer(SummarizerConfig(max_sentences=10, min_sentences=6))

    result = summarizer.summarize(
        context,
        query="fix failing billing tests and preserve user constraints",
    )

    assert "Нельзя менять migrations/20260211_add_invoice_status.sql" in result.text
    assert "pytest tests/billing/test_invoice_retry.py::test_retries_are_idempotent" in result.text
    assert "Decision: keep SQLite for local tests" in result.text
    assert "OPENAI_API_KEY" in result.text
    assert result.compression_ratio < 0.45
    assert {"constraint", "error", "decision", "security"}.issubset(set(result.labels))


def test_balances_central_topic_with_anchor_facts():
    context = _long_product_research_context()
    result = AgentContextSummarizer(SummarizerConfig(max_sentences=8, min_sentences=5)).summarize(
        context,
        query="summarize launch blockers and agreed implementation plan",
    )

    assert "Decision: ship the CSV importer behind feature flag import_v2" in result.text
    assert "User requested: do not change the public REST contract" in result.text
    assert "Error: production sample file contains Windows-1251 headers" in result.text
    assert "The main product goal is reducing onboarding time" in result.text
    assert "docs/importer/spec.md" in result.text


def test_sentence_splitter_keeps_tool_lines_and_bullets_as_units():
    text = """
    User: Please fix the tests. Do not touch migrations.
    Tool: python -m pytest tests/api/test_users.py failed with AssertionError.
    - Decision: keep the old endpoint /v1/users for mobile clients.

    This paragraph has two sentences. It should be split cleanly.
    """

    sentences = split_sentences(text)

    assert "User: Please fix the tests. Do not touch migrations." in sentences
    assert "Tool: python -m pytest tests/api/test_users.py failed with AssertionError." in sentences
    assert "- Decision: keep the old endpoint /v1/users for mobile clients." in sentences
    assert "This paragraph has two sentences." in sentences
    assert "It should be split cleanly." in sentences


def test_empty_input_returns_empty_result():
    result = AgentContextSummarizer().summarize("   ")

    assert result.text == ""
    assert result.sentences == ()
    assert result.compression_ratio == 0.0


def test_extracts_structured_agent_state_from_tool_trace():
    text = """
    Dataset: tau-style trace
    user: Hello! I need to return order #W6067464.
    assistant tool_call: get_order_details({"order_id":"#W6067464"})
    tool[get_order_details]: {"items": [{"item_id": "8917609800"}]}
    expected_next_action: [{'role': 'assistant', 'tool_calls': [{'function': {'name': 'return_delivered_order_items', 'arguments': '{"order_id":"#W6067464","item_ids":["8917609800"]}'}}]}]
    other real tau-bench retail traces as long-context noise:
    noise tau retail trace: first_user=I need a different order ids=['#W0000000'] tools=['wrong_tool']
    """

    units = extract_agent_state(text, "return #W6067464 8917609800 return_delivered_order_items", max_units=10)
    rendered = "\n".join(unit.text for unit in units)

    assert "#W6067464" in rendered
    assert "8917609800" in rendered
    assert "return_delivered_order_items" in rendered
    assert "#W0000000" not in rendered


def _long_agent_context() -> str:
    filler = "\n".join(
        f"Assistant: Refactor pass {i} mostly discussed naming, formatting, and ordinary cleanup around invoice services."
        for i in range(35)
    )
    return f"""
    System: You are helping inside a Python billing repository.
    User: Нужно починить flaky billing retry tests, но не делать большой рефакторинг.
    Assistant: I inspected the repository and found service, model, and API layers.
    {filler}
    User: Нельзя менять migrations/20260211_add_invoice_status.sql без моего явного разрешения.
    Tool: python -m pytest tests/billing/test_invoice_retry.py::test_retries_are_idempotent failed with AssertionError: expected 1 retry event, got 2.
    Assistant: The repeated event seems related to InvoiceRetryPolicy.record_attempt being called before transaction rollback.
    Decision: keep SQLite for local tests because CI also uses SQLite in this package.
    Tool: git diff showed edits in billing/retry.py and tests/billing/test_invoice_retry.py only.
    User requested: preserve the public behavior of POST /v1/invoices/retry for mobile clients.
    Warning: OPENAI_API_KEY appeared in a copied shell transcript and must not be echoed back in final logs.
    Assistant: The most likely fix is to make retry recording idempotent by invoice_id and attempt_number.
    Tool: python -m pytest tests/billing/test_invoice_retry.py passed after the idempotency guard.
    """


def _long_product_research_context() -> str:
    filler = "\n".join(
        f"Research note {i}: customers repeatedly mention import screens, onboarding docs, and confusing status messages."
        for i in range(45)
    )
    return f"""
    The main product goal is reducing onboarding time for operations teams importing legacy customer files.
    The team compared spreadsheet cleanup, guided mapping, automatic validation, and better examples.
    {filler}
    User requested: do not change the public REST contract because three enterprise clients call it directly.
    Error: production sample file contains Windows-1251 headers and the current parser assumes UTF-8.
    Decision: ship the CSV importer behind feature flag import_v2 and enable it only for internal workspaces.
    Open issue: docs/importer/spec.md still says duplicate emails are rejected, but the new behavior merges them.
    Tool: npm test -- importer failed in packages/importer/src/encoding.test.ts with expected "ФИО", received "���".
    The main implementation plan is detecting encoding, normalizing headers, preserving the old endpoint, and logging merge decisions.
    """

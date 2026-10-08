"""tests/test_codeexec.py - Acceptance tests for Sandboxed Code Execution & Auto-Heal (M12 / Exp 6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from guardian.genai.codeexec import (
    SandboxedCodeExecutor,
    SecurityViolation,
    check_ast_security,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "t_s": [0.0, 10.0, 20.0, 30.0],
        "temp_c": [24.5, 25.1, 26.8, 28.2],
        "current_a": [1.2, 1.5, 2.1, 2.0],
        "voltage_v": [3.9, 3.85, 3.75, 3.7],
        "cycle": [10, 10, 10, 10],
        "battery_id": ["B0005", "B0005", "B0005", "B0005"],
    })


def test_malicious_code_rejected_before_execution(sample_df: pd.DataFrame):
    """Verify that malicious code triggers SecurityViolation and never executes."""
    malicious_snippets = [
        "import os\nresult = os.listdir('.')",
        "import sys\nresult = sys.version",
        "from subprocess import Popen\nresult = 1",
        "result = open('secret.txt', 'w')",
        "result = eval('2 + 2')",
        "result = __import__('os').system('echo hack')",
        "result = df.__class__.__bases__",
        "result = df._mgr",
    ]

    executor = SandboxedCodeExecutor()

    for code in malicious_snippets:
        with pytest.raises((SecurityViolation, SyntaxError)):
            check_ast_security(code)

        # Also verify executor rejects it cleanly without crashing
        res = executor.run_analysis(
            question="Analyze data",
            df=sample_df,
            initial_code_override=code,
        )
        assert res["status"] == "REJECTED_SECURITY"
        assert res["healed"] is False
        assert "security policy" in res["answer"].lower()


def test_valid_code_executes_successfully(sample_df: pd.DataFrame):
    """Verify clean pandas code executes and extracts expected scalar."""
    executor = SandboxedCodeExecutor()
    valid_code = "result = float(df['temp_c'].max())"

    res = executor.run_analysis(
        question="What is the maximum temperature?",
        df=sample_df,
        initial_code_override=valid_code,
    )

    assert res["status"] == "PASS"
    assert res["healed"] is False
    assert res["attempts"] == 1
    assert abs(res["answer"] - 28.2) < 1e-4


class MockHealingLLM:
    """Mock LLM that repairs column name KeyError in code generation."""

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Any = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> str:
        # Check prompt context
        user_msg = messages[-1]["content"]
        if "temperature_celsius" in user_msg or "KeyError" in user_msg:
            # Successfully heal column name
            return "```python\nresult = float(df['temp_c'].max())\n```"
        return "result = float(df['temp_c'].mean())"


def test_seeded_buggy_code_autoheals_within_cap(sample_df: pd.DataFrame):
    """Verify seeded KeyError snippet gets repaired within <= 3 attempts."""
    mock_llm = MockHealingLLM()
    executor = SandboxedCodeExecutor(llm_client=mock_llm, max_attempts=3)

    # Seeded buggy code using non-existent column
    buggy_code = "result = float(df['temperature_celsius'].max())"

    res = executor.run_analysis(
        question="What is the maximum temperature?",
        df=sample_df,
        initial_code_override=buggy_code,
    )

    assert res["status"] == "PASS"
    assert res["healed"] is True
    assert res["attempts"] == 2
    assert abs(res["answer"] - 28.2) < 1e-4


def test_runtime_codeexec_logged():
    """Verify that execution attempts are logged into reports/autoheal/log.jsonl."""
    log_file = Path("reports/autoheal/log.jsonl")
    assert log_file.exists()

    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    codeexec_records = [r for r in records if r.get("source") == "runtime_codeexec"]
    assert len(codeexec_records) > 0

    sample_rec = codeexec_records[-1]
    assert sample_rec["milestone"] == "M12"
    assert "status" in sample_rec
    assert "duration_s" in sample_rec

"""tests/test_chat_agent.py - Acceptance tests for Guardian Chat Agent (M12 / Exp 6)."""

from __future__ import annotations

import pandas as pd
import pytest

from guardian.genai.chat_agent import GuardianChatAgent
from guardian.genai.llm_client import TemplateClient


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame({
        "t_s": [0.0, 50.0, 100.0, 150.0],
        "temp_c": [25.0, 26.5, 28.0, 29.5],
        "current_a": [1.5, 1.5, 1.5, 1.5],
        "voltage_v": [3.8, 3.75, 3.7, 3.65],
        "cycle": [50, 50, 50, 50],
        "battery_id": ["B0005", "B0005", "B0005", "B0005"],
        "soc_pct": [80.0, 75.0, 70.0, 65.0],
        "soh_pct": [92.0, 92.0, 92.0, 92.0],
        "dT_dt": [0.03, 0.03, 0.03, 0.03],
    })


def test_agent_tools_direct(test_df: pd.DataFrame):
    """Verify all individual diagnostic tools return structured results."""
    agent = GuardianChatAgent(telemetry_df=test_df, llm_client=TemplateClient())

    # 1. get_current_state
    state = agent.get_current_state(battery_id="B0005", cycle=50, t_s=50.0)
    assert state["battery_id"] == "B0005"
    assert "temp_c" in state
    assert "voltage_v" in state

    # 2. get_risk
    risk = agent.get_risk(battery_id="B0005", cycle=50, t_s=50.0)
    assert "risk_score" in risk
    assert "risk_level" in risk

    # 3. list_events
    events = agent.list_events(battery_id="B0005", cycle=50)
    assert "events" in events

    # 4. run_whatif
    whatif = agent.run_whatif(battery_id="B0005", current_a=3.0, temp_c=45.0, soc_pct=60.0)
    assert "simulated_risk_level" in whatif
    assert "simulated_risk_score" in whatif

    # 5. search_kb
    kb_res = agent.search_kb("thermal runaway propagation")
    assert "passages" in kb_res
    assert len(kb_res["passages"]) > 0

    # 6. run_analysis
    analysis = agent.run_analysis("What is the max temp_c?")
    assert "answer" in analysis


def test_agent_conversation_flow(test_df: pd.DataFrame):
    """Verify chat agent conversation turn and memory update."""
    agent = GuardianChatAgent(telemetry_df=test_df, llm_client=TemplateClient())

    response = agent.chat("What was the maximum temperature observed?")
    assert "answer" in response
    assert isinstance(response["answer"], str)
    assert len(response["answer"]) > 0
    assert len(agent.history) == 2  # 1 user + 1 assistant turn

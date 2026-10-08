"""Tool-calling chat agent for AI Battery Safety Guardian (M12 / Exp 6).

Implements conversational assistant equipped with diagnostic tools:
- get_current_state(battery_id, cycle, t_s)
- get_risk(battery_id, cycle, t_s)
- list_events(battery_id, cycle)
- run_whatif(battery_id, current_a, temp_c, soc_pct)
- search_kb(query)
- run_analysis(question) [sandboxed pandas code execution with auto-heal]

Enforces conversation memory (last 6 turns) and max 5 tool calls per turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from guardian.genai.codeexec import SandboxedCodeExecutor
from guardian.genai.llm_client import LLMClient, get_llm_client
from guardian.genai.rag import RAGPipeline
from guardian.risk.engine import assess
from guardian.risk.events import detect_events
from guardian.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_AGENT_PROMPT = """You are the AI Battery Safety Guardian Assistant, an expert decision-support copilot for high-voltage battery health and safety monitoring.

You have access to the following diagnostic tools:
1. `get_current_state(battery_id, cycle, t_s)`: Returns real-time telemetry (temp, voltage, current, SOC, SOH).
2. `get_risk(battery_id, cycle, t_s)`: Returns deterministic risk assessment, risk score (0-100), level, and contributing factors.
3. `list_events(battery_id, cycle)`: Returns debounced safety events and advisories for this session.
4. `run_whatif(battery_id, current_a, temp_c, soc_pct)`: Evaluates simulated scenario risk and temperature trends.
5. `search_kb(query)`: Searches technical knowledge base for electrochemical failure mechanisms and SOPs.
6. `run_analysis(question)`: Generates and executes sandboxed pandas code to query the telemetry dataset.

SAFETY RULES:
- Never issue autonomous control commands (e.g., "shut down the battery", "disconnect now"). Always advise human inspection or escalation.
- Ground all numerical claims in tool output.
- When referencing knowledge base facts, preserve citations like [KB-xx-Cx].
- To use a tool, reply in valid JSON format: `{"tool": "<tool_name>", "args": {<arguments>}}`.
- When ready to give the final answer, reply in valid JSON format: `{"final_answer": "<your complete response>"}`.
"""


class GuardianChatAgent:
    """Conversational assistant with tool orchestration and sandboxed code execution."""

    def __init__(
        self,
        telemetry_df: Optional[pd.DataFrame] = None,
        llm_client: Optional[LLMClient] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        code_executor: Optional[SandboxedCodeExecutor] = None,
        max_turns_memory: int = 6,
        max_tool_calls_per_turn: int = 5,
    ) -> None:
        self.df = telemetry_df
        if self.df is None:
            data_path = Path("data/processed/battery_telemetry.csv")
            if data_path.exists():
                self.df = pd.read_csv(data_path)
            else:
                self.df = pd.DataFrame()

        self.llm_client = llm_client or get_llm_client()
        self.rag = rag_pipeline or RAGPipeline(llm_client=self.llm_client)
        self.code_executor = code_executor or SandboxedCodeExecutor(llm_client=self.llm_client)
        self.max_turns_memory = max_turns_memory
        self.max_tool_calls = max_tool_calls_per_turn

        self.history: List[Dict[str, str]] = []

    def get_current_state(self, battery_id: str = "B0005", cycle: int = 50, t_s: float = 300.0) -> Dict[str, Any]:
        """Fetch telemetry state matching battery, cycle, and time."""
        if self.df.empty:
            return {"error": "Telemetry dataset not loaded"}

        sub = self.df[(self.df["battery_id"] == battery_id) & (self.df["cycle"] == cycle)]
        if sub.empty:
            sub = self.df[self.df["battery_id"] == battery_id]
        if sub.empty:
            sub = self.df

        # Find closest row by t_s
        idx = (sub["t_s"] - t_s).abs().idxmin()
        row = sub.loc[idx]

        return {
            "battery_id": str(row.get("battery_id", battery_id)),
            "cycle": int(row.get("cycle", cycle)),
            "t_s": float(row.get("t_s", t_s)),
            "temp_c": float(row.get("temp_c", 25.0)),
            "voltage_v": float(row.get("voltage_v", 3.8)),
            "current_a": float(row.get("current_a", 1.5)),
            "soc_pct": float(row.get("soc_pct", 60.0)),
            "soh_pct": float(row.get("soh_pct", 90.0)),
            "dT_dt": float(row.get("dT_dt", 0.05)),
        }

    def get_risk(self, battery_id: str = "B0005", cycle: int = 50, t_s: float = 300.0) -> Dict[str, Any]:
        """Run deterministic risk assessment at the specified point."""
        state = self.get_current_state(battery_id, cycle, t_s)
        if "error" in state:
            return state

        res = assess(
            state=state,
            anomaly_score=0.15,
            pred_temp=state["temp_c"] + 0.5,
        )
        return {
            "battery_id": battery_id,
            "cycle": cycle,
            "t_s": t_s,
            "risk_score": res.risk_score,
            "risk_level": res.risk_level.value if hasattr(res.risk_level, "value") else str(res.risk_level),
            "hard_overrides": res.hard_overrides,
            "factors": [f.to_dict() for f in res.factors],
        }

    def list_events(self, battery_id: str = "B0005", cycle: int = 50) -> Dict[str, Any]:
        """List detected safety events and advisories for a battery cycle."""
        if self.df.empty:
            return {"events": []}

        sub = self.df[(self.df["battery_id"] == battery_id) & (self.df["cycle"] == cycle)]
        if sub.empty:
            sub = self.df.head(200)

        # Generate assessments for each step
        assessments_list = []
        for row in sub.to_dict(orient="records"):
            a = assess(state=row, anomaly_score=0.1, pred_temp=float(row.get("temp_c", 25.0)) + 0.2)
            assessments_list.append(a)

        events = detect_events(assessments_list)
        return {
            "battery_id": battery_id,
            "cycle": cycle,
            "event_count": len(events),
            "events": [
                {
                    "event_id": e.event_id,
                    "t_s": e.t_s,
                    "risk_level": e.risk_level.value if hasattr(e.risk_level, "value") else str(e.risk_level),
                    "risk_score": e.risk_score,
                    "reasons": e.reasons,
                    "recommendations": e.recommendations,
                }
                for e in events[:10]
            ],
        }

    def run_whatif(
        self,
        battery_id: str = "B0005",
        current_a: float = 2.0,
        temp_c: float = 30.0,
        soc_pct: float = 50.0,
    ) -> Dict[str, Any]:
        """Simulate modified conditions and predict safety metrics."""
        sim_state = {
            "battery_id": battery_id,
            "current_a": current_a,
            "temp_c": temp_c,
            "voltage_v": 3.7,
            "soc_pct": soc_pct,
            "dT_dt": (current_a / 2.0) * (0.8 / 60.0),
        }
        res = assess(
            state=sim_state,
            anomaly_score=0.2,
            pred_temp=temp_c + (current_a * 0.5),
        )
        return {
            "battery_id": battery_id,
            "simulated_current_a": current_a,
            "simulated_temp_c": temp_c,
            "simulated_soc_pct": soc_pct,
            "simulated_risk_level": res.risk_level.value if hasattr(res.risk_level, "value") else str(res.risk_level),
            "simulated_risk_score": res.risk_score,
            "hard_overrides": res.hard_overrides,
        }

    def search_kb(self, query: str) -> Dict[str, Any]:
        """Query technical knowledge base and return top passages."""
        results = self.rag.retrieve(query, top_k=3)
        return {
            "query": query,
            "retrieved_count": len(results),
            "passages": [
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "doc_id": r.get("doc_id", ""),
                    "section": r.get("section", r.get("doc_title", "")),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "text": r.get("text", ""),
                }
                for r in results
            ],
        }

    def run_analysis(self, question: str) -> Dict[str, Any]:
        """Execute sandboxed pandas code analysis on telemetry dataset."""
        return self.code_executor.run_analysis(question, self.df)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route tool invocation to appropriate handler."""
        tool_map: Dict[str, Callable[..., Any]] = {
            "get_current_state": self.get_current_state,
            "get_risk": self.get_risk,
            "list_events": self.list_events,
            "run_whatif": self.run_whatif,
            "search_kb": self.search_kb,
            "run_analysis": self.run_analysis,
        }
        if tool_name not in tool_map:
            return {"error": f"Unknown tool '{tool_name}'. Available: {list(tool_map.keys())}"}

        try:
            handler = tool_map[tool_name]
            return handler(**args)
        except Exception as exc:
            return {"error": f"Tool execution failed: {type(exc).__name__}: {exc}"}

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Execute single conversational turn with tool orchestration."""
        # Trim history to max turns
        active_history = self.history[-(self.max_turns_memory * 2) :]
        messages = [{"role": "system", "content": SYSTEM_AGENT_PROMPT}] + active_history
        messages.append({"role": "user", "content": user_message})

        tools_used: List[Dict[str, Any]] = []
        code_traces: List[Dict[str, Any]] = []

        # If query asks for specific data calculation, proactively route to run_analysis
        lower_q = user_message.lower()
        if any(w in lower_q for w in ("how many", "max temp", "maximum temperature", "average", "min voltage", "highest current")):
            analysis_res = self.run_analysis(user_message)
            tools_used.append({"tool": "run_analysis", "args": {"question": user_message}, "result": analysis_res})
            if "code" in analysis_res:
                code_traces.append(analysis_res)

            messages.append({
                "role": "system",
                "content": f"Analysis tool output: {json.dumps(analysis_res)}. Provide a helpful answer based on this result.",
            })

        for _ in range(self.max_tool_calls):
            raw_response = self.llm_client.generate(messages, temperature=0.1, max_tokens=600)

            # Try parsing tool invocation or final answer
            try:
                # Extract JSON if wrapped in markdown
                cleaned = raw_response.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_name = parsed["tool"]
                    tool_args = parsed.get("args", {})
                    tool_output = self._execute_tool(tool_name, tool_args)
                    tools_used.append({"tool": tool_name, "args": tool_args, "result": tool_output})

                    if tool_name == "run_analysis" and isinstance(tool_output, dict):
                        code_traces.append(tool_output)

                    # Feed tool output back to conversation
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' returned:\n{json.dumps(tool_output, default=str)}\nContinue answering the original question.",
                    })
                    continue

                elif isinstance(parsed, dict) and "final_answer" in parsed:
                    final_text = str(parsed["final_answer"])
                    self.history.append({"role": "user", "content": user_message})
                    self.history.append({"role": "assistant", "content": final_text})
                    return {
                        "answer": final_text,
                        "tools_called": tools_used,
                        "code_traces": code_traces,
                    }
            except Exception:
                # LLM output direct text
                pass

            # If plain text returned
            final_text = raw_response.strip()
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": final_text})
            return {
                "answer": final_text,
                "tools_called": tools_used,
                "code_traces": code_traces,
            }

        # Exhausted tool calls
        final_text = "I performed multiple diagnostic checks to investigate your query. Please review the attached tool telemetry."
        return {
            "answer": final_text,
            "tools_called": tools_used,
            "code_traces": code_traces,
        }

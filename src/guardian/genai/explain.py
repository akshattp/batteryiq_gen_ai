"""Safety explanation generation with deterministic template fallback (Section 8.4).

Ensures that even if external LLM APIs are unreachable or offline, the system
consistently produces faithful, non-hallucinatory explanations.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from guardian.genai.llm_client import LLMClient, LLMUnavailable, TemplateClient, get_llm_client
from guardian.genai.prompts import load_prompt
from guardian.genai.validator import Violation, validate_explanation
from guardian.utils.logging import get_logger
from guardian.utils.schemas import ExplanationOutput

logger = get_logger(__name__)


def _log_runtime_event(
    milestone: str,
    attempt: int,
    status: str,
    error_class: Optional[str] = None,
    error_excerpt: Optional[str] = None,
    fix_summary: Optional[str] = None,
    duration_s: float = 0.0,
    linked_to: Optional[str] = None,
) -> Optional[str]:
    """Helper to log runtime validation/heal events to reports/autoheal/log.jsonl."""
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tools.autoheal import log_event
        entry = log_event(
            source="runtime",
            milestone=milestone,
            attempt=attempt,
            command="generate_explanation",
            status=status,
            exit_code=0 if status == "PASS" else 1,
            error_class=error_class,
            error_excerpt=error_excerpt,
            fix_summary=fix_summary,
            duration_s=duration_s,
            linked_to=linked_to,
        )
        return entry.get("id")
    except Exception as exc:
        logger.warning(f"Could not write runtime autoheal log: {exc}")
        return None


def template_explanation(assessment: Dict[str, Any]) -> ExplanationOutput:
    """Generate a 100% deterministic, faithful explanation without external LLM calls.

    Args:
        assessment: Serialized RiskAssessment dictionary.

    Returns:
        ExplanationOutput with summary, reasons, and non-control recommendations.
    """
    level = str(assessment.get("risk_level", "NORMAL")).upper()
    score = float(assessment.get("risk_score", 0.0))
    factors = assessment.get("factors", [])
    overrides = assessment.get("hard_overrides", [])

    reasons: List[str] = []
    if overrides:
        for ov in overrides:
            reasons.append(f"Hard safety override triggered: {ov}")

    for f in factors[:3]:
        name = f.get("name", "Factor")
        weighted = float(f.get("weighted_score", 0.0))
        raw = float(f.get("raw_value", 0.0))
        reasons.append(f"{name} contributed {weighted:.1f} pts (raw value: {raw:.2f}).")

    if not reasons:
        reasons.append(f"Telemetry measurements conform to expected baseline parameters for {level} status.")

    summary = f"Battery risk level is {level} with deterministic risk score {score:.1f}/100."

    if level in ("CRITICAL", "HIGH"):
        actions = [
            "Reduce battery load immediately to lower thermal dissipation.",
            "Inspect thermal cooling management and check physical cell connections.",
            "Maintain active monitoring and escalate to battery engineer if conditions persist.",
            "Advisory recommendation only; human operator verification required before any intervention.",
        ]
    elif level == "WARNING":
        actions = [
            "Increase telemetry logging cadence and inspect thermal dissipation.",
            "Review cycle trends and verify sensor calibration.",
            "Advisory recommendation only; human operator verification required before any intervention.",
        ]
    else:
        actions = [
            "Maintain standard operational telemetry monitoring.",
            "Advisory recommendation only; human operator verification required before any intervention.",
        ]

    return ExplanationOutput(
        summary=summary,
        reasons=reasons,
        recommended_actions=actions,
        confidence_note="Generated via deterministic baseline template (ground truth).",
    )


def generate_explanation(
    assessment: Dict[str, Any],
    client: Optional[LLMClient] = None,
    prompt_name: str = "explain_v3",
    max_retries: int = 2,
) -> ExplanationOutput:
    """Generate a safety explanation with self-correction retry and deterministic fallback.

    Args:
        assessment: Serialized RiskAssessment dictionary.
        client: LLMClient instance (defaults to configured LLM via get_llm_client).
        prompt_name: Name of prompt template YAML to load.
        max_retries: Number of regeneration attempts if validator finds issues.

    Returns:
        Validated ExplanationOutput object.
    """
    if client is None:
        client = get_llm_client()

    if isinstance(client, TemplateClient):
        return template_explanation(assessment)

    try:
        prompt = load_prompt(prompt_name)
    except Exception as exc:
        logger.warning(f"Failed to load prompt {prompt_name}: {exc}. Using template fallback.")
        return template_explanation(assessment)

    assessment_json = json.dumps(assessment, indent=2)
    messages = prompt.render_messages(assessment_json=assessment_json)

    last_failed_id: Optional[str] = None

    for attempt in range(1, max_retries + 2):
        t0 = time.time()
        try:
            raw_text = client.generate(
                messages,
                json_schema=prompt.output_schema,
                temperature=0.2,
                max_tokens=600,
            )
        except Exception as gen_exc:
            duration = time.time() - t0
            logger.warning(f"LLM generation failed on attempt {attempt}: {gen_exc}")
            _log_runtime_event(
                milestone="M10",
                attempt=attempt,
                status="FAIL",
                error_class="LLMGenerationError",
                error_excerpt=str(gen_exc)[:200],
                duration_s=duration,
            )
            # Fall back to template on generation failure
            _log_runtime_event(
                milestone="M10",
                attempt=attempt + 1,
                status="FIX",
                fix_summary="Fell back to deterministic template due to LLM error.",
            )
            return template_explanation(assessment)

        duration = time.time() - t0
        violations = validate_explanation(raw_text, assessment)

        if not violations:
            if attempt > 1:
                # Successfully healed!
                _log_runtime_event(
                    milestone="M10",
                    attempt=attempt,
                    status="PASS",
                    duration_s=duration,
                    linked_to=last_failed_id,
                )
            try:
                data = json.loads(raw_text)
                return ExplanationOutput(
                    summary=data.get("summary", ""),
                    reasons=data.get("reasons", []),
                    recommended_actions=data.get("recommended_actions", []),
                    confidence_note=data.get("confidence_note", f"Generated via {prompt_name}."),
                )
            except Exception:
                return template_explanation(assessment)

        # Violations present
        excerpt = "; ".join(v.message for v in violations)
        last_failed_id = _log_runtime_event(
            milestone="M10",
            attempt=attempt,
            status="FAIL",
            error_class="ExplanationValidationFailure",
            error_excerpt=excerpt[:300],
            duration_s=duration,
        )

        if attempt <= max_retries:
            # Append feedback message for regeneration
            feedback = (
                f"Your response had {len(violations)} validation violations:\n"
                + "\n".join(f"- {v.message}" for v in violations)
                + "\nPlease regenerate the explanation correcting these issues. "
                "Ensure strict JSON output conforming to the schema and quoting only provided numbers."
            )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": feedback})
        else:
            # Retries exhausted -> fallback to template
            _log_runtime_event(
                milestone="M10",
                attempt=attempt + 1,
                status="FIX",
                fix_summary="Fell back to deterministic template after max retries.",
                linked_to=last_failed_id,
            )
            return template_explanation(assessment)

    return template_explanation(assessment)

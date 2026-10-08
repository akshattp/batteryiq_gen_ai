"""Explanation validator and self-correction loop (Section 8.3).

Enforces schema compliance, numeric fidelity, risk-level consistency,
and absence of forbidden autonomous control directives.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


@dataclass
class Violation:
    """Represents a specific validation rule failure."""
    rule: str
    message: str
    severity: str = "ERROR"

    def to_dict(self) -> Dict[str, str]:
        return {"rule": self.rule, "message": self.message, "severity": self.severity}


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract floating point and integer numbers from string text.

    Excludes standard markdown list bullets (like '1. ', '2. ') or version strings.
    """
    # Clean text of markdown enumeration like '1. ', '2. ' at start of lines
    cleaned = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Find all signed numbers (integer or decimal)
    matches = re.finditer(r"(?<![a-zA-Z_])[-+]?(?:\d*\.\d+|\d+)(?![a-zA-Z_])", cleaned)
    numbers: List[float] = []
    for m in matches:
        try:
            val = float(m.group(0))
            numbers.append(val)
        except ValueError:
            continue
    return numbers


def collect_numbers_from_data(data: Any) -> Set[float]:
    """Recursively collect numeric values present in input data structure."""
    numbers: Set[float] = set()
    if isinstance(data, dict):
        for v in data.values():
            numbers.update(collect_numbers_from_data(v))
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            numbers.update(collect_numbers_from_data(item))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        numbers.add(float(data))
    elif isinstance(data, str):
        # Also extract numbers embedded in string fields
        for n in extract_numbers_from_text(data):
            numbers.add(n)
    return numbers


def is_number_grounded(num: float, allowed_numbers: Set[float], abs_tol: float = 0.1, rel_tol: float = 0.01) -> bool:
    """Check if number matches any allowed number within absolute or relative tolerance."""
    # Standard invariant scale constants allowed in explanations
    standard_constants = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0}
    if num in standard_constants:
        return True

    for allowed in allowed_numbers:
        # Check absolute tolerance
        if abs(num - allowed) <= abs_tol:
            return True
        # Check relative tolerance if non-zero
        if allowed != 0.0 and abs(num - allowed) / abs(allowed) <= rel_tol:
            return True

    return False


def validate_explanation(
    output_text: str,
    structured_input: Dict[str, Any],
    forbidden_phrases: Optional[List[str]] = None,
) -> List[Violation]:
    """Validate model explanation output against strict safety, schema, and fidelity rules.

    Args:
        output_text: Raw string output from LLM (expected JSON).
        structured_input: Dictionary of assessment telemetry provided to the prompt.
        forbidden_phrases: List of prohibited control commands.

    Returns:
        List of detected Violation objects (empty if output is 100% valid).
    """
    violations: List[Violation] = []

    # 1. JSON parsing and schema
    try:
        data = json.loads(output_text)
    except json.JSONDecodeError as exc:
        violations.append(Violation(rule="json_syntax", message=f"Invalid JSON: {exc}"))
        return violations

    if not isinstance(data, dict):
        violations.append(Violation(rule="schema_type", message="Expected top-level JSON object."))
        return violations

    for required_key in ("summary", "reasons", "recommended_actions"):
        if required_key not in data:
            violations.append(Violation(rule="schema_missing_key", message=f"Missing required key '{required_key}'"))

    if violations:
        return violations

    summary = str(data.get("summary", "")).strip()
    reasons = data.get("reasons", [])
    recommended_actions = data.get("recommended_actions", [])

    if not summary:
        violations.append(Violation(rule="empty_field", message="Field 'summary' must not be empty."))

    if not isinstance(reasons, list) or len(reasons) == 0:
        violations.append(Violation(rule="empty_field", message="Field 'reasons' must be a non-empty list of strings."))

    if not isinstance(recommended_actions, list) or len(recommended_actions) == 0:
        violations.append(
            Violation(rule="empty_field", message="Field 'recommended_actions' must be a non-empty list of strings.")
        )

    # 2. Forbidden control phrases
    phrases = forbidden_phrases or [
        "shut down",
        "shutdown",
        "disconnect",
        "emergency stop",
        "disable bms",
        "cut off",
        "trip breaker",
        "ignore",
        "override safety",
    ]
    text_lower = output_text.lower()
    for phrase in phrases:
        if phrase in text_lower:
            violations.append(
                Violation(
                    rule="forbidden_control_verb",
                    message=f"Output contains forbidden autonomous directive: '{phrase}'",
                )
            )

    # 3. Level consistency
    input_level = str(structured_input.get("risk_level", "")).upper()
    if input_level:
        summary_upper = summary.upper()
        # The expected risk level must appear in summary or reasons
        level_found = input_level in summary_upper or any(
            input_level in str(r).upper() for r in (reasons if isinstance(reasons, list) else [])
        )
        if not level_found:
            violations.append(
                Violation(
                    rule="level_inconsistency",
                    message=f"Output does not mention expected risk level '{input_level}'",
                )
            )

        # Contradiction checks
        if input_level in ("HIGH", "CRITICAL"):
            contradictions = ["SAFE", "NOMINAL STATE", "ALL SYSTEMS NORMAL", "HEALTHY"]
            for contra in contradictions:
                if contra in summary_upper:
                    violations.append(
                        Violation(
                            rule="contradictory_claim",
                            message=f"Claimed '{contra}' when risk level is {input_level}",
                        )
                    )

        if input_level == "NORMAL":
            severe_words = ["CRITICAL", "HIGH RISK", "SEVERE DANGER"]
            for sev in severe_words:
                if sev in summary_upper:
                    violations.append(
                        Violation(
                            rule="level_inconsistency",
                            message=f"Claimed '{sev}' when risk level is NORMAL",
                        )
                    )

    # 4. Numeric fidelity
    # Collect all numbers from input telemetry
    allowed_numbers = collect_numbers_from_data(structured_input)

    # Extract numbers from summary and reasons
    text_to_check = summary + "\n" + "\n".join(str(r) for r in reasons if isinstance(r, str))
    extracted_numbers = extract_numbers_from_text(text_to_check)

    for num in extracted_numbers:
        if not is_number_grounded(num, allowed_numbers):
            violations.append(
                Violation(
                    rule="numeric_hallucination",
                    message=f"Number {num} in explanation is not grounded in structured input telemetry.",
                )
            )

    # 5. Length limits
    if len(summary) > 500:
        violations.append(Violation(rule="length_limit", message=f"Summary length ({len(summary)}) exceeds 500 characters."))

    return violations


validate = validate_explanation

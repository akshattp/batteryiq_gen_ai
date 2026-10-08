"""tests/test_reports.py - Acceptance test for M15 Reports (Model Card & Autoheal Audit)."""

from __future__ import annotations

import re
from pathlib import Path


def test_reports_exist_and_have_no_placeholders():
    """Verify MODEL_CARD.md and AUTOHEAL_REPORT.md exist and have zero placeholders."""
    reports_dir = Path("reports")
    assert reports_dir.exists(), "reports directory does not exist"

    model_card = reports_dir / "MODEL_CARD.md"
    autoheal_report = reports_dir / "AUTOHEAL_REPORT.md"

    assert model_card.exists(), "reports/MODEL_CARD.md missing"
    assert autoheal_report.exists(), "reports/AUTOHEAL_REPORT.md missing"

    card_text = model_card.read_text(encoding="utf-8")
    heal_text = autoheal_report.read_text(encoding="utf-8")

    assert len(card_text) > 500, "MODEL_CARD.md is suspiciously short"
    assert len(heal_text) > 500, "AUTOHEAL_REPORT.md is suspiciously short"

    # Search for forbidden placeholder tokens
    placeholder_pattern = re.compile(r"\b(TODO|TBD|XX)\b")

    card_matches = placeholder_pattern.findall(card_text)
    heal_matches = placeholder_pattern.findall(heal_text)

    assert not card_matches, f"Placeholders found in MODEL_CARD.md: {card_matches}"
    assert not heal_matches, f"Placeholders found in AUTOHEAL_REPORT.md: {heal_matches}"

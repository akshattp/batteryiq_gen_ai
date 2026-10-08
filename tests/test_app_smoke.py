"""tests/test_app_smoke.py - Acceptance test for Streamlit Dashboard (M13 / Exp 8).

Verifies that every page loads with zero exceptions using streamlit.testing.v1.AppTest.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


PAGES = [
    "app/Home.py",
    "app/pages/1_Live_Monitor.py",
    "app/pages/2_Events.py",
    "app/pages/3_Ask_Guardian.py",
    "app/pages/4_WhatIf.py",
    "app/pages/5_Incidents.py",
    "app/pages/6_Model_Card.py",
    "app/pages/7_AutoHeal_Report.py",
]


@pytest.mark.parametrize("page_path", PAGES)
def test_dashboard_page_smoke(page_path: str):
    """Verify each page renders cleanly with no uncaught exceptions."""
    assert Path(page_path).exists(), f"Page file missing: {page_path}"

    abs_path = str(Path(page_path).resolve())
    at = AppTest.from_file(abs_path, default_timeout=30)
    at.run()

    if at.exception:
        exc_msgs = [str(e.value) for e in at.exception]
        pytest.fail(f"Page {page_path} raised exceptions: {exc_msgs}")

    assert not at.exception, f"Uncaught exception on page: {page_path}"

"""tools/autoheal.py - Development and Runtime Auto-Heal Harness.

Implements the auto-healing loop, execution logging, failure classification,
and fix recording specified in Section 10 of the implementation plan.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


LOG_PATH = Path("reports/autoheal/log.jsonl")
BLOCKED_PATH = Path("docs/BLOCKED.md")
DEFAULT_MAX_ATTEMPTS = 5


def ensure_log_dir() -> None:
    """Ensure destination directory for autoheal logs exists."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_logs() -> List[Dict[str, Any]]:
    """Read all log entries from the JSONL log file."""
    if not LOG_PATH.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_log(record: Dict[str, Any]) -> None:
    """Append a single record to the JSONL log file."""
    ensure_log_dir()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_event(
    *,
    source: str = "dev",
    milestone: str = "M0",
    attempt: int = 1,
    command: str = "",
    status: str = "PASS",
    exit_code: int = 0,
    error_class: Optional[str] = None,
    error_excerpt: Optional[str] = None,
    hypothesis: Optional[str] = None,
    files_changed: Optional[List[str]] = None,
    fix_summary: Optional[str] = None,
    duration_s: float = 0.0,
    linked_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to construct and append a log entry."""
    entry: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "milestone": milestone,
        "attempt": attempt,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "error_class": error_class,
        "error_excerpt": error_excerpt,
        "hypothesis": hypothesis,
        "files_changed": files_changed or [],
        "fix_summary": fix_summary,
        "duration_s": round(duration_s, 2),
        "linked_to": linked_to,
    }
    append_log(entry)
    return entry


def classify_error(output: str) -> str:
    """Classify standard Python / shell errors into known taxonomies."""
    out_lower = output.lower()
    if "importerror" in out_lower or "modulenotfounderror" in out_lower:
        return "ImportError"
    if "syntaxerror" in out_lower or "indentationerror" in out_lower:
        return "SyntaxError"
    if "assertionerror" in out_lower or "assert " in out_lower:
        return "AssertionError"
    if "shapemismatch" in out_lower or "shape mismatch" in out_lower or "size mismatch" in out_lower:
        return "ShapeMismatch"
    if "keyerror" in out_lower:
        return "KeyError"
    if "connectionerror" in out_lower or "networkerror" in out_lower or "httperror" in out_lower or "socket" in out_lower:
        return "NetworkError"
    if "memoryerror" in out_lower or "oom" in out_lower or "out of memory" in out_lower:
        return "OOM"
    if "timeout" in out_lower or "timed out" in out_lower:
        return "Timeout"
    if "typeerror" in out_lower:
        return "TypeError"
    if "valueerror" in out_lower:
        return "ValueError"
    return "Other"


def extract_error_excerpt(output: str, max_lines: int = 40) -> str:
    """Extract up to max_lines of error excerpt from process output."""
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def get_current_attempt_count(milestone: str, command: str) -> int:
    """Count how many attempts have occurred for this milestone and command."""
    logs = read_logs()
    count = 0
    for r in logs:
        if r.get("milestone") == milestone and r.get("command") == command and r.get("status") in ("FAIL", "PASS"):
            count += 1
    return count + 1


def get_last_failed_entry(milestone: str) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent FAIL record for a given milestone."""
    logs = read_logs()
    for r in reversed(logs):
        if r.get("milestone") == milestone and r.get("status") == "FAIL":
            return r
    for r in reversed(logs):
        if r.get("status") == "FAIL":
            return r
    return None


def record_fix(
    milestone: str,
    hypothesis: str,
    files_changed: List[str],
    summary: str,
) -> Dict[str, Any]:
    """Record a FIX action before retrying a failed command."""
    last_fail = get_last_failed_entry(milestone)
    linked_id = last_fail.get("id") if last_fail else None
    cmd = last_fail.get("command", "") if last_fail else ""
    attempt = last_fail.get("attempt", 1) if last_fail else 1

    entry = log_event(
        source="dev",
        milestone=milestone,
        attempt=attempt,
        command=cmd,
        status="FIX",
        exit_code=0,
        hypothesis=hypothesis,
        files_changed=files_changed,
        fix_summary=summary,
        linked_to=linked_id,
    )
    print(f"[autoheal] FIX recorded (linked to {linked_id})")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Files: {files_changed}")
    print(f"  Summary: {summary}")
    return entry


def mark_blocked(milestone: str, command: str, error_class: str, excerpt: str, attempts: int) -> None:
    """Write docs/BLOCKED.md when max heal attempts are exceeded."""
    BLOCKED_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# BLOCKED: Milestone {milestone} Exceeded Auto-Heal Cap

- **Timestamp**: {datetime.now(timezone.utc).isoformat()}
- **Milestone**: {milestone}
- **Command**: `{command}`
- **Attempts**: {attempts}
- **Error Class**: {error_class}

## Error Excerpt
```
{excerpt}
```

## Diagnosis and Next Step
Maximum autoheal attempts reached. Human review or fallback ladder activation required.
"""
    with open(BLOCKED_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[autoheal] BLOCKED threshold reached ({attempts} attempts). Wrote {BLOCKED_PATH}")


def run_command_under_autoheal(
    command: str,
    milestone: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> int:
    """Run a shell command, time it, capture output, and log result."""
    attempt = get_current_attempt_count(milestone, command)
    print(f"[autoheal] Running milestone={milestone} attempt={attempt}/{max_attempts}: {command}")

    last_fail = get_last_failed_entry(milestone)
    linked_id = last_fail.get("id") if (last_fail and attempt > 1) else None

    start_t = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        duration_s = time.time() - start_t
        output = proc.stdout or ""
        exit_code = proc.returncode
    except Exception as exc:
        duration_s = time.time() - start_t
        output = str(exc)
        exit_code = 1

    try:
        print(output)
    except Exception:
        try:
            sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")
        except Exception:
            pass

    if exit_code == 0:
        log_event(
            source="dev",
            milestone=milestone,
            attempt=attempt,
            command=command,
            status="PASS",
            exit_code=0,
            duration_s=duration_s,
            linked_to=linked_id,
        )
        print(f"[autoheal] PASS ({duration_s:.2f}s) for {command}")
        return 0
    else:
        err_class = classify_error(output)
        excerpt = extract_error_excerpt(output, 40)
        log_event(
            source="dev",
            milestone=milestone,
            attempt=attempt,
            command=command,
            status="FAIL",
            exit_code=exit_code,
            error_class=err_class,
            error_excerpt=excerpt,
            duration_s=duration_s,
            linked_to=linked_id,
        )
        print("\n" + "=" * 60)
        print("HEAL REQUIRED: diagnose, patch, then re-run")
        print(f"Error class: {err_class} (Exit code: {exit_code})")
        print("=" * 60 + "\n")

        if attempt >= max_attempts:
            mark_blocked(milestone, command, err_class, excerpt, attempt)
            return 3

        return exit_code


def run_selftest() -> int:
    """Execute automated self-test of autoheal loop."""
    print("[autoheal] Starting self-test cycle...")
    temp_script = Path("tools/_selftest_tmp.py")

    try:
        # Step 1: Write deliberate buggy script
        temp_script.write_text("def test_dummy():\n    assert 1 == 2, 'Deliberate test failure'\n\ntest_dummy()\n", encoding="utf-8")
        test_cmd = f"python {temp_script}"

        # Run command expecting failure
        print("[autoheal] Step 1: Running deliberately failing test...")
        rc1 = run_command_under_autoheal(test_cmd, milestone="M0_SELFTEST", max_attempts=5)
        if rc1 == 0:
            print("[autoheal] Selftest error: Expected step 1 to fail, but it passed.")
            return 1

        # Step 2: Record fix
        print("[autoheal] Step 2: Recording fix...")
        record_fix(
            milestone="M0_SELFTEST",
            hypothesis="Deliberate assertion 1 == 2 failed as designed",
            files_changed=[str(temp_script)],
            summary="Corrected assertion to 1 == 1",
        )

        # Step 3: Patch the script
        temp_script.write_text("def test_dummy():\n    assert 1 == 1, 'Self test pass'\n\ntest_dummy()\n", encoding="utf-8")

        # Step 4: Re-run expecting pass
        print("[autoheal] Step 3: Re-running patched test...")
        rc2 = run_command_under_autoheal(test_cmd, milestone="M0_SELFTEST", max_attempts=5)
        if rc2 != 0:
            print("[autoheal] Selftest error: Expected step 3 to pass, but failed.")
            return 1

        # Step 5: Verify log contents
        logs = [r for r in read_logs() if r.get("milestone") == "M0_SELFTEST"]
        statuses = [r.get("status") for r in logs]
        print(f"[autoheal] Verified selftest records: {statuses}")
        if statuses[-3:] != ["FAIL", "FIX", "PASS"]:
            print(f"[autoheal] Selftest warning: Sequence was {statuses[-3:]}, expected ['FAIL', 'FIX', 'PASS']")
            return 1

        print("[autoheal] Selftest SUCCESSFUL! The auto-heal loop is functioning as specified.")
        return 0
    finally:
        if temp_script.exists():
            temp_script.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Heal Harness for AI Battery Safety Guardian")
    parser.add_argument("--milestone", type=str, default="M0", help="Milestone identifier (e.g. M0, M1)")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="Max retry attempts before blocking")
    parser.add_argument("--record-fix", action="store_true", help="Record a diagnosis and patch before re-running")
    parser.add_argument("--hypothesis", type=str, default="", help="Hypothesis for root cause of failure")
    parser.add_argument("--files", type=str, default="", help="Comma-separated or space-separated list of modified files")
    parser.add_argument("--summary", type=str, default="", help="Summary of changes made")
    parser.add_argument("--selftest", action="store_true", help="Run automated self-test of the auto-heal harness")
    parser.add_argument("command", nargs="*", help="Command to execute")

    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    if args.record_fix:
        files = [f.strip() for f in re.split(r"[, ]+", args.files) if f.strip()]
        record_fix(
            milestone=args.milestone,
            hypothesis=args.hypothesis or "Auto-heal fix applied",
            files_changed=files,
            summary=args.summary or "Code modified to resolve failure",
        )
        sys.exit(0)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmd_str = " ".join(args.command)
    code = run_command_under_autoheal(cmd_str, milestone=args.milestone, max_attempts=args.max_attempts)
    sys.exit(code)


if __name__ == "__main__":
    main()

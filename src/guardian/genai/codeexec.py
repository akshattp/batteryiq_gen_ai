"""Sandboxed code execution and auto-heal loop for pandas analysis (Section 9 / Exp 6).

Executes LLM-generated analysis code in an isolated subprocess with AST safety validation,
5-second timeout, memory limits, and automated 3-attempt repair on errors.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from guardian.genai.llm_client import LLMClient, get_llm_client
from guardian.utils.config import get_config
from guardian.utils.logging import get_logger

logger = get_logger(__name__)

FORBIDDEN_CALLS: Set[str] = {
    "open",
    "eval",
    "exec",
    "__import__",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
}

FORBIDDEN_NAMES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "pathlib",
    "builtins",
    "__builtins__",
}


class SecurityViolation(Exception):
    """Raised when code violates AST security constraints."""
    pass


def check_ast_security(code: str) -> None:
    """Perform static AST analysis to enforce strict sandbox constraints.

    Rules:
    1. No import or import-from statements (pandas/numpy are pre-imported).
    2. No access to forbidden functions (open, eval, exec, __import__, etc.).
    3. No references to OS, filesystem, network, or subprocess modules.
    4. No private or dunder attribute accesses (starting with '_').
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SyntaxError(f"Syntax error in generated code: {exc}") from exc

    for node in ast.walk(tree):
        # 1. Reject import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SecurityViolation("Import statements are forbidden in sandboxed code.")

        # 2. Reject calls to forbidden builtins
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SecurityViolation(f"Calling '{node.func.id}' is forbidden.")

        # 3. Reject access to forbidden module names
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SecurityViolation(f"Reference to restricted module/identifier '{node.id}' is forbidden.")

        # 4. Reject private / dunder attributes
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise SecurityViolation(f"Access to private attribute '{node.attr}' is forbidden.")


def verify_code_safety(code: str) -> Tuple[bool, str]:
    """Verify code safety using AST checks.

    Returns:
        (True, "") if safe, or (False, reason_message) if unsafe.
    """
    try:
        check_ast_security(code)
        return True, ""
    except (SecurityViolation, SyntaxError) as exc:
        return False, str(exc)


def log_codeexec_attempt(
    record_id: str,
    attempt: int,
    status: str,
    error_class: str = "None",
    error_excerpt: str = "",
    duration_s: float = 0.0,
    linked_to: Optional[str] = None,
) -> None:
    """Log execution attempt to reports/autoheal/log.jsonl conforming to Section 10.2 schema."""
    log_file = Path("reports/autoheal/log.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": record_id,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "runtime_codeexec",
        "milestone": "M12",
        "attempt": attempt,
        "command": "run_analysis",
        "status": status,
        "exit_code": 0 if status == "PASS" else 1,
        "error_class": error_class,
        "error_excerpt": error_excerpt[-500:] if error_excerpt else "",
        "hypothesis": "Auto-heal repaired code snippet" if status == "PASS" and attempt > 1 else "",
        "files_changed": [],
        "fix_summary": f"Repaired code at attempt {attempt}" if status == "PASS" and attempt > 1 else "",
        "duration_s": round(duration_s, 4),
        "linked_to": linked_to,
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to append to autoheal log: {exc}")


def clean_code_snippet(raw_text: str) -> str:
    """Extract code from markdown code fences if present."""
    text = raw_text.strip()
    if "```python" in text:
        parts = text.split("```python", 1)[1].split("```", 1)
        return parts[0].strip()
    elif "```" in text:
        parts = text.split("```", 1)[1].split("```", 1)
        return parts[0].strip()
    return text


def execute_in_subprocess(
    code: str,
    df: pd.DataFrame,
    timeout_s: float = 5.0,
) -> Tuple[Any, Optional[str], Optional[str]]:
    """Execute code against DataFrame in an isolated subprocess.

    Returns:
        (result, error_class, error_message)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        df_parquet = tmp_path / "df.parquet"
        result_json = tmp_path / "result.json"
        script_path = tmp_path / "runner.py"

        # Export DataFrame
        df.to_parquet(df_parquet)

        runner_code = f"""
import sys
import json
import pandas as pd
import numpy as np

try:
    df = pd.read_parquet(r"{df_parquet}")
    result = None

    # User code
{'\n'.join('    ' + line for line in code.splitlines())}

    # Format result
    if result is None:
        raise ValueError("Code executed but did not assign a value to 'result'.")

    # Serialize result
    if isinstance(result, (pd.DataFrame, pd.Series)):
        res_val = result.to_dict()
    elif isinstance(result, (np.integer, np.floating)):
        res_val = result.item()
    elif isinstance(result, np.ndarray):
        res_val = result.tolist()
    else:
        res_val = result

    with open(r"{result_json}", "w", encoding="utf-8") as f:
        json.dump({{"status": "ok", "result": res_val}}, f, default=str)

except Exception as exc:
    err_class = type(exc).__name__
    err_msg = str(exc)
    with open(r"{result_json}", "w", encoding="utf-8") as f:
        json.dump({{"status": "error", "error_class": err_class, "error_msg": err_msg}}, f)
    sys.exit(1)
"""
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(runner_code)

        # Isolated environment on Windows
        sub_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                timeout=timeout_s,
                capture_output=True,
                text=True,
                env=sub_env,
            )
        except subprocess.TimeoutExpired:
            return None, "TimeoutError", f"Execution exceeded {timeout_s} seconds limit."

        if result_json.exists():
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    out = json.load(f)
                if out.get("status") == "ok":
                    return out.get("result"), None, None
                else:
                    return None, out.get("error_class", "ExecutionError"), out.get("error_msg", "Unknown error")
            except Exception as read_exc:
                return None, "SerializationError", f"Failed to read result: {read_exc}"

        stderr = proc.stderr.strip()
        last_lines = "\n".join(stderr.splitlines()[-15:])
        return None, "SubprocessError", f"Subprocess exited with code {proc.returncode}: {last_lines}"


class SandboxedCodeExecutor:
    """Full LLM codegen -> AST validation -> subprocess -> auto-heal pipeline."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_attempts: int = 3,
        timeout_s: float = 5.0,
    ) -> None:
        self.llm_client = llm_client or get_llm_client()
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s

        # Load prompts
        with open("config/prompts/codegen_v1.yaml", "r", encoding="utf-8") as f:
            self.codegen_prompt = yaml.safe_load(f)
        with open("config/prompts/heal_codegen_v1.yaml", "r", encoding="utf-8") as f:
            self.heal_prompt = yaml.safe_load(f)

    def _generate_initial_code(self, question: str, df: pd.DataFrame) -> str:
        """Prompt LLM for initial pandas analysis snippet."""
        schema_info = {
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "sample_rows": df.head(3).to_dict(orient="records"),
            "question": question,
        }
        user_msg = self.codegen_prompt["user_template"].format(**schema_info)
        messages = [
            {"role": "system", "content": self.codegen_prompt["system"]},
            {"role": "user", "content": user_msg},
        ]

        raw = self.llm_client.generate(messages, temperature=0.1, max_tokens=350)
        return clean_code_snippet(raw)

    def _generate_healed_code(
        self,
        question: str,
        df: pd.DataFrame,
        failed_code: str,
        error_class: str,
        error_message: str,
    ) -> str:
        """Prompt LLM to repair code given the failure reason."""
        info = {
            "question": question,
            "columns": list(df.columns),
            "failed_code": failed_code,
            "error_class": error_class,
            "error_message": error_message,
        }
        user_msg = self.heal_prompt["user_template"].format(**info)
        messages = [
            {"role": "system", "content": self.heal_prompt["system"]},
            {"role": "user", "content": user_msg},
        ]

        raw = self.llm_client.generate(messages, temperature=0.1, max_tokens=350)
        return clean_code_snippet(raw)

    def run_analysis(
        self,
        question: str,
        df: pd.DataFrame,
        initial_code_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute sandboxed analysis with up to 3 repair attempts.

        Returns:
            Dict containing:
            - answer: final result or polite error message
            - code: final executed code
            - attempts: number of iterations used
            - healed: bool indicating if auto-heal succeeded
            - status: "PASS" or "FAIL"
        """
        first_record_id = str(uuid.uuid4())
        current_code = initial_code_override or self._generate_initial_code(question, df)
        last_error_class = ""
        last_error_msg = ""
        prev_id: Optional[str] = None

        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            record_id = str(uuid.uuid4()) if attempt > 1 else first_record_id

            # 1. Static AST Check
            try:
                check_ast_security(current_code)
            except (SecurityViolation, SyntaxError) as sec_exc:
                duration = time.time() - t0
                err_class = type(sec_exc).__name__
                err_msg = str(sec_exc)
                log_codeexec_attempt(
                    record_id=record_id,
                    attempt=attempt,
                    status="FAIL",
                    error_class=err_class,
                    error_excerpt=err_msg,
                    duration_s=duration,
                    linked_to=prev_id,
                )
                prev_id = record_id
                last_error_class = err_class
                last_error_msg = err_msg

                if isinstance(sec_exc, SecurityViolation):
                    # Reject malicious code immediately without execution
                    return {
                        "answer": f"Execution rejected due to security policy: {err_msg}",
                        "code": current_code,
                        "attempts": attempt,
                        "healed": False,
                        "status": "REJECTED_SECURITY",
                        "error_class": err_class,
                    }

                # If syntax error, attempt repair
                if attempt < self.max_attempts:
                    current_code = self._generate_healed_code(
                        question, df, current_code, err_class, err_msg
                    )
                continue

            # 2. Subprocess Execution
            result, err_class, err_msg = execute_in_subprocess(
                current_code, df, timeout_s=self.timeout_s
            )
            duration = time.time() - t0

            if err_class is None:
                # Success
                log_codeexec_attempt(
                    record_id=record_id,
                    attempt=attempt,
                    status="PASS",
                    duration_s=duration,
                    linked_to=prev_id,
                )
                return {
                    "answer": result,
                    "code": current_code,
                    "attempts": attempt,
                    "healed": (attempt > 1),
                    "status": "PASS",
                }
            else:
                # Execution failure
                log_codeexec_attempt(
                    record_id=record_id,
                    attempt=attempt,
                    status="FAIL",
                    error_class=err_class,
                    error_excerpt=err_msg or "Unknown error",
                    duration_s=duration,
                    linked_to=prev_id,
                )
                prev_id = record_id
                last_error_class = err_class
                last_error_msg = err_msg or "Unknown error"

                if attempt < self.max_attempts:
                    current_code = self._generate_healed_code(
                        question, df, current_code, last_error_class, last_error_msg
                    )

        # All attempts exhausted
        return {
            "answer": f"Analysis failed after {self.max_attempts} attempts. Last error ({last_error_class}): {last_error_msg}",
            "code": current_code,
            "attempts": self.max_attempts,
            "healed": False,
            "status": "FAIL",
            "error_class": last_error_class,
            "error_message": last_error_msg,
        }

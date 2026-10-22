"""
Pylint and Bandit static analysis runners.
"""
import os
import json
import subprocess
import logging
from typing import Optional

log = logging.getLogger(__name__)


def run_pylint(file_path: str) -> list[dict]:
    """
    Run pylint on a Python file and return a list of issue dicts.
    Each dict: {line, column, type, symbol, message, file}
    """
    issues = []
    try:
        result = subprocess.run(
            [
                "python", "-m", "pylint",
                "--output-format=json",
                "--disable=C0114,C0115,C0116",  # ignore missing docstrings for speed
                file_path,
            ],
            capture_output=True, text=True, timeout=30
        )
        raw = result.stdout.strip()
        if raw:
            data = json.loads(raw)
            for item in data:
                issues.append({
                    "file":    item.get("path", file_path),
                    "line":    item.get("line", 0),
                    "column":  item.get("column", 0),
                    "type":    item.get("type", ""),
                    "symbol":  item.get("symbol", ""),
                    "message": item.get("message", ""),
                    "tool":    "pylint",
                })
    except Exception as e:
        log.warning("Pylint failed for %s: %s", file_path, e)
    return issues


def run_bandit(file_path: str) -> list[dict]:
    """
    Run bandit on a Python file and return security issues.
    Each dict: {line, severity, confidence, test_id, issue_text, file}
    """
    issues = []
    try:
        result = subprocess.run(
            ["python", "-m", "bandit", "-f", "json", file_path],
            capture_output=True, text=True, timeout=30
        )
        raw = result.stdout.strip()
        if raw:
            data = json.loads(raw)
            for item in data.get("results", []):
                issues.append({
                    "file":      item.get("filename", file_path),
                    "line":      item.get("line_number", 0),
                    "column":    0,
                    "type":      "security",
                    "symbol":    item.get("test_id", ""),
                    "message":   item.get("issue_text", ""),
                    "severity":  item.get("issue_severity", "LOW"),
                    "confidence":item.get("issue_confidence", "LOW"),
                    "tool":      "bandit",
                })
    except Exception as e:
        log.warning("Bandit failed for %s: %s", file_path, e)
    return issues


def run_eslint(file_path: str, config: Optional[str] = None) -> list[dict]:
    """
    Run ESLint on a JS/TS file (requires npx / globally installed ESLint).
    Returns list of issues.
    """
    issues = []
    cmd = ["npx", "--yes", "eslint", "--format", "json"]
    if config:
        cmd += ["--config", config]
    cmd.append(file_path)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        raw = result.stdout.strip()
        if raw:
            data = json.loads(raw)
            for file_result in data:
                for msg in file_result.get("messages", []):
                    issues.append({
                        "file":    file_result.get("filePath", file_path),
                        "line":    msg.get("line", 0),
                        "column":  msg.get("column", 0),
                        "type":    "error" if msg.get("severity") == 2 else "warning",
                        "symbol":  msg.get("ruleId", ""),
                        "message": msg.get("message", ""),
                        "tool":    "eslint",
                    })
    except Exception as e:
        log.warning("ESLint failed for %s: %s", file_path, e)
    return issues


def run_static_analysis(file_path: str, config: Optional[str] = None) -> list[dict]:
    """
    Auto-detect file type and run the appropriate static analyser.
    Always runs independent of the ML risk tier.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".py":
        issues = run_pylint(file_path) + run_bandit(file_path)
    elif ext in (".js", ".ts", ".jsx", ".tsx"):
        issues = run_eslint(file_path, config)
    else:
        issues = []
    return issues

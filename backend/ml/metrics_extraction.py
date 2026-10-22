"""
Code metrics extraction for real Python/JS repositories using radon + AST.

For PROMISE CSV data this module is bypassed — the CSV columns ARE the metrics.
For real git repos, this extracts: LOC, cyclomatic complexity, coupling,
number of methods, and code churn.
"""
import os
import ast
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LOC + Cyclomatic Complexity via radon
# ---------------------------------------------------------------------------

def extract_radon_metrics(file_path: str) -> dict:
    """
    Run radon cc and radon raw on a Python file.
    Returns {loc, cyclomatic_complexity, num_methods}.
    """
    metrics = {"loc": 0, "cyclomatic_complexity": 0, "num_methods": 0}
    try:
        # Raw metrics (LOC)
        raw_result = subprocess.run(
            ["python", "-m", "radon", "raw", "-j", file_path],
            capture_output=True, text=True, timeout=15
        )
        if raw_result.returncode == 0 and raw_result.stdout.strip():
            raw_data = json.loads(raw_result.stdout)
            for path_key, data in raw_data.items():
                metrics["loc"] = data.get("loc", 0)
                break

        # CC (cyclomatic complexity)
        cc_result = subprocess.run(
            ["python", "-m", "radon", "cc", "-j", "-a", file_path],
            capture_output=True, text=True, timeout=15
        )
        if cc_result.returncode == 0 and cc_result.stdout.strip():
            cc_data = json.loads(cc_result.stdout)
            for path_key, blocks in cc_data.items():
                if isinstance(blocks, list):
                    complexities = [b.get("complexity", 1) for b in blocks]
                    metrics["cyclomatic_complexity"] = max(complexities) if complexities else 0
                    metrics["num_methods"] = len(complexities)
                break
    except Exception as e:
        log.debug("radon failed for %s: %s", file_path, e)

    return metrics


# ---------------------------------------------------------------------------
# Coupling via AST import analysis
# ---------------------------------------------------------------------------

def extract_coupling(file_path: str, project_root: str) -> int:
    """
    Count how many other project files this file imports (afferent/efferent coupling).
    Uses the Python AST — works without executing the file.
    """
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return 0

    # Collect all import targets
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    # Count how many of those correspond to local project files
    py_files = {
        Path(f).stem
        for f in Path(project_root).rglob("*.py")
        if f != file_path
    }
    coupling = len(imports & py_files)
    return coupling


# ---------------------------------------------------------------------------
# Code churn via git log
# ---------------------------------------------------------------------------

def extract_churn(file_path: str, repo_root: str, n_commits: int = 100) -> int:
    """
    Count how many of the last N commits touched this file.
    Returns 0 if git is unavailable or the file is not in a repo.
    """
    try:
        rel_path = os.path.relpath(file_path, repo_root)
        result = subprocess.run(
            ["git", "log", f"--max-count={n_commits}", "--oneline", "--", rel_path],
            capture_output=True, text=True, cwd=repo_root, timeout=15
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().splitlines() if l]
            return len(lines)
    except Exception as e:
        log.debug("git log failed: %s", e)
    return 0


# ---------------------------------------------------------------------------
# Test coverage via coverage.py JSON report
# ---------------------------------------------------------------------------

def extract_coverage(file_path: str, coverage_json: Optional[str] = None) -> Optional[float]:
    """
    Read line coverage % for a file from a coverage.py JSON report.
    Run `coverage json -o coverage.json` first.
    Returns None if unavailable.
    """
    if not coverage_json or not os.path.exists(coverage_json):
        return None
    try:
        with open(coverage_json) as f:
            data = json.load(f)
        files = data.get("files", {})
        # Try exact path, then basename match
        for path_key, file_data in files.items():
            if path_key == file_path or os.path.basename(path_key) == os.path.basename(file_path):
                summary = file_data.get("summary", {})
                pct = summary.get("percent_covered", None)
                return float(pct) if pct is not None else None
    except Exception as e:
        log.debug("Coverage parse failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Dependency graph (for hotspot centrality)
# ---------------------------------------------------------------------------

def build_dependency_graph(project_root: str):
    """
    Build a directed graph of Python imports using networkx.
    Returns a networkx.DiGraph where nodes are file paths.
    """
    try:
        import networkx as nx
    except ImportError:
        log.warning("networkx not installed — hotspot centrality unavailable.")
        return None

    G = nx.DiGraph()
    py_files = list(Path(project_root).rglob("*.py"))

    # Map stem → file path
    stem_map = {f.stem: str(f) for f in py_files}

    for f in py_files:
        fstr = str(f)
        G.add_node(fstr)
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except Exception:
            continue

        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets.append(node.module.split(".")[0])

            for t in targets:
                if t in stem_map:
                    G.add_edge(fstr, stem_map[t])

    return G


def compute_centrality(G, file_path: str) -> float:
    """
    Return in-degree centrality of a node (how many files depend on it).
    Normalised to [0, 1].
    """
    if G is None:
        return 0.0
    try:
        import networkx as nx
        centrality = nx.in_degree_centrality(G)
        return float(centrality.get(file_path, 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# All-in-one extractor for a real Python repo
# ---------------------------------------------------------------------------

def extract_all_metrics(file_path: str, project_root: str, coverage_json: Optional[str] = None) -> dict:
    """
    Run all extractors for a single Python file.
    Returns a dict ready for FileMetrics insertion.
    """
    r = extract_radon_metrics(file_path)
    coupling = extract_coupling(file_path, project_root)
    churn = extract_churn(file_path, project_root)
    coverage_pct = extract_coverage(file_path, coverage_json)

    return {
        "file_path": file_path,
        "loc": r.get("loc", 0),
        "cyclomatic_complexity": r.get("cyclomatic_complexity", 0),
        "num_methods": r.get("num_methods", 0),
        "coupling": coupling,
        "churn_count": churn,
        "test_coverage_pct": coverage_pct,
    }

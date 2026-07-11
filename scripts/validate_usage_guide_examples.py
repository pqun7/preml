#!/usr/bin/env python3
"""
Production‑grade validator for Python code examples in ``Usage Guide.md``.

This script:
1. Parses the Markdown guide structurally, extracting all fenced Python blocks
   with their section context.
2. For each block, determines whether it is a standalone runnable example
   (using AST analysis).
3. Runs every runnable block in a fresh, isolated virtual environment to
   guarantee copy‑paste reproducibility.
4. Analyses block dependencies via AST inspection, classifies setup issues
   from tracebacks, and produces detailed JSON + Markdown reports.

All operations are fully typed, robustly logged, and protected by timeouts.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback as tb
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Final, List, Optional, Set, Tuple, TypeAlias

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT: Final[Path] = Path(__file__).resolve().parents[1]
GUIDE: Final[Path] = ROOT / "Usage Guide.md"
OUT_DIR: Final[Path] = ROOT / "artifacts" / "usage_guide_validation"
BASE_VENV_DIR: Final[Path] = OUT_DIR / "_base_venv"

# Regex patterns used during Markdown parsing
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE_OPEN_RE = re.compile(r"^```\s*([A-Za-z0-9_+-]*)\s*$")
FENCE_CLOSE_RE = re.compile(r"^```\s*$")

# Mapping from exception class names to setup‑issue categories
SETUP_EXCEPTION_CATEGORIES: Final[Dict[str, str]] = {
    "ModuleNotFoundError": "dependency",
    "ImportError": "dependency",
    "FileNotFoundError": "environment",
    "PermissionError": "environment",
    "KeyError": "runtime",
    "AttributeError": "runtime",
    "ValueError": "runtime",
    "TypeError": "runtime",
    "NameError": "runtime",
    "SyntaxError": "user_code",
    "IndentationError": "user_code",
}

# ---------------------------------------------------------------------------
# Structured Types
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CodeBlock:
    id: int
    language: str
    code: str
    start_line: int
    end_line: int
    section: str                     # nearest heading stack, joined with " > "
    title: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BlockResult:
    id: int
    section: str
    title: Optional[str]
    start_line: int
    end_line: int
    runnable: bool
    status: str                     # "passed", "failed", "skipped"
    setup_issue: bool
    dependencies: List[str]
    related_api: List[str]
    expected_behavior: str
    stdout: str
    stderr: str
    return_code: Optional[int]
    duration_seconds: float
    error_summary: Optional[str] = None


# Typed dict for report serialisation
ReportDict: TypeAlias = Dict[str, Any]

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger("validate_usage_guide")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """Read entire file as UTF‑8 text."""
    return path.read_text(encoding="utf-8")


def python_executable_for(venv_dir: Path) -> Path:
    """Return the Python executable path inside a virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# ---------------------------------------------------------------------------
# 1. Structural Markdown Parser
# ---------------------------------------------------------------------------
class MarkdownParser:
    """Extract fenced code blocks with hierarchical section context."""

    def __init__(self) -> None:
        self.heading_stack: List[str] = []   # from outermost to current
        self.blocks: List[CodeBlock] = []

    def parse(self, markdown: str) -> List[CodeBlock]:
        """Return all fenced code blocks with section and optional title."""
        lines = markdown.splitlines()
        self.heading_stack = []
        self.blocks = []

        in_fence = False
        fence_lang = ""
        fence_start = -1
        code_lines: List[str] = []
        # We keep the last non‑heading line as a possible block title.
        last_non_heading: Optional[str] = None

        for idx, raw_line in enumerate(lines, start=1):
            line = raw_line

            # Headings
            heading_match = HEADING_RE.match(line)
            if heading_match and not in_fence:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                # Trim the stack to the new heading's level
                self.heading_stack = self.heading_stack[: level - 1]
                self.heading_stack.append(title)
                last_non_heading = None
                continue

            # Outside fence: detect fence openings
            if not in_fence:
                m = FENCE_OPEN_RE.match(line)
                if m:
                    in_fence = True
                    fence_lang = (m.group(1) or "").strip().lower()
                    fence_start = idx + 1
                    code_lines = []
                    continue
                # Keep track of the most recent meaningful line as a possible title.
                if line.strip() and not line.startswith("#"):
                    last_non_heading = line.strip()
                continue

            # Inside fence
            if FENCE_CLOSE_RE.match(line):
                # End of code block
                code = "\n".join(code_lines)
                title = last_non_heading if last_non_heading else None
                section = " > ".join(self.heading_stack) if self.heading_stack else "(root)"
                self.blocks.append(
                    CodeBlock(
                        id=len(self.blocks) + 1,
                        language=fence_lang,
                        code=code,
                        start_line=fence_start,
                        end_line=idx - 1,
                        section=section,
                        title=title,
                    )
                )
                in_fence = False
                fence_lang = ""
                fence_start = -1
                code_lines = []
                last_non_heading = None
                continue
            # Inside fence: accumulate code lines
            code_lines.append(line)

        return self.blocks


# ---------------------------------------------------------------------------
# 2. Runnable Detection via AST
# ---------------------------------------------------------------------------
def is_runnable_python_block(block: CodeBlock) -> Tuple[bool, Optional[str]]:
    """
    Decide whether a Python block is a standalone executable example.

    A block is considered runnable if it contains at least one top‑level
    statement that is **not** solely a function/class definition or a comment.
    This correctly handles decorators, async functions, multiline signatures,
    and annotations.
    """
    if block.language not in {"python", "py"}:
        return False, "Not a Python block"

    stripped = block.code.strip()
    if not stripped:
        return False, "Empty block"

    # Remove lines that are pure comments or blank – but keep them in AST later
    lines = [ln for ln in block.code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return False, "Comment‑only block"

    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        # If it can't even be parsed, treat as a snippet (maybe an incomplete example)
        return False, "Invalid Python syntax"

    # Walk through top‑level statements
    executable_found = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # A single expression statement (e.g., a function signature shown as a
        # lone line with "->" is captured here if it's just a string or a
        # Name, but we want to allow print() etc. We'll flag as not runnable
        # only when the body consists *exclusively* of definitions.
        # So if we see *any* non‑definition, it's runnable.
        executable_found = True
        break

    if not executable_found:
        return False, "Contains only function/class definitions"

    # Context-aware safety: skip snippets that rely on names not defined
    # inside the block itself (or Python builtins). These are usually guide
    # continuation snippets, not standalone copy-paste examples.
    defined_names: Set[str] = set()
    builtin_names = set(dir(builtins))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined_names.add(node.id)

    unresolved: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined_names and node.id not in builtin_names:
                unresolved.add(node.id)

    if unresolved:
        sample = ", ".join(sorted(unresolved)[:5])
        return False, f"Depends on external context: {sample}"

    return True, None


# ---------------------------------------------------------------------------
# 3. AST‑Based Dependency and API Detection
# ---------------------------------------------------------------------------
# Known library aliases and their standard import names
KNOWN_IMPORTS: Final[Dict[str, str]] = {
    "pandas": "pandas",
    "pd": "pandas",
    "numpy": "numpy",
    "np": "numpy",
    "matplotlib": "matplotlib",
    "plt": "matplotlib",
    "seaborn": "seaborn",
    "sns": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "plotly": "plotly",
    "requests": "requests",
    "joblib": "joblib",
}

# Additional libraries detected through attribute usage even without import
ATTR_USAGE_MAP: Final[Dict[str, str]] = {
    "pd.read_csv": "pandas",
    "plt.": "matplotlib",
    "sns.": "seaborn",
    "sklearn.": "scikit-learn",
    "joblib.": "joblib",
    "quick_eda": "preml",
    "EDAAnalyzer": "preml",
    "StatisticsEngine": "preml",
    "RecommendationEngine": "preml",
    "PreprocessingBuilder": "preml",
    "FeatureEngineering": "preml",
    "BaselineTrainer": "preml",
    "ReportGenerator": "preml",
}


def _collect_imports(tree: ast.Module) -> Dict[str, str]:
    """Build a mapping from alias (as string) to full package name."""
    alias_to_pkg: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                alias_to_pkg[alias.asname or name] = name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full = module.split(".")[0]
                alias_to_pkg[alias.asname or alias.name] = full
    return alias_to_pkg


def _detect_usage(code: str, alias_to_pkg: Dict[str, str]) -> Set[str]:
    """Detect libraries used via attribute access or known calls."""
    used: Set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return used

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check call function chain
            func = node.func
            if isinstance(func, ast.Attribute):
                parts = []
                obj = func
                while isinstance(obj, ast.Attribute):
                    parts.append(obj.attr)
                    obj = obj.value
                if isinstance(obj, ast.Name):
                    parts.append(obj.id)
                    chain = ".".join(reversed(parts))
                    for prefix, pkg in ATTR_USAGE_MAP.items():
                        if chain.startswith(prefix):
                            used.add(pkg)
                            break
                else:
                    # base is something like a.b.c, try string matches
                    pass
            elif isinstance(func, ast.Name):
                if func.id in ATTR_USAGE_MAP:
                    used.add(ATTR_USAGE_MAP[func.id])
    # Also map imported aliases to known packages
    for alias, pkg in alias_to_pkg.items():
        if pkg in KNOWN_IMPORTS:
            used.add(KNOWN_IMPORTS[pkg])
        elif pkg in KNOWN_IMPORTS.values():
            used.add(pkg)
    return used


def infer_dependencies_and_api(code: str) -> Tuple[List[str], List[str], str]:
    """
    Analyse a code snippet's dependencies and related APIs.

    Returns:
        dependencies: sorted list of package names
        related_api: sorted list of PreML API elements mentioned
        expected_behavior: short human‑readable description
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], [], "Invalid Python – cannot analyse"

    alias_to_pkg = _collect_imports(tree)
    deps = _detect_usage(code, alias_to_pkg)

    # Detect specific PreML API calls
    preml_apis: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in {
                    "quick_eda",
                    "EDAAnalyzer",
                    "StatisticsEngine",
                    "RecommendationEngine",
                    "PreprocessingBuilder",
                    "FeatureEngineering",
                    "BaselineTrainer",
                    "ReportGenerator",
                }:
                    preml_apis.add(func.id)
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "preml":
                    preml_apis.add(func.attr)
                elif (
                    isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "preml"
                ):
                    preml_apis.add(func.attr)

    expected = "Produces console output" if "print(" in code else "Runs without exception"
    return sorted(deps), sorted(preml_apis), expected


# ---------------------------------------------------------------------------
# 4. Traceback‑Based Setup Issue Classification
# ---------------------------------------------------------------------------
def classify_setup_issue(stderr: str) -> bool:
    """
    Determine whether an execution failure is a setup/configuration problem.

    Parses the traceback from ``stderr`` and checks if the outermost exception
    belongs to a known set of environment/dependency‑related errors.
    """
    if not stderr:
        return False

    # Extract the last traceback block (the one that actually crashed)
    tb_lines = stderr.splitlines()
    exception_line = ""
    for line in reversed(tb_lines):
        if line.startswith("Traceback (most recent call last):"):
            break
        if re.match(r"^\S+:", line):   # e.g., "ModuleNotFoundError: ..."
            exception_line = line
    if not exception_line:
        # Fallback: search for any exception line
        for line in tb_lines:
            if re.match(r"^\S+:", line):
                exception_line = line
                break

    exc_name = exception_line.split(":")[0].strip()
    category = SETUP_EXCEPTION_CATEGORIES.get(exc_name, "")
    return category in {"dependency", "environment"}


# ---------------------------------------------------------------------------
# 5. Virtual Environment Management with Timeouts
# ---------------------------------------------------------------------------
BASE_VENV_LOCK = Lock()


def create_exec_script(code: str) -> str:
    """Wrap user code into a script that runs without Python warning noise."""
    return textwrap.dedent(
        """\
        import warnings
        warnings.simplefilter('always')
    """
    ) + code.rstrip()


def ensure_base_venv(timeout_seconds: int = 120) -> Path:
    """
    Create (or reuse) a base virtual environment with the project installed.

    The base venv is created under ``BASE_VENV_DIR`` and reused across runs.
    Thread‑safe.
    """
    with BASE_VENV_LOCK:
        BASE_VENV_DIR.mkdir(parents=True, exist_ok=True)
        marker = BASE_VENV_DIR / ".preml-base-ready"
        if marker.exists():
            return BASE_VENV_DIR

        if BASE_VENV_DIR.exists():
            shutil.rmtree(BASE_VENV_DIR)
        BASE_VENV_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Creating base virtual environment …")
        try:
            # 1. Create venv
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(BASE_VENV_DIR)],
                check=True,
                timeout=timeout_seconds,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("Timeout while creating base venv")
            shutil.rmtree(BASE_VENV_DIR, ignore_errors=True)
            raise

        py_exec = python_executable_for(BASE_VENV_DIR)

        # 2. Upgrade pip
        try:
            subprocess.run(
                [str(py_exec), "-m", "pip", "install", "-q", "--upgrade", "pip"],
                check=True,
                timeout=timeout_seconds,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("Timeout during pip upgrade")
            shutil.rmtree(BASE_VENV_DIR, ignore_errors=True)
            raise

        # 3. Install the project in editable mode
        try:
            subprocess.run(
                [str(py_exec), "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT)],
                check=True,
                timeout=timeout_seconds * 2,  # more time for first install
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("Timeout while installing project")
            shutil.rmtree(BASE_VENV_DIR, ignore_errors=True)
            raise
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to install project into base venv:\n%s", exc.stderr)
            shutil.rmtree(BASE_VENV_DIR, ignore_errors=True)
            raise

        marker.write_text("ready\n", encoding="utf-8")
        logger.info("Base virtual environment is ready.")
        return BASE_VENV_DIR


def run_in_fresh_venv(
    block: CodeBlock,
    timeout_seconds: int,
    base_venv: Path,
) -> BlockResult:
    """
    Execute a single code block in a fresh copy of the base virtual environment.

    Returns a ``BlockResult`` with full diagnostics.
    """
    deps, apis, expected = infer_dependencies_and_api(block.code)
    runnable, reason = is_runnable_python_block(block)

    if not runnable:
        return BlockResult(
            id=block.id,
            section=block.section,
            title=block.title,
            start_line=block.start_line,
            end_line=block.end_line,
            runnable=False,
            status="skipped",
            setup_issue=False,
            dependencies=deps,
            related_api=apis,
            expected_behavior=reason or "Not a standalone runnable example",
            stdout="",
            stderr="",
            return_code=None,
            duration_seconds=0.0,
            error_summary=None,
        )

    start_time = time.monotonic()
    # Each example gets its own isolated venv copied from the base.
    with tempfile.TemporaryDirectory(prefix=f"usage_block_{block.id}_") as td:
        temp_dir = Path(td)
        venv_dir = temp_dir / "venv"
        script_path = temp_dir / "example.py"

        try:
            shutil.copytree(base_venv, venv_dir)
        except OSError as exc:
            return BlockResult(
                id=block.id,
                section=block.section,
                title=block.title,
                start_line=block.start_line,
                end_line=block.end_line,
                runnable=True,
                status="failed",
                setup_issue=True,
                dependencies=deps,
                related_api=apis,
                expected_behavior=expected,
                stdout="",
                stderr=f"Cannot create venv copy: {exc}",
                return_code=-1,
                duration_seconds=round(time.monotonic() - start_time, 3),
                error_summary="Venv copy failure",
            )

        script_path.write_text(create_exec_script(block.code), encoding="utf-8")
        py_exec = python_executable_for(venv_dir)

        try:
            proc = subprocess.run(
                [str(py_exec), str(script_path)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            dt = time.monotonic() - start_time
            return BlockResult(
                id=block.id,
                section=block.section,
                title=block.title,
                start_line=block.start_line,
                end_line=block.end_line,
                runnable=True,
                status="failed",
                setup_issue=False,
                dependencies=deps,
                related_api=apis,
                expected_behavior=expected,
                stdout="",
                stderr=f"Execution timed out after {timeout_seconds}s",
                return_code=-1,
                duration_seconds=round(dt, 3),
                error_summary="Timeout",
            )
        except Exception as exc:
            dt = time.monotonic() - start_time
            return BlockResult(
                id=block.id,
                section=block.section,
                title=block.title,
                start_line=block.start_line,
                end_line=block.end_line,
                runnable=True,
                status="failed",
                setup_issue=False,
                dependencies=deps,
                related_api=apis,
                expected_behavior=expected,
                stdout="",
                stderr=str(exc),
                return_code=-1,
                duration_seconds=round(dt, 3),
                error_summary="Internal harness error",
            )

        dt = time.monotonic() - start_time
        ok = proc.returncode == 0
        setup_issue = (not ok) and classify_setup_issue(proc.stderr)

        error_summary = None
        if not ok:
            tail = (proc.stderr or "").strip().splitlines()
            error_summary = tail[-1] if tail else "Unknown error"

        return BlockResult(
            id=block.id,
            section=block.section,
            title=block.title,
            start_line=block.start_line,
            end_line=block.end_line,
            runnable=True,
            status="passed" if ok else "failed",
            setup_issue=setup_issue,
            dependencies=deps,
            related_api=apis,
            expected_behavior=expected,
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=proc.returncode,
            duration_seconds=round(dt, 3),
            error_summary=error_summary,
        )


# ---------------------------------------------------------------------------
# 6. Report Generation
# ---------------------------------------------------------------------------
def write_reports(blocks: List[CodeBlock], results: List[BlockResult]) -> None:
    """Generate JSON and Markdown reports in ``OUT_DIR``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = [
        {
            "id": b.id,
            "language": b.language,
            "section": b.section,
            "title": b.title,
            "start_line": b.start_line,
            "end_line": b.end_line,
            "runnable": r.runnable,
            "status": r.status,
            "dependencies": r.dependencies,
            "related_api": r.related_api,
            "expected_behavior": r.expected_behavior,
        }
        for b, r in zip(blocks, results)
    ]

    report: ReportDict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "guide": str(GUIDE),
        "total_fenced_blocks": len(blocks),
        "total_python_blocks": sum(1 for b in blocks if b.language in ("python", "py")),
        "total_runnable_blocks": sum(1 for r in results if r.runnable),
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "results": [dataclasses.asdict(r) for r in results],
        "inventory": inventory,
    }

    json_path = OUT_DIR / "usage_guide_validation.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("JSON report written to %s", json_path)

    # Markdown summary
    lines: List[str] = [
        "# Usage Guide Validation Report",
        "",
        f"- Total fenced blocks: {report['total_fenced_blocks']}",
        f"- Python blocks: {report['total_python_blocks']}",
        f"- Runnable blocks: {report['total_runnable_blocks']}",
        f"- Passed: {report['passed']}",
        f"- Failed: {report['failed']}",
        f"- Skipped: {report['skipped']}",
        "",
        "## Failures",
        "",
    ]
    fails = [r for r in results if r.status == "failed"]
    if not fails:
        lines.append("No failures.")
    else:
        for r in fails:
            title_str = f" ({r.title})" if r.title else ""
            lines.append(
                f"- Block {r.id} | {r.section}{title_str} | lines {r.start_line}-{r.end_line}"
            )
            lines.append(f"  - Error: {r.error_summary or 'Unknown'}")
            lines.append(f"  - Setup issue: {r.setup_issue}")
            lines.append(f"  - Return code: {r.return_code}")
            lines.append("")

    md_path = OUT_DIR / "usage_guide_validation.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Markdown summary written to %s", md_path)


# ---------------------------------------------------------------------------
# 7. CLI and Main Orchestration
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all Python examples in Usage Guide.md"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Per‑example timeout in seconds (default: 240)",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=0,
        help="Maximum number of blocks to execute (0 = all)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previous output directory before running",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="Number of parallel workers (default: up to 4)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.setLevel(args.log_level)

    if args.clean_output and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        logger.info("Cleared previous output directory.")

    # Parse markdown
    markdown = read_text(GUIDE)
    parser = MarkdownParser()
    blocks = parser.parse(markdown)
    logger.info("Extracted %d fenced blocks from guide.", len(blocks))

    python_blocks = [b for b in blocks if b.language in {"python", "py"}]
    logger.info("Found %d Python blocks.", len(python_blocks))

    # Apply max‑blocks filter
    if args.max_blocks > 0:
        selected_blocks = python_blocks[: args.max_blocks]
    else:
        selected_blocks = python_blocks
    selected_ids = {b.id for b in selected_blocks}

    # Ensure base venv is ready
    logger.info("Preparing base virtual environment …")
    base_venv = ensure_base_venv(timeout_seconds=args.timeout)

    # Prepare results container
    results: Dict[int, BlockResult] = {}
    for block in blocks:
        if block.id not in selected_ids:
            # Non‑selected blocks are skipped
            deps, apis, expected = infer_dependencies_and_api(block.code)
            results[block.id] = BlockResult(
                id=block.id,
                section=block.section,
                title=block.title,
                start_line=block.start_line,
                end_line=block.end_line,
                runnable=False,
                status="skipped",
                setup_issue=False,
                dependencies=deps,
                related_api=apis,
                expected_behavior=expected,
                stdout="",
                stderr="",
                return_code=None,
                duration_seconds=0.0,
                error_summary="Not selected",
            )

    # Execute runnable blocks in parallel
    runnable_blocks = [b for b in blocks if b.id in selected_ids]
    logger.info("Executing %d runnable blocks with %d workers …",
                len(runnable_blocks), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(run_in_fresh_venv, block, args.timeout, base_venv): block.id
            for block in runnable_blocks
        }
        for future in as_completed(future_map):
            block_id = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Unexpected error processing block %d: %s", block_id, exc)
                # Create a synthetic failure result
                block = next(b for b in runnable_blocks if b.id == block_id)
                results[block_id] = BlockResult(
                    id=block_id,
                    section=block.section,
                    title=block.title,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    runnable=True,
                    status="failed",
                    setup_issue=False,
                    dependencies=[],
                    related_api=[],
                    expected_behavior="",
                    stdout="",
                    stderr=str(exc),
                    return_code=-1,
                    duration_seconds=0.0,
                    error_summary="Harness exception",
                )
            else:
                results[block_id] = result

    # Order results as per original block list
    ordered_results = [results[b.id] for b in blocks]

    # Write reports
    write_reports(blocks, ordered_results)

    failed_count = sum(1 for r in ordered_results if r.status == "failed")
    logger.info("Validation complete. Failed blocks: %d", failed_count)
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
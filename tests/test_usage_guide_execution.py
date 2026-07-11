"""Runtime validation tests for Usage Guide code examples.

These tests parse Python blocks from Usage Guide.md and execute them through
scripts/validate_usage_guide_examples.py to ensure examples remain healthy.

Speed optimization:
    The subset test does NOT use --clean-output, allowing the base virtual
    environment to be reused across runs. The first run will be slower as it
    builds the base venv; subsequent runs are fast (<5s for 3 blocks).
    Set CLEAN_OUTPUT=1 to force cleaning.

Full test (set RUN_FULL_USAGE_GUIDE_VALIDATION=1) cleans the output directory
by default to guarantee a pristine environment, but you can override that
with CLEAN_OUTPUT=0.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.validate_usage_guide_examples as validator


ROOT = Path(__file__).resolve().parent.parent
GUIDE_PATH = ROOT / "Usage Guide.md"
REPORT_PATH = ROOT / "artifacts" / "usage_guide_validation" / "usage_guide_validation.json"


def _run_validator(
    *,
    max_blocks: int,
    workers: int,
    timeout: int,
    clean_output: bool = False,
) -> dict:
    """Invoke the validator script as a subprocess and parse the JSON report."""
    cmd = [
        sys.executable,
        "scripts/validate_usage_guide_examples.py",
        f"--workers={workers}",
        f"--timeout={timeout}",
    ]
    if max_blocks > 0:
        cmd.append(f"--max-blocks={max_blocks}")
    if clean_output:
        cmd.append("--clean-output")

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if not REPORT_PATH.exists():
        pytest.fail(
            "Validator did not generate report file.\n"
            f"Exit code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _format_failures(report: dict, limit: int = 10) -> str:
    failures = [r for r in report.get("results", []) if r.get("status") == "failed"]
    if not failures:
        return "No failures."

    parser = validator.MarkdownParser()
    blocks = parser.parse(GUIDE_PATH.read_text(encoding="utf-8"))
    block_by_id = {b.id: b for b in blocks}

    lines: list[str] = []
    for idx, row in enumerate(failures[:limit], start=1):
        block_id = row.get("id")
        block = block_by_id.get(block_id)

        lines.append("=" * 90)
        lines.append(f"Failure #{idx}")
        lines.append(
            "Block #{id} | section: {section} | lines: {start}-{end}".format(
                id=block_id,
                section=row.get("section"),
                start=row.get("start_line"),
                end=row.get("end_line"),
            )
        )
        lines.append(f"Title: {row.get('title')}")
        lines.append(f"Error summary: {row.get('error_summary') or 'Unknown error'}")
        lines.append(f"Setup issue: {row.get('setup_issue')}")
        lines.append(f"Return code: {row.get('return_code')}")

        stderr = (row.get("stderr") or "").strip()
        stdout = (row.get("stdout") or "").strip()
        if stderr:
            lines.append("\n--- STDERR ---")
            lines.append(stderr)
        if stdout:
            lines.append("\n--- STDOUT ---")
            lines.append(stdout)

        lines.append("\n--- CODE BLOCK ---")
        if block is not None:
            lines.append(block.code.rstrip() or "<empty>")
        else:
            lines.append("<code block not found in current Usage Guide.md>")

    if len(failures) > limit:
        lines.append("=" * 90)
        lines.append(f"... {len(failures) - limit} more failures were omitted.")

    return "\n".join(lines)


def test_usage_guide_python_blocks_are_extracted() -> None:
    """Verify that the guide file exists and contains Python code blocks."""
    assert GUIDE_PATH.exists(), "Usage Guide.md is missing"

    parser = validator.MarkdownParser()
    blocks = parser.parse(GUIDE_PATH.read_text(encoding="utf-8"))

    python_blocks = [b for b in blocks if b.language in {"python", "py"}]
    assert len(blocks) > 0, "No fenced blocks found in the guide"
    assert len(python_blocks) > 0, "No Python blocks found in the guide"


def test_runnable_usage_guide_blocks_have_valid_syntax() -> None:
    """Ensure that every runnable Python block parses correctly (AST check)."""
    parser = validator.MarkdownParser()
    blocks = parser.parse(GUIDE_PATH.read_text(encoding="utf-8"))

    for block in blocks:
        if block.language not in {"python", "py"}:
            continue
        runnable, _ = validator.is_runnable_python_block(block)
        if not runnable:
            continue

        try:
            ast.parse(block.code.strip())
        except SyntaxError as exc:
            pytest.fail(
                "Syntax error in runnable block "
                f"#{block.id} ({block.section}) lines {block.start_line}-{block.end_line}: {exc}"
            )


def test_usage_guide_examples_execute_subset_cleanly() -> None:
    """
    Execute a small number of examples to catch immediate breakage.

    By default, runs only 3 blocks without cleaning the output directory,
    reusing the previously built base virtual environment for speed.
    Override via environment variables:
      USAGE_GUIDE_SUBSET_BLOCKS  – number of blocks to run (default 3)
      USAGE_GUIDE_SUBSET_WORKERS – parallel workers (default 2)
      USAGE_GUIDE_SUBSET_TIMEOUT – per-block timeout in seconds (default 120)
      CLEAN_OUTPUT               – set to "1" to force cleaning the output dir
    """
    max_blocks = int(os.getenv("USAGE_GUIDE_SUBSET_BLOCKS", "3"))
    workers = int(os.getenv("USAGE_GUIDE_SUBSET_WORKERS", "2"))
    timeout = int(os.getenv("USAGE_GUIDE_SUBSET_TIMEOUT", "120"))
    clean = os.getenv("CLEAN_OUTPUT", "0") == "1"

    report = _run_validator(
        max_blocks=max_blocks,
        workers=workers,
        timeout=timeout,
        clean_output=clean,
    )

    assert report["total_python_blocks"] > 0
    assert report["total_runnable_blocks"] > 0

    if report["failed"] != 0:
        details = _format_failures(report)
        pytest.fail(
            "Usage Guide subset execution found failing examples.\n"
            f"Failures: {report['failed']}\n"
            f"Details:\n{details}"
        )


@pytest.mark.skipif(
    os.getenv("RUN_FULL_USAGE_GUIDE_VALIDATION") != "1",
    reason="Set RUN_FULL_USAGE_GUIDE_VALIDATION=1 to run full guide execution",
)
def test_usage_guide_examples_execute_fully() -> None:
    """
    Full end-to-end validation – executes *all* runnable Python examples.

    By default, cleans the output directory to start from a clean state.
    You can skip cleaning by setting CLEAN_OUTPUT=0 if you trust the existing
    base environment.

    Override via environment variables:
      FULL_VALIDATION_WORKERS – parallel workers (default 4)
      FULL_VALIDATION_TIMEOUT  – per-block timeout in seconds (default 300)
      CLEAN_OUTPUT             – set to "0" to keep existing output/base venv
    """
    workers = int(os.getenv("FULL_VALIDATION_WORKERS", "4"))
    timeout = int(os.getenv("FULL_VALIDATION_TIMEOUT", "300"))
    clean = os.getenv("CLEAN_OUTPUT", "1") == "1"   # full test cleans by default

    report = _run_validator(
        max_blocks=0,          # 0 means all blocks
        workers=workers,
        timeout=timeout,
        clean_output=clean,
    )

    if report["failed"] != 0:
        details = _format_failures(report)
        pytest.fail(
            "Full Usage Guide execution found failing examples.\n"
            f"Failures: {report['failed']}\n"
            f"Details:\n{details}"
        )
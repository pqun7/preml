"""
Comprehensive tests for the Usage Guide validator.

Covers parser behavior, runnable-block detection, setup-issue classification,
dependency/API inference, and a small end-to-end workflow.
"""

import json
import sys
from unittest import mock

import pytest

import scripts.validate_usage_guide_examples as validator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_markdown() -> str:
    return """# Introduction
Some text.

## Quick Start
```python
print(\"Hello\")
```

### Subsection
```python
x = 1
```

```bash
ls
```
"""


@pytest.fixture
def signature_block() -> str:
    """Function signature only - should be non-runnable."""
    return "def foo(a: int) -> str:\n    ..."


@pytest.fixture
def runnable_block() -> str:
    return "print('runs')"


@pytest.fixture
def block_with_defs() -> str:
    return "def f():\n    pass\nclass C:\n    pass"


# ---------------------------------------------------------------------------
# Unit Tests: MarkdownParser
# ---------------------------------------------------------------------------
class TestMarkdownParser:
    def test_extract_blocks_section_hierarchy(self, sample_markdown: str) -> None:
        parser = validator.MarkdownParser()
        blocks = parser.parse(sample_markdown)

        assert len(blocks) == 3

        # First block under "Quick Start"
        assert blocks[0].section == "Introduction > Quick Start"
        assert blocks[0].language == "python"
        assert "print" in blocks[0].code

        # Second block under "Subsection"
        assert blocks[1].section == "Introduction > Quick Start > Subsection"
        assert blocks[1].language == "python"
        assert "x = 1" in blocks[1].code

    def test_ignore_non_python_blocks(self, sample_markdown: str) -> None:
        blocks = validator.MarkdownParser().parse(sample_markdown)
        python_blocks = [b for b in blocks if b.language in {"python", "py"}]
        assert len(python_blocks) == 2

    def test_title_from_previous_line(self) -> None:
        md = """## Section
This is a title

```python
code = 1
```
"""
        blocks = validator.MarkdownParser().parse(md)
        assert len(blocks) == 1
        assert blocks[0].title == "This is a title"


# ---------------------------------------------------------------------------
# Unit Tests: is_runnable_python_block
# ---------------------------------------------------------------------------
class TestRunnableDetection:
    def test_empty_block_not_runnable(self) -> None:
        block = validator.CodeBlock(1, "python", "", 1, 1, "sec")
        runnable, _ = validator.is_runnable_python_block(block)
        assert not runnable

    def test_signature_only_block_not_runnable(self, signature_block: str) -> None:
        block = validator.CodeBlock(1, "python", signature_block, 1, 2, "sec")
        runnable, _ = validator.is_runnable_python_block(block)
        assert not runnable

    def test_runnable_with_print(self, runnable_block: str) -> None:
        block = validator.CodeBlock(1, "python", runnable_block, 1, 1, "sec")
        runnable, _ = validator.is_runnable_python_block(block)
        assert runnable

    def test_defs_only_not_runnable(self, block_with_defs: str) -> None:
        block = validator.CodeBlock(1, "python", block_with_defs, 1, 2, "sec")
        runnable, _ = validator.is_runnable_python_block(block)
        assert not runnable

    def test_mixed_def_and_call_runnable(self) -> None:
        code = "def f():\n    pass\nprint('hi')"
        block = validator.CodeBlock(1, "python", code, 1, 3, "sec")
        runnable, _ = validator.is_runnable_python_block(block)
        assert runnable


# ---------------------------------------------------------------------------
# Unit Tests: classify_setup_issue
# ---------------------------------------------------------------------------
class TestSetupIssueClassification:
    def test_module_not_found_is_setup(self) -> None:
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"example.py\", line 1, in <module>\n"
            "ModuleNotFoundError: No module named 'foo'\n"
        )
        assert validator.classify_setup_issue(stderr)

    def test_keyerror_is_not_setup(self) -> None:
        stderr = "Traceback (most recent call last):\nKeyError: 'col'"
        assert not validator.classify_setup_issue(stderr)

    def test_user_printed_exception_not_classified(self) -> None:
        stderr = "Some user output containing ModuleNotFoundError in a print statement"
        assert not validator.classify_setup_issue(stderr)


# ---------------------------------------------------------------------------
# Unit Tests: infer_dependencies_and_api
# ---------------------------------------------------------------------------
class TestDependencyDetection:
    def test_import_pandas_detected(self) -> None:
        code = "import pandas as pd\npd.read_csv('file')"
        deps, apis, _ = validator.infer_dependencies_and_api(code)
        assert "pandas" in deps
        assert apis == []

    def test_matplotlib_usage_detected(self) -> None:
        code = "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])"
        deps, _, _ = validator.infer_dependencies_and_api(code)
        assert "matplotlib" in deps

    def test_preml_apis_detected(self) -> None:
        code = "from preml import quick_eda\nquick_eda()"
        deps, apis, _ = validator.infer_dependencies_and_api(code)
        assert "preml" in deps
        assert "quick_eda" in apis


# ---------------------------------------------------------------------------
# Integration Tests (with temporary files)
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_full_workflow_with_two_examples(self, tmp_path) -> None:
        """Simulate a small guide with two examples and run end-to-end."""

        guide_content = """# Test

Example 1
```python
print(\"success\")
```

Example 2
```python
import sys
print(sys.version)
```
"""
        guide_file = tmp_path / "guide.md"
        guide_file.write_text(guide_content, encoding="utf-8")

        with (
            mock.patch.object(validator, "GUIDE", guide_file),
            mock.patch.object(validator, "OUT_DIR", tmp_path / "output"),
            mock.patch.object(validator, "ensure_base_venv", return_value=tmp_path / "base"),
            mock.patch.object(validator, "run_in_fresh_venv") as mock_run,
        ):

            def fake_run(block, timeout_seconds, base_venv):
                runnable, _ = validator.is_runnable_python_block(block)
                if runnable:
                    return validator.BlockResult(
                        id=block.id,
                        section=block.section,
                        title=block.title,
                        start_line=block.start_line,
                        end_line=block.end_line,
                        runnable=True,
                        status="passed",
                        setup_issue=False,
                        dependencies=[],
                        related_api=[],
                        expected_behavior="Runs without exception",
                        stdout="success\n",
                        stderr="",
                        return_code=0,
                        duration_seconds=0.1,
                        error_summary=None,
                    )
                return validator.BlockResult(
                    id=block.id,
                    section=block.section,
                    title=block.title,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    runnable=False,
                    status="skipped",
                    setup_issue=False,
                    dependencies=[],
                    related_api=[],
                    expected_behavior="Not runnable",
                    stdout="",
                    stderr="",
                    return_code=None,
                    duration_seconds=0.0,
                    error_summary=None,
                )

            mock_run.side_effect = fake_run

            test_args = ["validate.py", "--workers=1", "--timeout=10"]
            with mock.patch.object(sys, "argv", test_args):
                exit_code = validator.main()

            assert exit_code == 0

        out_dir = tmp_path / "output"
        report_file = out_dir / "usage_guide_validation.json"
        assert report_file.exists()

        report = json.loads(report_file.read_text(encoding="utf-8"))
        assert report["passed"] == 2

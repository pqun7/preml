"""Tests for the high-level PreML facade and root convenience helpers."""

from __future__ import annotations

import pandas as pd

import preml
from preml.facade import PreML, analyze, models, pipeline, quick_eda, recommendations, report, visualize


def _make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["A", "B", "A", "C"],
            "target": [0, 1, 0, 1],
        }
    )


def test_root_exports_include_facade_helpers() -> None:
    assert preml.PreML is PreML
    assert preml.analyze is analyze
    assert preml.quick_eda is quick_eda
    assert preml.recommendations is recommendations
    assert preml.report is report
    assert preml.pipeline is pipeline
    assert preml.visualize is visualize
    assert preml.models is models


def test_facade_caches_analysis(monkeypatch) -> None:
    df = _make_frame()
    facade = PreML(df, target="target")

    call_count = {"run": 0}
    original_run = facade._analyzer.run

    def wrapped_run():
        call_count["run"] += 1
        return original_run()

    monkeypatch.setattr(facade._analyzer, "run", wrapped_run)

    first = facade.analyze()
    second = facade.analyze()
    summary = facade.summary()
    builder = facade.pipeline()
    recs = facade.recommendations()

    assert first is second
    assert call_count["run"] == 1
    assert isinstance(summary, str)
    assert builder is not None
    assert isinstance(recs, dict)


def test_convenience_helpers_execute() -> None:
    df = _make_frame()

    analysis = analyze(df, target="target")
    recs = recommendations(df, target="target")
    report_text = report(df, target="target")
    builder = pipeline(df, target="target")
    figures = visualize(df, target="target", kind="missing")

    assert "metadata" in analysis
    assert "models" in recs or "pipeline" in recs
    assert isinstance(report_text, str)
    assert builder is not None
    assert "missing_heatmap" in figures

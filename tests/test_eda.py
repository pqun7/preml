"""
Unit tests for preml.eda (EDAAnalyzer and quick_eda).

Tests cover:
- Proper orchestration of engines
- data_quality_score computation
- summary() output
- edge cases (empty dataframe, no target)
- error handling
"""

import numpy as np
import pandas as pd
import pytest

from preml.config import MLToolkitConfig
from preml.eda import EDAAnalyzer, quick_eda
from preml.exceptions import DataValidationError


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num1": np.random.normal(0, 1, n),
        "num2": np.random.uniform(0, 10, n),
        "cat1": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })
    # Add some missing
    df.loc[0:2, "num1"] = np.nan
    return df


@pytest.fixture
def analyzer_with_target(sample_df):
    return EDAAnalyzer(sample_df, target="target")


@pytest.fixture
def analyzer_no_target(sample_df):
    return EDAAnalyzer(sample_df)


class TestEDAAnalyzer:
    def test_run_returns_expected_keys(self, analyzer_with_target):
        result = analyzer_with_target.run()
        expected = {
            "metadata", "duplicates", "infinite", "missing",
            "outliers", "feature_profiles", "correlation_pairs",
            "target_profile", "recommendations",
            "data_quality_score", "data_quality_notes"
        }
        assert set(result.keys()) >= expected

    def test_quality_score_between_0_and_100(self, analyzer_with_target):
        result = analyzer_with_target.run()
        assert 0 <= result["data_quality_score"] <= 100

    def test_quality_notes_list(self, analyzer_with_target):
        result = analyzer_with_target.run()
        assert isinstance(result["data_quality_notes"], list)

    def test_summary_returns_string(self, analyzer_with_target):
        summary = analyzer_with_target.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_includes_scaling_recommendation(self, analyzer_with_target):
        summary = analyzer_with_target.summary()
        assert "scale numeric features" in summary.lower()

    def test_no_target_does_not_crash(self, analyzer_no_target):
        result = analyzer_no_target.run()
        assert result["target_profile"] is None

    def test_invalid_dataframe_raises(self):
        with pytest.raises(DataValidationError):
            EDAAnalyzer([1, 2, 3])

    def test_quick_eda_convenience(self, sample_df):
        result = quick_eda(sample_df)
        assert "metadata" in result
        assert "recommendations" in result
        assert result["target_profile"] is None
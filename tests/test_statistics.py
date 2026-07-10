"""
Unit tests for preml.statistics_engine.

Tests cover:
- Metadata computation
- Duplicate & infinite value detection
- Missing value reports
- Outlier detection (IQR, Z‑score)
- Numeric and categorical profiling
- Correlation pair extraction
- Target profile (regression, classification)
- Error handling (invalid input, missing target)
"""

import numpy as np
import pandas as pd
import pytest

from preml.config import MLToolkitConfig
from preml.exceptions import DataValidationError
from preml.schema import (
    CategoricalProfile,
    CorrelationPair,
    DatasetMetadata,
    DuplicateReport,
    FeatureProfile,
    InfiniteReport,
    MissingReport,
    NumericDistributionProfile,
    OutlierReport,
    TargetProfile,
)
from preml.statistics_engine import StatisticsEngine


# ------------------- Fixtures -------------------
@pytest.fixture
def sample_df():
    """Create a simple mixed DataFrame for testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num1": np.random.normal(0, 1, n),
        "num2": np.random.uniform(0, 10, n),
        "cat1": np.random.choice(["A", "B", "C"], n),
        "cat2": np.random.choice(["X", "Y"], n),
        "target": np.random.choice([0, 1], n),  # binary target
    })
    # Add some missing values
    df.loc[0:2, "num1"] = np.nan
    df.loc[10:12, "cat1"] = None
    return df


@pytest.fixture
def engine(sample_df):
    """Return a StatisticsEngine with target set."""
    return StatisticsEngine(sample_df, target="target")


@pytest.fixture
def engine_no_target(sample_df):
    """Return a StatisticsEngine without target."""
    return StatisticsEngine(sample_df)


# ------------------- Tests -------------------
class TestDatasetMetadata:
    def test_returns_correct_metadata(self, engine):
        meta = engine.compute_dataset_metadata()
        assert isinstance(meta, DatasetMetadata)
        assert meta.n_rows == 100
        assert meta.n_columns == 5
        assert meta.memory_mb > 0


class TestDuplicates:
    def test_no_duplicates(self, engine):
        dup = engine.compute_duplicate_report()
        assert isinstance(dup, DuplicateReport)
        assert dup.total_duplicates == 0
        assert dup.duplicate_percent == 0.0

    def test_with_duplicates(self, engine):
        df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [3, 4, 3, 4]})
        se = StatisticsEngine(df)
        dup = se.compute_duplicate_report()
        assert dup.total_duplicates == 2
        assert dup.duplicate_percent == 50.0
        assert len(dup.sample_indices) == 4


class TestInfinite:
    def test_no_infinite(self, engine):
        inf = engine.compute_infinite_report()
        assert inf.columns_with_inf == []

    def test_with_infinite(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, -np.inf]})
        se = StatisticsEngine(df)
        inf = se.compute_infinite_report()
        assert set(inf.columns_with_inf) == {"a", "b"}
        assert inf.counts["a"] == 1
        assert inf.counts["b"] == 1


class TestMissing:
    def test_missing_report(self, engine):
        miss = engine.compute_missing_report()
        assert isinstance(miss, MissingReport)
        assert miss.total_missing > 0
        assert "num1" in miss.columns_with_missing
        assert "cat1" in miss.columns_with_missing

    def test_missing_percentages(self, engine):
        miss = engine.compute_missing_report()
        for col_rpt in miss.column_reports:
            if col_rpt.column == "num1":
                assert col_rpt.missing_count == 3
                assert col_rpt.missing_percent == 3.0

    def test_missing_report_empty_dataframe(self):
        df = pd.DataFrame(columns=["a", "b"])
        se = StatisticsEngine(df)
        miss = se.compute_missing_report()
        assert miss.total_missing == 0
        assert miss.columns_with_missing == []
        assert miss.column_reports == []


class TestOutliers:
    def test_iqr_outlier_detection(self, engine):
        # Use config with IQR method
        config = MLToolkitConfig(outlier_method="iqr")
        se = StatisticsEngine(engine.df, config=config)
        outliers = se.compute_outlier_report()
        assert all(isinstance(o, OutlierReport) for o in outliers)
        assert len(outliers) > 0
        for o in outliers:
            assert o.method == "iqr"

    def test_zscore_outlier_detection(self, engine):
        config = MLToolkitConfig(outlier_method="zscore")
        se = StatisticsEngine(engine.df, config=config)
        outliers = se.compute_outlier_report()
        assert all(o.method == "zscore" for o in outliers)

    def test_invalid_method_raises(self, engine):
        config = MLToolkitConfig(outlier_method="bad")
        se = StatisticsEngine(engine.df, config=config)
        with pytest.raises(DataValidationError):
            se.compute_outlier_report()


class TestFeatureProfiles:
    def test_profiles_created_for_all_columns(self, engine):
        profiles = engine.compute_feature_profiles()
        assert len(profiles) == 4  # excluding target
        col_names = {p.column for p in profiles}
        assert col_names == {"num1", "num2", "cat1", "cat2"}

    def test_numeric_profile_content(self, engine):
        profiles = engine.compute_feature_profiles()
        num_prof = next(p for p in profiles if p.column == "num1")
        assert isinstance(num_prof.numeric_profile, NumericDistributionProfile)
        assert num_prof.numeric_profile.count == 97  # 100 - 3 NaN
        assert not np.isnan(num_prof.numeric_profile.mean)

    def test_categorical_profile_content(self, engine):
        profiles = engine.compute_feature_profiles()
        cat_prof = next(p for p in profiles if p.column == "cat1")
        assert isinstance(cat_prof.categorical_profile, CategoricalProfile)
        assert cat_prof.categorical_profile.unique_count == 3

    def test_constant_column_detected(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 3, 4]})
        se = StatisticsEngine(df)
        profiles = se.compute_feature_profiles()
        const = [p for p in profiles if p.column == "a"][0]
        assert const.is_constant is True


class TestCorrelations:
    def test_correlation_pairs_below_threshold(self, engine):
        pairs = engine.compute_correlation_pairs()
        # num1 and num2 have low correlation
        assert all(abs(p.coefficient) < engine.config.correlation_threshold for p in pairs)

    def test_high_correlation_detected(self):
        df = pd.DataFrame({
            "x": np.arange(100),
            "y": np.arange(100) + np.random.normal(0, 0.1, 100),
            "z": np.random.rand(100),
        })
        config = MLToolkitConfig(correlation_threshold=0.9)
        se = StatisticsEngine(df, config=config)
        pairs = se.compute_correlation_pairs()
        assert any(p.feature_a == "x" and p.feature_b == "y" for p in pairs)


class TestTargetProfile:
    def test_target_profile_regression(self):
        df = pd.DataFrame({"target": np.random.randn(100) + 50})
        se = StatisticsEngine(df, target="target")
        tp = se.compute_target_profile()
        assert tp.is_regression
        assert not tp.is_binary

    def test_target_profile_binary(self, engine):
        tp = engine.compute_target_profile()
        assert tp is not None
        assert tp.is_binary
        assert not tp.is_regression

    def test_target_profile_multiclass(self):
        df = pd.DataFrame({"target": np.random.choice([0, 1, 2], 100)})
        se = StatisticsEngine(df, target="target")
        tp = se.compute_target_profile()
        assert not tp.is_regression
        assert not tp.is_binary
        assert tp.n_unique == 3

    def test_missing_target_handled(self):
        df = pd.DataFrame({"target": [1.0, np.nan, 3.0]})
        se = StatisticsEngine(df, target="target")
        tp = se.compute_target_profile()
        assert tp.missing_count == 1


class TestFullAnalysis:
    def test_run_full_analysis_returns_all_keys(self, engine):
        result = engine.run_full_analysis()
        expected_keys = {
            "metadata", "duplicates", "infinite", "missing",
            "outliers", "feature_profiles", "correlation_pairs", "target_profile"
        }
        assert set(result.keys()) == expected_keys

    def test_run_full_analysis_no_target(self, engine_no_target):
        result = engine_no_target.run_full_analysis()
        assert result["target_profile"] is None

    def test_invalid_dataframe_raises(self):
        with pytest.raises(DataValidationError):
            StatisticsEngine([1, 2, 3])

    def test_invalid_target_raises(self, sample_df):
        with pytest.raises(DataValidationError):
            StatisticsEngine(sample_df, target="nonexistent")
"""
Comprehensive unit tests for preml.recommendation_engine.

Updated to cover:
- Input validation with clear error messages.
- Configurable threshold behaviour.
- Missing key / malformed analysis_results.
- Imputation, outlier, transformation, scaling, encoding recommendations.
- Feature engineering and feature selection with correlation threshold.
- Enhanced model recommendations (missing values, XGBoost/LightGBM if available).
- Pipeline suggestion based on real pre‑processing needs.
- summarise() output structure.
- Edge cases (empty data, no target, no numeric features).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from preml.config import MLToolkitConfig
from preml.recommendation_engine import RecommendationEngine
from preml.exceptions import RecommendationError
from preml.schema import (
    DatasetMetadata,
    DuplicateReport,
    InfiniteReport,
    MissingReport,
    MissingColumnReport,
    OutlierReport,
    FeatureProfile,
    NumericDistributionProfile,
    CategoricalProfile,
    CorrelationPair,
    TargetProfile,
    Recommendation,
    PipelineSuggestion,
    ModelRecommendation,
)


# ---------------------------------------------------------------------------
# Helper to build a minimal analysis_results dict
# ---------------------------------------------------------------------------
def _make_minimal_analysis(**overrides):
    """Create a minimal analysis dict with sensible defaults."""
    base = {
        "metadata": DatasetMetadata(100, 5, 0.5, {}),
        "duplicates": DuplicateReport(0, 0.0),
        "infinite": InfiniteReport(),
        "missing": MissingReport(0),
        "outliers": [],
        "feature_profiles": [],
        "correlation_pairs": [],
        "target_profile": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def engine():
    return RecommendationEngine()


@pytest.fixture
def engine_custom_config():
    """Engine with a custom config to test threshold overrides."""
    config = MLToolkitConfig()
    config.missing_threshold = 0.5
    config.skewness_threshold = 2.0
    config.correlation_threshold = 0.8
    setattr(config, "outlier_threshold_percent", 10.0)
    return RecommendationEngine(config=config)


@pytest.fixture
def numeric_profile():
    return NumericDistributionProfile(
        column="num",
        count=100,
        mean=10.0,
        median=9.0,
        std=2.0,
        cv=0.2,
        min=5.0,
        max=15.0,
        skewness=1.8,
        kurtosis=2.0,
        zero_percent=0.0,
        negative_percent=0.0,
        unique_count=85,
    )


@pytest.fixture
def categorical_profile():
    return CategoricalProfile(
        column="cat",
        unique_count=5,
        missing_count=0,
        missing_percent=0.0,
        mode="A",
    )


@pytest.fixture
def feature_profiles(numeric_profile, categorical_profile):
    return [
        FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile),
        FeatureProfile(column="cat", dtype="object", categorical_profile=categorical_profile),
    ]


@pytest.fixture
def target_profile_binary():
    return TargetProfile(
        column="target",
        dtype="int64",
        n_unique=2,
        is_regression=False,
        is_binary=True,
    )


@pytest.fixture
def target_profile_regression():
    return TargetProfile(
        column="target",
        dtype="float64",
        n_unique=100,
        is_regression=True,
        is_binary=False,
    )


# ---------------------------------------------------------------------------
# Input validation and error handling
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_missing_required_key_raises(self, engine):
        with pytest.raises(RecommendationError, match="Missing required key"):
            engine.generate_recommendations({})

    def test_non_dict_input_raises(self, engine):
        with pytest.raises(RecommendationError, match="must be a dictionary"):
            engine.generate_recommendations("not a dict")

    def test_wrong_type_feature_profiles(self, engine):
        analysis = _make_minimal_analysis(feature_profiles="not a list")
        with pytest.raises(RecommendationError, match="must be a list"):
            engine.generate_recommendations(analysis)

    def test_wrong_type_outlier_reports(self, engine):
        analysis = _make_minimal_analysis(outliers="wrong")
        with pytest.raises(RecommendationError, match="must be a list"):
            engine.generate_recommendations(analysis)

    def test_wrong_type_correlation_pairs(self, engine):
        analysis = _make_minimal_analysis(correlation_pairs="wrong")
        with pytest.raises(RecommendationError, match="must be a list"):
            engine.generate_recommendations(analysis)


# ---------------------------------------------------------------------------
# Imputation recommendations
# ---------------------------------------------------------------------------
class TestImputation:
    def test_no_missing_returns_empty(self, engine):
        analysis = _make_minimal_analysis(missing=MissingReport(0))
        recs = engine._imputation_recommendations([], analysis["missing"])
        assert recs == []

    def test_missing_numeric_with_outliers(self, engine, numeric_profile):
        missing = MissingReport(10, columns_with_missing=["num"], column_reports=[
            MissingColumnReport("num", 10, 10.0)
        ])
        engine._outlier_columns = ["num"]
        recs = engine._imputation_recommendations(
            [FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)],
            missing
        )
        assert len(recs) >= 1
        assert "median" in recs[0].action.lower()

    def test_missing_categorical(self, engine, categorical_profile):
        missing = MissingReport(5, columns_with_missing=["cat"], column_reports=[
            MissingColumnReport("cat", 5, 5.0)
        ])
        recs = engine._imputation_recommendations(
            [FeatureProfile(column="cat", dtype="object", categorical_profile=categorical_profile)],
            missing
        )
        assert any("mode" in r.action.lower() for r in recs)

    def test_high_missing_triggers_investigation(self, engine):
        missing = MissingReport(50, columns_with_missing=["num"], column_reports=[
            MissingColumnReport("num", 50, 50.0)
        ])
        recs = engine._imputation_recommendations([], missing)
        assert any("investigate" in r.action.lower() for r in recs)

    def test_config_missing_threshold_used(self, engine_custom_config):
        # Custom config sets threshold to 0.5, 40% missing should NOT trigger investigation
        missing = MissingReport(40, columns_with_missing=["col"], column_reports=[
            MissingColumnReport("col", 40, 40.0)
        ])
        recs = engine_custom_config._imputation_recommendations([], missing)
        # Since 40% < 50% threshold, no investigation message
        assert not any("investigate" in r.action.lower() for r in recs)


# ---------------------------------------------------------------------------
# Outlier recommendations
# ---------------------------------------------------------------------------
class TestOutlierRecommendations:
    def test_no_outliers(self, engine):
        recs = engine._outlier_recommendations([])
        assert recs == []

    def test_minor_outliers(self, engine):
        out = OutlierReport("col", "iqr", 1, 1.0)
        recs = engine._outlier_recommendations([out])
        assert len(recs) == 1
        assert recs[0].confidence < 0.8

    def test_significant_outliers(self, engine):
        out = OutlierReport("col", "iqr", 10, 10.0)
        recs = engine._outlier_recommendations([out])
        assert recs[0].confidence > 0.8

    def test_config_outlier_threshold_used(self, engine_custom_config):
        # Custom config sets threshold to 10%, 9% should be minor
        out = OutlierReport("col", "iqr", 9, 9.0)
        recs = engine_custom_config._outlier_recommendations([out])
        assert recs[0].confidence < 0.8   # minor
        out2 = OutlierReport("col2", "iqr", 15, 15.0)
        recs2 = engine_custom_config._outlier_recommendations([out2])
        assert recs2[0].confidence > 0.8  # significant


# ---------------------------------------------------------------------------
# Transformation recommendations
# ---------------------------------------------------------------------------
class TestTransformationRecommendations:
    def test_no_skewed_features(self, engine):
        prof = NumericDistributionProfile(
            column="x", count=100, mean=0, median=0, std=1, cv=0.0,
            min=-3, max=3, skewness=0.5
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine._transformation_recommendations([fp])
        assert recs == []

    def test_right_skewed_positive(self, engine):
        prof = NumericDistributionProfile(
            column="x", count=100, mean=10, median=5, std=5, cv=0.5,
            min=1, max=50, skewness=3.0
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine._transformation_recommendations([fp])
        assert len(recs) == 1
        assert "log" in recs[0].action.lower()

    def test_right_skewed_negative(self, engine):
        prof = NumericDistributionProfile(
            column="x", count=100, mean=-10, median=-5, std=5, cv=0.0,
            min=-50, max=0, skewness=3.0
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine._transformation_recommendations([fp])
        assert any("yeo" in r.action.lower() for r in recs)

    def test_custom_skewness_threshold(self, engine_custom_config):
        # Custom config sets skewness_threshold to 2.0, so skew=1.8 should be ignored
        prof = NumericDistributionProfile(
            column="x", count=100, mean=10, median=5, std=5, cv=0.5,
            min=1, max=50, skewness=1.8
        )
        fp = FeatureProfile(column="x", dtype="float64", numeric_profile=prof)
        recs = engine_custom_config._transformation_recommendations([fp])
        assert recs == []


# ---------------------------------------------------------------------------
# Scaling recommendation
# ---------------------------------------------------------------------------
class TestScalingRecommendation:
    def test_no_numeric(self, engine):
        rec = engine._scaling_recommendations([], None)
        assert "no numeric features" in rec.action.lower()

    def test_with_numeric(self, engine, numeric_profile):
        fp = FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)
        rec = engine._scaling_recommendations([fp], None)
        assert "scale" in rec.action.lower()


# ---------------------------------------------------------------------------
# Encoding recommendations
# ---------------------------------------------------------------------------
class TestEncodingRecommendations:
    def test_categorical_low_cardinality(self, engine):
        cp = CategoricalProfile(column="cat", unique_count=3)
        fp = FeatureProfile(column="cat", dtype="object", categorical_profile=cp)
        recs = engine._encoding_recommendations([fp])
        assert any("one-hot" in r.action.lower() for r in recs)

    def test_categorical_high_cardinality(self, engine):
        cp = CategoricalProfile(column="cat", unique_count=100)
        fp = FeatureProfile(column="cat", dtype="object", categorical_profile=cp)
        recs = engine._encoding_recommendations([fp])
        assert any("frequency" in r.action.lower() for r in recs)

    def test_numeric_categorical_like(self, engine):
        np = NumericDistributionProfile(
            column="x", count=100, mean=1, median=1, std=0.5, cv=0.0,
            min=0, max=2, unique_count=3, is_categorical_like=True
        )
        fp = FeatureProfile(column="x", dtype="int64", numeric_profile=np)
        recs = engine._encoding_recommendations([fp])
        assert any("categorical" in r.action.lower() for r in recs)

    def test_binary_categorical(self, engine):
        cp = CategoricalProfile(column="bin", unique_count=2)
        fp = FeatureProfile(column="bin", dtype="object", categorical_profile=cp)
        recs = engine._encoding_recommendations([fp])
        assert any("binary" in r.action.lower() or "0/1" in r.action.lower() for r in recs)

    def test_low_cardinality_threshold_from_config(self, engine_custom_config):
        # Custom low_cardinality_threshold is 10; 8 categories -> one-hot
        cp = CategoricalProfile(column="cat", unique_count=8)
        fp = FeatureProfile(column="cat", dtype="object", categorical_profile=cp)
        recs = engine_custom_config._encoding_recommendations([fp])
        assert any("one-hot" in r.action.lower() for r in recs)


# ---------------------------------------------------------------------------
# Feature selection (correlation) recommendations
# ---------------------------------------------------------------------------
class TestFeatureSelectionRecommendations:
    def test_no_correlation(self, engine):
        recs = engine._correlation_recommendations([])
        assert recs == []

    def test_high_correlation(self, engine):
        cp = CorrelationPair("a", "b", 0.95)
        recs = engine._correlation_recommendations([cp])
        assert any("drop" in r.action.lower() for r in recs)

    def test_below_threshold_ignored(self, engine):
        cp = CorrelationPair("a", "b", 0.5)  # below default 0.7
        recs = engine._correlation_recommendations([cp])
        assert recs == []

    def test_custom_correlation_threshold(self, engine_custom_config):
        # threshold set to 0.8, so 0.75 is ignored, 0.9 is included
        cp_low = CorrelationPair("a", "b", 0.75)
        cp_high = CorrelationPair("c", "d", 0.9)
        recs = engine_custom_config._correlation_recommendations([cp_low, cp_high])
        assert len(recs) == 1
        assert "c" in recs[0].action.lower()


# ---------------------------------------------------------------------------
# Feature engineering recommendations
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_disabled(self, engine):
        engine.enable_feature_engineering = False
        recs = engine._feature_engineering_recommendations([], [CorrelationPair("a", "b", 0.9)])
        assert recs == []

    def test_no_strong_correlation(self, engine):
        recs = engine._feature_engineering_recommendations([], [CorrelationPair("a", "b", 0.5)])
        assert recs == []  # below default 0.7 threshold

    def test_strong_correlation_suggestion(self, engine):
        cp = CorrelationPair("a", "b", 0.95)
        recs = engine._feature_engineering_recommendations([], [cp])
        assert any("redundant" in r.action.lower() for r in recs)

    def test_cv_ratio_suggestion(self, engine):
        prof1 = NumericDistributionProfile(
            column="x", count=100, mean=10, median=5, std=2, cv=0.2,
            min=1, max=20, skewness=0.0
        )
        prof2 = NumericDistributionProfile(
            column="y", count=100, mean=20, median=10, std=4, cv=0.2,
            min=2, max=40, skewness=0.0
        )
        fp1 = FeatureProfile(column="x", dtype="float64", numeric_profile=prof1)
        fp2 = FeatureProfile(column="y", dtype="float64", numeric_profile=prof2)
        recs = engine._feature_engineering_recommendations([fp1, fp2], [])
        # Should suggest a ratio (if engine enabled)
        assert any("ratio" in r.action.lower() for r in recs)


# ---------------------------------------------------------------------------
# Model recommendations
# ---------------------------------------------------------------------------
class TestModelRecommendations:
    @pytest.fixture
    def dummy_feature_profiles(self):
        # A simple numeric feature profile
        np = NumericDistributionProfile(
            column="f1", count=100, mean=10, median=10, std=2, cv=0.2,
            min=5, max=15, skewness=0.5
        )
        return [FeatureProfile(column="f1", dtype="float64", numeric_profile=np)]

    def test_no_target_profile_returns_na(self, engine, dummy_feature_profiles):
        recs = engine._model_recommendations(
            target_profile=None,
            feature_profiles=dummy_feature_profiles,
            correlation_pairs=[],
            outlier_reports=[],
            missing_report=MissingReport(0)
        )
        assert len(recs) == 1
        assert recs[0].model_name == "N/A"
        assert recs[0].suitability == "none"

    def test_regression_models_present(self, engine, target_profile_regression, dummy_feature_profiles):
        recs = engine._model_recommendations(
            target_profile=target_profile_regression,
            feature_profiles=dummy_feature_profiles,
            correlation_pairs=[],
            outlier_reports=[],
            missing_report=MissingReport(0),
            metadata=DatasetMetadata(500, 5, 0.5, {})
        )
        model_names = [m.model_name for m in recs]
        assert "LinearRegression" in model_names
        assert "RandomForestRegressor" in model_names

    def test_binary_classification_models(self, engine, target_profile_binary, dummy_feature_profiles):
        recs = engine._model_recommendations(
            target_profile=target_profile_binary,
            feature_profiles=dummy_feature_profiles,
            correlation_pairs=[],
            outlier_reports=[],
            missing_report=MissingReport(0),
            metadata=DatasetMetadata(500, 5, 0.5, {})
        )
        model_names = [m.model_name for m in recs]
        assert "LogisticRegression" in model_names

    def test_multiclass_models(self, engine, dummy_feature_profiles):
        tp = TargetProfile(column="target", dtype="int64", n_unique=5, is_regression=False, is_binary=False)
        recs = engine._model_recommendations(
            target_profile=tp,
            feature_profiles=dummy_feature_profiles,
            correlation_pairs=[],
            outlier_reports=[],
            missing_report=MissingReport(0),
            metadata=DatasetMetadata(500, 5, 0.5, {})
        )
        model_names = [m.model_name for m in recs]
        assert any("Logistic" in name for name in model_names)

    def test_missing_values_suggests_histgb(self, engine, target_profile_regression, dummy_feature_profiles):
        missing = MissingReport(20, columns_with_missing=["f1"], column_reports=[
            MissingColumnReport("f1", 20, 20.0)
        ])
        recs = engine._model_recommendations(
            target_profile=target_profile_regression,
            feature_profiles=dummy_feature_profiles,
            correlation_pairs=[],
            outlier_reports=[],
            missing_report=missing,
            metadata=DatasetMetadata(500, 5, 0.5, {})
        )
        model_names = [m.model_name for m in recs]
        assert "HistGradientBoostingRegressor" in model_names

    def test_xgboost_included_if_available(self, engine, target_profile_binary, dummy_feature_profiles):
        import preml.recommendation_engine as rec_mod
        with patch.object(rec_mod, 'XGBOOST_AVAILABLE', True):
            recs = engine._model_recommendations(
                target_profile=target_profile_binary,
                feature_profiles=dummy_feature_profiles,
                correlation_pairs=[],
                outlier_reports=[],
                missing_report=MissingReport(0),
                metadata=DatasetMetadata(500, 5, 0.5, {})
            )
            assert any("XGB" in m.model_name for m in recs)

    def test_lightgbm_included_if_available(self, engine, target_profile_regression, dummy_feature_profiles):
        import preml.recommendation_engine as rec_mod
        with patch.object(rec_mod, 'LIGHTGBM_AVAILABLE', True):
            recs = engine._model_recommendations(
                target_profile=target_profile_regression,
                feature_profiles=dummy_feature_profiles,
                correlation_pairs=[],
                outlier_reports=[],
                missing_report=MissingReport(0),
                metadata=DatasetMetadata(500, 5, 0.5, {})
            )
            assert any("LGBM" in m.model_name for m in recs)


# ---------------------------------------------------------------------------
# Pipeline suggestion
# ---------------------------------------------------------------------------
class TestPipelineSuggestion:
    def test_empty_pipeline(self, engine):
        pipeline = engine._pipeline_suggestion([])
        assert isinstance(pipeline, PipelineSuggestion)
        # Should still return a valid object, possibly with "passthrough"
        assert len(pipeline.steps) >= 1

    def test_pipeline_with_numeric_and_outliers(self, engine, numeric_profile):
        fp = FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)
        pipeline = engine._pipeline_suggestion(
            [fp],
            missing_report=MissingReport(0),
            outlier_columns=["num"],
            transformation_recs=[]
        )
        steps = dict(pipeline.steps)
        assert "scaler" in steps
        assert "RobustScaler" in steps["scaler"]

    def test_pipeline_with_missing_values(self, engine, numeric_profile):
        missing = MissingReport(10, columns_with_missing=["num"], column_reports=[
            MissingColumnReport("num", 10, 10.0)
        ])
        fp = FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)
        pipeline = engine._pipeline_suggestion(
            [fp],
            missing_report=missing,
            outlier_columns=[],
            transformation_recs=[]
        )
        steps = dict(pipeline.steps)
        assert "imputation" in steps

    def test_pipeline_with_skew_transformation(self, engine, numeric_profile):
        # Create a transformation recommendation (simulate output)
        trans_rec = Recommendation(
            category="transformation",
            action="apply log",
            confidence=0.9,
            evidence=[],
        )
        fp = FeatureProfile(column="num", dtype="float64", numeric_profile=numeric_profile)
        pipeline = engine._pipeline_suggestion(
            [fp],
            missing_report=MissingReport(0),
            outlier_columns=[],
            transformation_recs=[trans_rec]
        )
        steps = dict(pipeline.steps)
        assert "transformation" in steps
        assert "log" in steps["transformation"].lower() or "powertransformer" in steps["transformation"].lower()


# ---------------------------------------------------------------------------
# Integration: generate_recommendations output structure
# ---------------------------------------------------------------------------
class TestGenerateRecommendations:
    def test_returns_all_categories(self, engine, feature_profiles, target_profile_binary):
        analysis = _make_minimal_analysis(
            feature_profiles=feature_profiles,
            target_profile=target_profile_binary,
        )
        result = engine.generate_recommendations(analysis)
        expected_cats = [
            "imputation", "outlier_handling", "transformation", "scaling",
            "encoding", "feature_engineering", "feature_selection",
            "pipeline", "models", "data_quality_notes"
        ]
        for cat in expected_cats:
            assert cat in result

    def test_pipeline_suggestion_type(self, engine, feature_profiles):
        analysis = _make_minimal_analysis(feature_profiles=feature_profiles)
        result = engine.generate_recommendations(analysis)
        assert isinstance(result["pipeline"], PipelineSuggestion)

    def test_model_recommendations_type(self, engine, feature_profiles, target_profile_binary):
        analysis = _make_minimal_analysis(
            feature_profiles=feature_profiles,
            target_profile=target_profile_binary
        )
        result = engine.generate_recommendations(analysis)
        assert all(isinstance(m, ModelRecommendation) for m in result["models"])

    def test_data_quality_notes_populated(self, engine):
        duplicates = DuplicateReport(5, 5.0)
        missing = MissingReport(10, columns_with_missing=["c1"], column_reports=[
            MissingColumnReport("c1", 10, 10.0)
        ])
        analysis = _make_minimal_analysis(duplicates=duplicates, missing=missing)
        result = engine.generate_recommendations(analysis)
        notes = result["data_quality_notes"]
        assert any("duplicate" in n.lower() for n in notes)
        assert any("missing" in n.lower() for n in notes)


# ---------------------------------------------------------------------------
# Summarize method
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_summarize_returns_string(self, engine, feature_profiles, target_profile_binary):
        analysis = _make_minimal_analysis(
            feature_profiles=feature_profiles,
            target_profile=target_profile_binary,
        )
        result = engine.generate_recommendations(analysis)
        summary = engine.summarize(result)
        assert isinstance(summary, str)
        # Contains major sections
        assert "DATA QUALITY NOTES" in summary
        assert "MODEL RECOMMENDATIONS" in summary
        assert "SUGGESTED PIPELINE" in summary
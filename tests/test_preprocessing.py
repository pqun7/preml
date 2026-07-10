"""
Unit tests for preml.preprocessing.

Tests cover:
- PreprocessingBuilder pipeline creation
- Column type detection (numeric, categorical, categorical-like)
- Imputation, scaling, and transformation strategies based on recommendations
- fit_transform method
- Error handling
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler

from preml.config import MLToolkitConfig
from preml.exceptions import PreprocessingError
from preml.preprocessing import PreprocessingBuilder
from preml.schema import (
    FeatureProfile,
    NumericDistributionProfile,
    CategoricalProfile,
    Recommendation,
    OutlierReport,
)


# ------------------- Helpers -------------------
def _make_minimal_analysis(**overrides):
    base = {
        "feature_profiles": [],
        "target_profile": None,
        "recommendations": {
            "imputation": [],
            "outlier_handling": [],
            "transformation": [],
            "scaling": {},
            "encoding": [],
        },
        "duplicates": None,
        "metadata": None,
        "missing": None,
        "outliers": [],
        "correlation_pairs": [],
    }
    base.update(overrides)
    return base


def _numeric_profile(column, skewness=0.2, is_categorical_like=False,
                     unique_count=100, std=1.0, median=0.0):
    return NumericDistributionProfile(
        column=column, count=100, mean=0.0, median=median, std=std, cv=0.0,
        min=-3, max=3, skewness=skewness, kurtosis=0.0, zero_percent=0.0,
        negative_percent=0.0, is_categorical_like=is_categorical_like,
        unique_count=unique_count,
    )


def _categorical_profile(column, unique_count=5):
    return CategoricalProfile(
        column=column, unique_count=unique_count,
        missing_count=0, missing_percent=0.0, mode="A",
    )


# ------------------- Fixtures -------------------
@pytest.fixture
def simple_analysis():
    """A simple analysis with one numeric and one categorical column."""
    profiles = [
        FeatureProfile(column="num", dtype="float64",
                       numeric_profile=_numeric_profile("num")),
        FeatureProfile(column="cat", dtype="object",
                       categorical_profile=_categorical_profile("cat")),
    ]
    return _make_minimal_analysis(feature_profiles=profiles)


@pytest.fixture
def builder(simple_analysis):
    return PreprocessingBuilder(simple_analysis)


# ------------------- Tests -------------------
class TestPreprocessingBuilder:
    def test_column_type_detection(self, builder):
        assert builder.numeric_cols == ["num"]
        assert builder.categorical_cols == ["cat"]
        assert builder.categorical_like_cols == []

    def test_build_pipeline_returns_column_transformer(self, builder):
        pipe = builder.build_pipeline()
        assert isinstance(pipe, ColumnTransformer)

    def test_fit_transform_returns_ndarray(self, builder):
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["A", "B", "A", "C", "B"]
        })
        transformed = builder.fit_transform(df)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == 5

    def test_constant_columns_are_dropped(self):
        profiles = [
            FeatureProfile(column="const", dtype="float64",
                           numeric_profile=_numeric_profile("const", std=0.0),
                           is_constant=True),
            FeatureProfile(column="num", dtype="float64",
                           numeric_profile=_numeric_profile("num")),
        ]
        analysis = _make_minimal_analysis(feature_profiles=profiles)
        builder = PreprocessingBuilder(analysis)
        assert "const" not in builder.numeric_cols

    def test_categorical_like_numeric_handled(self):
        np_like = _numeric_profile("id", is_categorical_like=True, unique_count=10)
        profiles = [FeatureProfile(column="id", dtype="int64", numeric_profile=np_like)]
        analysis = _make_minimal_analysis(feature_profiles=profiles)
        builder = PreprocessingBuilder(analysis)
        assert builder.categorical_like_cols == ["id"]

    def test_skewed_transformation_applied(self):
        skewed_prof = _numeric_profile("skewed", skewness=2.0)
        profiles = [FeatureProfile(column="skewed", dtype="float64",
                                   numeric_profile=skewed_prof)]
        rec = Recommendation(
            category="transformation",
            action="Apply log transform to skewed",
            confidence=0.9,
            evidence=[],
        )
        analysis = _make_minimal_analysis(
            feature_profiles=profiles,
            recommendations={"transformation": [rec]}
        )
        builder = PreprocessingBuilder(analysis)
        pipe = builder.build_pipeline()

        # Fit on dummy data so named_transformers_ becomes available
        df = pd.DataFrame({"skewed": [1.0, 2.0, 3.0, 4.0, 5.0]})
        pipe.fit(df)

        # The sub-pipeline for skewed columns is named "num_skewed"
        # and should contain a PowerTransformer step.
        assert any(isinstance(step[1], PowerTransformer)
                   for step in pipe.named_transformers_["num_skewed"].steps)

    def test_outlier_detection_triggers_robust_scaler(self):
        num_prof = _numeric_profile("x")
        profiles = [FeatureProfile(column="x", dtype="float64",
                                   numeric_profile=num_prof)]
        rec = Recommendation(
            category="outlier_handling",
            action="Handle outliers for x",
            confidence=0.8,
            evidence=[],
        )
        analysis = _make_minimal_analysis(
            feature_profiles=profiles,
            recommendations={"outlier_handling": [rec]}
        )
        builder = PreprocessingBuilder(analysis)
        pipe = builder.build_pipeline()

        # Fit on dummy data so named_transformers_ becomes available
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        pipe.fit(df)

        # The numeric pipeline for non-skewed columns is "num_normal"
        num_pipe = pipe.named_transformers_["num_normal"]
        assert any(isinstance(step[1], RobustScaler) for step in num_pipe.steps)

    def test_empty_after_dropping_constants_raises(self):
        analysis = _make_minimal_analysis(feature_profiles=[])
        builder = PreprocessingBuilder(analysis)
        with pytest.raises(PreprocessingError):
            builder.build_pipeline()
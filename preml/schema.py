"""Data models (dataclasses) for preml.

All classes in this module are pure data containers. They hold facts,
never compute them. This strict separation allows the statistics engine,
recommendation engine, and reporting modules to work with well‑defined,
type‑safe objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Datetime / metadata helpers (used inside profiles)
# ----------------------------------------------------------------------
@dataclass
class DatasetMetadata:
    """Lightweight metadata describing the dataset as a whole.

    Attributes:
        n_rows: Number of rows.
        n_columns: Number of columns.
        memory_mb: Memory usage in megabytes.
        column_types: Mapping from dtype to count.
    """
    n_rows: int
    n_columns: int
    memory_mb: float
    column_types: Dict[str, int]


@dataclass
class DuplicateReport:
    """Information about duplicate rows.

    Attributes:
        total_duplicates: Count of completely duplicated rows.
        duplicate_percent: Percentage of rows that are duplicates.
        sample_indices: Indices of duplicate rows (capped at 100).
    """
    total_duplicates: int
    duplicate_percent: float
    sample_indices: List[int] = field(default_factory=list)


@dataclass
class InfiniteReport:
    """Detection of infinite values in the dataset.

    Attributes:
        columns_with_inf: Names of columns containing +-inf values.
        counts: Dict mapping column name to number of inf values.
    """
    columns_with_inf: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Missingness & Outlier reports (facts only)
# ----------------------------------------------------------------------
@dataclass
class MissingColumnReport:
    """Missing value statistics for a single column.

    Attributes:
        column: Column name.
        missing_count: Number of missing values.
        missing_percent: Percentage of missing values.
        pattern: Optional label, e.g., 'MCAR', 'MAR', 'MNAR' (if
            statistical tests are applied in the future; currently None).
    """
    column: str
    missing_count: int
    missing_percent: float
    pattern: Optional[str] = None


@dataclass
class MissingReport:
    """Aggregate missing value information for the whole dataset.

    Attributes:
        total_missing: Sum of all missing values across all columns.
        columns_with_missing: List of column names that have at least one missing.
        column_reports: Per‑column detailed reports.
    """
    total_missing: int
    columns_with_missing: List[str] = field(default_factory=list)
    column_reports: List[MissingColumnReport] = field(default_factory=list)


@dataclass
class OutlierReport:
    """Outlier detection results for a single column.

    Attributes:
        column: Column name.
        method: Method used ('iqr' or 'zscore').
        outlier_count: Number of detected outliers.
        outlier_percent: Percentage of data that are outliers.
        lower_bound: Lower bound of the normal range (if applicable).
        upper_bound: Upper bound of the normal range (if applicable).
    """
    column: str
    method: str
    outlier_count: int
    outlier_percent: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


# ----------------------------------------------------------------------
# Distribution & statistical profile (facts)
# ----------------------------------------------------------------------
@dataclass
class NumericDistributionProfile:
    """Statistical facts about a numeric column.

    Attributes:
        column: Column name.
        count: Number of non‑null values.
        mean: Mean value.
        median: Median value.
        std: Standard deviation.
        cv: Coefficient of variation.
        min: Minimum value.
        max: Maximum value.
        percentiles: Dict of percentile -> value (e.g., {1: ...}).
        skewness: Sample skewness.
        kurtosis: Sample excess kurtosis.
        zero_percent: Percentage of values equal to zero.
        negative_percent: Percentage of values less than zero.
        is_categorical_like: True if the number of unique values is small
            enough to consider the column as categorical.
        unique_count: Number of distinct values.
    """
    column: str
    count: int
    mean: float
    median: float
    std: float
    cv: float
    min: float
    max: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    skewness: float = 0.0
    kurtosis: float = 0.0
    zero_percent: float = 0.0
    negative_percent: float = 0.0
    is_categorical_like: bool = False
    unique_count: int = 0


@dataclass
class CategoricalProfile:
    """Statistical facts about a categorical / object column.

    Attributes:
        column: Column name.
        unique_count: Number of distinct categories.
        top_categories: List of (category, count) for the 5 most frequent.
        missing_count: Number of missing values in the column.
        missing_percent: Percentage of missing values.
        mode: Most frequent category (or None if column is empty).
    """
    column: str
    unique_count: int
    top_categories: List[Dict[str, Union[str, int]]] = field(default_factory=list)
    missing_count: int = 0
    missing_percent: float = 0.0
    mode: Optional[Any] = None


@dataclass
class FeatureProfile:
    """Unified profile for a single column (numeric or categorical).

    One of `numeric_profile` or `categorical_profile` will be set.
    """
    column: str
    dtype: str
    numeric_profile: Optional[NumericDistributionProfile] = None
    categorical_profile: Optional[CategoricalProfile] = None
    is_constant: bool = False
    is_quasi_constant: bool = False


# ----------------------------------------------------------------------
# Correlation & multicollinearity (facts)
# ----------------------------------------------------------------------
@dataclass
class CorrelationPair:
    """A pair of features and their Pearson correlation coefficient.

    Attributes:
        feature_a: First column name.
        feature_b: Second column name.
        coefficient: Correlation coefficient (float).
    """
    feature_a: str
    feature_b: str
    coefficient: float


# ----------------------------------------------------------------------
# Target analysis (facts)
# ----------------------------------------------------------------------
@dataclass
class TargetProfile:
    """Facts about the target variable.

    Attributes:
        column: Target column name.
        dtype: Data type of the target.
        n_unique: Number of unique values (for regression: high; for
            classification: number of classes).
        missing_count: Number of missing values in the target.
        missing_percent: Percentage of missing target values.
        is_regression: True if target is numeric and appears continuous.
        is_binary: True if target has exactly 2 unique values.
        class_distribution: For classification, mapping of class label
            to count (sorted by count descending).
    """
    column: str
    dtype: str
    n_unique: int
    missing_count: int = 0
    missing_percent: float = 0.0
    is_regression: bool = False
    is_binary: bool = False
    class_distribution: Dict[Any, int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Recommendation data models (decisions, but pure data)
# ----------------------------------------------------------------------
@dataclass
class Evidence:
    """Supporting evidence for a recommendation.

    Attributes:
        reason: Short description of why this recommendation is made.
        statistics: Dict of statistic name -> value that backs the decision.
    """
    reason: str
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """A single actionable recommendation.

    Attributes:
        category: Type of recommendation (e.g., 'imputation', 'scaling',
            'encoding', 'transformation', 'feature_engineering',
            'feature_selection', 'model', 'general').
        action: Human‑readable suggestion.
        confidence: Float in [0, 1] indicating how confident the engine is
            about this suggestion (based on evidence).
        evidence: List of Evidence objects supporting the recommendation.
        alternative_options: List of alternative actions the user could take.
        risks: List of potential pitfalls if this recommendation is followed.
    """
    category: str
    action: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    alternative_options: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class PipelineSuggestion:
    """A complete preprocessing pipeline suggestion.

    Attributes:
        name: A short descriptive name (e.g., 'Standard numeric pipeline').
        steps: List of (step_name, transformer_or_estimator) tuples
            compatible with sklearn Pipeline.
        explanation: Why this pipeline was suggested.
    """
    name: str
    steps: List[Any]  # In practice, list of (str, transformer) tuples
    explanation: str = ""


@dataclass
class ModelRecommendation:
    """A model recommendation for a given task.

    Attributes:
        model_name: Name of the model (e.g., 'RandomForestRegressor').
        suitability: Qualitative judgement ('excellent', 'good', 'baseline').
        reason: Why this model is recommended.
        hyperparams: Suggested hyperparameters (as a dict).
        conditions: Pre‑conditions that must be met for the model to work well.
    """
    model_name: str
    suitability: str
    reason: str = ""
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
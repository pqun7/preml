# config.py
"""Central configuration module for preml.

This module defines all tunable parameters, thresholds, and defaults
used throughout the library. It is designed to be imported anywhere
without creating circular dependencies and does not rely on any
heavy external packages (only the standard library).
"""

from dataclasses import dataclass
from typing import Literal, Tuple

import pandas as pd


@dataclass
class MLToolkitConfig:
    """Immutable configuration container for preml.

    All parameters have sensible defaults, but can be overridden
    by creating a new instance or modifying attributes directly.

    Attributes:
        missing_threshold: Ratio above which a column is flagged as
            having high missingness (0.0 to 1.0).
        high_cardinality_threshold: Number of unique values above which
            a categorical column is considered high cardinality.
        max_unique_for_categorical_like: Maximum unique values for a
            numeric column to be considered as categorical-like.
        correlation_threshold: Absolute correlation coefficient above
            which a pair of features is flagged as highly correlated.
        skewness_threshold: Absolute skewness above which a distribution
            is considered heavily skewed.
        cv_threshold: Coefficient of Variation above which a feature
            is marked as highly dispersed.
        zero_percent_threshold: Percentage of zeros in a numeric column
            that triggers a "consider binary flag" suggestion.
        negative_percent_threshold: Similar for negative values.
        outlier_method: Method used for outlier detection.
            'iqr': Interquartile Range (robust).
            'zscore': Z‑score (sensitive to skew, generally discouraged).
        iqr_multiplier: Multiplier for IQR bounds (default 1.5).
        zscore_threshold: Threshold for Z‑score outlier detection.
        constant_variance_threshold: Variance below this value is
            considered constant/quasi‑constant.
        enable_feature_engineering: If True, the recommendation engine
            may suggest feature engineering techniques (ratios, interactions,
            etc.). They are always based on statistical evidence, never on
            column names.
        max_plot_cols: Maximum number of individual feature plots to
            generate in one go (avoid overwhelming the user).
        plot_style: Seaborn style to use for all visualizations.
        color_palette: Seaborn color palette.
        figure_size: Default (width, height) for main figures.
        random_state: Random seed for reproducibility in imputation,
            encoding, etc.
    """

    # ----------------------------------------------------------------
    # Missingness & Data Quality
    # ----------------------------------------------------------------
    missing_threshold: float = 0.25
    low_cardinality_threshold: int = 10
    high_cardinality_threshold: int = 50
    max_unique_for_categorical_like: int = 15

    # ----------------------------------------------------------------
    # Correlation & Distribution
    # ----------------------------------------------------------------
    correlation_threshold: float = 0.8
    skewness_threshold: float = 1.0
    cv_threshold: float = 2.0
    zero_percent_threshold: float = 50.0
    negative_percent_threshold: float = 5.0

    # ----------------------------------------------------------------
    # Outlier Detection
    # ----------------------------------------------------------------
    outlier_threshold_percent: float = 5.0
    outlier_method: Literal["iqr", "zscore"] = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0

    # ----------------------------------------------------------------
    # Variance & Constants
    # ----------------------------------------------------------------
    constant_variance_threshold: float = 1e-6

    # ----------------------------------------------------------------
    # Feature Engineering Toggle
    # ----------------------------------------------------------------
    enable_feature_engineering: bool = True

    # ----------------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------------
    max_plot_cols: int = 20
    plot_style: str = "whitegrid"
    color_palette: str = "muted"
    figure_size: Tuple[int, int] = (12, 6)

    # ----------------------------------------------------------------
    # Reproducibility
    # ----------------------------------------------------------------
    random_state: int = 42
    n_jobs: int = -1

    def adapt_to_dataset(self, df: pd.DataFrame) -> "MLToolkitConfig":
        """Adapt threshold defaults based on dataset shape and profile.

        This method updates the current instance in-place and returns it,
        allowing fluent usage:

        ``config = MLToolkitConfig().adapt_to_dataset(df)``
        """
        if not isinstance(df, pd.DataFrame):
            return self
        if df.empty:
            return self

        n_rows, n_cols = df.shape
        numeric_ratio = (
            float(len(df.select_dtypes(include=["number"]).columns)) / n_cols
            if n_cols > 0
            else 0.0
        )

        # Large, sparse datasets benefit from stricter missingness handling.
        if n_rows >= 100_000:
            self.missing_threshold = min(self.missing_threshold, 0.2)
            self.high_cardinality_threshold = max(self.high_cardinality_threshold, 100)

        # Small datasets: avoid over-reacting to correlation/skew noise.
        if n_rows < 2_000:
            self.correlation_threshold = max(self.correlation_threshold, 0.85)
            self.skewness_threshold = max(self.skewness_threshold, 1.2)

        # Very wide datasets: keep correlation filtering practical.
        if n_cols >= 200:
            self.correlation_threshold = min(self.correlation_threshold, 0.75)

        # Mostly categorical data often needs lower high-cardinality split.
        if numeric_ratio < 0.35:
            self.high_cardinality_threshold = min(self.high_cardinality_threshold, 40)

        return self


# ----------------------------------------------------------------------
# Default configuration instance
# ----------------------------------------------------------------------
# Users can import this singleton and modify attributes at runtime,
# or pass their own MLToolkitConfig instance to functions that accept one.
default_config = MLToolkitConfig()
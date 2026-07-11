"""Preprocessing layer — builds scikit‑learn compatible pipelines from
statistical evidence and recommendations.

This module consumes the output of the EDA analysis and creates
a sklearn ColumnTransformer / Pipeline tailored to the dataset.
It does NOT recompute statistics; it only translates facts and
recommendations into preprocessing steps.

Example
-------
>>> from preml.eda import EDAAnalyzer
>>> from preml.preprocessing import PreprocessingBuilder
>>> analyzer = EDAAnalyzer(df, target='price')
>>> result = analyzer.run()
>>> builder = PreprocessingBuilder(result)
>>> pipeline = builder.build_pipeline()
>>> X_transformed = builder.fit_transform(df)  # or pipeline.fit_transform(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from preml._analysis import resolve_analysis_result
from preml.config import MLToolkitConfig, default_config
from preml.exceptions import PreprocessingError
from preml.schema import (
    FeatureProfile,
    OutlierReport,
    Recommendation,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _get_feature_profiles(
    analysis_result: Dict[str, Any],
) -> List[FeatureProfile]:
    return analysis_result.get("feature_profiles", [])


def _get_outlier_reports(
    analysis_result: Dict[str, Any],
) -> List[OutlierReport]:
    return analysis_result.get("outliers", [])


def _get_recommendations(
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    return analysis_result.get("recommendations", {})


# ------------------------------------------------------------------
# PreprocessingBuilder
# ------------------------------------------------------------------
class PreprocessingBuilder:
    """Translates analysis results into a sklearn preprocessing pipeline.

    Parameters
    ----------
    analysis_result : dict
        The dictionary returned by `EDAAnalyzer.run()`.
    config : MLToolkitConfig, optional
        Configuration object (used for random state and thresholds).

    Raises
    ------
    PreprocessingError
        If required keys are missing from *analysis_result*.
    """

    def __init__(
        self,
        analysis_result: Dict[str, Any],
        config: Optional[MLToolkitConfig] = None,
    ) -> None:
        try:
            analysis_result = resolve_analysis_result(analysis_result)
        except TypeError as exc:
            raise PreprocessingError(
                "analysis_result must be a dictionary (the output of EDAAnalyzer.run()) or an EDAAnalyzer instance."
            ) from exc

        # Validate required sections
        if "feature_profiles" not in analysis_result:
            raise PreprocessingError(
                "Missing key 'feature_profiles' in analysis_result."
            )
        if "recommendations" not in analysis_result:
            raise PreprocessingError(
                "Missing key 'recommendations' in analysis_result."
            )

        self.analysis = analysis_result
        self.config = config or default_config
        self.feature_profiles = _get_feature_profiles(analysis_result)
        self.recommendations = _get_recommendations(analysis_result)
        self._fitted_pipeline: Optional[ColumnTransformer] = None

        # Categorise columns
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.categorical_like_cols: List[str] = []

        for prof in self.feature_profiles:
            if prof.is_constant:
                continue  # will be dropped automatically
            if prof.numeric_profile and not prof.numeric_profile.is_categorical_like:
                self.numeric_cols.append(prof.column)
            elif prof.numeric_profile and prof.numeric_profile.is_categorical_like:
                self.categorical_like_cols.append(prof.column)
            elif prof.categorical_profile:
                self.categorical_cols.append(prof.column)

        # Determine which numeric columns are skewed based on actual skewness
        self.skewed_cols: List[str] = []
        for prof in self.feature_profiles:
            if prof.column in self.numeric_cols and prof.numeric_profile:
                if abs(prof.numeric_profile.skewness) >= self.config.skewness_threshold:
                    self.skewed_cols.append(prof.column)

        # Detect if any numeric column has outliers (using actual outlier reports)
        outlier_reports = _get_outlier_reports(analysis_result)
        self.has_outliers = any(
            o.column in self.numeric_cols and o.outlier_count > 0
            for o in outlier_reports
        ) or bool(self.recommendations.get("outlier_handling"))

        logger.debug(
            "Columns: numeric=%d skewed=%d categorical=%d cat_like=%d has_outliers=%s",
            len(self.numeric_cols),
            len(self.skewed_cols),
            len(self.categorical_cols),
            len(self.categorical_like_cols),
            self.has_outliers,
        )

    def _validate_input_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the incoming DataFrame matches expected feature columns."""
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError(
                "Input must be a pandas DataFrame.",
                details="Use the same feature DataFrame structure used during EDA analysis.",
            )

        expected_cols = {
            p.column for p in self.feature_profiles if not p.is_constant
        }
        missing = sorted(expected_cols - set(df.columns))
        if missing:
            raise PreprocessingError(
                "Input DataFrame is missing required feature columns.",
                details=(
                    f"Missing columns: {missing[:10]}. "
                    "Use the full feature set from analysis_result['feature_profiles'] "
                    "and exclude the target column from preprocessing input."
                ),
            )

    def fit(self, df: pd.DataFrame) -> "PreprocessingBuilder":
        """Fit and store a preprocessing pipeline on ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Input features DataFrame.

        Returns
        -------
        PreprocessingBuilder
            The builder instance itself for fluent usage.
        """
        self._validate_input_dataframe(df)
        pipeline = self.build_pipeline()
        pipeline.fit(df)
        self._fitted_pipeline = pipeline
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using a previously fitted pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input features DataFrame.

        Returns
        -------
        np.ndarray
            Dense transformed feature array.

        Raises
        ------
        PreprocessingError
            If the builder has not been fitted yet.
        """
        self._validate_input_dataframe(df)
        if self._fitted_pipeline is None:
            raise PreprocessingError(
                "PreprocessingBuilder is not fitted.",
                details=(
                    "Call builder.fit(df) before builder.transform(df), or use "
                    "builder.fit_transform(df) for one-step training transformations."
                ),
            )

        transformed = self._fitted_pipeline.transform(df)
        return np.asarray(transformed)

    def build_pipeline(self) -> ColumnTransformer:
        """Build a scikit‑learn ColumnTransformer with appropriate
        preprocessing steps for each column group.

        Returns
        -------
        ColumnTransformer
            A fully configured ColumnTransformer ready to fit/transform.
        """
        transformers: List[Tuple[str, Any, List[str]]] = []
        all_used_columns: List[str] = []

        # ------------------------------------------------------------------
        # 1. Numeric pipelines (split skewed / normal)
        # ------------------------------------------------------------------
        if self.numeric_cols:
            skewed = self.skewed_cols
            normal = [c for c in self.numeric_cols if c not in skewed]

            if normal:
                num_pipe = self._build_numeric_pipeline(apply_power=False)
                transformers.append(("num_normal", num_pipe, normal))
                all_used_columns.extend(normal)

            if skewed:
                num_pipe_skew = self._build_numeric_pipeline(apply_power=True)
                transformers.append(("num_skewed", num_pipe_skew, skewed))
                all_used_columns.extend(skewed)

        # ------------------------------------------------------------------
        # 2. Categorical pipelines
        # ------------------------------------------------------------------
        if self.categorical_cols:
            low_card, high_card = self._split_categorical_by_cardinality()
            if low_card:
                cat_low_pipe = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ])
                transformers.append(("cat_low", cat_low_pipe, low_card))
                all_used_columns.extend(low_card)
            if high_card:
                cat_high_pipe = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ])
                transformers.append(("cat_high", cat_high_pipe, high_card))
                all_used_columns.extend(high_card)

        # ------------------------------------------------------------------
        # 3. Categorical‑like numeric columns
        # ------------------------------------------------------------------
        if self.categorical_like_cols:
            catlike_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ])
            transformers.append(
                ("cat_like", catlike_pipe, self.categorical_like_cols)
            )
            all_used_columns.extend(self.categorical_like_cols)

        # ------------------------------------------------------------------
        # 4. Any remaining columns (should be none, but fallback)
        # ------------------------------------------------------------------
        all_dataset_cols = [
            p.column for p in self.feature_profiles if not p.is_constant
        ]
        remaining = sorted(set(all_dataset_cols) - set(all_used_columns))
        if remaining:
            # Fallback: pass-through (should not happen normally)
            transformers.append(("remainder", "passthrough", remaining))

        if not transformers:
            raise PreprocessingError(
                "No columns to preprocess after dropping constants.",
                details="Check if all features are constant.",
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # drop any column not explicitly handled
            verbose_feature_names_out=False,
        )

    def _build_numeric_pipeline(self, apply_power: bool = False) -> Pipeline:
        """Internal helper to create a numeric sub‑pipeline."""
        # Imputation strategy
        strategy = "median" if self.has_outliers else "mean"
        steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy=strategy))
        ]

        # Optional power transformation for skewed data
        if apply_power:
            steps.append(("power", PowerTransformer(method="yeo-johnson")))

        # Scaling
        if self.has_outliers:
            steps.append(("scaler", RobustScaler()))
        else:
            steps.append(("scaler", StandardScaler()))

        return Pipeline(steps=steps)

    def _split_categorical_by_cardinality(self) -> Tuple[List[str], List[str]]:
        """Split categorical columns into low and high cardinality groups.

        The threshold is taken from the config (`high_cardinality_threshold`).
        """
        threshold = self.config.high_cardinality_threshold
        low, high = [], []
        for col in self.categorical_cols:
            prof = next(
                (p for p in self.feature_profiles if p.column == col), None
            )
            if prof and prof.categorical_profile:
                n_unique = prof.categorical_profile.unique_count
            else:
                logger.warning(
                    "Categorical column '%s' not found in feature profiles; "
                    "assuming low cardinality.",
                    col,
                )
                n_unique = threshold
            if n_unique <= threshold:
                low.append(col)
            else:
                high.append(col)
        return low, high

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience method: fit pipeline on `df` and return transformed array.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (must contain the same columns as the original).

        Returns
        -------
        np.ndarray
            Transformed feature array (dense, since all encoders use
            ``sparse_output=False``).
        """
        self._validate_input_dataframe(df)
        pipeline = self.build_pipeline()
        transformed = pipeline.fit_transform(df)
        self._fitted_pipeline = pipeline
        # The pipeline guarantees dense output (no sparse matrices).
        return np.asarray(transformed)
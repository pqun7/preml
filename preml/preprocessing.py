"""Preprocessing layer — builds scikit‑learn compatible pipelines from
statistical evidence and recommendations.

This module consumes the output of the EDA analysis and creates
a sklearn ColumnTransformer / Pipeline tailored to the dataset.
It does NOT recompute statistics; it only translates facts and
recommendations into preprocessing steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from preml.config import MLToolkitConfig, default_config
from preml.exceptions import PreprocessingError
from preml.schema import (
    FeatureProfile,
    Recommendation,
    TargetProfile,
)

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _get_feature_profiles(
    analysis_result: Dict[str, Any],
) -> List[FeatureProfile]:
    return analysis_result.get("feature_profiles", [])


def _get_target_profile(
    analysis_result: Dict[str, Any],
) -> Optional[TargetProfile]:
    return analysis_result.get("target_profile")


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
        Configuration object (used for random state).
    """

    def __init__(
        self,
        analysis_result: Dict[str, Any],
        config: Optional[MLToolkitConfig] = None,
    ) -> None:
        self.analysis = analysis_result
        self.config = config or default_config
        self.feature_profiles = _get_feature_profiles(analysis_result)
        self.target_profile = _get_target_profile(analysis_result)
        self.recommendations = _get_recommendations(analysis_result)

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

        # Determine which numeric columns are flagged for transformation
        self.skewed_cols: List[str] = []
        transformation_recs = self.recommendations.get("transformation", [])
        for rec in transformation_recs:
            for col in self.numeric_cols:
                if col in rec.action:
                    self.skewed_cols.append(col)

        # Detect if any numeric column has outliers (for scaling strategy)
        self.has_outliers = False
        outlier_recs = self.recommendations.get("outlier_handling", [])
        for rec in outlier_recs:
            if any(col in rec.action for col in self.numeric_cols):
                self.has_outliers = True
                break

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
            # Split by cardinality
            low_card, high_card = self._split_categorical_by_cardinality()
            if low_card:
                cat_low_pipe = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ])
                transformers.append(("cat_low", cat_low_pipe, low_card))
                all_used_columns.extend(low_card)
            if high_card:
                cat_high_pipe = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ])
                transformers.append(("cat_high", cat_high_pipe, high_card))
                all_used_columns.extend(high_card)

        # ------------------------------------------------------------------
        # 3. Categorical‑like numeric columns
        # ------------------------------------------------------------------
        if self.categorical_like_cols:
            catlike_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat_like", catlike_pipe, self.categorical_like_cols))
            all_used_columns.extend(self.categorical_like_cols)

        # ------------------------------------------------------------------
        # 4. Any remaining columns (should be none, but fallback)
        # ------------------------------------------------------------------
        all_dataset_cols = [p.column for p in self.feature_profiles if not p.is_constant]
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
        steps = []
        steps.append(("imputer", SimpleImputer(strategy=strategy)))

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
        low = []
        high = []
        for col in self.categorical_cols:
            prof = next((p for p in self.feature_profiles if p.column == col), None)
            if prof and prof.categorical_profile:
                n_unique = prof.categorical_profile.unique_count
            else:
                # Fallback: assume low cardinality
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
            Transformed feature array.
        """
        pipeline = self.build_pipeline()
        transformed = pipeline.fit_transform(df)
        return np.asarray(transformed)
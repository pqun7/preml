"""
Facts extraction layer — computes statistical profiles without
making any decisions or recommendations.

All functions and methods in this module are purely computational.
They accept a DataFrame and configuration, and return dataclass
instances from `preml.schema`. No side effects, no global state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from preml.config import MLToolkitConfig, default_config
from preml.exceptions import DataValidationError
from preml.schema import (
    CategoricalProfile,
    CorrelationPair,
    DatasetMetadata,
    DuplicateReport,
    FeatureProfile,
    InfiniteReport,
    MissingColumnReport,
    MissingReport,
    NumericDistributionProfile,
    OutlierReport,
    TargetProfile,
)


class StatisticsEngine:
    """Computes factual statistics about a tabular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target : str, optional
        Name of the target column. If provided, the engine will compute
        a `TargetProfile`.
    config : MLToolkitConfig, optional
        Configuration for thresholds and methods. If not supplied, the
        global `default_config` is used.

    Raises
    ------
    DataValidationError
        If `df` is not a pandas DataFrame, or if `target` is not a column
        in `df`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[MLToolkitConfig] = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                "Input must be a pandas DataFrame.",
                details=type(df),
            )
        if df.empty or df.shape[1] == 0:
            raise DataValidationError(
                "Input DataFrame is empty.",
                details=(
                    "PreML requires at least one row and one column to perform analysis. "
                    "Load data and verify filtering steps before calling StatisticsEngine."
                ),
            )
        if target is not None and target not in df.columns:
            raise DataValidationError(
                f"Target column '{target}' not found in DataFrame.",
                details=(
                    f"Available columns (sample): {list(df.columns[:10])}. "
                    "Pass the exact target column name or set target=None for unsupervised analysis."
                ),
            )

        self.df = df.copy()
        self.target = target
        self.config = config or default_config

        # Precompute numeric and categorical column lists for reuse
        self._numeric_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        # Store config values directly for performance and clarity.
        self.iqr_multiplier = self.config.iqr_multiplier
        self.zscore_threshold = self.config.zscore_threshold
        self.outlier_method = self.config.outlier_method
        self.constant_variance_threshold = self.config.constant_variance_threshold
        self.max_unique_for_categorical_like = self.config.max_unique_for_categorical_like
        self.correlation_threshold = self.config.correlation_threshold

    # ------------------------------------------------------------------
    # Dataset‑level checks
    # ------------------------------------------------------------------
    def compute_dataset_metadata(self) -> DatasetMetadata:
        """Return basic metadata about the dataset.

        Returns
        -------
        DatasetMetadata
        """
        n_rows, n_columns = self.df.shape
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024**2)
        col_types = self.df.dtypes.value_counts().to_dict()
        # Convert dtype objects to string for serialisability
        col_types = {str(k): v for k, v in col_types.items()}
        return DatasetMetadata(
            n_rows=n_rows,
            n_columns=n_columns,
            memory_mb=round(memory_mb, 2),
            column_types=col_types,
        )

    def compute_duplicate_report(self) -> DuplicateReport:
        """Analyse duplicate rows.

        Returns
        -------
        DuplicateReport
        """
        n_duplicates = self.df.duplicated().sum()
        dup_percent = (
            100.0 * n_duplicates / len(self.df) if len(self.df) > 0 else 0.0
        )
        sample_idx = (
            self.df[self.df.duplicated(keep=False)].index[:100].tolist()
            if n_duplicates > 0
            else []
        )
        return DuplicateReport(
            total_duplicates=n_duplicates,
            duplicate_percent=round(dup_percent, 2),
            sample_indices=sample_idx,
        )

    def compute_infinite_report(self) -> InfiniteReport:
        """Detect columns containing positive or negative infinity.

        Returns
        -------
        InfiniteReport
        """
        inf_mask = self.df.isin([np.inf, -np.inf])
        inf_counts = inf_mask.sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        return InfiniteReport(
            columns_with_inf=cols_with_inf.index.tolist(),
            counts=cols_with_inf.to_dict(),
        )

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    def compute_missing_report(self) -> MissingReport:
        """Compute missing values per column and overall.

        Returns
        -------
        MissingReport
        """
        total_rows = len(self.df)
        total_missing = self.df.isna().sum().sum()
        missing_by_col = self.df.isna().sum()
        if total_rows > 0:
            missing_pct = (100.0 * missing_by_col / total_rows).round(2)
        else:
            missing_pct = pd.Series(0.0, index=missing_by_col.index, dtype=float)

        cols_with_missing = missing_by_col[missing_by_col > 0].index.tolist()

        reports = [
            MissingColumnReport(
                column=col,
                missing_count=int(missing_by_col[col]),
                missing_percent=float(missing_pct[col]),
            )
            for col in cols_with_missing
        ]

        return MissingReport(
            total_missing=int(total_missing),
            columns_with_missing=cols_with_missing,
            column_reports=reports,
        )

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------
    def _iqr_outliers(
        self, series: pd.Series
    ) -> Tuple[float, float, pd.Series]:
        """IQR‑based outlier detection.

        Parameters
        ----------
        series : pd.Series
            Clean numeric series (no NaN).

        Returns
        -------
        lower_bound : float
        upper_bound : float
        outlier_mask : pd.Series (boolean)
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        mask = (series < lower) | (series > upper)
        return lower, upper, mask

    def _zscore_outliers(
        self, series: pd.Series
    ) -> Tuple[Optional[float], Optional[float], pd.Series]:
        """Z‑score based outlier detection.

        Parameters
        ----------
        series : pd.Series
            Clean numeric series (no NaN).

        Returns
        -------
        lower_bound : None (not applicable)
        upper_bound : None (not applicable)
        outlier_mask : pd.Series (boolean)
        """
        z = np.abs(sp_stats.zscore(series, nan_policy="omit"))
        mask = z > self.zscore_threshold
        return None, None, mask

    def compute_outlier_report(self) -> List[OutlierReport]:
        """Compute outlier statistics for every numeric column.

        Returns
        -------
        List[OutlierReport]
        """
        reports: List[OutlierReport] = []
        method = self.outlier_method

        for col in self._numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            if method == "iqr":
                lower, upper, mask = self._iqr_outliers(series)
            elif method == "zscore":
                lower, upper, mask = self._zscore_outliers(series)
            else:
                raise DataValidationError(
                    f"Unsupported outlier method: {method}",
                    details="Use 'iqr' or 'zscore'.",
                )

            n_outliers = int(mask.sum())
            pct = (100.0 * n_outliers / len(series)) if len(series) > 0 else 0.0

            reports.append(
                OutlierReport(
                    column=col,
                    method=method,
                    outlier_count=n_outliers,
                    outlier_percent=round(pct, 2),
                    lower_bound=lower if lower is not None else None,
                    upper_bound=upper if upper is not None else None,
                )
            )

        return reports

    # ------------------------------------------------------------------
    # Distribution profiles (numeric / categorical)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_round(value: Any, ndigits: int = 4) -> float:
        """Round a value safely, returning NaN on failure."""
        try:
            return round(float(value), ndigits)
        except (TypeError, ValueError):
            return np.nan

    def _compute_numeric_profile(self, col: str) -> NumericDistributionProfile:
        """Create a NumericDistributionProfile for a single column."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return NumericDistributionProfile(
                column=col,
                count=0,
                mean=np.nan,
                median=np.nan,
                std=np.nan,
                cv=np.nan,
                min=np.nan,
                max=np.nan,
            )

        q_vals = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        percentiles: Dict[str, float] = {
            "1%": self._safe_round(q_vals[0.01], 4),
            "5%": self._safe_round(q_vals[0.05], 4),
            "25%": self._safe_round(q_vals[0.25], 4),
            "50%": self._safe_round(q_vals[0.50], 4),
            "75%": self._safe_round(q_vals[0.75], 4),
            "95%": self._safe_round(q_vals[0.95], 4),
            "99%": self._safe_round(q_vals[0.99], 4),
        }

        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        cv_val = (std_val / mean_val) if mean_val != 0 else np.nan

        try:
            sk = series.skew()
        except Exception:
            sk = np.nan
        try:
            ku = series.kurtosis()
        except Exception:
            ku = np.nan

        zero_pct = (series == 0).mean() * 100
        neg_pct = (series < 0).mean() * 100

        unique_count = series.nunique()
        is_categorical_like = unique_count <= self.max_unique_for_categorical_like

        return NumericDistributionProfile(
            column=col,
            count=len(series),
            mean=self._safe_round(mean_val, 4),
            median=self._safe_round(median_val, 4),
            std=self._safe_round(std_val, 4),
            cv=self._safe_round(cv_val, 4),
            min=self._safe_round(series.min(), 4),
            max=self._safe_round(series.max(), 4),
            percentiles=percentiles,
            skewness=self._safe_round(sk, 4),
            kurtosis=self._safe_round(ku, 4),
            zero_percent=self._safe_round(zero_pct, 2),
            negative_percent=self._safe_round(neg_pct, 2),
            is_categorical_like=is_categorical_like,
            unique_count=unique_count,
        )

    def _compute_categorical_profile(self, col: str) -> CategoricalProfile:
        """Create a CategoricalProfile for a single column."""
        series = self.df[col].dropna()
        missing = self.df[col].isna().sum()
        missing_pct = (100.0 * missing / len(self.df)) if len(self.df) > 0 else 0.0

        if len(series) == 0:
            return CategoricalProfile(
                column=col,
                unique_count=0,
                missing_count=missing,
                missing_percent=round(missing_pct, 2),
            )

        unique_count = series.nunique()
        top_series = series.value_counts().head(5)
        top_categories: List[Dict[str, Any]] = [
            {"category": idx, "count": int(cnt)}
            for idx, cnt in top_series.items()
        ]
        mode = series.mode().iloc[0] if not series.mode().empty else None

        return CategoricalProfile(
            column=col,
            unique_count=unique_count,
            top_categories=top_categories,
            missing_count=missing,
            missing_percent=round(missing_pct, 2),
            mode=mode,
        )

    def compute_feature_profiles(self) -> List[FeatureProfile]:
        """Build a FeatureProfile for every column in the dataset.

        Returns
        -------
        List[FeatureProfile]
        """
        profiles: List[FeatureProfile] = []

        if self._numeric_cols:
            variances = self.df[self._numeric_cols].var(numeric_only=False)
            quasi_const_mask = variances < self.constant_variance_threshold
            quasi_const = variances[quasi_const_mask].index.tolist()
        else:
            quasi_const = []

        const_cols = self.df.columns[self.df.nunique() <= 1].tolist()

        for col in self.df.columns:
            if col == self.target:
                continue

            dtype = str(self.df[col].dtype)
            is_const = col in const_cols
            is_quasi = col in quasi_const if col in self._numeric_cols else False

            if pd.api.types.is_numeric_dtype(self.df[col]):
                num_profile = self._compute_numeric_profile(col)
                profiles.append(
                    FeatureProfile(
                        column=col,
                        dtype=dtype,
                        numeric_profile=num_profile,
                        is_constant=is_const,
                        is_quasi_constant=is_quasi,
                    )
                )
            else:
                cat_profile = self._compute_categorical_profile(col)
                profiles.append(
                    FeatureProfile(
                        column=col,
                        dtype=dtype,
                        categorical_profile=cat_profile,
                        is_constant=is_const,
                        is_quasi_constant=False,
                    )
                )

        return profiles

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------
    def compute_correlation_pairs(self) -> List[CorrelationPair]:
        """Identify all pairwise Pearson correlations exceeding the
        configured threshold.

        Returns
        -------
        List[CorrelationPair]
        """
        if len(self._numeric_cols) < 2:
            return []

        corr_mat = self.df[self._numeric_cols].corr()
        pairs: List[CorrelationPair] = []
        threshold = self.correlation_threshold

        for i in range(len(corr_mat.columns)):
            for j in range(i + 1, len(corr_mat.columns)):
                val = corr_mat.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append(
                        CorrelationPair(
                            feature_a=corr_mat.columns[i],
                            feature_b=corr_mat.columns[j],
                            coefficient=round(val, 4),
                        )
                    )

        return pairs

    # ------------------------------------------------------------------
    # Target profile
    # ------------------------------------------------------------------
    def compute_target_profile(self) -> Optional[TargetProfile]:
        """Analyse the target column, if defined.

        Returns
        -------
        TargetProfile or None
        """
        if self.target is None:
            return None

        series = self.df[self.target]
        missing = series.isna().sum()
        missing_pct = (100.0 * missing / len(series)) if len(series) > 0 else 0.0
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            unique_vals = series.dropna().nunique()
            is_binary = unique_vals == 2
            is_regression = unique_vals >= 20
            class_dist = {}
            if not is_regression:
                class_dist = series.value_counts().sort_index().to_dict()
            return TargetProfile(
                column=self.target,
                dtype=dtype,
                n_unique=unique_vals,
                missing_count=int(missing),
                missing_percent=round(missing_pct, 2),
                is_regression=is_regression,
                is_binary=is_binary,
                class_distribution=class_dist,
            )
        else:
            unique_vals = series.nunique()
            class_dist = series.value_counts().to_dict()
            return TargetProfile(
                column=self.target,
                dtype=dtype,
                n_unique=unique_vals,
                missing_count=int(missing),
                missing_percent=round(missing_pct, 2),
                is_regression=False,
                is_binary=unique_vals == 2,
                class_distribution=class_dist,
            )

    # ------------------------------------------------------------------
    # Full analysis convenience
    # ------------------------------------------------------------------
    def run_full_analysis(self) -> Dict[str, Any]:
        """Execute all statistical computations and return a dictionary
        of dataclass objects.

        Returns
        -------
        dict
            Keys include:
                - 'metadata' : DatasetMetadata
                - 'duplicates' : DuplicateReport
                - 'infinite' : InfiniteReport
                - 'missing' : MissingReport
                - 'outliers' : List[OutlierReport]
                - 'feature_profiles' : List[FeatureProfile]
                - 'correlation_pairs' : List[CorrelationPair]
                - 'target_profile' : TargetProfile or None
        """
        return {
            "metadata": self.compute_dataset_metadata(),
            "duplicates": self.compute_duplicate_report(),
            "infinite": self.compute_infinite_report(),
            "missing": self.compute_missing_report(),
            "outliers": self.compute_outlier_report(),
            "feature_profiles": self.compute_feature_profiles(),
            "correlation_pairs": self.compute_correlation_pairs(),
            "target_profile": self.compute_target_profile(),
        }
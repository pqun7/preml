"""Orchestration layer — combines facts from the statistics engine with
recommendations from the recommendation engine to produce a complete EDA
result.

This module does not compute statistics nor make decisions; it only
coordinates the two engines and packages their outputs together with
a data quality assessment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.exceptions import DataValidationError
from ml_toolkit.recommendation_engine import RecommendationEngine
from ml_toolkit.statistics_engine import StatisticsEngine
from ml_toolkit.schema import Recommendation


class EDAAnalyzer:
    """Full EDA orchestrator for tabular datasets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.
    target : str, optional
        Name of the target column for supervised analysis.
    config : MLToolkitConfig, optional
        Configuration object. If None, the global `default_config` is used.
    enable_feature_engineering : bool, default True
        Passed to the recommendation engine; controls whether feature
        engineering suggestions are emitted.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[MLToolkitConfig] = None,
        enable_feature_engineering: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                "Input must be a pandas DataFrame.", details=type(df)
            )
        self.df = df
        self.target = target
        self.config = config or default_config
        self.enable_feature_engineering = enable_feature_engineering

        # Engines are created lazily inside run()
        self._stats_engine: Optional[StatisticsEngine] = None
        self._recommendation_engine: Optional[RecommendationEngine] = None
        self._analysis_result: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        """Execute full analysis: gather facts, generate recommendations,
        and compute a data quality score.

        Returns
        -------
        dict
            A comprehensive dictionary containing:

            - 'metadata' : DatasetMetadata
            - 'duplicates' : DuplicateReport
            - 'infinite' : InfiniteReport
            - 'missing' : MissingReport
            - 'outliers' : List[OutlierReport]
            - 'feature_profiles' : List[FeatureProfile]
            - 'correlation_pairs' : List[CorrelationPair]
            - 'target_profile' : Optional[TargetProfile]
            - 'recommendations' : dict from RecommendationEngine
            - 'data_quality_score' : float (0-100)
            - 'data_quality_notes' : List[str]

        """
        # 1. Compute facts
        self._stats_engine = StatisticsEngine(
            self.df, target=self.target, config=self.config
        )
        stats = self._stats_engine.run_full_analysis()

        # 2. Generate recommendations
        self._recommendation_engine = RecommendationEngine(
            config=self.config,
            enable_feature_engineering=self.enable_feature_engineering,
        )
        recommendations = self._recommendation_engine.generate_recommendations(stats)

        # 3. Compute data quality score
        score, notes = self._compute_quality_score(stats)

        # 4. Merge everything
        self._analysis_result = {
            **stats,
            "recommendations": recommendations,
            "data_quality_score": score,
            "data_quality_notes": notes,
        }
        return self._analysis_result

    def _compute_quality_score(
        self, stats: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Derive a heuristic data quality score (0–100) and explanatory notes.

        The score starts at 100 and is reduced for issues like
        duplicates, missing values, infinite values, excessive outliers,
        or constant columns.
        """
        score = 100.0
        notes: List[str] = []

        # Duplicates penalty
        dup = stats.get("duplicates")
        if dup and dup.total_duplicates > 0:
            penalty = min(dup.duplicate_percent * 0.2, 15)  # max 15 points
            score -= penalty
            notes.append(
                f"Duplicate rows ({dup.duplicate_percent:.1f}%) reduce quality "
                f"by {penalty:.1f} points."
            )

        # Infinite values penalty
        inf = stats.get("infinite")
        if inf and inf.columns_with_inf:
            penalty = min(len(inf.columns_with_inf) * 5, 15)
            score -= penalty
            notes.append(
                f"Infinite values in {len(inf.columns_with_inf)} column(s) "
                f"reduce quality by {penalty} points."
            )

        # Missing values penalty (safe division by zero)
        miss = stats.get("missing")
        if miss and miss.total_missing > 0:
            total_cells = len(self.df) * self.df.shape[1]
            if total_cells > 0:
                miss_ratio = miss.total_missing / total_cells
                penalty = min(miss_ratio * 100, 25)  # up to 25 points
            else:
                miss_ratio = 0.0
                penalty = 0.0
            score -= penalty
            notes.append(
                f"Missing values ({miss_ratio:.2%} of all cells) reduce "
                f"quality by {penalty:.1f} points."
            )

        # Outlier penalty (excessive outliers across features)
        outliers = stats.get("outliers", [])
        if outliers:
            high_outlier_cols = [o for o in outliers if o.outlier_percent > 10]
            if high_outlier_cols:
                penalty = min(len(high_outlier_cols) * 3, 10)
                score -= penalty
                notes.append(
                    f"{len(high_outlier_cols)} column(s) have >10% outliers; "
                    f"penalising {penalty} points."
                )

        # Constant/quasi-constant columns
        profiles = stats.get("feature_profiles", [])
        const_cols = [p for p in profiles if p.is_constant or p.is_quasi_constant]
        if const_cols:
            penalty = min(len(const_cols) * 2, 10)
            score -= penalty
            notes.append(
                f"{len(const_cols)} column(s) are constant or quasi-constant; "
                f"penalising {penalty} points."
            )

        return max(score, 0.0), notes

    def summary(self) -> str:
        """Return a plain‑text summary of the dataset and major findings.

        This is a lightweight textual overview; for full reports use
        `ml_toolkit.report`.
        """
        if self._analysis_result is None:
            self.run()

        result = self._analysis_result
        meta = result["metadata"]
        dup = result["duplicates"]
        miss = result["missing"]
        rec = result["recommendations"]

        lines = []
        lines.append("=" * 60)
        lines.append("         ML TOOLKIT EDA SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Rows: {meta.n_rows}   Columns: {meta.n_columns}")
        lines.append(f"Memory: {meta.memory_mb:.2f} MB")
        lines.append(f"Duplicate rows: {dup.total_duplicates} ({dup.duplicate_percent:.2f}%)")
        lines.append(f"Missing cells: {miss.total_missing}")
        lines.append(f"Data quality score: {result['data_quality_score']:.1f}/100")
        lines.append("")
        lines.append("Key recommendations:")
        for category in ["imputation", "outlier_handling", "transformation", "scaling", "encoding"]:
            cat_recs = rec.get(category, [])
            if isinstance(cat_recs, list):
                for r in cat_recs[:2]:  # top 2 per category
                    lines.append(f"  [{r.category}] {r.action}")
            elif isinstance(cat_recs, Recommendation):
                lines.append(f"  [scaling] {cat_recs.action}")
        return "\n".join(lines)


def quick_eda(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function: instantiate and run EDA with default settings.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target : str, optional
        Target column.

    Returns
    -------
    dict
        The complete analysis result (same as `EDAAnalyzer.run()`).
    """
    analyzer = EDAAnalyzer(df, target=target)
    return analyzer.run()
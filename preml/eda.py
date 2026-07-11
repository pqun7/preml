# eda.py
"""Orchestration layer — combines facts from the statistics engine with
recommendations from the recommendation engine to produce a complete EDA
result.

This module does not compute statistics nor make decisions; it only
coordinates the two engines and packages their outputs together with
a data quality assessment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

import pandas as pd

from preml.config import MLToolkitConfig, default_config
from preml.exceptions import DataValidationError
from preml.recommendation_engine import RecommendationEngine
from preml.recommendation_utils import normalize_recommendation_items
from preml.statistics_engine import StatisticsEngine

logger = logging.getLogger(__name__)


class _AnalysisResult(TypedDict, total=False):
    """Structure of the dictionary returned by `EDAAnalyzer.run()`."""

    metadata: Any
    duplicates: Any
    infinite: Any
    missing: Any
    outliers: List[Any]
    feature_profiles: List[Any]
    correlation_pairs: List[Any]
    target_profile: Optional[Any]
    recommendations: Dict[str, List[Any]]
    data_quality_score: float
    data_quality_notes: List[str]


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
                "Input must be a pandas DataFrame.",
                details=(
                    f"Received type: {type(df)}. "
                    "Convert your data with pd.DataFrame(...) before calling EDAAnalyzer."
                ),
            )
        self.df = df
        self.target = target
        self.config = config or default_config
        self.enable_feature_engineering = enable_feature_engineering

        # Engines are created lazily inside run()
        self._stats_engine: Optional[StatisticsEngine] = None
        self._recommendation_engine: Optional[RecommendationEngine] = None
        self._analysis_result: Optional[_AnalysisResult] = None

    def run(self) -> _AnalysisResult:
        """Execute full analysis: gather facts, generate recommendations,
        and compute a data quality score.

        Returns
        -------
        _AnalysisResult
            A comprehensive dictionary containing:

            - 'metadata' : DatasetMetadata
            - 'duplicates' : DuplicateReport
            - 'infinite' : InfiniteReport
            - 'missing' : MissingReport
            - 'outliers' : List[OutlierReport]
            - 'feature_profiles' : List[FeatureProfile]
            - 'correlation_pairs' : List[CorrelationPair]
            - 'target_profile' : Optional[TargetProfile]
            - 'recommendations' : dict (imputation, outlier_handling, etc.)
            - 'data_quality_score' : float (0-100)
            - 'data_quality_notes' : List[str]
        """
        logger.debug("Starting EDA run. DataFrame shape: %s", self.df.shape)

        # 1. Compute facts
        self._stats_engine = StatisticsEngine(
            self.df, target=self.target, config=self.config
        )
        stats = self._stats_engine.run_full_analysis()
        logger.debug("Statistics engine finished.")

        # 2. Generate recommendations
        self._recommendation_engine = RecommendationEngine(
            config=self.config,
            enable_feature_engineering=self.enable_feature_engineering,
        )
        recommendations = self._recommendation_engine.generate_recommendations(stats)
        logger.debug("Recommendation engine finished.")

        # 3. Compute data quality score
        score, notes = self._compute_quality_score(stats)
        logger.debug("Data quality score: %.1f", score)

        # 4. Merge everything
        self._analysis_result = cast(_AnalysisResult, {
            **stats,
            "recommendations": recommendations,
            "data_quality_score": score,
            "data_quality_notes": notes,
        })
        logger.info("EDA run complete.")
        return self._analysis_result

    def _compute_quality_score(
        self, stats: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Derive a heuristic data quality score (0–100) and explanatory notes.

        The score starts at 100 and is reduced for issues like
        duplicates, missing values, infinite values, excessive outliers,
        or constant columns.

        Returns
        -------
        Tuple[float, List[str]]
            (score, list of note strings)
        """
        score = 100.0
        notes: List[str] = []

        def _apply_penalty(penalty: float, message: str) -> None:
            nonlocal score
            score -= penalty
            notes.append(message)

        # Duplicates penalty
        dup = stats.get("duplicates")
        if dup is not None:
            dup_total = getattr(dup, "total_duplicates", 0)
            dup_pct = getattr(dup, "duplicate_percent", 0.0)
            if dup_total > 0:
                penalty = min(dup_pct * 0.2, 15)
                _apply_penalty(
                    penalty,
                    f"Duplicate rows ({dup_pct:.1f}%) reduce quality by {penalty:.1f} points.",
                )

        # Infinite values penalty
        inf = stats.get("infinite")
        if inf is not None:
            inf_cols = getattr(inf, "columns_with_inf", [])
            if inf_cols:
                penalty = min(len(inf_cols) * 5, 15)
                _apply_penalty(
                    penalty,
                    f"Infinite values in {len(inf_cols)} column(s) reduce quality by {penalty} points.",
                )

        # Missing values penalty (safe division by zero)
        miss = stats.get("missing")
        if miss is not None:
            miss_total = getattr(miss, "total_missing", 0)
            if miss_total > 0:
                total_cells = len(self.df) * self.df.shape[1]
                if total_cells > 0:
                    miss_ratio = miss_total / total_cells
                    penalty = min(miss_ratio * 100, 25)
                else:
                    miss_ratio = 0.0
                    penalty = 0.0
                _apply_penalty(
                    penalty,
                    f"Missing values ({miss_ratio:.2%} of all cells) reduce quality by {penalty:.1f} points.",
                )

        # Outlier penalty (excessive outliers across features)
        outliers = stats.get("outliers", [])
        if outliers:
            high_outlier_cols = [
                o for o in outliers if getattr(o, "outlier_percent", 0) > 10
            ]
            if high_outlier_cols:
                penalty = min(len(high_outlier_cols) * 3, 10)
                _apply_penalty(
                    penalty,
                    f"{len(high_outlier_cols)} column(s) have >10% outliers; penalising {penalty} points.",
                )

        # Constant/quasi-constant columns
        profiles = stats.get("feature_profiles", [])
        const_cols = [
            p
            for p in profiles
            if getattr(p, "is_constant", False) or getattr(p, "is_quasi_constant", False)
        ]
        if const_cols:
            penalty = min(len(const_cols) * 2, 10)
            _apply_penalty(
                penalty,
                f"{len(const_cols)} column(s) are constant or quasi-constant; penalising {penalty} points.",
            )

        return max(score, 0.0), notes

    def summary(self) -> str:
        """Return a plain‑text summary of the dataset and major findings.

        This is a lightweight textual overview; for full reports use
        `preml.report`.
        """
        if self._analysis_result is None:
            logger.debug("Running analysis because no cached result exists.")
            self.run()

        result = self._analysis_result
        if result is None:
            raise RuntimeError("Analysis failed to produce a result.")

        meta = result.get("metadata")
        dup = result.get("duplicates")
        miss = result.get("missing")
        rec = result.get("recommendations")

        # TypedDict keys are optional; ensure required pieces exist at runtime
        if meta is None or dup is None or miss is None or rec is None:
            raise RuntimeError(
                "Analysis result is missing required sections "
                "(metadata/duplicates/missing/recommendations)."
            )

        lines = []
        lines.append("=" * 60)
        lines.append("         ML TOOLKIT EDA SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Rows: {meta.n_rows}   Columns: {meta.n_columns}")
        lines.append(f"Memory: {meta.memory_mb:.2f} MB")
        lines.append(
            f"Duplicate rows: {dup.total_duplicates} ({dup.duplicate_percent:.2f}%)"
        )
        lines.append(f"Missing cells: {miss.total_missing}")
        dq = result.get("data_quality_score")
        try:
            dq_val = float(dq) if dq is not None else 0.0
        except (TypeError, ValueError):
            dq_val = 0.0
        lines.append(f"Data quality score: {dq_val:.1f}/100")
        lines.append("")
        lines.append("Key recommendations:")
        for category in [
            "imputation",
            "outlier_handling",
            "transformation",
            "scaling",
            "encoding",
        ]:
            cat_recs = normalize_recommendation_items(rec.get(category))
            for r in cat_recs[:2]:  # top 2 per category
                lines.append(f"  [{r.category}] {r.action}")
        logger.debug("Summary generated successfully.")
        return "\n".join(lines)


def quick_eda(df: pd.DataFrame, target: Optional[str] = None) -> _AnalysisResult:
    """Convenience function: instantiate and run EDA with default settings.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target : str, optional
        Target column.

    Returns
    -------
    _AnalysisResult
        The complete analysis result (same as `EDAAnalyzer.run()`).
    """
    analyzer = EDAAnalyzer(df, target=target)
    return analyzer.run()
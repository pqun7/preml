"""
Feature engineering suggestion layer — generates proposals for new
features based solely on statistical evidence, never on column names.

All suggestions are data‑driven and include confidence, evidence, and
explanations.  This module can optionally accept the original DataFrame
to detect datetime columns, but never recomputes statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import pandas as pd
import numpy as np

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.schema import (
    FeatureProfile,
    Recommendation,
    Evidence,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _get_profiles(analysis_result: Dict[str, Any]) -> List[FeatureProfile]:
    """Extract feature profiles from the analysis result dictionary."""
    return analysis_result.get("feature_profiles", [])


def _make_recommendation(
    action: str,
    confidence: float,
    reasons: List[str],
    stats: Dict[str, Any],
    alternatives: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
) -> Recommendation:
    """Build a Recommendation with a list of Evidence."""
    evidence = [Evidence(reason=r, statistics=stats) for r in reasons]
    return Recommendation(
        category="feature_engineering",
        action=action,
        confidence=min(max(confidence, 0.0), 1.0),
        evidence=evidence,
        alternative_options=alternatives or [],
        risks=risks or [],
    )


# ------------------------------------------------------------------
# FeatureEngineering
# ------------------------------------------------------------------
class FeatureEngineering:
    """Proposes new features using statistical evidence from the EDA.

    Parameters
    ----------
    analysis_result : dict
        Output of `EDAAnalyzer.run()`.
    df : pd.DataFrame, optional
        Original DataFrame. Required to detect datetime columns and
        to compute exact cardinalities for feature crossing.
    config : MLToolkitConfig, optional
        Configuration thresholds (skewness, correlation, etc.).
    """

    def __init__(
        self,
        analysis_result: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
        config: Optional[MLToolkitConfig] = None,
    ) -> None:
        self.analysis = analysis_result
        self.df = df
        self.config = config or default_config
        self.profiles = _get_profiles(analysis_result)
        self.correlation_pairs = analysis_result.get("correlation_pairs", [])

        # Build lookups for numeric and categorical profiles
        self.numeric_map: Dict[str, FeatureProfile] = {
            p.column: p for p in self.profiles if p.numeric_profile
        }
        self.categorical_map: Dict[str, FeatureProfile] = {
            p.column: p for p in self.profiles if p.categorical_profile
        }

        # Precompute highly correlated pairs (both directions) for reuse
        self._highly_corr_set: FrozenSet[Tuple[str, str]] = frozenset(
            (p.feature_a, p.feature_b) for p in self.correlation_pairs
        ) | frozenset(
            (p.feature_b, p.feature_a) for p in self.correlation_pairs
        )

        # Safe configuration thresholds
        self.skewness_threshold = getattr(self.config, "skewness_threshold", 1.0)
        self.max_unique_for_categorical_like = getattr(
            self.config, "max_unique_for_categorical_like", 10
        )

    def suggest_features(self) -> List[Recommendation]:
        """Generate a list of feature engineering suggestions.

        Returns
        -------
        List[Recommendation]
            Each recommendation has category ``feature_engineering``.
        """
        suggestions: List[Recommendation] = []
        suggestions.extend(self._suggest_ratios())
        suggestions.extend(self._suggest_interactions())
        suggestions.extend(self._suggest_binning())
        suggestions.extend(self._suggest_power_transforms())
        suggestions.extend(self._suggest_datetime_features())
        suggestions.extend(self._suggest_feature_crossing())

        logger.info("Generated %d feature engineering suggestions.", len(suggestions))
        return suggestions

    # ------------------------------------------------------------------
    # 1. Ratios
    # ------------------------------------------------------------------
    def _suggest_ratios(self) -> List[Recommendation]:
        """Suggest ratios for pairs of numeric columns that are safe to divide.

        Confidence is raised when the two columns have similar coefficients
        of variation, indicating a plausible relative measure.
        """
        recs: List[Recommendation] = []
        numeric_cols = list(self.numeric_map.keys())

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a, col_b = numeric_cols[i], numeric_cols[j]
                if (col_a, col_b) in self._highly_corr_set:
                    continue  # avoid redundant ratios for highly correlated pairs

                prof_a = self.numeric_map[col_a].numeric_profile
                prof_b = self.numeric_map[col_b].numeric_profile
                if not prof_a or not prof_b:
                    continue
                if prof_a.median == 0 or prof_b.median == 0:
                    continue
                if prof_a.min <= 0 or prof_b.min <= 0:
                    continue  # ratio only meaningful for positive values

                base_confidence = 0.5
                # Raise confidence if coefficients of variation are similar
                if (
                    prof_a.cv is not None and prof_b.cv is not None
                    and not (np.isnan(prof_a.cv) or np.isnan(prof_b.cv))
                    and abs(prof_a.cv - prof_b.cv) < 0.5
                ):
                    base_confidence = 0.65

                stats = {
                    f"{col_a}_median": prof_a.median,
                    f"{col_b}_median": prof_b.median,
                    f"{col_a}_cv": prof_a.cv,
                    f"{col_b}_cv": prof_b.cv,
                }
                recs.append(
                    _make_recommendation(
                        action=f"Create ratio feature: {col_a} / {col_b} (or vice versa).",
                        confidence=base_confidence,
                        reasons=[
                            f"Both columns are numeric with non‑zero medians and positive values, "
                            f"suggesting a possible relative measure."
                            + (
                                f" Similar CV ({prof_a.cv:.2f} vs {prof_b.cv:.2f}) strengthens the case."
                                if base_confidence > 0.5 else ""
                            )
                        ],
                        stats=stats,
                        alternatives=["Consider difference or product instead."],
                        risks=["Ratio may become unbounded or create division‑by‑zero in edge cases."],
                    )
                )
        return recs

    # ------------------------------------------------------------------
    # 2. Interactions
    # ------------------------------------------------------------------
    def _suggest_interactions(self) -> List[Recommendation]:
        """Suggest interaction (product) features for pairs of numeric
        columns that are NOT highly correlated (orthogonal information)."""
        recs: List[Recommendation] = []
        numeric_cols = list(self.numeric_map.keys())

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a, col_b = numeric_cols[i], numeric_cols[j]
                if (col_a, col_b) in self._highly_corr_set:
                    continue
                prof_a = self.numeric_map[col_a].numeric_profile
                prof_b = self.numeric_map[col_b].numeric_profile
                if not prof_a or not prof_b:
                    continue
                if prof_a.std == 0 or prof_b.std == 0:
                    continue  # constant column, interaction useless

                stats = {
                    f"{col_a}_std": prof_a.std,
                    f"{col_b}_std": prof_b.std,
                }
                recs.append(
                    _make_recommendation(
                        action=f"Consider adding interaction feature: {col_a} * {col_b}.",
                        confidence=0.4,
                        reasons=[
                            "The two features are not highly correlated, so their interaction "
                            "might capture combined effects."
                        ],
                        stats=stats,
                        alternatives=["Polynomial features (degree=2) can include this automatically."],
                        risks=["Interaction may increase dimensionality and noise."],
                    )
                )
        return recs

    # ------------------------------------------------------------------
    # 3. Binning
    # ------------------------------------------------------------------
    def _suggest_binning(self) -> List[Recommendation]:
        """Suggest discretisation for numeric columns with high cardinality
        and strong skewness."""
        recs: List[Recommendation] = []
        for col, prof in self.numeric_map.items():
            num = prof.numeric_profile
            if not num:
                continue
            if num.unique_count < 100 or abs(num.skewness) < self.skewness_threshold:
                continue
            recs.append(
                _make_recommendation(
                    action=f"Consider binning '{col}' (e.g., KBinsDiscretizer) to create ordinal feature.",
                    confidence=0.6,
                    reasons=[
                        f"High cardinality ({num.unique_count}) and skewness ({num.skewness:.2f}) "
                        "suggest discretisation may help model capture non‑linear patterns."
                    ],
                    stats={"unique_count": num.unique_count, "skewness": num.skewness},
                    alternatives=["Use as continuous with a tree‑based model."],
                    risks=["Binning loses information and may reduce predictive power if not careful."],
                )
            )
        return recs

    # ------------------------------------------------------------------
    # 4. Power transformations as new features
    # ------------------------------------------------------------------
    def _suggest_power_transforms(self) -> List[Recommendation]:
        """Suggest adding log‑ or Yeo‑Johnson transformed versions of
        highly skewed numeric columns."""
        recs: List[Recommendation] = []
        for col, prof in self.numeric_map.items():
            num = prof.numeric_profile
            if not num or abs(num.skewness) < self.skewness_threshold:
                continue
            if num.min > 0:
                action = f"Create a log‑transformed feature for '{col}' (e.g., log1p)."
                confidence = 0.8
                reasons = [
                    f"Highly right‑skewed ({num.skewness:.2f}) with positive values; "
                    "log transform can help linear models."
                ]
            else:
                action = f"Create a Yeo‑Johnson transformed feature for '{col}'."
                confidence = 0.7
                reasons = [
                    f"Highly skewed ({num.skewness:.2f}) and contains non‑positive values; "
                    "Yeo‑Johnson is applicable."
                ]
            recs.append(
                _make_recommendation(
                    action=action,
                    confidence=confidence,
                    reasons=reasons,
                    stats={"skewness": num.skewness},
                    alternatives=["Box‑Cox if strictly positive."],
                    risks=["Transformation alters interpretability."],
                )
            )
        return recs

    # ------------------------------------------------------------------
    # 5. Datetime features
    # ------------------------------------------------------------------
    def _suggest_datetime_features(self) -> List[Recommendation]:
        """If the DataFrame is available, detect datetime columns and
        propose extracting common date parts."""
        if self.df is None:
            return []
        dt_cols = self.df.select_dtypes(include=["datetime", "datetime64"]).columns
        recs: List[Recommendation] = []
        for col in dt_cols:
            recs.append(
                _make_recommendation(
                    action=f"Extract datetime features from '{col}': year, month, day, dayofweek, hour (if present).",
                    confidence=0.9,
                    reasons=["Datetime column detected; time components often carry predictive signal."],
                    stats={"column": col},
                    alternatives=["Cyclical encoding for month/day/dayofweek."],
                    risks=["Too many features can cause overfitting; use only relevant ones."],
                )
            )
        return recs

    # ------------------------------------------------------------------
    # 6. Feature crossing (categorical)
    # ------------------------------------------------------------------
    def _suggest_feature_crossing(self) -> List[Recommendation]:
        """Suggest combining two low‑cardinality categorical columns."""
        cat_cols = list(self.categorical_map.keys())
        recs: List[Recommendation] = []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                col_a, col_b = cat_cols[i], cat_cols[j]
                prof_a = self.categorical_map[col_a].categorical_profile
                prof_b = self.categorical_map[col_b].categorical_profile
                if not prof_a or not prof_b:
                    continue
                if (
                    prof_a.unique_count > self.max_unique_for_categorical_like
                    or prof_b.unique_count > self.max_unique_for_categorical_like
                ):
                    continue
                if prof_a.unique_count * prof_b.unique_count > 50:
                    continue
                recs.append(
                    _make_recommendation(
                        action=f"Create a crossed feature from '{col_a}' and '{col_b}' (e.g., concatenation).",
                        confidence=0.7,
                        reasons=[
                            f"Both have low cardinality ({prof_a.unique_count}, {prof_b.unique_count}); "
                            "their combination may reveal interactions."
                        ],
                        stats={
                            f"{col_a}_card": prof_a.unique_count,
                            f"{col_b}_card": prof_b.unique_count,
                        },
                        alternatives=["One‑Hot encode both separately and add interaction terms."],
                        risks=["Crossed feature cardinality may still be high and lead to overfitting."],
                    )
                )
        return recs
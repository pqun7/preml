"""Decision layer — interprets statistical facts and produces
evidence‑based recommendations.

This module reads facts (dataclass instances from
ml_toolkit.statistics_engine) and outputs Recommendations,
PipelineSuggestions, and ModelRecommendations.  No computation of
statistics happens here; all decisions are derived from the supplied
evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.exceptions import RecommendationError
from ml_toolkit.schema import (
    CategoricalProfile,
    CorrelationPair,
    DuplicateReport,
    Evidence,
    FeatureProfile,
    InfiniteReport,
    MissingColumnReport,
    MissingReport,
    ModelRecommendation,
    NumericDistributionProfile,
    OutlierReport,
    PipelineSuggestion,
    Recommendation,
    TargetProfile,
)


class RecommendationEngine:
    """Generates recommendations based on statistical evidence.

    Parameters
    ----------
    config : MLToolkitConfig, optional
        Configuration controlling thresholds and behaviour.
    enable_feature_engineering : bool
        If True, the engine may suggest feature engineering steps
        (e.g., ratios, interactions) based on statistical patterns.
        Such suggestions never rely on column names.
    """

    def __init__(
        self,
        config: Optional[MLToolkitConfig] = None,
        enable_feature_engineering: bool = True,
    ) -> None:
        self.config = config or default_config
        self.enable_feature_engineering = enable_feature_engineering
        self._outlier_columns: List[str] = []   


    # ------------------------------------------------------------------
    # Helper: build a Recommendation
    # ------------------------------------------------------------------
    def _make_recommendation(
        self,
        category: str,
        action: str,
        confidence: float,
        reasons: List[str],
        stats: Dict[str, Any],
        alternatives: Optional[List[str]] = None,
        risks: Optional[List[str]] = None,
    ) -> Recommendation:
        """Create a Recommendation with a list of Evidence."""
        evidence_list = [
            Evidence(reason=reason, statistics=stats) for reason in reasons
        ]
        return Recommendation(
            category=category,
            action=action,
            confidence=min(max(confidence, 0.0), 1.0),  # clamp
            evidence=evidence_list,
            alternative_options=alternatives or [],
            risks=risks or [],
        )

    # ------------------------------------------------------------------
    # Individual recommendation methods
    # ------------------------------------------------------------------
    def _imputation_recommendations(
        self,
        feature_profiles: List[FeatureProfile],
        missing_report: MissingReport,
        outlier_columns: Optional[List[str]] = None,
    ) -> List[Recommendation]:
        """Suggest imputation strategies per column with missing values.

        High‑missing columns are flagged for investigation regardless of
        whether a FeatureProfile is available. Others use the profile to
        choose between mean, median, or mode imputation.
        """
        recs: List[Recommendation] = []
        if not missing_report.columns_with_missing:
            return recs

        profile_map = {p.column: p for p in feature_profiles}
        self._outlier_columns = outlier_columns or []

        for col_report in missing_report.column_reports:
            col = col_report.column
            miss_pct = col_report.missing_percent
            stats: Dict[str, Any] = {"missing_percent": miss_pct}

            # 1. High missing ratio → investigate (no profile needed)
            if miss_pct > self.config.missing_threshold * 100:
                recs.append(
                    self._make_recommendation(
                        category="imputation",
                        action=f"High missing ratio ({miss_pct:.1f}%). Investigate "
                        "column importance before dropping or imputing.",
                        confidence=0.9,
                        reasons=[
                            f"Missing ratio exceeds threshold "
                            f"({self.config.missing_threshold*100:.0f}%)."
                        ],
                        stats=stats,
                        risks=[
                            "Dropping may lose critical information.",
                            "Imputation with high missing may introduce bias.",
                        ],
                    )
                )
                continue  # no need to check the profile further

            # 2. Lower missing ratio → rely on column profile
            profile = profile_map.get(col)
            if profile is None:
                continue

            if profile.numeric_profile:
                has_outliers = col in self._outlier_columns  # safe now
                if has_outliers:
                    action = "Use median imputation (outliers present)."
                    reasons = ["Outliers detected; median is robust."]
                else:
                    action = "Use mean or median imputation (no significant outliers)."
                    reasons = [
                        "No outliers detected. Mean is acceptable if distribution is symmetric."
                    ]
                recs.append(
                    self._make_recommendation(
                        category="imputation",
                        action=action,
                        confidence=0.8,
                        reasons=reasons,
                        stats={**stats, "has_outliers": has_outliers},
                        alternatives=[
                            "KNNImputer if missingness is MAR.",
                            "IterativeImputer for complex patterns.",
                        ],
                        risks=["Mean imputation distorts variance."],
                    )
                )
            else:
                # Categorical
                recs.append(
                    self._make_recommendation(
                        category="imputation",
                        action="Impute categorical missing with mode or a 'missing' category.",
                        confidence=0.7,
                        reasons=["Categorical feature with missing values."],
                        stats=stats,
                        alternatives=["Create a separate 'Unknown' category."],
                        risks=["Mode may overrepresent majority class."],
                    )
                )

        return recs

    def _outlier_recommendations(
        self, outlier_reports: List[OutlierReport]
    ) -> List[Recommendation]:
        """Generate recommendations for features with outliers."""
        recs = []
        for out in outlier_reports:
            if out.outlier_count == 0:
                continue
            stats = {
                "outlier_count": out.outlier_count,
                "outlier_percent": out.outlier_percent,
            }
            if out.outlier_percent > 5.0:
                action = (
                    f"Significant outlier ratio ({out.outlier_percent:.1f}%). "
                    "Investigate source before removal. Consider "
                    "winsorization or robust scaling if erroneous."
                )
                confidence = 0.85
            else:
                action = (
                    f"Minor outliers ({out.outlier_percent:.1f}%). "
                    "Likely natural extreme values. Keep unless domain suggests otherwise."
                )
                confidence = 0.6

            recs.append(
                self._make_recommendation(
                    category="outlier_handling",
                    action=action,
                    confidence=confidence,
                    reasons=[
                        f"IQR method detected {out.outlier_count} outliers "
                        f"({out.outlier_percent:.1f}%)."
                    ],
                    stats=stats,
                    alternatives=["Winsorization", "Capping", "Transformation"],
                    risks=["Removing valid extremes may harm model generalization."],
                )
            )
        return recs

    def _transformation_recommendations(
        self, feature_profiles: List[FeatureProfile]
    ) -> List[Recommendation]:
        """Suggest transformations for skewed numeric features."""
        recs = []
        for prof in feature_profiles:
            if not prof.numeric_profile:
                continue
            num = prof.numeric_profile
            sk = num.skewness
            if abs(sk) < self.config.skewness_threshold:
                continue

            col = prof.column
            stats = {"skewness": sk}
            if sk > self.config.skewness_threshold:
                if num.min >= 0:
                    action = f"Apply log1p transform to reduce right skew (skew={sk:.2f})."
                    reasons = ["Highly right‑skewed distribution."]
                    confidence = 0.9
                else:
                    action = f"Apply Yeo‑Johnson transform (right‑skewed, negative values present)."
                    reasons = ["Highly right‑skewed with negative values."]
                    confidence = 0.85
            else:  # left skewed
                action = f"Apply Yeo‑Johnson to correct left skew (skew={sk:.2f})."
                reasons = ["Highly left‑skewed distribution."]
                confidence = 0.8

            recs.append(
                self._make_recommendation(
                    category="transformation",
                    action=action,
                    confidence=confidence,
                    reasons=reasons,
                    stats=stats,
                    alternatives=[
                        "Box‑Cox (if all values > 0).",
                        "QuantileTransformer for non‑linear normalization.",
                    ],
                    risks=["Transformation may harm interpretability."],
                )
            )
        return recs

    def _scaling_recommendations(
        self, feature_profiles: List[FeatureProfile], target_profile: Optional[TargetProfile]
    ) -> Recommendation:
        """Global recommendation about scaling (one for all numeric features)."""
        numeric_cols = [p.column for p in feature_profiles if p.numeric_profile]
        if not numeric_cols:
            return self._make_recommendation(
                category="scaling",
                action="No numeric features to scale.",
                confidence=1.0,
                reasons=["No numeric features in dataset."],
                stats={},
            )
        # General advice: scaling needed for distance‑based models
        return self._make_recommendation(
            category="scaling",
            action="Scale numeric features if using distance‑based models (KNN, SVM, NN). "
            "Tree‑based models do not require scaling.",
            confidence=1.0,
            reasons=["General ML best practice for numeric features."],
            stats={"num_numeric_features": len(numeric_cols)},
            alternatives=["StandardScaler (normal distribution)", "RobustScaler (outliers)"],
        )

    def _encoding_recommendations(
        self, feature_profiles: List[FeatureProfile]
    ) -> List[Recommendation]:
        """Suggest encoding strategies for categorical features."""
        recs = []
        for prof in feature_profiles:
            if prof.categorical_profile:
                cat = prof.categorical_profile
                stats = {"unique_count": cat.unique_count}
                if cat.unique_count == 0:
                    continue
                if cat.unique_count <= 2:
                    action = "Binary encoding or keep as 0/1."
                    confidence = 0.95
                elif cat.unique_count <= 10:
                    action = "One-Hot Encoding (low cardinality)."   # plain ASCII hyphen
                    confidence = 0.9
                    
                else:
                    action = (
                        f"High cardinality ({cat.unique_count} categories). "
                        "Consider frequency encoding or target encoding."
                    )
                    confidence = 0.8
                recs.append(
                    self._make_recommendation(
                        category="encoding",
                        action=action,
                        confidence=confidence,
                        reasons=[f"Categorical with {cat.unique_count} unique values."],
                        stats=stats,
                        risks=["One‑hot encoding may explode dimensionality."],
                        alternatives=[
                            "OrdinalEncoder if order matters.",
                            "TargetEncoder (watch for leakage).",
                        ],
                    )
                )
        # Also handle numeric categorical‑like
        for prof in feature_profiles:
            if prof.numeric_profile and prof.numeric_profile.is_categorical_like:
                recs.append(
                    self._make_recommendation(
                        category="encoding",
                        action=f"Treat '{prof.column}' as categorical (only "
                        f"{prof.numeric_profile.unique_count} unique values).",
                        confidence=0.85,
                        reasons=["Numeric column with very few unique values."],
                        stats={"unique_count": prof.numeric_profile.unique_count},
                        alternatives=["One‑Hot encode or leave as integer."],
                    )
                )
        return recs

    def _feature_engineering_recommendations(
        self,
        feature_profiles: List[FeatureProfile],
        correlation_pairs: List[CorrelationPair],
    ) -> List[Recommendation]:
        """Suggest feature engineering based on statistical patterns."""
        if not self.enable_feature_engineering:
            return []

        recs = []
        # Suggestion: log transform for right‑skewed (already covered by transformation,
        # but we could add interaction suggestions from high correlations)
        if correlation_pairs:
            # Pick top correlated pair as example
            top_pair = sorted(correlation_pairs, key=lambda x: abs(x.coefficient), reverse=True)[0]
            recs.append(
                self._make_recommendation(
                    category="feature_engineering",
                    action=f"High correlation between '{top_pair.feature_a}' and "
                    f"'{top_pair.feature_b}' (r={top_pair.coefficient:.2f}). "
                    "These features may be redundant; consider dropping one, "
                    "using PCA, or applying regularization instead of creating a ratio or interaction.",
                    confidence=0.75,
                    reasons=[
                        "Highly correlated features often carry overlapping information."
                    ],
                    stats={"correlation": top_pair.coefficient},
                    risks=["May introduce multicollinearity."],
                    alternatives=["PCA to combine features."],
                )
            )

        # Suggestion: ratio features for features with similar magnitude
        # Use CV and range to detect potential ratio candidates
        # (no column names! purely statistical)
        numeric_profiles = [
            p.numeric_profile for p in feature_profiles if p.numeric_profile
        ]
        if len(numeric_profiles) >= 2:
            # Find pairs with similar CV (within 0.5) and non‑zero median
            pairs = []
            for i in range(len(numeric_profiles)):
                for j in range(i + 1, len(numeric_profiles)):
                    pi, pj = numeric_profiles[i], numeric_profiles[j]
                    if (
                        pi.cv and pj.cv and
                        abs(pi.cv - pj.cv) < 0.5 and
                        pi.median != 0 and pj.median != 0
                    ):
                        pairs.append((pi.column, pj.column))
            if pairs:
                # Take first pair
                col_a, col_b = pairs[0]
                recs.append(
                    self._make_recommendation(
                        category="feature_engineering",
                        action=f"Features '{col_a}' and '{col_b}' have similar "
                        "coefficients of variation. A ratio (e.g., a/b) may capture "
                        "relative information.",
                        confidence=0.5,  # weak suggestion
                        reasons=["Similar dispersion patterns may hide interactions."],
                        stats={"cv_a": numeric_profiles[0].cv, "cv_b": numeric_profiles[1].cv},
                        risks=["Ratio may become unbounded or create division‑by‑zero."],
                        alternatives=["Create polynomial features."],
                    )
                )
        return recs

    def _correlation_recommendations(
        self, correlation_pairs: List[CorrelationPair]
    ) -> List[Recommendation]:
        """Advise on highly correlated features."""
        recs = []
        for pair in correlation_pairs:
            recs.append(
                self._make_recommendation(
                    category="feature_selection",
                    action=f"High correlation between '{pair.feature_a}' and "
                    f"'{pair.feature_b}' (r={pair.coefficient:.2f}). "
                    "Consider dropping one to reduce multicollinearity.",
                    confidence=0.9,
                    reasons=["Multicollinearity can destabilize linear models."],
                    stats={"correlation": pair.coefficient},
                    risks=["Dropping may lose unique information if correlation is not perfect."],
                    alternatives=["Regularization (L1/L2) to handle collinearity."],
                )
            )
        return recs

    def _pipeline_suggestion(
        self, feature_profiles: List[FeatureProfile]
    ) -> PipelineSuggestion:
        """Create a suggested sklearn‑compatible pipeline based on profiles."""
        steps = []
        # Numeric pipeline
        num_cols = [p.column for p in feature_profiles if p.numeric_profile]
        cat_cols = [p.column for p in feature_profiles if p.categorical_profile]

        if num_cols:
            # Determine scaler based on outlier presence
            has_outliers = any(
                p.numeric_profile and p.numeric_profile.skewness > self.config.skewness_threshold
                for p in feature_profiles
            )
            scaler = "RobustScaler()" if has_outliers else "StandardScaler()"
            steps.append(("numeric_preprocessing", f"ColumnTransformer for {len(num_cols)} columns"))
            # In practice, we'd return the actual sklearn objects, but here we keep it conceptual.
            steps.append(("scaler", scaler))
        if cat_cols:
            steps.append(("categorical_preprocessing", "OneHotEncoder()"))

        return PipelineSuggestion(
            name="Recommended base pipeline",
            steps=steps,
            explanation="Automatically generated based on feature types and outlier presence.",
        )

    def _model_recommendations(
        self, target_profile: Optional[TargetProfile]
    ) -> List[ModelRecommendation]:
        """Suggest baseline models based on target type."""
        if not target_profile:
            return []
        if target_profile.is_regression:
            return [
                ModelRecommendation(
                    model_name="LinearRegression",
                    suitability="baseline",
                    reason="Simple, interpretable baseline for regression.",
                    conditions=["Features should be scaled.", "No multicollinearity."],
                ),
                ModelRecommendation(
                    model_name="RandomForestRegressor",
                    suitability="good",
                    reason="Handles non‑linearity and feature interactions well.",
                    conditions=["Handles raw features well.", "No scaling required."],
                ),
            ]
        else:
            # classification
            if target_profile.is_binary:
                models = [
                    ModelRecommendation(
                        model_name="LogisticRegression",
                        suitability="baseline",
                        reason="Probabilistic baseline for binary classification.",
                        conditions=["Features should be scaled."],
                    ),
                    ModelRecommendation(
                        model_name="RandomForestClassifier",
                        suitability="good",
                        reason="Robust to many data issues, good default.",
                        conditions=["No scaling required."],
                    ),
                ]
            else:
                models = [
                    ModelRecommendation(
                        model_name="LogisticRegression (OvR)",
                        suitability="baseline",
                        reason="Multiclass extension of logistic regression.",
                        conditions=["Scaled features."],
                    ),
                    ModelRecommendation(
                        model_name="RandomForestClassifier",
                        suitability="good",
                        reason="Handles multiclass naturally.",
                    ),
                ]
            return models

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate_recommendations(
        self,
        analysis_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce a full set of recommendations from analysis facts.

        Parameters
        ----------
        analysis_results : dict
            The dictionary returned by
            `StatisticsEngine.run_full_analysis()`. Expected keys:
            'metadata', 'duplicates', 'infinite', 'missing', 'outliers',
            'feature_profiles', 'correlation_pairs', 'target_profile'.

        Returns
        -------
        dict
            Keys:
                - 'imputation': List[Recommendation]
                - 'outlier_handling': List[Recommendation]
                - 'transformation': List[Recommendation]
                - 'scaling': Recommendation
                - 'encoding': List[Recommendation]
                - 'feature_engineering': List[Recommendation]
                - 'feature_selection': List[Recommendation]
                - 'pipeline': PipelineSuggestion
                - 'models': List[ModelRecommendation]
                - 'data_quality_notes': List[str]  (simple warnings)
        """
        # Extract facts
        duplicates = analysis_results.get("duplicates")
        infinite = analysis_results.get("infinite")
        missing = analysis_results.get("missing")
        # Ensure we pass a MissingReport to imputation recommendations
        if missing is None:
            # type: ignore[arg-type]
            missing = MissingReport(columns_with_missing=[], column_reports=[], total_missing=0)
        outlier_reports = analysis_results.get("outliers", [])
        feature_profiles = analysis_results.get("feature_profiles", [])
        correlation_pairs = analysis_results.get("correlation_pairs", [])
        target_profile = analysis_results.get("target_profile")

        outlier_columns = [o.column for o in outlier_reports if o.outlier_count > 0]

        # Build recommendations
        imputation_recs = self._imputation_recommendations(
            feature_profiles,
            missing,
            outlier_columns,
        )
        outlier_recs = self._outlier_recommendations(outlier_reports)
        transformation_recs = self._transformation_recommendations(feature_profiles)
        scaling_rec = self._scaling_recommendations(feature_profiles, target_profile)
        encoding_recs = self._encoding_recommendations(feature_profiles)
        feature_eng_recs = self._feature_engineering_recommendations(
            feature_profiles, correlation_pairs
        )
        corr_recs = self._correlation_recommendations(correlation_pairs)
        pipeline = self._pipeline_suggestion(feature_profiles)
        model_recs = self._model_recommendations(target_profile)

        # Data quality notes
        data_quality_notes = []
        if duplicates and duplicates.total_duplicates > 0:
            data_quality_notes.append(
                f"Found {duplicates.total_duplicates} duplicate rows "
                f"({duplicates.duplicate_percent:.2f}%)."
            )
        if infinite and infinite.columns_with_inf:
            data_quality_notes.append(
                f"Infinite values in columns: {infinite.columns_with_inf}. "
                "Treat as missing or drop."
            )
        if missing and missing.total_missing > 0:
            data_quality_notes.append(
                f"Total missing values: {missing.total_missing} across "
                f"{len(missing.columns_with_missing)} columns."
            )

        return {
            "imputation": imputation_recs,
            "outlier_handling": outlier_recs,
            "transformation": transformation_recs,
            "scaling": scaling_rec,
            "encoding": encoding_recs,
            "feature_engineering": feature_eng_recs,
            "feature_selection": corr_recs,
            "pipeline": pipeline,
            "models": model_recs,
            "data_quality_notes": data_quality_notes,
        }
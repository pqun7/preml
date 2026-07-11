"""
Decision layer — interprets statistical facts and produces
evidence‑based recommendations.

This module reads facts (dataclass instances from
preml.statistics_engine) and outputs Recommendations,
PipelineSuggestions, and ModelRecommendations.  No computation of
statistics happens here; all decisions are derived from the supplied
evidence.

Thresholds are configured via :class:`MLToolkitConfig` with sensible
defaults.  The engine considers missing values, outliers, skewness,
cardinality, and multicollinearity to generate actionable advice.
Model recommendations adapt to dataset size, presence of missing
values, and (optionally) the availability of XGBoost / LightGBM.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional, Tuple

from preml._analysis import resolve_analysis_result
from preml.config import MLToolkitConfig, default_config
from preml.exceptions import RecommendationError
from preml.schema import (
    CorrelationPair,
    DatasetMetadata,
    DuplicateReport,
    Evidence,
    FeatureProfile,
    InfiniteReport,
    MissingReport,
    ModelRecommendation,
    OutlierReport,
    PipelineSuggestion,
    Recommendation,
    TargetProfile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional gradient boosting libraries
# ---------------------------------------------------------------------------
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None


class RecommendationEngine:
    """Generates recommendations based on statistical evidence.

    Parameters
    ----------
    config : MLToolkitConfig, optional
        Configuration controlling thresholds and behaviour.
        See :class:`MLToolkitConfig` for all available parameters.
    enable_feature_engineering : bool
        If True, the engine may suggest feature engineering steps
        (e.g. ratios, interactions) based on statistical patterns.
        These suggestions never rely on column names.
    """

    def __init__(
        self,
        config: Optional[MLToolkitConfig] = None,
        enable_feature_engineering: bool = True,
    ) -> None:
        self.config = config or default_config
        self.enable_feature_engineering = enable_feature_engineering

        # Internal cache for outlier column names (set during imputation step).
        self._outlier_columns: List[str] = []

    # ------------------------------------------------------------------
    # Helper: build a Recommendation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_recommendation(
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
            confidence=min(max(confidence, 0.0), 1.0),
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
        """Suggest imputation strategies per column with missing values."""
        recs: List[Recommendation] = []
        if not missing_report.columns_with_missing:
            return recs

        profile_map: Dict[str, FeatureProfile] = {p.column: p for p in feature_profiles}
        self._outlier_columns = outlier_columns or []

        for col_report in missing_report.column_reports:
            col = col_report.column
            miss_pct = col_report.missing_percent  # already a percentage (0-100)
            stats: Dict[str, Any] = {"missing_percent": miss_pct}

            # When missing ratio exceeds the threshold, raise a high‑level flag.
            if miss_pct > self.config.missing_threshold * 100:
                recs.append(
                    self._make_recommendation(
                        category="imputation",
                        action=(
                            f"High missing ratio ({miss_pct:.1f}%) in column "
                            f"'{col}'. Investigate column importance before "
                            "dropping or imputing."
                        ),
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
                continue

            profile = profile_map.get(col)
            if profile is None:
                continue

            if profile.numeric_profile:
                has_outliers = col in self._outlier_columns
                if has_outliers:
                    action = (
                        f"Column '{col}': use median imputation "
                        "(outliers present)."
                    )
                    reasons = [f"Outliers detected in '{col}'; median is robust."]
                else:
                    action = (
                        f"Column '{col}': use mean or median imputation "
                        "(no significant outliers)."
                    )
                    reasons = [
                        f"No outliers detected in '{col}'. "
                        "Mean is acceptable if distribution is symmetric."
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
                recs.append(
                    self._make_recommendation(
                        category="imputation",
                        action=(
                            f"Column '{col}': impute categorical missing "
                            "with mode or a 'missing' category."
                        ),
                        confidence=0.7,
                        reasons=[f"Categorical feature '{col}' with missing values."],
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
        recs: List[Recommendation] = []
        threshold = getattr(self.config, "outlier_threshold_percent", 5.0)
        for out in outlier_reports:
            if out.outlier_count == 0:
                continue
            stats = {
                "outlier_count": out.outlier_count,
                "outlier_percent": out.outlier_percent,
            }
            if out.outlier_percent > threshold:
                action = (
                    f"Column '{out.column}': significant outlier ratio "
                    f"({out.outlier_percent:.1f}%). Investigate source "
                    "before removal. Consider winsorization or robust "
                    "scaling if erroneous."
                )
                confidence = 0.85
            else:
                action = (
                    f"Column '{out.column}': minor outliers "
                    f"({out.outlier_percent:.1f}%). Likely natural "
                    "extreme values. Keep unless domain suggests otherwise."
                )
                confidence = 0.6

            recs.append(
                self._make_recommendation(
                    category="outlier_handling",
                    action=action,
                    confidence=confidence,
                    reasons=[
                        f"IQR method detected {out.outlier_count} outliers "
                        f"({out.outlier_percent:.1f}%) in '{out.column}'."
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
        recs: List[Recommendation] = []
        skew_threshold = self.config.skewness_threshold
        for prof in feature_profiles:
            if not prof.numeric_profile:
                continue
            num = prof.numeric_profile
            sk = num.skewness
            if abs(sk) < skew_threshold:
                continue

            col = prof.column
            stats = {"skewness": sk}
            if sk > skew_threshold:
                if num.min is not None and num.min >= 0:
                    action = (
                        f"Column '{col}': apply log1p transform to "
                        f"reduce right skew (skew={sk:.2f})."
                    )
                    reasons = ["Highly right‑skewed distribution."]
                    confidence = 0.9
                else:
                    action = (
                        f"Column '{col}': apply Yeo‑Johnson transform "
                        f"(right‑skewed, negative values present)."
                    )
                    reasons = ["Highly right‑skewed with negative values."]
                    confidence = 0.85
            else:
                action = (
                    f"Column '{col}': apply Yeo‑Johnson to correct "
                    f"left skew (skew={sk:.2f})."
                )
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
        self,
        feature_profiles: List[FeatureProfile],
        target_profile: Optional[TargetProfile],
    ) -> Recommendation:
        """Global recommendation about feature scaling."""
        numeric_cols = [p.column for p in feature_profiles if p.numeric_profile]
        if not numeric_cols:
            return self._make_recommendation(
                category="scaling",
                action="No numeric features to scale.",
                confidence=1.0,
                reasons=["No numeric features in dataset."],
                stats={},
            )
        return self._make_recommendation(
            category="scaling",
            action=(
                "Scale numeric features if using distance‑based models "
                "(KNN, SVM, NN). Tree‑based models do not require scaling."
            ),
            confidence=1.0,
            reasons=["General ML best practice for numeric features."],
            stats={"num_numeric_features": len(numeric_cols)},
            alternatives=[
                "StandardScaler (normal distribution)",
                "RobustScaler (outliers)",
            ],
        )

    def _encoding_recommendations(
        self, feature_profiles: List[FeatureProfile]
    ) -> List[Recommendation]:
        """Suggest encoding strategies for categorical features."""
        recs: List[Recommendation] = []
        low_threshold = getattr(
            self.config, "low_cardinality_threshold", 10
        )
        high_threshold = self.config.high_cardinality_threshold  # not used directly but available

        for prof in feature_profiles:
            if prof.categorical_profile:
                cat = prof.categorical_profile
                stats = {"unique_count": cat.unique_count}
                if cat.unique_count == 0:
                    continue
                if cat.unique_count <= 2:
                    action = (
                        f"Column '{prof.column}': binary encoding or keep as 0/1."
                    )
                    confidence = 0.95
                elif cat.unique_count <= low_threshold:
                    action = (
                        f"Column '{prof.column}': One-Hot Encoding "
                        "(low cardinality)."
                    )
                    confidence = 0.9
                else:
                    action = (
                        f"Column '{prof.column}': high cardinality "
                        f"({cat.unique_count} categories). Consider "
                        "frequency encoding or target encoding."
                    )
                    confidence = 0.8
                recs.append(
                    self._make_recommendation(
                        category="encoding",
                        action=action,
                        confidence=confidence,
                        reasons=[
                            f"Categorical with {cat.unique_count} unique values."
                        ],
                        stats=stats,
                        risks=["One‑hot encoding may explode dimensionality."],
                        alternatives=[
                            "OrdinalEncoder if order matters.",
                            "TargetEncoder (watch for leakage).",
                        ],
                    )
                )
        # Numeric columns that look categorical
        for prof in feature_profiles:
            if prof.numeric_profile and prof.numeric_profile.is_categorical_like:
                recs.append(
                    self._make_recommendation(
                        category="encoding",
                        action=(
                            f"Treat '{prof.column}' as categorical (only "
                            f"{prof.numeric_profile.unique_count} unique values)."
                        ),
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
        """Suggest feature engineering based on statistical patterns.

        Only activated when `self.enable_feature_engineering` is True.

        .. todo::
            The ratio/interaction logic here duplicates functionality in
            :class:`~preml.feature_engineering.FeatureEngineering`.  In a
            future refactoring, the recommendation engine should delegate
            to that module instead of maintaining parallel code.
        """
        if not self.enable_feature_engineering:
            return []

        recs: List[Recommendation] = []
        corr_threshold = self.config.correlation_threshold

        # Strong correlations → redundancy warning
        strong_pairs = [
            pair
            for pair in correlation_pairs
            if abs(pair.coefficient) >= corr_threshold
        ]
        if strong_pairs:
            top_pair = max(strong_pairs, key=lambda x: abs(x.coefficient))
            abs_corr = abs(top_pair.coefficient)
            # Confidence increases with correlation strength
            confidence = 0.65 + (min(abs_corr, 1.0) - corr_threshold) * (
                0.3 / (1.0 - corr_threshold + 1e-9)
            )
            recs.append(
                self._make_recommendation(
                    category="feature_engineering",
                    action=(
                        f"High correlation between '{top_pair.feature_a}' and "
                        f"'{top_pair.feature_b}' (r={top_pair.coefficient:.2f}). "
                        "These features may be redundant; consider dropping one, "
                        "using PCA, or applying regularization instead of creating "
                        "a ratio or interaction."
                    ),
                    confidence=min(confidence, 0.95),
                    reasons=[
                        "Highly correlated features often carry overlapping information."
                    ],
                    stats={"correlation": top_pair.coefficient},
                    risks=["May introduce multicollinearity."],
                    alternatives=["PCA to combine features."],
                )
            )

        # Ratio suggestion based on similar coefficients of variation
        numeric_profiles = [
            p.numeric_profile for p in feature_profiles if p.numeric_profile
        ]
        if len(numeric_profiles) >= 2:
            for i in range(len(numeric_profiles)):
                for j in range(i + 1, len(numeric_profiles)):
                    pi, pj = numeric_profiles[i], numeric_profiles[j]
                    if (
                        pi.cv is not None
                        and pj.cv is not None
                        and abs(pi.cv - pj.cv) < 0.5
                        and pi.median != 0
                        and pj.median != 0
                    ):
                        recs.append(
                            self._make_recommendation(
                                category="feature_engineering",
                                action=(
                                    f"Features '{pi.column}' and '{pj.column}' have "
                                    "similar coefficients of variation. A ratio "
                                    "(e.g., a/b) may capture relative information."
                                ),
                                confidence=0.5,
                                reasons=[
                                    "Similar dispersion patterns may hide interactions."
                                ],
                                stats={"cv_a": pi.cv, "cv_b": pj.cv},
                                risks=[
                                    "Ratio may become unbounded or create "
                                    "division‑by‑zero."
                                ],
                                alternatives=["Create polynomial features."],
                            )
                        )
                        # Only suggest one ratio pair to avoid noise.
                        return recs
        return recs

    def _correlation_recommendations(
        self, correlation_pairs: List[CorrelationPair]
    ) -> List[Recommendation]:
        """Advise on highly correlated features (above config threshold)."""
        recs: List[Recommendation] = []
        corr_threshold = self.config.correlation_threshold
        for pair in correlation_pairs:
            if abs(pair.coefficient) < corr_threshold:
                continue
            recs.append(
                self._make_recommendation(
                    category="feature_selection",
                    action=(
                        f"High correlation between '{pair.feature_a}' and "
                        f"'{pair.feature_b}' (r={pair.coefficient:.2f}). "
                        "Consider dropping one to reduce multicollinearity."
                    ),
                    confidence=0.9,
                    reasons=[
                        "Multicollinearity can destabilize linear models."
                    ],
                    stats={"correlation": pair.coefficient},
                    risks=[
                        "Dropping may lose unique information if correlation "
                        "is not perfect."
                    ],
                    alternatives=[
                        "Regularization (L1/L2) to handle collinearity."
                    ],
                )
            )
        return recs

    def _pipeline_suggestion(
        self,
        feature_profiles: List[FeatureProfile],
        missing_report: Optional[MissingReport] = None,
        outlier_columns: Optional[List[str]] = None,
        transformation_recs: Optional[List[Recommendation]] = None,
    ) -> PipelineSuggestion:
        """Create a suggested sklearn‑compatible pipeline based on profiles.

        The pipeline reflects detected issues: missing values, skewness,
        outliers, and categorical columns.
        """
        steps: List[Tuple[str, str]] = []
        num_cols = [p.column for p in feature_profiles if p.numeric_profile]
        cat_cols = [p.column for p in feature_profiles if p.categorical_profile]

        has_outlier_column = (
            outlier_columns is not None
            and any(col in outlier_columns for col in num_cols)
        )
        has_skew = (
            transformation_recs is not None
            and any(rec.category == "transformation" for rec in transformation_recs)
        )

        if missing_report and missing_report.total_missing > 0:
            steps.append(
                (
                    "imputation",
                    "SimpleImputer(strategy='median' for numeric, "
                    "'most_frequent' for categorical)",
                )
            )

        if has_skew and num_cols:
            steps.append(
                (
                    "transformation",
                    "PowerTransformer(method='yeo-johnson') or "
                    "FunctionTransformer(log1p)",
                )
            )

        if num_cols:
            scaler = "RobustScaler()" if has_outlier_column else "StandardScaler()"
            steps.append(("scaler", scaler))

        if cat_cols:
            steps.append(
                ("categorical_encoder", "OneHotEncoder(handle_unknown='ignore')")
            )

        if not steps:
            steps.append(("passthrough", "No preprocessing required"))

        return PipelineSuggestion(
            name="Recommended base pipeline",
            steps=steps,
            explanation=(
                "Automatically generated based on feature types, "
                "missing values, outliers, and skewness."
            ),
        )

    # ==================================================================
    # EVIDENCE‑BASED MODEL RECOMMENDATIONS
    # ==================================================================

    # ------------------------------------------------------------------
    # Internal helpers for model recommendation building
    # ------------------------------------------------------------------
    def _add_baseline_model(
        self, models: List[ModelRecommendation], task_type: str
    ) -> None:
        """Append the default linear baseline (Linear/LogisticRegression)."""
        if task_type == "regression":
            models.append(
                ModelRecommendation(
                    model_name="LinearRegression",
                    suitability="baseline",
                    reason="Simple, interpretable baseline.",
                    conditions=["Scale features."],
                    hyperparams={},
                )
            )
        else:
            models.append(
                ModelRecommendation(
                    model_name="LogisticRegression",
                    suitability="baseline",
                    reason="Probabilistic baseline for classification.",
                    conditions=["Scale features."],
                    hyperparams={
                        "max_iter": 1000,
                        "random_state": self.config.random_state,
                    },
                )
            )

    def _add_tree_ensemble_models(
        self,
        models: List[ModelRecommendation],
        task_type: str,
        has_multicollinearity: bool,
        is_large: bool,
        n_features: int,
        has_missing: bool,
    ) -> None:
        """Add RandomForest, GradientBoosting, and optionally HistGradientBoosting."""
        reg = task_type == "regression"
        rf_name = "RandomForestRegressor" if reg else "RandomForestClassifier"
        gb_name = "GradientBoostingRegressor" if reg else "GradientBoostingClassifier"
        hist_name = (
            "HistGradientBoostingRegressor"
            if reg
            else "HistGradientBoostingClassifier"
        )

        models.append(
            ModelRecommendation(
                model_name=rf_name,
                suitability="excellent" if not has_multicollinearity else "good",
                reason="Handles non‑linearity and does not require scaling.",
                conditions=["Tune n_estimators (100‑500) and max_depth (5‑30)."],
                hyperparams={
                    "n_estimators": 200,
                    "random_state": self.config.random_state,
                },
            )
        )

        gb_suitability = "excellent" if (is_large and n_features > 3) else "good"
        models.append(
            ModelRecommendation(
                model_name=gb_name,
                suitability=gb_suitability,
                reason="Sequential ensemble; often best performance on tabular data.",
                conditions=[
                    "Tune learning_rate (0.01‑0.2), n_estimators (100‑1000), max_depth (3‑10)."
                ],
                hyperparams={
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "random_state": self.config.random_state,
                },
            )
        )

        if has_missing:
            models.append(
                ModelRecommendation(
                    model_name=hist_name,
                    suitability="good",
                    reason="Native support for missing values; faster than standard GBDT.",
                    conditions=[
                        "No need for imputation.",
                        "Tune max_iter, learning_rate, max_depth.",
                    ],
                    hyperparams={
                        "max_iter": 200,
                        "learning_rate": 0.1,
                        "random_state": self.config.random_state,
                    },
                )
            )

    def _add_optional_xgboost_lightgbm(
        self,
        models: List[ModelRecommendation],
        task_type: str,
        is_small: bool,
    ) -> None:
        """Add XGBoost and LightGBM models if libraries are available."""
        reg = task_type == "regression"
        if XGBOOST_AVAILABLE:
            xgb_name = "XGBRegressor" if reg else "XGBClassifier"
            models.append(
                ModelRecommendation(
                    model_name=xgb_name,
                    suitability="excellent",
                    reason="XGBoost: highly optimized, handles missing values and regularization.",
                    conditions=[
                        "Tune n_estimators, max_depth, learning_rate, subsample."
                    ],
                    hyperparams={
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "random_state": self.config.random_state,
                    },
                )
            )
        if LIGHTGBM_AVAILABLE:
            lgb_name = "LGBMRegressor" if reg else "LGBMClassifier"
            models.append(
                ModelRecommendation(
                    model_name=lgb_name,
                    suitability="excellent" if not is_small else "good",
                    reason="LightGBM: fast, handles large data and categorical features.",
                    conditions=["Tune num_leaves, learning_rate, n_estimators."],
                    hyperparams={
                        "n_estimators": 200,
                        "num_leaves": 31,
                        "learning_rate": 0.1,
                        "random_state": self.config.random_state,
                    },
                )
            )

    def _model_recommendations(
        self,
        target_profile: Optional[TargetProfile],
        feature_profiles: List[FeatureProfile],
        correlation_pairs: List[CorrelationPair],
        outlier_reports: List[OutlierReport],
        missing_report: Optional[MissingReport],
        metadata: Optional[DatasetMetadata] = None,
    ) -> List[ModelRecommendation]:
        """Suggest the most suitable models based on data characteristics.

        Returns
        -------
        List[ModelRecommendation]
            Sorted by suitability: 'excellent' → 'good' → 'baseline' → 'conditional'.
        """
        if not target_profile:
            return [
                ModelRecommendation(
                    model_name="N/A",
                    suitability="none",
                    reason="Target variable not specified; cannot recommend models.",
                    conditions=[],
                    hyperparams={},
                )
            ]

        is_regression = target_profile.is_regression
        is_binary = target_profile.is_binary

        # Estimate sample size
        if metadata and metadata.n_rows:
            n_samples = metadata.n_rows
        else:
            counts = [
                p.numeric_profile.count
                for p in feature_profiles
                if p.numeric_profile and p.numeric_profile.count > 0
            ]
            if counts:
                n_samples = max(counts)
            else:
                logger.debug("No numeric profile counts available; assuming n_samples=1000.")
                n_samples = 1000

        n_features = len([p for p in feature_profiles if not p.is_constant])

        has_multicollinearity = any(
            abs(pair.coefficient) >= self.config.correlation_threshold
            for pair in correlation_pairs
        )
        has_outliers = any(o.outlier_count > 0 for o in outlier_reports)
        has_missing = bool(missing_report and missing_report.total_missing > 0)

        is_large = n_samples > 10_000
        is_small = n_samples < 1_000
        is_high_dimensional = n_features > 50

        if is_regression:
            models = self._regression_model_recommendations(
                n_samples, n_features, has_multicollinearity, has_outliers,
                has_missing, is_large, is_small, is_high_dimensional,
            )
        else:
            n_classes = target_profile.n_unique
            models = self._classification_model_recommendations(
                n_samples, n_features, is_binary, n_classes,
                has_multicollinearity, has_outliers,
                has_missing, is_large, is_small, is_high_dimensional,
            )

        priority = {"excellent": 0, "good": 1, "baseline": 2, "conditional": 3}
        models.sort(key=lambda m: priority.get(m.suitability, 99))
        return models

    def _regression_model_recommendations(
        self,
        n_samples: int,
        n_features: int,
        has_multicollinearity: bool,
        has_outliers: bool,
        has_missing: bool,
        is_large: bool,
        is_small: bool,
        is_high_dimensional: bool,
    ) -> List[ModelRecommendation]:
        """Build regression model suggestions based on data flags."""
        models: List[ModelRecommendation] = []

        self._add_baseline_model(models, "regression")

        if has_multicollinearity:
            models.append(
                ModelRecommendation(
                    model_name="ElasticNetCV",
                    suitability="excellent",
                    reason="Multicollinearity detected. ElasticNet combines L1/L2 regularization.",
                    conditions=["Scale features.", "Use l1_ratio in [.1,.5,.7,.9,.99,1]."],
                    hyperparams={"cv": 5, "random_state": self.config.random_state},
                )
            )
            models.append(
                ModelRecommendation(
                    model_name="Ridge",
                    suitability="good",
                    reason="L2 regularization handles correlated features.",
                    conditions=["Scale features.", "Tune alpha via RidgeCV."],
                    hyperparams={"alpha": 1.0},
                )
            )
            if n_features > 20:
                models.append(
                    ModelRecommendation(
                        model_name="Lasso",
                        suitability="good",
                        reason="L1 regularization for automatic feature selection.",
                        conditions=["Scale features.", "Use LassoCV."],
                        hyperparams={"cv": 5, "random_state": self.config.random_state},
                    )
                )

        if has_outliers:
            models.append(
                ModelRecommendation(
                    model_name="HuberRegressor",
                    suitability="good",
                    reason="Outliers detected; Huber loss is robust.",
                    conditions=["Scale features."],
                    hyperparams={"epsilon": 1.35},
                )
            )

        self._add_tree_ensemble_models(
            models, "regression", has_multicollinearity,
            is_large, n_features, has_missing,
        )
        self._add_optional_xgboost_lightgbm(models, "regression", is_small)

        if not is_large:
            models.append(
                ModelRecommendation(
                    model_name="SVR",
                    suitability="good" if n_samples < 5_000 else "conditional",
                    reason="Can capture complex relationships; works well on small data.",
                    conditions=["Scale features.", "Tune C and gamma."],
                    hyperparams={"kernel": "rbf", "C": 1.0},
                )
            )

        if is_small:
            models.append(
                ModelRecommendation(
                    model_name="KNeighborsRegressor",
                    suitability="conditional",
                    reason="Simple non‑parametric baseline for small data.",
                    conditions=["Scale features.", "Tune n_neighbors."],
                    hyperparams={"n_neighbors": 5},
                )
            )

        models.append(
            ModelRecommendation(
                model_name="DecisionTreeRegressor",
                suitability="conditional",
                reason="Highly interpretable; prone to overfitting.",
                conditions=["Tune max_depth (3‑10)."],
                hyperparams={"max_depth": 5, "random_state": self.config.random_state},
            )
        )

        if is_large:
            models.append(
                ModelRecommendation(
                    model_name="SGDRegressor",
                    suitability="conditional",
                    reason="Scales well to large datasets; supports L1/L2/ElasticNet.",
                    conditions=["Scale features.", "Tune alpha and penalty."],
                    hyperparams={
                        "penalty": "elasticnet",
                        "random_state": self.config.random_state,
                    },
                )
            )

        return models

    def _classification_model_recommendations(
        self,
        n_samples: int,
        n_features: int,
        is_binary: bool,
        n_classes: int,
        has_multicollinearity: bool,
        has_outliers: bool,
        has_missing: bool,
        is_large: bool,
        is_small: bool,
        is_high_dimensional: bool,
    ) -> List[ModelRecommendation]:
        """Build classification model suggestions."""
        models: List[ModelRecommendation] = []

        self._add_baseline_model(models, "classification")

        if has_multicollinearity or is_high_dimensional:
            models.append(
                ModelRecommendation(
                    model_name="LogisticRegressionCV",
                    suitability="good",
                    reason="Built‑in CV and regularization (L1/L2).",
                    conditions=["Scale features.", "solver='saga' for ElasticNet."],
                    hyperparams={
                        "cv": 5,
                        "solver": "saga",
                        "max_iter": 2000,
                        "random_state": self.config.random_state,
                    },
                )
            )

        self._add_tree_ensemble_models(
            models, "classification", has_multicollinearity,
            is_large, n_features, has_missing,
        )
        self._add_optional_xgboost_lightgbm(models, "classification", is_small)

        if not is_large:
            models.append(
                ModelRecommendation(
                    model_name="SVC",
                    suitability="good" if n_samples < 5_000 else "conditional",
                    reason="RBF kernel captures complex boundaries.",
                    conditions=[
                        "Scale features.",
                        "Tune C and gamma.",
                        "probability=True for probabilities.",
                    ],
                    hyperparams={"kernel": "rbf", "C": 1.0, "probability": True},
                )
            )

        if is_small:
            models.append(
                ModelRecommendation(
                    model_name="KNeighborsClassifier",
                    suitability="conditional",
                    reason="Simple, local decision boundaries.",
                    conditions=["Scale features.", "Tune n_neighbors."],
                    hyperparams={"n_neighbors": 5},
                )
            )
            models.append(
                ModelRecommendation(
                    model_name="GaussianNB",
                    suitability="conditional",
                    reason="Fast probabilistic classifier, good on small data.",
                    conditions=["Assumes feature independence."],
                    hyperparams={},
                )
            )

        models.append(
            ModelRecommendation(
                model_name="DecisionTreeClassifier",
                suitability="conditional",
                reason="Interpretable, but prone to overfitting.",
                conditions=["Tune max_depth."],
                hyperparams={"max_depth": 5, "random_state": self.config.random_state},
            )
        )

        if is_large:
            models.append(
                ModelRecommendation(
                    model_name="SGDClassifier",
                    suitability="conditional",
                    reason="Scales to large data; supports various losses.",
                    conditions=["Scale features.", "loss='log' for probabilities."],
                    hyperparams={"loss": "log", "random_state": self.config.random_state},
                )
            )

        return models

    # ==================================================================
    # Main entry point
    # ==================================================================
    def generate_recommendations(
        self,
        analysis_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce a full set of recommendations from analysis facts.

        Parameters
        ----------
        analysis_results : dict
            The dictionary returned by `StatisticsEngine.run_full_analysis()`.
            Expected keys: 'metadata', 'duplicates', 'infinite', 'missing',
            'outliers', 'feature_profiles', 'correlation_pairs', 'target_profile'.

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
                - 'data_quality_notes': List[str]

        Raises
        ------
        RecommendationError
            If the input dictionary is malformed or required keys are missing.
        """
        # ---- Input validation ----
        try:
            analysis_results = resolve_analysis_result(analysis_results)
        except TypeError as exc:
            raise RecommendationError(
                "analysis_results must be a dictionary returned by StatisticsEngine.run_full_analysis() or an EDAAnalyzer instance."
            ) from exc

        required = [
            "duplicates", "infinite", "missing", "outliers",
            "feature_profiles", "correlation_pairs", "target_profile",
        ]
        for key in required:
            if key not in analysis_results:
                raise RecommendationError(
                    f"Missing required key '{key}' in analysis_results."
                )

        # Extract facts with safe defaults
        duplicates: Optional[DuplicateReport] = analysis_results["duplicates"]
        infinite: Optional[InfiniteReport] = analysis_results["infinite"]
        missing: Optional[MissingReport] = analysis_results["missing"]
        if missing is None:
            missing = MissingReport(
                columns_with_missing=[], column_reports=[], total_missing=0
            )
        outlier_reports: List[OutlierReport] = analysis_results["outliers"]
        feature_profiles: List[FeatureProfile] = analysis_results["feature_profiles"]
        correlation_pairs: List[CorrelationPair] = analysis_results["correlation_pairs"]
        target_profile: Optional[TargetProfile] = analysis_results["target_profile"]
        metadata: Optional[DatasetMetadata] = analysis_results.get("metadata")

        if not isinstance(feature_profiles, list):
            raise RecommendationError("'feature_profiles' must be a list.")
        if not isinstance(outlier_reports, list):
            raise RecommendationError(
                "'outliers' must be a list.",
                details=(
                    "Pass the full output of StatisticsEngine.run_full_analysis() "
                    "or EDAAnalyzer.run() so the 'outliers' key is present and typed correctly."
                ),
            )
        if not isinstance(correlation_pairs, list):
            raise RecommendationError("'correlation_pairs' must be a list.")

        outlier_columns = [o.column for o in outlier_reports if o.outlier_count > 0]

        # Build recommendations
        imputation_recs = self._imputation_recommendations(
            feature_profiles, missing, outlier_columns
        )
        outlier_recs = self._outlier_recommendations(outlier_reports)
        transformation_recs = self._transformation_recommendations(feature_profiles)
        scaling_rec = self._scaling_recommendations(feature_profiles, target_profile)
        encoding_recs = self._encoding_recommendations(feature_profiles)
        feature_eng_recs = self._feature_engineering_recommendations(
            feature_profiles, correlation_pairs
        )
        corr_recs = self._correlation_recommendations(correlation_pairs)
        pipeline = self._pipeline_suggestion(
            feature_profiles,
            missing_report=missing,
            outlier_columns=outlier_columns,
            transformation_recs=transformation_recs,
        )

        model_recs = self._model_recommendations(
            target_profile=target_profile,
            feature_profiles=feature_profiles,
            correlation_pairs=correlation_pairs,
            outlier_reports=outlier_reports,
            missing_report=missing,
            metadata=metadata,
        )

        # Data quality notes
        data_quality_notes: List[str] = []
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
        if missing.total_missing > 0:
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

    # ------------------------------------------------------------------
    # Public helper: produce a formatted summary
    # ------------------------------------------------------------------
    @staticmethod
    def summarize(recommendations: Dict[str, Any]) -> str:
        """Return a human‑readable summary of the recommendations.

        .. note::
            This static method is a presentation utility.  It may be
            relocated to a dedicated formatter module in a future version.

        Parameters
        ----------
        recommendations : dict
            The dictionary returned by `generate_recommendations()`.

        Returns
        -------
        str
            Formatted text that can be printed or saved.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DATA QUALITY NOTES")
        lines.append("=" * 60)
        for note in recommendations.get("data_quality_notes", []):
            lines.append(f"  - {note}")

        sections = [
            ("IMPUTATION", recommendations.get("imputation", [])),
            ("OUTLIER HANDLING", recommendations.get("outlier_handling", [])),
            ("TRANSFORMATIONS", recommendations.get("transformation", [])),
            (
                "SCALING",
                [recommendations.get("scaling")]
                if recommendations.get("scaling")
                else [],
            ),
            ("ENCODING", recommendations.get("encoding", [])),
            ("FEATURE ENGINEERING", recommendations.get("feature_engineering", [])),
            (
                "FEATURE SELECTION (COLLINEARITY)",
                recommendations.get("feature_selection", []),
            ),
        ]
        for title, rec_list in sections:
            if not rec_list:
                continue
            lines.append("")
            lines.append("=" * 60)
            lines.append(f"  {title}")
            lines.append("=" * 60)
            for idx, rec in enumerate(rec_list, start=1):
                if rec is None:
                    continue
                lines.append(f"  [{idx}] {rec.action}")
                lines.append(f"       Confidence: {rec.confidence:.0%}")
                if rec.risks:
                    lines.append(f"       Risks: {'; '.join(rec.risks)}")
                if rec.alternative_options:
                    lines.append(
                        f"       Alternatives: {'; '.join(rec.alternative_options)}"
                    )
                lines.append("")

        # Pipeline
        pipeline = recommendations.get("pipeline")
        if pipeline:
            lines.append("=" * 60)
            lines.append("  SUGGESTED PIPELINE")
            lines.append("=" * 60)
            lines.append(f"  Name: {pipeline.name}")
            lines.append(f"  Explanation: {pipeline.explanation}")
            lines.append("  Steps:")
            for step in pipeline.steps:
                lines.append(f"    - {step[0]}: {step[1]}")
            lines.append("")

        # Models
        models = recommendations.get("models", [])
        if models:
            lines.append("=" * 60)
            lines.append("  MODEL RECOMMENDATIONS")
            lines.append("=" * 60)
            for m in models:
                lines.append(f"  [{m.suitability.upper()}] {m.model_name}")
                lines.append(f"       {m.reason}")
                if m.conditions:
                    lines.append(f"       Conditions: {'; '.join(m.conditions)}")
                if m.hyperparams:
                    hp = ", ".join(f"{k}={v}" for k, v in m.hyperparams.items())
                    lines.append(f"       Suggested hyperparams: {hp}")
                lines.append("")

        return "\n".join(lines)
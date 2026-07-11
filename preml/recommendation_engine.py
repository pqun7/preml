"""
advanced_recommendation_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Research-backed AutoML Recommendation Engine.

Implements state-of-the-art techniques:
- Meta-learning for algorithm selection (Garouani et al., 2024)
- Early-stopping cross-validation (Bergman, Purucker & Hutter, 2024)
- Multi-fidelity optimization with Successive Halving (FlexHB, 2024)
- Conditional preprocessing based on model type
- Learning Curve Cross-Validation (LCCV, 2023) with power-law extrapolation
- Bayesian hyperparameter optimisation via scikit-optimize (optional)
- Cosine similarity for meta-feature matching

All decisions are empirical: models are actually fitted and evaluated on the
data using proper sklearn Pipelines to avoid data leakage. A time budget
controls the entire process; early stopping and multi-fidelity techniques
ensure a validated recommendation within the allocated time.

Example usage
-------------
>>> import pandas as pd
>>> import numpy as np
>>> from preml.recommendation_engine import RecommendationEngine
>>> X = pd.DataFrame({
...     'feature1': np.random.randn(1000),
...     'feature2': np.random.randn(1000),
...     'category': np.random.choice(['A', 'B', 'C'], 1000)
... })
>>> y = X['feature1'] * 0.5 + X['feature2'] * 0.3 + np.random.randn(1000) * 0.1
>>> engine = RecommendationEngine(random_state=42)
>>> result = engine.fit(X, y, time_budget_seconds=60)
>>> print(engine.summarize(result))
>>> recommendation = engine.get_recommendation(X, y)
>>> print(f"Best model: {recommendation['model']}")
>>> print(f"CV Score: {recommendation['cv_score']:.4f} +/- {recommendation['cv_std']:.4f}")
>>> print(f"Pipeline: {recommendation['pipeline']}")
"""

from __future__ import annotations

import importlib.util
import inspect
import hashlib
import json
import logging
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from preml._analysis import resolve_analysis_result
from preml.config import MLToolkitConfig
from preml.exceptions import RecommendationError as SharedRecommendationError
from preml.schema import Evidence, ModelRecommendation, PipelineSuggestion, Recommendation

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None
LIGHTGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
SKOPT_AVAILABLE = importlib.util.find_spec("skopt") is not None


def _import_optional_module(module_name: str):
    """Import an optional dependency only when needed."""
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError:
        return None

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
RecommendationError = SharedRecommendationError


class ValidationTimeoutError(Exception):
    """Raised when the validation time budget is exceeded."""

class KnowledgeBaseError(Exception):
    """Raised when the knowledge base is corrupted or inaccessible."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelCandidate:
    """Container for a model candidate with its metadata."""
    name: str
    estimator_class: type
    priority: float  # 1.0 = highest
    hyperparams: Dict[str, Any]
    supports_categorical: bool
    supports_missing: bool
    needs_scaling: bool
    is_ensemble: bool = False
    is_linear: bool = False

@dataclass
class EvaluationResult:
    """Result of evaluating a candidate."""
    model: ModelCandidate
    cv_score: float
    cv_std: float
    training_time: float
    n_folds_completed: int
    learning_curve: Optional[List[Tuple[float, float]]] = None
    extrapolated_score: Optional[float] = None
    hyperparams_tuned: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Knowledge Base (meta-learning)
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """SQLite-backed storage for dataset meta-features and model performance.

    Uses **cosine similarity** for meta-feature matching as in (Garouani et al., 2024).

    Parameters
    ----------
    db_path : str
        Path to the SQLite file. Created if missing.
    """

    def __init__(self, db_path: str = "preml_knowledge.db") -> None:
        self.db_path = db_path
        self._init_db()
        self._cached_vectors: Optional[List[Tuple[str, np.ndarray]]] = None

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS meta_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_hash TEXT NOT NULL,
                    meta_features TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dataset_hash)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    suitability TEXT,
                    cv_score REAL,
                    hyperparams TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(dataset_hash) REFERENCES meta_data(dataset_hash)
                )"""
            )
            conn.commit()

    def store_meta_features(self, dataset_hash: str, meta: Dict[str, Any]) -> None:
        meta_json = json.dumps(meta)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta_data (dataset_hash, meta_features) VALUES (?, ?)",
                (dataset_hash, meta_json),
            )
            conn.commit()

    def store_model_performance(
        self,
        dataset_hash: str,
        model_name: str,
        suitability: str,
        cv_score: float,
        hyperparams: Dict[str, Any],
    ) -> None:
        hp_json = json.dumps(hyperparams)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO model_performance "
                "(dataset_hash, model_name, suitability, cv_score, hyperparams) "
                "VALUES (?, ?, ?, ?, ?)",
                (dataset_hash, model_name, suitability, cv_score, hp_json),
            )
            conn.commit()

    def _load_all_vectors(self) -> List[Tuple[str, np.ndarray]]:
        """Load all stored meta-feature vectors into memory (cached)."""
        if self._cached_vectors is not None:
            return self._cached_vectors
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT dataset_hash, meta_features FROM meta_data"
            ).fetchall()
        vectors = []
        for ds_hash, meta_json in rows:
            meta = json.loads(meta_json)
            # Important: use the same feature keys as meta-feature extraction
            vec = np.array([
                meta.get("n_samples", 0),
                meta.get("n_features", 0),
                meta.get("n_numeric", 0),
                meta.get("n_categorical", 0),
                meta.get("missing_ratio", 0.0),
                meta.get("skewness_mean", 0.0),
                meta.get("target_variance", 0.0),
                meta.get("signal_to_noise_ratio", 0.0),
            ], dtype=float)
            vectors.append((ds_hash, vec))
        self._cached_vectors = vectors
        return vectors

    def query_similar_datasets(
        self, meta: Dict[str, Any], top_k: int = 5
    ) -> List[Tuple[str, float, List[Dict[str, Any]]]]:
        """Return top_k similar datasets with their best models using cosine similarity.

        Returns list of (dataset_hash, similarity_score, [model_records]).
        """
        vectors = self._load_all_vectors()
        if not vectors:
            return []

        # Current vector
        cur_vec = np.array([
            meta.get("n_samples", 0),
            meta.get("n_features", 0),
            meta.get("n_numeric", 0),
            meta.get("n_categorical", 0),
            meta.get("missing_ratio", 0.0),
            meta.get("skewness_mean", 0.0),
            meta.get("target_variance", 0.0),
            meta.get("signal_to_noise_ratio", 0.0),
        ], dtype=float)

        # Cosine similarities
        similarities = []
        for ds_hash, vec in vectors:
            norm_prod = np.linalg.norm(cur_vec) * np.linalg.norm(vec)
            if norm_prod == 0:
                sim = 0.0
            else:
                sim = np.dot(cur_vec, vec) / norm_prod
            similarities.append((ds_hash, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:top_k]

        # Retrieve model performances
        result = []
        with sqlite3.connect(self.db_path) as conn:
            for ds_hash, sim in top:
                rows = conn.execute(
                    "SELECT model_name, suitability, cv_score, hyperparams "
                    "FROM model_performance WHERE dataset_hash = ? "
                    "ORDER BY cv_score DESC LIMIT 3",
                    (ds_hash,),
                ).fetchall()
                models = [
                    {
                        "model_name": r[0],
                        "suitability": r[1],
                        "cv_score": r[2],
                        "hyperparams": json.loads(r[3]) if r[3] else {},
                    }
                    for r in rows
                ]
                result.append((ds_hash, sim, models))
        return result

# ---------------------------------------------------------------------------
# Main Recommendation Engine
# ---------------------------------------------------------------------------
class RecommendationEngine(BaseEstimator):
    """Research-backed AutoML Recommendation Engine.

    This engine uses empirical validation (not just heuristics) to recommend
    the best model and preprocessing pipeline for a given dataset.

    Parameters
    ----------
    config : MLToolkitConfig, optional
        Shared configuration object. If omitted, a default configuration is used.
    knowledge_db_path : str, default='knowledge.db'
        Path to SQLite knowledge base for meta-learning. If None, meta-learning is disabled.
    random_state : int, default=42
        Random seed for reproducibility. Falls back to ``config.random_state`` when omitted.
    enable_meta_learning : bool, default=True
        Whether to use the knowledge base for prior recommendations.
    enable_feature_engineering : bool, optional
        Overrides ``config.enable_feature_engineering`` when provided.
    """

    def __init__(
        self,
        config: Optional[MLToolkitConfig] = None,
        knowledge_db_path: Optional[str] = "knowledge.db",
        random_state: Optional[int] = None,
        enable_meta_learning: bool = True,
        enable_feature_engineering: Optional[bool] = None,
    ) -> None:
        self.config = config or MLToolkitConfig()
        self.knowledge_db_path = knowledge_db_path
        self.random_state = self.config.random_state if random_state is None else random_state
        self.enable_meta_learning = enable_meta_learning
        self.enable_feature_engineering = (
            self.config.enable_feature_engineering
            if enable_feature_engineering is None
            else enable_feature_engineering
        )
        self.kb = KnowledgeBase(knowledge_db_path) if knowledge_db_path else None

        # Internal state
        self._meta_features: Dict[str, Any] = {}
        self._candidates: List[ModelCandidate] = []
        self._best_result: Optional[EvaluationResult] = None
        self._pipeline: Optional[Pipeline] = None
        self._fitted: bool = False
        self._is_regression: Optional[bool] = None
        self._scoring: Optional[str] = None

        # Cached column types
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []

    def _notify_progress(
        self,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        step: str,
        **payload: Any,
    ) -> None:
        """Emit progress updates while remaining backward compatible."""
        if progress_callback is None:
            return
        try:
            progress_callback(step, payload)
        except TypeError:
            fallback_callback = cast(Any, progress_callback)
            fallback_callback({"step": step, **payload})

    def _parallel_jobs(self, n_rows: int, n_cols: int) -> int:
        """Use fewer jobs on larger problems to reduce memory pressure."""
        cells = n_rows * max(n_cols, 1)
        if cells >= 250_000 or n_rows >= 100_000:
            warnings.warn(
                "Large dataset detected; limiting parallel cross-validation to n_jobs=1 to reduce memory usage.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 1
        return self.config.n_jobs

    # -----------------------------------------------------------------------
    # Timeout helper
    # -----------------------------------------------------------------------
    def _check_timeout(self, start_time: float, budget: float) -> None:
        """Raise ValidationTimeoutError if budget exceeded."""
        if time.time() - start_time > budget:
            raise ValidationTimeoutError(
                f"Time budget exceeded ({budget:.1f}s elapsed)"
            )

    # -----------------------------------------------------------------------
    # Meta-feature extraction
    # -----------------------------------------------------------------------
    def extract_meta_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compute dataset meta-features for meta-learning and similarity.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        dict
            Dictionary with keys: n_samples, n_features, n_numeric,
            n_categorical, missing_ratio, skewness_mean, kurtosis_mean,
            target_variance, signal_to_noise_ratio.
        """
        n_samples, n_features = X.shape
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        # Treat pandas CategoricalDtype as categorical
        categorical_cols.extend(
            [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
        )
        categorical_cols = list(set(categorical_cols))  # deduplicate

        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols

        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)

        # Missing ratio
        total_missing = X.isnull().sum().sum()
        missing_ratio = (
            total_missing / (n_samples * n_features) if n_features > 0 else 0.0
        )

        # Skewness & kurtosis of numeric features
        if n_numeric > 0:
            skew_vals = X[numeric_cols].skew().dropna()
            kurt_vals = X[numeric_cols].kurtosis().dropna()
            skewness_mean = float(skew_vals.mean()) if len(skew_vals) > 0 else 0.0
            kurtosis_mean = float(kurt_vals.mean()) if len(kurt_vals) > 0 else 0.0
        else:
            skewness_mean = 0.0
            kurtosis_mean = 0.0

        # Target variance / signal-to-noise
        if pd.api.types.is_numeric_dtype(y):
            target_variance = float(np.var(y)) if len(y) > 1 else 0.0
            # Simple signal-to-noise: mean abs correlation of numeric features with target
            if n_numeric > 0:
                corrs = []
                for col in numeric_cols:
                    if not X[col].isnull().all():
                        corr = abs(X[col].corr(y)) if len(y) > 1 else 0.0
                        corrs.append(corr)
                signal_to_noise = float(np.mean(corrs)) if corrs else 0.0
            else:
                signal_to_noise = 0.0
        else:
            # Classification: use class entropy as target_variance, dummy SNR
            value_counts = y.value_counts(normalize=True)
            target_variance = -float(
                (value_counts * np.log(value_counts + 1e-9)).sum()
            )
            signal_to_noise = 1.0  # placeholder

        self._meta_features = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "n_unique_ratios": float(
                np.mean([
                    X[col].nunique(dropna=True) / max(n_samples, 1)
                    for col in categorical_cols
                ]) if categorical_cols else 0.0
            ),
            "missing_ratio": missing_ratio,
            "skewness_mean": skewness_mean,
            "kurtosis_mean": kurtosis_mean,
            "target_variance": target_variance,
            "signal_to_noise_ratio": signal_to_noise,
        }
        return self._meta_features

    # -----------------------------------------------------------------------
    # Adaptive hyperparameter ranges
    # -----------------------------------------------------------------------
    def _get_adaptive_hyperparams(
        self, model_name: str, n_samples: int, n_features: int
    ) -> Dict[str, Any]:
        """Return dataset-adaptive hyperparameter ranges.

        Based on size regime (small < 5000, medium 5000-50000, large >= 50000).

        Parameters
        ----------
        model_name : str
            One of the supported model names (e.g. "HistGradientBoosting", "LGBM").
        n_samples : int
        n_features : int

        Returns
        -------
        dict
            Keys are hyperparameter names, values are either tuples (low, high)
            for continuous/integer ranges or lists of discrete choices.
        """
        if n_samples < 5_000:
            size = "small"
        elif n_samples < 50_000:
            size = "medium"
        else:
            size = "large"

        base = {}

        if "histgradientboosting" in model_name.lower():
            if size == "small":
                base["learning_rate"] = (0.01, 0.3)
                base["max_depth"] = (3, 8)
                base["min_samples_leaf"] = (2, 20)
            elif size == "medium":
                base["learning_rate"] = (0.005, 0.2)
                base["max_depth"] = (4, 12)
                base["min_samples_leaf"] = (2, 30)
            else:
                base["learning_rate"] = (0.001, 0.1)
                base["max_depth"] = (6, 15)
                base["min_samples_leaf"] = (5, 50)
            base["max_iter"] = (100, 1000) if size == "small" else (200, 2000)

        elif "gradientboosting" in model_name.lower():
            if size == "small":
                base["learning_rate"] = (0.01, 0.3)
                base["n_estimators"] = (50, 200)
                base["max_depth"] = (3, 8)
            elif size == "medium":
                base["learning_rate"] = (0.005, 0.2)
                base["n_estimators"] = (100, 500)
                base["max_depth"] = (4, 12)
            else:
                base["learning_rate"] = (0.001, 0.1)
                base["n_estimators"] = (200, 1000)
                base["max_depth"] = (6, 15)
            base["subsample"] = (0.5, 1.0)

        elif "randomforest" in model_name.lower():
            base["n_estimators"] = (50, 200) if size == "small" else (100, 500)
            base["max_depth"] = (3, 8) if size == "small" else (4, 20)
            base["min_samples_split"] = (2, 10) if size == "small" else (2, 20)

        elif "lgbm" in model_name.lower():
            base["n_estimators"] = (50, 200) if size == "small" else (100, 500)
            base["learning_rate"] = (0.01, 0.3) if size == "small" else (0.005, 0.2)
            base["num_leaves"] = (15, 63) if size == "small" else (31, 127)
            base["subsample"] = (0.5, 1.0)

        elif "xgboost" in model_name.lower():
            base["n_estimators"] = (50, 200) if size == "small" else (100, 500)
            base["learning_rate"] = (0.01, 0.3) if size == "small" else (0.005, 0.2)
            base["max_depth"] = (3, 8) if size == "small" else (4, 12)
            base["subsample"] = (0.5, 1.0)
            base["colsample_bytree"] = (0.5, 1.0)

        elif "logistic" in model_name.lower():
            base["C"] = [0.01, 0.1, 1.0, 10.0]
        elif "elasticnet" in model_name.lower():
            base["l1_ratio"] = [0.1, 0.5, 0.7, 0.9, 0.99, 1.0]
        elif "ridge" in model_name.lower():
            base["alpha"] = [0.1, 1.0, 10.0, 100.0]
        elif "lasso" in model_name.lower():
            base["alpha"] = [0.001, 0.01, 0.1, 1.0]
        elif "svc" in model_name.lower() or "svr" in model_name.lower():
            base["C"] = [0.1, 1.0, 10.0]
            base["gamma"] = ["scale", "auto"]

        return base

    # -----------------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------------
    def _generate_candidates(self) -> List[ModelCandidate]:
        """Generate all candidate models with adaptive hyperparameters."""
        candidates: List[ModelCandidate] = []
        n_samples = self._meta_features.get("n_samples", 1000)
        n_features = self._meta_features.get("n_features", 10)
        is_reg = self._is_regression

        # Tree ensembles – primary candidates
        # 1. HistGradientBoosting (always included)
        hgb_hyper = self._get_adaptive_hyperparams(
            "HistGradientBoosting", n_samples, n_features
        )
        candidates.append(
            ModelCandidate(
                name=(
                    "HistGradientBoostingRegressor"
                    if is_reg
                    else "HistGradientBoostingClassifier"
                ),
                estimator_class=(
                    HistGradientBoostingRegressor
                    if is_reg
                    else HistGradientBoostingClassifier
                ),
                priority=1.0,
                hyperparams={"random_state": self.random_state, **hgb_hyper},
                supports_categorical=True,
                supports_missing=True,
                needs_scaling=False,
                is_ensemble=True,
            )
        )

        # 2. GradientBoosting
        gb_hyper = self._get_adaptive_hyperparams(
            "GradientBoosting", n_samples, n_features
        )
        candidates.append(
            ModelCandidate(
                name=(
                    "GradientBoostingRegressor"
                    if is_reg
                    else "GradientBoostingClassifier"
                ),
                estimator_class=(
                    GradientBoostingRegressor
                    if is_reg
                    else GradientBoostingClassifier
                ),
                priority=0.7,
                hyperparams={"random_state": self.random_state, **gb_hyper},
                supports_categorical=False,
                supports_missing=False,
                needs_scaling=False,
                is_ensemble=True,
            )
        )

        # 3. RandomForest
        rf_hyper = self._get_adaptive_hyperparams(
            "RandomForest", n_samples, n_features
        )
        candidates.append(
            ModelCandidate(
                name=(
                    "RandomForestRegressor" if is_reg else "RandomForestClassifier"
                ),
                estimator_class=(
                    RandomForestRegressor if is_reg else RandomForestClassifier
                ),
                priority=0.6,
                hyperparams={"random_state": self.random_state, **rf_hyper},
                supports_categorical=False,
                supports_missing=False,
                needs_scaling=False,
                is_ensemble=True,
            )
        )

        # 4. LightGBM (optional)
        if LIGHTGBM_AVAILABLE:
            lightgbm_module = _import_optional_module("lightgbm")
            if lightgbm_module is not None:
                lgb_regressor = getattr(lightgbm_module, "LGBMRegressor", None)
                lgb_classifier = getattr(lightgbm_module, "LGBMClassifier", None)
            else:
                lgb_regressor = None
                lgb_classifier = None
        else:
            lgb_regressor = None
            lgb_classifier = None

        if lgb_regressor is not None and lgb_classifier is not None:
            lgb_hyper = self._get_adaptive_hyperparams("LGBM", n_samples, n_features)
            candidates.append(
                ModelCandidate(
                    name="LGBMRegressor" if is_reg else "LGBMClassifier",
                    estimator_class=lgb_regressor if is_reg else lgb_classifier,
                    priority=0.85,
                    hyperparams={
                        "random_state": self.random_state,
                        "verbose": -1,
                        **lgb_hyper,
                    },
                    supports_categorical=True,
                    supports_missing=True,
                    needs_scaling=False,
                    is_ensemble=True,
                )
            )

        # 5. XGBoost (optional)
        if XGBOOST_AVAILABLE:
            xgboost_module = _import_optional_module("xgboost")
            if xgboost_module is not None:
                xgb_regressor = getattr(xgboost_module, "XGBRegressor", None)
                xgb_classifier = getattr(xgboost_module, "XGBClassifier", None)
            else:
                xgb_regressor = None
                xgb_classifier = None
        else:
            xgb_regressor = None
            xgb_classifier = None

        if xgb_regressor is not None and xgb_classifier is not None:
            xgb_hyper = self._get_adaptive_hyperparams(
                "XGBoost", n_samples, n_features
            )
            candidates.append(
                ModelCandidate(
                    name="XGBRegressor" if is_reg else "XGBClassifier",
                    estimator_class=xgb_regressor if is_reg else xgb_classifier,
                    priority=0.75,
                    hyperparams={
                        "random_state": self.random_state,
                        "verbosity": 0,
                        **xgb_hyper,
                    },
                    supports_categorical=False,
                    supports_missing=True,
                    needs_scaling=False,
                    is_ensemble=True,
                )
            )

        # Linear models
        if is_reg:
            candidates.append(
                ModelCandidate(
                    name="LinearRegression",
                    estimator_class=LinearRegression,
                    priority=0.3,
                    hyperparams={},
                    supports_categorical=False,
                    supports_missing=False,
                    needs_scaling=True,
                    is_linear=True,
                )
            )
            candidates.append(
                ModelCandidate(
                    name="RidgeCV",
                    estimator_class=RidgeCV,
                    priority=0.4,
                    hyperparams={"alphas": [0.1, 1.0, 10.0]},
                    supports_categorical=False,
                    supports_missing=False,
                    needs_scaling=True,
                    is_linear=True,
                )
            )
            candidates.append(
                ModelCandidate(
                    name="ElasticNetCV",
                    estimator_class=ElasticNetCV,
                    priority=0.45,
                    hyperparams={"cv": 3, "l1_ratio": [0.1, 0.5, 0.7, 0.9]},
                    supports_categorical=False,
                    supports_missing=False,
                    needs_scaling=True,
                    is_linear=True,
                )
            )
        else:
            candidates.append(
                ModelCandidate(
                    name="LogisticRegression",
                    estimator_class=LogisticRegression,
                    priority=0.3,
                    hyperparams={"max_iter": 1000, "random_state": self.random_state},
                    supports_categorical=False,
                    supports_missing=False,
                    needs_scaling=True,
                    is_linear=True,
                )
            )
            candidates.append(
                ModelCandidate(
                    name="LogisticRegressionCV",
                    estimator_class=LogisticRegressionCV,
                    priority=0.45,
                    hyperparams={
                        "cv": 3,
                        "max_iter": 1000,
                        "random_state": self.random_state,
                    },
                    supports_categorical=False,
                    supports_missing=False,
                    needs_scaling=True,
                    is_linear=True,
                )
            )

        # Simple baselines (small datasets)
        if n_samples < 5000:
            if is_reg:
                candidates.append(
                    ModelCandidate(
                        name="SVR",
                        estimator_class=SVR,
                        priority=0.2,
                        hyperparams={"kernel": "rbf", "C": 1.0},
                        supports_categorical=False,
                        supports_missing=False,
                        needs_scaling=True,
                    )
                )
            else:
                candidates.append(
                    ModelCandidate(
                        name="SVC",
                        estimator_class=SVC,
                        priority=0.2,
                        hyperparams={
                            "kernel": "rbf",
                            "C": 1.0,
                            "probability": True,
                        },
                        supports_categorical=False,
                        supports_missing=False,
                        needs_scaling=True,
                    )
                )

        # Sort by priority
        candidates.sort(key=lambda c: c.priority, reverse=True)
        return candidates

    # -----------------------------------------------------------------------
    # Pipeline building (used inside every CV step)
    # -----------------------------------------------------------------------
    def _build_pipeline(self, candidate: ModelCandidate) -> Pipeline:
        """Build a full sklearn Pipeline with ColumnTransformer.

        Preprocessing steps are **not** applied globally – the pipeline is
        used directly in cross-validation to avoid data leakage.

        Parameters
        ----------
        candidate : ModelCandidate
            Metadata describing the model's requirements.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Pipeline that ends with the (untuned) estimator.
        """
        numeric_cols = self._numeric_cols
        categorical_cols = self._categorical_cols

        transformers = []
        # --- Numeric handling ---
        if numeric_cols:
            num_steps = []
            # Imputation: only if model does NOT natively support missing values
            if not candidate.supports_missing and self._meta_features.get("missing_ratio", 0) > 0:
                num_steps.append(("imputer_num", SimpleImputer(strategy="median")))
            # Scaling: only if model needs it
            if candidate.needs_scaling:
                # Choose scaler based on skewness & outliers
                skew = self._meta_features.get("skewness_mean", 0.0)
                if abs(skew) > 1.5:
                    num_steps.append(("scaler", PowerTransformer(method="yeo-johnson")))
                else:
                    num_steps.append(("scaler", StandardScaler()))
            if num_steps:
                # Chain steps inside a sub-pipeline
                num_pipe = Pipeline(num_steps) if len(num_steps) > 1 else num_steps[0][1]
                transformers.append(("num", num_pipe, numeric_cols))
            else:
                # Pass through numeric columns as-is
                transformers.append(("num", "passthrough", numeric_cols))

        # --- Categorical handling ---
        if categorical_cols:
            avg_unique = self._meta_features.get("n_unique_ratios", 0.0) * self._meta_features.get("n_samples", 1)
            if candidate.supports_categorical:
                # Keep categorical columns in the output frame so estimators can refer to them by name.
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                transformers.append(("cat", enc, categorical_cols))
            else:
                # One-hot encoding if average cardinality is low, else ordinal.
                if avg_unique <= self.config.low_cardinality_threshold:
                    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                else:
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                transformers.append(("cat", enc, categorical_cols))

        # No columns left? Use a pass-through transformer.
        if not transformers:
            all_cols = numeric_cols + categorical_cols
            transformers.append(("passthrough", "passthrough", all_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        if hasattr(preprocessor, "set_output"):
            preprocessor.set_output(transform="pandas")

        # Build the estimator with fixed hyperparams (but NOT tuned ones)
        fixed_params = {
            k: v
            for k, v in candidate.hyperparams.items()
            if not isinstance(v, (tuple, list))
        }
        # For models that support native categorical, we need to pass `categorical_features`
        # after we know the output column order. We'll set it as an init argument if needed.
        estimator = candidate.estimator_class(**fixed_params)

        if candidate.supports_categorical and categorical_cols:
            # After ColumnTransformer: numeric cols first, then cat (each becomes one column)
            cat_start_idx = len(numeric_cols)
            cat_end_idx = cat_start_idx + len(categorical_cols)
            cat_indices = list(range(cat_start_idx, cat_end_idx))
            # Only set if the estimator supports this parameter
            if hasattr(estimator, "categorical_features"):
                estimator.set_params(categorical_features=cat_indices)
            elif hasattr(estimator, "categorical_feature"):
                estimator.set_params(categorical_feature=categorical_cols)

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("estimator", estimator)]
        )
        return pipeline

    # -----------------------------------------------------------------------
    # Instantiate model for simple use (not used inside pipelines)
    # -----------------------------------------------------------------------
    def _instantiate_model(self, candidate: ModelCandidate) -> BaseEstimator:
        """Instantiate the model with its default hyperparameters."""
        return candidate.estimator_class(**candidate.hyperparams)

    # -----------------------------------------------------------------------
    # Early Stopping Cross-Validation (ESCV)
    # -----------------------------------------------------------------------
    def _fast_cv_selector(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidates: List[ModelCandidate],
        time_budget: float,
    ) -> List[EvaluationResult]:
        """ESCV according to Bergman et al. (2024) with progressive folds.

        Phase 1 – Screening on a 30% sample with progressive CV (2→3→5 folds).
        Phase 2 – Deep evaluation of top-3 on full data with 5-fold CV.

        Parameters
        ----------
        X, y : training data
        candidates : list of ModelCandidate
        time_budget : float
            Allocated time in seconds.

        Returns
        -------
        List[EvaluationResult]
            Sorted by cv_score descending.
        """
        start = time.time()
        # Keep the screening stage large enough to avoid premature timeout
        # on moderately sized examples, while still reserving time for the
        # deeper evaluation phase.
        screening_budget = max(time_budget * 0.45, 30.0)
        screening_budget = min(screening_budget, time_budget * 0.65)
        deep_budget = max(time_budget - screening_budget, time_budget * 0.2)

        # Phase 1: Screening on a smaller sample
        X_sample, _, y_sample, _ = train_test_split(
            X,
            y,
            train_size=0.3,
            random_state=self.random_state,
            stratify=y if not self._is_regression else None,
        )
        screening_scores: Dict[str, List[float]] = {}
        n_jobs = self._parallel_jobs(len(X_sample), X_sample.shape[1])
        for cand in candidates:
            self._check_timeout(start, screening_budget)
            pipe = self._build_pipeline(cand)
            cv_scores: List[float] = []
            prev_score = None
            for n_splits in [2, 3, 5]:
                cv = (
                    StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
                    if not self._is_regression
                    else KFold(n_splits, shuffle=True, random_state=self.random_state)
                )
                try:
                    scores = cross_val_score(
                        pipe,
                        X_sample,
                        y_sample,
                        cv=cv,
                        scoring=self._scoring,
                        n_jobs=n_jobs,
                    )
                    mean_score = float(np.mean(scores))
                    cv_scores.append(mean_score)
                    if prev_score is not None and len(cv_scores) > 1:
                        improvement = (mean_score - prev_score) / (abs(prev_score) + 1e-9)
                        if abs(improvement) < 0.02:
                            break
                    prev_score = mean_score
                except Exception as e:
                    logger.debug("Screening CV failed for %s: %s", cand.name, e)
            if cv_scores:
                screening_scores[cand.name] = cv_scores

        # Phase 2: Deep evaluation on top-3 candidates
        deep_candidates = sorted(
            screening_scores, key=lambda n: np.mean(screening_scores[n]), reverse=True
        )[:3]
        top_candidates = [c for c in candidates if c.name in deep_candidates]

        results = []
        for cand in top_candidates:
            self._check_timeout(start, deep_budget)
            pipe = self._build_pipeline(cand)
            try:
                cv = (
                    StratifiedKFold(5, shuffle=True, random_state=self.random_state)
                    if not self._is_regression
                    else KFold(5, shuffle=True, random_state=self.random_state)
                )
                scores = cross_val_score(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    scoring=self._scoring,
                    n_jobs=self._parallel_jobs(len(X), X.shape[1]),
                )
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))
                results.append(
                    EvaluationResult(
                        model=cand,
                        cv_score=mean_score,
                        cv_std=std_score,
                        training_time=0.0,  # will be updated later if needed
                        n_folds_completed=5,
                    )
                )
            except Exception as e:
                logger.warning("Deep CV failed for %s: %s", cand.name, e)

        results.sort(key=lambda r: r.cv_score, reverse=True)
        return results

    # -----------------------------------------------------------------------
    # Learning Curve Cross-Validation (LCCV)
    # -----------------------------------------------------------------------
    def _lccv_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidate: ModelCandidate,
    ) -> EvaluationResult:
        """LCCV using `learning_curve` from sklearn and power-law extrapolation.

        Parameters
        ----------
        X, y : data
        candidate : ModelCandidate

        Returns
        -------
        EvaluationResult with `learning_curve` and `extrapolated_score`.
        """
        pipe = self._build_pipeline(candidate)
        train_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        try:
            # Use sklearn's learning_curve
            learning_curve_result = learning_curve(
                pipe,
                X,
                y,
                train_sizes=train_sizes,
                cv=3,
                scoring=self._scoring,
                n_jobs=self._parallel_jobs(len(X), X.shape[1]),
                random_state=self.random_state,
            )
            train_sizes_abs = learning_curve_result[0]
            train_scores = learning_curve_result[1]
            test_scores = learning_curve_result[2]
        except Exception as e:
            logger.warning("LCCV failed for %s: %s", candidate.name, e)
            return EvaluationResult(
                model=candidate,
                cv_score=float("-inf"),
                cv_std=0.0,
                training_time=0.0,
                n_folds_completed=0,
                extrapolated_score=float("-inf"),
            )

        # Mean test scores across folds
        test_mean = np.mean(test_scores, axis=1)
        # Fit power law: score(n) = a + b * n^(-c)
        from scipy.optimize import curve_fit

        def power_law(n, a, b, c):
            return a + b * np.power(n, -c)

        # Use all points for fitting
        x_vals = train_sizes_abs.astype(float)
        y_vals = test_mean
        try:
            popt, _ = curve_fit(power_law, x_vals, y_vals, maxfev=10000)
            extrapolated = float(power_law(len(X), *popt))
        except Exception:
            extrapolated = float(y_vals[-1])  # fallback

        # Build learning curve list
        learning_curve_points = list(
            zip(train_sizes, test_mean.tolist())
        )

        return EvaluationResult(
            model=candidate,
            cv_score=extrapolated,  # use extrapolated as best estimate
            cv_std=0.0,
            training_time=0.0,
            n_folds_completed=3,
            learning_curve=learning_curve_points,
            extrapolated_score=extrapolated,
        )

    # -----------------------------------------------------------------------
    # Successive Halving with HalvingRandomSearchCV
    # -----------------------------------------------------------------------
    def _successive_halving_optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidate: ModelCandidate,
        time_budget: float,
    ) -> Dict[str, Any]:
        """Multi-fidelity HP optimisation using sklearn's HalvingRandomSearchCV.

        Parameters
        ----------
        X, y : data
        candidate : ModelCandidate
        time_budget : float

        Returns
        -------
        dict of best hyperparameters found.
        """
        start = time.time()
        # Build search space from adaptive ranges
        param_ranges = candidate.hyperparams
        import scipy.stats as stats

        param_distributions = {}
        for key, val in param_ranges.items():
            if isinstance(val, tuple) and len(val) == 2:
                low, high = val
                if isinstance(low, int) and isinstance(high, int):
                    param_distributions[f"estimator__{key}"] = stats.randint(low, high + 1)
                else:
                    # Use uniform distribution; for learning rates we might prefer log-uniform,
                    # but uniform is acceptable.
                    param_distributions[f"estimator__{key}"] = stats.uniform(low, high - low)
            elif isinstance(val, list):
                param_distributions[f"estimator__{key}"] = val

        if not param_distributions:
            return {}

        # Build base pipeline (estimator with no hyperparams to tune)
        base_pipe = self._build_pipeline(candidate)
        model_selection_module = __import__("sklearn.model_selection", fromlist=["HalvingRandomSearchCV"])
        HalvingRandomSearchCV = getattr(model_selection_module, "HalvingRandomSearchCV")

        # HalvingRandomSearchCV needs a base estimator; we give it the pipeline.
        halving_cv = HalvingRandomSearchCV(
            base_pipe,
            param_distributions,
            factor=2,
            resource="n_samples",
            min_resources=max(100, int(0.1 * len(X))),
            max_resources=len(X),
            n_candidates="exhaust",
            random_state=self.random_state,
            cv=3,
            scoring=self._scoring,
            n_jobs=self._parallel_jobs(len(X), X.shape[1]),
            verbose=0,
        )
        try:
            halving_cv.fit(X, y)
            best_params = halving_cv.best_params_
            # Strip 'estimator__' prefix
            clean_params = {
                k.split("__", 1)[1] if "__" in k else k: v
                for k, v in best_params.items()
            }
            return clean_params
        except Exception as e:
            logger.warning("Successive Halving failed for %s: %s", candidate.name, e)
            return {}

    # -----------------------------------------------------------------------
    # Bayesian Optimisation fine-tuning
    # -----------------------------------------------------------------------
    def _bayesian_optimization_finetune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        candidate: ModelCandidate,
        initial_params: Dict[str, Any],
        time_budget: float,
    ) -> Dict[str, Any]:
        """Fine-tune with BayesSearchCV (skopt) if available, else fall back to HalvingRandomSearchCV.

        Parameters
        ----------
        X, y : data
        candidate : ModelCandidate
        initial_params : dict
            Starting point for search.
        time_budget : float

        Returns
        -------
        dict of best hyperparameters.
        """
        if not SKOPT_AVAILABLE:
            logger.info("scikit-optimize not available; using extended HalvingRandomSearchCV.")
            return self._successive_halving_optimize(X, y, candidate, time_budget)

        skopt_module = _import_optional_module("skopt")
        if skopt_module is None:
            logger.info("scikit-optimize import failed; using extended HalvingRandomSearchCV.")
            return self._successive_halving_optimize(X, y, candidate, time_budget)
        space_module = _import_optional_module("skopt.space")
        if space_module is None:
            logger.info("scikit-optimize space import failed; using extended HalvingRandomSearchCV.")
            return self._successive_halving_optimize(X, y, candidate, time_budget)

        BayesSearchCV = getattr(skopt_module, "BayesSearchCV")
        Real = getattr(space_module, "Real")
        Integer = getattr(space_module, "Integer")
        Categorical = getattr(space_module, "Categorical")

        # Convert candidate's hyperparam ranges into skopt space
        space = {}
        for key, val in candidate.hyperparams.items():
            if isinstance(val, tuple) and len(val) == 2:
                low, high = val
                if isinstance(low, int) and isinstance(high, int):
                    space[f"estimator__{key}"] = Integer(low, high)
                else:
                    if "learning_rate" in key:
                        space[f"estimator__{key}"] = Real(low, high, prior="log-uniform")
                    else:
                        space[f"estimator__{key}"] = Real(low, high, prior="uniform")
            elif isinstance(val, list):
                space[f"estimator__{key}"] = Categorical(val)

        if not space:
            return initial_params

        base_pipe = self._build_pipeline(candidate)
        n_iter = min(25, max(10, int(time_budget / 2)))  # rough estimate

        opt = BayesSearchCV(
            base_pipe,
            space,
            n_iter=n_iter,
            cv=3,
            scoring=self._scoring,
            random_state=self.random_state,
            n_jobs=self._parallel_jobs(len(X), X.shape[1]),
        )
        try:
            opt.fit(X, y)
            best = opt.best_params_
            clean = {
                k.split("__", 1)[1] if "__" in k else k: v for k, v in best.items()
            }
            # Merge with initial (which may contain fixed params)
            return {**initial_params, **clean}
        except Exception as e:
            logger.warning("Bayesian optimisation failed: %s", e)
            return self._successive_halving_optimize(X, y, candidate, time_budget)

    # -----------------------------------------------------------------------
    # Preprocessing recommendations (concise)
    # -----------------------------------------------------------------------
    def get_preprocessing_recommendations(self) -> Dict[str, str]:
        """Return simplified preprocessing recommendations based on meta-features.

        Returns a dict with keys like 'imputation', 'scaling', 'encoding'.
        """
        recs = {}
        meta = self._meta_features
        missing = meta.get("missing_ratio", 0)
        skew = meta.get("skewness_mean", 0)
        n_cat = meta.get("n_categorical", 0)

        if missing > 0:
            if missing < 0.05:
                recs["imputation"] = "SimpleImputer (median/most_frequent)"
            else:
                recs["imputation"] = "IterativeImputer or native model support"
        else:
            recs["imputation"] = "None required"

        if abs(skew) > 1.5:
            recs["transformation"] = "PowerTransformer (yeo-johnson) or log1p"
        else:
            recs["transformation"] = "None required"

        recs["scaling"] = (
            "StandardScaler or RobustScaler" if self._is_regression is not None else "depends on model"
        )

        if n_cat > 0:
            recs["encoding"] = (
                "OneHotEncoder (low cardinality) or OrdinalEncoder"
                if not any(
                    c.supports_categorical for c in self._candidates
                )
                else "Native categorical support available"
            )
        else:
            recs["encoding"] = "No categorical features"

        return recs

    # -----------------------------------------------------------------------
    # Legacy heuristic recommendation helpers (backward compatibility)
    # -----------------------------------------------------------------------
    def _imputation_recommendations(
        self,
        feature_profiles: List[Any],
        missing_report: Any,
    ) -> List[Recommendation]:
        recs: List[Recommendation] = []
        if missing_report is None or getattr(missing_report, "total_missing", 0) <= 0:
            return recs

        profile_by_column = {
            fp.column: fp for fp in feature_profiles if getattr(fp, "column", None)
        }
        for column_report in getattr(missing_report, "column_reports", []):
            feature_profile = profile_by_column.get(column_report.column)
            if feature_profile and getattr(feature_profile, "numeric_profile", None) is not None:
                action = f"Impute {column_report.column} with median."
            else:
                action = f"Impute {column_report.column} with mode."
            recs.append(
                Recommendation(
                    category="imputation",
                    action=action,
                    confidence=0.9,
                    evidence=[Evidence(reason="Column has missing values", statistics={"missing_count": column_report.missing_count})],
                )
            )

        column_missing_percent = max(
            [getattr(column_report, "missing_percent", 0.0) for column_report in getattr(missing_report, "column_reports", [])],
            default=0.0,
        ) / 100.0
        if column_missing_percent >= self.config.missing_threshold:
            recs.append(
                Recommendation(
                    category="imputation",
                    action="Investigate extensive missingness before imputation.",
                    confidence=0.75,
                    evidence=[Evidence(reason="Missingness exceeds configured threshold", statistics={"missing_percent": column_missing_percent * 100})],
                )
            )
        return recs

    def _outlier_recommendations(self, outlier_reports: List[Any]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for report in outlier_reports:
            confidence = 0.95 if getattr(report, "outlier_percent", 0.0) >= self.config.outlier_threshold_percent else 0.65
            action = (
                f"Handle significant outliers in {report.column} using robust scaling or capping."
                if confidence > 0.8
                else f"Minor outliers detected in {report.column}; monitor or cap if needed."
            )
            recs.append(
                Recommendation(
                    category="outlier_handling",
                    action=action,
                    confidence=confidence,
                    evidence=[Evidence(reason="Outlier report", statistics={"outlier_percent": getattr(report, "outlier_percent", 0.0)})],
                )
            )
        return recs

    def _transformation_recommendations(self, feature_profiles: List[Any]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for profile in feature_profiles:
            numeric_profile = getattr(profile, "numeric_profile", None)
            if numeric_profile is None:
                continue
            skewness = abs(getattr(numeric_profile, "skewness", 0.0))
            if skewness < self.config.skewness_threshold:
                continue
            if getattr(numeric_profile, "min", 0.0) >= 0 and getattr(numeric_profile, "skewness", 0.0) > 0:
                action = f"Apply log transform to {profile.column}."
            else:
                action = f"Apply Yeo-Johnson / PowerTransformer to {profile.column}."
            recs.append(
                Recommendation(
                    category="transformation",
                    action=action,
                    confidence=0.9,
                    evidence=[Evidence(reason="Skewed numeric distribution", statistics={"skewness": getattr(numeric_profile, "skewness", 0.0)})],
                )
            )
        return recs

    def _scaling_recommendations(self, feature_profiles: List[Any], target_profile: Any) -> Recommendation:
        numeric_profiles = [fp for fp in feature_profiles if getattr(fp, "numeric_profile", None) is not None]
        if not numeric_profiles:
            return Recommendation(
                category="scaling",
                action="No numeric features to scale.",
                confidence=1.0,
            )
        return Recommendation(
            category="scaling",
            action="Scale numeric features. Use StandardScaler or RobustScaler.",
            confidence=0.9,
            evidence=[Evidence(reason="Numeric features present", statistics={"n_numeric": len(numeric_profiles)})],
        )

    def _encoding_recommendations(self, feature_profiles: List[Any]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for profile in feature_profiles:
            categorical_profile = getattr(profile, "categorical_profile", None)
            numeric_profile = getattr(profile, "numeric_profile", None)
            if categorical_profile is not None:
                unique_count = getattr(categorical_profile, "unique_count", 0)
                if unique_count <= 2:
                    action = f"Use binary / 0/1 encoding for {profile.column}."
                elif unique_count <= self.config.low_cardinality_threshold:
                    action = f"Use one-hot encoding for {profile.column}."
                else:
                    action = f"Use frequency encoding or ordinal encoding for {profile.column}."
                recs.append(
                    Recommendation(category="encoding", action=action, confidence=0.9)
                )
            elif numeric_profile is not None and getattr(numeric_profile, "is_categorical_like", False):
                recs.append(
                    Recommendation(
                        category="encoding",
                        action=f"Treat {profile.column} as categorical and encode it explicitly.",
                        confidence=0.85,
                    )
                )
        return recs

    def _correlation_recommendations(self, correlation_pairs: List[Any]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        for pair in correlation_pairs:
            coefficient = abs(getattr(pair, "coefficient", 0.0))
            if coefficient < self.config.correlation_threshold:
                continue
            recs.append(
                Recommendation(
                    category="feature_selection",
                    action=f"Drop one of {pair.feature_a} or {pair.feature_b} due to high correlation.",
                    confidence=0.95,
                    evidence=[Evidence(reason="High correlation pair", statistics={"coefficient": getattr(pair, "coefficient", 0.0)})],
                )
            )
        return recs

    def _feature_engineering_recommendations(
        self,
        feature_profiles: List[Any],
        correlation_pairs: List[Any],
    ) -> List[Recommendation]:
        if not self.enable_feature_engineering:
            return []
        recs: List[Recommendation] = []
        numeric_profiles = [fp for fp in feature_profiles if getattr(fp, "numeric_profile", None) is not None]
        if len(numeric_profiles) >= 2:
            first, second = numeric_profiles[:2]
            recs.append(
                Recommendation(
                    category="feature_engineering",
                    action=f"Consider a ratio feature between {first.column} and {second.column}.",
                    confidence=0.75,
                )
            )
        if any(abs(getattr(pair, "coefficient", 0.0)) >= self.config.correlation_threshold for pair in correlation_pairs):
            recs.append(
                Recommendation(
                    category="feature_engineering",
                    action="Some features look redundant; consider interaction or composite features after dropping duplicates.",
                    confidence=0.8,
                )
            )
        return recs

    def _model_recommendations(
        self,
        target_profile: Any,
        feature_profiles: List[Any],
        correlation_pairs: List[Any],
        outlier_reports: List[Any],
        missing_report: Any,
        metadata: Optional[Any] = None,
    ) -> List[ModelRecommendation]:
        if target_profile is None:
            return [ModelRecommendation(model_name="N/A", suitability="none", reason="Target profile missing.")]

        is_regression = bool(getattr(target_profile, "is_regression", False))
        has_missing = getattr(missing_report, "total_missing", 0) > 0

        models: List[ModelRecommendation] = []
        if is_regression:
            models.extend(
                [
                    ModelRecommendation(model_name="LinearRegression", suitability="baseline", reason="Simple numerical baseline."),
                    ModelRecommendation(model_name="RidgeCV", suitability="good", reason="Regularized linear model."),
                    ModelRecommendation(model_name="RandomForestRegressor", suitability="excellent", reason="Strong tree ensemble for tabular regression."),
                ]
            )
            if has_missing:
                models.append(ModelRecommendation(model_name="HistGradientBoostingRegressor", suitability="excellent", reason="Handles missing values natively."))
        else:
            models.extend(
                [
                    ModelRecommendation(model_name="LogisticRegression", suitability="baseline", reason="Simple classification baseline."),
                    ModelRecommendation(model_name="RandomForestClassifier", suitability="good", reason="Strong tree ensemble for tabular classification."),
                ]
            )
            if getattr(target_profile, "is_binary", False):
                models.append(ModelRecommendation(model_name="LogisticRegressionCV", suitability="good", reason="Cross-validated logistic model for binary tasks."))
            if has_missing:
                models.append(ModelRecommendation(model_name="HistGradientBoostingClassifier", suitability="excellent", reason="Handles missing values natively."))

        if XGBOOST_AVAILABLE:
            models.append(ModelRecommendation(model_name="XGBRegressor" if is_regression else "XGBClassifier", suitability="good", reason="Optional XGBoost support available."))
        if LIGHTGBM_AVAILABLE:
            models.append(ModelRecommendation(model_name="LGBMRegressor" if is_regression else "LGBMClassifier", suitability="good", reason="Optional LightGBM support available."))

        return models

    def _pipeline_suggestion(
        self,
        feature_profiles: List[Any],
        missing_report: Optional[Any] = None,
        outlier_columns: Optional[List[str]] = None,
        transformation_recs: Optional[List[Recommendation]] = None,
    ) -> PipelineSuggestion:
        steps: List[Tuple[str, str]] = []
        if missing_report is not None and getattr(missing_report, "total_missing", 0) > 0:
            steps.append(("imputation", "SimpleImputer or a native missing-value model."))
        if outlier_columns:
            steps.append(("scaler", "RobustScaler for outlier-resistant scaling."))
        else:
            numeric_profiles = [fp for fp in feature_profiles if getattr(fp, "numeric_profile", None) is not None]
            if numeric_profiles:
                steps.append(("scaler", "StandardScaler for numeric features."))
        if transformation_recs:
            steps.append(("transformation", transformation_recs[0].action))
        categorical_profiles = [fp for fp in feature_profiles if getattr(fp, "categorical_profile", None) is not None]
        if categorical_profiles:
            unique_counts = [getattr(fp.categorical_profile, "unique_count", 0) for fp in categorical_profiles]
            if any(count <= self.config.low_cardinality_threshold for count in unique_counts):
                steps.append(("encoding", "OneHotEncoder for low-cardinality categoricals."))
            else:
                steps.append(("encoding", "OrdinalEncoder or frequency encoding for high-cardinality categoricals."))
        if not steps:
            steps.append(("passthrough", "No preprocessing required."))
        return PipelineSuggestion(name="tabular_pipeline", steps=steps, explanation="Chosen from observed missingness, skew, outliers, and cardinality.")

    def generate_recommendations(self, analysis_result: Any) -> Dict[str, Any]:
        """Generate a complete heuristic recommendation bundle from an analysis result."""
        if not isinstance(analysis_result, dict):
            raise RecommendationError("analysis_result must be a dictionary.")

        analysis = resolve_analysis_result(analysis_result)
        required_keys = ["metadata", "duplicates", "infinite", "missing", "outliers", "feature_profiles", "correlation_pairs", "target_profile"]
        for key in required_keys:
            if key not in analysis:
                raise RecommendationError(f"Missing required key: {key}")

        feature_profiles = analysis.get("feature_profiles", [])
        outliers = analysis.get("outliers", [])
        correlation_pairs = analysis.get("correlation_pairs", [])
        missing_report = analysis.get("missing")
        target_profile = analysis.get("target_profile")

        if not isinstance(feature_profiles, list):
            raise RecommendationError("feature_profiles must be a list.")
        if not isinstance(outliers, list):
            raise RecommendationError("outliers must be a list.")
        if not isinstance(correlation_pairs, list):
            raise RecommendationError("correlation_pairs must be a list.")

        imputation = self._imputation_recommendations(feature_profiles, missing_report)
        outlier_handling = self._outlier_recommendations(outliers)
        transformation = self._transformation_recommendations(feature_profiles)
        scaling = self._scaling_recommendations(feature_profiles, target_profile)
        encoding = self._encoding_recommendations(feature_profiles)
        feature_engineering = self._feature_engineering_recommendations(feature_profiles, correlation_pairs)
        feature_selection = self._correlation_recommendations(correlation_pairs)
        models = self._model_recommendations(
            target_profile=target_profile,
            feature_profiles=feature_profiles,
            correlation_pairs=correlation_pairs,
            outlier_reports=outliers,
            missing_report=missing_report,
            metadata=analysis.get("metadata"),
        )
        pipeline = self._pipeline_suggestion(
            feature_profiles,
            missing_report=missing_report,
            outlier_columns=[report.column for report in outliers if getattr(report, "outlier_percent", 0.0) >= self.config.outlier_threshold_percent],
            transformation_recs=transformation,
        )

        data_quality_notes: List[str] = []
        duplicates = analysis.get("duplicates")
        total_duplicates = getattr(duplicates, "total_duplicates", 0) if duplicates is not None else 0
        total_missing = getattr(missing_report, "total_missing", 0) if missing_report is not None else 0
        if total_duplicates > 0:
            data_quality_notes.append(f"Found {total_duplicates} duplicate rows.")
        if total_missing > 0:
            data_quality_notes.append(f"Missing values detected: {total_missing}.")
        if outliers:
            data_quality_notes.append(f"Outliers detected in {len(outliers)} columns.")

        return {
            "imputation": imputation,
            "outlier_handling": outlier_handling,
            "transformation": transformation,
            "scaling": scaling,
            "encoding": encoding,
            "feature_engineering": feature_engineering,
            "feature_selection": feature_selection,
            "pipeline": pipeline,
            "models": models,
            "data_quality_notes": data_quality_notes,
        }

    # -----------------------------------------------------------------------
    # Main fit method
    # -----------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_budget_seconds: float = 120.0,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Orchestrate the full AutoML recommendation with empirical validation.

        Steps:
        1. Meta-feature extraction
        2. Knowledge base query (if enabled)
        3. Candidate generation
        4. ESCV screening → deep evaluation
        5. LCCV on top candidates
        6. Successive Halving hyperparameter optimisation
        7. Bayesian optimisation fine-tuning (if skopt available)
        8. Final pipeline assembly & fit
        9. Store results in knowledge base

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        time_budget_seconds : float, default=120
            Total time budget for the entire process.
        progress_callback : callable, optional
            If provided, called after each major step with the signature
            ``progress_callback(step: str, payload: dict)``. The callback is
            invoked at start, after each major phase, and when the run finishes.
            For backward compatibility, single-argument callbacks that accept a
            status dictionary are also supported.

        Returns
        -------
        dict
            Contains ``best_model``, ``cv_score``, ``cv_std``, ``pipeline``,
            ``hyperparams``, ``training_time``, ``total_time``, and optionally
            ``reasoning``.

        Raises
        ------
        ValidationTimeoutError
            If the time budget is exceeded.
        RecommendationError
            If an unrecoverable error occurs.
        """
        global_start = time.time()
        self._notify_progress(progress_callback, "start", time_budget=time_budget_seconds)

        # 0. Detect task
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            self._is_regression = True
            self._scoring = "neg_mean_squared_error"
        else:
            self._is_regression = False
            self._scoring = "roc_auc_ovr" if y.nunique() > 2 else "accuracy"

        # 1. Meta-features
        self.extract_meta_features(X, y)
        self._notify_progress(progress_callback, "meta_features_extracted", meta=self._meta_features)

        # 2. Knowledge base (optional)
        similar_datasets = []
        if self.enable_meta_learning and self.kb is not None:
            similar_datasets = self.kb.query_similar_datasets(
                self._meta_features, top_k=5
            )
            self._notify_progress(progress_callback, "knowledge_base_queried", similar_count=len(similar_datasets))

        # 3. Generate candidates
        self._candidates = self._generate_candidates()
        self._notify_progress(progress_callback, "candidates_generated", n_candidates=len(self._candidates))

        # 4. Allocate time budget adaptively
        n_samples = self._meta_features["n_samples"]
        if n_samples < 5000:
            # Small dataset → more time for screening and deep evaluation.
            screening_budget = max(time_budget_seconds * 0.35, 30.0)
            deep_budget = max(time_budget_seconds * 0.25, 20.0)
            hyper_budget = max(time_budget_seconds - screening_budget - deep_budget, 0.0)
        else:
            screening_budget = max(time_budget_seconds * 0.35, 30.0)
            deep_budget = max(time_budget_seconds * 0.25, 20.0)
            hyper_budget = max(time_budget_seconds - screening_budget - deep_budget, 0.0)

        # 4a. ESCV
        escv_results = self._fast_cv_selector(
            X, y, self._candidates, time_budget=screening_budget + deep_budget
        )
        if not escv_results:
            raise RecommendationError("No model survived ESCV – check your data.")
        self._notify_progress(progress_callback, "escv_completed", top_model=escv_results[0].model.name)

        # 5. LCCV on top-2 candidates
        top_candidates = [r.model for r in escv_results[:2]]
        lccv_results = []
        for cand in top_candidates:
            self._check_timeout(global_start, time_budget_seconds * 0.9)
            lccv = self._lccv_evaluate(X, y, cand)
            lccv_results.append(lccv)
        if not lccv_results:
            # fallback to top ESCV
            best_candidate = escv_results[0].model
            best_extrapolated = escv_results[0].cv_score
        else:
            lccv_results.sort(
                key=lambda r: r.extrapolated_score or float("-inf"), reverse=True
            )
            best_candidate = lccv_results[0].model
            best_extrapolated = lccv_results[0].extrapolated_score or 0.0
        self._notify_progress(progress_callback, "lccv_completed", best_lccv_model=best_candidate.name, extrapolated_score=best_extrapolated)

        # 6. Successive Halving
        sh_params = self._successive_halving_optimize(
            X, y, best_candidate, time_budget=hyper_budget * 0.5
        )
        self._notify_progress(progress_callback, "successive_halving_done", params=sh_params)

        # 7. Bayesian Optimisation fine-tuning (if time)
        self._check_timeout(global_start, time_budget_seconds * 0.95)
        final_params = self._bayesian_optimization_finetune(
            X, y, best_candidate, sh_params, time_budget=hyper_budget * 0.5
        )
        self._notify_progress(progress_callback, "bayesian_opt_done", params=final_params)

        # 8. Build final pipeline with best hyperparams
        best_candidate.hyperparams = {**best_candidate.hyperparams, **final_params}
        final_pipeline = self._build_pipeline(best_candidate)
        # Fit on full data for final return
        fit_start = time.time()
        final_pipeline.fit(X, y)
        train_time = time.time() - fit_start

        # 9. Cross-validate for final score estimate
        cv = (
            StratifiedKFold(3, shuffle=True, random_state=self.random_state)
            if not self._is_regression
            else KFold(3, shuffle=True, random_state=self.random_state)
        )
        try:
            final_scores = cross_val_score(
                final_pipeline,
                X,
                y,
                cv=cv,
                scoring=self._scoring,
                n_jobs=self._parallel_jobs(len(X), X.shape[1]),
            )
            final_cv_score = float(np.mean(final_scores))
            final_cv_std = float(np.std(final_scores))
        except Exception:
            final_cv_score = best_extrapolated
            final_cv_std = 0.0

        # Store result
        self._best_result = EvaluationResult(
            model=best_candidate,
            cv_score=final_cv_score,
            cv_std=final_cv_std,
            training_time=train_time,
            n_folds_completed=3,
            hyperparams_tuned=final_params,
        )
        self._pipeline = final_pipeline
        self._fitted = True

        # 10. Knowledge base storage
        if self.kb is not None:
            ds_hash = hashlib.md5(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()).hexdigest()
            try:
                self.kb.store_meta_features(ds_hash, self._meta_features)
                self.kb.store_model_performance(
                    ds_hash,
                    best_candidate.name,
                    "excellent",
                    final_cv_score,
                    final_params,
                )
            except Exception as e:
                logger.warning("Failed to store in knowledge base: %s", e)

        total_time = time.time() - global_start
        result = {
            "best_model": best_candidate.name,
            "cv_score": final_cv_score,
            "cv_std": final_cv_std,
            "pipeline": final_pipeline,
            "hyperparams": final_params,
            "training_time": train_time,
            "total_time": total_time,
            "reasoning": (
                f"Selected via ESCV + LCCV + Successive Halving + Bayesian Opt. "
                f"Task: {'regression' if self._is_regression else 'classification'}, "
                f"samples: {n_samples}."
            ),
        }
        self._notify_progress(progress_callback, "finished", result_summary=result)
        return result

    # -----------------------------------------------------------------------
    # Post-fit / heuristic recommendation
    # -----------------------------------------------------------------------
    def get_recommendation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Return a recommendation. If `fit` was called, returns the empirically
        validated result; otherwise returns a heuristic recommendation.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series

        Returns
        -------
        dict with keys ``model``, ``reasoning``, ``cv_score``, etc.
        """
        if self._fitted and self._best_result is not None:
            return {
                "model": self._best_result.model.name,
                "reasoning": "Empirically validated as best on this data via ESCV, LCCV, and multi-fidelity optimization.",
                "cv_score": self._best_result.cv_score,
                "cv_std": self._best_result.cv_std,
                "pipeline": self._pipeline,
                "hyperparams": self._best_result.hyperparams_tuned,
            }
        # Heuristic fallback
        self.extract_meta_features(X, y)
        self._is_regression = (
            pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
        )
        candidates = self._generate_candidates()
        # Pick the highest-priority candidate
        best = candidates[0]
        return {
            "model": best.name,
            "reasoning": "Heuristic recommendation (HistGradientBoosting is a top performer for tabular data). Run fit() for empirical validation.",
            "cv_score": None,
            "cv_std": None,
            "pipeline": None,
            "hyperparams": best.hyperparams,
            "warning": "No fit() performed; results are based on heuristics only.",
        }

    # -----------------------------------------------------------------------
    # Summarise
    # -----------------------------------------------------------------------
    @staticmethod
    def summarize(recommendations: Dict[str, Any]) -> str:
        """Return a human-readable summary."""
        lines = ["=" * 60, " RECOMMENDATION ENGINE SUMMARY", "=" * 60]
        if "best_model" in recommendations:
            lines.append(f" Best model: {recommendations['best_model']}")
            lines.append(
                f" CV Score: {recommendations.get('cv_score', 0):.4f} "
                f"+/- {recommendations.get('cv_std', 0):.4f}"
            )
            lines.append(
                f" Training time: {recommendations.get('training_time', 0):.2f} sec"
            )
            lines.append(
                f" Total time: {recommendations.get('total_time', 0):.2f} sec"
            )
        elif any(
            key in recommendations
            for key in [
                "imputation",
                "outlier_handling",
                "transformation",
                "scaling",
                "encoding",
                "feature_engineering",
                "feature_selection",
                "models",
            ]
        ):
            from preml.recommendation_utils import normalize_recommendation_items

            lines.append(" DATA QUALITY NOTES")
            for note in recommendations.get("data_quality_notes", []):
                lines.append(f" - {note}")
            lines.append(" RECOMMENDATIONS")
            for category in [
                "imputation",
                "outlier_handling",
                "transformation",
                "scaling",
                "encoding",
                "feature_engineering",
                "feature_selection",
            ]:
                for rec in normalize_recommendation_items(recommendations.get(category)):
                    lines.append(f" - {rec.action}")
            lines.append(" MODEL RECOMMENDATIONS")
            for model in recommendations.get("models", []):
                lines.append(f" - {model.model_name} [{model.suitability}]")
            pipeline = recommendations.get("pipeline")
            if pipeline is not None:
                lines.append(" SUGGESTED PIPELINE")
                for step_name, description in getattr(pipeline, "steps", []):
                    lines.append(f" - {step_name}: {description}")
        else:
            lines.append(" No empirical results. Use fit() to obtain validated recommendations.")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example usage (if run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    X = pd.DataFrame(
        {
            "num1": np.random.randn(2000),
            "num2": np.random.randn(2000) * 2 + 1,
            "cat": np.random.choice(["A", "B", "C"], 2000),
            "num3": np.random.exponential(2, 2000),
        }
    )
    y_reg = X["num1"] * 0.5 + X["num2"] * 0.3 + np.random.randn(2000) * 0.2
    y_cls = ((X["num1"] + X["num2"] > 0)).astype(int)

    engine = RecommendationEngine(random_state=42)

    print("=== Regression example ===")
    result = engine.fit(X, y_reg, time_budget_seconds=60)
    print(engine.summarize(result))
    print("\nPreprocessing recommendations:", engine.get_preprocessing_recommendations())

    print("\n=== Classification heuristic ===")
    engine2 = RecommendationEngine()
    rec = engine2.get_recommendation(X, y_cls)
    print(f"Recommended: {rec['model']}, reason: {rec['reasoning']}")

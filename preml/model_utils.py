"""
model_utils.py — Baseline model training, cross‑validation, and metrics.

This module provides production‑ready utilities for:
- Creating simple baseline pipelines from a preprocessing ColumnTransformer.
- Evaluating models via cross‑validation.
- Computing standard regression/classification metrics.
- Training recommended baseline models based on the results of a full EDA.

All computations use the existing configuration for reproducibility and
never recompute statistics or recommendations on their own. The module
depends only on the official public APIs of `preml` (config, schema,
exceptions) and scikit‑learn.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline

from preml.config import MLToolkitConfig, default_config
from preml.exceptions import ModelError
from preml.schema import ModelRecommendation, TargetProfile

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_REGRESSION_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
    "mae": mean_absolute_error,
    "r2": r2_score,
}

_CLASSIFICATION_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy_score,
    "precision": lambda yt, yp: precision_score(
        yt, yp, average="weighted", zero_division=0
    ),
    "recall": lambda yt, yp: recall_score(
        yt, yp, average="weighted", zero_division=0
    ),
    "f1": lambda yt, yp: f1_score(
        yt, yp, average="weighted", zero_division=0
    ),
}


def _get_task_type(target_profile: TargetProfile) -> str:
    """Determine task string from a TargetProfile."""
    if target_profile.is_regression:
        return "regression"
    elif target_profile.is_binary:
        return "binary_classification"
    return "multiclass_classification"


# ---------------------------------------------------------------------------
# Public API: Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    extra_metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
) -> Dict[str, float]:
    """Compute standard evaluation metrics for a single train/test split.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    task_type : str
        One of ``'regression'``, ``'classification'``,
        ``'binary_classification'``, or ``'multiclass_classification'``.
    extra_metrics : dict, optional
        Additional metric functions keyed by name. Each callable must
        accept ``(y_true, y_pred)`` and return a float.

    Returns
    -------
    dict
        Mapping of metric name → score.

    Raises
    ------
    ModelError
        If *task_type* is unsupported.
    """
    if task_type == "regression":
        metric_funcs = dict(_REGRESSION_METRICS)
    elif task_type in ("classification", "binary_classification", "multiclass_classification"):
        metric_funcs = dict(_CLASSIFICATION_METRICS)
    else:
        raise ModelError(f"Unsupported task type: '{task_type}'.")

    results = {name: func(y_true, y_pred) for name, func in metric_funcs.items()}
    if extra_metrics is not None:
        results.update({name: func(y_true, y_pred) for name, func in extra_metrics.items()})
    return results


# ---------------------------------------------------------------------------
# Public API: Cross‑validation
# ---------------------------------------------------------------------------

def cross_validate(
    model: BaseEstimator,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: int = 5,
    scoring: Union[str, List[str]] = "r2",
    random_state: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Run k‑fold cross‑validation on a (preprocessing + estimator) pipeline.

    Parameters
    ----------
    model : BaseEstimator
        A scikit‑learn compatible estimator (can be a ``Pipeline``).
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target vector.
    cv : int, default 5
        Number of folds.
    scoring : str or list of str
        Scoring metric(s) compatible with
        :func:`sklearn.model_selection.cross_validate`.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    dict
        Mapping ``metric_name`` → list of fold scores.
    """
    cv_results = sk_cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
        error_score="raise",
    )
    # Extract test‑ scores and drop the 'test_' prefix
    scores: Dict[str, List[float]] = {}
    for key, values in cv_results.items():
        if key.startswith("test_"):
            metric_name = key[len("test_"):]
            scores[metric_name] = values.tolist()
    return scores


# ---------------------------------------------------------------------------
# Public API: BaselineTrainer
# ---------------------------------------------------------------------------

class BaselineTrainer:
    """Trains and evaluates simple baseline models.

    The trainer uses an already‑built ``ColumnTransformer`` (the output
    of :class:`~preml.preprocessing.PreprocessingBuilder`) and a
    target profile to create a full machine‑learning pipeline. It can
    either accept an explicit estimator or infer sensible defaults from
    the task type.

    Parameters
    ----------
    config : MLToolkitConfig, optional
        Configuration object; used for the random state in estimators.
    """

    def __init__(self, config: Optional[MLToolkitConfig] = None) -> None:
        self.config = config or default_config

    # ------------------------------------------------------------------
    # Estimator selection
    # ------------------------------------------------------------------
    def _get_default_estimator(self, task_type: str) -> BaseEstimator:
        """Return a sensible default estimator for the given task."""
        if task_type == "regression":
            return LinearRegression()
        elif task_type in ("classification", "binary_classification", "multiclass_classification"):
            return LogisticRegression(
                max_iter=1000, random_state=self.config.random_state
            )
        else:
            raise ModelError(f"Unsupported task type: '{task_type}'.")

    def _estimator_from_recommendation(
        self, rec: ModelRecommendation, task_type: str
    ) -> BaseEstimator:
        """Map a ``ModelRecommendation`` to a scikit‑learn estimator.

        Parameters
        ----------
        rec : ModelRecommendation
            A model recommendation from the EDA.
        task_type : str
            Task type string.

        Returns
        -------
        BaseEstimator
            Instantiated estimator.

        Raises
        ------
        ModelError
            If the model name cannot be mapped.
        """
        name_lower = rec.model_name.lower().replace(" ", "")
        rs = self.config.random_state

        if task_type == "regression":
            if "linearregression" in name_lower:
                return LinearRegression()
            elif "randomforest" in name_lower:
                return RandomForestRegressor(random_state=rs)
            else:
                raise ModelError(
                    f"No estimator mapping for '{rec.model_name}' in regression."
                )
        else:  # classification
            if "logisticregression" in name_lower:
                return LogisticRegression(max_iter=1000, random_state=rs)
            elif "randomforest" in name_lower:
                return RandomForestClassifier(random_state=rs)
            else:
                raise ModelError(
                    f"No estimator mapping for '{rec.model_name}' in classification."
                )

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------
    def build_model_pipeline(
        self,
        preprocessing_pipeline: ColumnTransformer,
        task_type: str,
        estimator: Optional[BaseEstimator] = None,
    ) -> Pipeline:
        """Attach an estimator to a preprocessing pipeline.

        Parameters
        ----------
        preprocessing_pipeline : ColumnTransformer
            The preprocessing pipeline returned by
            :meth:`PreprocessingBuilder.build_pipeline()
            <preml.preprocessing.PreprocessingBuilder.build_pipeline>`.
        task_type : str
            One of ``'regression'``, ``'classification'``,
            ``'binary_classification'``, ``'multiclass_classification'``.
        estimator : BaseEstimator, optional
            If ``None``, a sensible default is chosen based on *task_type*.

        Returns
        -------
        Pipeline
            A scikit‑learn ``Pipeline`` with steps ``('preprocessor', ...)``
            and ``('estimator', ...)``.
        """
        if estimator is None:
            estimator = self._get_default_estimator(task_type)
        return Pipeline(
            steps=[
                ("preprocessor", preprocessing_pipeline),
                ("estimator", clone(estimator)),
            ]
        )

    # ------------------------------------------------------------------
    # Single model evaluation
    # ------------------------------------------------------------------
    def evaluate_baseline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv: int = 5,
        scoring: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Fit and cross‑validate a single baseline pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            Full pipeline (preprocessor + estimator).
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        task_type : str
            Task type string.
        cv : int, default 5
            Number of cross‑validation folds.
        scoring : str or list of str, optional
            Scikit‑learn scoring string(s). If ``None``, sensible defaults
            are chosen: ``'neg_mean_squared_error'`` and ``'r2'`` for
            regression, ``'accuracy'`` for classification.

        Returns
        -------
        dict
            Contains:
            - ``'cv_scores'`` : dict of metric → list of fold scores
            - ``'mean_scores'`` : dict of metric → mean over folds
            - ``'std_scores'`` : dict of metric → standard deviation
            - ``'pipeline'`` : the pipeline fitted on the *full* dataset
        """
        if scoring is None:
            scoring = (
                ["neg_mean_squared_error", "r2"]
                if task_type == "regression"
                else ["accuracy"]
            )

        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring,
            random_state=self.config.random_state,
        )
        mean_scores = {k: float(np.mean(v)) for k, v in cv_results.items()}
        std_scores = {k: float(np.std(v)) for k, v in cv_results.items()}

        # Fit on full data so the user can inspect the final model
        pipeline.fit(X, y)
        return {
            "cv_scores": cv_results,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "pipeline": pipeline,
        }

    # ------------------------------------------------------------------
    # Batch training of recommended baselines
    # ------------------------------------------------------------------
    def train_baselines(
        self,
        analysis_result: Dict[str, Any],
        df: pd.DataFrame,
        target_col: str,
        preprocessing_pipeline: ColumnTransformer,
        cv: int = 5,
    ) -> List[Dict[str, Any]]:
        """Train all recommended baseline models using a shared preprocessor.

        Iterates over the model recommendations contained in
        ``analysis_result['recommendations']['models']``, builds a
        pipeline for each, cross‑validates, and collects the results.

        If no models are recommended, a single default estimator is
        trained (LinearRegression for regression, LogisticRegression
        for classification).

        Parameters
        ----------
        analysis_result : dict
            The full output of :meth:`EDAAnalyzer.run()
            <preml.eda.EDAAnalyzer.run>`.
        df : pd.DataFrame
            Original DataFrame containing the feature columns and the
            target column.
        target_col : str
            Name of the target column.
        preprocessing_pipeline : ColumnTransformer
            Preprocessing pipeline (without estimator) as returned by
            :meth:`PreprocessingBuilder.build_pipeline()
            <preml.preprocessing.PreprocessingBuilder.build_pipeline>`.
        cv : int, default 5
            Number of cross‑validation folds.

        Returns
        -------
        list of dict
            Each dict contains:
            - ``'model_name'`` : str
            - ``'cv_scores'``, ``'mean_scores'``, ``'std_scores'``
            - ``'pipeline'`` : fitted Pipeline

        Raises
        ------
        ModelError
            If no target profile is present in the analysis.
        """
        target_profile = analysis_result.get("target_profile")
        if not target_profile:
            raise ModelError(
                "No target profile found in analysis. "
                "Make sure `target` was specified during EDA."
            )
        task_type = _get_task_type(target_profile)

        # Retrieve model recommendations
        recs = analysis_result.get("recommendations", {}).get("models", [])
        if not recs:
            # Fallback to a single default
            recs = [
                ModelRecommendation(
                    model_name="LinearRegression"
                    if task_type == "regression"
                    else "LogisticRegression",
                    suitability="baseline",
                )
            ]

        X = df.drop(columns=[target_col])
        y = df[target_col]

        results: List[Dict[str, Any]] = []
        for rec in recs:
            # rec is a ModelRecommendation instance
            estimator = self._estimator_from_recommendation(rec, task_type)
            pipeline = self.build_model_pipeline(
                preprocessing_pipeline, task_type, estimator
            )
            eval_out = self.evaluate_baseline(
                pipeline, X, y, task_type, cv=cv
            )
            results.append({"model_name": rec.model_name, **eval_out})
        return results
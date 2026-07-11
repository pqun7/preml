from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from preml.recommendation_engine import (
    EvaluationResult,
    KnowledgeBase,
    ModelCandidate,
    RecommendationEngine,
    ValidationTimeoutError,
)


@pytest.fixture
def mixed_dataset():
    X = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "cat": ["A", "B", "A", "B", "A", "B"],
        }
    )
    y_reg = pd.Series([1.0, 1.8, 3.1, 4.0, 5.2, 5.8])
    y_cls = pd.Series([0, 1, 0, 1, 0, 1])
    return X, y_reg, y_cls


def _patch_fast_fit(monkeypatch, engine: RecommendationEngine, candidate: ModelCandidate) -> None:
    monkeypatch.setattr(engine, "_generate_candidates", lambda: [candidate])
    monkeypatch.setattr(
        engine,
        "_fast_cv_selector",
        lambda X, y, candidates, time_budget: [
            EvaluationResult(
                model=candidate,
                cv_score=0.9,
                cv_std=0.01,
                training_time=0.0,
                n_folds_completed=5,
                extrapolated_score=0.9,
            )
        ],
    )
    monkeypatch.setattr(
        engine,
        "_lccv_evaluate",
        lambda X, y, cand: EvaluationResult(
            model=cand,
            cv_score=0.92,
            cv_std=0.0,
            training_time=0.0,
            n_folds_completed=3,
            extrapolated_score=0.92,
        ),
    )
    monkeypatch.setattr(engine, "_successive_halving_optimize", lambda *args, **kwargs: {})
    monkeypatch.setattr(engine, "_bayesian_optimization_finetune", lambda *args, **kwargs: {})


@pytest.mark.parametrize(
    "candidate_class,is_regression",
    [
        (HistGradientBoostingRegressor, True),
        (HistGradientBoostingClassifier, False),
    ],
)
def test_fit_handles_missing_and_categorical_data(monkeypatch, mixed_dataset, candidate_class, is_regression):
    X, y_reg, y_cls = mixed_dataset
    y = y_reg if is_regression else y_cls
    engine = RecommendationEngine(random_state=42)
    candidate = ModelCandidate(
        name=candidate_class.__name__,
        estimator_class=candidate_class,
        priority=1.0,
        hyperparams={"random_state": 42},
        supports_categorical=True,
        supports_missing=True,
        needs_scaling=False,
    )
    _patch_fast_fit(monkeypatch, engine, candidate)

    result = engine.fit(X, y, time_budget_seconds=5.0)

    assert result["best_model"] == candidate_class.__name__
    assert result["pipeline"] is not None
    assert result["cv_score"] is not None
    assert result["hyperparams"] == {}


def test_fit_timeout_raises(monkeypatch, mixed_dataset):
    X, y_reg, _ = mixed_dataset
    engine = RecommendationEngine(random_state=42)
    candidate = ModelCandidate(
        name="HistGradientBoostingRegressor",
        estimator_class=HistGradientBoostingRegressor,
        priority=1.0,
        hyperparams={"random_state": 42},
        supports_categorical=True,
        supports_missing=True,
        needs_scaling=False,
    )
    _patch_fast_fit(monkeypatch, engine, candidate)

    with pytest.raises(ValidationTimeoutError):
        engine.fit(X, y_reg, time_budget_seconds=0.0)


def test_knowledge_base_store_and_query(tmp_path):
    db_path = tmp_path / "knowledge.db"
    kb = KnowledgeBase(str(db_path))
    dataset_hash = "abc123"
    meta = {
        "n_samples": 100,
        "n_features": 5,
        "n_numeric": 4,
        "n_categorical": 1,
        "missing_ratio": 0.1,
        "skewness_mean": 0.3,
        "target_variance": 1.2,
        "signal_to_noise_ratio": 0.8,
    }
    kb.store_meta_features(dataset_hash, meta)
    kb.store_model_performance(
        dataset_hash,
        "HistGradientBoostingRegressor",
        "excellent",
        0.94,
        {"random_state": 42},
    )

    rows = kb.query_similar_datasets(meta, top_k=1)

    assert rows
    assert rows[0][0] == dataset_hash
    assert rows[0][2][0]["model_name"] == "HistGradientBoostingRegressor"


def test_generate_recommendations_supports_missing_and_categorical_data(mixed_dataset):
    X, _, y_cls = mixed_dataset
    engine = RecommendationEngine(random_state=42)
    engine.extract_meta_features(X, y_cls)
    result = engine.generate_recommendations(
        {
            "metadata": None,
            "duplicates": type("D", (), {"total_duplicates": 0})(),
            "infinite": type("I", (), {})(),
            "missing": type("M", (), {"total_missing": 1, "column_reports": [type("R", (), {"column": "num", "missing_count": 1})()]})(),
            "outliers": [],
            "feature_profiles": [],
            "correlation_pairs": [],
            "target_profile": type("T", (), {"is_regression": False, "is_binary": True})(),
        }
    )

    assert "pipeline" in result
    assert "models" in result
    assert result["data_quality_notes"]

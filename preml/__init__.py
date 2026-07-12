# Copyright (c) 2026 Ali Nazer
# Licensed under the MIT License.
# See the LICENSE file in the project root for license information.
"""
preml — A professional machine learning toolkit for EDA,
preprocessing, feature engineering, and modeling.

The package is organized into clearly separated modules, each
responsible for a single domain concern:

- config          : Central configuration and thresholds.
- exceptions      : Custom exception hierarchy.
- schema          : Strongly‑typed data models (dataclasses).
- statistics_engine: Extracts statistical facts from data.
- recommendation_engine: Generates evidence‑based recommendations.
- eda             : Orchestrates analysis and produces insights.
- visualization   : Creates plots (statistics‑free).
- preprocessing   : Builds sklearn‑compatible pipelines.
- feature_engineering: Suggests and creates new features.
- model_utils     : Baseline models, cross‑validation, metrics.
- report          : Generates reports in various formats.
"""

__version__ = "0.1.8"
__author__ = "Ali Nazer <alinazer30@gmail.com>"

# Expose the most commonly used classes and convenience functions
from preml.config import MLToolkitConfig, default_config
from preml.exceptions import (
    MLToolkitError,
    DataValidationError,
    StatisticsError,
    RecommendationError,
    PreprocessingError,
    FeatureEngineeringError,
    ModelError,
    ReportError,
    VisualizationError,
)
from preml.schema import (
    DatasetMetadata,
    DuplicateReport,
    InfiniteReport,
    MissingColumnReport,
    MissingReport,
    OutlierReport,
    NumericDistributionProfile,
    CategoricalProfile,
    FeatureProfile,
    CorrelationPair,
    TargetProfile,
    Evidence,
    Recommendation,
    PipelineSuggestion,
    ModelRecommendation,
)
from preml.facade import (
    PreML,
    analyze,
    feature_engineering,
    models,
    pipeline,
    quick_eda,
    recommendations,
    report,
    visualize,
)

__all__ = [
    # Facade
    "PreML",
    "analyze",
    "recommendations",
    "report",
    "pipeline",
    "visualize",
    "feature_engineering",
    "models",
    # Configuration
    "MLToolkitConfig",
    "default_config",
    # Exceptions
    "MLToolkitError",
    "DataValidationError",
    "StatisticsError",
    "RecommendationError",
    "PreprocessingError",
    "FeatureEngineeringError",
    "ModelError",
    "ReportError",
    "VisualizationError",
    # Schema
    "DatasetMetadata",
    "DuplicateReport",
    "InfiniteReport",
    "MissingColumnReport",
    "MissingReport",
    "OutlierReport",
    "NumericDistributionProfile",
    "CategoricalProfile",
    "FeatureProfile",
    "CorrelationPair",
    "TargetProfile",
    "Evidence",
    "Recommendation",
    "PipelineSuggestion",
    "ModelRecommendation",
    # Convenience
    "quick_eda",
]
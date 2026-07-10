"""
Custom exception classes for ml_toolkit.

All exceptions inherit from MLToolkitError, so users can catch a single
base exception if needed. Each exception conveys clear, actionable error
messages and supports optional payloads for debugging.

Design decisions:
- One base class and specialized subclasses for common error categories.
- All exceptions are picklable (rely on plain Python attributes).
- No heavy dependencies; pure standard library.
- Exception names follow the naming pattern <Category>Error.
- Optional `details` attribute to attach extra debugging information.
"""

from typing import Any, Optional


class MLToolkitError(Exception):
    """Base exception for all ml_toolkit errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    details : Any, optional
        Additional debugging information (e.g., invalid value, column name).
    """

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details is not None:
            return f"{self.message} [Details: {self.details}]"
        return self.message

    def __repr__(self) -> str:
        """Return a developer‑friendly representation for debugging."""
        cls_name = type(self).__name__
        if self.details is None:
            return f"{cls_name}({self.message!r})"
        return f"{cls_name}({self.message!r}, details={self.details!r})"


class DataValidationError(MLToolkitError):
    """Raised when input data fails validation.

    Examples:
        - DataFrame is empty.
        - Column type mismatch.
        - Invalid configuration.
    """
    pass


class StatisticsError(MLToolkitError):
    """Raised when statistical computation fails.

    Examples:
        - Attempt to compute correlation on all‑null columns.
        - Skewness on constant column.
    """
    pass


class RecommendationError(MLToolkitError):
    """Raised when the recommendation engine encounters a problem.

    Examples:
        - No numeric features to recommend preprocessing for.
        - Incompatible configuration settings.
    """
    pass


class PreprocessingError(MLToolkitError):
    """Raised during pipeline generation or data transformation.

    Examples:
        - Invalid column names provided for encoding.
        - Missing required parameters.
    """
    pass


class FeatureEngineeringError(MLToolkitError):
    """Raised when feature engineering cannot proceed.

    Examples:
        - No eligible numeric pairs for ratio creation.
        - Not enough features to compute interactions.
    """
    pass


class ModelError(MLToolkitError):
    """Raised for model‑related issues.

    Examples:
        - Unsupported metric for regression/classification.
        - Incompatible model type.
    """
    pass


class ReportError(MLToolkitError):
    """Raised when report generation fails.

    Examples:
        - Invalid output format requested.
        - Missing required statistics.
    """
    pass


class VisualizationError(MLToolkitError):
    """Raised when visualization cannot be created.

    Examples:
        - Empty data for plot.
        - Invalid figure size.
    """
    pass
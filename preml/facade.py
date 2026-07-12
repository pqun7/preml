"""High-level public facade for PreML.

This module provides the user-facing entry point for the library while
preserving the existing internal orchestration layers. The facade caches
analysis lazily so expensive statistics are computed once and reused by
downstream workflows such as recommendations, preprocessing, reports,
visualizations, and feature engineering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import warnings

import pandas as pd

from preml.config import MLToolkitConfig, default_config
from preml.eda import EDAAnalyzer
from preml.feature_engineering import FeatureEngineering
from preml.preprocessing import PreprocessingBuilder
from preml.recommendation_engine import RecommendationEngine
from preml.report import ReportGenerator
from preml.statistics_engine import StatisticsEngine
from preml.visualization import (
    explain_visualizations,
    plot_correlation_heatmap,
    plot_missing_heatmap,
    plot_numeric_distributions,
    plot_outlier_summary,
    plot_target_correlations,
    plot_target_distribution,
)


class PreML:
    """Facade object for tabular analysis workflows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str, optional
        Target column name for supervised workflows.
    config : MLToolkitConfig, optional
        Configuration object. Defaults to :data:`preml.config.default_config`.
    enable_feature_engineering : bool, default True
        Whether recommendation workflows should include feature-engineering
        suggestions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[MLToolkitConfig] = None,
        enable_feature_engineering: bool = True,
    ) -> None:
        self.df = df
        self.target = target
        self.config = config or default_config
        self.enable_feature_engineering = enable_feature_engineering
        self._analyzer = EDAAnalyzer(
            df,
            target=target,
            config=self.config,
            enable_feature_engineering=enable_feature_engineering,
        )
        self._analysis_result: Optional[Dict[str, Any]] = None

    def analyze(self) -> Dict[str, Any]:
        """Run EDA once and cache the result."""
        if self._analysis_result is None:
            result = self._analyzer.run()
            self._analysis_result = cast(Dict[str, Any], result)
            self._analyzer._analysis_result = result
        return cast(Dict[str, Any], self._analysis_result)

    def run(self) -> Dict[str, Any]:
        """Backward-compatible alias for :meth:`analyze`."""
        warnings.warn(
            "PreML.run() is deprecated; use PreML.analyze() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.analyze()

    @property
    def analysis(self) -> Dict[str, Any]:
        """Cached analysis result.

        Accessing this property triggers analysis on first use.
        """
        return self.analyze()

    def summary(self) -> str:
        """Return the plain-text analysis summary."""
        self.analyze()
        return self._analyzer.summary()

    def statistics(self) -> StatisticsEngine:
        """Create a statistics engine for advanced workflows."""
        return StatisticsEngine(self.df, target=self.target, config=self.config)

    def recommendations(self) -> Dict[str, Any]:
        """Return heuristic recommendations derived from cached analysis."""
        engine = RecommendationEngine(
            config=self.config,
            enable_feature_engineering=self.enable_feature_engineering,
        )
        return engine.generate_recommendations(self.analyze())

    def generate_recommendations(self) -> Dict[str, Any]:
        """Backward-compatible alias for :meth:`recommendations`."""
        warnings.warn(
            "PreML.generate_recommendations() is deprecated; use PreML.recommendations() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.recommendations()

    def pipeline(self) -> PreprocessingBuilder:
        """Return a preprocessing builder configured from cached analysis."""
        return PreprocessingBuilder(self.analyze(), config=self.config)

    def build_pipeline(self) -> PreprocessingBuilder:
        """Backward-compatible alias for :meth:`pipeline`."""
        warnings.warn(
            "PreML.build_pipeline() is deprecated; use PreML.pipeline() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.pipeline()

    def feature_engineering(self):
        """Return feature-engineering suggestions from cached analysis."""
        return FeatureEngineering(self.analyze(), df=self.df, config=self.config).suggest_features()

    def report(
        self,
        format: str = "text",
        embed_plots: bool = False,
        df: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Generate a report from cached analysis.

        Parameters
        ----------
        format : str, default "text"
            One of ``"text"``, ``"markdown"``, or ``"html"``.
        embed_plots : bool, default False
            Forwarded to HTML report generation.
        df : pandas.DataFrame, optional
            DataFrame used for plot rendering. Defaults to the input data.
        """
        report_df = df if df is not None else self.df
        generator = ReportGenerator(self.analyze(), df=report_df, config=self.config)
        if format == "text":
            return generator.generate_text()
        if format == "markdown":
            return generator.generate_markdown()
        if format == "html":
            return generator.generate_html(embed_plots=embed_plots)
        raise ValueError("format must be one of 'text', 'markdown', or 'html'.")

    def visualize(self, kind: str = "all", df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate one or more visualizations from cached analysis.

        The return value is a dictionary keyed by plot name. Missing plots are
        omitted.
        """
        data = df if df is not None else self.df
        analysis = self.analyze()
        figures: Dict[str, Any] = {}

        if kind in ("all", "numeric"):
            figures["numeric_distributions"] = plot_numeric_distributions(
                data, analysis, config=self.config
            )
        if kind in ("all", "missing"):
            figures["missing_heatmap"] = plot_missing_heatmap(
                data, config=self.config
            )
        if kind in ("all", "outliers"):
            figures["outlier_summary"] = plot_outlier_summary(
                analysis, config=self.config
            )
        if kind in ("all", "target"):
            figures["target_distribution"] = plot_target_distribution(
                data, analysis, config=self.config
            )
        if kind in ("all", "correlations"):
            figures["correlation_heatmap"] = plot_correlation_heatmap(
                data, analysis, config=self.config
            )
            figures["target_correlations"] = plot_target_correlations(
                data, analysis, config=self.config
            )
        if kind in ("all", "explanations"):
            figures["explanations"] = explain_visualizations(
                analysis, recommendations=self.recommendations()
            )
        return figures

    def models(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Return model recommendations or empirical model selection results.

        If ``X`` and ``y`` are omitted, the facade uses the configured target
        column from the stored DataFrame when available.
        """
        engine = RecommendationEngine(
            config=self.config,
            enable_feature_engineering=self.enable_feature_engineering,
        )

        if X is None or y is None:
            if self.target is None:
                raise ValueError(
                    "models() requires X and y, or a target column passed to PreML."
                )
            X = self.df.drop(columns=[self.target])
            y = self.df[self.target]

        return engine.get_recommendation(X, y)

    def quick_eda(self) -> Dict[str, Any]:
        """Backward-compatible alias for one-line analysis."""
        return self.analyze()


def analyze(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
    enable_feature_engineering: bool = True,
) -> Dict[str, Any]:
    """One-line convenience wrapper around :class:`PreML`."""
    return PreML(
        df,
        target=target,
        config=config,
        enable_feature_engineering=enable_feature_engineering,
    ).analyze()


def recommendations(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
    enable_feature_engineering: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper for recommendation generation."""
    return PreML(
        df,
        target=target,
        config=config,
        enable_feature_engineering=enable_feature_engineering,
    ).recommendations()


def report(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
    format: str = "text",
    embed_plots: bool = False,
) -> Any:
    """Convenience wrapper for report generation."""
    return PreML(
        df,
        target=target,
        config=config,
    ).report(format=format, embed_plots=embed_plots)


def pipeline(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
) -> PreprocessingBuilder:
    """Convenience wrapper for preprocessing pipeline construction."""
    return PreML(df, target=target, config=config).pipeline()


def visualize(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
    kind: str = "all",
) -> Dict[str, Any]:
    """Convenience wrapper for visualization generation."""
    return PreML(df, target=target, config=config).visualize(kind=kind)


def feature_engineering(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
) -> Any:
    """Convenience wrapper for feature-engineering suggestions."""
    return PreML(df, target=target, config=config).feature_engineering()


def models(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for model recommendations."""
    return PreML(df, target=target, config=config).models(X=X, y=y)


def quick_eda(
    df: pd.DataFrame,
    target: Optional[str] = None,
    config: Optional[MLToolkitConfig] = None,
    enable_feature_engineering: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible helper mirroring the legacy convenience API."""
    return analyze(
        df,
        target=target,
        config=config,
        enable_feature_engineering=enable_feature_engineering,
    )


__all__ = [
    "PreML",
    "analyze",
    "quick_eda",
    "recommendations",
    "report",
    "pipeline",
    "visualize",
    "feature_engineering",
    "models",
]
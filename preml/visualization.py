"""
Visualization layer — creates informative plots from pre‑computed
statistical facts and the original data.

This module NEVER computes statistics; it uses the supplied DataFrame
only for plotting raw values.  All plotting functions accept an
optional `ax` for composability and return the figure.

A new utility function `explain_visualizations` provides
human-readable explanations of each plot type, based on the actual
data characteristics, and integrates recommendations from the
RecommendationEngine to suggest concrete next steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preml.config import MLToolkitConfig, default_config
from preml.schema import (
    CorrelationPair,
    FeatureProfile,
    OutlierReport,
    TargetProfile,
)

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _get_profiles(analysis_result: Dict[str, Any]) -> List[FeatureProfile]:
    return analysis_result.get("feature_profiles", [])


def _get_outliers(analysis_result: Dict[str, Any]) -> List[OutlierReport]:
    return analysis_result.get("outliers", [])


def _get_correlations(analysis_result: Dict[str, Any]) -> List[CorrelationPair]:
    return analysis_result.get("correlation_pairs", [])


def _get_target_profile(analysis_result: Dict[str, Any]) -> Optional[TargetProfile]:
    return analysis_result.get("target_profile")


def _get_missing_columns(df: pd.DataFrame, max_cols: int = 50) -> List[str]:
    """Return up to ``max_cols`` columns that contain missing values.

    Columns are ordered by descending missing count so the heatmap shows the
    most informative subset when the dataset has many sparse columns.
    """
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    return missing_counts.index[:max_cols].tolist()


def _apply_style(cfg: MLToolkitConfig) -> None:
    """Set global Seaborn style and palette from config with safe defaults."""
    style = getattr(cfg, "plot_style", "whitegrid")
    palette = getattr(cfg, "color_palette", "muted")
    sns.set_style(style)
    sns.set_palette(palette)


def _safe_figsize(cfg: MLToolkitConfig, default_size=(10, 6)) -> tuple:
    """Return a valid figure size tuple, falling back to `default_size`."""
    sz = getattr(cfg, "figure_size", default_size)
    if isinstance(sz, (tuple, list)) and len(sz) == 2:
        return tuple(sz)
    return default_size


# ------------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------------
def plot_numeric_distributions(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    max_cols: int = 20,
    show_outlier_lines: bool = True,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Combined histogram + boxplot for numeric features.

    The histogram shows the distribution shape (skewness, peaks) and
    overlays the mean, median, and optional IQR outlier boundaries.
    The boxplot summarizes quartiles and potential outliers.

    Interpretation: Skewed distributions often benefit from
    transformations (log, Yeo-Johnson).  Outlier boundaries help
    decide whether to winsorize or cap extreme values.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame (must contain the numeric columns).
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    max_cols : int
        Maximum number of numeric columns to plot.
    show_outlier_lines : bool
        If True, draw IQR outlier bounds as dashed lines on the histogram.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    profiles = _get_profiles(analysis_result)
    outliers = _get_outliers(analysis_result)
    numeric_profiles = [
        p for p in profiles if p.numeric_profile and not p.is_constant
    ]
    if not numeric_profiles:
        return None

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    profs_to_plot = numeric_profiles[:max_cols]
    n = len(profs_to_plot)
    fig, axes = plt.subplots(
        n, 2, figsize=(figsize[0] * 1.2, 4 * n), squeeze=False
    )
    fig.suptitle("Numeric Feature Distributions", fontsize=16)

    outlier_dict = {o.column: o for o in outliers}

    for i, prof in enumerate(profs_to_plot):
        col = prof.column
        num = prof.numeric_profile  # guaranteed not None
        data = df[col].dropna()

        # Histogram + KDE
        ax_hist = axes[i, 0]
        sns.histplot(data, kde=True, ax=ax_hist, color="steelblue",
                     edgecolor="white")
        if num:
            ax_hist.axvline(num.mean, color="red", linestyle="--",
                            label=f"Mean={num.mean:.2f}")
            ax_hist.axvline(num.median, color="green", linestyle="-",
                            label=f"Median={num.median:.2f}")
        if show_outlier_lines and col in outlier_dict:
            o = outlier_dict[col]
            if o.lower_bound is not None:
                ax_hist.axvline(o.lower_bound, color="orange",
                                linestyle=":", label="IQR lower")
            if o.upper_bound is not None:
                ax_hist.axvline(o.upper_bound, color="orange",
                                linestyle=":", label="IQR upper")
        ax_hist.set_title(f"{col} (skew={num.skewness:.2f})")
        ax_hist.legend(loc="upper right")

        # Boxplot
        ax_box = axes[i, 1]
        sns.boxplot(x=data, ax=ax_box, color="lightblue")
        ax_box.set_title(f"{col} boxplot")
        if num:
            ax_box.set_xlabel(f"Min={num.min:.2f}, Max={num.max:.2f}")

    plt.tight_layout()
    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Plot the distribution of the target variable.

    For regression targets, a histogram + KDE and a boxplot are shown.
    For classification targets, a bar chart of class frequencies is
    displayed.

    Interpretation: Highly imbalanced classes or extreme skew in a
    regression target can guide preprocessing (e.g., stratification,
    transformation).

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame (must contain the target column).
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    target_profile = _get_target_profile(analysis_result)
    if target_profile is None:
        return None

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    target_col = target_profile.column
    data = df[target_col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0], 5))
    if target_profile.is_regression:
        sns.histplot(data, kde=True, ax=axes[0], color="teal")
        axes[0].set_title(f"Target distribution: {target_col}")
        sns.boxplot(x=data, ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Target boxplot: {target_col}")
    else:
        value_counts = data.value_counts()
        axes[0].bar(value_counts.index.astype(str), value_counts.values,
                    color="salmon")
        axes[0].set_title(f"Target classes: {target_col}")
        axes[0].set_ylabel("Count")
        axes[1].axis("off")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Correlation plots
# ------------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Plot a Pearson correlation heatmap for numeric features.

    Only columns that appear in correlation pairs above the configured
    threshold are shown.  The heatmap uses a coolwarm colormap centered
    at 0.

    Interpretation: Dark red/blue cells indicate strong positive or
    negative linear relationships.  High multicollinearity can
    destabilize linear models – consider dropping one of each highly
    correlated pair or using regularization.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    correlation_pairs = _get_correlations(analysis_result)
    if not correlation_pairs:
        return None

    cols_in_corr = set()
    for pair in correlation_pairs:
        cols_in_corr.add(pair.feature_a)
        cols_in_corr.add(pair.feature_b)
    numeric_cols = [c for c in cols_in_corr if c in df.columns]
    if len(numeric_cols) < 2:
        return None

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return fig


def plot_top_correlations_bar(
    analysis_result: Dict[str, Any],
    top_n: int = 10,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Bar chart of the top absolute correlations.

    Interpretation: The tallest bars show feature pairs with the
    strongest linear relationship.  Investigate these for potential
    redundancy or interaction effects.

    Parameters
    ----------
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    top_n : int
        Number of pairs to display.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    pairs = _get_correlations(analysis_result)
    if not pairs:
        return None

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    sorted_pairs = sorted(pairs, key=lambda x: abs(x.coefficient), reverse=True)
    top_pairs = sorted_pairs[:top_n]

    labels = [f"{p.feature_a}\nvs {p.feature_b}" for p in top_pairs]
    values = [abs(p.coefficient) for p in top_pairs]

    fig, ax = plt.subplots(figsize=(figsize[0], 0.5 * len(labels)))
    ax.barh(range(len(labels)), values, color="purple", edgecolor="black")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute Pearson Correlation")
    ax.set_title("Top Feature Correlations")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Missing values heatmap
# ------------------------------------------------------------------
def plot_missing_heatmap(
    df: pd.DataFrame,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Heatmap showing missing values across columns (yellow = missing).

    If the dataset has many rows, a random sample of up to 5000 rows
    is displayed for performance.

    Interpretation: Dense yellow blocks indicate columns with high
    missingness.  These may need imputation or removal.  Patterns
    across rows can hint at systematic missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    cols = _get_missing_columns(df)
    if not cols:
        return None

    missing_data = df[cols].isnull()

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    if len(df) > 5000:
        idx = np.random.choice(len(df), 5000, replace=False)
        missing_data = missing_data.iloc[idx]

    fig, ax = plt.subplots(figsize=(figsize[0], max(0.5 * len(cols), 2)))
    sns.heatmap(missing_data.T, cmap=["#ffffff", "#f1c40f"],
                cbar=False, ax=ax, xticklabels=False)
    ax.set_xlabel("Rows (sample)")
    ax.set_ylabel("Columns")
    ax.set_title("Missing Values Heatmap (yellow = missing)")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Outlier summary
# ------------------------------------------------------------------
def plot_outlier_summary(
    analysis_result: Dict[str, Any],
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Horizontal bar chart showing outlier percentages per numeric column.

    Interpretation: Columns with a high outlier percentage may contain
    erroneous data or genuine extreme values.  Investigate the source
    and consider winsorization, capping, or robust scaling.

    Parameters
    ----------
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    outliers = _get_outliers(analysis_result)
    if not outliers:
        return None

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    non_zero = [o for o in outliers if o.outlier_count > 0]
    if not non_zero:
        return None

    labels = [o.column for o in non_zero]
    percentages = [o.outlier_percent for o in non_zero]

    fig, ax = plt.subplots(figsize=(figsize[0], 0.4 * len(labels)))
    ax.barh(range(len(labels)), percentages, color="coral", edgecolor="black")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Outlier Percentage (%)")
    ax.set_title("Outlier Percentage per Numeric Feature (IQR)")
    ax.invert_yaxis()
    for i, p in enumerate(percentages):
        ax.text(p + 0.5, i, f"{p:.1f}%", va="center")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Target correlations bar
# ------------------------------------------------------------------
def plot_target_correlations(
    df: pd.DataFrame,
    analysis_result: Dict[str, Any],
    top_n: int = 15,
    config: Optional[MLToolkitConfig] = None,
) -> Optional[plt.Figure]:
    """Bar chart of Pearson correlations between features and a numeric target.

    Positive correlations are shown in teal, negative in coral.
    Interpretation: The strongest predictors (by linear correlation)
    are shown.  Non‑linear relationships will not be captured; consider
    also mutual information or tree‑based feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    top_n : int
        Number of most correlated features to show.
    config : MLToolkitConfig, optional

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    target_profile = _get_target_profile(analysis_result)
    if not target_profile or not target_profile.is_regression:
        return None

    target_col = target_profile.column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols or len(numeric_cols) < 2:
        return None

    cfg = config or default_config
    _apply_style(cfg)
    figsize = _safe_figsize(cfg, (10, 6))

    corrs = df[numeric_cols].corrwith(df[target_col]).drop(target_col)
    corrs_sorted = corrs.abs().sort_values(ascending=False).head(top_n)
    signed_corrs = corrs[corrs_sorted.index]

    fig, ax = plt.subplots(figsize=(figsize[0], 0.5 * len(signed_corrs)))
    colors = ["teal" if c >= 0 else "coral" for c in signed_corrs]
    ax.barh(range(len(signed_corrs)), signed_corrs.values, color=colors,
            edgecolor="black")
    ax.set_yticks(range(len(signed_corrs)))
    ax.set_yticklabels(signed_corrs.index)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(f"Feature Correlations with Target: {target_col}")
    ax.invert_yaxis()
    for i, v in enumerate(signed_corrs.values):
        ax.text(v + 0.01 if v >= 0 else v - 0.08, i, f"{v:.2f}", va="center")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Explanatory text generation (enhanced with recommendations)
# ------------------------------------------------------------------
def explain_visualizations(
    analysis_result: Dict[str, Any],
    recommendations: Optional[Dict[str, Any]] = None,
    config: Optional[MLToolkitConfig] = None,
) -> Dict[str, str]:
    """Produce organized, actionable explanations for each visualization.

    Each explanation includes:
    - What the plot shows.
    - Key observations derived from the actual data.
    - Recommended actions, pulled from the `RecommendationEngine`
      (either supplied via `recommendations` or computed automatically).

    Parameters
    ----------
    analysis_result : dict
        Output of `StatisticsEngine.run_full_analysis()`.
    recommendations : dict, optional
        Pre‑computed recommendations from
        `RecommendationEngine.generate_recommendations()`. If not
        provided, they will be generated internally using the default
        configuration.
    config : MLToolkitConfig, optional
        Configuration used when generating recommendations on the fly.

    Returns
    -------
    dict
        Keys:
        - 'numeric_distributions'
        - 'target_distribution'
        - 'correlation_heatmap'
        - 'missing_values_heatmap'
        - 'outlier_summary'
        - 'target_correlations'

        Each value is a formatted multi‑line string.
    """
    # ------------------------------------------------------------------
    # Ensure we have recommendations
    # ------------------------------------------------------------------
    if recommendations is None:
        # Local import to avoid circular dependency at module level
        from preml.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine(config=config, enable_feature_engineering=False)
        recommendations = engine.generate_recommendations(analysis_result)

    # ------------------------------------------------------------------
    # Extract relevant parts from recommendations
    # ------------------------------------------------------------------
    imputation_recs = recommendations.get("imputation", [])
    outlier_recs = recommendations.get("outlier_handling", [])
    transformation_recs = recommendations.get("transformation", [])
    scaling_rec = recommendations.get("scaling")
    encoding_recs = recommendations.get("encoding", [])
    feature_selection_recs = recommendations.get("feature_selection", [])
    data_quality_notes = recommendations.get("data_quality_notes", [])

    # ------------------------------------------------------------------
    # Helper to format a list of recommendation strings
    # ------------------------------------------------------------------
    def _format_recs(rec_list, indent="  - "):
        if not rec_list:
            return ""
        items = []
        for rec in rec_list:
            items.append(f"{indent}{rec.action} (confidence: {rec.confidence:.0%})")
        return "\n".join(items)

    explanations: Dict[str, str] = {}

    # ==================================================================
    # 1. Numeric Distributions
    # ==================================================================
    profiles = _get_profiles(analysis_result)
    numeric = [p for p in profiles if p.numeric_profile and not p.is_constant]
    if numeric:
        lines = [
            "**What this plot shows:**",
            "  Histogram and boxplot for each numeric feature. The histogram",
            "  displays the distribution with mean (red dashed) and median",
            "  (green solid). IQR bounds (orange dotted) are shown if enabled.",
            "  The boxplot summarizes quartiles and outliers.",
            "",
            "**Key observations:**"
        ]
        skewed_cols = [p.column for p in numeric if abs(p.numeric_profile.skewness) > 1.0]
        if skewed_cols:
            lines.append(f"  Skewed features (|skew| > 1): {', '.join(skewed_cols)}")
        else:
            lines.append("  No features with strong skew detected.")
        lines.append("")
        lines.append("**Recommended actions:**")
        if transformation_recs:
            lines.append("  Transformations to reduce skew:")
            lines.append(_format_recs(transformation_recs))
        else:
            lines.append("  No transformation needed for skew.")
        if outlier_recs:
            lines.append("  Outlier handling:")
            lines.append(_format_recs(outlier_recs))
        if scaling_rec:
            lines.append(f"  Scaling: {scaling_rec.action}")
        explanations["numeric_distributions"] = "\n".join(lines)
    else:
        explanations["numeric_distributions"] = (
            "No non‑constant numeric features to plot."
        )

    # ==================================================================
    # 2. Target Distribution
    # ==================================================================
    target_profile = _get_target_profile(analysis_result)
    if target_profile:
        lines = [
            "**What this plot shows:**",
            "  Distribution of the target variable. For regression: histogram",
            "  and boxplot. For classification: bar chart of class counts.",
            "",
            "**Key observations:**"
        ]
        if target_profile.is_regression:
            # We don't have skew directly, but we can mention if high missing or n_unique
            lines.append(f"  Regression target with {target_profile.n_unique} unique values.")
        else:
            lines.append(f"  Classification target with {target_profile.n_unique} classes.")
            if target_profile.class_distribution:
                max_cls = max(target_profile.class_distribution.values())
                min_cls = min(target_profile.class_distribution.values())
                ratio = max_cls / min_cls if min_cls > 0 else float("inf")
                if ratio > 5:
                    lines.append(f"  Class imbalance detected (max/min ratio = {ratio:.1f}).")
        lines.append("")
        lines.append("**Recommended actions:**")
        if target_profile.is_regression:
            # Suggest target transformation if distribution is heavily skewed (heuristic)
            lines.append("  Consider a log or Box‑Cox transformation if the target is skewed.")
            lines.append("  Ensure the evaluation metric is appropriate (e.g., RMSLE for skewed targets).")
        else:
            if target_profile.class_distribution and ratio > 5:
                lines.append("  Use stratified sampling during train/test split.")
                lines.append("  Consider class weights, oversampling (SMOTE), or undersampling.")
            lines.append("  Choose metrics robust to imbalance (F1, AUC‑ROC).")
        explanations["target_distribution"] = "\n".join(lines)
    else:
        explanations["target_distribution"] = "No target variable defined; plot not available."

    # ==================================================================
    # 3. Correlation Heatmap
    # ==================================================================
    correlation_pairs = _get_correlations(analysis_result)
    if correlation_pairs:
        lines = [
            "**What this plot shows:**",
            "  Heatmap of Pearson correlation coefficients between numeric",
            "  features that appear in pairs above the threshold. The upper",
            "  triangle is masked.",
            "",
            "**Key observations:**"
        ]
        high_pairs = [p for p in correlation_pairs if abs(p.coefficient) > 0.9]
        if high_pairs:
            lines.append("  Very strong correlations (|r| > 0.9):")
            for p in high_pairs[:5]:
                lines.append(f"    {p.feature_a} vs {p.feature_b}: {p.coefficient:.2f}")
        else:
            lines.append("  No extremely high pairwise correlations detected.")
        lines.append("")
        lines.append("**Recommended actions:**")
        if feature_selection_recs:
            lines.append("  Handle multicollinearity:")
            lines.append(_format_recs(feature_selection_recs))
        else:
            lines.append("  No collinearity issues above threshold.")
        explanations["correlation_heatmap"] = "\n".join(lines)
    else:
        explanations["correlation_heatmap"] = (
            "No correlations above threshold; heatmap not generated."
        )

    # ==================================================================
    # 4. Missing Values Heatmap
    # ==================================================================
    missing_cols = _get_missing_columns(
        pd.DataFrame()  # df not needed for explanation; we can rely on analysis data
    )
    # Instead, use missing report from analysis_result if available
    missing_report = analysis_result.get("missing")
    if missing_report and missing_report.total_missing > 0:
        lines = [
            "**What this plot shows:**",
            "  Yellow cells indicate missing values; rows are sampled for large",
            "  datasets. Each row is a record, each column a feature.",
            "",
            "**Key observations:**"
        ]
        lines.append(f"  Total missing values: {missing_report.total_missing}")
        lines.append(f"  Columns with missing: {len(missing_report.columns_with_missing)}")
        # show top 3 columns with highest missing
        top_missing = sorted(
            missing_report.column_reports,
            key=lambda x: x.missing_percent,
            reverse=True
        )[:3]
        for col_rep in top_missing:
            lines.append(f"    {col_rep.column}: {col_rep.missing_percent:.1f}% missing")
        lines.append("")
        lines.append("**Recommended actions:**")
        if imputation_recs:
            lines.append("  Imputation strategies:")
            lines.append(_format_recs(imputation_recs))
        else:
            lines.append("  No imputation recommendations (may be below threshold).")
        explanations["missing_values_heatmap"] = "\n".join(lines)
    else:
        explanations["missing_values_heatmap"] = (
            "No missing values detected; heatmap not generated."
        )

    # ==================================================================
    # 5. Outlier Summary
    # ==================================================================
    outliers = _get_outliers(analysis_result)
    if outliers:
        lines = [
            "**What this plot shows:**",
            "  Bar chart of outlier percentages (IQR method) for numeric features.",
            "",
            "**Key observations:**"
        ]
        high_outliers = [o for o in outliers if o.outlier_percent > 5.0]
        if high_outliers:
            lines.append("  Features with >5% outliers:")
            for o in high_outliers:
                lines.append(f"    {o.column}: {o.outlier_percent:.1f}%")
        else:
            lines.append("  All outlier percentages ≤ 5%.")
        lines.append("")
        lines.append("**Recommended actions:**")
        if outlier_recs:
            lines.append("  Outlier treatment:")
            lines.append(_format_recs(outlier_recs))
        else:
            lines.append("  No outlier handling needed.")
        explanations["outlier_summary"] = "\n".join(lines)
    else:
        explanations["outlier_summary"] = "No outlier data available."

    # ==================================================================
    # 6. Target Correlations
    # ==================================================================
    if target_profile and target_profile.is_regression:
        lines = [
            "**What this plot shows:**",
            "  Horizontal bar chart of Pearson correlations between each numeric",
            "  feature and the regression target. Teal = positive, coral = negative.",
            "",
            "**Key observations:**"
        ]
        # We don't have the actual correlation values here (plot depends on df),
        # but we can mention that it highlights linear predictors.
        lines.append("  Strong linear predictors will have tall bars.")
        lines.append("")
        lines.append("**Recommended actions:**")
        lines.append("  Focus on features with high absolute correlation, but be aware that")
        lines.append("  non‑linear relationships won't be captured. Supplement with mutual")
        lines.append("  information or tree‑based feature importance.")
        explanations["target_correlations"] = "\n".join(lines)
    else:
        explanations["target_correlations"] = (
            "Target correlations plot is only available for numeric regression targets."
        )

    return explanations
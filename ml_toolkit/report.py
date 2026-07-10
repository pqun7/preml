"""
report.py — Automated, insight‑rich report generation from EDA results.

This module creates self‑contained reports in HTML, Markdown, and plain text.
Reports include an executive summary, statistical details, data quality
assessment, actionable preprocessing and model recommendations, and optional
embedded visualizations with explanatory captions. The module never
recomputes statistics; it builds upon the analysis dictionary produced by
:class:`ml_toolkit.eda.EDAAnalyzer`.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from ml_toolkit.config import MLToolkitConfig, default_config
from ml_toolkit.exceptions import ReportError
from ml_toolkit.recommendation_utils import normalize_recommendation_items
from ml_toolkit.schema import (
    FeatureProfile,
    CorrelationPair,
    OutlierReport,
    TargetProfile,
)
from ml_toolkit.visualization import (
    explain_visualizations,
    plot_correlation_heatmap,
    plot_missing_heatmap,
    plot_numeric_distributions,
    plot_outlier_summary,
    plot_target_correlations,
    plot_target_distribution,
)

# ---------------------------------------------------------------------------
# Embedded CSS for HTML reports
# ---------------------------------------------------------------------------
_REPORT_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       margin: 40px; color: #333; max-width: 1200px; }
h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
h2 { color: #2980b9; margin-top: 40px; border-bottom: 1px solid #ddd; }
h3 { color: #2c3e50; margin-top: 25px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #3498db; color: white; }
tr:nth-child(even) { background-color: #f2f2f2; }
pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
.quality-score { font-size: 2.5em; font-weight: bold; }
.warning { color: #e74c3c; }
.good { color: #27ae60; }
.moderate { color: #f39c12; }
.collapsible { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 15px; }
.caption { font-style: italic; color: #555; margin-top: 5px; border-left: 3px solid #3498db; padding-left: 10px; }
.summary-box { background: #eaf2f8; border-left: 5px solid #2980b9; padding: 15px; margin: 20px 0; }
.score-breakdown { margin-top: 10px; }
"""


# ---------------------------------------------------------------------------
# Helper: encode figure to HTML img tag
# ---------------------------------------------------------------------------
def _fig_to_html(fig: matplotlib.figure.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)  # prevent memory leaks
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;" />'


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------
class ReportGenerator:
    """Generates formatted EDA reports with actionable insights.

    The generator consumes the complete analysis dictionary produced by
    :meth:`EDAAnalyzer.run() <ml_toolkit.eda.EDAAnalyzer.run>` and
    optionally the original DataFrame for embedded plots.

    Parameters
    ----------
    analysis_result : dict
        The full analysis result dictionary.
    df : pd.DataFrame, optional
        Original dataset. Required for generating plots. If omitted, the
        report will contain only textual/statistical information.
    config : MLToolkitConfig, optional
        Configuration object; used for visualisation settings.
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

        # ------------------------------------------------------------------
        # Extract commonly used sections with safe defaults
        # ------------------------------------------------------------------
        self.metadata = analysis_result.get("metadata")
        self.duplicates = analysis_result.get("duplicates")
        self.infinite = analysis_result.get("infinite")
        self.missing = analysis_result.get("missing")
        self.outliers: List[OutlierReport] = analysis_result.get("outliers", [])
        self.feature_profiles: List[FeatureProfile] = analysis_result.get("feature_profiles", [])
        self.correlation_pairs: List[CorrelationPair] = analysis_result.get("correlation_pairs", [])
        self.target_profile: Optional[TargetProfile] = analysis_result.get("target_profile")
        self.recommendations: Dict[str, Any] = analysis_result.get("recommendations", {})
        self.quality_score = analysis_result.get("data_quality_score", 0.0)
        self.quality_notes: List[str] = analysis_result.get("data_quality_notes", [])

        # Ensure recommendations exist (generate if missing, but only once)
        self._recommendations_ensured = False

    # ------------------------------------------------------------------
    # Internal: ensure recommendations are available
    # ------------------------------------------------------------------
    def _ensure_recommendations(self) -> None:
        """If no recommendations exist, compute them using the engine."""
        if self._recommendations_ensured:
            return
        if not self.recommendations:
            try:
                from ml_toolkit.recommendation_engine import RecommendationEngine
                engine = RecommendationEngine(config=self.config, enable_feature_engineering=False)
                self.recommendations = engine.generate_recommendations(self.analysis)
            except Exception:
                # If recommendation generation fails, leave empty
                pass
        self._recommendations_ensured = True

    # ------------------------------------------------------------------
    # Helper: build a "key findings" summary
    # ------------------------------------------------------------------
    def _key_findings(self) -> List[str]:
        findings = []

        # Missing values
        if self.missing and self.missing.total_missing > 0:
            pct = (self.missing.total_missing / (self.metadata.n_rows * self.metadata.n_columns)) * 100 if self.metadata else 0
            if pct > 5:
                findings.append(f"High missing data ({pct:.1f}% of cells). Imputation or column removal is recommended.")
            else:
                findings.append(f"Minor missing data ({pct:.1f}% of cells). Imputation should be straightforward.")

        # Outliers
        outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
        if outlier_cols:
            high_outlier = [o for o in outlier_cols if o.outlier_percent > 5]
            if high_outlier:
                cols = ", ".join(o.column for o in high_outlier[:3])
                findings.append(f"Significant outliers in {cols} (and others). Consider robust scaling or capping.")
            else:
                findings.append("Outliers are present but within acceptable limits.")

        # Correlations
        if self.correlation_pairs:
            strong = [p for p in self.correlation_pairs if abs(p.coefficient) > 0.9]
            if strong:
                pairs = ", ".join(f"{p.feature_a}/{p.feature_b}" for p in strong[:3])
                findings.append(f"Very high correlation (|r|>0.9) between {pairs}. Multicollinearity may affect linear models.")

        # Constant / quasi-constant features
        const = [p.column for p in self.feature_profiles if p.is_constant]
        quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
        if const or quasi:
            findings.append(f"Dataset contains {len(const)} constant and {len(quasi)} quasi-constant features; they carry no information and should be dropped.")

        # Target
        if self.target_profile:
            if self.target_profile.is_regression:
                # just mention if many unique values
                pass
            else:
                if self.target_profile.n_unique == 2 and self.target_profile.is_binary:
                    # check imbalance
                    class_dist = self.target_profile.class_distribution
                    if class_dist:
                        vals = list(class_dist.values())
                        if min(vals)/max(vals) < 0.3:
                            findings.append("Target class imbalance detected. Use stratified sampling and consider class weights.")
                elif self.target_profile.n_unique > 10:
                    findings.append("Multi‑class target with many classes; evaluate if grouping is possible.")

        # Infinity
        if self.infinite and self.infinite.columns_with_inf:
            findings.append("Infinite values found in columns: " + ", ".join(self.infinite.columns_with_inf[:3]) + ". Replace or remove them.")

        return findings

    # ------------------------------------------------------------------
    # Quality score breakdown (simple)
    # ------------------------------------------------------------------
    def _score_breakdown(self) -> str:
        parts = []
        # Assume the quality score starts at 100 and deductions are made
        # We'll create a human-readable text based on quality_notes and thresholds.
        # This is illustrative; real implementation would use the actual scoring logic.
        if self.quality_score >= 70:
            status = "Good"
        elif self.quality_score >= 40:
            status = "Fair"
        else:
            status = "Poor"
        parts.append(f"Overall Quality: {status} ({self.quality_score:.1f}/100)")
        if self.quality_notes:
            parts.append("Contributing factors:")
            for note in self.quality_notes:
                parts.append(f"  • {note}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Plain‑text report (enhanced)
    # ------------------------------------------------------------------
    def generate_text(self) -> str:
        """Generate a plain‑text EDA report with summary and recommendations.

        Returns
        -------
        str
            Formatted plain‑text report.
        """
        self._ensure_recommendations()
        lines = []
        lines.append("=" * 70)
        lines.append("                    ML TOOLKIT – EDA REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Key Findings
        findings = self._key_findings()
        if findings:
            lines.append(">>> KEY FINDINGS <<<")
            for f in findings:
                lines.append(f"  • {f}")
            lines.append("")

        # Dataset overview
        if self.metadata:
            lines.append("[Dataset Overview]")
            lines.append(f"  Rows: {self.metadata.n_rows}")
            lines.append(f"  Columns: {self.metadata.n_columns}")
            lines.append(f"  Memory: {self.metadata.memory_mb:.2f} MB")
            lines.append("")

        # Data quality
        lines.append("[Data Quality]")
        lines.append(self._score_breakdown())
        lines.append("")

        # Duplicates
        if self.duplicates:
            lines.append("[Duplicates]")
            lines.append(f"  Total duplicates: {self.duplicates.total_duplicates} "
                         f"({self.duplicates.duplicate_percent:.2f}%)")
            lines.append("")

        # Infinite
        if self.infinite and self.infinite.columns_with_inf:
            lines.append("[Infinite Values]")
            for col, cnt in self.infinite.counts.items():
                lines.append(f"  {col}: {cnt} infinite values")
            lines.append("")

        # Missing values
        if self.missing and self.missing.total_missing > 0:
            lines.append("[Missing Values]")
            lines.append(f"  Total missing cells: {self.missing.total_missing}")
            lines.append("  Top columns:")
            top_miss = sorted(
                self.missing.column_reports,
                key=lambda x: x.missing_percent, reverse=True
            )[:5]
            for col_rpt in top_miss:
                lines.append(f"    {col_rpt.column}: {col_rpt.missing_percent:.2f}%")
            lines.append("")

        # Outliers
        if self.outliers:
            outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
            if outlier_cols:
                lines.append("[Outliers (IQR)]")
                for o in outlier_cols[:10]:
                    lines.append(f"  {o.column}: {o.outlier_count} ({o.outlier_percent:.2f}%)")
                if len(outlier_cols) > 10:
                    lines.append(f"  ... and {len(outlier_cols)-10} more columns.")
                lines.append("")

        # Feature profiles summary
        if self.feature_profiles:
            lines.append("[Feature Profiles Summary]")
            const = [p.column for p in self.feature_profiles if p.is_constant]
            quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
            if const:
                lines.append(f"  Constant columns: {', '.join(const)}")
            if quasi:
                lines.append(f"  Quasi-constant columns: {', '.join(quasi)}")
            lines.append("")

        # Correlations
        if self.correlation_pairs:
            lines.append("[Highly Correlated Feature Pairs]")
            for pair in self.correlation_pairs[:10]:
                lines.append(f"  {pair.feature_a} vs {pair.feature_b}: r={pair.coefficient:.2f}")
            if len(self.correlation_pairs) > 10:
                lines.append(f"  ... and {len(self.correlation_pairs)-10} more pairs.")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("[Recommendations Summary]")
            try:
                from ml_toolkit.recommendation_engine import RecommendationEngine
                summary_text = RecommendationEngine.summarize(self.recommendations)
                lines.append(summary_text)
            except Exception:
                # fallback to simple listing
                for category in ["imputation", "outlier_handling", "transformation",
                                 "scaling", "encoding", "feature_engineering", "feature_selection"]:
                    recs = normalize_recommendation_items(self.recommendations.get(category))
                    for r in recs[:3]:
                        lines.append(f"  [{category}] {r.action}")
            lines.append("")

        # Models
        model_recs = self.recommendations.get("models", [])
        if model_recs:
            lines.append("[Recommended Baseline Models]")
            for m in model_recs:
                lines.append(f"  {m.model_name} ({m.suitability}): {m.reason}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown report (enhanced)
    # ------------------------------------------------------------------
    def generate_markdown(self) -> str:
        """Generate a Markdown formatted EDA report with insights.

        Returns
        -------
        str
            Markdown string.
        """
        self._ensure_recommendations()
        md = []
        md.append("# ML Toolkit – EDA Report\n")

        # Key Findings
        findings = self._key_findings()
        if findings:
            md.append("## Key Findings\n")
            for f in findings:
                md.append(f"- {f}")
            md.append("")

        # Dataset overview
        if self.metadata:
            md.append("## Dataset Overview\n")
            md.append(f"- **Rows:** {self.metadata.n_rows}")
            md.append(f"- **Columns:** {self.metadata.n_columns}")
            md.append(f"- **Memory:** {self.metadata.memory_mb:.2f} MB\n")

        # Data quality
        md.append("## Data Quality\n")
        score_color = "green" if self.quality_score >= 70 else "orange" if self.quality_score >= 40 else "red"
        md.append(f"**Score:** <span style='color:{score_color};font-size:1.5em;'>{self.quality_score:.1f}/100</span>\n")
        if self.quality_notes:
            for note in self.quality_notes:
                md.append(f"- {note}")
        md.append("")

        # Duplicates, Infinite, Missing, Outliers (tables)
        if self.duplicates:
            md.append("## Duplicates\n")
            md.append(f"Total duplicate rows: {self.duplicates.total_duplicates} ({self.duplicates.duplicate_percent:.2f}%)\n")
        if self.infinite and self.infinite.columns_with_inf:
            md.append("## Infinite Values\n")
            md.append("| Column | Count |")
            md.append("|--------|-------|")
            for col, cnt in self.infinite.counts.items():
                md.append(f"| {col} | {cnt} |")
            md.append("")
        if self.missing and self.missing.total_missing > 0:
            md.append("## Missing Values\n")
            md.append(f"Total missing cells: {self.missing.total_missing}\n")
            md.append("| Column | Missing Count | Missing % |")
            md.append("|--------|---------------|-----------|")
            for col_rpt in self.missing.column_reports:
                md.append(f"| {col_rpt.column} | {col_rpt.missing_count} | {col_rpt.missing_percent:.2f}% |")
            md.append("")
        if self.outliers:
            outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
            if outlier_cols:
                md.append("## Outliers (IQR)\n")
                md.append("| Column | Outlier Count | Outlier % |")
                md.append("|--------|---------------|-----------|")
                for o in outlier_cols:
                    md.append(f"| {o.column} | {o.outlier_count} | {o.outlier_percent:.2f}% |")
                md.append("")

        # Feature profiles
        if self.feature_profiles:
            const = [p.column for p in self.feature_profiles if p.is_constant]
            quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
            if const or quasi:
                md.append("## Feature Status\n")
                if const:
                    md.append(f"**Constant columns:** {', '.join(const)}")
                if quasi:
                    md.append(f"**Quasi-constant columns:** {', '.join(quasi)}")
                md.append("")

        # Correlations
        if self.correlation_pairs:
            md.append("## Highly Correlated Pairs\n")
            md.append("| Feature A | Feature B | Coefficient |")
            md.append("|-----------|-----------|-------------|")
            for pair in self.correlation_pairs:
                md.append(f"| {pair.feature_a} | {pair.feature_b} | {pair.coefficient:.2f} |")
            md.append("")

        # Recommendations (using summarizer for better formatting)
        if self.recommendations:
            md.append("## Recommendations\n")
            try:
                from ml_toolkit.recommendation_engine import RecommendationEngine
                summary = RecommendationEngine.summarize(self.recommendations)
                # Convert the plain-text summary to markdown (basic)
                summary = summary.replace("=" * 60, "---\n")
                md.append(summary)
            except Exception:
                # fallback to simple list
                for cat in ["imputation", "outlier_handling", "transformation",
                            "scaling", "encoding", "feature_engineering", "feature_selection"]:
                    recs = normalize_recommendation_items(self.recommendations.get(cat))
                    if recs:
                        md.append(f"### {cat.replace('_', ' ').title()}\n")
                        for r in recs:
                            md.append(f"- **{r.action}** (confidence: {r.confidence:.2f})")
                md.append("")

        # Models
        model_recs = self.recommendations.get("models", [])
        if model_recs:
            md.append("## Recommended Models\n")
            for m in model_recs:
                md.append(f"- **{m.model_name}** ({m.suitability}): {m.reason}")
            md.append("")

        return "\n".join(md)

    # ------------------------------------------------------------------
    # HTML report (enriched with explanations)
    # ------------------------------------------------------------------
    def generate_html(self, embed_plots: bool = True) -> str:
        """Generate a self‑contained HTML EDA report with insights.

        Parameters
        ----------
        embed_plots : bool
            If True and *df* was provided, embed base64‑encoded
            visualisations with explanatory captions.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        self._ensure_recommendations()

        # Prepare explanations dictionary (once)
        try:
            explanations = explain_visualizations(self.analysis, recommendations=self.recommendations, config=self.config)
        except Exception:
            explanations = {}

        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head><meta charset='utf-8'><title>EDA Report</title>")
        html_parts.append(f"<style>{_REPORT_CSS}</style></head><body>")

        # Title & Quality Score
        html_parts.append("<h1>ML Toolkit – EDA Report</h1>")
        score_class = "good" if self.quality_score >= 70 else "moderate" if self.quality_score >= 40 else "warning"
        html_parts.append(
            f"<p>Data Quality Score: <span class='quality-score {score_class}'>{self.quality_score:.1f}</span>/100</p>"
        )

        # Key Findings
        findings = self._key_findings()
        if findings:
            html_parts.append("<div class='summary-box'><h2>Key Findings</h2><ul>")
            for f in findings:
                html_parts.append(f"<li>{f}</li>")
            html_parts.append("</ul></div>")

        # Dataset Overview
        if self.metadata:
            html_parts.append("<h2>Dataset Overview</h2>")
            html_parts.append("<ul>")
            html_parts.append(f"<li><strong>Rows:</strong> {self.metadata.n_rows}</li>")
            html_parts.append(f"<li><strong>Columns:</strong> {self.metadata.n_columns}</li>")
            html_parts.append(f"<li><strong>Memory:</strong> {self.metadata.memory_mb:.2f} MB</li>")
            html_parts.append("</ul>")

        # Quality Score Breakdown
        html_parts.append("<h2>Data Quality Breakdown</h2>")
        html_parts.append("<div class='score-breakdown'>")
        html_parts.append(self._score_breakdown().replace("\n", "<br>"))
        html_parts.append("</div>")

        # Duplicates, Infinite, Missing, Outliers (tables)
        if self.duplicates and self.duplicates.total_duplicates > 0:
            html_parts.append("<h2>Duplicates</h2>")
            html_parts.append(f"<p>Total duplicates: {self.duplicates.total_duplicates} ({self.duplicates.duplicate_percent:.2f}%)</p>")
        if self.infinite and self.infinite.columns_with_inf:
            html_parts.append("<h2>Infinite Values</h2><table><tr><th>Column</th><th>Count</th></tr>")
            for col, cnt in self.infinite.counts.items():
                html_parts.append(f"<tr><td>{col}</td><td>{cnt}</td></tr>")
            html_parts.append("</table>")
        if self.missing and self.missing.total_missing > 0:
            html_parts.append("<h2>Missing Values</h2>")
            html_parts.append(f"<p>Total missing cells: {self.missing.total_missing}</p>")
            html_parts.append("<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>")
            for col_rpt in self.missing.column_reports:
                html_parts.append(f"<tr><td>{col_rpt.column}</td><td>{col_rpt.missing_count}</td><td>{col_rpt.missing_percent:.2f}%</td></tr>")
            html_parts.append("</table>")
        if self.outliers:
            outlier_cols = [o for o in self.outliers if o.outlier_count > 0]
            if outlier_cols:
                html_parts.append("<h2>Outliers (IQR)</h2><table><tr><th>Column</th><th>Outlier Count</th><th>Outlier %</th></tr>")
                for o in outlier_cols:
                    html_parts.append(f"<tr><td>{o.column}</td><td>{o.outlier_count}</td><td>{o.outlier_percent:.2f}%</td></tr>")
                html_parts.append("</table>")

        # Feature Profiles
        if self.feature_profiles:
            const = [p.column for p in self.feature_profiles if p.is_constant]
            quasi = [p.column for p in self.feature_profiles if p.is_quasi_constant]
            if const or quasi:
                html_parts.append("<h2>Feature Status</h2>")
                if const:
                    html_parts.append(f"<p><strong>Constant columns:</strong> {', '.join(const)}</p>")
                if quasi:
                    html_parts.append(f"<p><strong>Quasi-constant columns:</strong> {', '.join(quasi)}</p>")

        # Correlations
        if self.correlation_pairs:
            html_parts.append("<h2>Highly Correlated Pairs</h2><table><tr><th>Feature A</th><th>Feature B</th><th>Coefficient</th></tr>")
            for pair in self.correlation_pairs:
                html_parts.append(f"<tr><td>{pair.feature_a}</td><td>{pair.feature_b}</td><td>{pair.coefficient:.2f}</td></tr>")
            html_parts.append("</table>")

        # Recommendations (using summarizer)
        if self.recommendations:
            html_parts.append("<h2>Recommendations</h2>")
            try:
                from ml_toolkit.recommendation_engine import RecommendationEngine
                summary_txt = RecommendationEngine.summarize(self.recommendations)
                # Convert plain text to HTML with monospace pre block
                html_parts.append("<pre>" + summary_txt + "</pre>")
            except Exception:
                # fallback
                for cat in ["imputation", "outlier_handling", "transformation",
                            "scaling", "encoding", "feature_engineering", "feature_selection"]:
                    recs = normalize_recommendation_items(self.recommendations.get(cat))
                    if recs:
                        html_parts.append(f"<h3>{cat.replace('_', ' ').title()}</h3><ul>")
                        for r in recs:
                            html_parts.append(f"<li><strong>{r.action}</strong> (confidence: {r.confidence:.2f})</li>")
                        html_parts.append("</ul>")

        # Model Recommendations
        model_recs = self.recommendations.get("models", [])
        if model_recs:
            html_parts.append("<h2>Recommended Models</h2><ul>")
            for m in model_recs:
                html_parts.append(f"<li><strong>{m.model_name}</strong> ({m.suitability}): {m.reason}</li>")
            html_parts.append("</ul>")

        # Visualizations with explanations
        if embed_plots and self.df is not None:
            html_parts.append("<h2>Visualisations</h2>")

            def _add_plot_with_caption(fig, caption_key: str, section_title: str):
                if fig is not None:
                    html_parts.append(f"<h3>{section_title}</h3>")
                    if caption_key in explanations:
                        html_parts.append(f"<div class='caption'>{explanations[caption_key]}</div>")
                    html_parts.append(_fig_to_html(fig))

            # Missing heatmap
            try:
                fig = plot_missing_heatmap(self.df, config=self.config)
                _add_plot_with_caption(fig, "missing_values_heatmap", "Missing Values Heatmap")
            except Exception:
                pass

            # Numeric distributions
            try:
                fig = plot_numeric_distributions(self.df, self.analysis,
                                                 max_cols=getattr(self.config, "max_plot_cols", 12),
                                                 config=self.config)
                _add_plot_with_caption(fig, "numeric_distributions", "Numeric Distributions")
            except Exception:
                pass

            # Outlier summary
            try:
                fig = plot_outlier_summary(self.analysis, config=self.config)
                _add_plot_with_caption(fig, "outlier_summary", "Outlier Summary")
            except Exception:
                pass

            # Correlation heatmap
            try:
                fig = plot_correlation_heatmap(self.df, self.analysis, config=self.config)
                _add_plot_with_caption(fig, "correlation_heatmap", "Correlation Heatmap")
            except Exception:
                pass

            # Target plots
            if self.target_profile is not None:
                try:
                    fig = plot_target_distribution(self.df, self.analysis, config=self.config)
                    _add_plot_with_caption(fig, "target_distribution", "Target Distribution")
                except Exception:
                    pass
                if self.target_profile.is_regression:
                    try:
                        fig = plot_target_correlations(self.df, self.analysis, config=self.config)
                        _add_plot_with_caption(fig, "target_correlations", "Feature Correlations with Target")
                    except Exception:
                        pass

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    # ------------------------------------------------------------------
    # Save reports to files
    # ------------------------------------------------------------------
    def save_report(self, filepath: str, format: str = "html", embed_plots: bool = True) -> None:
        """Generate a report and save it to a file.

        Parameters
        ----------
        filepath : str
            Path to the output file (extension will be forced to match format
            if not already correct).
        format : str
            One of ``'html'``, ``'md'``, ``'txt'``.
        embed_plots : bool
            Only relevant for HTML; see :meth:`generate_html`.

        Raises
        ------
        ReportError
            If the format is unsupported.
        """
        format = format.lower()
        if format == "html":
            content = self.generate_html(embed_plots=embed_plots)
        elif format in ("md", "markdown"):
            content = self.generate_markdown()
            if not filepath.endswith(".md"):
                filepath += ".md"
        elif format in ("txt", "text"):
            content = self.generate_text()
            if not filepath.endswith(".txt"):
                filepath += ".txt"
        else:
            raise ReportError(f"Unsupported report format: '{format}'.")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
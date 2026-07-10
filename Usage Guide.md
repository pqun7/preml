# PreML

**PreML** is a Python library that automates exploratory data analysis, preprocessing, feature engineering, baseline model training, and report generation for tabular machine learning projects.

It generates evidence-based recommendations and builds `scikit-learn`-compatible preprocessing pipelines while remaining transparent, configurable, and production-ready.

---

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core Components](#core-components)
  - [EDAAnalyzer](#edaanalyzer)
  - [StatisticsEngine](#statisticsengine)
  - [RecommendationEngine](#recommendationengine)
  - [Visualization](#visualization)
  - [PreprocessingBuilder](#preprocessingbuilder)
  - [FeatureEngineering](#featureengineering)
  - [BaselineTrainer](#baselinetrainer)
  - [ReportGenerator](#reportgenerator)
- [End-to-End Workflow](#end-to-end-workflow)
- [Performance](#performance)
- [Thread Safety](#thread-safety)
- [Reproducibility](#reproducibility)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)
- [FAQ](#faq)
- [Component Relationships](#component-relationships)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [License](#license)
- [Citation](#citation)

---

# Introduction

**PreML** automates the repetitive, data-driven stages of traditional machine learning workflows.

The library combines statistical analysis, intelligent preprocessing recommendations, feature engineering suggestions, visualization, baseline model evaluation, and professional report generation into a unified workflow.

Every recommendation is generated from statistical evidence rather than hard-coded assumptions, ensuring transparency and interpretability throughout the analysis process.

---

# Requirements

PreML requires the following software and dependencies.

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn *(used for visualization)*
- scipy *(used for statistical methods)*

---

# Installation

## Install from PyPI

```bash
pip install preml
```

## Install from Source

Clone the repository and install it in editable mode.

```bash
git clone https://github.com/alinazer30/preml.git

cd preml

pip install -e .
```

---

# Quick Start

The following example demonstrates a minimal end-to-end workflow.

```python
import pandas as pd

from preml.eda import EDAAnalyzer
from preml.preprocessing import PreprocessingBuilder

df = pd.read_csv("housing.csv")

analysis = EDAAnalyzer(
    df,
    target="SalePrice",
).run()

builder = PreprocessingBuilder(analysis)

pipeline = builder.build_pipeline()

feature_cols = [
    p.column
    for p in analysis["feature_profiles"]
]

X_transformed, = builder.fit_transform(
    df[feature_cols]
)
```

> **Tip**
>
> For a complete machine learning workflow including feature engineering, baseline model evaluation, and report generation, see the **End-to-End Workflow** section.

---

# Configuration

All configurable thresholds and runtime settings are managed through `MLToolkitConfig`.

Whenever a configuration parameter is omitted, the toolkit automatically falls back to safe default values.

This ensures stable behavior while allowing advanced users to customize every stage of the workflow.

---

## Create a Custom Configuration

```python
from preml.config import MLToolkitConfig

config = MLToolkitConfig(
    missing_threshold=0.30,
    correlation_threshold=0.85,
    skewness_threshold=1.0,
    outlier_method="iqr",      # "iqr" or "zscore"
    random_state=42,
    max_plot_cols=12,
    figure_size=(12, 8),
    plot_style="whitegrid",
    color_palette="muted",
)
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `missing_threshold` | Missing-value threshold used when generating recommendations. |
| `correlation_threshold` | Minimum absolute correlation considered significant. |
| `skewness_threshold` | Threshold used when recommending feature transformations. |
| `outlier_method` | Outlier detection algorithm (`iqr` or `zscore`). |
| `random_state` | Random seed used for reproducible workflows. |
| `max_plot_cols` | Maximum number of columns displayed in generated visualizations. |
| `figure_size` | Default figure size used by visualization functions. |
| `plot_style` | Global Matplotlib plotting style. |
| `color_palette` | Default visualization color palette. |

---

## Use the Default Configuration

```python
from preml.config import default_config

config = default_config
```

---

## Pass the Configuration to Components

Every component that supports configuration accepts the same `MLToolkitConfig` instance.

```python
from preml.eda import EDAAnalyzer

analyzer = EDAAnalyzer(
    df,
    target="target",
    config=config,
)
```

> **Best Practice**
>
> Create a single configuration instance and reuse it across the entire workflow to ensure consistent thresholds, preprocessing behavior, visualization settings, and model recommendations.

---

# Core Components

## EDAAnalyzer

### Overview

`EDAAnalyzer` is the central component of PreML.

It orchestrates the complete exploratory data analysis (EDA) workflow by combining statistical analysis, recommendation generation, and data quality assessment into a single interface.

---

### Highlights

- Computes dataset metadata.
- Detects missing values, duplicates, and infinite values.
- Identifies outliers using configurable detection methods.
- Profiles numerical and categorical features.
- Computes feature correlations.
- Profiles the target variable.
- Generates evidence-based preprocessing recommendations.
- Calculates an overall data quality score with explanatory notes.

---

### Example

```python
from preml.eda import EDAAnalyzer

analyzer = EDAAnalyzer(
    df,
    target="price",
)

analysis = analyzer.run()
```

**Returned Object**

```python
analysis: dict[str, Any]
```

Available keys include:

```text
metadata
duplicates
infinite
missing
outliers
feature_profiles
correlation_pairs
target_profile
recommendations
data_quality_score
data_quality_notes
```

---

### Generate a Summary

```python
print(analyzer.summary())
```

The summary provides a concise overview of the dataset, highlighting important quality issues and preprocessing recommendations.

---

### Notes

- The `target` parameter is optional.
- When no target is provided, all target-dependent analyses are skipped automatically.
- Statistical outputs are represented by strongly typed dataclasses.

---

## StatisticsEngine

### Overview

`StatisticsEngine` provides the statistical foundation for the toolkit.

Unlike `EDAAnalyzer`, it performs descriptive analysis only and does not generate preprocessing recommendations or quality scores.

It is intended for workflows that require statistical facts without higher-level interpretation.

---

### Highlights

- Computes dataset statistics.
- Provides independent methods for each analysis stage.
- Returns strongly typed dataclasses.
- Uses vectorized computations for improved performance.
- Designed to be lightweight and reusable.

---

### Example

```python
from preml.statistics_engine import StatisticsEngine

engine = StatisticsEngine(
    df,
    target="price",
)

stats = engine.run_full_analysis()
```

**Returned Object**

```python
stats: dict[str, Any]
```

The returned dictionary contains the same statistical outputs as `EDAAnalyzer`, excluding recommendations and quality metrics.

---

### Access Individual Reports

```python
missing = engine.compute_missing_report()

print(missing.total_missing)
```

Example:

```python
for report in missing.column_reports:
    print(
        report.column,
        report.missing_percent,
    )
```

---

### Available Analysis Methods

```python
engine.compute_dataset_metadata()

engine.compute_duplicate_report()

engine.compute_infinite_report()

engine.compute_missing_report()

engine.compute_outlier_report()

engine.compute_feature_profiles()

engine.compute_correlation_pairs()

engine.compute_target_profile()

engine.run_full_analysis()
```

---

### Notes

- The input DataFrame is copied during initialization to preserve the original dataset.
- For very large datasets, sampling before analysis is recommended.
- Use `EDAAnalyzer` whenever preprocessing recommendations or quality scoring are also required.

---

## RecommendationEngine

### Overview

`RecommendationEngine` converts statistical evidence into actionable machine learning recommendations.

It never recomputes statistics. Instead, it interprets the outputs generated by `EDAAnalyzer` or `StatisticsEngine`.

---

### Highlights

- Configurable decision thresholds.
- Missing-value recommendations.
- Outlier handling recommendations.
- Feature transformation suggestions.
- Scaling recommendations.
- Encoding recommendations.
- Feature engineering suggestions.
- Feature selection suggestions.
- Ranked baseline model recommendations.

---

### Example

```python
from preml.recommendation_engine import RecommendationEngine

engine = RecommendationEngine(
    config=config,
)

recommendations = engine.generate_recommendations(
    analysis,
)
```

**Returned Object**

```python
recommendations: dict[str, Any]
```

Available categories include:

```text
imputation
outlier_handling
transformation
scaling
encoding
feature_engineering
feature_selection
models
data_quality_notes
```

---

### Scaling Recommendation

```python
print(
    recommendations["scaling"].action
)
```

---

### Recommended Models

```python
for model in recommendations["models"]:
    print(
        f"{model.model_name:<30}"
        f"{model.suitability:<12}"
        f"{model.reason}"
    )
```

Example output:

```text
RandomForestClassifier      Excellent
XGBoostClassifier           Excellent
LogisticRegression          Baseline
```

---

### Generate a Formatted Summary

```python
print(
    RecommendationEngine.summarize(
        recommendations
    )
)
```

The generated summary is suitable for:

- Console output
- Markdown reports
- HTML reports
- Plain-text documentation
- Logging

---

### Notes

- Statistics are never recomputed.
- Recommendations are generated entirely from the supplied analysis results.
- Missing required analysis keys raise a `RecommendationError`.

## Visualization

### Overview

The visualization module generates publication-ready figures directly from precomputed analysis results.

Unlike traditional plotting utilities, visualization functions never perform statistical analysis internally. They reuse the analysis dictionary produced by `EDAAnalyzer`, ensuring consistency while avoiding unnecessary computation.

---

### Highlights

- Lightweight and efficient.
- No statistical recomputation.
- Returns standard Matplotlib figures.
- Supports automatic explanatory captions.
- Integrates seamlessly with `ReportGenerator`.

---

### Importing Visualization Functions

```python
from preml.visualization import (
    plot_numeric_distributions,
    plot_missing_heatmap,
    plot_correlation_heatmap,
    plot_outlier_summary,
    plot_target_distribution,
    plot_target_correlations,
    plot_top_correlations_bar,
    explain_visualizations,
)
```

---

### Numeric Feature Distributions

Visualize the distribution of numerical features using combined histograms and box plots.

```python
fig = plot_numeric_distributions(
    df,
    analysis,
    max_cols=8,
)

if fig:
    fig.savefig("distributions.png")
```

**Returns**

```python
matplotlib.figure.Figure | None
```

---

### Missing Values Heatmap

Visualize missing-value patterns across the dataset.

```python
fig = plot_missing_heatmap(df)

if fig:
    fig.savefig("missing.png")
```

---

### Correlation Heatmap

Display statistically significant feature correlations.

```python
fig = plot_correlation_heatmap(
    df,
    analysis,
)

if fig:
    fig.savefig("correlation.png")
```

---

### Outlier Summary

Summarize detected outliers across numeric features.

```python
fig = plot_outlier_summary(analysis)

if fig:
    fig.savefig("outliers.png")
```

---

### Target Distribution

Automatically adapts to regression or classification targets.

```python
fig = plot_target_distribution(
    df,
    analysis,
)

if fig:
    fig.savefig("target_distribution.png")
```

---

### Target Correlations

Display correlations between numerical features and the target variable.

```python
fig = plot_target_correlations(
    df,
    analysis,
    top_n=10,
)

if fig:
    fig.savefig("target_correlations.png")
```

---

### Top Feature Correlations

Display the strongest absolute correlations among numerical features.

```python
fig = plot_top_correlations_bar(
    analysis,
    top_n=10,
)

if fig:
    fig.savefig("top_correlations.png")
```

---

### Explanatory Captions

Generate human-readable explanations for each visualization.

```python
captions = explain_visualizations(
    analysis,
    recommendations,
)

print(
    captions["numeric_distributions"]
)
```

**Returns**

```python
dict[str, str]
```

Available keys include:

```text
numeric_distributions
target_distribution
correlation_heatmap
missing_values_heatmap
outlier_summary
target_correlations
```

---

### Notes

- All plotting functions optionally accept a configuration object for styling.
- HTML reports automatically embed generated captions.
- Every plotting function returns either a `Figure` or `None`.

---

## PreprocessingBuilder

### Overview

`PreprocessingBuilder` converts analysis results into a production-ready `scikit-learn` preprocessing pipeline.

All preprocessing decisions are based on statistical evidence rather than manually specified rules.

---

### Highlights

- Removes constant and quasi-constant features.
- Automatically imputes missing values.
- Applies power transformations to highly skewed features.
- Scales numerical variables appropriately.
- Encodes categorical features according to cardinality.
- Produces a standard `ColumnTransformer`.

---

### Example

```python
from preml.preprocessing import PreprocessingBuilder

builder = PreprocessingBuilder(
    analysis,
)

pipeline = builder.build_pipeline()
```

**Returns**

```python
sklearn.compose.ColumnTransformer
```

---

### Transform Features

```python
feature_cols = [
    p.column
    for p in analysis["feature_profiles"]
]

X = df[feature_cols]

X_transformed, = builder.fit_transform(X)
```

**Returns**

```python
tuple[np.ndarray]
```

The transformed feature matrix is obtained using tuple unpacking.

---

### Train a Model

```python
from sklearn.linear_model import LogisticRegression

y = df["target"]

model = LogisticRegression()

model.fit(
    X_transformed,
    y,
)
```

---

### Automatic Pipeline Decisions

The generated preprocessing pipeline automatically performs the following operations when appropriate:

| Operation | Strategy |
|-----------|----------|
| Constant features | Removed |
| Missing numerical values | Mean or Median |
| Missing categorical values | Most Frequent |
| Skewed numerical features | Yeo–Johnson transformation |
| Numerical scaling | `StandardScaler` or `RobustScaler` |
| Low-cardinality categorical features | `OneHotEncoder` |
| High-cardinality categorical features | `OrdinalEncoder` |

---

### Best Practices

- Always remove the target column before calling `fit_transform()`.
- Reuse the same preprocessing pipeline during training and inference.
- Serialize the fitted pipeline with `joblib` for deployment.

---

### Notes

- `fit_transform()` returns a one-element tuple.
- The generated pipeline is fully compatible with `scikit-learn`.
- Additional preprocessing steps can be appended using a standard `Pipeline`.

## FeatureEngineering

### Overview

`FeatureEngineering` analyzes statistical properties of the dataset and proposes meaningful feature engineering opportunities.

Suggestions are generated entirely from statistical evidence rather than feature names, making the recommendations domain-independent and reproducible.

---

### Highlights

- Data-driven feature suggestions.
- Ratio feature recommendations.
- Interaction feature suggestions.
- Numerical binning recommendations.
- Power transformation suggestions.
- Datetime decomposition.
- Categorical feature crossing.
- Dynamic confidence scoring.

---

### Example

```python
from preml.feature_engineering import FeatureEngineering

fe = FeatureEngineering(
    analysis,
    df=df,
)

suggestions = fe.suggest_features()
```

**Returns**

```python
list[Recommendation]
```

Each recommendation contains information such as:

- Category
- Action
- Confidence
- Reason

---

### Display Suggestions

```python
for suggestion in suggestions:
    print(
        f"[{suggestion.confidence:.0%}] "
        f"{suggestion.action}"
    )
```

Example output:

```text
[92%] Create a ratio between AnnualIncome and SpendingScore.

[74%] Apply a Yeo-Johnson transformation to TotalSales.

[68%] Extract Month and DayOfWeek from OrderDate.
```

---

### Best Practices

- Treat generated features as hypotheses rather than guaranteed improvements.
- Validate engineered features using a hold-out validation set.
- Evaluate feature importance before adding engineered variables to production pipelines.

---

### Notes

- Statistical information is reused from the analysis object.
- Datetime-related recommendations require the original DataFrame.
- Confidence scores are calculated dynamically based on the underlying statistics.

---

## BaselineTrainer

### Overview

`BaselineTrainer` builds complete machine learning pipelines and evaluates multiple baseline models using cross-validation.

It provides a fast way to establish reference model performance before hyperparameter tuning.

---

### Highlights

- Supports regression and classification tasks.
- Automatic preprocessing integration.
- Cross-validation.
- Multiple baseline algorithms.
- Metric computation helpers.
- Configurable evaluation strategy.

---

### Example

```python
from preml.model_utils import BaselineTrainer

trainer = BaselineTrainer(
    config=config,
)
```

---

### Build a Model Pipeline

```python
target_profile = analysis["target_profile"]

task_type = (
    "regression"
    if target_profile.is_regression
    else "classification"
)

model_pipeline = trainer.build_model_pipeline(
    preprocessing_pipeline=pipeline,
    task_type=task_type,
)
```

---

### Evaluate a Baseline Model

```python
evaluation = trainer.evaluate_baseline(
    model_pipeline,
    X=X,
    y=y,
    task_type=task_type,
    cv=5,
)

print(
    evaluation["mean_scores"]
)
```

---

### Train Recommended Models

```python
results = trainer.train_baselines(
    analysis,
    df=df,
    target_col="target",
    preprocessing_pipeline=pipeline,
    cv=5,
)

for result in results:
    print(
        result["model_name"],
        result["mean_scores"],
    )
```

---

### Compute Metrics

```python
from preml.model_utils import compute_metrics

metrics = compute_metrics(
    y_true,
    y_pred,
    task_type="regression",
)
```

---

### Cross Validation Helper

```python
from preml.model_utils import cross_validate

scores = cross_validate(
    estimator,
    X,
    y,
    cv=5,
    scoring=[
        "accuracy",
        "f1_macro",
    ],
)
```

---

### Notes

- `train_baselines()` requires the original DataFrame, including the target column.
- Model pipelines are fully compatible with `scikit-learn`.
- Use a fixed `random_state` to ensure reproducible experiments.

---

## ReportGenerator

### Overview

`ReportGenerator` creates comprehensive reports from analysis results.

Reports combine statistics, recommendations, visualizations, and explanatory text into a single document suitable for sharing with stakeholders or documenting experiments.

---

### Highlights

- HTML reports.
- Markdown reports.
- Plain-text reports.
- Embedded visualizations.
- Automatic explanatory captions.
- Executive summary.
- Data quality score.
- Recommendation summary.

---

### Example

```python
from preml.report import ReportGenerator

report = ReportGenerator(
    analysis,
    df=df,
)
```

---

### Save Reports

```python
report.save_report(
    "report.html",
    format="html",
)

report.save_report(
    "report.md",
    format="md",
)

report.save_report(
    "report.txt",
    format="txt",
)
```

---

### Generate Raw Content

```python
html = report.generate_html(
    embed_plots=True,
)

markdown = report.generate_markdown()

text = report.generate_text()
```

**Returns**

```python
str
```

---

### HTML Report Contents

Generated HTML reports include:

- Executive summary.
- Dataset metadata.
- Missing value analysis.
- Duplicate analysis.
- Outlier analysis.
- Correlation analysis.
- Target profile.
- Data quality score.
- Evidence-based recommendations.
- Ranked model suggestions.
- Embedded visualizations.
- Automatically generated explanatory captions.

---

### Notes

- If the original DataFrame is unavailable, statistical information is still included while visualizations are omitted.
- Visualization captions are generated automatically through `explain_visualizations()`.
- HTML reports are fully self-contained and suitable for sharing without additional resources.

---

## Next Section

The following sections cover:

- End-to-End Workflow
- Performance
- Thread Safety
- Reproducibility
- Best Practices
- Common Pitfalls
- FAQ
- Component Relationships
- Architecture
- API Reference
- License
- Citation

# End-to-End Workflow

The following example demonstrates a complete machine learning workflow using PreML.

```python
import pandas as pd

from preml.eda import EDAAnalyzer
from preml.preprocessing import PreprocessingBuilder
from preml.feature_engineering import FeatureEngineering
from preml.model_utils import BaselineTrainer
from preml.report import ReportGenerator

# Load the dataset
df = pd.read_csv("data.csv")
target = "target_column"

# 1. Exploratory Data Analysis
analysis = EDAAnalyzer(
    df,
    target=target,
).run()

# 2. Build preprocessing pipeline
builder = PreprocessingBuilder(analysis)
pipeline = builder.build_pipeline()

feature_cols = [
    profile.column
    for profile in analysis["feature_profiles"]
]

X_transformed, = builder.fit_transform(
    df[feature_cols]
)

# 3. Feature engineering suggestions
feature_engineering = FeatureEngineering(
    analysis,
    df=df,
)

for suggestion in feature_engineering.suggest_features():
    print(suggestion.action)

# 4. Train baseline models
trainer = BaselineTrainer()

results = trainer.train_baselines(
    analysis,
    df=df,
    target_col=target,
    preprocessing_pipeline=pipeline,
    cv=5,
)

# 5. Generate report
ReportGenerator(
    analysis,
    df=df,
).save_report(
    "report.html",
    format="html",
)
```

---

# Performance

PreML is designed to minimize unnecessary computation while remaining fully transparent.

## Performance Characteristics

- Statistics are computed once and reused throughout the workflow.
- Visualization functions never recompute statistics.
- Report generation reuses the existing analysis results.
- Vectorized NumPy and pandas operations are used whenever possible.
- Percentile calculations are optimized for numerical features.
- Correlation analysis avoids redundant computations.

---

# Thread Safety

Each component is independent after construction.

Parallel workflows are supported provided that each thread owns its own analyzer or statistics engine instance.

Sharing a read-only analysis dictionary between threads is safe.

Avoid modifying the analysis dictionary concurrently.

---

# Reproducibility

For deterministic experiments, specify a fixed `random_state` in `MLToolkitConfig`.

```python
from preml.config import MLToolkitConfig

config = MLToolkitConfig(
    random_state=42,
)
```

The statistics engine itself is deterministic and performs no random sampling.

---

# Best Practices

- Reuse a single `MLToolkitConfig` instance throughout the workflow.
- Sample extremely large datasets before running analysis.
- Validate engineered features on an independent validation set.
- Exclude the target column before preprocessing.
- Check whether visualization functions return `None`.
- Reuse fitted preprocessing pipelines during inference.
- Persist trained pipelines using `joblib`.

---

# Common Pitfalls

## `fit_transform()` Returns a Tuple

```python
# Correct
X_transformed, = builder.fit_transform(X)
```

---

## `train_baselines()` Requires the Original DataFrame

```python
trainer.train_baselines(
    analysis,
    df=df,
    target_col="target",
)
```

Do not remove the target column before calling this method.

---

## Plot Functions May Return `None`

```python
fig = plot_missing_heatmap(df)

if fig:
    fig.savefig("missing.png")
```

---

## Missing Scaling Recommendation

Access recommendations directly.

```python
analysis["recommendations"]["scaling"]
```

---

# FAQ

## Why does `fit_transform()` return a tuple?

The preprocessing builder reserves the ability to return additional artifacts in future releases while maintaining backward compatibility.

---

## Why are some plots empty?

Visualization functions return `None` whenever no meaningful visualization can be generated.

---

## Can I use only the statistical analysis?

Yes.

Use `StatisticsEngine` instead of `EDAAnalyzer`.

---

## Can I customize the preprocessing pipeline?

Yes.

`PreprocessingBuilder` produces a standard `scikit-learn` `ColumnTransformer` that can be extended using `Pipeline`.

---

# Component Relationships

| Component | Depends On |
|-----------|------------|
| `StatisticsEngine` | DataFrame |
| `RecommendationEngine` | Analysis dictionary |
| `Visualization` | Analysis dictionary + DataFrame |
| `PreprocessingBuilder` | Analysis dictionary |
| `FeatureEngineering` | Analysis dictionary (+ optional DataFrame) |
| `BaselineTrainer` | Analysis dictionary + preprocessing pipeline |
| `ReportGenerator` | Analysis dictionary (+ optional DataFrame) |

---

# Architecture

```text
                    Dataset
                        │
                        ▼
                 EDAAnalyzer
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   Statistics    Recommendations   Quality Score
        │               │
        └───────────────┘
                │
                ▼
      PreprocessingBuilder
                │
                ▼
      FeatureEngineering
                │
                ▼
        BaselineTrainer
                │
                ▼
        ReportGenerator
```

---

# API Reference

| Module | Main Classes / Functions |
|---------|--------------------------|
| `preml.eda` | `EDAAnalyzer`, `quick_eda` |
| `preml.statistics_engine` | `StatisticsEngine` |
| `preml.recommendation_engine` | `RecommendationEngine` |
| `preml.visualization` | Plotting functions, `explain_visualizations()` |
| `preml.preprocessing` | `PreprocessingBuilder` |
| `preml.feature_engineering` | `FeatureEngineering` |
| `preml.model_utils` | `BaselineTrainer`, `compute_metrics()`, `cross_validate()` |
| `preml.report` | `ReportGenerator` |
| `preml.config` | `MLToolkitConfig`, `default_config` |
| `preml.exceptions` | All toolkit exceptions |

For detailed information about every class and method, refer to the source docstrings.

---

# License

PreML is released under the **MIT License**.

See the `LICENSE` file for complete licensing information.

---

# Citation

If PreML contributes to your research, publication, or production system, please consider citing the project.

```bibtex
@software{preml,
  author    = {Ali Nazer},
  title     = {PreML: Automated EDA, Preprocessing, and Baseline Modeling},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/alinazer30/preml}
}
```
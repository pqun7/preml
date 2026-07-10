# ML Toolkit — Complete Usage Guide

> **ML Toolkit** is a Python library that automates the most repetitive parts of traditional machine learning workflows. It performs Exploratory Data Analysis (EDA), generates evidence-based recommendations, builds production-ready preprocessing pipelines, suggests feature engineering opportunities, trains baseline models, and produces professional reports.

---

# Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Configuration](#configuration)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Statistics Engine](#statistics-engine)
* [Recommendation Engine](#recommendation-engine)
* [Visualization](#visualization)
* [Preprocessing](#preprocessing)
* [Feature Engineering](#feature-engineering)
* [Model Utilities](#model-utilities)
* [Reporting](#reporting)
* [Complete Workflow Example](#complete-workflow-example)
* [Exception Handling](#exception-handling)
* [Common Pitfalls](#common-pitfalls)
* [Advanced Tips](#advanced-tips)
* [Recommended Workflow](#recommended-workflow)

---

# Overview

ML Toolkit provides an end-to-end workflow for classical machine learning projects.

It helps you:

* Perform automated Exploratory Data Analysis (EDA)
* Generate data-driven recommendations
* Build preprocessing pipelines automatically
* Suggest statistically supported feature engineering
* Train baseline machine learning models
* Generate Markdown, HTML, and plain-text reports

Each module can be used independently or combined into a complete machine learning pipeline.

---

# Installation

## Install from PyPI

```bash
pip install ml-toolkit
```

## Install from Source

```bash
git clone https://github.com/alinazer30/ml-toolkit.git

cd ml-toolkit

pip install -e .
```

---

# Configuration

All configurable thresholds and runtime settings are managed through `MLToolkitConfig`.

## Use the Default Configuration

```python
from ml_toolkit.config import default_config

config = default_config
```

## Create a Custom Configuration

```python
from ml_toolkit.config import MLToolkitConfig

config = MLToolkitConfig(
    missing_threshold=0.30,
    correlation_threshold=0.85,
    outlier_method="iqr",   # "iqr" or "zscore"
    random_state=42,
)
```

## Pass Configuration to Components

```python
from ml_toolkit.eda import EDAAnalyzer

analyzer = EDAAnalyzer(
    df,
    config=config,
)
```

---

# Exploratory Data Analysis (EDA)

`EDAAnalyzer` orchestrates the complete analysis pipeline.

## What It Computes

* Dataset metadata
* Missing values
* Duplicate rows
* Infinite values
* Outliers
* Feature profiles
* Correlation analysis
* Target analysis
* Machine learning recommendations
* Data quality score

---

## Run a Complete Analysis

```python
from ml_toolkit.eda import EDAAnalyzer

analyzer = EDAAnalyzer(
    housing_df,
    target="MedHouseVal",
)

analysis = analyzer.run()
```

---

## Quick Helper

```python
from ml_toolkit.eda import quick_eda

analysis = quick_eda(
    df,
    target="price",
)
```

---

## Available Analysis Outputs

```python
analysis.keys()
```

```python
dict_keys([
    "metadata",
    "duplicates",
    "infinite",
    "missing",
    "outliers",
    "feature_profiles",
    "correlation_pairs",
    "target_profile",
    "recommendations",
    "data_quality_score",
    "data_quality_notes",
])
```

---

## Generate a Text Summary

```python
print(analyzer.summary())
```

---

# Statistics Engine

Use `StatisticsEngine` when you only need statistical analysis without ML recommendations.

```python
from ml_toolkit.statistics_engine import StatisticsEngine

engine = StatisticsEngine(
    df,
    target="price",
)
```

## Run Individual Analyses

```python
metadata = engine.compute_dataset_metadata()
duplicates = engine.compute_duplicate_report()
missing = engine.compute_missing_report()
outliers = engine.compute_outlier_report()
profiles = engine.compute_feature_profiles()
corr_pairs = engine.compute_correlation_pairs()
target_prof = engine.compute_target_profile()
```

## Run the Entire Analysis

```python
all_stats = engine.run_full_analysis()
```

All returned objects are strongly typed dataclasses, including:

* `MissingReport`
* `OutlierReport`
* `NumericDistributionProfile`

---

# Recommendation Engine

`RecommendationEngine` transforms statistical findings into actionable machine learning recommendations.

```python
from ml_toolkit.recommendation_engine import RecommendationEngine

engine = RecommendationEngine(config=config)

recommendations = engine.generate_recommendations(analysis)
```

## Recommendation Categories

* Imputation
* Scaling
* Encoding
* Feature Engineering
* Feature Selection
* Model Selection

---

## Example: Scaling Recommendation

```python
scaling = recommendations["scaling"]

print(scaling.action)
```

---

## Example: Recommended Models

```python
for model in recommendations["models"]:
    print(
        model.model_name,
        model.suitability,
        model.reason,
    )
```

---

# Visualization

Visualization functions consume existing EDA results.

> **Important**
>
> These functions **never recompute statistics**, making them lightweight and efficient.

```python
from ml_toolkit.visualization import (
    plot_numeric_distributions,
    plot_missing_heatmap,
    plot_correlation_heatmap,
    plot_outlier_summary,
    plot_target_distribution,
    plot_target_correlations,
    plot_top_correlations_bar,
)
```

---

## Numeric Distributions

```python
fig = plot_numeric_distributions(
    df,
    analysis,
    max_cols=10,
)

if fig:
    fig.savefig("distributions.png")
```

---

## Missing Values Heatmap

```python
fig = plot_missing_heatmap(df)

if fig:
    fig.savefig("missing_heatmap.png")
```

---

## Correlation Heatmap

```python
fig = plot_correlation_heatmap(
    df,
    analysis,
)

if fig:
    fig.savefig("corr_heatmap.png")
```

---

## Outlier Summary

```python
fig = plot_outlier_summary(analysis)

if fig:
    fig.savefig("outliers.png")
```

---

## Target Distribution

```python
fig = plot_target_distribution(
    df,
    analysis,
)

if fig:
    fig.savefig("target.png")
```

---

## Feature–Target Correlations

```python
fig = plot_target_correlations(
    df,
    analysis,
    top_n=10,
)

if fig:
    fig.savefig("target_corrs.png")
```

---

## Strongest Feature Correlations

```python
fig = plot_top_correlations_bar(
    analysis,
    top_n=10,
)

if fig:
    fig.savefig("top_corrs.png")
```

> All visualization functions accept an optional `ax` parameter and return either:
>
> * `matplotlib.figure.Figure`
> * `None` (when there is nothing meaningful to visualize)

---

# Preprocessing

`PreprocessingBuilder` converts EDA results into a production-ready Scikit-learn `ColumnTransformer`.

```python
from ml_toolkit.preprocessing import PreprocessingBuilder

builder = PreprocessingBuilder(analysis)

pipeline = builder.build_pipeline()
```

---

## Important: `fit_transform()` Returns a Tuple

Current versions return:

```python
(transformed_array,)
```

instead of:

```python
numpy.ndarray
```

### Correct Usage

```python
X_transformed, = builder.fit_transform(feature_df)
```

or

```python
X_transformed = builder.fit_transform(feature_df)[0]
```

---

## Always Remove the Target Column

The preprocessing pipeline should receive **only feature columns**.

```python
feature_cols = [
    p.column
    for p in analysis["feature_profiles"]
]

X = df[feature_cols]

X_transformed, = builder.fit_transform(X)
```

---

## Train a Scikit-learn Model

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

## Automatic Preprocessing Steps

The generated pipeline automatically:

* Removes constant features
* Removes quasi-constant features
* Handles missing values
* Detects categorical-like numeric columns
* Applies power transformations
* Scales numerical features
* Encodes categorical features

---

## Missing Value Strategy

| Feature Type               | Strategy      |
| -------------------------- | ------------- |
| Numeric (with outliers)    | Median        |
| Numeric (without outliers) | Mean          |
| Categorical                | Most Frequent |

---

## Scaling Strategy

| Condition        | Scaler           |
| ---------------- | ---------------- |
| Outliers Present | `RobustScaler`   |
| Otherwise        | `StandardScaler` |

---

## Encoding Strategy

| Category Type    | Encoder          |
| ---------------- | ---------------- |
| Low Cardinality  | `OneHotEncoder`  |
| High Cardinality | `OrdinalEncoder` |

---

# Feature Engineering

`FeatureEngineering` analyzes statistical evidence and suggests engineered features.

```python
from ml_toolkit.feature_engineering import FeatureEngineering

fe = FeatureEngineering(
    analysis,
    df=df,
)

suggestions = fe.suggest_features()

for suggestion in suggestions:
    print(
        suggestion.action,
        suggestion.confidence,
    )
```

## Possible Suggestions

* Ratio Features
* Interaction Features
* Feature Binning
* Power Transformations
* Datetime Decomposition
* Crossed Categorical Variables

> **Note**
>
> Feature engineering suggestions are statistically motivated recommendations.
> Always validate them on a validation set before using them in production.

---

# Model Utilities

`BaselineTrainer` builds complete machine learning pipelines and evaluates them using cross-validation.

```python
from ml_toolkit.model_utils import BaselineTrainer

trainer = BaselineTrainer(config=config)
```

---

## Build a Model Pipeline

```python
target_prof = analysis["target_profile"]

task_type = (
    "regression"
    if target_prof.is_regression
    else "classification"
)

model_pipeline = trainer.build_model_pipeline(
    preprocessing_pipeline=pipeline,
    task_type=task_type,
)
```

---

## Evaluate a Baseline Model

```python
X = df.drop(columns=["target"])
y = df["target"]

evaluation = trainer.evaluate_baseline(
    model_pipeline,
    X=X,
    y=y,
    task_type=task_type,
    cv=5,
)

print(evaluation["mean_scores"])
```

---

## Train Recommended Baseline Models

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

## Compute Metrics

```python
from ml_toolkit.model_utils import compute_metrics

metrics = compute_metrics(
    y_true,
    y_pred,
    task_type="regression",
)
```

---

## Custom Metrics

```python
from sklearn.metrics import f1_score

def f1_macro(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
    )

metrics = compute_metrics(
    y_true,
    y_pred,
    task_type="classification",
    extra_metrics={
        "f1_macro": f1_macro,
    },
)
```

---

## Cross Validation

```python
from ml_toolkit.model_utils import cross_validate

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

# Reporting

`ReportGenerator` creates professional reports directly from EDA results.

```python
from ml_toolkit.report import ReportGenerator

report = ReportGenerator(
    analysis,
    df=df,
)
```

---

## Plain Text

```python
text = report.generate_text()

print(text)
```

---

## Markdown

```python
markdown = report.generate_markdown()

with open("report.md", "w") as f:
    f.write(markdown)
```

---

## HTML

```python
html = report.generate_html(
    embed_plots=True,
)

with open(
    "report.html",
    "w",
    encoding="utf-8",
) as f:
    f.write(html)
```

---

## Save Automatically

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

### Generated HTML Includes

* Dataset Statistics
* Recommendations
* Correlation Heatmaps
* Missing Value Heatmaps
* Distribution Plots
* Data Quality Summary

---

# Complete Workflow Example

```python
import pandas as pd

from ml_toolkit.eda import EDAAnalyzer
from ml_toolkit.preprocessing import PreprocessingBuilder
from ml_toolkit.model_utils import BaselineTrainer
from ml_toolkit.report import ReportGenerator

# Load dataset
df = pd.read_csv("housing.csv")
target = "SalePrice"

# Run EDA
analyzer = EDAAnalyzer(df, target=target)
analysis = analyzer.run()

print(
    "Data quality:",
    analysis["data_quality_score"],
)

# Build preprocessing pipeline
builder = PreprocessingBuilder(analysis)
preprocessor = builder.build_pipeline()

feature_cols = [
    p.column
    for p in analysis["feature_profiles"]
]

X_df = df[feature_cols]

X_transformed, = builder.fit_transform(X_df)

# Train baseline models
trainer = BaselineTrainer()

results = trainer.train_baselines(
    analysis,
    df=df,
    target_col=target,
    preprocessing_pipeline=preprocessor,
    cv=5,
)

for result in results:
    print(
        result["model_name"],
        result["mean_scores"],
    )

# Generate report
report = ReportGenerator(
    analysis,
    df=df,
)

report.save_report(
    "housing_report.html",
    format="html",
)
```

---

# Exception Handling

All custom exceptions inherit from `MLToolkitError`.

```python
from ml_toolkit.exceptions import (
    MLToolkitError,
    DataValidationError,
    StatisticsError,
    PreprocessingError,
    ModelError,
    ReportError,
    VisualizationError,
)
```

## Example

```python
try:
    analysis = quick_eda(df)

except DataValidationError as e:
    print("Invalid data:", e)

except MLToolkitError as e:
    print("ML Toolkit error:", e)
```

---

# Common Pitfalls

## 1. `fit_transform()` Returns a Tuple

**Problem**

```python
builder.fit_transform(df)
```

returns:

```python
(array,)
```

instead of a NumPy array.

**Solution**

```python
X_transformed, = builder.fit_transform(df)
```

or

```python
X_transformed = builder.fit_transform(df)[0]
```

---

## 2. Only Pass Feature Columns

Always remove the target column.

```python
feature_cols = [
    p.column
    for p in analysis["feature_profiles"]
]

X = df[feature_cols]

X_transformed, = builder.fit_transform(X)
```

---

## 3. `train_baselines()` Requires the Full DataFrame

Correct:

```python
trainer.train_baselines(
    analysis,
    df=df,
    target_col="target",
)
```

Incorrect:

```python
trainer.train_baselines(
    analysis,
    df=df.drop(columns=["target"]),
    target_col="target",
)
```

---

## 4. Plot Functions May Return `None`

```python
fig = plot_missing_heatmap(df)

if fig:
    fig.savefig("missing.png")
```

---

## 5. Missing Scaling Recommendation in `summary()`

In some cases, `EDAAnalyzer.summary()` may omit scaling recommendations.

Instead, access them directly:

```python
analysis["recommendations"]["scaling"]
```

---

# Advanced Tips

## Customize the Preprocessing Pipeline

The generated pipeline is a standard Scikit-learn `ColumnTransformer`.

Feel free to extend it with custom transformers before model training.

---

## Validate Feature Engineering Suggestions

Always verify suggested engineered features on your validation dataset before deployment.

---

## Reuse a Single Configuration

Create one `MLToolkitConfig` instance and reuse it across the entire workflow for consistent behavior.

---

## Working with Large Datasets

`StatisticsEngine` works on a copy of the DataFrame.

For very large datasets:

* Sample the data before running EDA.
* Reduce memory usage.
* Improve execution time.

---

# Recommended Workflow

```text
Dataset
   │
   ▼
EDAAnalyzer
   │
   ▼
Recommendations
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

# Design Philosophy

ML Toolkit automates the repetitive aspects of machine learning while keeping every recommendation:

* Transparent
* Interpretable
* Statistically grounded
* Production-oriented
* Easy to customize

This balance enables faster experimentation without sacrificing explainability or engineering best practices.

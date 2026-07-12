![CI](https://github.com/pqun7/preml/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/pypreml.svg)
![Python](https://img.shields.io/pypi/pyversions/pypreml.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
# PreML

Production-ready machine learning toolkit for EDA, preprocessing, and modelling.

## 📌 Features
- Fast and automated Exploratory Data Analysis (EDA).
- Robust preprocessing pipelines.
- Production-ready machine learning modelling wrappers.
- sklearn-style preprocessing lifecycle (`fit`, `transform`, `fit_transform`).

## ⚙️ Installation

You can install `pypreml` via pip:

```bash
pip install pypreml
```


PreML is a modular Python library for exploratory data analysis, statistical recommendations, preprocessing pipeline generation, and feature engineering guidance for tabular datasets.

The recommended entry point is the high-level facade:

```python
from preml import PreML

ml = PreML(df, target="target_column")
analysis = ml.analyze()
report = ml.report()
pipeline = ml.pipeline()
```

The source code now lives inside the `preml/` package directory, which keeps the project organized and matches the installed namespace.

## What This Project Does

- Computes statistical facts from tabular data.
- Turns those facts into evidence-based recommendations.
- Builds scikit-learn compatible preprocessing pipelines.
- Suggests feature engineering ideas from measurable patterns.
- Keeps analysis, recommendations, and preprocessing separated into focused modules.

## Quick Start

Use this map to understand how the current files fit together:

| File | Purpose |
from preml import PreML, analyze
| [preml/config.py](preml/config.py) | Central thresholds and defaults via `MLToolkitConfig`. |
| [preml/statistics_engine.py](preml/statistics_engine.py) | Computes dataset facts such as profiles, missingness, outliers, and correlations. |
| [preml/recommendation_engine.py](preml/recommendation_engine.py) | Converts statistics into recommendations and model guidance. |
ml = PreML(df, target="target_column")
analysis = ml.analyze()
print(ml.summary())

# Or use the one-line helper for immediate analysis
analysis2 = analyze(df, target="target_column")

builder = ml.pipeline()
X_train = df.drop(columns=["target_column"])
X = builder.fit_transform(X_train)
- Feature engineering suggestions grounded in statistical evidence rather than column names.
- Clean module boundaries that make the package easier to test, extend, and maintain.

## Installation

### From source

```bash
git clone https://github.com/pqun7/preml.git
cd preml
python -m pip install -e .
```

### Dependencies

The project targets Python 3.9+ and uses:

- pandas
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn

## Quick Start

```python
import pandas as pd

from preml.eda import EDAAnalyzer, quick_eda
from preml.preprocessing import PreprocessingBuilder

df = pd.read_csv("your_dataset.csv")

# Run the full analysis
analysis = quick_eda(df, target="target_column")

# Or use the orchestrator directly
analyzer = EDAAnalyzer(df, target="target_column")
analysis = analyzer.run()
print(analyzer.summary())

# Build a preprocessing pipeline from the analysis output
builder = PreprocessingBuilder(analysis)
X_train = df.drop(columns=["target_column"])
pipeline = builder.build_pipeline()
X = builder.fit_transform(X_train)

# Later, transform new data using the fitted builder
# X_new = builder.transform(new_features_df)
```

## Typical Workflow

1. Load your tabular dataset into a pandas DataFrame.
2. Start with `PreML(df, target=...)` or `analyze(df, target=...)`.
3. Use `ml.pipeline()`, `ml.recommendations()`, and `ml.report()` as needed.
4. Drop down to `EDAAnalyzer`, `StatisticsEngine`, or `RecommendationEngine` only for advanced workflows.

## Public API

The package exposes the most common shared types and facade helpers at the package root:

```python
from preml import PreML, analyze, quick_eda
from preml import MLToolkitConfig, default_config
from preml import DataValidationError, RecommendationError
from preml import FeatureProfile, Recommendation, TargetProfile
```

Useful module entry points:

- `preml.PreML`
- `preml.analyze`
- `preml.eda.EDAAnalyzer`
- `preml.eda.quick_eda`
- `preml.preprocessing.PreprocessingBuilder`
- `preml.recommendation_engine.RecommendationEngine`
- `preml.statistics_engine.StatisticsEngine`

## Configuration

All thresholds and defaults are defined in `MLToolkitConfig`.

```python
from preml.config import MLToolkitConfig
from preml.eda import EDAAnalyzer

config = MLToolkitConfig(
    missing_threshold=0.2,
    correlation_threshold=0.85,
    random_state=42,
    n_jobs=-1,
)

analyzer = EDAAnalyzer(df, target="target_column", config=config)
analysis = analyzer.run()
```

Key configuration values include:

- `missing_threshold`
- `high_cardinality_threshold`
- `max_unique_for_categorical_like`
- `correlation_threshold`
- `skewness_threshold`
- `outlier_method`
- `iqr_multiplier`
- `random_state`

## GitHub Workflow

Recommended commands to publish the project cleanly:

```bash
git init
git add README.md pyproject.toml requirements.txt .gitignore preml/ tests/
git status
git commit -m "Prepare packaged ML toolkit"
git branch -M main
git remote add origin https://github.com/<your-username>/preml.git
git push -u origin main
```

Best practices when pushing:

- Commit only the files you intended to change.
- Run the test suite before pushing.
- Keep the README synchronized with the real repository layout.
- Avoid committing virtual environments, caches, notebooks, and build artifacts.
- Prefer small, descriptive commits such as `docs: reorganize README` or `fix: align packaging layout`.

## Development

Run the test suite with:

```bash
pytest tests/
```

If you want to sanity-check the edited modules locally, you can also run:

```bash
python -m py_compile preml/preprocessing.py preml/eda.py preml/recommendation_engine.py
```

## Contributing

Contributions are welcome. Keep changes aligned with the existing architecture, preserve the separation between facts and recommendations, and add tests when you change behaviour.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Security

Please report vulnerabilities according to [SECURITY.md](SECURITY.md).

## Changelog

Release history is documented in [CHANGELOG.md](CHANGELOG.md).

## Code of Conduct

Community standards are documented in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT License.

## Author

Ali Nazer – alinazer30@gmail.com

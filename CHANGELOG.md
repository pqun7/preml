# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [0.1.8] - 2026-07-11

### Added

- Added a dedicated executable test suite for validating Usage Guide examples end to end.
- Added a small Usage Guide test README with commands for running the validator-focused tests.

### Changed

- Clarified Usage Guide guidance for `RecommendationEngine.fit()` and documented `ValidationTimeoutError` handling.
- Added fallback behavior to the empirical model-selection example so it remains copy-paste runnable when the time budget is exceeded.
- Adjusted recommendation-engine timeout budgeting to avoid premature screening timeouts on documented examples.

### Fixed

- Replaced hidden file-based sample data in Usage Guide examples with inline synthetic data.
- Corrected Usage Guide field names for feature profiles and correlation pairs.
- Corrected Usage Guide evaluation output key usage to `cv_scores`.

## [0.1.7] - 2026-07-11

### Changed

- Aligned runtime package version metadata with the published `pyproject.toml` version.
- Clarified the `RecommendationEngine.fit(..., progress_callback=...)` callback contract in the public API docs.

## [0.1.6] - 2026-07-11

### Changed

- Synchronized Usage Guide examples with the empirical `RecommendationEngine.fit()` workflow.
- Expanded recommendation-engine documentation to show both descriptive and empirical usage paths.
- Bumped package version metadata for the new documentation release.

## [0.1.5] - 2026-07-11

### Added

- Added release audit closure document (`AUDIT_CLOSURE.md`) for issue-to-evidence traceability.
- Added executable documentation workflow tests to prevent docs/API drift.
- Added issue and pull request templates for contribution quality.
- Added `fit` and `transform` lifecycle methods to `PreprocessingBuilder`.
- Added adaptive config method `MLToolkitConfig.adapt_to_dataset(df)`.
- Added config field `n_jobs` for parallel model evaluation controls.
- Added CI workflow for multi-version Python test validation.
- Added contributing and governance documents.

### Changed

- Finalized release documentation for API-stable workflows in README and Usage Guide.
- Standardized packaging license metadata to modern `pyproject.toml` format.
- Improved release hygiene by tightening generated artifact ignore rules.
- Improved defaults for missingness, correlation, and skewness thresholds.
- Improved cross-validation splitter behavior to avoid integer-regression misclassification.
- Improved user-facing validation and error guidance in key modules.
- Updated dependency version bounds for compatibility clarity.
- Synchronized runtime package metadata with project metadata.

### Fixed

- Removed packaging deprecation warnings related to license metadata format.
- Eliminated visualization test warning source in outlier summary layout.
- Fixed recommendation input validation message key mismatch (`outliers` vs `outlier_reports`).
- Fixed visualization docstring contamination in `explain_visualizations`.
- Corrected repository URL inconsistency in documentation.

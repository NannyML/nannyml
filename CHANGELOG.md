# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased
### Added
- Added support for both predicted labels and predicted probabilities in ``ModelMetadata``.
- Support for monitoring model performance metrics using the ``PerformanceCalculator``.
- Support for monitoring target distribution using the ``TargetDistributionCalculator``

### Changed
- Plotting will default to using step plots.
- Restructured the ``nannyml.drift`` package and subpackages. *Breaking changes*!
- Metadata completeness check will now fail when there are features of ``FeatureType.UNKNOWN``.
- Chunk date boundaries are now calculated differently for a ``PeriodBasedChunker``, using the
  theoretical period for boundaries as opposed to the observed boundaries within the chunk observations.
- Updated version of the ``black`` pre-commit hook due to breaking changes in its ``click`` dependency.
- The *minimum chunk size* will now be provided by each individual ``calculator`` / ``estimator`` / ``metric``,
  allowing for each of them to warn the end user when chunk sizes are suboptimal.

### Fixed
- Restrict version of the ``scipy`` dependency to be ``>=1.7.3, <1.8.0``. Planned to be relaxed ASAP.
- Deal with missing values in chunks causing ``NaN`` values when concatenating.
- Crash when estimating CBPE without a target column present
- Incorrect label in ``ModelMetadata`` printout

## [0.2.1] - 2022-03-22
### Changed
- Allow calculators/estimators to provide appropriate ``min_chunk_size`` upon splitting into ``chunks``.

### Fixed
- Data reconstruction drift calculation failing when there are no categorical or continuous features
  [(#36)](https://github.com/NannyML/nannyml/issues/36)
- Incorrect scaling on continuous feature distribution plot [(#39)](https://github.com/NannyML/nannyml/issues/39)
- Missing ``needs_calibration`` checks before performing score calibration in CBPE
- Fix crash on chunking when missing target values in reference data

## [0.2.0] - 2022-03-03
### Added
- Result classes for Calculators and Estimators.
### Changed
- Updated the documentation to reflect the changes introduced by result classes,
  specifically to plotting functionality.
- Add support for imputing of missing values in the ``DataReconstructionDriftCalculator``.
### Removed
- ``nannyml.plots.plots`` was removed.
  Plotting is now meant to be done using ``DriftResult.plot()`` or ``EstimatorResult.plot()``.


## [0.1.1] - 2022-03-03
### Fixed
- Fixed an issue where data reconstruction drift calculation also used model predictions during decomposition.


## [0.1.0] - 2022-03-03
### Added
- Chunking base classes and implementations
- Metadata definitions and utilities
- Drift calculator base classes and implementations
  - Univariate statistical drift calculator
  - Multivariate data reconstruction drift calculator
- Drifted feature ranking base classes and implementations
  - Alert count based ranking
- Performance estimator base classes and implementations
  - Certainty based performance estimator
- Plotting utilities with support for
  - Stacked bar plots
  - Line plots
  - Joy plots
- Documentation
  - Quick start guide
  - User guides
  - Deep dives
  - Example notebooks
  - Technical reference documentation

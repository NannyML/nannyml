# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.1] - 2022-03-22
### Fixed
- Data reconstruction drift calculation failing when there are no categorical or continuous features
  [(#36)](https://github.com/NannyML/nannyml/issues/36)
- Incorrect scaling on continuous feature distribution plot [(#39)](https://github.com/NannyML/nannyml/issues/39)
- Missing ``needs_calibration`` checks before performing score calibration in CBPE
- Fix crash on chunking when missing target values in reference data

### Changed
- Allow calculators/estimators to provide appropriate ``min_chunk_size`` upon splitting into ``chunks``.


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

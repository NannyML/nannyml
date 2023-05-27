# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.6] - 2023-05-24

### Changed

- Significant QA work on all the documentation, thanks [@santiviquez](https://github.com/santiviquez) and
  [@maciejbalawejder](https://github.com/maciejbalawejder)
- Reworked the [`nannyml.runner`](nannyml/runner.py) and the accompanying configuration format to improve flexibility (e.g. setting
  custom initialization parameters, running a calculator multiple times, excluding a calculator, ...).
- Added support for custom thresholds to the [`nannyml.runner`](nannyml/runner.py)
- Simplified some of the `nannyml.io` interfaces, especially the [`nannyml.io.RawFilesWriter`](nannyml/io/raw_files_writer.py)
- Reworked the [`nannyml.base.Result`](nannyml/base.py)
- Totally revamped [quickstart documentation](docs/quick.rst) based on a real life dataset, thanks [@jakubnml](https://github.com/jakubnml)

### Added

- Added new calculators to support simple data quality metrics such as counting missing or unseen values.
  For more information, check out the [data quality tutorials](https://nannyml.readthedocs.io/en/main/tutorials/data_quality.html).

### Fixed

- Fixed an issue where x-axis titles would appear on top of plots
- Removed erroneous checks during calculation of realized regression performance metrics. [(#279)](https://github.com/NannyML/nannyml/issues/279)
- Fixed an issue dealing with `az://` URLs in the CLI, thanks [@michael-nml](https://github.com/michael-nml) [(#283)](https://github.com/NannyML/nannyml/issues/283)

## [0.8.5] - 2023-03-29

### Changed

- Applied new rules for visualizations. Estimated values will be the color indigo and represented with a dashed line.
  Calculated values will be blue and have a solid line. This color coding might be overridden in comparison plots.
  Data periods will no longer have different colors, we've added some additional text fields to the plot to indicate the data period.
- Cleaned up legends in plots, since there will no longer be a different entry for reference and analysis periods of metrics.
- Removed the lower threshold for default thresholds of the KS and Wasserstein drift detection methods.

### Added

- We've added the `business_value` metric for both estimated and realized binary classification performance. It allows
  you to assign a value (or cost) to true positive, true negative, false positive and false negative occurrences.
  This can help you track something like a monetary value or business impact of a model as a metric. Read more in the
  business value tutorials ([estimated](https://nannyml.readthedocs.io/en/latest/tutorials/performance_estimation/binary_performance_estimation/business_value_estimation.html)
  or [realized](https://nannyml.readthedocs.io/en/latest/tutorials/performance_calculation/binary_performance_calculation/business_value_calculation.html))
  or the [how it works](https://nannyml.readthedocs.io/en/latest/how_it_works/business_value.html) page.

### Fixed

- Sync quickstart of the README with the dedicated quickstart page. [(#256)](https://github.com/NannyML/nannyml/issues/256)
  Thanks [@NeoKish](https://github.com/NeoKish)!
- Fixed incorrect code snippet order in the thresholding tutorial. [(#258)](https://github.com/NannyML/nannyml/issues/258)
  Thanks once more to the one and only [@NeoKish](https://github.com/NeoKish)!
- Fixed broken container build that had sneakily been going on for a while
- Fixed incorrect confidence band color in comparison plots [(#259)](https://github.com/NannyML/nannyml/issues/259)
- Fixed incorrect titles and missing legends in comparison plots [(#264)](https://github.com/NannyML/nannyml/issues/264)
- Fixed an issue where numerical series marked as category would cause issues during Chi2 calculation

## [0.8.4] - 2023-03-17

### Changed

- Updated univariate drift methods to no longer store all reference data by default [(#182)](https://github.com/NannyML/nannyml/issues/182)
- Updated univariate drift methods to deal better with missing data [(#202)](https://github.com/NannyML/nannyml/issues/202)
- Updated the included example datasets
- Critical security updates for dependencies
- Updated visualization of multi-level table headers in the docs [(#242)](https://github.com/NannyML/nannyml/issues/242)
- Improved typing support for Result classes using generics

### Added

- Support for estimating the confusion matrix for binary classification [(#191)](https://github.com/NannyML/nannyml/issues/191)
- Added `treat_as_categorical` parameter to univariate drift calculator [(#239)](https://github.com/NannyML/nannyml/issues/239)
- Added comparison plots to help visualize two different metrics at once

### Fixed

- Fix missing confidence boundaries in some plots [(#193)](https://github.com/NannyML/nannyml/issues/193)
- Fix incorrect metric names on plot y-axes [(#195)](https://github.com/NannyML/nannyml/issues/195)
- Fix broken links to external docs [(#196)](https://github.com/NannyML/nannyml/issues/196)
- Fix missing display name to performance calculation and estimation charts [(#200)](https://github.com/NannyML/nannyml/issues/200)
- Fix missing confidence boundaries for single metric plots [(#203)](https://github.com/NannyML/nannyml/issues/203)
- Fix incorrect code in example notebook for ranking
- Fix result corruption when re-using calculators [(#206)](https://github.com/NannyML/nannyml/issues/206)
- Fix unintentional period filtering [(#199)](https://github.com/NannyML/nannyml/issues/199)
- Fixed some typing issues [(#213)](https://github.com/NannyML/nannyml/issues/213)
- Fixed missing data requirements documentation on regression [(#215)](https://github.com/NannyML/nannyml/issues/215)
- Corrections in the glossary [(#214)](https://github.com/NannyML/nannyml/issues/214), thanks [@sebasmos](https://github.com/sebasmos)!
- Fix missing treshold in plotting legend [(#219)](https://github.com/NannyML/nannyml/issues/219)
- Fix missing annotation in single row & column charts [(#221)](https://github.com/NannyML/nannyml/issues/221)
- Fix outdated performance estimation and calculation docs [(#223)](https://github.com/NannyML/nannyml/issues/223)
- Fix categorical encoding of unseen values for DLE [(#224)](https://github.com/NannyML/nannyml/issues/224)
- Fix incorrect legend for None timeseries [(#235)](https://github.com/NannyML/nannyml/issues/235)

## [0.8.3] - 2023-01-31

### Added

- Added some extra semantic methods on results for easy property access. No dealing with multilevel indexes required.
- Added functionality to compare results and plot that comparison. Early release version.

### Fixed

- Pinned Sphinx version to 4.5.0 in the [documentation requirements](docs/requirements.txt).
  Version selector, copy toggle buttons and some styling were broken on RTD due to unintended usage of Sphinx 6 which
  treats jQuery in a different way.

## [0.8.2] - 2023-01-24

### Changed

- Log Ranker usage logging
- Remove some redundant parameters in `plot()` function calls for data reconstruction results, univariate drift results,
  CBPE results and DLE results.
- Support "single metric/column" arguments in addition to lists in class creation [(#165)](https://github.com/NannyML/nannyml/issues/165)
- Fix incorrect 'None' checks when dealing with defaults in univariate drift calculator
- Multiple updates and corrections to the docs (thanks [@nikml](https://github.com/nikml)!), including:
  - Updating univariate drift tutorial
  - Updating README
  - Update PCA: How it works
  - Fix incorrect plots
  - Fix quickstart [(#171)](https://github.com/NannyML/nannyml/issues/171)
- Update chunker docstrings to match parameter names, thanks [@mrggementiza](https://github.com/jrggementiza)!
- Make sequence 'None' checks more readable, thanks [@mrggementiza](https://github.com/jrggementiza)!
- Ensure error handling in usage logging does not cause errors...
- Start using `OrdinalEncoder` instead of `LabelEncorder` in DLE. This allows us to deal with "unseen" values in the
  analysis period.

### Added

- Added a Store to provide persistence for objects. Main use case for now is storing fitted calculators to be reused
  later without needing to fit on reference again. Current store implementation uses a local or remote filesystem as a
  persistence layer. Check out the documentation on [persisting calculators](https://nannyml.readthedocs.io/en/latest/tutorials/persisting_calculators.html).

### Fixed

- Fix incorrect interpretation of `y_pred` column as continuous values for the included sample binary classification data.
  Converting the column explicitly to "category" data type for now, update of the dataset to follow soon.
  [(#171)](https://github.com/NannyML/nannyml/issues/171)
- Fix broken image link in README, thanks [@mrggementiza](https://github.com/jrggementiza)!
- Fix missing key in the CLI section on raw files output, thanks [@CoffiDev](https://github.com/CoffiDev)!
- Fix upper and lower thresholds for data reconstruction being swapped [(#179)](https://github.com/NannyML/nannyml/issues/179)
- Fix stacked bar chart plots (missing bars + too many categories shown)


## [0.8.1] - 2022-12-01

### Changed

- Thorough refactor of the `nannyml.drift.ranker` module. The abstract base class and factory have been dropped in favor
  of a more flexible approach.
- Thorough refactor of our Plotly-based plotting modules. These have been rewritten from scratch to make them more
  modular and composable. This will allow us to deliver more powerful and meaningful visualizations faster.

### Added

- Added a new univariate drift method. The [`Hellinger distance`](https://nannyml.readthedocs.io/en/v0.8.1/how_it_works/univariate_drift_detection.html#hellinger-distance), used for continuous variables.
- Added an [extensive write-up]() on when to use which univariate drift method.
- Added a new way to rank the results of univariate drift calculation. The `CorrelationRanker` ranks columns based on
  the correlation between the drift value and the change in realized or estimated performance. Read all about it in the
  [ranking documentation](https://nannyml.readthedocs.io/en/v0.8.1/how_it_works/ranking.html)

### Fixed

- Disabled usage logging for or GitHub workflows
- Allow passing a single string to the `metrics` parameter of the `result.filter()` function, as per special request.

## [0.8.0] - 2022-11-22

### Changed

- Updated `mypy` to a new version, immediately resulting in some new checks that failed.

### Added

- Added new univariate drift methods. The [`Wasserstein distance`](https://nannyml.readthedocs.io/en/latest/how_it_works/univariate_drift_detection.html#wasserstein-distance) for continuous variables,
  and the [`L-Infinity distance`](https://nannyml.readthedocs.io/en/main/how_it_works/univariate_drift_detection.html#l-infinity-distance) for categorical variables.
- Added usage logging to our key functions. Check out the [docs](https://nannyml.readthedocs.io/en/latest/usage_logging.html#providing-a-env-file) to find out more on what, why, how, and how to
  disable it if you want to.

### Fixed

- Fixed and updated various parts of the docs, reported at warp speed! Thanks [@NeoKish](https://github.com/NeoKish)!
- Fixed `mypy` issues concerning 'implicit optionals'.

## [0.7.0] - 2022-11-07

### Changed

- Updated the handling of "leftover" observations when using the `SizeBasedChunker` and `CountBasedChunker`.
  Renamed the parameter for tweaking that behavior to `incomplete`, that can be set to `keep`, `drop` or `append`.
  Default behavior for both is now to append leftover observations to the last _full_ chunk.
- Refactored the `nannyml.drift` module. The intermediate structural level (`model_inputs`, `model_outputs`, `targets`)
  has been removed and turned into a single unified `UnivariateDriftCalculator`. The old built-in statistics have been
  re-implemented as `Methods`, allowing us to add new methods to detect univariate drift.
- Simplified a lot of the codebase (but also complicated some bits) by storing results internally as multilevel-indexed
  DataFrames. This means we no longer have to 'convey information' by encoding data column names and method names in
  the names of result columns. We've introduced a new paradigm to deal with results. Drill down to the data you really
  need by using the `filter` method, which returns a new `Result` instance, with a smaller 'scope'. Then turn this
  `Result` into a DataFrame using the `to_df` method.
- Changed the structure of the [pyproject.toml](pyproject.toml) file due to a Poetry upgrade to version 1.2.1.

### Added

- Expanded the `nannyml.io` module with new `Writer` implementations: `DatabaseWriter` that exports data into multiple
  tables in a relational database and the `PickleFileWriter` which stores the
  pickled `Results` on local/remote/cloud disk.
- Added a new univariate drift detection method based on the Jensen-Shannon distance.
  Used within the `UnivariateDriftCalculator`.

### Fixed

- Added [lightgbm](https://github.com/microsoft/LightGBM) installation instructions to our installation guide.

## [0.6.3] - 2022-09-22

### Changed

- `dependencybot` dependency updates
- `stalebot` setup

### Fixed

- CBPE now uses uncalibrated `y_pred_proba` values to calculate realized performance. Fixed for both binary and
  multiclass use cases [(#98)](https://github.com/NannyML/nannyml/issues/98)
- Fix an issue where reference data was rendered incorrectly on joy plots
- Updated the 'California Housing' example docs, thanks for the help [@NeoKish](https://github.com/NeoKish)
- Fix lower confidence bounds and thresholds under zero for regression cases. When the lower limit is set to 0,
  the lower threshold will not be plotted. [(#127)](https://github.com/NannyML/nannyml/issues/127)

## [0.6.2] - 2022-09-16

### Changed

- Made the `timestamp_column_name` required by all calculators and estimators optional. The main consequences of this
  are plots have a chunk-index based x-axis now when no timestamp column name was given. You can also not chunk by
  period when the timestamp column name is not specified.

### Fixed

- Added missing `s3fs` dependency
- Fixed outdated plotting kind constants in the runner (used by CLI)
- Fixed some missing images and incorrect version numbers in the README, thanks [@NeoKish](https://github.com/NeoKish)!

### Added

- Added a lot of additional tests, mainly concerning plotting and the [`Runner`](nannyml/runner.py) class

## [0.6.1] - 2022-09-09

### Changed

- Use the `problem_type` parameter to determine the correct graph to output when plotting model output drift

### Fixed

- Showing the wrong plot title for DLE estimation result plots, thanks [@NeoKish](https://github.com/NeoKish)
- Fixed incorrect plot kinds in some error feedback for the model output drift calculator
- Fixed missing `problem_type` argument in the Quickstart guide
- Fix incorrect visualization of confidence bands on reference data in DEE and CBPE result plots

## [0.6.0] - 2022-09-07

### Added

- Added support for regression problems across all calculators and estimators.
  In some cases a required `problem_type` parameter is required during calculator/estimator initialization, this
  is a breaking change. Read more about using regression in our
  [tutorials](https://nannyml.readthedocs.io/en/main/tutorials.html) and about our new performance estimation
  for regression using the [Direct Loss Estimation (DLE)](https://nannyml.readthedocs.io/en/main/how_it_works/performance_estimation.html#direct-loss-estimation-dle) algorithm.

### Changed

- Improved `tox` running speed by skipping some unnecessary package installations.
  Thanks [@baskervilski](https://github.com/baskervilski)!

### Fixed

- Fixed an issue where some Pandas column datatypes were not recognized as continuous by NannyML, causing them to be
  dropped in calculations. Thanks for reporting [@Dbhasin1](https://github.com/Dbhasin1)!
- Fixed an issue where some helper columns for visualization crept into the stored reference results. Good catch
  [@Dbhasin1](https://github.com/Dbhasin1)!
- Fixed an issue where a `Reader` instance would raise a `WriteException`. Thanks for those eagle eyes
  [@baskervilski](https://github.com/baskervilski)!

## [0.5.3] - 2022-08-30

### Changed

- We've completely overhauled the way we determine the "stability" of our estimations. We've moved on from determining
  a minimum `Chunk` size to estimating the *sampling error* for an operation on a `Chunk`.
  - A **sampling error** value will be provided per metric per `Chunk` in the result data for
    **reconstruction error multivariate drift calculator**, all **performance calculation metrics** and
    all **performance estimation metrics**.
  - Confidence bounds are now also based on this *sampling error* and will display a range around an estimation +/- 3
    times the *sampling error* in **CBPE** and **reconstruction error multivariate drift calculator**.
  Be sure to check out our [in-depth documentation](https://nannyml.readthedocs.io/en/main/how_it_works/estimation_of_standard_error.html#estimation-of-standard-error)
  on how it works or dive right into the [implementation](nannyml/sampling_error).

### Fixed

- Fixed issue where an outdated version of Numpy caused Pandas to fail reading string columns in some scenarios
  [(#93)](https://github.com/NannyML/nannyml/issues/93). Thank you, [@bernhardbarker](https://github.com/bernhardbarker) and
  [@ga-tardochisalles](https://github.com/ga-tardochisalles) for the investigative work!

## [0.5.2] - 2022-08-17

### Changed

- Swapped out ASCII art library from 'art' to 'PyFiglet' because the former was not yet present in conda-forge.

### Fixed

- Some leftover parameter was forgotten during cleanup, breaking CLI functionality
- CLI progressbar was broken due to a boolean check with task ID 0.


## [0.5.1] - 2022-08-16

### Added

- Added simple CLI implementation to support automation and MLOps toolchain use cases. Supports reading/writing to
  cloud storage using S3, GCS, ADL, ABFS and AZ protocols. Containerized version available at
  [dockerhub](https://hub.docker.com/repository/docker/nannyml/nannyml).

### Changed

- `make clean` now also clears `__pycache__`
- Fixed some inconsistencies in docstrings (they still need some additional love though)

## [0.5.0] - 2022-07-07

### Changed
- Replaced the whole Metadata system by a more intuitive approach.

### Fixed
- Fix docs [(#87)](https://github.com/NannyML/nannyml/issues/79) and [(#89)](https://github.com/NannyML/nannyml/issues/89), thanks [@NeoKish](https://github.com/NeoKish)
- Fix confidence bounds for binary settings [(#86)](https://github.com/NannyML/nannyml/issues/86), thanks [@rfrenoy](https://github.com/rfrenoy)
- Fix README [(#87)](https://github.com/NannyML/nannyml/issues/79), thanks [@NeoKish](https://github.com/NeoKish)
- Fix index misalignment on calibration [(#79)](https://github.com/NannyML/nannyml/issues/79)
- Fix Poetry dev-dependencies issues [(#78)](https://github.com/NannyML/nannyml/issues/78), thanks [@rfrenoy](https://github.com/rfrenoy)
- Fix incorrect documentation links [(#76)](https://github.com/NannyML/nannyml/issues/76), thanks [@SoyGema](https://github.com/SoyGema)

## [0.4.1] - 2022-05-19

### Added
- Added limited support for ``regression`` use cases: create or extract ``RegressionMetadata`` and use it for drift
  detection. Performance estimation and calculation require more research.

### Changed
- ``DefaultChunker`` splits into 10 chunks of equal size.
- ``SizeBasedChunker`` no longer drops incomplete last chunk by default, but this is now configurable behavior.

## [0.4.0] - 2022-05-13

### Added
- Added support for new metrics in the Confidence Based Performance Estimator (CBPE). It now estimates ``roc_auc``,
  ``f1``, ``precision``, ``recall`` and ``accuracy``.
- Added support for **multiclass classification**. This includes
  - Specifying ``multiclass classification metadata`` + support in automated metadata extraction (by introducing a
    ``model_type`` parameter).
  - Support for all ``CBPE`` metrics.
  - Support for realized performance calculation using the ``PerformanceCalculator``.
  - Support for all types of drift detection (model inputs, model output, target distribution).
  - A new synthetic toy dataset.

### Changed
- Removed the ``identifier`` property from the ``ModelMetadata`` class. Joining ``analysis`` data and
  ``analysis target`` values should be done upfront or index-based.
- Added an ``exclude_columns`` parameter to the ``extract_metadata`` function. Use it to specify the columns that should
  not be considered as model metadata or features.
- All ``fit`` methods now return the fitted object. This allows chaining ``Calculator``/``Estimator`` instantiation
  and fitting into a single line.
- Custom metrics are no longer supported in the ``PerformanceCalculator``. Only the predefined metrics remain supported.
- Big documentation revamp: we've tweaked overall structure, page structure and incorporated lots of feedback.
- Improvements to consistency and readability for the 'hover' visualization in the step plots, including consistent
  color usage, conditional formatting, icon usage etc.
- Improved indication of "realized" and "estimated" performance in all ``CBPE`` step plots
  (changes to hover, axes and legends)

### Fixed
- Updated homepage in project metadata
- Added missing metadata modification to the *quickstart*
- Perform some additional check on reference data during preprocessing
- Various documentation suggestions [(#58)](https://github.com/NannyML/nannyml/issues/58)

## [0.3.2] - 2022-05-03

### Fixed
- Deal with out-of-time-order data when chunking
- Fix reversed Y-axis and plot labels in continuous distribution plots

## [0.3.1] - 2022-04-11
### Changed
- Publishing to PyPi did not like raw sections in ReST, replaced by Markdown version.

## [0.3.0] - 2022-04-08
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

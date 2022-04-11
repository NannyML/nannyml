<p align="center">
    <img src="https://assets.website-files.com/6099466e98d9381b3f745b9a/60994ab2b5bd890780db9c84_NannyML%20logo%20horizontal%20typfont.png">
</p>
<p align="center">
    <a href="https://pypi.org/project/nannyml/">
        <img src="https://img.shields.io/pypi/v/nannyml.svg">
    </a>
    <a href="https://pypi.org/project/nannyml/">
        <img src="https://img.shields.io/pypi/pyversions/nannyml.svg">
    </a>
    <a href="https://github.com/nannyml/nannyml/actions/workflows/dev.yml">
        <img src="https://github.com/NannyML/nannyml/actions/workflows/dev.yml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/NannyML/nannyml">
        <img src="https://codecov.io/gh/NannyML/nannyml/branch/main/graph/badge.svg?token=OGpF5gVzfR">
    </a>
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/nannyml?color=green">
</p>

<p align="center">
    <strong>
        <a href="https://nannyml.com/">Website</a>
        •
        <a href="https://docs.nannyml.com">Docs</a>
        •
        <a href="https://nannymlbeta.slack.com">Community Slack</a>
    </strong>
</p>

# Basic overview

NannyML helps you monitor your ML models in production by:

* estimating performance in absence of ground truth
* calculating realized performance metrics
* detecting data drift on model inputs, model outputs and targets

# Installing the latest stable release

```python
pip install nannyml
```


# Installing the latest development version

```python
python -m pip install git+https://github.com/NannyML/nannyml
```


# Getting started

```python
import nannyml as nml
import pandas as pd

# Load some data
reference_data, analysis_data, _ = nml.load_synthetic_sample()
data = pd.concat([reference_data, analysis_data])
metadata = nml.extract_metadata(reference_data)

# Estimate performance
estimator = nml.CBPE(metadata).fit(reference_data)
estimates = estimator.estimate(data)

estimates.plot(kind='performance').show()
```

# Examples

* [Performance estimation](<https://docs.nannyml.com/latest/guides/performance_estimation.html>)
* [Realized performance calculation](https://docs.nannyml.com/latest/guides/performance_calculation.html)
* [Univariate model input drift detection](https://docs.nannyml.com/latest/guides/data_drift.html#univariate-drift-detection)
* [Multivariate model input drift detection](https://docs.nannyml.com/latest/guides/data_drift.html#drift-detection-for-model-outputs)
* [Model output drift detection](https://docs.nannyml.com/latest/guides/data_drift.html#drift-detection-for-model-outputs)
* [Model target distribution](https://docs.nannyml.com/latest/guides/data_drift.html#drift-detection-for-model-targets)

# Development setup

* Read the docs on [how to contribute](CONTRIBUTING.md)

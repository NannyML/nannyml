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
        <a href="https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ">Community Slack</a>
    </strong>
</p>

# What is NannyML?
NannyML is an open source python library that allows you to estimate real-world model performance (without access to targets),
detect multivariate data drift, and link data drift to changes in model performance. Built for data scientist, NannyML has a easy to use interface, beautiful and interactive visualizations, and currently
works on all binary classification models with tabular data.

NannyMLs performance estimated algorithm, confidence based performance estimated, was researched
and developed by NannyML core contributors. NannyML has also developed it's one multivariate
drift detection based on PCA reconstruction error.

#Why use NannyML?
NannyML is built to allow data scientist to easily and automatically detect silent model failure. Data scientists can
then explore the root cause of the changes in model performance. NannyML also outputs alerts when performance has changes
and for data drift that is correlated with the performance change.

By using NannyML you get the following benifits:

* **Automatic** detection of silent model failure
* Discovering the **root cause** to why your model performance has changes
* No alert fatiuge, react only when necessary model performance is impacted
* **Painless** setup in any enviroment
* An end to sleepless nights caused by not knowing your model performance 😴

# GO DEEP
| NannyML Resources | Description |
| ------------- | ------------- |
| **[NannyML 101]** | New to NannyML? Start here! |
| **[Key Concepts]** | Glossary of key concepts we use. |
| **[Technical Reference]** | Monitor the performance of your ML models. |
| **[New in v0.3.1]** | New features, bug fixes. |
| **[Docs]** | Full documentation for using NannyML. |
| **[Real World Example]** |  Take a look a real world example of NannyML. |
| **[Blog]** | Thoughts on post deployment data science from the NannyML team. |
| **[Newsletter]** | All things post deployment data science. Subscribe to see the latest papers and blogs. |
| **[Join Slack]** | Need help with your specific use case? Say hi on Slack! |
| **[Contribute]** | How to contribute to the NannyML project and code base. |

[NannyML 101]: https://docs.nannyml.com/
[Core Concepts]: https://docs.nannyml.com/latest/glossary.html
[Technical Reference]:https://docs.nannyml.com/latest/nannyml/nannyml.html
[New in v0.3.1]: https://nannyml.substack.com/p/nannyml-031-release-soft-launch?s=w
[Docs]: https://docs.nannyml.com/
[Real World Example]: https://docs.nannyml.com/v0.3.1/guides/real_world_data_example.html
[Blog]: https://www.nannyml.com/blog
[Newsletter]:  https://mailchi.mp/022c62281d13/postdeploymentnewsletter
[Join Slack]: https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ
[Contribute]: https://github.com/NannyML/nannyml/blob/main/CONTRIBUTING.md

#Features

### 1. Estimating real-world model performance in absence of ground truth
Using an algorithm researched by NannyML contributors called **Confidence Based Performance Estimation**, NannyML is
able to estimate model performance. NannyML reconstructs the expected confusion matrix and calculates ROC AUC.

![image](https://drive.google.com/file/d/1mNML2fhpU1J6mjYIkQehw87oXIJsGOvA/view?usp=sharing)

### 2. Multivariate Data drift
NannyML uses Data Reconstruction with PCA to detect multivariate changes.
NannyML monitors the reconstruction error over time for the monitored model and raises an alert if the values get
outside the range observed in the reference partition.
### 3. Univariate drift
NannyML uses the KS Test for continuous features and the 2 sample Chi squared test for categorical features. Both tests
provide a statistic where they measure the observed drift and a p-value that shows how likely we are to get the observed
sample under the assumption that there was no drift.

### 4. Target Shift
Monitor target fshit by calculating the mean
occurance of positive events as well as the chi-squared statistic, from the 2 sample Chi Squared test, of the target
values for each chunk.
### 5. Model output Drift
NannyML also detects data drift in the Model Outputs. It uses the same univariate methodology as for a continuous feature.
### 6. Calculating realized performance metrics
NannyML uses TargetDistributionCalculator in order to monitor drift in Target distribution.
It can calculate the mean occurance of positive events as well as the chi-squared statistic,
from the 2 sample Chi Squared test, of the target values for each chunk.

### 7. Interactive Visualization
A whole suite of beautiful visualization that visualize all of the data that comes out NannyML. They are designed
to help you explore your model performance and data.

### 8. Meta data extraction
NannyML provides the nannyml.metadata.extract_metadata() function to automatically extract the required metadata from a
given DataFrame. It does so by following some simple naming conventions and heuristics to column names and data.
 It returns a prepopulated instance of the ModelMetadata class.

# Getting started

## Install NannyML

 *Requirments*: NannyML requires Python >=3.7.1, <3.11

 Easily install via PyPI:

```bash
pip install nannyml
```


*HERE BE DRAGONS*:

You can install the latest development version of NannyML here

**USE AT YOUR OWN RISK**
```bash
python -m pip install git+https://github.com/NannyML/nannyml
```

##Quick Start

#### Step 1: Import NannyML and Pandas
```python
import nannyml as nml
import pandas as pd
```
#### Step 2: Import data from you ML system and split it in reference and analysis period

**Reference refers to the period you want to compare your production data too**

**Analysis is your production data**

NannyML provides synthetic data for you play around with =)

```python
reference_data, analysis_data, _ = nml.load_synthetic_sample()
data = pd.concat([reference_data, analysis_data])
metadata = nml.extract_metadata(reference_data)
metadata.target_column_name = 'work_home_actual'
```
#### Step 3: Estimate the performance of your model
```python
# Estimate performance
estimator = nml.CBPE(metadata)
estimator.fit(reference_data)
estimates = estimator.estimate(data)

estimates.plot(kind='performance').show()
```

#### Step 4: Univarte Drift Detection
```python
univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=chunk_size)
univariate_calculator.fit(reference_data=reference)
univariate_results = univariate_calculator.calculate(data=data)

# let's plot drift results for all model inputs
for feature in metadata.features:
    figure = univariate_results.plot(kind='feature_drift', metric='statistic', feature_label=feature.label)
    figure.show()
```

#### Step 5: Multivariate Drift Detection

```python
# Let's initialize the object that will perform Data Reconstruction with PCA
rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=metadata, chunk_size=chunk_size)
# NannyML compares drift versus the full reference dataset.
rcerror_calculator.fit(reference_data=reference)
# let's see Reconstruction error statistics for all available data
rcerror_results = rcerror_calculator.calculate(data=data)
figure = rcerror_results.plot(kind='drift')
figure.show()
```

#### Step 6: Model Output drift
```python
figure = univariate_results.plot(kind='prediction_drift', metric='statistic')
figure.show()
```

#### Step 7: Alerts

```python
ranker = nml.Ranker.by('alert_count')
ranked_features = ranker.rank(univariate_results, model_metadata=metadata, only_drifting = False)
ranked_features
```


# Detailed documention and guides

* [Performance estimation](<https://docs.nannyml.com/latest/guides/performance_estimation.html>)
* [Realized performance calculation](https://docs.nannyml.com/latest/guides/performance_calculation.html)
* [Univariate model input drift detection](https://docs.nannyml.com/latest/guides/data_drift.html#univariate-drift-detection)
* [Multivariate model input drift detection](https://docs.nannyml.com/latest/guides/data_drift.html#multivariate-drift-detection)
* [Model output drift detection](https://docs.nannyml.com/latest/guides/data_drift.html#drift-detection-for-model-outputs)
* [Model target distribution](https://docs.nannyml.com/latest/guides/data_drift.html#drift-detection-for-model-targets)

# Contributing and Community

We want to build NannyML together with the community! The best way to contributeat the moment is to
propose new features, or log bugs under [issues](https://github.com/NannyML/nannyml/issues).

Also we would love if you joined some discussions in the community [slack](https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ)
Read the docs on [how to contribute](CONTRIBUTING.md).

# Asking for help

The best place to ask for help is on [slack](https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ).
Feel free to join and ask questions or raise issues. Someone will definitely respond to you.

# License

NannyML is distributed under an Apache License Version 2.0. A complete version is found [here](LICENSE.MD). All contributions
will be made under distributed under this license.



<p align="center">
    <img src="media/thumbnail-4.png">
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
    <a href='https://nannyml.readthedocs.io/en/main/?badge=main'>
        <img src='https://readthedocs.org/projects/nannyml/badge/?version=main' alt='Documentation Status' />
    </a>
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/nannyml?color=green">
    <br>
    <br>
    <a href="https://www.producthunt.com/posts/nannyml?utm_source=badge-top-post-badge&utm_medium=badge&utm_souce=badge-nannyml" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=346412&theme=light&period=daily" alt="NannyML - OSS&#0032;Python&#0032;library&#0032;for&#0032;detecting&#0032;silent&#0032;ML&#0032;model&#0032;failure | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>

</p>

<p align="center">
    <strong>
        <a href="https://nannyml.com/">Website</a>
        •
        <a href="https://nannyml.readthedocs.io/en/stable/">Docs</a>
        •
        <a href="https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ">Community Slack</a>
    </strong>
</p>

<p align="center">
  <img src="media/cbpe_v3.gif" alt="animated">
</p>


# 💡 What is NannyML?
NannyML is an open-source python library that allows you to **estimate post-deployment model performance** (without access to targets), detect data drift, and intelligently link data drift alerts back to changes in model performance. Built for data scientists, NannyML has an easy-to-use interface, interactive visualizations, is completely model-agnostic and currently supports all tabular classification use cases.

The core contributors of NannyML have researched and developed a novel algorithm for estimating model performance: [confidence-based performance estimation (CBPE)](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html). The nansters also invented a new approach to detect [multivariate data drift](https://nannyml.readthedocs.io/en/stable/how_it_works/data_reconstruction.html) using PCA-based data reconstruction.

If you like what we are working on, be sure to become a Nanster yourself, join our [community slack](https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ) <img src="media/slack.png" width='25'> and support us with a GitHub <img src="media/github.png" width='25'> star ⭐.

# ☔ Why use NannyML?
NannyML closes the loop with performance monitoring and post deployment data science, empowering data scientist to quickly understand and **automatically detect silent model failure**. By using NannyML, data scientists can finally maintain complete visibility and trust in their deployed machine learning models.
Allowing you to have the following benefits:

* End sleepless nights caused by not knowing your model performance 😴
* Analyse data drift and model performance **over time**
* Discover the **root cause** to why your models are not performing as expected
* **No alert fatigue!** React only when necessary if model performance is impacted
* **Painless** setup in any environment

# 🧠 GO DEEP
| NannyML Resources                                       | Description                                                                            |
|---------------------------------------------------------|----------------------------------------------------------------------------------------|
| ☎️ **[NannyML 101]**                                     | New to NannyML? Start here!                                                            |
| 🔮 **[Performance estimation]**                         | How the magic works.                                                                   |
| 🌍 **[Real world example]**                             | Take a look at a real-world example of NannyML.                                        |
| 🔑 **[Key concepts]**                                   | Glossary of key concepts we use.                                                       |
| 🔬 **[Technical reference]**                            | Monitor the performance of your ML models.                                             |
| 🔎 **[Blog]**                                           | Thoughts on post-deployment data science from the NannyML team.                        |
| 📬 **[Newsletter]**                                     | All things post-deployment data science. Subscribe to see the latest papers and blogs. |
| 💎 **[New in v0.4.1]**                                  | New features, bug fixes.                                                               |
| 🧑‍💻 **[Contribute]**                                     | How to contribute to the NannyML project and codebase.                                 |
| <img src="media/slack.png" width='25'> **[Join slack]** | Need help with your specific use case? Say hi on slack!                                |

[NannyML 101]: https://nannyml.readthedocs.io/en/stable/
[Performance Estimation]: https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html
[Key Concepts]: https://nannyml.readthedocs.io/en/stable/glossary.html
[Technical Reference]:https://nannyml.readthedocs.io/en/stable/nannyml/modules.html
[New in v0.4.1]: https://github.com/NannyML/nannyml/releases/latest/
[Real World Example]: https://nannyml.readthedocs.io/en/stable/examples/california_housing.html
[Blog]: https://www.nannyml.com/blog
[Newsletter]:  https://mailchi.mp/022c62281d13/postdeploymentnewsletter
[Join Slack]: https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ
[Contribute]: https://github.com/NannyML/nannyml/blob/main/CONTRIBUTING.rst

# 🔱 Features

### 1. Performance estimation and monitoring

When the actual outcome of your deployed prediction models is delayed, or even when post-deployment target labels are completely absent, you can use NannyML's [CBPE-algorithm](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html) to **estimate model performance**. This algorithm requires the predicted probabilities of your machine learning model and leverages probability calibration to estimate any traditional binary classification metric (ROC_AUC, Precision, Recall, F1, etc.). Rather than estimating the performance of future model predictions, CBPE estimates the expected model performance of the predictions made at inference time.

<p><img src="docs/_static/tutorial-perf-est-guide-with-ref-roc_auc.svg"></p>

NannyML can also **track the realised performance** of your machine learning model once targets are available.

### 2. Data drift detection

To detect **multivariate feature drift** NannyML uses [PCA-based data reconstruction](https://nannyml.readthedocs.io/en/main/how_it_works/data_reconstruction.html). Changes in the resulting reconstruction error are monitored over time and data drift alerts are logged when the reconstruction error in a certain period exceeds a threshold. This threshold is calculated based on the reconstruction error observed in the reference period.

<p><img src="docs/_static/drift-guide-multivariate.svg"></p>

NannyML utilises statistical tests to detect **univariate feature drift**. The Kolmogorov–Smirnov test is used for continuous features and the 2-sample chi-squared test for categorical features. The results of these tests are tracked over time, properly corrected to counteract multiplicity and overlayed on the temporal feature distributions. (It is also possible to visualise the test-statistics over time, to get a notion of the drift magnitude.)

<p><img src="docs/_static/drift-guide-joyplot-distance_from_office.svg"><img src="docs/_static/drift-guide-stacked-salary_range.svg"></p>

NannyML uses the same statistical tests to detected **model output drift**.

<p><img src="docs/_static/drift-guide-predictions-joyplot.svg"></p>

**Target distribution drift** is monitored by calculating the mean occurrence of positive events in combination with the 2-sample chi-squared test. Bear in mind that this operation requires the presence of actuals.

<p><img src="docs/_static/target_distribution_metric.svg"></p>

### 3. Intelligent alerting

Because NannyML can estimate performance, it is possible to weed out data drift alerts that do not impact expected performance, combatting alert fatigue. Besides linking data drift issues to drops in performance it is also possible to prioritise alerts according to other criteria using NannyML's Ranker.


# 🚀 Getting started

### Install NannyML

From PyPI:

```bash
pip install nannyml
```

**Here be dragons!** Use the latest development version of NannyML at your own risk:

```bash
python -m pip install git+https://github.com/NannyML/nannyml
```

### Quick Start

*The following snippet is based on our [latest release](https://github.com/NannyML/nannyml/releases/latest)*.

```python
import pandas as pd
import nannyml as nml

# Load synthetic data
reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
data = pd.concat([reference, analysis], ignore_index=True)

# Extract meta data
metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor', model_type='classification_binary', exclude_columns=['identifier'])
metadata.target_column_name = 'work_home_actual'

# Choose a chunker or set a chunk size
chunk_size = 5000

# Estimate model performance
estimator = nml.CBPE(model_metadata=metadata, metrics=['roc_auc'], chunk_size=chunk_size)
estimator.fit(reference)
estimated_performance = estimator.estimate(data=data)

figure = estimated_performance.plot(metric='roc_auc', kind='performance')
figure.show()

# Detect multivariate feature drift
multivariate_calculator = nml.DataReconstructionDriftCalculator(model_metadata=metadata, chunk_size=chunk_size)
multivariate_calculator.fit(reference_data=reference)
multivariate_results = multivariate_calculator.calculate(data=data)

figure = multivariate_results.plot(kind='drift')
figure.show()

# Detect univariate feature drift
univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=chunk_size)
univariate_calculator.fit(reference_data=reference)
univariate_results = univariate_calculator.calculate(data=data)

# Rank features based on number of alerts
ranker = nml.Ranker.by('alert_count')
ranked_features = ranker.rank(univariate_results, model_metadata=metadata, only_drifting = False)

for feature in ranked_features.feature:
    figure = univariate_results.plot(kind='feature_distribution', feature_label=feature)
    figure.show()
```

# 📖 Documentation

* Performance monitoring
  - [Estimated performance](https://nannyml.readthedocs.io/en/main/tutorials/performance_estimation.html)
  - [Realized performance](https://nannyml.readthedocs.io/en/main/tutorials/performance_calculation.html)
* Drift detection
  * [Multivariate feature drift](https://nannyml.readthedocs.io/en/main/tutorials/detecting_data_drift/multivariate_drift_detection.html)
  - [Univariate feature drift](https://nannyml.readthedocs.io/en/main/tutorials/detecting_data_drift/univariate_drift_detection.html)
  - [Model output drift](https://nannyml.readthedocs.io/en/main/tutorials/detecting_data_drift/drift_detection_for_model_outputs.html)
  - [Target distribution drift](https://nannyml.readthedocs.io/en/main/tutorials/detecting_data_drift/drift_detection_for_model_targets.html)

# 🦸 Contributing and Community

We want to build NannyML together with the community! The easiest to contribute at the moment is to propose new features or log bugs under [issues](https://github.com/NannyML/nannyml/issues). For more information, have a look at [how to contribute](CONTRIBUTING.rst).

# 🙋 Get help

The best place to ask for help is in the [community slack](https://join.slack.com/t/nannymlbeta/shared_invite/zt-16fvpeddz-HAvTsjNEyC9CE6JXbiM7BQ). Feel free to join and ask questions or raise issues. Someone will definitely respond to you.

# 🥷 Stay updated

If you want to stay up to date with recent changes to the NannyML library, you can subscribe to our [release notes](https://nannyml.substack.com). For thoughts on post-deployment data science from the NannyML team, feel free to visit our [blog](https://www.nannyml.com/blog). You can also sing up for our [newsletter](https://mailchi.mp/022c62281d13/postdeploymentnewsletter), which brings together the best papers, articles, news, and open-source libraries highlighting the ML challenges after deployment.

# 📄 License

NannyML is distributed under an Apache License Version 2.0. A complete version can be found [here](LICENSE.MD). All contributions will be distributed under this license.

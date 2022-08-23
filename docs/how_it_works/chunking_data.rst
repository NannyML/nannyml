.. _chunk-data:

Chunking Considerations
=======================


For an introduction on :term:`chunks<Data Chunk>` please have a look at
:ref:`Setting Up: Chunking<chunking>`. This guide focuses on the
potential issues that may arise from using chunks. They are described below.


Not Enough Chunks
-----------------

Sometimes the selected chunking method might not generate enough chunks in the reference period.
NannyML calculates thresholds based on the variability of metrics measured in the ``reference`` chunks (see how thresholds
are calculated for :ref:`performance estimation<performance-estimation-thresholds>`). Having 6 chunks is
far from optimal, but is a reasonable minimum. If there are less than 6 chunks, a warning will be raised.

.. code-block:: python

    >>> cbpe = nml.CBPE(
    ...     y_pred_proba='y_pred_proba',
    ...     y_pred='y_pred',
    ...     y_true='work_home_actual',
    ...     timestamp_column_name='timestamp',
    ...     chunk_number=5,
    ...     metrics=['roc_auc']
    >>> ).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(analysis)
    UserWarning: The resulting number of chunks is too low. Please consider splitting your data in a different way or continue at your own risk.



Not Enough Observations in Chunk: Infeasible Calculations
---------------------------------------------------------

Sometimes selected chunking method may result in some chunks being relatively small. This may lead to a situation
when it is infeasible to calculate some metric on such chunk. Imagine binary classification task with some chunks
that do not contain positive targets at all. In that case performance metrics like precision (True Positives/All
positives) just cannot be calculated. In such situations NannyML will just return ``NaN`` in the results data for
that specific chunk.


.. _sampling-error-introduction:

Too Few Observations in Chunk: Unreliable Calculations
------------------------------------------------------

Small sample size strongly affects the reliability of any ML or statistical analysis, including data drift detection
and performance estimation. NannyML allows splitting data in chunks in different ways to let users choose chunks that
are meaningful for them. However, when the chunks are too small, statistical results may become unreliable. In such
cases results are governed by sampling noise rather than the actual signal. If possible, NannyML quantifies the
amount of variability in calculated statistic/metric coming from sampling only. This variability is expressed
using the most common measure of sampling error - the standard error (standard deviation of the sampling distribution).
Sections below explain what sampling error is and how NannyML estimates it for its different features.


.. _sampling-error-performance-metrics:

Sampling Error: Performance Metrics
+++++++++++++++++++++++++++++++++++

When the chunk size is small what looks like a significant drop in performance of the monitored model may only be a sampling effect.
This effect is relevant for both: calculated and estimated performance. To better understand that, have a look at the
histogram below. It shows distribution of accuracy for a random
model predicting a random binary target (which by definition should be 0.5)
for samples containing 100 observations. It is not uncommon to get accuracy of 0.6 for some samples. The effect is even
stronger for more complex metrics like ROC AUC.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.metrics import accuracy_score

    >>> sample_size = 100
    >>> dataset_size = 10_000
    >>> # random model
    >>> np.random.seed(23)
    >>> y_true = np.random.binomial(1, 0.5, dataset_size)
    >>> y_pred = np.random.binomial(1, 0.5, dataset_size)
    >>> accuracy_scores = []

    >>> for experiment in range(10_000):
    >>>     subset_indexes = np.random.choice(dataset_size, sample_size, replace=False) # get random indexes
    >>>     y_true_subset = y_true[subset_indexes]
    >>>     y_pred_subset = y_pred[subset_indexes]
    >>>     accuracy_scores.append(accuracy_score(y_true_subset, y_pred_subset))

    >>> plt.hist(accuracy_scores, bins=20, density=True)
    >>> plt.title("Accuracy of random classifier\n for randomly selected samples of 100 observations.");

.. image:: ../_static/deep_dive_data_chunks_stability_of_accuracy.svg
    :width: 400pt

When there are many chunks, it is easy to spot the noisy nature of fluctuations. However, with only a few chunks, it
is difficult to tell whether the observed changes are significant or not. In this situation estimation of sampling
error provided by NannyML becomes useful.

Since reference data should always contain targets, standard error is estimated based on it. The easiest way to
calculate standard error for a chunk of size *n* would be to just run thousands of experiments and in each experiment
sample *n* observations from the reference set, calculate performance metric on
that sample and store it. Then we would just calculate standard deviation of the stored distribution of metric values
(exactly the way it is done in the example above). Given the number of experiments is large, this
approach gives precise results but it comes at relatively high computation cost (especially with many chunks of
different sizes). This is why NannyML estimates it instead. Selecting a chunk of data and calculating performance for
it is similar to sampling a set from a population and calculating a statistic. When
the statistic is a mean, the Standard Error of the Mean (SEM) formula [1]_ can be used to estimate the standard
deviation of the sampled means:

    .. math::
        {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}

Let's go through the process of estimating the standard error of accuracy score distribution
from the example above using SEM.
In order to take advantage of the SEM formula, accuracy for each observation separately needs to be calculated.
Accuracy for a single observation is simply equal to 1 when the prediction is correct and equal to 0 otherwise.
With observation-level accuracies (i.e. calculated separately for each observation) in place, accuracy for the whole
sample can be calculated as the mean of them.

.. code-block:: python

    >>> obs_level_accuracy = y_true == y_pred
    >>> np.mean(obs_level_accuracy), accuracy_score(y_true, y_pred)
    (0.5045, 0.5045)

Now SEM formula can be used directly to estimate the standard error of accuracy: :math:`\sigma` from the
formula above is the standard deviation of the observation-level accuracies and :math:`n` is the chunk size.
The code below implements it and compares it with the standard deviation from the direct repeated sampling
experiments above.

.. code-block:: python

    >>> SEM_std = np.std(obs_level_accuracy)/np.sqrt(sample_size)
    >>> np.round(SEM_std, 3), np.round(np.std(accuracy_scores), 3)
    (0.05, 0.05)


So for the analyzed case, the chunk size of 100 observations will result in a standard error of accuracy equal to 0.05.
This dispersion will be purely the effect of sampling because model quality and data distribution remain unchanged.
Generally the SEM formula gives the exact value when:

    * The standard deviation of the population is known.
    * The samples drawn from the population are statistically independent.

Both of these requirements are in fact violated. The true standard deviation of the population is
unknown and we can only use the standard deviation of the reference dataset as a proxy value.
We then treat the chunks as samples of the reference dataset and use the SEM formula accordingly.
In many cases chunks are not independent either e.g. when observations in chunks are selected chronologically, not
randomly. They are also drawn without replacement, meaning the same instance (set of inputs and output) won't be
selected twice. Nevertheless, this approach provides an estimation with good enough precision for our use case while
keeping the computation cost very low.

Another thing to keep in mind is that regardless of the method chosen to calculate it, the standard error is based
on reference data. The only information it takes from the analysis chunk is its size (in case of SEM, it is the
denominator). Therefore, it provides accurate estimations for the analysis period as long as i.i.d holds. Or in other words - it
assumes that *variability* of metric on analysis set will be the same as on reference set.

Estimation of standard error for other classification and regression metrics is performed in a similar manner. The
key is to express a specific metric on observation level in such a way that the mean of observation-level
metrics equals to the value of the metric on the set level.


Sampling Error: Multivariate Drift Detection with PCA
+++++++++++++++++++++++++++++++++++++++++++++++++++++

Standard error for :ref:`Multivariate Drift Detection<multivariate_drift_detection>` is calculated in the exact same way as for
:ref:`Performance Metrics <sampling-error-performance-metrics>`. For each observation the multivariate drift detection
with PCA  process calculates a :term:`reconstruction error<Reconstruction Error>` value. The mean of those values for all observations in a chunk
is the reconstruction error per chunk. The process is described in detail in
:ref:`How it works: Data Reconstruction with PCA Chunking<data-reconstruction-pca>`.
Therefore the standard error of the mean formula can be used without any intermediate steps - to get standard error we just divide standard deviation of
reconstruction error for each observation on the reference dataset with the square root of chunk size of interest.



Sampling Error: Univariate Drift Detection
++++++++++++++++++++++++++++++++++++++++++

Currently :ref:`Univariate Drift Detection<univariate_drift_detection>` for both continuous and categorical variables is
based on two-sample statistical tests. These statistical tests return the value of test static together with the associated p-value.
The p-value takes into account sizes of compared samples and in a sense it contains information about the sampling error. Therefore
additional information about sampling errors is not needed. To make sure you
interpret p-values correctly have a look at the American Statistical Association statement on p-values [2]_.



**References**

.. [1] https://en.wikipedia.org/wiki/Standard_error

.. [2] https://amstat.tandfonline.com/doi/full/10.1080/00031305.2016.1154108#.YvIj6XZBzFe

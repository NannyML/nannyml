.. _estimation_of_standard_error:

Calculating Sampling Error
==========================

This page explains how NannyML calculates :term:`Sampling Error`. The effect of sampling error
on model monitoring results is described in our guide of :ref:`chunk size and reliability of results<sampling-error-introduction>`.

In order to quantify sampling error we rely on :term:`Standard Error`.
The standard error of a statistic is the standard deviation of its sampling distribution. Hence below we will be discussing
how we estimate the standard error and then we will show how we define sampling error from it.

.. _introducing_sem:

Defining Sampling Error from Standard Error of the Mean
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

Let us recall the example of random binary classification model, predicting random binary targets from
our guide of :ref:`chunk size and reliability of results<sampling-error-introduction>`.
The histogram below shows the sampling distribution of accuracy,
which has a true value of 0.5, for samples containing 100 observations.

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

Calculating standard error for the example above is simple. Since the (10 000) sampling experiments are already done
and the accuracy for each sample is stored, in ``accuracy_scores``, it is just a matter of
calculating standard deviation:

.. code-block:: python

    >>>  np.round(np.std(accuracy_scores), 3)
    0.05

With a large enough number of experiments, this approach gives precise results, but it comes with a relatively high computation cost.
There are less precise but significantly faster ways. Selecting a sample (chunk) of data and calculating its performance
is similar to sampling from a population and calculating a statistic.
When the statistic is a mean, the :term:`Standard Error of the Mean (SEM)<Standard Error>` formula [1]_ can be
used to estimate the standard deviation of the sampled means:

    .. math::
        {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}

In order to take advantage of the :term:`SEM formula<Standard Error>` in the analyzed example,
the accuracy of each observation needs to be
calculated in such a way that a mean of this observation-level accuracies equals the accuracy of the whole sample. This
sounds complicated, but the following solution makes it simple.

Accuracy of a single observation is simply equal to 1
when the prediction is correct and equal to 0 otherwise. When the mean of such observation-level accuracies is
calculated, it is equal to the sample-level accuracy, as demonstrated below:

.. code-block:: python

    >>> obs_level_accuracy = y_true == y_pred
    >>> np.mean(obs_level_accuracy), accuracy_score(y_true, y_pred)
    (0.5045, 0.5045)

Now the SEM formula can be used directly to estimate the
standard error of accuracy for a sample of size :math:`n`.

:math:`\sigma`, from the
formula above, is the standard deviation of the observation-level accuracies and :math:`n` is the sample size (chunk
size). The code below calculates standard error with SEM and compares it with the standard error from a
repeated experiments approach:

.. code-block:: python

    >>> SEM_std = np.std(obs_level_accuracy)/np.sqrt(sample_size)
    >>> np.round(SEM_std, 3), np.round(np.std(accuracy_scores), 3)
    (0.05, 0.05)

So for the analyzed case, the sample size of 100 observations will result in a standard error of accuracy equal to 0.05.
This dispersion in measured values will be purely the effect of sampling because model quality and data distribution remain unchanged.

What does this mean when we calculate a statistic from a sample?
It means that when we take a sample of 100 points the accuracy we will calculate has a
68,2% chance of being in the range [0.45, 0.55]. If we extend the range to [0.4, 0.6] then there is a 95% chance of the accuracy
we calculate from the sample being in that range. And if we extend the range to [0.35, 0.65] then there is a 99.7% chance that
the measured accuracy will fall within the specified range. Within NannyML we define :term:`Sampling Error` to be +/- 3
:term:`standard errors<Standard Error>`, and this is the :term:`Confidence Band` that appears as a shaded purple area
in our plots.


Sampling Error Estimation and Interpretation for NannyML features
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Performance Estimation
**********************


As discussed we first calculate :term:`Standard Error` for performance estimation using SEM [1]_ in the way described in
:ref:`previous section<introducing_sem>`. Since targets are available only in the reference dataset,
the nominator of the
SEM formula is calculated based on observation-level metrics from the reference dataset.
The sample size in the denominator is the size of the chunk for which standard error is estimated.

Given that the assumptions of performance estimation methods
are met, the estimated performance is the expected performance of the monitored model on the chunk. Sampling error
informs how much the actual (calculated) performance might be different from the expected one due to sampling effects
only. The sampling error in the results is expressed as 3 standard errors. So the estimated performance +/- 3 standard
errors creates an interval which should contain the actual value of performance metric in about 99% of cases (given
the assumptions of the performance estimation algorithm are met).


Performance Monitoring
**********************

:term:`Standard Error` for realized performance monitoring is calculated using SEM [1]_ in a way described in
:ref:`Adapting Standard Error of the Mean Formula<introducing_sem>`. The nominator of the SEM formula is calculated
based on observation-level metrics from the reference
dataset. The sample size in the denominator is the size of a chunk for which standard error is estimated.

Since realized performance is the actual performance of
the monitored model in the chunk, the standard error has a different interpretation than in estimated performance case.
It informs what the *true performance* of the monitored model might be for a given chunk. In the random model example
:ref:`described above<introducing_sem>` the true accuracy of the model is 0.5. However for some chunks
that contain 100 observations the calculated accuracy can be 0.4, while for other 0.65 etc. This is due to sampling
effects only. NannyML performance
calculation results for these chunks will come together with value of 3 standard errors, which quantifies the
sampling error. For the analyzed example this is equal to 0.15. This tells us that, for
99% of the cases, the true model performance will be found in the +/- 0.15 range from the calculated one. This helps to
evaluate whether performance changes are significant or are just caused by sampling effects.



Multivariate Drift Detection with PCA
*************************************

:term:`Standard Error` for :ref:`Multivariate Drift Detection<multivariate_drift_detection>` is calculated using the approach
introduced in :ref:`Adapting Standard Error of the Mean Formula<introducing_sem>`. For each observation the
multivariate drift detection with PCA process calculates a :term:`reconstruction error<Reconstruction Error>` value.
The mean of those values for all observations in a chunk is the reconstruction error per chunk.
The process is described in detail in :ref:`How it works: Data Reconstruction with PCA Chunking<data-reconstruction-pca>`.
Therefore the standard error of the mean formula can be used without any intermediate steps. We calculate the standard error
of the mean of reconstruction error values within a chunk by dividing the standard deviation of
reconstruction error for each observation on the reference dataset with the square root of the size of the chunk of interest.

Just like in Performance Monitoring, in multivariate drift detection with PCA, the reconstruction error we measure for each chunk
is affected by sampling error and is not the actual reconstruction error of the monitored population.
Again the requirement is that with around 99% certainly we want the true reconstruction error value of the monitored population to be
with in the range of values defined by sampling error. Hence we use +/- 3 standard errors to define our sampling error range.
The validity of sampling error range results are constrained by the :ref:`limitations of our approach<limitations_sem>`
that should be taken in mind when interpreting model monitoring results.


Univariate Drift Detection
**************************

Currently :ref:`Univariate Drift Detection<univariate_drift_detection>` for both continuous and categorical variables is
based on two-sample statistical tests. These statistical tests return the value of the test static together with the associated p-value.
The p-value takes into account sizes of compared samples and in a sense it contains information about the sampling error. Therefore
additional information about sampling errors is not needed. To make sure you
interpret p-values correctly have a look at the American Statistical Association statement on p-values [2]_.

.. _limitations_sem:

Assumptions and Limitations
+++++++++++++++++++++++++++

Generally the :term:`SEM formula<Standard Error>` gives the exact value when:

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
on reference data. The only information it takes from the analysis chunk is its size. Therefore, it provides
accurate estimations for the analysis period as long as the i.i.d (independent and identically distributed) assumption holds.
Or in other words - it assumes that the *variability* of a metric on analysis set will be the same as on reference set.


**References**

.. [1] https://en.wikipedia.org/wiki/Standard_error

.. [2] https://amstat.tandfonline.com/doi/full/10.1080/00031305.2016.1154108#.YvIj6XZBzFe

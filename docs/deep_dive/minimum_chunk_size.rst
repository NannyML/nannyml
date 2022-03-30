.. _minimum-chunk-size:

==================
Minimum chunk size
==================

.. note::
    Not sure what data chunk is in the first place? Read about :term:`Data Chunk` for a short explanation or go through
    the :ref:`guide<chunk-data>` for practical implementations.


Small sample size strongly affects the reliability of any ML or statistical analysis, including data drift detection
and performance estimation. NannyML allows splitting data in chunks in different ways to let users choose chunks that
are meaningful for them. However, when the chunks are too small, statistical results may become unreliable.
In this case NannyML will issue a warning. The user can then chose to ignore it and continue or use a chunking
method that will result in bigger chunks.

Minimum Chunk for Performance Estimation and Performance Calculation
====================================================================

When the chunk size is small
**what looks like a significant drop in performance of the monitored model may be only a sampling effect**.
To better understand that, have a look at the histogram below.
It shows dispersion of accuracy for a random model predicting a random binary target (which by definition should be 0.5)
for a sample of 100 observations. It is not uncommon to get accuracy of 0.6 for some samples. The effect is even
stronger for more complex metrics like AUROC.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> sample_size = 100
    >>> dataset_size = 10_000
    >>> # random model
    >>> y_true = np.random.binomial(1, 0.5, dataset_size)
    >>> y_pred = np.random.binomial(1, 0.5, dataset_size)
    >>> accuracy_scores = []
    >>>
    >>> for experiment in range(10_000):
    >>>     subset_indexes = np.random.choice(dataset_size, sample_size, replace=False) # get random indexes
    >>>     y_true_subset = y_true[subset_indexes]
    >>>     y_pred_subset = y_pred[subset_indexes]
    >>>     accuracy_scores.append(accuracy_score(y_true_subset, y_pred_subset))
    >>>
    >>> plt.hist(accuracy_scores, bins=20, density=True)
    >>> plt.title("Accuracy of random classifier\n for randomly selected samples of 100 observations.");

.. image:: ../_static/deep_dive_data_chunks_stability_of_accuracy.svg
    :width: 400pt

When there are many chunks, it is easy to spot the noisy nature of fluctuations. However, with only a few chunks, it
is difficult to tell whether the observed changes are significant. To minimize this risk, NannyML
estimates a minimum chunk size for the monitored data and raises a warning if the selected chunking method results in
chunks that are smaller. The minimum chunk size is estimated in order to
keep variation of performance of the monitored model low. The variation is expressed in terms of standard deviation and
it is considered *low* when it is below 0.02. In other words, for the selected evaluation metric, NannyML
estimates chunk size for which standard deviation of performance on chunks resulting purely from sampling is lower
than 0.02.

Let's go through the estimation process for accuracy score from the example above. Selecting chunk in the data and
calculating performance for it is similar to sampling a set from a population and calculating a statistic. When
the statistic is a mean, Standard Error (SE) formula [1]_ can be used to estimate the standard deviation of sampled
means:

    .. math::
        {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}

To directly use it for computation of standard deviation of accuracy, the metric needs to be expressed for each
observation in the way that mean of observation-level accuracies gives the whole sample accuracy. Observation-level
accuracy is simply equal to 1 when the prediction is correct and 0 when it is not. Therefore:

.. code-block:: python

    >>> obs_level_accuracy = y_true == y_pred
    >>> np.mean(obs_level_accuracy), accuracy_score(y_true, y_pred)
    (0.4988, 0.4988)

Now SE formula can be used to estimate standard deviation and compare it with standard deviation from sampling
experiments
above:

.. code-block:: python

    >>> SE_std = np.std(obs_level_accuracy)/np.sqrt(sample_size)
    >>> SE_std, np.std(accuracy_scores)
    (0.04999932399543018, 0.04946720594494903)

The same formula can be used to estimate sample size for required standard deviation:

.. code-block:: python

    >>> required_std = 0.02
    >>> sample_size = (np.std(correct_predictions)**2)/required_std**2
    >>> sample_size
    624.99

So for the analyzed case chunk should contain at least 625 observations to keep dispersion of
accuracy on chunks coming from random effect of sampling below 0.02 SD. In the actual implementation the final value
is rounded to full hundredths and limited from the bottom to 300.

Generally SE formula gives the exact value when:

    * standard deviation of the population is known,
    * samples are statistically independent.

Both of these requirements are in fact violated. When data is split into chunks it is not sampled from population -
it comes from a finite set. Therefore standard deviation of **population** is unknown. Moreover, chunks are not
independent - observations in chunks are selected chronologically, not randomly. They are drawn *without replacement* (the same observation
cannot be selected twice). Nevertheless, this approach provides estimation with good enough precision for our use
case while keeping the computation time very low.

Estimation of minimum chunk size for other metrics, such as AUROC, precision, recall etc. is performed in similar
manner.

Minimum Chunk for Data Reconstruction
=====================================

To ensure that there is no significant noise present in data recontruction results NannyML suggests a minimum chunk size
based on the number of features user to perform data reconstruction according to this function:

.. math::

    f(x) = \textrm{Int}( 20 * x ^ {\frac{5}{6}})

The result based on internal testing. It is merely a suggestion because multidimensional data can have difficult to foresee
instabilities. A better suggestion could be derived by inspecting the data used to look for
:ref:`multivariate drift<data-drift-multivariate>` but at the cost of increased computation time.

Minimum Chunk for Univariate Drift
==================================

To ensure that there is no significant noise present in :ref:`Univariate Drift Detection<data-drift-univariate>`
the recommended minimum chunk size is 500. It is a rule of thumb
choice that should cover most common cases. A better suggestion could be derived by inspecting the data used
for Univariate Drift detection but at the cost of increased computation time.


**References**

.. [1] https://en.wikipedia.org/wiki/Standard_error

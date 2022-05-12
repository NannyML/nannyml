.. _chunk-data:

=============
Chunking data
=============

Chunking considerations
----------------------------------

For an introduction on :term:`chunks<Data Chunk>` please have a loot at
:ref:`Setting Up: Chunking<chunking>`.
This guide focuses on the potential issues that may arise from using chunks.
They are described below.

Different periods within one chunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to get performance estimation or data drift results for a dataset that contains two
periods - ``reference`` and ``analysis``, most likely there will be a chunk that contains  observations from both of
them. Such a chunk will be considered as an ``analysis`` chunk, even if only one observation belongs to ``analysis``.
In the example below, chunk which contains observations from 44444 to 55554 is considered an analysis
chunk but indices from 44444 to 49999 point to reference observations:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> reference, analysis, _ = nml.datasets.load_synthetic_binary_classification_dataset()
    >>> metadata = nml.extract_metadata(reference, model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_number=9, metrics=['roc_auc']).fit(reference_data=reference)
    >>> # Estimate on concatenated reference and analysis
    >>> est_perf = cbpe.estimate(pd.concat([reference, analysis]))
    >>> est_perf.data.iloc[3:5,:7]


+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   estimated_roc_auc |
+====+===============+===============+=============+=====================+=====================+=============+=====================+
|  3 | [33333:44443] |         33333 |       44443 | 2016-07-25 00:00:00 | 2017-04-19 23:59:59 | reference   |            0.968876 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+
|  4 | [44444:55554] |         44444 |       55554 | 2017-04-19 00:00:00 | 2018-01-15 23:59:59 | analysis    |            0.968921 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+

.. code-block:: python

    >>> reference.index.max()
    49999

.. note::
    This is especially important for Performance Estimation. Since the Performance Estimation algorithm is calibrated
    on the ``reference`` dataset (see :ref:`PE deep dive <performance-estimation-deep-dive>`), it will perform better on
    it. If the first ``analysis`` chunk contains ``reference`` data, the performance estimation may perform better on this
    chunk as well. Keep this in mind when interpreting the results.


Underpopulated chunks
~~~~~~~~~~~~~~~~~~~~~

Depending on the selected chunking method and the provided datasets, some chunks may not have a lot of observations
inside them. In fact, they might be so small that results obtained are governed by noise rather than actual signal.
NannyML estimates a minimum chunk size based on the reference data
(see how in the :ref:`section on minimum chunk size<minimum-chunk-size>`). If some of the created chunks
are smaller than the minimum chunk size, a warning will be raised. For example:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_period="Q", metrics=['roc_auc']).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(analysis)
    UserWarning: The resulting list of chunks contains 1 underpopulated chunks. They contain too few records to be
    statistically relevant and might negatively influence the quality of calculations. Please consider splitting
    your data in a different way or continue at your own risk.

When the warning is about a single chunk, it is usually the last chunk and this is due to the reasons described in
:ref:`Setting Up: Chunking<chunking>`.
When there are more than one underpopulated chunks staying with the selected chunking method
may be suboptimal.
Read :ref:`minimum chunk size <minimum-chunk-size>` to get more information about the effect of
small chunks. Beware of the trade-offs involved, when selecting the chunking method.


Not enough chunks
~~~~~~~~~~~~~~~~~

Sometimes the selected chunking method might not genereate enough chunks in the rerfence period.
NannyML calculates thresholds based on the variability of metrics measured in the ``reference`` chunks (see how thresholds
are calculated for :ref:`performance estimation<performance-estimation-thresholds>`). Having 6 chunks is
far from optimal but a reasonable minimum. If there are less than 6 chunks, a warning will be raised:

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_number=5, metrics=['roc_auc']).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(analysis)
    UserWarning: The resulting number of chunks is too low. Please consider splitting your data in a different way or
    continue at your own risk.


.. _minimum-chunk-size:

Minimum chunk size
------------------

Small sample size strongly affects the reliability of any ML or statistical analysis, including data drift detection
and performance estimation. NannyML allows splitting data in chunks in different ways to let users choose chunks that
are meaningful for them. However, when the chunks are too small, statistical results may become unreliable.
In this case NannyML will issue a warning. The user can then chose to ignore it and continue or use a chunking
method that will result in bigger chunks.

.. _chunk-data-minimum-chunk:

Minimum Chunk for Performance Estimation and Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the chunk size is small
**what looks like a significant drop in performance of the monitored model may only be a sampling effect**.
To better understand that, have a look at the histogram below.
It shows dispersion of accuracy for a random model predicting a random binary target (which by definition should be 0.5)
for a sample of 100 observations. It is not uncommon to get accuracy of 0.6 for some samples. The effect is even
stronger for more complex metrics like ROC AUC.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.metrics import accuracy_score

    >>> sample_size = 100
    >>> dataset_size = 10_000
    >>> # random model
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
is difficult to tell whether the observed changes are significant. To minimize this risk, NannyML
estimates a minimum chunk size for the monitored data and raises a warning if the selected chunking method results in
chunks that are smaller. The minimum chunk size is estimated in order to
keep variation of performance of the monitored model low. The variation is expressed in terms of standard deviation and
it is considered *low* when it is below 0.02. In other words, for the selected evaluation metric, NannyML
estimates a chunk size for which the standard deviation of performance on chunks resulting purely from sampling is lower
than 0.02.

Let's go through the estimation process for accuracy score from the example above. Selecting a chunk in the data and
calculating performance for it is similar to sampling a set from a population and calculating a statistic. When
the statistic is a mean, the Standard Error (SE) formula [1]_ can be used to estimate the standard deviation of
the sampled means:

    .. math::
        {\sigma }_{\bar {x}}\ ={\frac {\sigma }{\sqrt {n}}}

In order to take advantage of the SE formula, accuracy for each observation separately needs to be calculated.
Accuracy for a single observation is simply equal to 1 when the prediction is correct and equal to 0 otherwise.
With observation-level accuracies in place, accuracy for the whole sample can be calculated as the mean of them.
After this transformation the SE formula can be used directly to estimate the standard error of accuracy as a
function of sample:

.. code-block:: python

    >>> obs_level_accuracy = y_true == y_pred
    >>> np.mean(obs_level_accuracy), accuracy_score(y_true, y_pred)
    (0.4988, 0.4988)

Now the SE formula can be used to estimate the standard deviation and compare it with
the standard deviation from the sampling experiments above:

.. code-block:: python

    >>> SE_std = np.std(obs_level_accuracy)/np.sqrt(sample_size)
    >>> SE_std, np.std(accuracy_scores)
    (0.04999932399543018, 0.04946720594494903)

The same formula can be used to estimate the sample size for the required standard deviation:

.. code-block:: python

    >>> required_std = 0.02
    >>> sample_size = (np.std(correct_predictions)**2)/required_std**2
    >>> sample_size
    624.99

So for the analyzed case, the chunk size of 625 observations will result with standard error of accuracy equal to 0.02.
In other words, if we calculate accuracy of this model on a large number of samples with 625 observations each,
standard deviation of these accuracies will be about 0.02. This dispersion will be purely the effect of sampling
because model quality and data distribution remain unchanged. In the current NannyML implementation, the estimated chunk
size is rounded to full hundredths, 600 in the example above. Additionally, if the estimation returns a number lower
than 300, the minimum chunk size suggested is 300.

Generally the SE formula gives the exact value when:

    * The standard deviation of the population is known,
    * The samples are statistically independent.

Both of these requirements are in fact violated. When the data is split into chunks it is not sampled from population,
it comes from a finite set. Therefore standard deviation of **population** is unknown. Moreover, chunks are not
independent - observations in chunks are selected chronologically, not randomly. They are also drawn *without replacement*,
meaning the same observation cannot be selected twice. Nevertheless, this approach provides an estimation with good enough
precision for our use case while keeping the computation time very low.

Estimation of minimum chunk size for other metrics, such as ROC AUC, precision, recall etc. is performed in similar
manner.

Minimum Chunk for Multivariate Drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure that there is no significant noise present in :ref:`multivariate drift<multivariate_drift_detection>`
results NannyML suggests a minimum chunk size
based on the number of features used to perform data reconstruction according to this function:

.. math::

    f(x) = \textrm{Int}( 20 * x ^ {\frac{5}{6}})

This result is based on internal testing. It is merely a suggestion because multidimensional data can have difficult to foresee
instabilities.

Minimum Chunk for Univariate Drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure that there is no significant noise present in :ref:`Univariate Drift Detection<univariate_drift_detection>`
the recommended minimum chunk size is 500. It is a rule of thumb
that should cover most common cases.


**References**

.. [1] https://en.wikipedia.org/wiki/Standard_error

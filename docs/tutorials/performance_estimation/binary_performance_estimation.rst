.. _binary-performance-estimation:

========================================================================================
Estimating Performance for Binary Classification
========================================================================================

Why Perform Performance Estimation
============================================

NannyML allows estimating the performance of a classification model when :term:`targets<Target>` are absent.
This can be very helpful when targets are delayed, only partially available, or not available at all, because 
it allows you to potentially identify performance issues before they would otherwise be detected. 

Some specific examples of when you could benefit from estimating your performance include:

- When predicting loan defaults, to estimate model performance before the end of the repayment periods.
- When performing sentiment analysis, targets may be entirely unavailable without significant human effort, so estimation is the only feasible way to attain metrics.
- When dealing with huge datasets, where human verification can only cover a small sample, estimation of performance can help confirm confidence or question the efficacy.

This tutorial explains how to use NannyML to estimate the performance of binary classification 
models in the absence of target data. To find out how CBPE estimates performance, read the :ref:`explanation of Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.

.. _performance-estimation-binary-just-the-code:

Just The Code
----------------

.. code-block:: python

  import pandas as pd
  import nannyml as nml
  from IPython.display import display

  reference_df = nml.load_synthetic_binary_classification_dataset()[0]
  analysis_df = nml.load_synthetic_binary_classification_dataset()[1]

  display(reference_df.head(3))

  estimator = nml.CBPE(
      y_pred_proba='y_pred_proba',
      y_pred='y_pred',
      y_true='y_true',
      timestamp_column_name='timestamp',
      metrics=['roc_auc', 'f1']
      chunk_size=5000,
  )

  estimator.fit(reference_df)

  results = estimator.estimate(analysis_df)

  display(results.data.head(3))

  for metric in estimator.metrics:
      fig1 = results.plot(kind='performance', metric=metric)
      fig1.show()

  for metric in estimator.metrics:
      fig2 = results.plot(kind='performance', plot_reference=True, metric=metric)
      fig2.show()


Walkthrough
--------------

Prepare the data
^^^^^^^^^^^^^^^^^^

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model predicts
whether an employee will work from home. You can :ref:`read more about this synthetic dataset<dataset-synthetic-binary>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`

.. code-block:: python

    import pandas as pd
    import nannyml as nml
    from IPython.display import display

    reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    analysis_df = nml.load_synthetic_binary_classification_dataset()[1]

    display(reference_df.head(3))

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+


Next we create the Confidence-based Performance Estimation
(:class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`)
estimator with a list of metrics, and an optional
:ref:`chunking<chunking>` specification. 

The list of metrics specifies which performance metrics of the monitored model will be estimated. 
The following metrics are currently supported:

- ``roc_auc`` - one vs. the rest, macro averaged
- ``f1`` - macro averaged
- ``precision`` - macro averaged
- ``recall`` - macro averaged
- ``specificity`` - macro averaged
- ``accuracy``

For more information about :term:`chunking<Data Chunk>` you can check the :ref:`setting up page<chunking>` and :ref:`advanced guide<chunk-data>`.
.. code-block:: python

    estimator = nml.CBPE(
      y_pred_proba='y_pred_proba',
      y_pred='y_pred',
      y_true='y_true',
      timestamp_column_name='timestamp',
      metrics=['roc_auc', 'f1']
      chunk_size=5000,
    )

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
estimator is then fitted using the
:meth:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE.fit` method on the ``reference`` data.

The fitted ``cbpe`` can be used to estimate performance on other data, for which performance cannot be calculated.
Typically, this would be used on the latest production data where target is missing. In our example this is
the ``analysis_df`` data.

NannyML can then output a dataframe that contains all the results. Let's have a look at the results for analysis period
only.

.. _performance-estimation-thresholds:

Apart from chunking and chunk and partition-related data, the results data have the following columns for each metric
that was estimated:

 - ``estimated_<metric>`` - the estimate of a metric for a specific chunk,
 - ``confidence_<metric>`` - the width of the confidence band. It is equal to 1 standard deviation of performance estimates on
   `reference` data (hence calculated during ``fit`` phase).
 - ``upper_threshold_<metric>`` and ``lower_threshold_<metric>`` - crossing these thresholds will raise an alert on significant
   performance change. The thresholds are calculated based on the actual performance of the monitored model on chunks in
   the ``reference`` partition. The thresholds are 3 standard deviations away from the mean performance calculated on
   chunks.
   They are calculated during ``fit`` phase.
 - ``realized_<metric>`` - when ``target`` values are available for a chunk, the realized performance metric will also
   be calculated and included within the results.
 - ``alert_<metric>`` - flag indicating potentially significant performance change. ``True`` if estimated performance crosses
   upper or lower threshold.

   .. code-block:: python

  estimator.fit(reference_df)

  results = estimator.estimate(analysis_df)

  display(results.data.head(3))

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   confidence_roc_auc |   realized_roc_auc |   estimated_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |   confidence_f1 |   realized_f1 |   estimated_f1 |   upper_threshold_f1 |   lower_threshold_f1 | alert_f1   |
+====+===============+===============+=============+=====================+=====================+=============+======================+====================+=====================+===========================+===========================+=================+=================+===============+================+======================+======================+============+
|  0 | [0:4999]      |             0 |        4999 | 2017-08-31 04:20:00 | 2018-01-02 00:45:44 | analysis    |           0.00035752 |                nan |            0.968631 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.948555 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2018-01-02 01:13:11 | 2018-05-01 13:10:10 | analysis    |           0.00035752 |                nan |            0.969044 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.946578 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|  2 | [10000:14999] |         10000 |       14999 | 2018-05-01 14:25:25 | 2018-09-01 15:40:40 | analysis    |           0.00035752 |                nan |            0.969444 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.948807 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+


These results can be also plotted. Our plot contains several key elements.

* The purple dashed step plot shows the estimated performance in each chunk of the analysis period. Thick squared point
  marker indicates the middle of this period.

* The solid, low-saturated purple line *behind* indicates the confidence band.

* The red horizontal dashed lines show upper and lower thresholds.

* If the estimated performance crosses the upper or lower threshold an alert is raised which is indicated with a red,
  low-saturated background in the whole width of the relevant chunk. This is additionally
  indicated by a red point marker in the middle of the chunk.

Description of tabular results above explains how the
confidence bands and thresholds are calculated. Additional information is shown in the hover (these are
interactive plots, though only static views are included here).

.. code-block:: python

    for metric in estimator.metrics:
      fig1 = results.plot(kind='performance', metric=metric)
      fig1.show()


.. image:: ../../_static/tutorial-perf-est-guide-analysis-roc_auc.svg

.. image:: ../../_static/tutorial-perf-est-guide-analysis-f1.svg


To get a better context let's also plot estimation of performance on analysis data together with calculated
performance on the reference period (where the target was available).

* The right-hand side of the plot shows the estimated performance for the analysis period, as before.

* The purple dashed vertical line splits the reference and analysis periods.

* On the left-hand side of the line, the actual model performance (not estimation!) is plotted with a solid light blue
  line. This facilitates comparison of the estimation against the reference period, and sets expectations on the 
  variability of the performance.

.. code-block:: python

    for metric in estimator.metrics:
      fig2 = results.plot(kind='performance', plot_reference=True, metric=metric)
      fig2.show()


.. image:: ../../_static/tutorial-perf-est-guide-with-ref-roc_auc.svg

.. image:: ../../_static/tutorial-perf-est-guide-with-ref-f1.svg


Insights
==========================

After reviewing the performance estimation results, we should be able to see any indications of performance change that
NannyML has detected based upon the model's inputs and outputs alone.


What's next
==========================

The :ref:`Data Drift<data-drift>` functionality can help us to understand whether data drift is causing the performance problem. 
When the target results become available they can be :ref:`compared with the estimated results<compare_estimated_and_realized_performance>`. 

You can learn more about the Confidence Based Performance Estimation and its limitations in the
:ref:`How it Works page<performance-estimation-deep-dive>`
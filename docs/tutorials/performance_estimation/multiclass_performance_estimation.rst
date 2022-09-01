.. _multiclass-performance-estimation:

========================================================================================
Estimating Performance for Multiclass Classification
========================================================================================

This tutorial explains how to use NannyML to estimate the performance of multiclass classification
models in the absence of target data. To find out how CBPE estimates performance, read the :ref:`explanation of Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.

Just The Code
-------------

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]

    >>> display(reference_df.head(3))

    >>> estimator = nml.CBPE(
    ...     y_pred_proba={
    ...       'prepaid_card': 'y_pred_proba_prepaid_card',
    ...       'highstreet_card': 'y_pred_proba_highstreet_card',
    ...       'upmarket_card': 'y_pred_proba_upmarket_card'},
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='classification_multiclass',
    ...     metrics=['roc_auc', 'f1'],
    ...     chunk_size=6000,
    >>> )

    >>> estimator.fit(reference_df)

    >>> results = estimator.estimate(analysis_df)

    >>> display(results.data.head(3))

    >>> for metric in estimator.metrics:
    ...     fig1 = results.plot(kind='performance', metric=metric)
    ...     fig1.show()

    >>> for metric in estimator.metrics:
    ...     fig2 = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     fig2.show()


Walkthrough
------------------------



For simplicity the guide is based on a synthetic dataset where the monitored model predicts
which type of credit card product new customers should be assigned to. You can :ref:`learn more about this dataset<dataset-synthetic-multiclass>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]

    >>> display(reference_df.head(3))


+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|    | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | partition   |   identifier | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred       | y_true        |
+====+===============+========================+==========================+===============+=======================+=================+===============+=============+==============+=====================+=============================+================================+==============================+==============+===============+
|  0 | Partner3      |               1.80823  |                      350 | web           |                   309 |           15000 | True          | reference   |        60000 | 2020-05-02 02:01:30 |                        0.97 |                           0.03 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  1 | Partner2      |               4.38257  |                      500 | mobile        |                   418 |           23000 | True          | reference   |        60001 | 2020-05-02 02:03:33 |                        0.87 |                           0.13 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  2 | Partner2      |              -0.787575 |                      400 | web           |                   507 |           24000 | False         | reference   |        60002 | 2020-05-02 02:04:49 |                        0.47 |                           0.35 |                         0.18 | prepaid_card | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+

Next we create the Confidence-based Performance Estimation
(:class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`)
estimator with a list of metrics, and an optional
:ref:`chunking<chunking>` specification.

The list of metrics specifies which performance metrics of the monitored model will be estimated.
The following metrics are currently supported:

- ``roc_auc`` - one-vs-the-rest, macro-averaged
- ``f1`` - macro-aveaged
- ``precision`` - macro-averaged
- ``recall`` - macro-averaged
- ``specificity`` - macro-averaged
- ``accuracy``

For more information about :term:`chunking<Data Chunk>` you can check the :ref:`setting up page<chunking>` and :ref:`advanced guide<chunk-data>`.

.. code-block:: python

    >>> estimator = nml.CBPE(
    ...     y_pred_proba={
    ...         'prepaid_card': 'y_pred_proba_prepaid_card',
    ...         'highstreet_card': 'y_pred_proba_highstreet_card',
    ...         'upmarket_card': 'y_pred_proba_upmarket_card'},
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='classification_multiclass',
    ...     metrics=['roc_auc', 'f1'],
    ...     chunk_size=6000,
    >>> )
    >>> estimator.fit(reference_df)

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
estimator is then fitted using the
:meth:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE.fit` method on the ``reference`` data.

The fitted ``cbpe`` can be used to estimate performance on other data, for which performance cannot be calculated.
Typically, this would be used on the latest production data where target is missing. In our example this is
the ``analysis_df`` data.

NannyML can then output a dataframe that contains all the results. Let's have a look at the results for analysis period
only.

.. code-block:: python

  >>> results = estimator.estimate(analysis_df)
  >>> display(results.data.head(3))


+----+---------------+---------------+-------------+---------------------+---------------------+--------------------+---------------------+----------------------------+----------------------------+---------------------------+---------------------------+-----------------+---------------+----------------+-----------------------+-----------------------+----------------------+----------------------+------------+
|    | key           |   start_index |   end_index | start_date          | end_date            |   realized_roc_auc |   estimated_roc_auc |   upper_confidence_roc_auc |   lower_confidence_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |   realized_f1 |   estimated_f1 |   upper_confidence_f1 |   lower_confidence_f1 |   upper_threshold_f1 |   lower_threshold_f1 | alert_f1   |
+====+===============+===============+=============+=====================+=====================+====================+=====================+============================+============================+===========================+===========================+=================+===============+================+=======================+=======================+======================+======================+============+
|  0 | [0:5999]      |             0 |        5999 | 2020-09-01 03:10:01 | 2020-09-13 16:15:10 |                nan |            0.907037 |                   0.907864 |                   0.90621  |                  0.900902 |                  0.913516 | False           |           nan |       0.753301 |              0.755053 |              0.75155  |             0.741254 |             0.764944 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+--------------------+---------------------+----------------------------+----------------------------+---------------------------+---------------------------+-----------------+---------------+----------------+-----------------------+-----------------------+----------------------+----------------------+------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2020-09-13 16:15:32 | 2020-09-25 19:48:42 |                nan |            0.909948 |                   0.910776 |                   0.909121 |                  0.900902 |                  0.913516 | False           |           nan |       0.756422 |              0.758173 |              0.75467  |             0.741254 |             0.764944 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+--------------------+---------------------+----------------------------+----------------------------+---------------------------+---------------------------+-----------------+---------------+----------------+-----------------------+-----------------------+----------------------+----------------------+------------+
|  2 | [12000:17999] |         12000 |       17999 | 2020-09-25 19:50:04 | 2020-10-08 02:53:47 |                nan |            0.909958 |                   0.910786 |                   0.909131 |                  0.900902 |                  0.913516 | False           |           nan |       0.758166 |              0.759917 |              0.756414 |             0.741254 |             0.764944 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+--------------------+---------------------+----------------------------+----------------------------+---------------------------+---------------------------+-----------------+---------------+----------------+-----------------------+-----------------------+----------------------+----------------------+------------+

Apart from chunk-related data, the results data have the following columns for each metric
that was estimated:

 - ``realized_<metric>`` - when ``target`` values are available for a chunk, the realized performance metric will also
   be calculated and included within the results.
 - ``estimated_<metric>`` - the estimate of a metric for a specific chunk,
 - ``upper_confidence_<metric>`` and ``lower_confidence_<metric>`` - These values show the :term:`Confidence Band` of the relevant metric
   and are equal to estimated value +/- 3 times the estimated :term:`Sampling Error`.
 - ``upper_threshold_<metric>`` and ``lower_threshold_<metric>`` - crossing these thresholds will raise an alert on significant
   performance change. The thresholds are calculated based on the actual performance of the monitored model on chunks in
   the ``reference`` partition. The thresholds are 3 standard deviations away from the mean performance calculated on
   chunks.
   They are calculated during ``fit`` phase.
 - ``alert_<metric>`` - flag indicating potentially significant performance change. ``True`` if estimated performance crosses
   upper or lower threshold.


These results can be also plotted. Our plot contains several key elements.

* The purple dashed step plot shows the estimated performance in each chunk of the analysis period. Thick squared point
  markers indicate the middle of these chunks.

* The low-saturated purple area around the estimated performance indicates the :ref:`sampling error<estimation_of_standard_error>`.

* The red horizontal dashed lines show upper and lower thresholds for alerting purposes.

* If the estimated performance crosses the upper or lower threshold an alert is raised which is indicated with a red,
  low-saturated background in the whole width of the relevant chunk. This is additionally
  indicated by a red, diamond-shaped point marker in the middle of the chunk.

Description of tabular results above explains how the
:term:`confidence bands<Confidence Band>` and thresholds are calculated. Additional information is shown in the hover (these are
interactive plots, though only static views are included here).


.. code-block:: python

    >>> for metric in estimator.metrics:
    ...     fig1 = results.plot(kind='performance', metric=metric)
    ...     fig1.show()


.. image:: ../../_static/tutorial-perf-est-mc-guide-analysis-roc_auc.svg

.. image:: ../../_static/tutorial-perf-est-mc-guide-analysis-f1.svg

To get a better context let's additionally plot estimation of performance on analysis data together with calculated
performance on reference period (where the target was available).

* The right-hand side of the plot shows the estimated performance for the
  analysis period as before.

* The purple dashed vertical line splits the reference and analysis periods.

* On the left-hand side of the line, the actual model performance (not estimation!) is plotted with a solid light blue
  line. This facilitates interpretation of the estimation, as it helps to set expectations on the variability of
  the realised performance.

.. code-block:: python

    >>> for metric in estimator.metrics:
    ...     fig2 = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     fig2.show()


.. image:: ../../_static/tutorial-perf-est-mc-guide-with-ref-roc_auc.svg

.. image:: ../../_static/tutorial-perf-est-mc-guide-with-ref-f1.svg


Insights
--------

After reviewing the performance estimation results, we should be able to see any indications of performance change that
NannyML has detected based upon the model's inputs and outputs alone.


What's next
-----------

The :ref:`Data Drift<data-drift>` functionality can help us to understand whether data drift is causing the performance problem.
When the target results become available they can be :ref:`compared with the estimated results<compare_estimated_and_realized_performance>`.

You can learn more about the Confidence Based Performance Estimation and its limitations in the
:ref:`How it Works page<performance-estimation-deep-dive>`.

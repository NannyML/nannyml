.. _multiclass-performance-calculation:

================================================================
Monitoring Realized Performance for Multiclass Classification
================================================================


Just The Code
==============

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_multiclass_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')

    >>> display(reference_df.head(3))

    >>> calc = nml.PerformanceCalculator(
    ...     y_pred_proba={
    ...         'prepaid_card': 'y_pred_proba_prepaid_card',
    ...         'highstreet_card': 'y_pred_proba_highstreet_card',
    ...         'upmarket_card': 'y_pred_proba_upmarket_card'
    ...     },
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['f1', 'roc_auc'],
    ...     chunk_size=6000)

    >>> calc.fit(reference_df)

    >>> results = calc.calculate(analysis_df)

    >>> display(results.data.head(3))

    >>> for metric in calc.metrics:
    ...     figure = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     figure.show()



Walkthrough
=============


For simplicity the guide is based on a synthetic dataset where the monitored model predicts
which type of credit card product new customers should be assigned to. You can :ref:`learn more about this dataset<dataset-synthetic-multiclass>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to calculate performance, so the first thing we need to in this case is set up the right data in the right dataframes.  The analysis target values are joined on the analysis frame by the ``identifier`` column.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_multiclass_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')

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


Next a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using a list of metrics to calculate (or just one metric), the data columns required for these metrics, and an optional :ref:`chunking<chunking>` specification.

The list of metrics specifies which performance metrics of the monitored model will be calculated.
The following metrics are currently supported:

- ``roc_auc`` - one-vs-the-rest, macro-averaged
- ``f1`` - macro-averaged
- ``precision`` - macro-averaged
- ``recall`` - macro-averaged
- ``specificity`` - macro-averaged
- ``accuracy``

For more information on metrics, check the :mod:`~nannyml.performance_calculation.metrics` module.

.. code-block:: python

    >>> calc = nml.PerformanceCalculator(
    ...     y_pred_proba={
    ...         'prepaid_card': 'y_pred_proba_prepaid_card',
    ...         'highstreet_card': 'y_pred_proba_highstreet_card',
    ...         'upmarket_card': 'y_pred_proba_upmarket_card'
    ...     },
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['f1', 'roc_auc'],
    ...     chunk_size=6000)

    >>> calc.fit(reference_df)


The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available.

.. code-block:: python

    >>> results = calc.calculate(analysis_df)
    >>> display(results.data.head(3))


+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+----------+----------------------+----------------------+------------+-----------+---------------------------+---------------------------+-----------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | period   |   targets_missing_rate |       f1 |   f1_lower_threshold |   f1_upper_threshold | f1_alert   |   roc_auc |   roc_auc_lower_threshold |   roc_auc_upper_threshold | roc_auc_alert   |
+====+===============+===============+=============+=====================+=====================+==========+========================+==========+======================+======================+============+===========+===========================+===========================+=================+
|  0 | [0:5999]      |             0 |        5999 | 2020-09-01 03:10:01 | 2020-09-13 16:15:10 |          |                      0 | 0.751103 |             0.741254 |             0.764944 | False      |  0.907595 |                  0.900902 |                  0.913516 | False           |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+----------+----------------------+----------------------+------------+-----------+---------------------------+---------------------------+-----------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2020-09-13 16:15:32 | 2020-09-25 19:48:42 |          |                      0 | 0.763045 |             0.741254 |             0.764944 | False      |  0.910534 |                  0.900902 |                  0.913516 | False           |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+----------+----------------------+----------------------+------------+-----------+---------------------------+---------------------------+-----------------+
|  2 | [12000:17999] |         12000 |       17999 | 2020-09-25 19:50:04 | 2020-10-08 02:53:47 |          |                      0 | 0.758487 |             0.741254 |             0.764944 | False      |  0.909414 |                  0.900902 |                  0.913516 | False           |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+----------+----------------------+----------------------+------------+-----------+---------------------------+---------------------------+-----------------+

NannyML can output a dataframe that contains all the results.

Apart from chunking and chunk and period-related columns, the results data have the a set of columns for each
calculated metric. When taking ``roc_auc`` as an example:

 - ``targets_missing_rate`` - The fraction of missing target data.
 - ``<metric>`` - The value of the metric for a specific chunk.
 - ``<metric>_lower_threshold>`` and ``<metric>_upper_threshold>`` - Lower and upper thresholds for performance metric.
   Crossing them will raise an alert that there is a significant
   metric change. The thresholds are calculated based on the realized performance of chunks in the ``reference`` period.
   The thresholds are 3 standard deviations away from the mean performance calculated on ``reference`` chunks.
 - ``<metric>_alert`` - A flag indicating potentially significant performance change. ``True`` if realized performance
   crosses
   upper or lower threshold.

The results can be plotted for visual inspection:

.. code-block:: python

    >>> for metric in calc.metrics:
    ...     figure = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     figure.show()


.. image:: /_static/tutorial-perf-guide-mc-F1.svg

.. image:: /_static/tutorial-perf-guide-mc-ROC_AUC.svg


Insights
=======================

After reviewing the performance calculation results, we should be able to clearly see how the model is performing against
the targets, according to whatever metrics we wish to track.



What Next
=======================

If we decide further investigation is needed, the :ref:`Data Drift<data-drift>` functionality can help us to see
what feature changes may be contributing to any performance changes.

It is also wise to check whether the model's performance is satisfactory
according to business requirements. This is an ad-hoc investigation that is not covered by NannyML.

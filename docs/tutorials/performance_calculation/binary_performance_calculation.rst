.. _binary-performance-calculation:

================================================================
Monitoring Realized Performance for Binary Classification
================================================================

Just The Code
==============

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_binary_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')

    >>> display(reference_df.head(3))

    >>> calc = nml.PerformanceCalculator(
    ...     y_pred_proba='y_pred_proba',
    ...     y_pred='y_pred',
    ...     y_true='work_home_actual',
    ...     timestamp_column_name='timestamp',
    ...     metrics=nml.performance_estimation.confidence_based.results.SUPPORTED_METRIC_VALUES,
    ...     chunk_size=5000)

    >>> calc.fit(reference_df)

    >>> results = calc.calculate(analysis_df)

    >>> display(results.data.head(3))

    >>> for metric in calc.metrics:
    ...     figure = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     figure.show()


Walkthrough
===============

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model predicts
whether an employee will work from home. You can :ref:`read more about this synthetic dataset<dataset-synthetic-binary>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to calculate performance, so the first thing we need to in this case is set up the right data in the right dataframes.  The analysis target values are joined on the analysis frame by the ``identifier`` column.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_binary_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')

    >>> display(reference_df.head(3))

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+


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
    >>>     y_pred_proba='y_pred_proba',
    >>>     y_pred='y_pred',
    >>>     y_true='work_home_actual',
    >>>     timestamp_column_name='timestamp',
    >>>     metrics=nml.performance_estimation.confidence_based.results.SUPPORTED_METRIC_VALUES,
    >>>     chunk_size=5000)

    >>> calc.fit(reference_df)

The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available.

NannyML can output a dataframe that contains all the results.

.. code-block:: python

    >>> results = calc.calculate(analysis_df)
    >>> display(results.data.head(3))


+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------+---------------------------+---------------------------+-----------------+----------+----------------------+----------------------+------------+-------------+-----------------------------+-----------------------------+-------------------+----------+--------------------------+--------------------------+----------------+---------------+-------------------------------+-------------------------------+---------------------+------------+----------------------------+----------------------------+------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | period   |   targets_missing_rate |   roc_auc |   roc_auc_lower_threshold |   roc_auc_upper_threshold | roc_auc_alert   |       f1 |   f1_lower_threshold |   f1_upper_threshold | f1_alert   |   precision |   precision_lower_threshold |   precision_upper_threshold | precision_alert   |   recall |   recall_lower_threshold |   recall_upper_threshold | recall_alert   |   specificity |   specificity_lower_threshold |   specificity_upper_threshold | specificity_alert   |   accuracy |   accuracy_lower_threshold |   accuracy_upper_threshold | accuracy_alert   |
+====+===============+===============+=============+=====================+=====================+==========+========================+===========+===========================+===========================+=================+==========+======================+======================+============+=============+=============================+=============================+===================+==========+==========================+==========================+================+===============+===============================+===============================+=====================+============+============================+============================+==================+
|  0 | [0:4999]      |             0 |        4999 | 2017-08-31 04:20:00 | 2018-01-02 00:45:44 |          |                      0 |  0.970962 |                  0.963317 |                   0.97866 | False           | 0.949549 |             0.935047 |             0.961094 | False      |    0.942139 |                    0.924741 |                    0.961131 | False             | 0.957077 |                 0.940831 |                 0.965726 | False          |      0.937034 |                      0.924741 |                      0.960113 | False               |     0.9474 |                   0.935079 |                   0.960601 | False            |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------+---------------------------+---------------------------+-----------------+----------+----------------------+----------------------+------------+-------------+-----------------------------+-----------------------------+-------------------+----------+--------------------------+--------------------------+----------------+---------------+-------------------------------+-------------------------------+---------------------+------------+----------------------------+----------------------------+------------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2018-01-02 01:13:11 | 2018-05-01 13:10:10 |          |                      0 |  0.970248 |                  0.963317 |                   0.97866 | False           | 0.946686 |             0.935047 |             0.961094 | False      |    0.943434 |                    0.924741 |                    0.961131 | False             | 0.949959 |                 0.940831 |                 0.965726 | False          |      0.944925 |                      0.924741 |                      0.960113 | False               |     0.9474 |                   0.935079 |                   0.960601 | False            |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------+---------------------------+---------------------------+-----------------+----------+----------------------+----------------------+------------+-------------+-----------------------------+-----------------------------+-------------------+----------+--------------------------+--------------------------+----------------+---------------+-------------------------------+-------------------------------+---------------------+------------+----------------------------+----------------------------+------------------+
|  2 | [10000:14999] |         10000 |       14999 | 2018-05-01 14:25:25 | 2018-09-01 15:40:40 |          |                      0 |  0.976282 |                  0.963317 |                   0.97866 | False           | 0.950459 |             0.935047 |             0.961094 | False      |    0.941438 |                    0.924741 |                    0.961131 | False             | 0.959654 |                 0.940831 |                 0.965726 | False          |      0.943602 |                      0.924741 |                      0.960113 | False               |     0.9514 |                   0.935079 |                   0.960601 | False            |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------+---------------------------+---------------------------+-----------------+----------+----------------------+----------------------+------------+-------------+-----------------------------+-----------------------------+-------------------+----------+--------------------------+--------------------------+----------------+---------------+-------------------------------+-------------------------------+---------------------+------------+----------------------------+----------------------------+------------------+

Apart from chunking and chunk and period-related columns, the results data have a set of columns for each
calculated metric. When taking ``roc_auc`` as an example:

 - ``targets_missing_rate`` - The fraction of missing target data.
 - ``<metric>`` - The value of the metric for a specific chunk.
 - ``<metric>_thresholds`` - A tuple containing the lower and upper thresholds. Crossing them will raise an alert that
   there is a significant
   metric change. The thresholds are calculated based on the realized performance of chunks in    the ``reference`` period.
   The thresholds are 3 standard deviations away from the mean performance calculated on ``reference`` chunks.
 - ``<metric>_alert`` - A flag indicating potentially significant performance change. ``True`` if realized performance
   crosses
   upper or lower threshold.

The results can be plotted for visual inspection.

.. code-block:: python

    >>> for metric in calc.metrics:
    ...     figure = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     figure.show()

.. image:: /_static/tutorial-perf-guide-Accuracy.svg

.. image:: /_static/tutorial-perf-guide-F1.svg

.. image:: /_static/tutorial-perf-guide-Precision.svg

.. image:: /_static/tutorial-perf-guide-ROC_AUC.svg

.. image:: /_static/tutorial-perf-guide-Recall.svg

.. image:: /_static/tutorial-perf-guide-Specificity.svg


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

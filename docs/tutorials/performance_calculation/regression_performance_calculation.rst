.. _regression-performance-calculation:

================================================================
Monitoring Realized Performance for Regression
================================================================

Just The Code
==============

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_car_price_dataset()[2]
    >>> analysis_df = analysis_df.join(analysis_target_df)

    >>> display(reference_df.head(3))

    >>> calc = nml.PerformanceCalculator(
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='regression',
    ...     metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    ...     chunk_size=6000)

    >>> calc.fit(reference_df)

    >>> results = calc.calculate(analysis_df)

    >>> display(results.data.head(3))

    >>> for metric in calc.metrics:
    ...     figure = results.plot(kind='performance', plot_reference=True, metric=metric)
    ...     figure.show()



Walkthrough
=============


For simplicity the guide is based on a synthetic dataset where the monitored model predicts the selling price of a used car.
You can :ref:`learn more about this dataset<dataset-synthetic-regression>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`.
But as it is required to calculate performance, the first thing to do in this case is to join the analysis target values with the analysis data.


.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_car_price_dataset()[2]
    >>> analysis_df = analysis_df.join(analysis_target_df)

    >>> display(reference_df.head(3))


+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|    |   car_age |   km_driven |   price_new |   accident_count |   door_count | fuel     | transmission   |   y_true |   y_pred | timestamp               |
+====+===========+=============+=============+==================+==============+==========+================+==========+==========+=========================+
|  0 |        15 |      144020 |       42810 |                4 |            3 | diesel   | automatic      |      569 |     1246 | 2017-01-24 08:00:00.000 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  1 |        12 |       57078 |       31835 |                3 |            3 | electric | automatic      |     4277 |     4924 | 2017-01-24 08:00:33.600 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  2 |         2 |       76288 |       31851 |                3 |            5 | diesel   | automatic      |     7011 |     5744 | 2017-01-24 08:01:07.200 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+

Next a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using a list of metrics to calculate (or just one metric), the data columns required for these metrics, and an optional :ref:`chunking<chunking>` specification.

The list of metrics specifies which performance metrics of the monitored model will be calculated.
The following metrics are currently supported:

- ``mae``
- ``mape``
- ``mse``
- ``msle``
- ``rmse``
- ``rmsle``

For more information on metrics, check the :mod:`~nannyml.performance_calculation.metrics` module.

.. code-block:: python

    >>> calc = nml.PerformanceCalculator(
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='regression',
    ...     metrics=['mae', 'mape', 'mse', 'msle', 'rmse', 'rmsle'],
    ...     chunk_size=6000)

    >>> calc.fit(reference_df)


The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can then be used to calculate
realized performance metrics on all data which has target values available.

.. code-block:: python

    >>> results = calc.calculate(analysis_df)
    >>> display(results.data.head(3))


+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+---------+-----------------------+-----------------------+----------------------+-------------+----------+------------------------+------------------------+-----------------------+--------------+-------------+-----------------------+-----------------------+----------------------+-------------+-----------+------------------------+------------------------+-----------------------+--------------+---------+------------------------+------------------------+-----------------------+--------------+----------+-------------------------+-------------------------+------------------------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   | period   |   targets_missing_rate |     mae |   mae_lower_threshold |   mae_upper_threshold |   mae_sampling_error | mae_alert   |     mape |   mape_lower_threshold |   mape_upper_threshold |   mape_sampling_error | mape_alert   |         mse |   mse_lower_threshold |   mse_upper_threshold |   mse_sampling_error | mse_alert   |      msle |   msle_lower_threshold |   msle_upper_threshold |   msle_sampling_error | msle_alert   |    rmse |   rmse_lower_threshold |   rmse_upper_threshold |   rmse_sampling_error | rmse_alert   |    rmsle |   rmsle_lower_threshold |   rmsle_upper_threshold |   rmsle_sampling_error | rmsle_alert   |
+====+===============+===============+=============+=====================+============================+==========+========================+=========+=======================+=======================+======================+=============+==========+========================+========================+=======================+==============+=============+=======================+=======================+======================+=============+===========+========================+========================+=======================+==============+=========+========================+========================+=======================+==============+==========+=========================+=========================+========================+===============+
|  0 | [0:5999]      |             0 |        5999 | 2017-02-16 16:00:00 | 2017-02-18 23:59:26.400000 |          |                      0 | 853.4   |               817.855 |               874.805 |              8.21576 | False       | 0.228707 |               0.229456 |               0.237019 |            0.00248466 | True         | 1.14313e+06 |           1.02681e+06 |           1.21572e+06 |                21915 | False       | 0.0704883 |              0.0696521 |              0.0737091 |             0.0011989 | False        | 1069.17 |                1014.28 |                1103.31 |                10.348 | False        | 0.265496 |                0.263948 |                0.271511 |               0.002239 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+---------+-----------------------+-----------------------+----------------------+-------------+----------+------------------------+------------------------+-----------------------+--------------+-------------+-----------------------+-----------------------+----------------------+-------------+-----------+------------------------+------------------------+-----------------------+--------------+---------+------------------------+------------------------+-----------------------+--------------+----------+-------------------------+-------------------------+------------------------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-02-19 00:00:00 | 2017-02-21 07:59:26.400000 |          |                      0 | 853.137 |               817.855 |               874.805 |              8.21576 | False       | 0.230818 |               0.229456 |               0.237019 |            0.00248466 | False        | 1.13987e+06 |           1.02681e+06 |           1.21572e+06 |                21915 | False       | 0.0699896 |              0.0696521 |              0.0737091 |             0.0011989 | False        | 1067.65 |                1014.28 |                1103.31 |                10.348 | False        | 0.264556 |                0.263948 |                0.271511 |               0.002239 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+---------+-----------------------+-----------------------+----------------------+-------------+----------+------------------------+------------------------+-----------------------+--------------+-------------+-----------------------+-----------------------+----------------------+-------------+-----------+------------------------+------------------------+-----------------------+--------------+---------+------------------------+------------------------+-----------------------+--------------+----------+-------------------------+-------------------------+------------------------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-02-21 08:00:00 | 2017-02-23 15:59:26.400000 |          |                      0 | 846.304 |               817.855 |               874.805 |              8.21576 | False       | 0.229042 |               0.229456 |               0.237019 |            0.00248466 | True         | 1.12872e+06 |           1.02681e+06 |           1.21572e+06 |                21915 | False       | 0.0696923 |              0.0696521 |              0.0737091 |             0.0011989 | False        | 1062.41 |                1014.28 |                1103.31 |                10.348 | False        | 0.263993 |                0.263948 |                0.271511 |               0.002239 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+---------+-----------------------+-----------------------+----------------------+-------------+----------+------------------------+------------------------+-----------------------+--------------+-------------+-----------------------+-----------------------+----------------------+-------------+-----------+------------------------+------------------------+-----------------------+--------------+---------+------------------------+------------------------+-----------------------+--------------+----------+-------------------------+-------------------------+------------------------+---------------+

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


.. image:: /_static/tutorial-perf-guide-regression-mse.svg

.. image:: /_static/tutorial-perf-guide-regression-msle.svg


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

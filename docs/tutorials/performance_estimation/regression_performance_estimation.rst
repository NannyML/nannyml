.. _regression-performance-estimation:

========================================================================================
Estimating Performance for Regression
========================================================================================

This tutorial explains how to use NannyML to estimate the performance of regression
models in the absence of target data. To find out how DEE estimates performance,
read the :ref:`explanation of how Direct Error Estimation works<how-it-works-dee>`.

.. TODO: Add reference link.


.. _performance-estimation-regression-just-the-code:

Just The Code
----------------

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
    >>> display(reference_df.head(3))

    >>> estimator = nml.DEE(
    ...     feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count', 'door_count', 'fuel', 'transmission'],
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['rmse', 'rmsle'],
    ...     chunk_size=6000,
    >>> )
    >>> estimator.fit(reference_df)

    >>> results = estimator.estimate(analysis_df)
    >>> display(results.data.head(3))

    >>> for metric in estimator.metrics:
    ...     fig1 = results.plot(kind='performance', metric=metric, plot_reference=True)
    ...     fig1.show()


Walkthrough
--------------

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model predicts
whether the market price of a used car. You can :ref:`read more about this synthetic dataset<dataset-synthetic-regression>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset.
Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

We start by loading the dataset we 'll be using:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference, analysis, analysis_target = nml.load_synthetic_car_price_dataset()
    >>> display(reference.head(3))

+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|    |   car_age |   km_driven |   price_new |   accident_count |   door_count | fuel     | transmission   |   y_true |   y_pred | timestamp               |
+====+===========+=============+=============+==================+==============+==========+================+==========+==========+=========================+
|  0 |        15 |      144020 |       42810 |                4 |            3 | diesel   | automatic      |      569 |     1246 | 2017-01-24 08:00:00.000 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  1 |        12 |       57078 |       31835 |                3 |            3 | electric | automatic      |     4277 |     4924 | 2017-01-24 08:00:33.600 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  2 |         2 |       76288 |       31851 |                3 |            5 | diesel   | automatic      |     7011 |     5744 | 2017-01-24 08:01:07.200 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+

Next we create the Direct Error Estimation
(:class:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE`)
estimator with a list of metrics, and an optional
:ref:`chunking<chunking>` specification.

The list of metrics specifies which performance metrics of the monitored model will be estimated.
The following metrics are currently supported:

- ``mae`` - mean absolute error
- ``mape`` - mean absolute percentage error
- ``mse`` - mean squared error
- ``rmse`` - root mean squared error
- ``msle`` - mean squared logarithmic error
- ``rmsle`` - root mean squared logarithmic error


For more information about :term:`chunking<Data Chunk>` you can check the :ref:`setting up page<chunking>` and :ref:`advanced guide<chunk-data>`.

.. code-block:: python

    >>> estimator = nml.DEE(
    ...     feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count', 'door_count', 'fuel', 'transmission'],
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['rmse', 'rmsle'],
    ...     chunk_size=6000,
    >>> )
    >>> estimator.fit(reference)

    >>> results = estimator.estimate(analysis)
    >>> display(results.data.head(3))


+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   |   realized_rmse |   estimated_rmse |   upper_confidence_rmse |   lower_confidence_rmse |   sampling_error_rmse |   upper_threshold_rmse |   lower_threshold_rmse | alert_rmse   |   realized_rmsle |   estimated_rmsle |   upper_confidence_rmsle |   lower_confidence_rmsle |   sampling_error_rmsle |   upper_threshold_rmsle |   lower_threshold_rmsle | alert_rmsle   |
+====+===============+===============+=============+=====================+============================+=================+==================+=========================+=========================+=======================+========================+========================+==============+==================+===================+==========================+==========================+========================+=========================+=========================+===============+
|  0 | [0:5999]      |             0 |        5999 | 2017-02-16 16:00:00 | 2017-02-18 23:59:26.400000 |             nan |          1067.42 |                 1098.46 |                 1036.37 |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.265777 |                 0.272494 |                 0.25906  |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-02-19 00:00:00 | 2017-02-21 07:59:26.400000 |             nan |          1062.74 |                 1093.79 |                 1031.7  |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.266766 |                 0.273483 |                 0.260049 |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-02-21 08:00:00 | 2017-02-23 15:59:26.400000 |             nan |          1054.53 |                 1085.58 |                 1023.49 |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.267806 |                 0.274523 |                 0.261089 |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+


.. _performance-estimation-regression-thresholds:

Apart from chunk-related data, the results data have the following columns for each metric
that was estimated:

 - ``realized_<metric>`` - when ``target`` values are available for a chunk, the realized performance metric will also
   be calculated and included within the results.
 - ``estimated_<metric>`` - the estimate of a metric for a specific chunk,
 - ``upper_confidence_<metric>`` and ``lower_confidence_<metric>`` - these correspond to the sampling error and are equal to +/-
   3 times the calculated standard error.
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

* The low-saturated purple area around the estimated performance indicates the :ref:`sampling
  error<estimation_of_standard_error>`.

* The red horizontal dashed lines show upper and lower thresholds for alerting purposes.

* If the estimated performance crosses the upper or lower threshold an alert is raised which is indicated with a red,
  low-saturated background in the whole width of the relevant chunk. This is additionally
  indicated by a red, diamond-shaped point marker in the middle of the chunk.

Description of tabular results above explains how the confidence bands and thresholds are calculated.
Additional information is shown in the hover (these are interactive plots, though only static views are included here).

.. code-block:: python

    >>> for metric in estimator.metrics:
    ...     fig1 = results.plot(kind='performance', metric=metric, plot_reference=True)
    ...     fig1.show()

.. image:: ../../_static/tutorial-perf-est-regression-RMSE.svg

.. image:: ../../_static/tutorial-perf-est-regression-RMSLE.svg


Insights
--------

After reviewing the performance estimation results, we should be able to see any indications of performance change that
NannyML has detected based upon the model's inputs and outputs alone.


What's next
-----------

The :ref:`Data Drift<data-drift>` functionality can help us to understand whether data drift is causing the performance problem.
When the target values become available they can be :ref:`compared with the estimated
results<compare_estimated_and_realized_performance>`.

You can learn more about Direct Error Estimation and its limitations in the
:ref:`How it Works page<performance-estimation-deep-dive>`.

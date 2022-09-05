.. _regression-performance-estimation:

=====================================
Estimating Performance for Regression
=====================================

This tutorial explains how to use NannyML to estimate the performance of regression
models in the absence of target data. To find out how DEE estimates performance,
read the :ref:`explanation of how Direct Error Estimation works<how-it-works-dle>`.

.. _performance-estimation-regression-just-the-code:

Just The Code
-------------

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
    ...     tune_hyperparameters=False
    >>> )
    >>> estimator.fit(reference_df)

    >>> results = estimator.estimate(analysis_df)
    >>> display(results.data)
    >>> display(results.estimator.previous_reference_results)

    >>> for metric in estimator.metrics:
    ...     fig1 = results.plot(kind='performance', metric=metric, plot_reference=True)
    ...     fig1.show()


Walkthrough
-----------

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model predicts
whether the market price of a used car. You can read more about this synthetic dataset :ref:`here<dataset-synthetic-regression>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset.
Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

We start by loading the dataset we 'll be using:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
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

The next step is to instantiate the Direct Error Estimation
(:class:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE`)
estimator. For the instantiation we need to provide:

* The list of column names for the features our model uses.
* The column name for the model output.
* The column name for the model targets.
* The list of regression performance metrics we are interested in estimating. Currently the supported metrics are:

  * ``mae`` - mean absolute error
  * ``mape`` - mean absolute percentage error
  * ``mse`` - mean squared error
  * ``rmse`` - root mean squared error
  * ``msle`` - mean squared logarithmic error
  * ``rmsle`` - root mean squared logarithmic error

* Optionally we can provide a :ref:`chunking<chunking>` specification, otherwise the NannyML default will be used.
  For more information about :term:`chunking<Data Chunk>` you can check the :ref:`setting up page<chunking>` and :ref:`advanced guide<chunk-data>`.
* Optionally we can provide selected hyperparamters for the model that will make the error estimation. If not, the
  `LGBMRegressor defaults`_ will be used.
* Optionally we can tell the estimator to use FLAML to perform hyperparamter tuning. By default no hyperparamter tuning is performed.
* Optionally we can provide `configuration options`_ to perform hyperparamter tuning instead of using the ones set by NannyML.

More information can be found on the API documentation for the :class:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE` estimator.
During this tutorial the NannyML default settings are used regarding hyperparameter tuning.


.. code-block:: python

    >>> estimator = nml.DEE(
    ...     feature_column_names=['car_age', 'km_driven', 'price_new', 'accident_count', 'door_count', 'fuel', 'transmission'],
    ...     y_pred='y_pred',
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['rmse', 'rmsle'],
    ...     chunk_size=6000,
    ...     tune_hyperparameters=False
    >>> )

The new :class:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE` is fitted using the
:meth:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE.fit` method on the ``reference`` data.

The fitted :class:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE` can then be used to calculate
estimated performance metrics on all data which has target values available with the
:meth:`~nannyml.performance_estimation.direct_error_estimation.dee.DEE.estimate` method.
NannyML can output a dataframe that contains all the results of the analysis data.


.. code-block:: python

    >>> estimator.fit(reference_df)
    >>> results = estimator.estimate(analysis_df)
    >>> display(results.data)

+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   |   realized_rmse |   estimated_rmse |   upper_confidence_rmse |   lower_confidence_rmse |   sampling_error_rmse |   upper_threshold_rmse |   lower_threshold_rmse | alert_rmse   |   realized_rmsle |   estimated_rmsle |   upper_confidence_rmsle |   lower_confidence_rmsle |   sampling_error_rmsle |   upper_threshold_rmsle |   lower_threshold_rmsle | alert_rmsle   |
+====+===============+===============+=============+=====================+============================+=================+==================+=========================+=========================+=======================+========================+========================+==============+==================+===================+==========================+==========================+========================+=========================+=========================+===============+
|  0 | [0:5999]      |             0 |        5999 | 2017-02-16 16:00:00 | 2017-02-18 23:59:26.400000 |             nan |         1067.42  |                1098.46  |                1036.37  |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.265777 |                 0.272494 |                 0.25906  |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-02-19 00:00:00 | 2017-02-21 07:59:26.400000 |             nan |         1062.74  |                1093.79  |                1031.7   |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.266766 |                 0.273483 |                 0.260049 |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-02-21 08:00:00 | 2017-02-23 15:59:26.400000 |             nan |         1054.53  |                1085.58  |                1023.49  |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.267806 |                 0.274523 |                 0.261089 |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  3 | [18000:23999] |         18000 |       23999 | 2017-02-23 16:00:00 | 2017-02-25 23:59:26.400000 |             nan |         1062.54  |                1093.58  |                1031.49  |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.265634 |                 0.272351 |                 0.258917 |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  4 | [24000:29999] |         24000 |       29999 | 2017-02-26 00:00:00 | 2017-02-28 07:59:26.400000 |             nan |         1054.79  |                1085.83  |                1023.74  |                10.348 |                1103.31 |                1014.28 | False        |              nan |          0.268348 |                 0.275065 |                 0.261631 |               0.002239 |                0.271511 |                0.263948 | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  5 | [30000:35999] |         30000 |       35999 | 2017-02-28 08:00:00 | 2017-03-02 15:59:26.400000 |             nan |          930.497 |                 961.541 |                 899.453 |                10.348 |                1103.31 |                1014.28 | True         |              nan |          0.305148 |                 0.311865 |                 0.298431 |               0.002239 |                0.271511 |                0.263948 | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  6 | [36000:41999] |         36000 |       41999 | 2017-03-02 16:00:00 | 2017-03-04 23:59:26.400000 |             nan |          930.34  |                 961.384 |                 899.296 |                10.348 |                1103.31 |                1014.28 | True         |              nan |          0.306772 |                 0.313489 |                 0.300055 |               0.002239 |                0.271511 |                0.263948 | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  7 | [42000:47999] |         42000 |       47999 | 2017-03-05 00:00:00 | 2017-03-07 07:59:26.400000 |             nan |          928.593 |                 959.637 |                 897.549 |                10.348 |                1103.31 |                1014.28 | True         |              nan |          0.306629 |                 0.313346 |                 0.299912 |               0.002239 |                0.271511 |                0.263948 | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  8 | [48000:53999] |         48000 |       53999 | 2017-03-07 08:00:00 | 2017-03-09 15:59:26.400000 |             nan |          932.237 |                 963.281 |                 901.193 |                10.348 |                1103.31 |                1014.28 | True         |              nan |          0.30611  |                 0.312827 |                 0.299393 |               0.002239 |                0.271511 |                0.263948 | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+
|  9 | [54000:59999] |         54000 |       59999 | 2017-03-09 16:00:00 | 2017-03-11 23:59:26.400000 |             nan |          921.73  |                 952.774 |                 890.686 |                10.348 |                1103.31 |                1014.28 | True         |              nan |          0.308825 |                 0.315542 |                 0.302108 |               0.002239 |                0.271511 |                0.263948 | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+


There results from the reference data are also available.

.. code-block:: python

    >>> display(results.estimator.previous_reference_results)

+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   |   realized_rmse |   estimated_rmse |   upper_confidence_rmse |   lower_confidence_rmse |   sampling_error_rmse |   upper_threshold_rmse |   lower_threshold_rmse | alert_rmse   |   realized_rmsle |   estimated_rmsle |   upper_confidence_rmsle |   lower_confidence_rmsle |   sampling_error_rmsle |   upper_threshold_rmsle |   lower_threshold_rmsle | alert_rmsle   | period    | estimated   |
+====+===============+===============+=============+=====================+============================+=================+==================+=========================+=========================+=======================+========================+========================+==============+==================+===================+==========================+==========================+========================+=========================+=========================+===============+===========+=============+
|  0 | [0:5999]      |             0 |        5999 | 2017-01-24 08:00:00 | 2017-01-26 15:59:26.400000 |         1086.31 |          1073.4  |                 1104.44 |                 1042.35 |                10.348 |                1103.31 |                1014.28 | False        |         0.267475 |          0.266626 |                 0.273343 |                 0.259909 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-01-26 16:00:00 | 2017-01-28 23:59:26.400000 |         1060.22 |          1056.19 |                 1087.24 |                 1025.15 |                10.348 |                1103.31 |                1014.28 | False        |         0.268573 |          0.268918 |                 0.275635 |                 0.262201 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-01-29 00:00:00 | 2017-01-31 07:59:26.400000 |         1038.42 |          1054.96 |                 1086.01 |                 1023.92 |                10.348 |                1103.31 |                1014.28 | False        |         0.266343 |          0.268806 |                 0.275523 |                 0.262089 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  3 | [18000:23999] |         18000 |       23999 | 2017-01-31 08:00:00 | 2017-02-02 15:59:26.400000 |         1038.4  |          1055.05 |                 1086.09 |                 1024    |                10.348 |                1103.31 |                1014.28 | False        |         0.266362 |          0.267155 |                 0.273872 |                 0.260438 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  4 | [24000:29999] |         24000 |       29999 | 2017-02-02 16:00:00 | 2017-02-04 23:59:26.400000 |         1072.02 |          1066.57 |                 1097.61 |                 1035.52 |                10.348 |                1103.31 |                1014.28 | False        |         0.269812 |          0.26687  |                 0.273587 |                 0.260153 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  5 | [30000:35999] |         30000 |       35999 | 2017-02-05 00:00:00 | 2017-02-07 07:59:26.400000 |         1074.97 |          1064.57 |                 1095.61 |                 1033.52 |                10.348 |                1103.31 |                1014.28 | False        |         0.266937 |          0.266295 |                 0.273012 |                 0.259578 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  6 | [36000:41999] |         36000 |       41999 | 2017-02-07 08:00:00 | 2017-02-09 15:59:26.400000 |         1058.48 |          1058.17 |                 1089.21 |                 1027.12 |                10.348 |                1103.31 |                1014.28 | False        |         0.267517 |          0.267456 |                 0.274173 |                 0.260739 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  7 | [42000:47999] |         42000 |       47999 | 2017-02-09 16:00:00 | 2017-02-11 23:59:26.400000 |         1050.7  |          1054.74 |                 1085.78 |                 1023.69 |                10.348 |                1103.31 |                1014.28 | False        |         0.270036 |          0.268755 |                 0.275472 |                 0.262038 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  8 | [48000:53999] |         48000 |       53999 | 2017-02-12 00:00:00 | 2017-02-14 07:59:26.400000 |         1048.4  |          1051.64 |                 1082.68 |                 1020.59 |                10.348 |                1103.31 |                1014.28 | False        |         0.266767 |          0.267832 |                 0.274549 |                 0.261115 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+
|  9 | [54000:59999] |         54000 |       59999 | 2017-02-14 08:00:00 | 2017-02-16 15:59:26.400000 |         1060.04 |          1053.51 |                 1084.55 |                 1022.47 |                10.348 |                1103.31 |                1014.28 | False        |         0.267471 |          0.268593 |                 0.27531  |                 0.261876 |               0.002239 |                0.271511 |                0.263948 | False         | reference | False       |
+----+---------------+---------------+-------------+---------------------+----------------------------+-----------------+------------------+-------------------------+-------------------------+-----------------------+------------------------+------------------------+--------------+------------------+-------------------+--------------------------+--------------------------+------------------------+-------------------------+-------------------------+---------------+-----------+-------------+


.. _performance-estimation-regression-thresholds:

Apart from chunk-related data, the results data have the following columns for each metric
that was estimated:

 - ``realized_<metric>`` - when ``target`` values are available for a chunk, the realized performance metric will also
   be calculated and included within the results.
 - ``estimated_<metric>`` - the estimate of a metric for a specific chunk,
 - ``upper_confidence_<metric>`` and ``lower_confidence_<metric>`` - these correspond to the sampling error and are equal to +/-
   3 times the calculated :term:`standard error<Standard Error>`.
 - ``<metric>_lower_threshold>`` and ``<metric>_upper_threshold>`` - Lower and upper thresholds for performance metric.
   Crossing them will raise an alert that there is a significant metric change. The thresholds are calculated based
   on the realized performance of chunks in the ``reference`` period.
   The thresholds are 3 standard deviations away from the mean performance calculated on ``reference`` chunks.
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

Additional information is shown in the hover (these are interactive plots, though only static views are included here).
The plots can be created with the following code:

.. code-block:: python

    >>> for metric in estimator.metrics:
    ...     fig1 = results.plot(kind='performance', metric=metric, plot_reference=True)
    ...     fig1.show()

.. image:: ../../_static/tutorial-perf-est-regression-RMSE.svg

.. image:: ../../_static/tutorial-perf-est-regression-RMSLE.svg


Insights
--------


From looking at the RMSE and RMSLE performance results we can observe an interesting effect. We know that RMSE penalizes
mispredictions symmetrically while RMSLE penalizes underprediction more than overprediction. Hence performance estimator tells us
that while our model will become a little bit more accurate according to RMSE, the increase in RMSLE suggests us that our model will
be underpredicting more than it was before!


What's next
-----------

The :ref:`Data Drift<data-drift>` functionality can help us to understand whether data drift is causing the performance problem.
When the target values become available they can be :ref:`compared with the estimated
results<compare_estimated_and_realized_performance>`.

You can learn more about Direct Error Estimation and its limitations in the
:ref:`How it Works page<performance-estimation-deep-dive>`.


.. _LGBMRegressor defaults: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
.. _configuration options: https://microsoft.github.io/FLAML/docs/reference/automl#automl-objects

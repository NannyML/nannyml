.. _drift_detection_for_regression_model_outputs:

=======================================================
Drift Detection for Regression Model Outputs
=======================================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of the model outputs tells us the model's evaluation of the expected
outcome across the model's population.
If the model's population changes, then the outcome will be different.
The difference in actions is very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.


Just The Code
-------------

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
    >>> display(reference_df.head())

    >>> calc = nml.StatisticalOutputDriftCalculator(
    ...     y_pred='y_pred',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='regression'
    >>> )

    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)
    >>> display(results.data)
    >>> display(results.calculator.previous_reference_results)

    >>> prediction_drift_fig = results.plot(kind='prediction_drift', plot_reference=True)
    >>> prediction_drift_fig.show()

    >>> prediction_distribution_fig = results.plot(kind='prediction_distribution', plot_reference=True)
    >>> prediction_distribution_fig.show()


Walkthrough
-----------

NannyML detects data drift for :term:`Model Outputs` using the
:ref:`Univariate Drift Detection methodology<univariate_drift_detection_walkthrough>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset.
Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference
and analysis dataframes. This synthetic data is for a regression model predicting used car prices. You can find more
details about it :ref:`here<dataset-synthetic-regression>`.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
    >>> display(reference_df.head())

+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|    |   car_age |   km_driven |   price_new |   accident_count |   door_count | fuel     | transmission   |   y_true |   y_pred | timestamp               |
+====+===========+=============+=============+==================+==============+==========+================+==========+==========+=========================+
|  0 |        15 |      144020 |       42810 |                4 |            3 | diesel   | automatic      |      569 |     1246 | 2017-01-24 08:00:00.000 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  1 |        12 |       57078 |       31835 |                3 |            3 | electric | automatic      |     4277 |     4924 | 2017-01-24 08:00:33.600 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  2 |         2 |       76288 |       31851 |                3 |            5 | diesel   | automatic      |     7011 |     5744 | 2017-01-24 08:01:07.200 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  3 |         7 |       97593 |       29288 |                2 |            3 | electric | manual         |     5576 |     6781 | 2017-01-24 08:01:40.800 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+
|  4 |        13 |        9985 |       41350 |                1 |            5 | diesel   | automatic      |     6456 |     6822 | 2017-01-24 08:02:14.400 |
+----+-----------+-------------+-------------+------------------+--------------+----------+----------------+----------+----------+-------------------------+

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator`
class implements the functionality needed for drift detection in model outputs. First, the class is instantiated with appropriate parameters.
To check the model outputs for data drift, we need to pass the name of the predictions column, the name of the timestamp column and the
type of the machine learning problem our model is addressing. In our case the problem type is regression.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.fit` method
is called on the reference data, so that the data baseline can be established.
Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.calculate` method
calculates the drift results on the data provided. An example using it can be seen below.

.. code-block:: python

    >>> calc = nml.StatisticalOutputDriftCalculator(
    ...     y_pred='y_pred',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='regression'
    >>> )

    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)

We can then display the results in a table.

.. code-block:: python

    >>> display(results.data)

+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   |   y_pred_dstat |   y_pred_p_value | y_pred_alert   |   y_pred_threshold |
+====+===============+===============+=============+=====================+============================+================+==================+================+====================+
|  0 | [0:5999]      |             0 |        5999 | 2017-02-16 16:00:00 | 2017-02-18 23:59:26.400000 |     0.00918333 |            0.743 | False          |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-02-19 00:00:00 | 2017-02-21 07:59:26.400000 |     0.01635    |            0.107 | False          |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-02-21 08:00:00 | 2017-02-23 15:59:26.400000 |     0.0108     |            0.544 | False          |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  3 | [18000:23999] |         18000 |       23999 | 2017-02-23 16:00:00 | 2017-02-25 23:59:26.400000 |     0.0101833  |            0.62  | False          |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  4 | [24000:29999] |         24000 |       29999 | 2017-02-26 00:00:00 | 2017-02-28 07:59:26.400000 |     0.01065    |            0.562 | False          |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  5 | [30000:35999] |         30000 |       35999 | 2017-02-28 08:00:00 | 2017-03-02 15:59:26.400000 |     0.202883   |            0     | True           |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  6 | [36000:41999] |         36000 |       41999 | 2017-03-02 16:00:00 | 2017-03-04 23:59:26.400000 |     0.20735    |            0     | True           |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  7 | [42000:47999] |         42000 |       47999 | 2017-03-05 00:00:00 | 2017-03-07 07:59:26.400000 |     0.204683   |            0     | True           |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  8 | [48000:53999] |         48000 |       53999 | 2017-03-07 08:00:00 | 2017-03-09 15:59:26.400000 |     0.207133   |            0     | True           |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+
|  9 | [54000:59999] |         54000 |       59999 | 2017-03-09 16:00:00 | 2017-03-11 23:59:26.400000 |     0.215883   |            0     | True           |               0.05 |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+

The drift results from the reference data are accessible though the ``previous_reference_results`` property of the drift calculator who is also accessible from the results object.

.. code-block:: python

    >>> display(results.calculator.previous_reference_results)

+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|    | key           |   start_index |   end_index | start_date          | end_date                   |   y_pred_dstat |   y_pred_p_value | y_pred_alert   |   y_pred_threshold | period    |
+====+===============+===============+=============+=====================+============================+================+==================+================+====================+===========+
|  0 | [0:5999]      |             0 |        5999 | 2017-01-24 08:00:00 | 2017-01-26 15:59:26.400000 |     0.0167667  |            0.092 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-01-26 16:00:00 | 2017-01-28 23:59:26.400000 |     0.0118833  |            0.421 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-01-29 00:00:00 | 2017-01-31 07:59:26.400000 |     0.0106667  |            0.56  | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  3 | [18000:23999] |         18000 |       23999 | 2017-01-31 08:00:00 | 2017-02-02 15:59:26.400000 |     0.00961667 |            0.69  | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  4 | [24000:29999] |         24000 |       29999 | 2017-02-02 16:00:00 | 2017-02-04 23:59:26.400000 |     0.00998333 |            0.645 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  5 | [30000:35999] |         30000 |       35999 | 2017-02-05 00:00:00 | 2017-02-07 07:59:26.400000 |     0.0086     |            0.811 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  6 | [36000:41999] |         36000 |       41999 | 2017-02-07 08:00:00 | 2017-02-09 15:59:26.400000 |     0.01265    |            0.344 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  7 | [42000:47999] |         42000 |       47999 | 2017-02-09 16:00:00 | 2017-02-11 23:59:26.400000 |     0.0146833  |            0.188 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  8 | [48000:53999] |         48000 |       53999 | 2017-02-12 00:00:00 | 2017-02-14 07:59:26.400000 |     0.0074     |            0.924 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+
|  9 | [54000:59999] |         54000 |       59999 | 2017-02-14 08:00:00 | 2017-02-16 15:59:26.400000 |     0.0145333  |            0.198 | False          |               0.05 | reference |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------------+------------------+----------------+--------------------+-----------+



NannyML can show the statistical properties of the drift in model outputs as a plot.

.. code-block:: python

    >>> predictions_drift_fig = results.plot(kind='prediction_drift', plot_reference=True)
    >>> predictions_drift_fig.show()

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/regression/drift_guide_prediction_drift.svg


NannyML can also visualise how the distributions of the model predictions evolved over time.

.. code-block:: python

    >>> predictions_distribution_fig = results.plot(kind='prediction_distribution', plot_reference=True)
    >>> predictions_distribution_fig.show()

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/regression/drift_guide_prediction_distribution.svg


Insights
--------

We can see that in the middle of the analysis period the model output distribution has changed significantly and
there is a good possiblity that the performance of our model has been impacted.

What Next
---------

If required, the :ref:`performance estimation<regression-performance-estimation>` functionality of NannyML can help
provide estimates of the impact of the observed changes to Model Outputs without having to wait for Model Targets to
become available.

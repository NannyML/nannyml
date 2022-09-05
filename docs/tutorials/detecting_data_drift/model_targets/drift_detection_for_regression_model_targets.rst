.. _drift_detection_for_regression_model_targets:

=======================================================
Drift Detection for Regression Model Targets
=======================================================

Why Perform Drift Detection for Model Targets
---------------------------------------------

The performance of a machine learning model can be affected if the distribution of targets changes.
The target distribution can change both because of data drift but also because of label shift.

A change in the target distribution may mean that business assumptions on which the model is
used may need to be revisited.

NannyML uses :class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
in order to monitor drift in the :term:`Target` distribution. It can calculate the KS
statistic (from the :term:`Kolmogorov-Smirnov test`) for aggregated drift results
but also show the target distribution results per chunk with joyploys.

.. note::
    The Target Drift detection process can handle missing target values across all :term:`data periods<Data Period>`.


Just The Code
-------------

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display

    >>> reference_df = nml.load_synthetic_car_price_dataset()[0]
    >>> analysis_df = nml.load_synthetic_car_price_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_car_price_dataset()[2]
    >>> analysis_df = analysis_df.join(analysis_target_df)

    >>> display(reference_df.head(3))

    >>> calc = nml.TargetDistributionCalculator(
    ...     y_true='y_true',
    ...     timestamp_column_name='timestamp',
    ...     problem_type='regression',
    >>> )

    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)
    >>> display(results.data)
    >>> display(results.calculator.previous_reference_results)
    

    >>> target_drift_fig = results.plot(kind='target_drift', plot_reference=True)
    >>> target_drift_fig.show()

    >>> target_distribution_fig = results.plot(kind='target_distribution', plot_reference=True)
    >>> target_distribution_fig.show()


Walkthrough
-----------

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some :ref:`synthetic car pricing data<dataset-synthetic-regression>` provided by the NannyML package, and setting it up as our reference and analysis dataframes.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation<performance-estimation>`. But it is required to detect drift for the targets, so the first thing we need to in this case is set up the right data in the right dataframes.
The analysis target values are expected to be ordered correctly, just like in sklearn.

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

Now that the data is in place we'll create a new
:class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
instantiating it with the appropriate parameters. We need the name for the target, ``y_true``, and the timestamp columns.
We also need to specify the machine learning problem we are working on.

.. code-block:: python

        >>> calc = nml.TargetDistributionCalculator(
        ...     y_true='y_true',
        ...     timestamp_column_name='timestamp',
        ...     problem_type='regression',
        >>> )


Afterwards, the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.fit`
method gets called on the reference :term:`period<Data Period>`, which represent an accepted target distribution
which we will compare against the analysis :term:`period<Data Period>`.

Then the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.calculate` method is
called to calculate the target drift results on the data provided. We use the previously assembled data as an argument.

We can display the results of this calculation in a dataframe.

.. code-block:: python

    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)
    >>> display(results.data)

+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   | period   |   targets_missing_rate |   metric_target_drift |   statistical_target_drift |      p_value |   thresholds | alert   | significant   |
+====+===============+===============+=============+=====================+============================+==========+========================+=======================+============================+==============+==============+=========+===============+
|  0 | [0:5999]      |             0 |        5999 | 2017-02-16 16:00:00 | 2017-02-18 23:59:26.400000 |          |                      0 |               4862.94 |                 0.01425    | 0.215879     |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-02-19 00:00:00 | 2017-02-21 07:59:26.400000 |          |                      0 |               4790.58 |                 0.0165667  | 0.0990255    |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-02-21 08:00:00 | 2017-02-23 15:59:26.400000 |          |                      0 |               4793.35 |                 0.0100667  | 0.634331     |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  3 | [18000:23999] |         18000 |       23999 | 2017-02-23 16:00:00 | 2017-02-25 23:59:26.400000 |          |                      0 |               4838.26 |                 0.0119167  | 0.4175       |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  4 | [24000:29999] |         24000 |       29999 | 2017-02-26 00:00:00 | 2017-02-28 07:59:26.400000 |          |                      0 |               4799.13 |                 0.00866667 | 0.803771     |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  5 | [30000:35999] |         30000 |       35999 | 2017-02-28 08:00:00 | 2017-03-02 15:59:26.400000 |          |                      0 |               4852.64 |                 0.171683   | 4.6704e-141  |         0.05 | True    | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  6 | [36000:41999] |         36000 |       41999 | 2017-03-02 16:00:00 | 2017-03-04 23:59:26.400000 |          |                      0 |               4875.46 |                 0.180117   | 2.5805e-155  |         0.05 | True    | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  7 | [42000:47999] |         42000 |       47999 | 2017-03-05 00:00:00 | 2017-03-07 07:59:26.400000 |          |                      0 |               4867.59 |                 0.179067   | 1.67957e-153 |         0.05 | True    | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  8 | [48000:53999] |         48000 |       53999 | 2017-03-07 08:00:00 | 2017-03-09 15:59:26.400000 |          |                      0 |               4885.11 |                 0.183233   | 9.21428e-161 |         0.05 | True    | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+
|  9 | [54000:59999] |         54000 |       59999 | 2017-03-09 16:00:00 | 2017-03-11 23:59:26.400000 |          |                      0 |               4787.09 |                 0.187383   | 3.64399e-168 |         0.05 | True    | True          |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+--------------+--------------+---------+---------------+

We can also display the results from the reference dataframe.

.. code-block:: python

    >>> display(results.calculator.previous_reference_results)

+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   | period   |   targets_missing_rate |   metric_target_drift |   statistical_target_drift |   p_value |   thresholds | alert   | significant   |
+====+===============+===============+=============+=====================+============================+==========+========================+=======================+============================+===========+==============+=========+===============+
|  0 | [0:5999]      |             0 |        5999 | 2017-01-24 08:00:00 | 2017-01-26 15:59:26.400000 |          |                      0 |               4894.48 |                 0.0161     |  0.116973 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-01-26 16:00:00 | 2017-01-28 23:59:26.400000 |          |                      0 |               4767.13 |                 0.0120167  |  0.407002 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-01-29 00:00:00 | 2017-01-31 07:59:26.400000 |          |                      0 |               4744    |                 0.0119667  |  0.412231 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  3 | [18000:23999] |         18000 |       23999 | 2017-01-31 08:00:00 | 2017-02-02 15:59:26.400000 |          |                      0 |               4791.32 |                 0.0122667  |  0.381454 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  4 | [24000:29999] |         24000 |       29999 | 2017-02-02 16:00:00 | 2017-02-04 23:59:26.400000 |          |                      0 |               4827.68 |                 0.0100833  |  0.632257 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  5 | [30000:35999] |         30000 |       35999 | 2017-02-05 00:00:00 | 2017-02-07 07:59:26.400000 |          |                      0 |               4811.59 |                 0.0095     |  0.704746 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  6 | [36000:41999] |         36000 |       41999 | 2017-02-07 08:00:00 | 2017-02-09 15:59:26.400000 |          |                      0 |               4773.53 |                 0.00613333 |  0.985698 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  7 | [42000:47999] |         42000 |       47999 | 2017-02-09 16:00:00 | 2017-02-11 23:59:26.400000 |          |                      0 |               4766    |                 0.01155    |  0.457293 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  8 | [48000:53999] |         48000 |       53999 | 2017-02-12 00:00:00 | 2017-02-14 07:59:26.400000 |          |                      0 |               4750.15 |                 0.00665    |  0.968017 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  9 | [54000:59999] |         54000 |       59999 | 2017-02-14 08:00:00 | 2017-02-16 15:59:26.400000 |          |                      0 |               4744.8  |                 0.00786667 |  0.885651 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+

The results can be also easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.
We first plot the KS Statistic drift results for each chunk.

.. code-block:: python

    >>> target_drift_fig = results.plot(kind='target_drift', plot_reference=True)
    >>> target_drift_fig.show()
 
Note that a dashed line, instead of a solid line, will be used for chunks that have missing target values.

.. image:: /_static/tutorials/detecting_data_drift/model_targets/regression/target-drift.svg

And then we create the joyplot to visualize the target distribution values for each chunk.

.. code-block:: python

    >>> target_distribution_fig = results.plot(kind='target_distribution', plot_reference=True)
    >>> target_distribution_fig.show()

.. image:: /_static/tutorials/detecting_data_drift/model_targets/regression/target-distribution.svg


Insights
--------

Looking at the results we can see that there has been some target drift towards lower car prices.
We should also check to see if the performance of our model has been affected through
:ref:`realized performance monitoring<regression-performance-calculation>`.
Lastly we would need to check with the business stakeholders to see if the changes observed can affect the company's 
sales and marketing policies.


What Next
---------

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes. Moreover the :ref:`Univariate Drift Detection<univariate_drift_detection>`
as well as the :ref:`Multivariate Drift Detection<multivariate_drift_detection>` can add further context if needed.

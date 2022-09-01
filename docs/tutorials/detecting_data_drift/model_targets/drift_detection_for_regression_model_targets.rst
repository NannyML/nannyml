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
in order to monitor drift in the :term:`Target` distribution. It can calculate the mean occurrence of positive
events for binary classification problems.

It can also calculate the chi squared statistic (from the :term:`Chi Squared test<Chi Squared test>`)
of the available target values for each chunk, for both binary and multiclass classification problems, or the KS
statistic (from the :term:`Kolmogorov-Smirnov test`) for regression problems.

.. note::
    The Target Drift detection process can handle missing target values across all :term:`data periods<Data Period>`.


Just The Code
------------------------------------

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
    >>> display(results.data.head(3))

    >>> target_drift_fig = results.plot(kind='target_drift', plot_reference=True)
    >>> target_drift_fig.show()

    >>> target_distribution_fig = results.plot(kind='target_distribution', plot_reference=True)
    >>> target_distribution_fig.show()


Walkthrough
------------------------------------------------

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some :ref:`synthetic data<dataset-synthetic-regression>` provided by the NannyML package, and setting it up as our reference and analysis dataframes.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to detect drift for the targets, so the first thing we need to in this case is set up the right data in the right dataframes.  The analysis target values are joined on the analysis frame by the ``identifier`` column.

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
instantiating it with the appropriate parameters. We only need the target (``y_true``) and timestamp.

.. code-block:: python

        >>> calc = nml.TargetDistributionCalculator(y_true='y_true', timestamp_column_name='timestamp')


Afterwards, the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.fit`
method gets called on the reference :term:`period<Data Period>`, which represent an accepted target distribution
which we will compare against the analysis :term:`period<Data Period>`.

Then the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.calculate` method is
called to calculate the target drift results on the data provided. We use the previously assembled data as an argument.

We can display the results of this calculation in a dataframe.

.. code-block:: python

    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)
    >>> display(results.data.head(3))

+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date                   | period   |   targets_missing_rate |   metric_target_drift |   statistical_target_drift |   p_value |   thresholds | alert   | significant   |
+====+===============+===============+=============+=====================+============================+==========+========================+=======================+============================+===========+==============+=========+===============+
|  0 | [0:5999]      |             0 |        5999 | 2017-02-16 16:00:00 | 2017-02-18 23:59:26.400000 |          |                      0 |               4862.94 |                  0.01425   | 0.215879  |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2017-02-19 00:00:00 | 2017-02-21 07:59:26.400000 |          |                      0 |               4790.58 |                  0.0165667 | 0.0990255 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2017-02-21 08:00:00 | 2017-02-23 15:59:26.400000 |          |                      0 |               4793.35 |                  0.0100667 | 0.634331  |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+----------------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+

The results can be also easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.

.. code-block:: python

    >>> target_drift_fig = results.plot(kind='target_drift', plot_reference=True)
    >>> target_drift_fig.show()

Note that a dashed line, instead of a solid line, will be used for chunks that have missing target values.

.. image:: /_static/tutorials/detecting_data_drift/model_targets/regression/target-drift.svg


.. code-block:: python

    >>> target_distribution_fig = results.plot(kind='target_distribution', plot_reference=True)
    >>> target_distribution_fig.show()

.. image:: /_static/tutorials/detecting_data_drift/model_targets/regression/target-distribution.svg


What Next
-----------------------

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes.

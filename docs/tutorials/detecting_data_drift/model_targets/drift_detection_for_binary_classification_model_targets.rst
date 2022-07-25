.. _drift_detection_for_binary_classification_model_targets:

=======================================================
Drift Detection for Binary Classification Model Targets
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
of the available target values for each chunk, for both binary and multiclass classification problems.

.. note::
    The Target Drift detection process can handle missing target values across all :term:`data periods<Data Period>`.


Just The Code
------------------------------------

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> 
    >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_binary_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')
    >>> 
    >>> display(reference_df.head(3))
    >>> 
    >>> calc = nml.TargetDistributionCalculator(
    ...     y_true='work_home_actual',
    ...     timestamp_column_name='timestamp'
    >>> )
    >>> 
    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)
    >>> display(results.data.head(3))
    >>> 
    >>> distribution_fig1 = results.plot(kind='distribution', distribution='metric', plot_reference=True)
    >>> distribution_fig1.show()
    >>> 
    >>> distribution_fig2 = results.plot(kind='distribution', distribution='statistical', plot_reference=True)
    >>> distribution_fig2.show()


Walkthrough
------------------------------------------------

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference and analysis dataframes. This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to detect drift for the targets, so the first thing we need to in this case is set up the right data in the right dataframes.  The analysis target values are joined on the analysis frame by the ``identifier`` column.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> 
    >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_binary_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')
    >>> 
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

Now that the data is in place we'll create a new
:class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
instantiating it with the appropriate parameters. We only need the target (``y_true``) and timestamp.

.. code-block:: python

    >>> calc = nml.TargetDistributionCalculator(
    ...     y_true='work_home_actual',
    ...     timestamp_column_name='timestamp'
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
    >>> display(results.data.head(3))

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   targets_missing_rate |   metric_target_drift |   statistical_target_drift |   p_value |   thresholds | alert   | significant   |
+====+===============+===============+=============+=====================+=====================+=============+========================+=======================+============================+===========+==============+=========+===============+
|  0 | [0:4999]      |             0 |        4999 | 2014-05-09 22:27:20 | 2014-09-09 08:18:27 | reference   |                      0 |                0.4944 |                   0.467363 |  0.494203 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2014-09-09 09:13:35 | 2015-01-09 00:02:51 | reference   |                      0 |                0.493  |                   0.76111  |  0.382981 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  2 | [10000:14999] |         10000 |       14999 | 2015-01-09 00:04:43 | 2015-05-09 15:54:26 | reference   |                      0 |                0.505  |                   0.512656 |  0.473991 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+

The results can be also easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.

.. code-block:: python

    >>> distribution_fig1 = results.plot(kind='distribution', distribution='metric', plot_reference=True)
    >>> distribution_fig1.show()

Note that a dashed line, instead of a solid line, will be used for chunks that have missing target values.

.. image:: /_static/target_distribution_metric.svg


.. code-block:: python

    >>> distribution_fig2 = results.plot(kind='distribution', distribution='statistical', plot_reference=True)
    >>> distribution_fig2.show()

.. image:: /_static/target_distribution_statistical.svg


Insights
-----------------------

Looking at the results we see that we have a false alert on the first chunk of the analysis data. This
can happen when the statistical tests consider a small change in the distribution of a variable to be significant.
But because the change is small it is usually not significant from a model monitoring perspective.



What Next
-----------------------

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes.

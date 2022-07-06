.. _drift_detection_for_multiclass_classification_model_targets:

===========================================================
Drift Detection for Multiclass Classification Model Targets
===========================================================

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
    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_multiclass_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')
    >>> 
    >>> display(reference_df.head(3))
    >>> 
    >>> calc = nml.TargetDistributionCalculator(
    ...     y_true='y_true',
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
    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]
    >>> analysis_target_df = nml.load_synthetic_multiclass_classification_dataset()[2]
    >>> analysis_df = analysis_df.merge(analysis_target_df, on='identifier')
    >>> 
    >>> display(reference_df.head(3))


+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|    | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | period    |   identifier | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred       | y_true        |
+====+===============+========================+==========================+===============+=======================+=================+===============+===========+==============+=====================+=============================+================================+==============================+==============+===============+
|  0 | Partner3      |               1.80823  |                      350 | web           |                   309 |           15000 | True          | reference |        60000 | 2020-05-02 02:01:30 |                        0.97 |                           0.03 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  1 | Partner2      |               4.38257  |                      500 | mobile        |                   418 |           23000 | True          | reference |        60001 | 2020-05-02 02:03:33 |                        0.87 |                           0.13 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  2 | Partner2      |              -0.787575 |                      400 | web           |                   507 |           24000 | False         | reference |        60002 | 2020-05-02 02:04:49 |                        0.47 |                           0.35 |                         0.18 | prepaid_card | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+

Now that the data is in place we'll create a new
:class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
instantiating it with the appropriate parameters. We only need the target (``y_true``) and timestamp.

.. code-block:: python

    >>> calc = nml.TargetDistributionCalculator(
    ...     y_true='y_true',
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

+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | period   |   targets_missing_rate |   metric_target_drift |   statistical_target_drift |   p_value |   thresholds | alert   | significant   |
+====+===============+===============+=============+=====================+=====================+==========+========================+=======================+============================+===========+==============+=========+===============+
|  0 | [0:5999]      |             0 |        5999 | 2020-09-01 03:10:01 | 2020-09-13 16:15:10 |          |                      0 |                   nan |                   0.521545 |  0.770456 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2020-09-13 16:15:32 | 2020-09-25 19:48:42 |          |                      0 |                   nan |                   2.11226  |  0.3478   |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  2 | [12000:17999] |         12000 |       17999 | 2020-09-25 19:50:04 | 2020-10-08 02:53:47 |          |                      0 |                   nan |                   0.940108 |  0.624969 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+

The results can be also easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.

.. code-block:: python

    >>> distribution_fig1 = results.plot(kind='distribution', distribution='metric', plot_reference=True)
    >>> distribution_fig1.show()


.. warning::
    Since our target data contains non-numerical values and over 3 values, we currently don't support plotting using the
    ``distribution='metric'`` parameter. NannyML will print out warnings to inform you about this:

    .. code-block::

        UserWarning: the target column contains 3 unique values. NannyML cannot provide a value for 'metric_target_drift' when there are more than 2 unique values. All 'metric_target_drift' values will be set to np.NAN
        UserWarning: the target column contains non-numerical values. NannyML cannot provide a value for 'metric_target_drift'.All 'metric_target_drift' values will be set to np.NAN



.. code-block:: python

    >>> distribution_fig2 = results.plot(kind='distribution', distribution='statistical', plot_reference=True)
    >>> distribution_fig2.show()

.. image:: /_static/tutorials/detecting_data_drift/model_targets/multiclass/target-distribution-statistical.svg


What Next
-----------------------

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes.

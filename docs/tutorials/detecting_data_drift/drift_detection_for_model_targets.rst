.. _drift_detection_for_model_targets:

=================================
Drift Detection for Model Targets
=================================

Why Perform Drift Detection for Model Targets
---------------------------------------------

The performance of a machine learning model can be affected if the distribution of targets changes.
The target distribution can change both because of data drift but also because of label shift.

Moreover a change in the target distribution may mean that business assumptions on which the model is
used may need to be revisited.

Just The Code
-------------

If you just want the code to experiment yourself within a Jupyter Notebook, here you go:

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor', model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(reference.head(3))

    >>> data = pd.concat([reference, analysis.set_index('identifier').join(analysis_target.set_index('identifier'), on='identifier', rsuffix='_r')], ignore_index=True).reset_index(drop=True)
    >>> display(data.loc[data['partition'] == 'analysis'].head(3))

    >>> target_distribution_calculator = nml.TargetDistributionCalculator(model_metadata=metadata, chunk_size=5000)
    >>> target_distribution_calculator = target_distribution_calculator.fit(reference_data=reference)

    >>> target_distribution = target_distribution_calculator.calculate(data)
    >>> display(target_distribution.data.head(3))

    >>> fig = target_distribution.plot(kind='distribution', distribution='metric')
    >>> fig.show()

    >>> fig = target_distribution.plot(kind='distribution', distribution='statistical')
    >>> fig.show()




Walkthrough on Drift Detection for Model Targets
------------------------------------------------

Let's start by loading some synthetic data provided by the NannyML package.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor', model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(reference.head(3))


+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+


NannyML uses :class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
in order to monitor drift in :term:`Target` distribution. It can calculate the mean occurrence of positive
events for binary classification problems. It can also  calculates the chi-squared statistic,
from the 2 sample Chi Squared test, of the target values for each chunk which is available for both binary
and multiclass classification problems.

In order to calculate target drift, the target values must be available. Let's manually add the target data to the analysis
data first.

.. note::
    The Target Drift detection process can handle missing target values across all :term:`data periods<Data Period>`.

.. code-block:: python

    >>> data = pd.concat([reference, analysis.set_index('identifier').join(analysis_target.set_index('identifier'), on='identifier', rsuffix='_r')], ignore_index=True).reset_index(drop=True)
    >>> display(data.loc[data['partition'] == 'analysis'].head(3))

+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|       |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+=======+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
| 50000 |               0.527691 | 0 - 20K €      |               1.8     |                      8.96072 | False              | Tuesday   |  4.22463 |          nan |                  1 | 2017-08-31 04:20:00 |           0.99 | analysis    |        1 |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
| 50001 |               8.48513  | 20K - 40K €    |               2.22207 |                      8.76879 | False              | Friday    |  4.9631  |          nan |                  1 | 2017-08-31 05:16:16 |           0.98 | analysis    |        1 |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
| 50002 |               2.07388  | 40K - 60K €    |               2.31008 |                      8.64998 | True               | Friday    |  4.58895 |          nan |                  1 | 2017-08-31 05:56:44 |           0.98 | analysis    |        1 |
+-------+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+

Now that the data is in place we'll create a new
:class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
instantiating it with appropriate parameters.
Afterwards, the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.fit`
method gets called on the reference :term:`period<Data Period>` which represent an accepted target distribution
against which we will compare data from the analysis :term:`period<Data Period>`.
Then the
:meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.calculate` method gets
called to calculate the target drift results on the data provided to it. We use the previously
assembled data as an argument.

.. code-block:: python

    >>> target_distribution_calculator = nml.TargetDistributionCalculator(model_metadata=metadata, chunk_size=5000)
    >>> target_distribution_calculator = target_distribution_calculator.fit(reference_data=reference)
    >>> target_distribution = target_distribution_calculator.calculate(data)
    >>> display(target_distribution.data.head(3))

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   targets_missing_rate |   metric_target_drift |   statistical_target_drift |   p_value |   thresholds | alert   | significant   |
+====+===============+===============+=============+=====================+=====================+=============+========================+=======================+============================+===========+==============+=========+===============+
|  0 | [0:4999]      |             0 |        4999 | 2014-05-09 22:27:20 | 2014-09-09 08:18:27 | reference   |                      0 |                0.4944 |                   0.467363 |  0.494203 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2014-09-09 09:13:35 | 2015-01-09 00:02:51 | reference   |                      0 |                0.493  |                   0.76111  |  0.382981 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+
|  2 | [10000:14999] |         10000 |       14999 | 2015-01-09 00:04:43 | 2015-05-09 15:54:26 | reference   |                      0 |                0.505  |                   0.512656 |  0.473991 |         0.05 | False   | False         |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------------------+----------------------------+-----------+--------------+---------+---------------+

The results can be easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.


.. code-block:: python

    >>> fig = target_distribution.plot(kind='distribution', distribution='metric')
    >>> fig.show()

Note that a dashed line, instead of a solid line, will be used for chunks that have missing target values.

.. image:: /_static/target_distribution_metric.svg


.. code-block:: python

    >>> fig = target_distribution.plot(kind='distribution', distribution='statistical')
    >>> fig.show()

.. image:: /_static/target_distribution_statistical.svg

Insights and Follow Ups
-----------------------

Looking at the results we see that we have a false alert on the first chunk of the analysis data. This
can happen when the statistical tests consider significant a small change in the distribution of a variable
in the chunks. But because the change is small it is usually not significant from a model monitoring perspective.

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes.

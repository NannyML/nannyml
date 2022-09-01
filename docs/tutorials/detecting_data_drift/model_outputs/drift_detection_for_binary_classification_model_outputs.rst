.. _drift_detection_for_binary_classification_model_outputs:

=======================================================
Drift Detection for Binary Classification Model Outputs
=======================================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of the model outputs tells us the model's evaluation of how likely
the predicted outcome is to happen across the model's population.
If the model's population changes, then its actions will be different.
The difference in actions is very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.

Just The Code
------------------------------------

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>>
    >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
    >>>
    >>> display(reference_df.head())
    >>>
    >>> calc = nml.StatisticalOutputDriftCalculator(y_pred='y_pred', y_pred_proba='y_pred_proba',
    ...                                             timestamp_column_name='timestamp', problem_type='classification_binary')
    >>>
    >>> calc.fit(reference_df)
    >>>
    >>> results = calc.calculate(analysis_df)
    >>>
    >>> display(results.data)
    >>>
    >>> score_drift_fig = results.plot(kind='score_drift', plot_reference=True)
    >>> score_drift_fig.show()
    >>>
    >>> score_distribution_fig = results.plot(kind='score_distribution', plot_reference=True)
    >>> score_distribution_fig.show()
    >>>
    >>> prediction_drift_fig = results.plot(kind='prediction_drift', plot_reference=True)
    >>> prediction_drift_fig.show()
    >>>
    >>> prediction_distribution_fig = results.plot(kind='prediction_distribution', plot_reference=True)
    >>> prediction_distribution_fig.show()

Walkthrough
------------------------------------------------

NannyML detects data drift for :term:`Model Outputs` using the
:ref:`Univariate Drift Detection methodology<univariate_drift_detection_walkthrough>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference and analysis dataframes. This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>>
    >>> reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_binary_classification_dataset()[1]
    >>>
    >>> display(reference_df.head())

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  3 |               2.53041  | 20K - 20K €    |               2.31872 |                      7.94425 | False              | Tuesday   | 0.453649 |            3 |                  1 | 2014-05-10 01:12:09 |           0.98 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  4 |               2.25364  | 60K+ €         |               2.22127 |                      8.88448 | True               | Thursday  | 5.69526  |            4 |                  1 | 2014-05-10 02:21:34 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator`
class implements the functionality needed for drift detection in model outputs. First, the class is instantiated with appropriate parameters.
To check the model outputs for data drift, we only need to pass in the column header of the outputs as `y_pred` and `y_pred_proba`.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.fit` method
is called on the reference data, so that the data baseline can be established.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.calculate` method
calculates the drift results on the data provided. An example using it can be seen below.

.. code-block:: python

    >>> calc = nml.StatisticalOutputDriftCalculator(y_pred='y_pred', y_pred_proba='y_pred_proba',
    ...                                             timestamp_column_name='timestamp', problem_type='classification_binary')
    >>> calc.fit(reference_df)
    >>> results = calc.calculate(analysis_df)

We can then display the results in a table, or as plots.

.. code-block:: python

    display(results.data)

+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | period   |   y_pred_chi2 |   y_pred_p_value | y_pred_alert   |   y_pred_threshold |   y_pred_proba_dstat |   y_pred_proba_p_value | y_pred_proba_alert   |   y_pred_proba_threshold |
+====+===============+===============+=============+=====================+=====================+==========+===============+==================+================+====================+======================+========================+======================+==========================+
|  0 | [0:4999]      |             0 |        4999 | 2017-08-31 04:20:00 | 2018-01-02 00:45:44 |          |     7.44238   |            0.006 | True           |               0.05 |              0.0253  |                  0.006 | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2018-01-02 01:13:11 | 2018-05-01 13:10:10 |          |     1.80017   |            0.18  | False          |               0.05 |              0.0123  |                  0.494 | False                |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  2 | [10000:14999] |         10000 |       14999 | 2018-05-01 14:25:25 | 2018-09-01 15:40:40 |          |     1.72853   |            0.189 | False          |               0.05 |              0.01642 |                  0.17  | False                |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  3 | [15000:19999] |         15000 |       19999 | 2018-09-01 16:19:07 | 2018-12-31 10:11:21 |          |     1.58961   |            0.207 | False          |               0.05 |              0.01058 |                  0.685 | False                |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  4 | [20000:24999] |         20000 |       24999 | 2018-12-31 10:38:45 | 2019-04-30 11:01:30 |          |     0.0608958 |            0.805 | False          |               0.05 |              0.01408 |                  0.325 | False                |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  5 | [25000:29999] |         25000 |       29999 | 2019-04-30 11:02:00 | 2019-09-01 00:24:27 |          |    12.5121    |            0     | True           |               0.05 |              0.1307  |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  6 | [30000:34999] |         30000 |       34999 | 2019-09-01 00:28:54 | 2019-12-31 09:09:12 |          |    11.3934    |            0.001 | True           |               0.05 |              0.1273  |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  7 | [35000:39999] |         35000 |       39999 | 2019-12-31 10:07:15 | 2020-04-30 11:46:53 |          |     9.81353   |            0.002 | True           |               0.05 |              0.1311  |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  8 | [40000:44999] |         40000 |       44999 | 2020-04-30 12:04:32 | 2020-09-01 02:46:02 |          |     3.78652   |            0.052 | False          |               0.05 |              0.1197  |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+
|  9 | [45000:49999] |         45000 |       49999 | 2020-09-01 02:46:13 | 2021-01-01 04:29:32 |          |    27.99      |            0     | True           |               0.05 |              0.13752 |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+----------------------+------------------------+----------------------+--------------------------+

NannyML can show the statistical properties of the drift in model scores as a plot.

.. code-block:: python

    >>> score_drift_fig = results.plot(kind='score_drift', plot_reference=True)
    >>> score_drift_fig.show()

.. image:: /_static/drift-guide-score-drift.svg

NannyML can also visualise how the distributions of the model scores evolved over time.

.. code-block:: python

    >>> score_distribution_fig = results.plot(kind='score_distribution', plot_reference=True)
    >>> score_distribution_fig.show()


.. image:: /_static/drift-guide-score-distribution.svg

NannyML can show the statistical properties of the drift in the model predictions as a plot.

.. code-block:: python

    >>> predicted_labels_drift_fig = results.plot(kind='prediction_drift', plot_reference=True)
    >>> predicted_labels_drift_fig.show()

.. image:: /_static/drift-guide-prediction-drift.svg

NannyML can also visualise how the distributions of the model predictions evolved over time.

.. code-block:: python

    >>> predicted_labels_distribution_fig = results.plot(kind='prediction_distribution', plot_reference=True)
    >>> predicted_labels_distribution_fig.show()

.. image:: /_static/drift-guide-prediction-distribution.svg


Insights
-----------------------

Looking at the results we can see that we have a false alert on the first chunk of the analysis data. This is similar
to the ``tenure`` variable in the :ref:`univariate drift results<univariate_drift_detection_tenure>`, where there is also
a false alert because the drift measured by the :term:`KS statistic<Kolmogorov-Smirnov test>` is very low. This
can happen when the statistical tests consider a small change in the distribution of a variable
to be significant. But because the change is small it is usually not significant from a model monitoring perspective.


What Next
-----------------------

If required, the :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Outputs.

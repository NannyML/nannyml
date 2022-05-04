.. _drift_detection_for_model_outputs:

=================================
Drift Detection for Model Outputs
=================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of model outputs gives us a summary of the population's
propensity with regards to the question the model is answering. If the model's
population changes, then the propensity of the population will change and vice versa.

Changes to a model's population are very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.


Just The Code
-------------

If you just want the code to experiment yourself withinb a Jupyter Notebook, here you go:

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> reference, analysis, analysis_target = nml.load_synthetic_sample()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    >>> metadata.target_column_name = 'work_home_actual'
    >>> reference.head()
    >>>
    >>> # Let's initialize the object that will perform the Univariate Drift calculations
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=5000).fit(reference_data=reference)
    >>> # let's see drift statistics for all available data
    >>> data = pd.concat([reference, analysis], ignore_index=True)
    >>> univariate_results = univariate_calculator.calculate(data=data)
    >>>
    >>> figure = univariate_results.plot(kind='prediction_drift', metric='statistic')
    >>> figure.show()
    >>>
    >>> figure = univariate_results.plot(kind='prediction_distribution', metric='statistic')
    >>> figure.show()


Walkthrough on Drift Detection for Model Outputs
------------------------------------------------

NannyML detects data drift for :term:`Model Outputs` in the same univariate drift methodology as
for a continuous feature. 

Let's start by loading some synthetic data provided by the NannyML package.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> reference, analysis, analysis_target = nml.load_synthetic_sample()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    >>> metadata.target_column_name = 'work_home_actual'
    >>> reference.head()


+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  3 |               2.53041  | 20K - 20K €    |               2.31872 |                      7.94425 | False              | Tuesday   | 0.453649 |            3 |                  1 | 2014-05-10 01:12:09 |           0.98 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  4 |               2.25364  | 60K+ €         |               2.22127 |                      8.88448 | True               | Thursday  | 5.69526  |            4 |                  1 | 2014-05-10 02:21:34 |           0.99 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator`
class implements the functionality needed for Univariate Drift Detection that we need for drift detection in model outputs as well.

.. code-block:: python

    >>> # Let's initialize the object that will perform the Univariate Drift calculations
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=5000).fit(reference_data=reference)
    >>> # let's see drift statistics for all available data
    >>> data = pd.concat([reference, analysis], ignore_index=True)
    >>> univariate_results = univariate_calculator.calculate(data=data)
    >>> # let's view a small subset of our results:
    >>> # We use the data property of the results class to view the relevant data.
    >>> univariate_results.data.iloc[:5, :9]

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+-------------------------+----------------------------+--------------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   wfh_prev_workday_chi2 |   wfh_prev_workday_p_value | wfh_prev_workday_alert   |
+====+===============+===============+=============+=====================+=====================+=============+=========================+============================+==========================+
|  5 | [25000:29999] |         25000 |       29999 | 2016-01-08 00:00:00 | 2016-05-09 23:59:59 | reference   |               3.61457   |                      0.057 | False                    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+-------------------------+----------------------------+--------------------------+
|  6 | [30000:34999] |         30000 |       34999 | 2016-05-09 00:00:00 | 2016-09-04 23:59:59 | reference   |               0.0757052 |                      0.783 | False                    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+-------------------------+----------------------------+--------------------------+
|  7 | [35000:39999] |         35000 |       39999 | 2016-09-04 00:00:00 | 2017-01-03 23:59:59 | reference   |               0.414606  |                      0.52  | False                    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+-------------------------+----------------------------+--------------------------+
|  8 | [40000:44999] |         40000 |       44999 | 2017-01-03 00:00:00 | 2017-05-03 23:59:59 | reference   |               0.0126564 |                      0.91  | False                    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+-------------------------+----------------------------+--------------------------+
|  9 | [45000:49999] |         45000 |       49999 | 2017-05-03 00:00:00 | 2017-08-31 23:59:59 | reference   |               2.20383   |                      0.138 | False                    |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+-------------------------+----------------------------+--------------------------+


NannyML can visualize the statistical properties of the drift in model outputs with:

.. code-block:: python

    >>> figure = univariate_results.plot(kind='prediction_drift', metric='statistic')
    >>> figure.show()

.. image:: /_static/drift-guide-predictions.svg

NannyML can also show how the distributions of the model predictions evolved over time:

.. code-block:: python

    >>> figure = univariate_results.plot(kind='prediction_distribution', metric='statistic')
    >>> figure.show()

.. image:: /_static/drift-guide-predictions-joyplot.svg


Insights and Follow Ups
-----------------------

Looking at the results we see that we have a false alert on the first chunk of the analysis data. This is similar
to the ``tenure`` variable in the :ref:`univariate drift results<univariate_drift_detection_tenure>` where there is also
a false alert because the drift measured by the KS d-statistic is very low. This
can happen when the statistical tests consider significant a small change in the distribtion of a variable
in the chunks. But becuase the change is small it is usually not significant from a model monitoring perspective.

If required the :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Outputs.

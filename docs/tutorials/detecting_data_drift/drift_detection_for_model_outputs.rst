.. _drift_detection_for_model_outputs:

=================================
Drift Detection for Model Outputs
=================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of model outputs tells us how likely it is that our population
will do what the model predicts. If the model's
population changes, then our populations' actions will be different.
The difference in actions is very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.


Just The Code
-------------

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor', model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(reference.head())

    >>> # Let's initialize the object that will perform the Univariate Drift calculations
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=5000)
    >>> univariate_calculator = univariate_calculator.fit(reference_data=reference)
    >>> # let's see drift statistics for all available data
    >>> data = pd.concat([reference, analysis], ignore_index=True)
    >>> univariate_results = univariate_calculator.calculate(data=data)
    >>> # let's view a small subset of our results:
    >>> # We use the data property of the results class to view the relevant data.
    >>> y_pred_proba_result_columns = list(univariate_results.data.columns)[:5] + [s for s in list(univariate_results.data.columns) if "y_pred_proba" in s]
    >>> display(univariate_results.data[y_pred_proba_result_columns][-7:-3])

    >>> figure = univariate_results.plot(kind='prediction_drift', metric='statistic')
    >>> figure.show()

    >>> figure = univariate_results.plot(kind='prediction_distribution', metric='statistic')
    >>> figure.show()


Walkthrough
------------------------------------------------

NannyML detects data drift for :term:`Model Outputs` using the
:ref:`Univariate Drift Detection methodology<univariate_drift_detection>`.

Let's start by loading some synthetic data provided by the NannyML package.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
    >>> metadata = nml.extract_metadata(data = reference, model_name='wfh_predictor', model_type='classification_binary', exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(reference.head())


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

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator`
class implements the functionality needed for drift detection in model outputs.

Following the process shown in :ref:`univariate drift detection<univariate_drift_detection>`,
:class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator`
is instantiated with appropriate parameters and the
:meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.fit` method
is called on the reference data that the results will be based off. 

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.calculate` method
calculates the drift results on the data provided. An example using it can be seen below.

.. code-block:: python

    >>> # Let's initialize the object that will perform the Univariate Drift calculations
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_size=5000)
    >>> univariate_calculator = univariate_calculator.fit(reference_data=reference)
    >>> # let's see drift statistics for all available data
    >>> data = pd.concat([reference, analysis], ignore_index=True)
    >>> univariate_results = univariate_calculator.calculate(data=data)
    >>> # let's view a small subset of our results:
    >>> # We use the data property of the results class to view the relevant data.
    >>> y_pred_proba_result_columns = list(univariate_results.data.columns)[:5] + [s for s in list(univariate_results.data.columns) if "y_pred_proba" in s]
    >>> display(univariate_results.data[y_pred_proba_result_columns][-7:-3])

+----+---------------+---------------+-------------+---------------------+---------------------+----------------------+------------------------+----------------------+--------------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            |   y_pred_proba_dstat |   y_pred_proba_p_value | y_pred_proba_alert   |   y_pred_proba_threshold |
+====+===============+===============+=============+=====================+=====================+======================+========================+======================+==========================+
| 13 | [65000:69999] |         65000 |       69999 | 2018-09-01 16:19:07 | 2018-12-31 10:11:21 |              0.01058 |                  0.685 | False                |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------------------+------------------------+----------------------+--------------------------+
| 14 | [70000:74999] |         70000 |       74999 | 2018-12-31 10:38:45 | 2019-04-30 11:01:30 |              0.01408 |                  0.325 | False                |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------------------+------------------------+----------------------+--------------------------+
| 15 | [75000:79999] |         75000 |       79999 | 2019-04-30 11:02:00 | 2019-09-01 00:24:27 |              0.1307  |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------------------+------------------------+----------------------+--------------------------+
| 16 | [80000:84999] |         80000 |       84999 | 2019-09-01 00:28:54 | 2019-12-31 09:09:12 |              0.1273  |                  0     | True                 |                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------------------+------------------------+----------------------+--------------------------+


NannyML can visualize the statistical properties of the drift in model outputs.

.. code-block:: python

    >>> figure = univariate_results.plot(kind='prediction_drift', metric='statistic')
    >>> figure.show()

.. image:: /_static/drift-guide-predictions.svg

NannyML can also show how the distributions of the model predictions evolved over time.

.. code-block:: python

    >>> figure = univariate_results.plot(kind='prediction_distribution', metric='statistic')
    >>> figure.show()

.. image:: /_static/drift-guide-predictions-joyplot.svg


Insights
-----------------------

Looking at the results we can see that we have a false alert on the first chunk of the analysis data. This is similar
to the ``tenure`` variable in the :ref:`univariate drift results<univariate_drift_detection_tenure>`, where there is also
a false alert because the drift measured by the :term:`KS statistic<Kolmogorov-Smirnov test>` is very low. This
can happen when the statistical tests consider a small change in the distribution of a variable
to be significant. But because the change is small it is usually not significant from a model monitoring perspective.

What Next
-----------------------

If required the :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Outputs.

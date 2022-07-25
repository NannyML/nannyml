.. _univariate_drift_detection:

==========================
Univariate Drift Detection
==========================

Why Perform Univariate Drift Detection
--------------------------------------

Univariate Drift Detection looks at each feature individually and checks whether its
distribution has changed. It's a simple, fully explainable form of data drift detection
and is the most straightforward to understand and communicate.

Just The Code
-------------

.. code-block:: python

    import nannyml as nml
    import pandas as pd
    from IPython.display import display

    reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    analysis_df = nml.load_synthetic_binary_classification_dataset()[1]

    display(reference_df.head())

    feature_column_names = [
        col for col in reference_df.columns if col not in ['timestamp', 'y_pred_proba', 'period', 'y_pred', 'work_home_actual', 'identifier']]

    calc = nml.UnivariateStatisticalDriftCalculator(feature_column_names=feature_column_names, timestamp_column_name='timestamp')

    calc.fit(reference_df)

    results = calc.calculate(analysis_df)

    display(results.data.iloc[:5, :9])
    display(results.data.iloc[:-5, :9])

    for feature in calc.feature_column_names:
        drift_fig = results.plot(kind='feature_drift', feature_column_name=feature, plot_reference=True)
        drift_fig.show()

    for cat_feat in calc.categorical_column_names:
        results.plot(kind='feature_distribution', feature_column_name=cat_feat, plot_reference=True).show()

    ranker = nml.Ranker.by('alert_count')
    ranked_features = ranker.rank(results, only_drifting = False)
    display(ranked_features)


Walkthrough
-----------------------------------------

NannyML's Univariate approach for data drift looks at each variable individually and conducts statistical tests comparing the :ref:`chunks<chunking>` created from the analysis :ref:`data period<data-drift-periods>` with the reference period. You can read more about the data required in our section on :ref:`data periods<data-drift-periods>`

NannyML uses the :term:`2 sample Kolmogorov-Smirnov Test<Kolmogorov-Smirnov test>` for continuous features and the
:term:`Chi squared test<Chi Squared test>` for categorical features. Both tests provide a statistic where they measure
the observed drift and a p-value that shows how likely we are to get the observed sample under the assumption that there was no drift.

If the p-value is less than 0.05 NannyML considers the result unlikely to be due to chance and issues an alert for the associated chunk and feature.

We begin by loading some synthetic data provided in the NannyML package. This is data for a binary classification model, but other model types operate in the same way.

.. code-block:: python

    import nannyml as nml
    import pandas as pd
    from IPython.display import display

    reference_df = nml.load_synthetic_binary_classification_dataset()[0]
    analysis_df = nml.load_synthetic_binary_classification_dataset()[1]

    display(reference_df.head())

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------------+---------------------+----------------+-------------+----------+
|  3 |               2.53041  | 20K - 40K €    |               2.31872 |                      7.94425 | False              | Tuesday   | 0.453649 |                  1 | 2014-05-10 01:12:09 |           0.98 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------------+---------------------+----------------+-------------+----------+
|  4 |               2.25364  | 60K+ €         |               2.22127 |                      8.88448 | True               | Thursday  | 5.69526  |                  1 | 2014-05-10 02:21:34 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------------+---------------------+----------------+-------------+----------+

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator`
class implements the functionality needed for Univariate Drift Detection. We need to instantiate it with appropriate parameters - the column headers of the features that we want to run drift detection on, and the timestamp column header. The features can be passed in as a simple list of strings, but here we have created this list by excluding the columns in the dataframe that are not features, and passed that into the argument.

.. code-block:: python

    feature_column_names = [
        col for col in reference_df.columns if col not in ['timestamp', 'y_pred_proba', 'period', 'y_pred', 'work_home_actual']]

    calc = nml.UnivariateStatisticalDriftCalculator(feature_column_names=feature_column_names, timestamp_column_name='timestamp')

Next, the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be compared with. Then the
:meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.calculate` method will
calculate the drift results on the data provided to it.

We then display a small subset of our results by specifying columns in the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.UnivariateStatisticalDriftCalculator.calculate.results` method.

NannyML returns a dataframe with 3 columns for each feature. The first column contains the corresponding test
statistic. The second column contains the corresponding p-value and the third column says whether there
is a drift alert for that feature and chunk.

.. code-block:: python

    calc.fit(reference_df)

    results = calc.calculate(analysis_df)

    display(results.data.iloc[:5, :9])

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   salary_range_chi2 |   salary_range_p_value | salary_range_alert   |
+====+===============+===============+=============+=====================+=====================+=============+=====================+========================+======================+
|  0 | [0:4999]      |             0 |        4999 | 2014-05-09 22:27:20 | 2014-09-09 08:18:27 | reference   |             2.89878 |                  0.407 | False                |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2014-09-09 09:13:35 | 2015-01-09 00:02:51 | reference   |             3.14439 |                  0.37  | False                |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
|  2 | [10000:14999] |         10000 |       14999 | 2015-01-09 00:04:43 | 2015-05-09 15:54:26 | reference   |             2.45188 |                  0.484 | False                |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
|  3 | [15000:19999] |         15000 |       19999 | 2015-05-09 16:02:08 | 2015-09-07 07:14:37 | reference   |             4.06262 |                  0.255 | False                |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
|  4 | [20000:24999] |         20000 |       24999 | 2015-09-07 07:27:47 | 2016-01-08 16:02:05 | reference   |             2.41399 |                  0.491 | False                |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+


.. code-block:: python

    display(univariate_results.data.iloc[-5:, :9])

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   salary_range_chi2 |   salary_range_p_value | salary_range_alert   |
+====+===============+===============+=============+=====================+=====================+=============+=====================+========================+======================+
| 15 | [75000:79999] |         75000 |       79999 | 2019-04-30 11:02:00 | 2019-09-01 00:24:27 | analysis    |             455.622 |                      0 | True                 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
| 16 | [80000:84999] |         80000 |       84999 | 2019-09-01 00:28:54 | 2019-12-31 09:09:12 | analysis    |             428.633 |                      0 | True                 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
| 17 | [85000:89999] |         85000 |       89999 | 2019-12-31 10:07:15 | 2020-04-30 11:46:53 | analysis    |             453.247 |                      0 | True                 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
| 18 | [90000:94999] |         90000 |       94999 | 2020-04-30 12:04:32 | 2020-09-01 02:46:02 | analysis    |             438.26  |                      0 | True                 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+
| 19 | [95000:99999] |         95000 |       99999 | 2020-09-01 02:46:13 | 2021-01-01 04:29:32 | analysis    |             474.892 |                      0 | True                 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+---------------------+------------------------+----------------------+

NannyML can also visualize those results on plots.

.. code-block:: python

    for feature in calc.feature_column_names:
        drift_fig = results.plot(kind='feature_drift', feature_column_name=feature, plot_reference=True)
        drift_fig.show()

.. image:: /_static/drift-guide-distance_from_office.svg

.. image:: /_static/drift-guide-gas_price_per_litre.svg

.. _univariate_drift_detection_tenure:
.. image:: /_static/drift-guide-tenure.svg

.. image:: /_static/drift-guide-wfh_prev_workday.svg

.. image:: /_static/drift-guide-workday.svg

.. image:: /_static/drift-guide-public_transportation_cost.svg

.. image:: /_static/drift-guide-salary_range.svg


NannyML can also plot details about the distributions of different features. In these plots, NannyML highlights the areas with possible data drift.

If we want to focus only on the categorical plots, we can specify that only these be plotted.

.. code-block:: python

    for cat_feat in calc.categorical_column_names:
        results.plot(kind='feature_distribution', feature_column_name=cat_feat, plot_reference=True).show()

.. image:: /_static/drift-guide-stacked-salary_range.svg

.. image:: /_static/drift-guide-stacked-wfh_prev_workday.svg

.. image:: /_static/drift-guide-stacked-workday.svg

NannyML can also rank features according to how many alerts they have had within the data analyzed
for data drift. NannyML allows viewing the ranking of all the model inputs, or just the ones that have drifted.
NannyML provides a dataframe with the resulting ranking of features.

.. code-block:: python

    ranker = nml.Ranker.by('alert_count')
    ranked_features = ranker.rank(results, only_drifting = False)
    display(ranked_features)

+----+----------------------------+--------------------+--------+
|    | feature                    |   number_of_alerts |   rank |
+====+============================+====================+========+
|  0 | salary_range               |                  5 |      1 |
+----+----------------------------+--------------------+--------+
|  1 | wfh_prev_workday           |                  5 |      2 |
+----+----------------------------+--------------------+--------+
|  2 | distance_from_office       |                  5 |      3 |
+----+----------------------------+--------------------+--------+
|  3 | public_transportation_cost |                  5 |      4 |
+----+----------------------------+--------------------+--------+
|  4 | tenure                     |                  2 |      5 |
+----+----------------------------+--------------------+--------+
|  5 | workday                    |                  0 |      6 |
+----+----------------------------+--------------------+--------+
|  6 | gas_price_per_litre        |                  0 |      7 |
+----+----------------------------+--------------------+--------+

Insights
-----------------------

After reviewing the above results we have a good understanding of what has changed in our
model's population.

What Next
-----------------------

The :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Performance.

If needed, we can investigate further as to why our population characteristics have
changed the way they did. This is an ad-hoc investigating that is not covered by NannyML.

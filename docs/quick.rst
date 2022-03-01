.. _quick-start:

=================
Quick Start Guide
=================

NannyML is a library that makes Model Monitoring more productive.
It estimates the performance of your models in absence of the target, detects data drift
and finds the data drift that's responsible for any drop in performance.

NannyML provides a sample synthetic dataset that can be used for testing purposes.


.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> reference, analysis, analysis_target = nml.load_synthetic_sample()
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

The synthetic dataset provided contains a binary classification model that predicts whether
and employee will work from home the next workday or not. The probability of the employee
working from home is included in the ``y_pred_proba`` column. The model inputs are ``distance_from_office``,
``salary_range``, ``gas_price_per_litre``, ``public_transportation_cost``, ``wfh_prev_workday``, ``workday`` and
``tenure``. ``identifier`` is the :term:`Identifier` column and ``timestamp`` is the :term:`Timestamp` column.

The next step is to have NannyML deduce some information about the model from the dataset.

.. code-block:: python

    >>> md = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    >>> md.timestamp_column_name = 'timestamp'
    >>> md.target_column_name = 'work_home_actual'

For help with using NannyML on other data see :ref:`import-data`.

The data are already split into a reference and an analysis partition. NannyML uses the reference partition to
establish a baseline for expected model performance and the analysis partition to check whether
the monitored model keeps performing as expected.
For more information about partitions look :ref:`data-drift-partitions`.

Estimating Performance without Targets
======================================

We see that our data drift detection results contain data drift. NannyML also investigates
the performance implications of this data drift. More information can be found at
:ref:`performance-estimation`.

.. code-block:: python

    >>> # fit estimator and estimate
    >>> cbpe = nml.CBPE(model_metadata=md, chunk_size=5000)
    >>> cbpe.fit(reference_data=df_ref)
    >>> est_perf = cbpe.estimate(data=data)
    >>> # show results
    >>> plots = nml.PerformancePlots(model_metadata=md, chunker=cbpe.chunker)
    >>> figure = plots.plot_cbpe_performance_estimation(est_perf)
    >>> figure.show()

.. image:: ./_static/perf-est-guide-syth-example.svg

We see that the drift we observed is likely to have a negative impact on performance.

Detecting Data Drift
====================

NannyML makes it easy to compute and visualize data drift for the model inputs.
See :ref:`data-drift-practice`.


.. code-block:: python

    >>> # Let's initialize the object that will perform the Univariate Drift calculations
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=md, chunk_size=5000)
    >>> univariate_calculator.fit(reference_data=reference)
    >>> data = pd.concat([reference, analysis])
    >>> univariate_results = univariate_calculator.calculate(data=data)
    >>> # Let's initialize the plotting class:
    >>> plots = nml.DriftPlots(model_metadata=univariate_calculator.model_metadata, chunker=univariate_calculator.chunker)
    >>> # let's plot drift results for all model inputs
    >>> for feature in md.features:
    ...     figure = plots.plot_univariate_statistical_drift(univariate_results, metric='statistic', feature_label=feature.label)
    ...     figure.show()

.. image:: ./_static/drift-guide-distance_from_office.svg

.. image:: ./_static/drift-guide-gas_price_per_litre.svg

.. image:: ./_static/drift-guide-tenure.svg

.. image:: ./_static/drift-guide-wfh_prev_workday.svg

.. image:: ./_static/drift-guide-workday.svg

.. image:: ./_static/drift-guide-public_transportation_cost.svg

.. image:: ./_static/drift-guide-salary_range.svg

When there are a lot of drifted features, NannyML can also rank them by the number of alerts they have raised:

.. code-block:: python

    >>> ranker = nml.Ranker.by('alert_count')
    >>> ranked_features = ranker.rank(univariate_results, only_drifted = False)
    >>> ranked_features

+----+----------------------------+--------------------+--------+
|    | feature                    |   number_of_alerts |   rank |
+====+============================+====================+========+
|  0 | wfh_prev_workday           |                  5 |      1 |
+----+----------------------------+--------------------+--------+
|  1 | salary_range               |                  5 |      2 |
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

NannyML can also look for drift in the model outputs:

.. code-block:: python

    >>> figure = plots.plot_univariate_statistical_prediction_drift(univariate_results, metric='statistic')
    >>> figure.show()

.. image:: ./_static/drift-guide-predictions.svg

More complex data drift cases can get detected by Data Reconstruction with PCA. For more information
see :ref:`Data Reconstruction with PCA Deep Dive<data-reconstruction-pca>`.


.. code-block:: python

    >>> # Let's initialize the object that will perform Data Reconstruction with PCA
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=md, chunk_size=5000)
    >>> # NannyML compares drift versus the full reference dataset.
    >>> rcerror_calculator.fit(reference_data=reference)
    >>> # let's see Reconstruction error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)

.. image:: ./_static/drift-guide-multivariate.svg

Putting everything together, we see that we have some false alerts for the early analysis data
and some true alerts for the late analysis data!

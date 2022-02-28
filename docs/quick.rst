.. _quick-start:

=================
Quick Start Guide
=================

NannyML is a library that makes Model Monitoring easier and more productive.

NannyML provides a sample synthetic dataset, containing a binary classification model,
so you can get started faster.


.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> reference, analysis, analysis_gt = nml.load_synthetic_sample()
    >>> md = nml.extract_metadata(data = reference, model_name='wfh_predictor')
    >>> md.timestamp_column_name = 'timestamp'
    >>> md.ground_truth_column_name = 'work_home_actual'

For help with using NannyML on other data look at :ref:`import-data`.

The data is already split into
a reference and an analysis partition. NannyML uses the reference partition to
establish a baseline for expected model performance and the analysis partition to check whether
the monitored moel will keep performing as expected.
For more information about partitions look :ref:`data-drift-partitions`.

Finding Data Drift
==================

NannyML makes it easy to compute and vizualize data drift for the model inputs.
For more details refer to :ref:`data-drift-practice`.


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
    >>> for itm in md.features:
    ...     fig = plots.plot_univariate_statistical_drift(univariate_results, metric='statistic', feature_label=itm.label)
    ...     fig.show()

.. image:: ../_static/drift-guide-distance_from_office.svg

.. image:: ../_static/drift-guide-gas_price_per_litre.svg

.. image:: ../_static/drift-guide-tenure.svg

.. image:: ../_static/drift-guide-wfh_prev_workday.svg

.. image:: ../_static/drift-guide-workday.svg

.. image:: ../_static/drift-guide-public_transportation_cost.svg

.. image:: ../_static/drift-guide-salary_range.svg

When there are a lot of drifted features NannyML can also rank them, ordering them by the number of alerts they have raised:

.. code-block:: python

    >>> ranker = nml.Ranker(by='alert_count')
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

NannyML also uses Data Reconstruction with PCA to compute the reconstruction error and detect more complex
data drift cases. More information can be found at
:ref:`Data Reconstruction with PCA Deep Dive<data-reconstruction-pca>`.


.. code-block:: python

    >>> # Let's initialize the object that will perform Data Reconstruction with PCA
    >>> # Let's use a chunk size of 5000 data points to create our drift statistics
    >>> rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=md, chunk_size=5000)
    >>> # NannyML compares drift versus the full reference dataset.
    >>> rcerror_calculator.fit(reference_data=reference)
    >>> # let's see RC error statistics for all available data
    >>> rcerror_results = rcerror_calculator.calculate(data=data)

.. image:: ../_static/drift-guide-multivariate.svg

Estimating Performance Impact
=============================

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
    >>> fig = plots.plot_cbpe_performance_estimation(est_perf)
    >>> fig.show()

.. image:: ../_static/perf-est-guide-syth-example.svg

We see that the drift we observed will likely cause a significant performance drop!

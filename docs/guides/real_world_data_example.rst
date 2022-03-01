=============================
Example on real world dataset
=============================

Below one can find an example usage of NannyML on modified California
Housing Prices dataset. Even though the original dataset does not fit the use case directly as
it is a regression task, its popularity and its size (small enough to be
quickly fetched yet large enough to show and interpret results) made it
suitable for quick example.

See what modifications were made to the data to make it suitable for the
use case :ref:`here<california-housing>`.

Load and prepare data
~~~~~~~~~~~~~~~~~~~~~~
Let's load the dataset from NannyML datasets:

.. code:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>>
    >>> # load data
    >>> df_ref, df_ana, df_ana_gt = nml.datasets.load_modified_california_housing_dataset()
    >>> df_ref.head(3)

+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+
|    |   MedInc |   HouseAge |   AveRooms |   AveBedrms |   Population |   AveOccup |   Latitude |   Longitude | timestamp           | partition   |   clf_target |   y_pred_proba |   identifier |
+====+==========+============+============+=============+==============+============+============+=============+=====================+=============+==============+================+==============+
|  0 |   9.8413 |         32 |    7.17004 |     1.01484 |         4353 |    2.93725 |      34.22 |     -118.19 | 2020-10-01 00:00:00 | reference   |            1 |           0.99 |            0 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+
|  1 |   8.3695 |         37 |    7.45875 |     1.06271 |          941 |    3.10561 |      34.22 |     -118.21 | 2020-10-01 01:00:00 | reference   |            1 |           1    |            1 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+
|  2 |   8.72   |         44 |    6.16318 |     1.04603 |          668 |    2.79498 |      34.2  |     -118.18 | 2020-10-01 02:00:00 | reference   |            1 |           1    |            2 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+


.. code:: python

    >>> # extract metadata, add gt column name
    >>> md = nml.extract_metadata(df_ref)
    >>> md.ground_truth_column_name = 'clf_target'
    >>> md.timestamp_column_name = 'timestamp'

Performance Estimation
~~~~~~~~~~~~~~~~~~~~~~
Let's estimate performance for reference and analysis partitions:

.. code:: python

    >>> # fit performance estimator and estimate for combined reference and analysis
    >>> cbpe = nml.CBPE(model_metadata=md, chunk_period='M')
    >>> cbpe.fit(reference_data=df_ref)
    >>> est_perf = cbpe.estimate(pd.concat([df_ref, df_ana]))

.. parsed-literal::

    UserWarning: The resulting list of chunks contains 1 underpopulated chunks.They contain too few records to be statistically relevant and might negatively influence the quality of calculations.Please consider splitting your data in a different way or continue at your own risk.

Some chunks are too small, most likely the last one, let's see:

.. code:: python

    >>> est_perf.tail(3)

+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+
|    | key     |   start_index |   end_index | start_date          | end_date            | partition   |   estimated_roc_auc |   confidence |   upper_threshold |   lower_threshold | alert   |
+====+=========+===============+=============+=====================+=====================+=============+=====================+==============+===================+===================+=========+
| 17 | 2022-03 |          6552 |        7295 | 2022-03-01 00:00:00 | 2022-03-31 23:59:59 | analysis    |            0.829077 |     0.051046 |                 1 |          0.708336 | False   |
+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+
| 18 | 2022-04 |          7296 |        8015 | 2022-04-01 00:00:00 | 2022-04-30 23:59:59 | analysis    |            0.910661 |     0.051046 |                 1 |          0.708336 | False   |
+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+
| 19 | 2022-05 |          8016 |        8231 | 2022-05-01 00:00:00 | 2022-05-09 23:59:59 | analysis    |            0.939883 |     0.051046 |                 1 |          0.708336 | False   |
+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+


Indeed, the last one is smaller than other due to selected chunking method. Let's remove it for clarity of
visualizations.

.. code:: python

    >>> est_perf = est_perf[:-1].copy()
    >>> est_perf.tail(2)

+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+---------------------------+-------------+------------------+
|    | key     |   start_index |   end_index | start_date          | end_date            | partition   |   estimated_roc_auc |   confidence |   upper_threshold |   lower_threshold | alert   | thresholds                | estimated   |   actual_roc_auc |
+====+=========+===============+=============+=====================+=====================+=============+=====================+==============+===================+===================+=========+===========================+=============+==================+
| 17 | 2022-03 |          6552 |        7295 | 2022-03-01 00:00:00 | 2022-03-31 23:59:59 | analysis    |            0.829077 |     0.051046 |                 1 |          0.708336 | False   | (0.7083356125891167, 1.0) | True        |         0.704867 |
+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+---------------------------+-------------+------------------+
| 18 | 2022-04 |          7296 |        8015 | 2022-04-01 00:00:00 | 2022-04-30 23:59:59 | analysis    |            0.910661 |     0.051046 |                 1 |          0.708336 | False   | (0.7083356125891167, 1.0) | True        |         0.975394 |
+----+---------+---------------+-------------+---------------------+---------------------+-------------+---------------------+--------------+-------------------+-------------------+---------+---------------------------+-------------+------------------+

Let's plot the estimated performance:

.. code:: python

    >>> plots = nml.PerformancePlots(model_metadata=md, chunker=cbpe.chunker)
    >>> fig = plots.plot_cbpe_performance_estimation(est_perf)
    >>> fig.show()

.. image:: ../_static/example_california_performance.svg

CBPE estimates significant performance drop in the chunk corresponding
to the month of September.

Comparison with the actual performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s use the ground truth that we have to
calculate AUROC on relevant chunks and compare:

.. code:: python

    >>> from sklearn.metrics import roc_auc_score
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # add ground truth to analysis
    >>> df_ana_full = pd.merge(df_ana,df_ana_gt, on = 'identifier')
    >>> df_all = pd.concat([df_ref, df_ana_full]).reset_index(drop=True)
    >>> df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    >>> # calculate actual ROC AUC
    >>> target_col = md.ground_truth_column_name
    >>> pred_score_col = 'y_pred_proba'
    >>> actual_performance = []
    >>>
    >>> for idx in est_perf.index:
    >>>     start_date, end_date = est_perf.loc[idx, 'start_date'], est_perf.loc[idx, 'end_date']
    >>>      sub = df_all[df_all['timestamp'].between(start_date, end_date)]
    >>>     actual_perf = roc_auc_score(sub[target_col], sub[pred_score_col])
    >>>     est_perf.loc[idx, 'actual_roc_auc'] = actual_perf
    >>> # plot
    >>> first_analysis = est_perf[est_perf['partition']=='analysis']['key'].values[0]
    >>> plt.plot(est_perf['key'], est_perf['estimated_roc_auc'], label='estimated AUC')
    >>> plt.plot(est_perf['key'], est_perf['actual_roc_auc'], label='actual ROC AUC')
    >>> plt.xticks(rotation=90)
    >>> plt.axvline(x=first_analysis, label='First analysis chunk', linestyle=':', color='grey')
    >>> plt.ylabel('ROC AUC')
    >>> plt.legend()
    >>> plt.show()

.. image:: ../_static/example_california_performance_estimation_tmp.svg

The significant drop at the first few chunks of the analysis period was
estimated accurately. After that the overall trend seems to be well
represented. The estimation of performance has lower variance than
actual performance. This is expected.

Drift detection
~~~~~~~~~~~~~~~

Let’s search for the reasons of this performance drop and investigate what
drifted using drift detection on univariate features.

.. code:: python

    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=md, chunk_period='M')
    >>> univariate_calculator.fit(reference_data=df_ref)
    >>> univariate_results = univariate_calculator.calculate(data=pd.concat([df_ana]))
    >>> nml.Ranker.by('alert_count').rank(univariate_results, only_drifting=True)


+----+--------------+--------------------+--------+
|    | feature      |   number_of_alerts |   rank |
+====+==============+====================+========+
|  0 | Latitude     |                 12 |      1 |
+----+--------------+--------------------+--------+
|  1 | AveOccup     |                 12 |      2 |
+----+--------------+--------------------+--------+
|  2 | Longitude    |                 12 |      3 |
+----+--------------+--------------------+--------+
|  3 | HouseAge     |                 12 |      4 |
+----+--------------+--------------------+--------+
|  4 | y_pred_proba |                 11 |      5 |
+----+--------------+--------------------+--------+
|  5 | MedInc       |                 11 |      6 |
+----+--------------+--------------------+--------+
|  6 | AveRooms     |                 11 |      7 |
+----+--------------+--------------------+--------+
|  7 | AveBedrms    |                  8 |      8 |
+----+--------------+--------------------+--------+
|  8 | Population   |                  8 |      9 |
+----+--------------+--------------------+--------+


It looks like there is a lot of drift in this dataset. Since we have 12
chunks in analysis period, top 4 features drifted in all analyzed
chunks. Let’s look at the intensity of this drift by looking at KS
distance statistics.

.. code:: python

    >>> # get columns with d statistics only
    >>> d_stat_cols = [x for x in univariate_results if 'dstat' in x]
    >>> univariate_results[d_stat_cols].mean().sort_values(ascending=False)

+------------------+-----------+
|                  |         0 |
+==================+===========+
| Longitude_dstat  | 0.836534  |
+------------------+-----------+
| Latitude_dstat   | 0.799592  |
+------------------+-----------+
| HouseAge_dstat   | 0.173479  |
+------------------+-----------+
| MedInc_dstat     | 0.158278  |
+------------------+-----------+
| AveOccup_dstat   | 0.133803  |
+------------------+-----------+
| AveRooms_dstat   | 0.110907  |
+------------------+-----------+
| AveBedrms_dstat  | 0.0786656 |
+------------------+-----------+
| Population_dstat | 0.0713122 |
+------------------+-----------+

The mean value of D-statistic for Longitude and Latitude on analysis chunks is the largest. Let’s plot their
distributions for the analysis period.

.. code:: python

    >>> plots = nml.DriftPlots(model_metadata=univariate_calculator.model_metadata, chunker=univariate_calculator.chunker)
    >>> for label in ['Longitude', 'Latitude']:
    >>>     fig = plots.plot_continuous_feature_distribution_over_time(
    >>>         data=df_ana,
    >>>         drift_results=univariate_results,
    >>>         feature_label=label)
    >>>     fig.show()


.. image:: ../_static/example_california_performance_distribution_Longitude.svg

.. image:: ../_static/example_california_performance_distribution_Latitude.svg

Indeed, distributions of these variables are completely different in each
chunk. This was expected as the original dataset has observations from
nearby locations next to each other. Let’s see it on a scatter plot:

.. code:: python

    >>> analysis_res = est_perf[est_perf['partition']=='analysis']
    >>> plt.figure(figsize=(8,6))
    >>> for idx in analysis_res.index[:10]:
    >>>     start_date, end_date = analysis_res.loc[idx, 'start_date'], analysis_res.loc[idx, 'end_date']
    >>>     sub = df_all[df_all['timestamp'].between(start_date, end_date)]
    >>>     plt.scatter(sub['Latitude'], sub['Longitude'], s=5, label="Chunk {}".format(str(idx)))
    >>> plt.legend()
    >>> plt.xlabel('Latitude')
    >>> plt.ylabel('Longitude')

.. image:: ../_static/example_california_latitude_longitude_scatter.svg

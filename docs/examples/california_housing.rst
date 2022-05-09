=============================
California Housing Dataset
=============================

This document outlines a typical workflow for estimating performance of a model without access to ground truth, detecting performance issues and identifying potential root causes for these issues.
Below, one can find an example use of NannyML on the modified California Housing Prices dataset.

See what modifications were made to the data to make it suitable for the
use case :ref:`here<california-housing-appendix>`.


---------------------------------------
Monitoring workflow with NannyML
---------------------------------------


Load and prepare data
=====================


Let's load the dataset from NannyML datasets:

.. code:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> # load data
    >>> reference, analysis, analysis_gt = nml.datasets.load_modified_california_housing_dataset()
    >>> reference.head(3)

+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+
|    |   MedInc |   HouseAge |   AveRooms |   AveBedrms |   Population |   AveOccup |   Latitude |   Longitude | timestamp           | partition   |   clf_target |   y_pred_proba |   identifier |
+====+==========+============+============+=============+==============+============+============+=============+=====================+=============+==============+================+==============+
|  0 |   9.8413 |         32 |    7.17004 |     1.01484 |         4353 |    2.93725 |      34.22 |     -118.19 | 2020-10-01 00:00:00 | reference   |            1 |           0.99 |            0 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+
|  1 |   8.3695 |         37 |    7.45875 |     1.06271 |          941 |    3.10561 |      34.22 |     -118.21 | 2020-10-01 01:00:00 | reference   |            1 |           1    |            1 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+
|  2 |   8.72   |         44 |    6.16318 |     1.04603 |          668 |    2.79498 |      34.2  |     -118.18 | 2020-10-01 02:00:00 | reference   |            1 |           1    |            2 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+

The data only contains the predicted probabilities in the ``y_pred_proba`` column right now. Performance estimation
requires access to the predicted labels as well. In the case of a binary classifier it is easy to add these
to the dataset by thresholding.

.. code-block:: python

    >>> reference['y_pred'] = reference['y_pred_proba'].map(lambda p: int(p >= 0.8))
    >>> analysis['y_pred'] = analysis['y_pred_proba'].map(lambda p: int(p >= 0.8))
    >>> reference.head(3)

+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+----------+
|    |   MedInc |   HouseAge |   AveRooms |   AveBedrms |   Population |   AveOccup |   Latitude |   Longitude | timestamp           | partition   |   clf_target |   y_pred_proba |   identifier |   y_pred |
+====+==========+============+============+=============+==============+============+============+=============+=====================+=============+==============+================+==============+==========+
|  0 |   9.8413 |         32 |    7.17004 |     1.01484 |         4353 |    2.93725 |      34.22 |     -118.19 | 2020-10-01 00:00:00 | reference   |            1 |           0.99 |            0 |        1 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+----------+
|  1 |   8.3695 |         37 |    7.45875 |     1.06271 |          941 |    3.10561 |      34.22 |     -118.21 | 2020-10-01 01:00:00 | reference   |            1 |           1    |            1 |        1 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+----------+
|  2 |   8.72   |         44 |    6.16318 |     1.04603 |          668 |    2.79498 |      34.2  |     -118.18 | 2020-10-01 02:00:00 | reference   |            1 |           1    |            2 |        1 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------------+-------------+--------------+----------------+--------------+----------+

.. code:: python

    >>> # extract metadata, add gt column name
    >>> metadata = nml.extract_metadata(reference, exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'clf_target'
    >>> metadata.timestamp_column_name = 'timestamp'

Performance Estimation
======================


Let's estimate performance for reference and analysis partitions:

.. code:: python

    >>> # fit performance estimator and estimate for combined reference and analysis
    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_period='M', metrics=['roc_auc']).fit(reference_data=reference)
    >>> est_perf = cbpe.estimate(pd.concat([reference, analysis]))

.. parsed-literal::

    UserWarning: The resulting list of chunks contains 1 underpopulated chunks.They contain too few records to be statistically relevant and might negatively influence the quality of calculations.Please consider splitting your data in a different way or continue at your own risk.

Some chunks are too small, most likely the last one, let's see:

.. code:: python

    >>> est_perf.data.tail(3)

+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+
|    | key     |   start_index |   end_index | start_date          | end_date                      | partition   |   confidence_roc_auc |   realized_roc_auc |   estimated_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |
+====+=========+===============+=============+=====================+===============================+=============+======================+====================+=====================+===========================+===========================+=================+
| 17 | 2022-03 |          6552 |        7295 | 2022-03-01 00:00:00 | 2022-03-31 23:59:59.999999999 | analysis    |             0.051046 |                nan |            0.829077 |                  0.708336 |                         1 | False           |
+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+
| 18 | 2022-04 |          7296 |        8015 | 2022-04-01 00:00:00 | 2022-04-30 23:59:59.999999999 | analysis    |             0.051046 |                nan |            0.910661 |                  0.708336 |                         1 | False           |
+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+
| 19 | 2022-05 |          8016 |        8231 | 2022-05-01 00:00:00 | 2022-05-31 23:59:59.999999999 | analysis    |             0.051046 |                nan |            0.939883 |                  0.708336 |                         1 | False           |
+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+

Indeed, the last one is smaller than the others due to the selected chunking method. Let's remove it for clarity of visualizations.

.. code:: python

    >>> est_perf.data = est_perf.data[:-1].copy()
    >>> est_perf.data.tail(2)

+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+
|    | key     |   start_index |   end_index | start_date          | end_date                      | partition   |   confidence_roc_auc |   realized_roc_auc |   estimated_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |
+====+=========+===============+=============+=====================+===============================+=============+======================+====================+=====================+===========================+===========================+=================+
| 16 | 2022-02 |          5880 |        6551 | 2022-02-01 00:00:00 | 2022-02-28 23:59:59.999999999 | analysis    |             0.051046 |                nan |            0.911054 |                  0.708336 |                         1 | False           |
+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+
| 17 | 2022-03 |          6552 |        7295 | 2022-03-01 00:00:00 | 2022-03-31 23:59:59.999999999 | analysis    |             0.051046 |                nan |            0.829077 |                  0.708336 |                         1 | False           |
+----+---------+---------------+-------------+---------------------+-------------------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+


Let's plot the estimated performance:

.. code:: python

    >>> fig = est_perf.plot(kind='performance', metric='roc_auc')
    >>> fig.show()

.. image:: ../_static/example_california_performance.svg

CBPE estimates a significant performance drop in the chunk corresponding
to the month of September.

Comparison with the actual performance
======================================

Let’s use the ground truth that we have to
calculate ROC AUC on relevant chunks and compare:

.. code:: python

    >>> from sklearn.metrics import roc_auc_score
    >>> import matplotlib.pyplot as plt
    >>> # add ground truth to analysis
    >>> analysis_full = pd.merge(analysis,analysis_gt, on = 'identifier')
    >>> df_all = pd.concat([reference, analysis_full]).reset_index(drop=True)
    >>> df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    >>> # calculate actual ROC AUC
    >>> target_col = metadata.target_column_name
    >>> pred_score_col = 'y_pred_proba'
    >>> actual_performance = []
    >>> for idx in est_perf.data.index:
    >>>     start_date, end_date = est_perf.data.loc[idx, 'start_date'], est_perf.data.loc[idx, 'end_date']
    >>>     sub = df_all[df_all['timestamp'].between(start_date, end_date)]
    >>>     actual_perf = roc_auc_score(sub[target_col], sub[pred_score_col])
    >>>     est_perf.data.loc[idx, 'actual_roc_auc'] = actual_perf
    >>> # plot
    >>> first_analysis = est_perf.data[est_perf.data['partition']=='analysis']['key'].values[0]
    >>> plt.plot(est_perf.data['key'], est_perf.data['estimated_roc_auc'], label='estimated AUC')
    >>> plt.plot(est_perf.data['key'], est_perf.data['actual_roc_auc'], label='actual ROC AUC')
    >>> plt.xticks(rotation=90)
    >>> plt.axvline(x=first_analysis, label='First analysis chunk', linestyle=':', color='grey')
    >>> plt.ylabel('ROC AUC')
    >>> plt.legend()
    >>> plt.show()

.. image:: ../_static/example_california_performance_estimation_tmp.svg

The significant drop at the first few chunks of the analysis period was
estimated accurately. After that the overall trend seems to be well
represented. The estimation of performance has a lower variance than
actual performance.

Drift detection
===============

The next step is to dig deeper to find out what might be responsible for this drop in ROC AUC. Let’s do it using
univariate drift detection.

.. code:: python

    >>> univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata=metadata, chunk_period='M').fit(reference_data=reference)
    >>> univariate_results = univariate_calculator.calculate(data=pd.concat([analysis]))
    >>> nml.Ranker.by('alert_count').rank(univariate_results, metadata)


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
|  4 | MedInc       |                 11 |      5 |
+----+--------------+--------------------+--------+
|  5 | AveRooms     |                 11 |      6 |
+----+--------------+--------------------+--------+
|  6 | AveBedrms    |                  8 |      7 |
+----+--------------+--------------------+--------+
|  7 | Population   |                  8 |      8 |
+----+--------------+--------------------+--------+


It looks like there is a lot of drift in this dataset. Since we have 12 chunks in the analysis period, top 4 features drifted in all analyzed chunks. Let’s look at the magnitude of this drift by looking at the KS distance statistics.

.. code:: python

    >>> # get columns with d statistics only
    >>> d_stat_cols = [x for x in univariate_results.data if 'dstat' in x]
    >>> univariate_results.data[d_stat_cols].mean().sort_values(ascending=False)

+------------------+-----------+
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

    >>> for label in ['Longitude', 'Latitude']:
    >>>     fig = univariate_results.plot(
    >>>         kind='feature_distribution',
    >>>         feature_label=label)
    >>>     fig.show()


.. image:: ../_static/example_california_performance_distribution_Longitude.svg

.. image:: ../_static/example_california_performance_distribution_Latitude.svg

Indeed, distributions of these variables are completely different in each
chunk. This was expected, as the original dataset has observations from
nearby locations next to each other. Let’s see it on a scatter plot:

.. code:: python

    >>> analysis_res = est_perf.data[est_perf.data['partition']=='analysis']
    >>> plt.figure(figsize=(8,6))
    >>> for idx in analysis_res.index[:10]:
    >>>     start_date, end_date = analysis_res.loc[idx, 'start_date'], analysis_res.loc[idx, 'end_date']
    >>>     sub = df_all[df_all['timestamp'].between(start_date, end_date)]
    >>>     plt.scatter(sub['Latitude'], sub['Longitude'], s=5, label="Chunk {}".format(str(idx)))
    >>> plt.legend()
    >>> plt.xlabel('Latitude')
    >>> plt.ylabel('Longitude')

.. image:: ../_static/example_california_latitude_longitude_scatter.svg

In summary, NannyML estimated the performance (ROC AUC) of a model without accessing the target data. The estimate is
quite accurate. Next, the potential root causes of the drop in performance were indicated by
detecting data drift. This was achieved using univariate methods that identify features which drifted the most.


.. _california-housing-appendix:

----------------------------------------------
Appendix: Modifying California Housing Dataset
----------------------------------------------

We are using the `California Housing Dataset`_ to create a real data example dataset for
NannyML. There are three steps needed for this process:

- Enriching the data
- Training a Machine Learning Model
- Meeting NannyML Data Requirements


Let’s start by loading the dataset:

.. code-block:: python

    >>> # Import required libraries
    >>> import pandas as pd
    >>> import numpy as np
    >>> import datetime as dt

    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import roc_auc_score

    >>> cali = fetch_california_housing(as_frame=True)
    >>> df = pd.concat([cali.data, cali.target], axis=1)
    >>> df.head(2)

+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------+
|    |   MedInc |   HouseAge |   AveRooms |   AveBedrms |   Population |   AveOccup |   Latitude |   Longitude |   MedHouseVal |
+====+==========+============+============+=============+==============+============+============+=============+===============+
|  0 |   8.3252 |         41 |    6.98413 |     1.02381 |          322 |    2.55556 |      37.88 |     -122.23 |         4.526 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------+
|  1 |   8.3014 |         21 |    6.23814 |     0.97188 |         2401 |    2.10984 |      37.86 |     -122.22 |         3.585 |
+----+----------+------------+------------+-------------+--------------+------------+------------+-------------+---------------+


Enriching the data
==================

The things that need to be added to the dataset are:

- A time dimension
- Splitting the data into reference and analysis sets
- A binary classification target

.. code-block:: python

    >>> # add artificial timestamp
    >>> timestamps = [dt.datetime(2020,1,1) + dt.timedelta(hours=x/2) for x in df.index]
    >>> df['timestamp'] = timestamps

    >>> # add partitions
    >>> train_beg = dt.datetime(2020,1,1)
    >>> train_end = dt.datetime(2020,5,1)
    >>> test_beg = dt.datetime(2020,5,1)
    >>> test_end = dt.datetime(2020,9,1)
    >>> df.loc[df['timestamp'].between(train_beg, train_end, inclusive='left'), 'partition'] = 'train'
    >>> df.loc[df['timestamp'].between(test_beg, test_end, inclusive='left'), 'partition'] = 'test'
    >>> df['partition'] = df['partition'].fillna('production')

    >>> # create new classification target - house value higher than mean
    >>> df_train = df[df['partition']=='train']
    >>> df['clf_target'] = np.where(df['MedHouseVal'] > df_train['MedHouseVal'].median(), 1, 0)
    >>> df = df.drop('MedHouseVal', axis=1)
    >>> del df_train

Training a Machine Learning Model
=================================

.. code-block:: python

    >>> # fit classifier
    >>> target = 'clf_target'
    >>> meta = 'partition'
    >>> features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']


    >>> df_train = df[df['partition']=='train']

    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(df_train[features], df_train[target])
    >>> df['y_pred_proba'] = clf.predict_proba(df[features])[:,1]

    >>> # Check roc auc score
    >>> for partition_name, partition_data in df.groupby('partition', sort=False):
    ...     print(partition_name, roc_auc_score(partition_data[target], partition_data['y_pred_proba']))
    train 1.0
    test 0.8737681614409617
    production 0.8224322932364313

Meeting NannyML Data Requirements
=================================

The data are now being splitted so they can be in a form required by NannyML.

.. code-block:: python

    >>> df_for_nanny = df[df['partition']!='train'].reset_index(drop=True)
    >>> df_for_nanny['partition'] = df_for_nanny['partition'].map({'test':'reference', 'production':'analysis'})
    >>> df_for_nanny['identifier'] = df_for_nanny.index

    >>> reference = df_for_nanny[df_for_nanny['partition']=='reference'].copy()
    >>> analysis = df_for_nanny[df_for_nanny['partition']=='analysis'].copy()
    >>> analysis_target = analysis[['identifier', 'clf_target']].copy()
    >>> analysis = analysis.drop('clf_target', axis=1)

The ``reference`` dataframe represents the reference :term:`Partition` and the ``analysis``
dataframe represents the analysis partition. The ``analysis_target`` dataframe contains the targets
for the analysis partition that is provided separately.


.. _California Housing Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

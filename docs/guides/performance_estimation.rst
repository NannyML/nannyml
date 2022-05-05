.. _performance-estimation:

======================
Performance Estimation
======================

This guide explains how to use NannyML to estimate the performance of a monitored model (in absence of ground truth). The guide is based on a synthetic dataset where the monitored model predicts whether an employee will work from home.

Prepare the data
================

Let's first load the data and have a quick look:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_sample()
    >>> reference.head(3)

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+

The data only contains the predicted probabilities in the ``y_pred_proba`` column right now. To address the full range
of metrics the CBPE supports it will need to access the predicted labels as well.
In the case of a binary classifier it is easy to add these to the dataset by thresholding.

.. code-block:: python

    >>> reference['y_pred'] = reference['y_pred_proba'].map(lambda p: int(p >= 0.8))
    >>> analysis['y_pred'] = analysis['y_pred_proba'].map(lambda p: int(p >= 0.8))
    >>> reference.head(3)

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier |   work_home_actual | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+====================+=====================+================+=============+==========+
|  0 |               5.96225  | 40K - 60K €    |               2.11948 |                      8.56806 | False              | Friday    | 0.212653 |            0 |                  1 | 2014-05-09 22:27:20 |           0.99 | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  1 |               0.535872 | 40K - 60K €    |               2.3572  |                      5.42538 | True               | Tuesday   | 4.92755  |            1 |                  0 | 2014-05-09 22:59:32 |           0.07 | reference   |        0 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+
|  2 |               1.96952  | 40K - 60K €    |               2.36685 |                      8.24716 | False              | Monday    | 0.520817 |            2 |                  1 | 2014-05-09 23:48:25 |           1    | reference   |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+--------------------+---------------------+----------------+-------------+----------+


.. code-block:: python

    >>> analysis.head(3)

+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+----------+
|    |   distance_from_office | salary_range   |   gas_price_per_litre |   public_transportation_cost | wfh_prev_workday   | workday   |   tenure |   identifier | timestamp           |   y_pred_proba | partition   |   y_pred |
+====+========================+================+=======================+==============================+====================+===========+==========+==============+=====================+================+=============+==========+
|  0 |               0.527691 | 0 - 20K €      |               1.8     |                      8.96072 | False              | Tuesday   |  4.22463 |        50000 | 2017-08-31 04:20:00 |           0.99 | analysis    |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+----------+
|  1 |               8.48513  | 20K - 20K €    |               2.22207 |                      8.76879 | False              | Friday    |  4.9631  |        50001 | 2017-08-31 05:16:16 |           0.98 | analysis    |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+----------+
|  2 |               2.07388  | 40K - 60K €    |               2.31008 |                      8.64998 | True               | Friday    |  4.58895 |        50002 | 2017-08-31 05:56:44 |           0.98 | analysis    |        1 |
+----+------------------------+----------------+-----------------------+------------------------------+--------------------+-----------+----------+--------------+---------------------+----------------+-------------+----------+


``reference`` and ``analysis`` correspond to ``reference`` and ``analysis`` partitions of the monitored data. To
understand what they are read :ref:`data partitions<data-drift-partitions>`. Let's leave
``analysis_gt`` for now, it will be described and used later.

Let's extract the metadata and complete the missing information:

.. code-block:: python

    >>> analysis.head(3)
    >>> metadata = nml.extract_metadata(reference, exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'


Full information on how the data should be prepared can be found in the guide on :ref:`importing data<import-data>`.

Creating and using the estimator
================================

In the next step Confidence-based Performance Estimation (CBPE) estimator is created and fitted on ``reference`` data.
Both the chunking method and the metrics to estimate need to be specified now.
Read more about chunking in relevant :ref:`guide<chunk-data>`.

.. code-block:: python

    >>> cbpe = nml.CBPE(model_metadata=metadata, chunk_size=5000, metrics=['roc_auc', 'f1', 'precision', 'recall']).fit(reference_data=reference)

The fitted ``cbpe`` can be used to estimate performance on other data, for which performance cannot be calculated.
Typically, this would be used on the latest production data where ground truth is missing (i.e. the ``analysis``
partition).

However, it can be also used on combined ``reference`` and ``analysis`` data, e.g. when comparing
estimations of ``reference`` and ``analysis`` data or comparing the estimated performance versus the realized
performance on ``reference`` data.

.. code-block:: python

    >>> est_perf = cbpe.estimate(pd.concat([reference, analysis], ignore_index=True))

To find out how CBPE estimates performance, read the relevant :ref:`deep dive<performance-estimation-deep-dive>`.

View the results
================

To get the data frame with results:

.. code-block:: python

    >>> est_perf.data.head(3)

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+------------------------+----------------------+-----------------------+-----------------------------+-----------------------------+-------------------+---------------------+-------------------+--------------------+--------------------------+--------------------------+----------------+------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   confidence_roc_auc |   realized_roc_auc |   estimated_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |   confidence_f1 |   realized_f1 |   estimated_f1 |   upper_threshold_f1 |   lower_threshold_f1 | alert_f1   |   confidence_precision |   realized_precision |   estimated_precision |   upper_threshold_precision |   lower_threshold_precision | alert_precision   |   confidence_recall |   realized_recall |   estimated_recall |   upper_threshold_recall |   lower_threshold_recall | alert_recall   |   actual_roc_auc |
+====+===============+===============+=============+=====================+=====================+=============+======================+====================+=====================+===========================+===========================+=================+=================+===============+================+======================+======================+============+========================+======================+=======================+=============================+=============================+===================+=====================+===================+====================+==========================+==========================+================+==================+
|  0 | [0:4999]      |             0 |        4999 | 2014-05-09 22:27:20 | 2014-09-09 08:18:27 | reference   |           0.00035752 |           0.976477 |            0.969051 |                  0.963317 |                   0.97866 | False           |      0.00145944 |      0.926044 |       0.921705 |             0.911932 |             0.928751 | False      |            0.000579414 |             0.972408 |              0.966623 |                    0.955649 |                    0.978068 | False             |          0.00270608 |          0.8839   |           0.880777 |                  0.86706 |                 0.889152 | False          |         0.976253 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+------------------------+----------------------+-----------------------+-----------------------------+-----------------------------+-------------------+---------------------+-------------------+--------------------+--------------------------+--------------------------+----------------+------------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2014-09-09 09:13:35 | 2015-01-09 00:02:51 | reference   |           0.00035752 |           0.968899 |            0.968909 |                  0.963317 |                   0.97866 | False           |      0.00145944 |      0.917111 |       0.917418 |             0.911932 |             0.928751 | False      |            0.000579414 |             0.965889 |              0.966807 |                    0.955649 |                    0.978068 | False             |          0.00270608 |          0.873022 |           0.87283  |                  0.86706 |                 0.889152 | False          |         0.969045 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+------------------------+----------------------+-----------------------+-----------------------------+-----------------------------+-------------------+---------------------+-------------------+--------------------+--------------------------+--------------------------+----------------+------------------+
|  2 | [10000:14999] |         10000 |       14999 | 2015-01-09 00:04:43 | 2015-05-09 15:54:26 | reference   |           0.00035752 |           0.972    |            0.968657 |                  0.963317 |                   0.97866 | False           |      0.00145944 |      0.917965 |       0.919083 |             0.911932 |             0.928751 | False      |            0.000579414 |             0.965066 |              0.96696  |                    0.955649 |                    0.978068 | False             |          0.00270608 |          0.875248 |           0.875723 |                  0.86706 |                 0.889152 | False          |         0.971742 |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+------------------------+----------------------+-----------------------+-----------------------------+-----------------------------+-------------------+---------------------+-------------------+--------------------+--------------------------+--------------------------+----------------+------------------+

.. _performance-estimation-thresholds:

Apart from chunking and chunk and partition-related data, the results data have the following columns for each metric
that was estimated:

 - ``estimated_<metric>`` - the estimate of ROC AUC for a specific chunk,
 - ``confidence_<metric>`` - the width of the confidence band. It is equal to 1 standard deviation of performance estimates on
   `reference` data (hence calculated during ``fit`` phase).
 - ``upper_threshold_<metric>`` and ``lower_threshold_<metric>`` - crossing these thresholds will raise an alert on significant
   performance change. The thresholds are calculated based on the actual performance of the monitored model on chunks in
   the ``reference`` partition. The thresholds are 3 standard deviations away from the mean performance calculated on
   chunks.
   They are calculated during ``fit`` phase.
 - ``realized_<metric>`` - when ``target`` values are available for a chunk, the realized performance metric will also
   be calculated and included within the results.
 - ``alert_<metric>`` - flag indicating potentially significant performance change. ``True`` if estimated performance crosses
   upper or lower threshold.


The results can be also plotted:

.. code-block:: python

    >>> for metric in cbpe.metrics:
            est_perf.plot(kind='performance', metric=metric).show()


.. image:: ../_static/perf-est-guide-roc_auc.svg

.. image:: ../_static/perf-est-guide-f1.svg

.. image:: ../_static/perf-est-guide-precision.svg

.. image:: ../_static/perf-est-guide-recall.svg


Compare with the actual performance
===================================

When the ground truth becomes available, the quality of estimation can be evaluated. For the synthetic dataset, the
ground truth is given in ``analysis_gt`` variable. It consists of ``identifier`` that allows to match it with
``analysis`` data and the target for monitored model - ``work_home_actual``:

.. code-block:: python

    >>> analysis_gt.head(3)


+----+--------------+--------------------+
|    |   identifier |   work_home_actual |
+====+==============+====================+
|  0 |        50000 |                  1 |
+----+--------------+--------------------+
|  1 |        50001 |                  1 |
+----+--------------+--------------------+
|  2 |        50002 |                  1 |
+----+--------------+--------------------+

.. code-block:: python

    >>> from sklearn.metrics import roc_auc_score
    >>> import matplotlib.pyplot as plt
    >>> # merge gt to analysis
    >>> analysis_full = pd.merge(analysis, analysis_gt, on = 'identifier')
    >>> df_all = pd.concat([reference, analysis_full]).reset_index(drop=True)
    >>> target_col = 'work_home_actual'
    >>> pred_score_col = 'y_pred_proba'
    >>> actual_performance = []
    >>> for idx in est_perf.data.index:
    >>>     start_index, end_index = est_perf.data.loc[idx, 'start_index'], est_perf.data.loc[idx, 'end_index']
    >>>     sub = df_all.loc[start_index:end_index]
    >>>     actual_perf = roc_auc_score(sub[target_col], sub[pred_score_col])
    >>>     est_perf.data.loc[idx, 'actual_roc_auc'] = actual_perf
    >>> # plot
    >>> est_perf.data[['estimated_roc_auc', 'actual_roc_auc']].plot()
    >>> plt.xlabel('chunk')
    >>> plt.ylabel('ROC AUC')
    >>> plt.show()


.. image:: ../_static/guide-performance_estimation_tmp.svg

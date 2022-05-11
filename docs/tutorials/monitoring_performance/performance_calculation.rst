.. _performance-calculation:

===============================
Monitoring Realized Performance
===============================

Why Monitoring Realized Performance
===================================

The realized performance of a machine learning model is the key driver for the busines results it will bring.
Therefore it is crucial to monitor it and make corrective actions when the results are not satisfactory.

This guide shows how to use NannyML to calculate the :term:`Realized Performance` of a model in order to monitor it.
:term:`Target` values need to be available in both the reference and analysis data.
All monitoring metrics available by NannyML for monitoring will be showed.

.. note::
    The performance monitoring process requires no missing values in the target data on the reference dataset. However
    the analysis data can contain missing values. In this case the entries with missing values will be ignored when
    calculating the performance results. If there are so many missing values that the available data are below the
    :ref:`minimum-chunk-size` then the performance results are ommited from the resulting visualizations because they are
    too noisy, due to low sample size, to be reliable.

Binary Classification
=====================


Just The Code
-------------

If you just want the code to experiment yourself, here you go:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_binary_classification_dataset()
    >>> display(reference.head(3))

    >>> data = pd.concat([reference, analysis.set_index('identifier').join(analysis_target.set_index('identifier'), on='identifier', rsuffix='_r')], ignore_index=True).reset_index(drop=True)
    >>> display(data.loc[data['partition'] == 'analysis'].head(3))

    >>> metadata = nml.extract_metadata(reference, model_type=nml.ModelType.CLASSIFICATION_BINARY, exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(metadata.is_complete())

    >>> performance_calculator = nml.PerformanceCalculator(
    ...     model_metadata=metadata,
    ...     # use NannyML to tell us what metrics are supported
    ...     metrics=nml.performance_estimation.confidence_based.results.SUPPORTED_METRIC_VALUES,
    ...     chunk_size=5000
    ... ).fit(reference_data=reference)

    >>> realized_performance = performance_calculator.calculate(data)

    >>> display(ealized_performance.data.head(3))

    >>> for metric in performance_calculator.metrics:
    ...     figure = realized_performance.plot(kind='performance', metric=metric)
    ...     figure.show()



Walkthrough on Monitoring Realized Performance
----------------------------------------------


Prepare the data
~~~~~~~~~~~~~~~~

For simplicity the guide is based on a synthetic dataset where the monitored model predicts
whether an employee will work from home.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_binary_classification_dataset()
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


The realized performance will be calculated on the combination of both reference and analysis data. The analysis target
values are joined on the analysis frame by the ``identifier`` column.

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


The ``reference`` and ``analysis`` dataframes correspond to ``reference`` and ``analysis`` periods of
the monitored data. To understand what they are read :ref:`data periods<data-drift-periods>`. The
``analysis_gt`` dataframe contains the target results of the analysis period and we will not be using
it during Performance Estimation.

One of the first steps in using NannyML is providing metadata information about the model we are monitoring.
Some information is infered automatically and we provide the rest.

.. code-block:: python

    >>> metadata = nml.extract_metadata(reference, model_type=nml.ModelType.CLASSIFICATION_BINARY, exclude_columns=['identifier'])
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(metadata.is_complete())
    (True, [])


We see that the metadata are complete. Full information on how to extract metadata can be found in the :ref:`providing metadata guide<import-data>`.

Fit calculator and calculate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the next step a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using the previously
extracted :class:`~nannyml.metadata.base.ModelMetadata`, a list of metrics and an optional :ref:`chunking<chunking>` specification.
The list of metrics specifies which metrics should be calculated. For an overview of all metrics,
check the :mod:`~nannyml.performance_calculation.metrics` module.

The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is then fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

.. code-block:: python

    >>> performance_calculator = nml.PerformanceCalculator(
    ...     model_metadata=metadata,
    ...     # use NannyML to tell us what metrics are supported
    ...     metrics=nml.performance_estimation.confidence_based.results.SUPPORTED_METRIC_VALUES,
    ...     chunk_size=5000
    ... ).fit(reference_data=reference)

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can be used to calculate
realized performance metrics on data for which target values are available.
This is typically done on all data for which target values are available. In our example this
includes both reference and analysis.

.. code-block:: python

    >>> realized_performance = performance_calculator.calculate(data)


View the results
~~~~~~~~~~~~~~~~

NannyML can output a dataframe that contains all the results:

.. code-block:: python

    >>> display(realized_performance.data.head(3))

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+------------------------------------------+------------+-------------+------------------------------------------+-------------------+----------+-----------------------------------------+----------------+---------------+------------------------------------------+---------------------+------------+------------------------------------------+------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   targets_missing_rate |   roc_auc | roc_auc_thresholds                      | roc_auc_alert   |       f1 | f1_thresholds                            | f1_alert   |   precision | precision_thresholds                     | precision_alert   |   recall | recall_thresholds                       | recall_alert   |   specificity | specificity_thresholds                   | specificity_alert   |   accuracy | accuracy_thresholds                      | accuracy_alert   |
+====+===============+===============+=============+=====================+=====================+=============+========================+===========+=========================================+=================+==========+==========================================+============+=============+==========================================+===================+==========+=========================================+================+===============+==========================================+=====================+============+==========================================+==================+
|  0 | [0:4999]      |             0 |        4999 | 2014-05-09 22:27:20 | 2014-09-09 08:18:27 | reference   |                      0 |  0.976253 | (0.963316535948479, 0.9786597341713761) | False           | 0.953803 | (0.9350467474218009, 0.9610943245280688) | False      |    0.951308 | (0.9247411224999635, 0.9611314708654666) | False             | 0.956311 | (0.940831383455992, 0.9657258748427315) | False          |      0.952136 | (0.9247408281519457, 0.9601131753790443) | False               |     0.9542 | (0.9350787461431096, 0.9606012538568904) | False            |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+------------------------------------------+------------+-------------+------------------------------------------+-------------------+----------+-----------------------------------------+----------------+---------------+------------------------------------------+---------------------+------------+------------------------------------------+------------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2014-09-09 09:13:35 | 2015-01-09 00:02:51 | reference   |                      0 |  0.969045 | (0.963316535948479, 0.9786597341713761) | False           | 0.940963 | (0.9350467474218009, 0.9610943245280688) | False      |    0.934748 | (0.9247411224999635, 0.9611314708654666) | False             | 0.947262 | (0.940831383455992, 0.9657258748427315) | False          |      0.9357   | (0.9247408281519457, 0.9601131753790443) | False               |     0.9414 | (0.9350787461431096, 0.9606012538568904) | False            |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+------------------------------------------+------------+-------------+------------------------------------------+-------------------+----------+-----------------------------------------+----------------+---------------+------------------------------------------+---------------------+------------+------------------------------------------+------------------+
|  2 | [10000:14999] |         10000 |       14999 | 2015-01-09 00:04:43 | 2015-05-09 15:54:26 | reference   |                      0 |  0.971742 | (0.963316535948479, 0.9786597341713761) | False           | 0.954483 | (0.9350467474218009, 0.9610943245280688) | False      |    0.949804 | (0.9247411224999635, 0.9611314708654666) | False             | 0.959208 | (0.940831383455992, 0.9657258748427315) | False          |      0.948283 | (0.9247408281519457, 0.9601131753790443) | False               |     0.9538 | (0.9350787461431096, 0.9606012538568904) | False            |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+------------------------------------------+------------+-------------+------------------------------------------+-------------------+----------+-----------------------------------------+----------------+---------------+------------------------------------------+---------------------+------------+------------------------------------------+------------------+


Apart from chunking and chunk and partition-related data, the results data have the a set of columns for each
calculated metric. When taking ``roc_auc`` as an example:

 - ``roc_auc`` - The value of the metric for a specific chunk.
 - ``roc_auc_thresholds`` - A tuple containing the lower and upper thresholds. Crossing them will raise an alert on significant
   metric change. The thresholds are calculated based on the realized performance metric of the monitored model on chunks in
   the ``reference`` period. The thresholds are 3 standard deviations away from the mean performance calculated on
   ``reference`` chunks.
 - ``roc_auc_alert`` - Flag indicating potentially significant performance change. ``True`` if realized performance crosses
   upper or lower threshold.


The results can be plotted for vizual inspection:

.. code-block:: python

    >>> for metric in performance_calculator.metrics:
    ...     figure = realized_performance.plot(kind='performance', metric=metric)
    ...     figure.show()

.. image:: /_static/tutorial-perf-guide-Accuracy.svg

.. image:: /_static/tutorial-perf-guide-F1.svg

.. image:: /_static/tutorial-perf-guide-Precision.svg

.. image:: /_static/tutorial-perf-guide-ROC_AUC.svg

.. image:: /_static/tutorial-perf-guide-Recall.svg

.. image:: /_static/tutorial-perf-guide-Specificity.svg


Multiclass Classification
=========================


Just The Code
-------------

If you just want the code to experiment yourself, here you go:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_multiclass_classification_dataset()
    >>> display(reference.head(3))

    >>> data = pd.concat([
    ...     reference,
    ...     analysis.set_index('identifier').join(analysis_gt.set_index('identifier'), on='identifier', rsuffix='_r')
    >>> ], ignore_index=True).reset_index(drop=True)
    >>> display(data.loc[data['partition'] == 'analysis'].head(3))

    >>> metadata = nml.extract_metadata(
    reference,
    ...     model_name='credit_card_segment',
    ...     model_type=nml.ModelType.CLASSIFICATION_MULTICLASS,
    ...     exclude_columns=['identifier']
    >>> )
    >>> metadata.target_column_name = 'y_true'
    >>> display(metadata.is_complete())

    >>> performance_calculator = nml.PerformanceCalculator(
    ...     model_metadata=metadata,
    ...     metrics=['roc_auc', 'f1'],
    ...     chunk_size=6000
    >>> ).fit(reference_data=reference)

    >>> realized_performance = performance_calculator.calculate(data)

    >>> display(realized_performance.data.head(3))

    >>> for metric in performance_calculator.metrics:
    ...     figure = realized_performance.plot(kind='performance', metric=metric)
    ...     figure.show()



Walkthrough on Monitoring Realized Performance
----------------------------------------------


Prepare the data
~~~~~~~~~~~~~~~~

For simplicity the guide is based on a synthetic dataset where the monitored model predicts
which type of credit card product new customers should be assigned to.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_multiclass_classification_dataset()
    >>> display(reference.head(3))

+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|    | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | partition   |   identifier | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred       | y_true        |
+====+===============+========================+==========================+===============+=======================+=================+===============+=============+==============+=====================+=============================+================================+==============================+==============+===============+
|  0 | Partner3      |               1.80823  |                      350 | web           |                   309 |           15000 | True          | reference   |        60000 | 2020-05-02 02:01:30 |                        0.97 |                           0.03 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  1 | Partner2      |               4.38257  |                      500 | mobile        |                   418 |           23000 | True          | reference   |        60001 | 2020-05-02 02:03:33 |                        0.87 |                           0.13 |                         0    | prepaid_card | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+
|  2 | Partner2      |              -0.787575 |                      400 | web           |                   507 |           24000 | False         | reference   |        60002 | 2020-05-02 02:04:49 |                        0.47 |                           0.35 |                         0.18 | prepaid_card | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+--------------+---------------+


The realized performance will be calculated on the combination of both reference and analysis data. The analysis target
values are joined on the analysis frame by the ``identifier`` column.

.. code-block:: python

    >>> data = pd.concat([
    ...     reference,
    ...     analysis.set_index('identifier').join(analysis_gt.set_index('identifier'), on='identifier', rsuffix='_r')
    >>> ], ignore_index=True).reset_index(drop=True)
    >>> display(data.loc[data['partition'] == 'analysis'].head(3))

+-------+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+-----------------+
|       | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | partition   |   identifier | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred          | y_true          |
+=======+===============+========================+==========================+===============+=======================+=================+===============+=============+==============+=====================+=============================+================================+==============================+=================+=================+
| 60000 | Organic       |              -1.64376  |                      300 | store         |                   439 |           15000 | False         | analysis    |          nan | 2020-09-01 03:10:01 |                        0.39 |                           0.35 |                         0.26 | prepaid_card    | upmarket_card   |
+-------+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+-----------------+
| 60001 | Partner2      |              -0.148435 |                      450 | store         |                   565 |           18000 | False         | analysis    |          nan | 2020-09-01 03:10:53 |                        0.72 |                           0.01 |                         0.27 | prepaid_card    | prepaid_card    |
+-------+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+-----------------+
| 60002 | Partner1      |              -2.28461  |                      600 | mobile        |                   691 |           28000 | False         | analysis    |          nan | 2020-09-01 03:11:39 |                        0.03 |                           0.75 |                         0.22 | highstreet_card | highstreet_card |
+-------+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-------------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+-----------------+


The ``reference`` and ``analysis`` dataframes correspond to ``reference`` and ``analysis`` periods of
the monitored data. To understand what they are read :ref:`data periods<data-drift-periods>`. The
``analysis_gt`` dataframe contains the target results of the analysis period and we will not be using
it during Performance Estimation.

One of the first steps in using NannyML is providing metadata information about the model we are monitoring.
Some information is infered automatically and we provide the rest.

.. code-block:: python

    >>> metadata = nml.extract_metadata(
    reference,
    ...     model_name='credit_card_segment',
    ...     model_type=nml.ModelType.CLASSIFICATION_MULTICLASS,
    ...     exclude_columns=['identifier']
    >>> )
    >>> metadata.target_column_name = 'y_true'
    >>> display(metadata.is_complete())
    (True, [])


We see that the metadata are complete. Full information on how to extract metadata can be found in the :ref:`providing metadata guide<import-data>`.

Fit calculator and calculate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the next step a :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is created using the previously
extracted :class:`~nannyml.metadata.base.ModelMetadata`, a list of metrics and an optional :ref:`chunking<chunking>` specification.
The list of metrics specifies which metrics should be calculated. For an overview of all metrics,
check the :mod:`~nannyml.performance_calculation.metrics` module.

The new :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` is then fitted using the
:meth:`~nannyml.performance_calculation.calculator.PerformanceCalculator.fit` method on the ``reference`` data.

.. code-block:: python

    >>> performance_calculator = nml.PerformanceCalculator(
    ...     model_metadata=metadata,
    ...     metrics=['roc_auc', 'f1'],
    ...     chunk_size=6000
    >>> ).fit(reference_data=reference)

The fitted :class:`~nannyml.performance_calculation.calculator.PerformanceCalculator` can be used to calculate
realized performance metrics on data for which target values are available.
This is typically done on all data for which target values are available. In our example this
includes both reference and analysis.

.. code-block:: python

    >>> realized_performance = performance_calculator.calculate(data)


View the results
~~~~~~~~~~~~~~~~

NannyML can output a dataframe that contains all the results:

.. code-block:: python

    >>> display(realized_performance.data.head(3))

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+-----------------------------------------+------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   targets_missing_rate |   roc_auc | roc_auc_thresholds                      | roc_auc_alert   |       f1 | f1_thresholds                           | f1_alert   |
+====+===============+===============+=============+=====================+=====================+=============+========================+===========+=========================================+=================+==========+=========================================+============+
|  0 | [0:5999]      |             0 |        5999 | 2020-05-02 02:01:30 | 2020-05-14 12:25:35 | reference   |                      0 |  0.90476  | (0.900902260737325, 0.9135156728918074) | False           | 0.750532 | (0.741253919065521, 0.7649438592270994) | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+-----------------------------------------+------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2020-05-14 12:29:25 | 2020-05-26 18:27:42 | reference   |                      0 |  0.905917 | (0.900902260737325, 0.9135156728918074) | False           | 0.751148 | (0.741253919065521, 0.7649438592270994) | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+-----------------------------------------+------------+
|  2 | [12000:17999] |         12000 |       17999 | 2020-05-26 18:31:06 | 2020-06-07 19:55:45 | reference   |                      0 |  0.909329 | (0.900902260737325, 0.9135156728918074) | False           | 0.75714  | (0.741253919065521, 0.7649438592270994) | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+------------------------+-----------+-----------------------------------------+-----------------+----------+-----------------------------------------+------------+


Apart from chunking and chunk and partition-related data, the results data have the a set of columns for each
calculated metric. When taking ``roc_auc`` as an example:

 - ``roc_auc`` - The value of the metric for a specific chunk.
 - ``roc_auc_thresholds`` - A tuple containing the lower and upper thresholds. Crossing them will raise an alert on significant
   metric change. The thresholds are calculated based on the realized performance metric of the monitored model on chunks in
   the ``reference`` period. The thresholds are 3 standard deviations away from the mean performance calculated on
   ``reference`` chunks.
 - ``roc_auc_alert`` - Flag indicating potentially significant performance change. ``True`` if realized performance crosses
   upper or lower threshold.


The results can be plotted for vizual inspection:

.. code-block:: python

    >>> for metric in performance_calculator.metrics:
    ...     figure = realized_performance.plot(kind='performance', metric=metric)
    ...     figure.show()

.. image:: /_static/tutorial-perf-guide-mc-F1.svg

.. image:: /_static/tutorial-perf-guide-mc-ROC_AUC.svg



Insights and Follow Ups
=======================

After reviewing the performance calculation results we have to decide if further investigation is needed.
The :ref:`Data Drift<data-drift>` functionality can help here.

If needed further investigation can be performed as to wheher the model's performance is satisfactory
according to business requirements. This is an ad-hoc investigation that is not covered by NannyML.

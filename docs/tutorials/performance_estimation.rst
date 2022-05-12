.. _performance-estimation:

============================================
Estimating Performance
============================================

Why Perform Performance Estimation
============================================

NannyML allows to estimate the performance of a classification model when :term:`targets<Target>` are absent.
This can be very helpful in situations where there is a significant delay
in when targets become available but any changes in the model's performance would have
a significant impact on business results. This tutorial explains how to use NannyML to estimate performance of binary
and multiclass classification models in the absence of ground truth. To find out how it works check
:ref:`the explanation of CBPE<performance-estimation-deep-dive>`.

Binary Classification
=====================

Just The Code
----------------

If you just want the code to experiment yourself, here you go:

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> reference, analysis, analysis_gt = nml.datasets.load_synthetic_binary_classification_dataset()
    >>> display(reference.head(3))

    >>> metadata = nml.extract_metadata(
    ...     reference,
    ...     model_type=nml.ModelType.CLASSIFICATION_BINARY,
    ...     exclude_columns=['identifier']
    ... )
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(metadata.is_complete())

    >>> cbpe = nml.CBPE(
    ...     model_metadata=metadata,
    ...     chunk_size=5000,
    ...     metrics=['roc_auc', 'f1']
    ... )
    >>> cbpe.fit(reference_data=reference)

    >>> est_perf_analysis = cbpe.estimate(analysis)
    >>> display(est_perf_analysis.data.head(3))
    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_analysis.plot(kind='performance', metric=metric)
    ...     figure.show()

    >>> est_perf_with_ref = cbpe.estimate(pd.concat([reference, analysis], ignore_index=True))
    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_with_ref.plot(kind='performance', metric=metric)
    ...     figure.show()



Walkthrough on Performance Estimation for Binary Classification
----------------------------------------------------------------

Prepare the data
^^^^^^^^^^^^^^^^^^

For simplicity the guide is based on a synthetic dataset where the monitored model predicts
whether an employee will work from home. Read more :ref:`here<dataset-synthetic-binary>`.


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


The ``reference`` and ``analysis`` dataframes correspond to ``reference`` and ``analysis`` periods of
the monitored data. To understand what they are read :ref:`data periods<data-drift-periods>`. The
``analysis_gt`` dataframe contains the target (ground truth) results of the analysis period and will not be used
during Performance Estimation.

One of the first steps in using NannyML is providing metadata information about the model that is monitored.
Some information is inferred automatically and the rest should be provided.

.. code-block:: python

    >>> metadata = nml.extract_metadata(
    ...     reference,
    ...     model_type=nml.ModelType.CLASSIFICATION_BINARY,
    ...     exclude_columns=['identifier']
    ... )
    >>> metadata.target_column_name = 'work_home_actual'
    >>> display(metadata.is_complete())
    (True, [])


We see that the metadata are complete. Full information on how the data should be prepared can be found in the guide on :ref:`importing data<import-data>`.

Create and fit the estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the next step Confidence-based Performance Estimation
(:class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`)
estimator is created using the previously
extracted :class:`~nannyml.metadata.base.ModelMetadata`, a list of metrics and an optional
:ref:`chunking<chunking>` specification. The list of metrics specifies the metrics
for which the performance of the monitored model will be estimated. The following metrics are currently supported:

- ``roc_auc``
- ``f1``
- ``precision``
- ``recall``
- ``specificity``
- ``accuracy``

For more information about :term:`chunking<Data Chunk>` you can check the :ref:`setting up page<chunking>` and :ref:`advanced guide<chunk-data>`.

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
estimator is then fitted using the
:meth:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE.fit` method on the ``reference`` data.




.. code-block:: python

    >>> cbpe = nml.CBPE(
    ...     model_metadata=metadata,
    ...     chunk_size=5000,
    ...     metrics=['roc_auc', 'f1']
    ... )
    >>> cbpe.fit(reference_data=reference)

The fitted ``cbpe`` can be used to estimate performance on other data, for which performance cannot be calculated.
Typically, this would be used on the latest production data where target is missing. In our example this is
the ``analysis`` data.

.. code-block:: python

    >>> est_perf_analysis = cbpe.estimate(analysis)

However, it can be also used on combined ``reference`` and ``analysis`` data. This might help to build better
understanding of the monitored model performance changes on analysis data as it can be then shown in the context of
changes of calculated performance on the reference period.

.. code-block:: python

    >>> est_perf_with_ref = cbpe.estimate(pd.concat([reference, analysis], ignore_index=True))

To find out how CBPE estimates performance, read about :ref:`Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.

View the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NannyML can output a dataframe that contains all the results. Let's have a look at the results for analysis period
only:

.. code-block:: python

    >>> display(est_perf_analysis.data.head(3))

+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   confidence_roc_auc |   realized_roc_auc |   estimated_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |   confidence_f1 |   realized_f1 |   estimated_f1 |   upper_threshold_f1 |   lower_threshold_f1 | alert_f1   |
+====+===============+===============+=============+=====================+=====================+=============+======================+====================+=====================+===========================+===========================+=================+=================+===============+================+======================+======================+============+
|  0 | [0:4999]      |             0 |        4999 | 2017-08-31 04:20:00 | 2018-01-02 00:45:44 | analysis    |           0.00035752 |                nan |            0.968631 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.948555 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2018-01-02 01:13:11 | 2018-05-01 13:10:10 | analysis    |           0.00035752 |                nan |            0.969044 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.946578 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|  2 | [10000:14999] |         10000 |       14999 | 2018-05-01 14:25:25 | 2018-09-01 15:40:40 | analysis    |           0.00035752 |                nan |            0.969444 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.948807 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+


.. _performance-estimation-thresholds:

Apart from chunk and partition-related data, the results data have the following columns for each metric
that was estimated:

 - ``estimated_<metric>`` - the estimate of selected ``metric`` for a specific chunk,
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


These results can be also plotted:

.. code-block:: python

    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_analysis.plot(kind='performance', metric=metric)
    ...     figure.show()


.. image:: ../_static/tutorial-perf-est-guide-analysis-roc_auc.svg

.. image:: ../_static/tutorial-perf-est-guide-analysis-f1.svg

The purple dashed step plot shows the estimated performance in each chunk of analysis period. Thick squared point
marker indicates the middle of this period. Solid, low-saturated purple line *behind* indicates the confidence band.
Red horizontal
dashed lines show upper and lower thresholds. If the estimated performance crosses upper or lower threshold and alert
is raised
which is indicated with red, low-saturated background in the whole width of the relevant chunk. This is additionally
indicated by red point marker in the middle of the chunk. Description of tabular results above explains how the
confidence bands and thresholds are calculated. Additional information is shown in the hover (these are
interactive plots).

To get a better context let's additionally plot estimation of performance on *analysis* data together with calculated
performance on reference period (where the target was available).

.. code-block:: python

    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_with_ref.plot(kind='performance', metric=metric)
    ...     figure.show()


.. image:: ../_static/tutorial-perf-est-guide-with-ref-roc_auc.svg

.. image:: ../_static/tutorial-perf-est-guide-with-ref-f1.svg

The right hand side of the plot is exactly the same as previously as it shows the estimated performance for the
analysis period. The purple dashed vertical line splits the reference and analysis periods. On the left hand side of
the line, the actual model performance (not estimation!) is plotted with solid light blue line. This facilitates
interpretation of the estimation on reference period as it helps to build expectations on variability of the
performance.


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

    >>> metadata = nml.extract_metadata(
    ...     reference,
    ...     model_name='credit_card_segment',
    ...     model_type=nml.ModelType.CLASSIFICATION_MULTICLASS,
    ...     exclude_columns=['identifier']
    >>> )
    >>> metadata.target_column_name = 'y_true'
    >>> display(metadata.is_complete())

    >>> cbpe = nml.CBPE(
    ...     model_metadata=metadata,
    ...     chunk_size=6000,
    ...     metrics=['roc_auc', 'f1']
    >>> )
    >>> cbpe = cbpe.fit(reference_data=reference)

    >>> est_perf_analysis = cbpe.estimate(analysis)
    >>> display(est_perf_analysis.data.head(3))
    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_analysis.plot(kind='performance', metric=metric)
    ...     figure.show()

    >>> est_perf_with_ref = cbpe.estimate(pd.concat([reference, analysis], ignore_index=True))
    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_with_ref.plot(kind='performance', metric=metric)
    ...     figure.show()


Walkthrough on Performance Estimation for Multiclass Classification
-------------------------------------------------------------------

Prepare the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simplicity the guide is based on a synthetic dataset where the monitored model predicts
which type of credit card product new customers should be assigned to. Read more :ref:`here<dataset-synthetic-multiclass>`.

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


The ``reference`` and ``analysis`` dataframes correspond to ``reference`` and ``analysis`` periods of
the monitored data. To understand what they are read :ref:`data periods<data-drift-periods>`. The
``analysis_gt`` dataframe contains the target results of the analysis period and will not be used
during Performance Estimation.

One of the first steps in using NannyML is providing metadata information about the model we are monitoring.
Some information is inferred automatically and we provide the rest.

.. code-block:: python

    >>> metadata = nml.extract_metadata(
    ...     reference,
    ...     model_name='credit_card_segment',
    ...     model_type=nml.ModelType.CLASSIFICATION_MULTICLASS,
    ...     exclude_columns=['identifier']
    >>> )
    >>> metadata.target_column_name = 'y_true'
    >>> display(metadata.is_complete())
    (True, [])

The difference between binary and multiclass classification is that metadata for multiclass classification should
contain mapping between classes (i.e. values that are in target and prediction columns) to column names with predicted
probabilities that correspond to these classes. This mapping can be specified or it can be automatically extracted
if predicted probability column names meet specific requirements as in the example presented. Read more in the
:ref:`Setting Up, Providing
Metadata<import-data>` section.

Create and fit the estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the next step Confidence-based Performance Estimation
(:class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`)
estimator is created using the previously
extracted :class:`~nannyml.metadata.base.ModelMetadata`, a list of metrics and an optional
:ref:`chunking<chunking>` specification. The list of metrics specifies the metrics
for which the performance of the monitored model will be estimated. The following metrics are currently supported:

- ``roc_auc``
- ``f1``
- ``precision``
- ``recall``
- ``specificity``
- ``accuracy``

For more information about :term:`chunking<Data Chunk>` you can check the :ref:`setting up page<chunking>` and :ref:`advanced guide<chunk-data>`.

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`
estimator is then fitted using the
:meth:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE.fit` method on the ``reference`` data.

.. code-block:: python

    >>> cbpe = nml.CBPE(
    ...     model_metadata=metadata,
    ...     chunk_size=6000,
    ...     metrics=['roc_auc', 'f1']
    >>> )
    >>> cbpe = cbpe.fit(reference_data=reference)

The fitted ``cbpe`` can be used to estimate performance on other data, for which performance cannot be calculated.
Typically, this would be used on the latest production data where target is missing. In our example this is
the ``analysis`` data.

.. code-block:: python

    >>> est_perf_analysis = cbpe.estimate(analysis)

However, it can be also used on combined ``reference`` and ``analysis`` data. This might help to build better
understanding of the monitored model performance changes on analysis data as it can be then shown in the context of
changes of calculated performance on the reference period.

.. code-block:: python

    >>> est_perf_with_ref = cbpe.estimate(pd.concat([reference, analysis], ignore_index=True))

To find out how CBPE estimates performance, read about :ref:`Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.

View the results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NannyML can output a dataframe that contains all the results. Let's have a look at the results for analysis period
only:

.. code-block:: python

    >>> display(est_perf_analysis.data.head(3))


+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | partition   |   confidence_roc_auc |   realized_roc_auc |   estimated_roc_auc |   upper_threshold_roc_auc |   lower_threshold_roc_auc | alert_roc_auc   |   confidence_f1 |   realized_f1 |   estimated_f1 |   upper_threshold_f1 |   lower_threshold_f1 | alert_f1   |
+====+===============+===============+=============+=====================+=====================+=============+======================+====================+=====================+===========================+===========================+=================+=================+===============+================+======================+======================+============+
|  0 | [0:4999]      |             0 |        4999 | 2017-08-31 04:20:00 | 2018-01-02 00:45:44 | analysis    |           0.00035752 |                nan |            0.968631 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.948555 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|  1 | [5000:9999]   |          5000 |        9999 | 2018-01-02 01:13:11 | 2018-05-01 13:10:10 | analysis    |           0.00035752 |                nan |            0.969044 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.946578 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+
|  2 | [10000:14999] |         10000 |       14999 | 2018-05-01 14:25:25 | 2018-09-01 15:40:40 | analysis    |           0.00035752 |                nan |            0.969444 |                  0.963317 |                   0.97866 | False           |     0.000951002 |           nan |       0.948807 |             0.935047 |             0.961094 | False      |
+----+---------------+---------------+-------------+---------------------+---------------------+-------------+----------------------+--------------------+---------------------+---------------------------+---------------------------+-----------------+-----------------+---------------+----------------+----------------------+----------------------+------------+



Apart from chunking and chunk and partition-related data, the results data have the following columns for each metric
that was estimated:

 - ``estimated_<metric>`` - the estimate of a metric for a specific chunk,
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


These results can be also plotted:

.. code-block:: python

    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_analysis.plot(kind='performance', metric=metric)
    ...     figure.show()


.. image:: ../_static/tutorial-perf-est-mc-guide-analysis-roc_auc.svg

.. image:: ../_static/tutorial-perf-est-mc-guide-analysis-f1.svg

The purple dashed step plot shows the estimated performance in each chunk of analysis period. Thick squared point
marker indicates the middle of this period. Solid, low-saturated purple line *behind* indicates the confidence band.
Red horizontal
dashed lines show upper and lower thresholds. If the estimated performance crosses upper or lower threshold and alert
is raised
which is indicated with red, low-saturated background in the whole width of the relevant chunk. This is additionally
indicated by red point marker in the middle of the chunk. Description of tabular results above explains how the
confidence bands and thresholds were calculated. Additional information is shown in the hover (these are
interactive plots).

To get a better context let's additionally plot estimation of performance on *analysis* data together with calculated
performance on reference period (where the target was available).

.. code-block:: python

    >>> for metric in cbpe.metrics:
    ...     figure = est_perf_with_ref.plot(kind='performance', metric=metric)
    ...     figure.show()


.. image:: ../_static/tutorial-perf-est-mc-guide-with-ref-roc_auc.svg

.. image:: ../_static/tutorial-perf-est-mc-guide-with-ref-f1.svg

The right hand side of the plot is exactly the same as previously as it shows the estimated performance for the
analysis period. The purple dashed vertical line splits the reference and analysis periods. On the left hand side of
the line, the actual model performance (not estimation!) is plotted with solid light blue line. This facilitates
interpretation of the estimation on reference period as it helps to build expectations on variability of the
performance.


Insights and Follow Ups
==========================

After reviewing the performance estimation results we have to decide if further investigation is needed.
The :ref:`Data Drift<data-drift>` functionality can help here.

This may help to indicate which of our population characteristics have
changed and how. This will sometimes lead to investigating *why* they changed which is not covered by NannyML.

When the target results become available they can be compared with the estimated results as
demonstrated :ref:`here<compare_estimated_and_realized_performance>`. You can learn more
about the Confidence Based Performance Estimation and its limitations in the
:ref:`How it Works page<performance-estimation-deep-dive>`

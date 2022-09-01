.. _drift_detection_for_multiclass_classification_model_outputs:

============================================================
Drift Detection for Multiclass Classification  Model Outputs
============================================================

Why Perform Drift Detection for Model Outputs
---------------------------------------------

The distribution of the model outputs tells us the model's evaluation of how likely
the predicted outcome is to happen across the model's population.
If the model's population changes, then our populations' actions will be different.
The difference in actions is very important to know as soon as possible because
they directly affect the business results from operating a machine learning model.


Just The Code
------------------------------------

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>>
    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]
    >>>
    >>> display(reference_df.head())
    >>>
    >>> calc = nml.StatisticalOutputDriftCalculator(
    ...     y_pred='y_pred',
    ...     y_pred_proba={
    ...         'prepaid_card': 'y_pred_proba_prepaid_card',
    ...         'upmarket_card': 'y_pred_proba_upmarket_card',
    ...         'highstreet_card': 'y_pred_proba_highstreet_card'
    ...     },
    ...     timestamp_column_name='timestamp',
    ...     problem_type='classification_multiclass')
    >>>
    >>> calc.fit(reference_df)
    >>>
    >>> results = calc.calculate(analysis_df)
    >>>
    >>> display(results.data)
    >>>
    >>> figure = results.plot(kind='prediction_drift', plot_reference=True)
    >>> figure.show()
    >>>
    >>> figure = results.plot(kind='prediction_distribution', plot_reference=True)
    >>> figure.show()
    >>>
    >>> for label in calc.y_pred_proba.keys():
    ...     figure = results.plot(kind='score_drift', class_label=label, plot_reference=True)
    ...     figure.show()
    >>>
    >>> for label in calc.y_pred_proba.keys():
    ...     figure = results.plot(kind='score_distribution', class_label=label, plot_reference=True)
    ...     figure.show()

Walkthrough
------------------------------------------------

NannyML detects data drift for :term:`Model Outputs` using the
:ref:`Univariate Drift Detection methodology<univariate_drift_detection_walkthrough>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference and analysis dataframes. This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

.. code-block:: python

    >>> import nannyml as nml
    >>> import pandas as pd
    >>> from IPython.display import display
    >>>
    >>> reference_df = nml.load_synthetic_multiclass_classification_dataset()[0]
    >>> analysis_df = nml.load_synthetic_multiclass_classification_dataset()[1]
    >>>
    >>> display(reference_df.head())

+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+---------------+
|    | acq_channel   |   app_behavioral_score |   requested_credit_limit | app_channel   |   credit_bureau_score |   stated_income | is_customer   | period    |   identifier | timestamp           |   y_pred_proba_prepaid_card |   y_pred_proba_highstreet_card |   y_pred_proba_upmarket_card | y_pred          | y_true        |
+====+===============+========================+==========================+===============+=======================+=================+===============+===========+==============+=====================+=============================+================================+==============================+=================+===============+
|  0 | Partner3      |               1.80823  |                      350 | web           |                   309 |           15000 | True          | reference |        60000 | 2020-05-02 02:01:30 |                        0.97 |                           0.03 |                         0    | prepaid_card    | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+---------------+
|  1 | Partner2      |               4.38257  |                      500 | mobile        |                   418 |           23000 | True          | reference |        60001 | 2020-05-02 02:03:33 |                        0.87 |                           0.13 |                         0    | prepaid_card    | prepaid_card  |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+---------------+
|  2 | Partner2      |              -0.787575 |                      400 | web           |                   507 |           24000 | False         | reference |        60002 | 2020-05-02 02:04:49 |                        0.47 |                           0.35 |                         0.18 | prepaid_card    | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+---------------+
|  3 | Partner3      |              -2.13177  |                      300 | mobile        |                   324 |           38000 | False         | reference |        60003 | 2020-05-02 02:07:59 |                        0.26 |                           0.5  |                         0.24 | highstreet_card | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+---------------+
|  4 | Partner3      |              -1.36294  |                      450 | mobile        |                   736 |           38000 | True          | reference |        60004 | 2020-05-02 02:20:19 |                        0.03 |                           0.04 |                         0.93 | upmarket_card   | upmarket_card |
+----+---------------+------------------------+--------------------------+---------------+-----------------------+-----------------+---------------+-----------+--------------+---------------------+-----------------------------+--------------------------------+------------------------------+-----------------+---------------+

The :class:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator`
class implements the functionality needed for drift detection in model outputs. First, the class is instantiated with appropriate parameters.
To check the model outputs for data drift, we only need to pass in the column header of the outputs as `y_pred` and `y_pred_proba`.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.fit` method
is called on the reference data, so that the data baseline can be established.

Then the :meth:`~nannyml.drift.model_inputs.univariate.statistical.calculator.StatisticalOutputDriftCalculator.calculate` method
calculates the drift results on the data provided. An example using it can be seen below.

.. code-block:: python

    >>> calc = nml.StatisticalOutputDriftCalculator(
    ...     y_pred='y_pred',
    ...     y_pred_proba={
    ...         'prepaid_card': 'y_pred_proba_prepaid_card',
    ...         'upmarket_card': 'y_pred_proba_upmarket_card',
    ...         'highstreet_card': 'y_pred_proba_highstreet_card'
    ...     },
    ...     timestamp_column_name='timestamp',
    ...     problem_type='classification_multiclass')
    >>>
    >>> calc.fit(reference_df)

We can then display the results in a table, or as plots.

.. code-block:: python

    >>> display(results.data)

+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|    | key           |   start_index |   end_index | start_date          | end_date            | period   |   y_pred_chi2 |   y_pred_p_value | y_pred_alert   |   y_pred_threshold |   y_pred_proba_prepaid_card_dstat |   y_pred_proba_prepaid_card_p_value | y_pred_proba_prepaid_card_alert   |   y_pred_proba_prepaid_card_threshold |   y_pred_proba_upmarket_card_dstat |   y_pred_proba_upmarket_card_p_value | y_pred_proba_upmarket_card_alert   |   y_pred_proba_upmarket_card_threshold |   y_pred_proba_highstreet_card_dstat |   y_pred_proba_highstreet_card_p_value | y_pred_proba_highstreet_card_alert   |   y_pred_proba_highstreet_card_threshold |
+====+===============+===============+=============+=====================+=====================+==========+===============+==================+================+====================+===================================+=====================================+===================================+=======================================+====================================+======================================+====================================+========================================+======================================+========================================+======================================+==========================================+
|  0 | [0:5999]      |             0 |        5999 | 2020-09-01 03:10:01 | 2020-09-13 16:15:10 |          |      2.41991  |            0.298 | False          |               0.05 |                        0.0133667  |                               0.281 | False                             |                                  0.05 |                         0.0122833  |                                0.38  | False                              |                                   0.05 |                            0.0057    |                                  0.994 | False                                |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  1 | [6000:11999]  |          6000 |       11999 | 2020-09-13 16:15:32 | 2020-09-25 19:48:42 |          |      1.26339  |            0.532 | False          |               0.05 |                        0.0220333  |                               0.01  | True                              |                                  0.05 |                         0.00845    |                                0.828 | False                              |                                   0.05 |                            0.0135667 |                                  0.265 | False                                |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  2 | [12000:17999] |         12000 |       17999 | 2020-09-25 19:50:04 | 2020-10-08 02:53:47 |          |      0.211705 |            0.9   | False          |               0.05 |                        0.00931667 |                               0.727 | False                             |                                  0.05 |                         0.00786667 |                                0.886 | False                              |                                   0.05 |                            0.00845   |                                  0.828 | False                                |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  3 | [18000:23999] |         18000 |       23999 | 2020-10-08 02:57:34 | 2020-10-20 15:48:19 |          |      1.04594  |            0.593 | False          |               0.05 |                        0.0068     |                               0.961 | False                             |                                  0.05 |                         0.0126167  |                                0.347 | False                              |                                   0.05 |                            0.02025   |                                  0.022 | True                                 |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  4 | [24000:29999] |         24000 |       29999 | 2020-10-20 15:49:06 | 2020-11-01 22:04:40 |          |      2.89101  |            0.236 | False          |               0.05 |                        0.0161333  |                               0.116 | False                             |                                  0.05 |                         0.0126167  |                                0.347 | False                              |                                   0.05 |                            0.01025   |                                  0.612 | False                                |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  5 | [30000:35999] |         30000 |       35999 | 2020-11-01 22:04:59 | 2020-11-14 03:55:33 |          |    131.238    |            0     | True           |               0.05 |                        0.174467   |                               0     | True                              |                                  0.05 |                         0.1468     |                                0     | True                               |                                   0.05 |                            0.2077    |                                  0     | True                                 |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  6 | [36000:41999] |         36000 |       41999 | 2020-11-14 03:55:49 | 2020-11-26 09:19:06 |          |    155.593    |            0     | True           |               0.05 |                        0.1713     |                               0     | True                              |                                  0.05 |                         0.144717   |                                0     | True                               |                                   0.05 |                            0.210867  |                                  0     | True                                 |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  7 | [42000:47999] |         42000 |       47999 | 2020-11-26 09:19:22 | 2020-12-08 14:33:56 |          |    182.001    |            0     | True           |               0.05 |                        0.170533   |                               0     | True                              |                                  0.05 |                         0.140967   |                                0     | True                               |                                   0.05 |                            0.2153    |                                  0     | True                                 |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  8 | [48000:53999] |         48000 |       53999 | 2020-12-08 14:34:25 | 2020-12-20 18:30:30 |          |    137.685    |            0     | True           |               0.05 |                        0.173467   |                               0     | True                              |                                  0.05 |                         0.14205    |                                0     | True                               |                                   0.05 |                            0.209533  |                                  0     | True                                 |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+
|  9 | [54000:59999] |         54000 |       59999 | 2020-12-20 18:31:09 | 2021-01-01 22:57:55 |          |    164.407    |            0     | True           |               0.05 |                        0.1673     |                               0     | True                              |                                  0.05 |                         0.14755    |                                0     | True                               |                                   0.05 |                            0.20505   |                                  0     | True                                 |                                     0.05 |
+----+---------------+---------------+-------------+---------------------+---------------------+----------+---------------+------------------+----------------+--------------------+-----------------------------------+-------------------------------------+-----------------------------------+---------------------------------------+------------------------------------+--------------------------------------+------------------------------------+----------------------------------------+--------------------------------------+----------------------------------------+--------------------------------------+------------------------------------------+

NannyML can show the statistical properties of the drift in model scores as a plot.

.. code-block:: python

    >>> for label in calc.y_pred_proba.keys():
    ...     figure = results.plot(kind='score_drift', class_label=label, plot_reference=True)
    ...     figure.show()

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-drift-prepaid_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-drift-upmarket_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-drift-highstreet_card.svg

NannyML can also visualise how the distributions of the model scores evolved over time.

.. code-block:: python

    >>> for label in calc.y_pred_proba.keys():
    ...     figure = results.plot(kind='score_distribution', class_label=label, plot_reference=True)
    ...     figure.show()

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-distribution-prepaid_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-distribution-upmarket_card.svg

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-score-distribution-highstreet_card.svg

NannyML can show the statistical properties of the drift in the predicted labels as a plot.

.. code-block:: python

     >>> figure = results.plot(kind='prediction_drift', plot_reference=True)
     >>> figure.show()

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-prediction-drift.svg

NannyML can also visualise how the distributions of the predicted labels evolved over time.

.. code-block:: python

     >>> figure = results.plot(kind='prediction_distribution', plot_reference=True)
     >>> figure.show()

.. image:: /_static/tutorials/detecting_data_drift/model_outputs/multiclass/drift-guide-prediction-distribution.svg



What Next
-----------------------

If required, the :ref:`Performance Estimation<performance-estimation>` functionality of NannyML can help provide estimates of the impact of the
observed changes to Model Outputs.

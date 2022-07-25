.. _compare_estimated_and_realized_performance:

============================================
Comparing Estimated and Realized Performance
============================================



When the :term:`targets<Target>` become available, the quality of estimations provided by NannyML can be evaluated.
The synthetic datasets provided with the library contain targets for analysis period.
It consists of ``identifier``, which allows to match it with
``analysis`` data, and the target for the monitored model - ``work_home_actual``. See:

.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
    >>> analysis_target.head(3)


+----+--------------+--------------------+
|    |   identifier |   work_home_actual |
+====+==============+====================+
|  0 |        50000 |                  1 |
+----+--------------+--------------------+
|  1 |        50001 |                  1 |
+----+--------------+--------------------+
|  2 |        50002 |                  1 |
+----+--------------+--------------------+

The beginning of the code below is similar to the one in :ref:`tutorial on
performance estimation with binary classification data<performance-estimation-binary-just-the-code>`.

Estimation results for ``reference`` and ``analysis`` are combined with realized and plot the two on the same graph.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> from IPython.display import display
    >>> from sklearn.metrics import roc_auc_score
    >>> import matplotlib.pyplot as plt

    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()

    >>> estimator = nml.CBPE(
    ...     y_pred_proba='y_pred_proba',
    ...     y_pred='y_pred',
    ...     y_true='work_home_actual',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['roc_auc', 'f1'],
    ...     chunk_size=5000
    >>> )

    >>> estimator.fit(reference)

    >>> results = estimator.estimate(pd.concat([reference, analysis], ignore_index=True))

    >>> analysis_full = pd.merge(analysis, analysis_target, on = 'identifier')
    >>> df_all = pd.concat([reference, analysis_full]).reset_index(drop=True)
    >>> target_col = 'work_home_actual'
    >>> pred_score_col = 'y_pred_proba'
    >>> actual_performance = []
    >>> for idx in results.data.index:
    ...     start_index, end_index = results.data.loc[idx, 'start_index'], results.data.loc[idx, 'end_index']
    ...     sub = df_all.loc[start_index:end_index]
    ...     actual_perf = roc_auc_score(sub[target_col], sub[pred_score_col])
    ...     results.data.loc[idx, 'actual_roc_auc'] = actual_perf


    >>> results.data[['estimated_roc_auc', 'actual_roc_auc']].plot()
    >>> plt.xlabel('chunk')
    >>> plt.ylabel('ROC AUC')
    >>> plt.show()


.. image:: /_static/guide-performance_estimation_tmp.svg

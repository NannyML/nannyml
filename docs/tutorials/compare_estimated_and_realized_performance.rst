.. _compare_estimated_and_realized_performance:

============================================
Comparing Estimated and Realized Performance
============================================



When the ground truth becomes available, the quality of estimation can be evaluated. For the synthetic dataset, the
ground truth is given in ``analysis_targets`` variable. It consists of ``identifier`` that allows to match it with
``analysis`` data and the target for monitored model - ``work_home_actual``. Start with the code from :ref:`tutorial on
performance estimation with binary classification data<performance-estimation-binary-just-the-code>` and continue:

.. code-block:: python

    >>> analysis_targets.head(3)


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
    >>> # merge target data to analysis
    >>> analysis_full = pd.merge(analysis, analysis_targets, on = 'identifier')
    >>> df_all = pd.concat([reference, analysis_full]).reset_index(drop=True)
    >>> target_col = 'work_home_actual'
    >>> pred_score_col = 'y_pred_proba'
    >>> actual_performance = []
    >>> for idx in est_perf_with_ref.data.index:
    >>>     start_index, end_index = est_perf_with_ref.data.loc[idx, 'start_index'], est_perf_with_ref.data.loc[idx, 'end_index']
    >>>     sub = df_all.loc[start_index:end_index]
    >>>     actual_perf = roc_auc_score(sub[target_col], sub[pred_score_col])
    >>>     est_perf_with_ref.data.loc[idx, 'actual_roc_auc'] = actual_perf
    >>> # plot
    >>> est_perf_with_ref.data[['estimated_roc_auc', 'actual_roc_auc']].plot()
    >>> plt.xlabel('chunk')
    >>> plt.ylabel('ROC AUC')
    >>> plt.show()


.. image:: /_static/guide-performance_estimation_tmp.svg

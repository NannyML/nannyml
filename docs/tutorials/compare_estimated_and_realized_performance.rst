.. _compare_estimated_and_realized_performance:

============================================
Comparing Estimated and Realized Performance
============================================



When the :term:`targets<Target>` become available, the quality of estimations provided by NannyML can be evaluated. For the synthetic datasets
included in the library, the targets are given in the ``analysis_targets`` variable. It consists of ``identifier``, which allows to match it with
``analysis`` data, and the target for the monitored model - ``work_home_actual``. 

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

Before following the guide below, you should run through the :ref:`tutorial on
performance estimation with binary classification data<performance-estimation-binary-just-the-code>`.

Then you can combine the estimated performance with the :ref:`realised performance calculation<performance-calculation>`
and plot the two on the same graph.

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

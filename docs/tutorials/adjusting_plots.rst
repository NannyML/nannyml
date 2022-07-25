.. _adjusting_plots:

======================================
Adjusting Plots
======================================

NannyML uses Plotly [1]_ to generate figures and returns ``plotly.graph_objs._figure.Figure`` from ``.plot`` methods.
When you need the plot to do something other than the default plotting, e.g. add another curve or indicate a specific
time period in the figure, you can do this by updating the plot.

The example below adds an additional indicator for a particular period of interest using this method.

.. code-block:: python

    >>> import pandas as pd
    >>> import nannyml as nml
    >>> reference, analysis, analysis_target = nml.load_synthetic_binary_classification_dataset()
    >>> estimator = nml.CBPE(
    ...     y_pred_proba='y_pred_proba',
    ...     y_pred='y_pred',
    ...     y_true='work_home_actual',
    ...     timestamp_column_name='timestamp',
    ...     metrics=['roc_auc'],
    ...     chunk_size=5000
    >>> ).fit(reference)
    >>> estimated_performance = estimator.estimate(analysis)
    >>> figure = estimated_performance.plot(kind='performance', metric='roc_auc', plot_reference=True)
    >>> 
    >>> # indicate period of interest
    >>> import datetime as dt
    >>> figure.add_vrect(x0=dt.datetime(2019,1,1), x1=dt.datetime(2020, 1,1),
    ...                  annotation_text="Strategy change", annotation_position="top left")
    >>> figure.show()

.. image:: /_static/adjusting_plots_time_periods_indication.svg

.. [1] https://plotly.com/python/

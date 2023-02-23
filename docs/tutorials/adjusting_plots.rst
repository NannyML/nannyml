.. _adjusting_plots:

======================================
Adjusting Plots
======================================

NannyML uses Plotly [1]_ to generate figures and returns ``plotly.graph_objs._figure.Figure`` from ``.plot`` methods.
When you need the plot to do something other than the default plotting, e.g. add another curve or indicate a specific
time period in the figure, you can do this by updating the plot.

The example below adds an additional indicator for a particular period of interest using this method.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Adjusting plots.ipynb
    :cells: 1

.. image:: /_static/tutorials/adjust_plots/adjusting_plots_time_periods_indication.svg

.. [1] https://plotly.com/python/

.. _compare_estimated_and_realized_performance:

============================================
Comparing Estimated and Realized Performance
============================================

When the :term:`targets<Target>` become available, the quality of estimations provided by NannyML can be evaluated.
The synthetic datasets provided with the library contain targets for analysis period.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to calculate performance, so the first thing we need to in this case is set up the right data in the right dataframes.

The analysis target values are joined on the analysis frame by their index. Your dataset may already contain the ``target`` column, so you may skip this join.

The beginning of the code below is similar to the one in :ref:`tutorial on
performance calculation with binary classification data<binary-performance-calculation>`.

Estimation results for ``reference`` and ``analysis`` are combined with realized and plot the two on the same graph.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cell: 2

... New estimator instance

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 3

... Fit with reference dataset, estimate analysis dataset and filter only the analysis periods

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cell: 5

... Compute actual roc auc

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cell: 7

... Plot the Estimated and Actual values to compare them

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 8


.. image:: /_static/tutorials/estimated_and_realized_performance/tutorial-binary-car-loan-roc-auc-estimated-and-actual.svg

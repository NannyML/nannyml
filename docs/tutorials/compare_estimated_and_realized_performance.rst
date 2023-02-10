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

Next we create the Confidence-based Performance Estimation (CBPE) estimator with a list of metrics, and an optional chunking specification.
For more information about chunking you can check the setting up page and advanced guide.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 3

... Fit with reference dataset, estimate analysis dataset and filter only the analysis periods

The CBPE estimator is then fitted using the fit() method on the reference data.

We estimate the performance of both the reference and analysis datasets,
to compare the estimated performance of the reference dataset with the actual performance.

We then filter the results to only have the estimated values.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cell: 5

We compute the actual performance with `sklearn` using the `Target` values.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cell: 7

Finally we can plot the Actual vs the Estimated performance:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance - Car Loan.ipynb
    :cells: 8


.. image:: /_static/tutorials/estimated_and_realized_performance/tutorial-binary-car-loan-roc-auc-estimated-and-actual.svg

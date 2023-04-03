.. _compare_estimated_and_realized_performance:

============================================
Comparing Estimated and Realized Performance
============================================


Just the code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 1 3 5 6 8 10


Walkthrough
------------

When the :term:`targets<Target>` become available, the quality of estimations provided by NannyML can be evaluated.

The beginning of the code below is similar to the one in :ref:`tutorial on
performance calculation with binary classification data<binary-performance-calculation>`.

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model
predicts whether a customer will repay a loan to buy a car.
Check out :ref:`Car Loan Dataset<dataset-synthetic-binary-car-loan>` to learn more about this dataset.

This datasets provided contains targets for analysis period. It has the :term:`target<Target>` values for the
monitored model in the **repaid** column.


.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cell: 2

For this example, the analysis targets and the analysis frame are joined by their index.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 3

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cell: 4

Estimating performance without targets
======================================

We create the Confidence-based Performance Estimation (:class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE`)
estimator with a list of metrics, and an optional :term:`chunking<Data Chunk>` specification.
For more information about chunking you can check the :ref:`chunking tutorial<chunking>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 5

The :class:`~nannyml.performance_estimation.confidence_based.cbpe.CBPE` estimator is then fitted using
the :meth:`~nannyml.base.AbstractEstimator.fit` method on the reference data.

We estimate the performance of both the reference and analysis datasets,
to compare the estimated and actual performance of the reference period.

We filter the results to only have the estimated values.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 6

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cell: 7

Comparing to realized performance
=================================

We'll first calculate the realized performance:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 8

.. nbtable::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cell: 9

We can then visualize both estimated and realized performance in a single comparison plot.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Compare Estimated and Realized Performance.ipynb
    :cells: 10

.. image:: /_static/tutorials/estimated_and_realized_performance/comparison_plot.svg

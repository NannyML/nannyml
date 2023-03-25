.. _custom-metric-estimation:

========================================================================================
Creating and Estimating a Custom Binary Classification Metric
========================================================================================
This tutorial explains how to use NannyML to estimate the a custom metric based on :term:`confusion matrix<Confusion Matrix>` for binary classification
models in the absence of target data. To find out how CBPE estimates performance, read the :ref:`explanation of Confidence-based
Performance Estimation<performance-estimation-deep-dive>`.

.. _custom-metric-estimation-binary-just-the-code:

Just the Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Creating and Estimating a Custom Metric - Binary Classification.ipynb
    :cells: 1 3 4 5 6 8 9 10

Walkthrough
--------------

For simplicity this guide is based on a synthetic dataset included in the library, where the monitored model
predicts whether a customer will repay a loan to buy a car.
You can read more about this synthetic dataset :ref:`here<dataset-synthetic-binary-car-loan>`.

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

We start by importing the libraries we'll nedd and loading the dataset we'll be using:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Creating and Estimating a Custom Metric - Binary Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Creating and Estimating a Custom Metric - Binary Classification.ipynb
    :cell: 2
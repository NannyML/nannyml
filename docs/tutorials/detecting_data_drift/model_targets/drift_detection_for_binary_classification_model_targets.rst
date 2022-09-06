.. _drift_detection_for_binary_classification_model_targets:

=======================================================
Drift Detection for Binary Classification Model Targets
=======================================================

Why Perform Drift Detection for Model Targets
---------------------------------------------

The performance of a machine learning model can be affected if the distribution of targets changes.
The target distribution can change both because of data drift but also because of label shift.

A change in the target distribution may mean that business assumptions on which the model is
used may need to be revisited.

NannyML uses :class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
in order to monitor drift in the :term:`Target` distribution. It can calculate the mean occurrence of positive
events for binary classification problems.

It can also calculate the chi squared statistic (from the :term:`Chi Squared test<Chi Squared test>`)
of the available target values for each chunk, for both binary and multiclass classification problems.

.. note::
    The Target Drift detection process can handle missing target values across all :term:`data periods<Data Period>`.


Just The Code
------------------------------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cells: 1 3 4 6 8


Walkthrough
------------------------------------------------

In order to monitor a model, NannyML needs to learn about it from a reference dataset. Then it can monitor the data that is subject to actual analysis, provided as the analysis dataset.
You can read more about this in our section on :ref:`data periods<data-drift-periods>`.

Let's start by loading some synthetic data provided by the NannyML package, and setting it up as our reference and analysis dataframes. This synthetic data is for a binary classification model, but multi-class classification can be handled in the same way.

The ``analysis_targets`` dataframe contains the target results of the analysis period. This is kept separate in the synthetic data because it is
not used during :ref:`performance estimation.<performance-estimation>`. But it is required to detect drift for the targets, so the first thing we need to in this case is set up the right data in the right dataframes.  The analysis target values are joined on the analysis frame by the ``identifier`` column.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cell: 2

Now that the data is in place we'll create a new
:class:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator`
instantiating it with the appropriate parameters. We only need the target (``y_true``) and timestamp.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cells: 3


Afterwards, the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.fit`
method gets called on the reference :term:`period<Data Period>`, which represent an accepted target distribution
which we will compare against the analysis :term:`period<Data Period>`.

Then the :meth:`~nannyml.drift.target.target_distribution.calculator.TargetDistributionCalculator.calculate` method is
called to calculate the target drift results on the data provided. We use the previously assembled data as an argument.

We can display the results of this calculation in a dataframe.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cell: 5

The results can be also easily plotted by using the
:meth:`~nannyml.drift.target.target_distribution.result.TargetDistributionResult.plot` method.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cells: 6


Note that a dashed line, instead of a solid line, will be used for chunks that have missing target values.

.. image:: /_static/target_distribution_metric.svg

.. nbimport::
    :path: ./example_notebooks/Tutorial - Drift - Model Targets - Binary Classification.ipynb
    :cells: 8

.. image:: /_static/target_distribution_statistical.svg


Insights
-----------------------

Looking at the results we see that we have a false alert on the first chunk of the analysis data. This
can happen when the statistical tests consider a small change in the distribution of a variable to be significant.
But because the change is small it is usually not significant from a model monitoring perspective.



What Next
-----------------------

The :ref:`performance-calculation` functionality of NannyML can can add context to the target drift results
showing whether there are associated performance changes.

.. _missing_values:

========================
Missing Values Detection
========================


Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cells: 1 3 4 6

.. _missing_values_walkthrough:

Walkthrough
-----------

NannyML's approach to missing values detection is quite straightforward.
For each :term:`chunk<Data Chunk>` NannyML calculates the number of missing values. There is an option, called
``normalize``, to convert the count of values to a relative ratio if needed. The resulting
values from the reference :ref:`data<data-drift-periods>` chunks are used to calculate the
alert :term:`thresholds<Threshold>`. The missing values results from the analysis chunks are
compared against those thresholds and generate alerts if applicable.

We begin by loading the :ref:`titanic dataset<dataset-titanic>` provided by the NannyML package.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cell: 2

The :class:`~nannyml.data_quality.missing.calculator.MissingValuesCalculator` class implements
the functionality needed for missing values calculations.
We need to instantiate it with appropriate parameters:

- The names of the columns to be evaluated.
- Optionally, a boolean option indicating whether we want the absolute count of the missing
  value instances or their relative ratio. By default it is set to true.
- Optionally, the name of the column containing the observation timestamps.
- Optionally, a chunking approach or a predefined chunker. If neither is provided, the default
  chunker creating 10 chunks will be used.
- Optionally, a threshold strategy to modify the default one. See available threshold options
  :ref:`here<thresholds>`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cells: 3

Next, the :meth:`~nannyml.data_quality.missing.calculator.MissingValuesCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be
compared with for :term:`alert<Alert>` generation. Then the
:meth:`~nannyml.data_quality.missing.calculator.MissingValuesCalculator.calculate` method will
calculate the data quality results on the data provided to it.

The results can be filtered to only include a certain data period, method or column by using the ``filter`` method.
You can evaluate the result data by converting the results into a `DataFrame`,
by calling the :meth:`~nannyml.data_quality.missing.result.Result.to_df` method.
By default this will return a `DataFrame` with a multi-level index. The first level represents the column, the second level
represents resulting information such as the data quality metric values, the alert thresholds or the associated sampling error.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cell: 5

More information on accessing the information contained in the
:class:`~nannyml.data_quality.missing.result.Result`
can be found on the :ref:`working_with_results` page.

The next step is visualizing the results, which is done using the
:meth:`~nannyml.data_quality.missing.result.Result.plot` method.
It is recommended to filter results for each column and plot separately.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Missing Values.ipynb
    :cells: 6

.. image:: /_static/tutorials/data_quality/missing-titanic-Age.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Cabin.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Embarked.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Fare.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Name.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Parch.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Pclass.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Sex.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-SibSp.svg
.. image:: /_static/tutorials/data_quality/missing-titanic-Ticket.svg

Insights
--------

We see that most of the dataset columns don't have missing values. The **Age** and **Cabin**
columns are the most interesting with regards to missing values.


What Next
---------

We can also inspect the dataset for :term:`Unseen Values` in the :ref:`Unseen Values Tutorial<unseen_values>`.
Then we can look for any :term:`Data Drift` present in the dataset using :ref:`data-drift` functionality of
NannyML.

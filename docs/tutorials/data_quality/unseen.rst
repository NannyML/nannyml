.. _unseen_values:

=======================
Unseen Values Detection
=======================

Just The Code
-------------

.. nbimport::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cells: 1 3 4 6

.. _unseen_values_walkthrough:

Walkthrough
-----------

NannyML defines :term:`unseen values<Unseen Values>` as categorical feature values that are not present
in the :term:`reference period<Data Period>`.
NannyML's approach to unseen values detection is simple.
The reference :term:`period <Data Period>` is used to create a set of expected values for
each categorical feature.
For each :term:`chunk<Data Chunk>` in the analysis :term:`period <Data Period>`
NannyML calculates the number of unseen values. There is an option,
called ``normalize``, to convert the count of values to a relative ratio if needed.
If unseen values are detected in a chunk, an alert is raised for the relevant feature.

We begin by loading the :ref:`titanic dataset<dataset-titanic>` provided by the NannyML package.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cells: 1

.. nbtable::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cell: 2

The :class:`~nannyml.data_quality.unseen.calculator.UnseenValuesCalculator` class implements
the functionality needed for unseen values calculations.
We need to instantiate it with appropriate parameters:

- **column_names:** A list with the names of columns to be evaluated. They need to be categorical columns.
- **normalize (Optional):** Optionally, a boolean option indicating whether we want the absolute count of the missing
  value instances or their relative ratio. By default it is set to true.
- **timestamp_column_name (Optional):** The name of the column in the reference data that
  contains timestamps.
- **chunk_size (Optional):** The number of observations in each chunk of data
  used. Only one chunking argument needs to be provided. For more information about
  :term:`chunking<Data Chunk>` configurations check out the :ref:`chunking tutorial<chunking>`.
- **chunk_number (Optional):** The number of chunks to be created out of data provided for each
  :ref:`period<data-drift-periods>`.
- **chunk_period (Optional):** The time period based on which we aggregate the provided data in
  order to create chunks.
- **chunker (Optional):** A NannyML :class:`~nannyml.chunk.Chunker` object that will handle the aggregation
  provided data in order to create chunks.
- **thresholds (Optional):** The threshold strategy used to calculate the alert threshold limits.
  For more information about thresholds, check out the :ref:`thresholds tutorial<thresholds>`.

.. warning::

    Note that because of how unseen values are defined they will be 0 by definition
    for the :term:`reference period<Data Period>`. Hence the
    :ref:`StandardDeviationThreshold<thresholds_std>`
    threshold option is not really applicable for this calculator.


.. nbimport::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cells: 3

Next, the :meth:`~nannyml.base.AbstractCalculator.fit` method needs
to be called on the reference data, which provides the baseline that the analysis data will be
compared with for :term:`alert<Alert>` generation. Then the
:meth:`~nannyml.base.AbstractCalculator.calculate` method will
calculate the data quality results on the data provided to it.

The results can be filtered to only include a certain data period, method or column by using the ``filter`` method.
You can evaluate the result data by converting the results into a `DataFrame`,
by calling the :meth:`~nannyml.base.AbstractResult.to_df` method.
By default this will return a `DataFrame` with a multi-level index. The first level represents the column, the second level
represents resulting information such as the data quality metric values and the alert thresholds.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cells: 4

.. nbtable::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cell: 5

More information on accessing the information contained in the
:class:`~nannyml.data_quality.unseen.result.Result`
can be found on the :ref:`working_with_results` page.

The next step is visualizing the results, which is done using the
:meth:`~nannyml.data_quality.unseen.result.Result.plot` method.
It is recommended to filter results for each column and plot separately.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Unseen Values.ipynb
    :cells: 6

.. image:: /_static/tutorials/data_quality/unseen-titanic-Cabin.svg
.. image:: /_static/tutorials/data_quality/unseen-titanic-Embarked.svg
.. image:: /_static/tutorials/data_quality/unseen-titanic-Sex.svg
.. image:: /_static/tutorials/data_quality/unseen-titanic-Ticket.svg

Insights
--------

We see that most of the dataset columns don't have unseen values. The **Ticket** and **Cabin**
columns are the most interesting with regards to unseen values.


What Next
---------

We can also inspect the dataset for missing values in the :ref:`Missing Values Tutorial<missing_values>`.
Then we can look for any :term:`Data Drift` present in the dataset using :ref:`data-drift` functionality of
NannyML.

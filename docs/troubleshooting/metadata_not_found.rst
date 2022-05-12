.. _metadata_not_found:

===========================================
Dealing with a ``MissingMetadataException``
===========================================

The problem
-----------

Running the ``fit``, ``calculate`` or ``estimate`` methods of a ``Calculator`` or ``Estimator`` fails by returning
a :class:`~nannyml.exceptions.MissingMetadataException`.

.. code-block::

    nannyml.exceptions.MissingMetadataException: metadata is still missing values for ['predicted_probability_column_name'].


The solution
------------

The :class:`~nannyml.exceptions.MissingMetadataException` is raised when the :class:`model metadata<nannyml.metadata.base.ModelMetadata>`
used to create the ``Calculator`` or ``Estimator`` is not complete, i.e. it is missing some required properties.

The exception will list the properties it is missing, as shown in the problem statement.

Assume ``md`` is the :class:`model metadata<nannyml.metadata.base.ModelMetadata>` object used,
``predicted_probability_column_name`` is the property missing and in your data the predicted probabilities are located
in the ``model_probas`` column.

The following snippet should help you prevent the exception by completing the metadata manually:

.. code-block:: python

    >>> md.is_complete()  # just checking
    (False, ['predicted_probability_column_name'])
    >>> md.predicted_probability_column_name = 'model_probas'
    >>> md.is_complete()
    (True, [])

Any metadata property can be set or updated.


Related reads
-------------

To read more on metadata, read the tutorial on :ref:`providing metadata<import-data>`.

NannyML can automatically extract metadata from a data sample if it follows some :ref:`naming conventions<deep_dive_metadata_extraction>`.

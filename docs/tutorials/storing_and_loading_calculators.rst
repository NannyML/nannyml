.. _storing_and_loading_calculators:

======================================
Storing and loading calculators
======================================

Fitting a calculator or estimator is only required when the reference data for a monitored model changes.
To avoid unnecessary calculations and speed up (repeating) runs of NannyML, you can store the fitted calculators
in a :class:`~nannyml.io.store.base.Store`.

.. note::

    We currently support persisting objects to a local or remote filesystem such as S3,
    Google Cloud Storage buckets, or Azure Blob Storage. You can find some :ref:`examples<storing_calculators_remote_examples>` in the walkthrough.

.. note::

    For more information on how to use this functionality with the CLI or container, check the
    :ref:`configuration file documentation<cli_configuration_store>`.

Just the code
--------------

Create the calculator and fit it on reference. Store the fitted calculator on a local disk.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 1 2 3

In a new session load the stored calculator and use it.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 4 5

Walkthrough
-----------

In the first part, we create a new :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` and fit it
to the reference data.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 1

In this snippet, we will set up the :class:`~nannyml.io.store.file_store.FilesystemStore`. It is a class responsible for
storing objects on a filesystem and retrieving it back.
We will first illustrate creating a store using the local filesystem. The `root_path` parameter configures the directory
on the filesystem that will be used as the root of our store. Additional directories and files can be created when
actually storing objects.

We will now provide a directory on the local filesystem.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 2

.. _storing_calculators_remote_examples:

Because we are using the `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ library under the covers we also
support a lot of remote filesystems out of the box.

The following snippet shows how to use S3 as a backing filesystem. See `AWS documentation <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html>`_
to learn more about the required access key id and secret access key credentials.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 7

This is how to use Google Cloud Storage as a backing system. See `Google Cloud documentation <https://cloud.google.com/iam/docs/creating-managing-service-account-keys>`_
to learn more about the required service account key credentials.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 8

This snippet illustrates how to do this using Azure Blob Storage. See `Azure support documentation <https://github.com/fsspec/adlfs#setting-credentials>`_
to learn more about the required credentials.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 9

The next step is using the :class:`~nannyml.io.store.file_store.FilesystemStore` to store our fitted calculator.
To do this, we can provide an optional `path` string parameter. It allows us to set a custom subdirectory and file name.
If no `path` is provided, a file will be created using a standard name within the root directory of the store.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 3

This concludes the first part: storing the fitted calculator.

When running NannyML in a new session to perform calculations on analysis data (e.g., repeated on a daily basis), we can load the pre-fitted calculator from the store.
But, first, we define the analysis data and declare the store:

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 4

Now we will use the store to load the pre-fitted calculator from the disk. By providing the optional `as_type` parameter,
we can have the store check the type of the loaded object before returning it. If it is not an instance of `as_type`, the
:meth:`~nannyml.io.store.base.Store.load` method will raise a :class:`~nannyml.exceptions.StoreException`.

If nothing is found at the given `path`, the :meth:`~nannyml.io.store.base.Store.load` method will return
`None`.

.. nbimport::
    :path: ./example_notebooks/Tutorial - Storing and Loading Calculators - Univariate.ipynb
    :cells: 5

.. nbtable::
    :path: ./example_notebooks/Tutorial - Ranking.ipynb
    :cell: 6

What's Next
===========

The :class:`~nannyml.io.store.file_store.FilesystemStore` can also be used when running NannyML using the CLI or as
a container. You can learn how in the :ref:`configuration file documentation<cli_configuration_store>`.

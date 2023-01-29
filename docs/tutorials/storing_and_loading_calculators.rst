.. _storing_and_loading_calculators:

======================================
Storing and loading calculators
======================================

Fitting a calculator or estimator is only required when the reference data for a monitored model changes.
To avoid unnecessary calculations and speed up (repeating) runs of NannyML, you can store the fitted calculators
to a :class:`~nannyml.io.store.base.Store`.

.. note::

    We currently support persisting objects to a local or remote filesystem such as S3,
    Google Cloud Storage buckets or Azure Blob Storage. You can find some :ref:`examples<storing_calculators_remote_examples>` in the walkthrough.

.. note::

    For more information on how to use this functionality with the CLI or container, check the
    :ref:`configuration file documentation<cli_configuration_store>`.

Just the code
--------------

Create the calculator and fit it on reference. Store the fitted calculator to local disk.

.. code-block:: python

    import nannyml as nml

    reference_df, _, _ = nml.load_synthetic_binary_classification_dataset()
    column_names = ['distance_from_office', 'salary_range', 'gas_price_per_litre', 'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure', 'y_pred_proba', 'y_pred']
    calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
    )
    calc.fit(reference_df)

    store = nml.io.store.FilesystemStore(root_path='/tmp/nml-cache')
    store.store(calc, path='example/calc.pkl')


In a new session load the stored calculator and use it.

.. code-block:: python

    import nannyml as nml

    _, analysis_df, _ = nml.load_synthetic_binary_classification_dataset()
    store = nml.io.store.FilesystemStore(root_path='/tmp/nml-cache')
    loaded_calc = store.load(path='example/calc.pkl', as_type=nml.UnivariateDriftCalculator)
    result = loaded_calc.calculate(analysis_df)
    display(result.to_df())

Walkthrough
-----------

In the first part we create a new :class:`~nannyml.drift.univariate.calculator.UnivariateDriftCalculator` and fit it
to the reference data.

.. code-block:: python

    import nannyml as nml

    reference_df, _, _ = nml.load_synthetic_binary_classification_dataset()
    column_names = ['distance_from_office', 'salary_range', 'gas_price_per_litre', 'public_transportation_cost', 'wfh_prev_workday', 'workday', 'tenure', 'y_pred_proba', 'y_pred']
    calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        timestamp_column_name='timestamp',
        continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
        categorical_methods=['chi2', 'jensen_shannon'],
    )
    calc.fit(reference_df)

In this snippet we'll set up the :class:`~nannyml.io.store.file_store.FilesystemStore`. It is a class responsible for
storing objects on a filesystem and retrieving it back.
We'll first illustrate creating a store using the local filesystem. The `root_path` parameter configures the directory
on the filesystem that will be used as the root of our store. Additional directories and files can be created when
actually storing objects.

We'll now provide a directory on the local filesystem.

.. code-block:: python

    store = nml.io.store.FilesystemStore(root_path='/opt/nml/cache')


.. _storing_calculators_remote_examples:

Because we're using the `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ library under the covers we also
support a lot of remote filesystems out of the box.

The following snippet shows how to use S3 as a backing filesystem. See https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html
to learn more about the required access key id and secret access key credentials.

.. code-block:: python

    store = FilesystemStore(
        root_path='s3://my-bucket-name/some/path',
        credentials={
            'client_kwargs': {
               'aws_access_key_id': '<ACCESS_KEY_ID>'
               'aws_secret_access_key': '<SECRET_ACCESS_KEY>'
            }
        }
    )

This is how to use Google Cloud Storage as a backing system. See https://cloud.google.com/iam/docs/creating-managing-service-account-keys
to learn more about the required service account key credentials.

.. code-block:: python

    store = FilesystemStore(
        root_path='gs://my-bucket-name/some/path',
        credentials={'token': 'service-account-access-key.json'}
    )

This snippet illustrates how to do this using Azure Blob Storage. See https://github.com/fsspec/adlfs#setting-credentials
to learn more about the required credentials.

.. code-block:: python

    store = FilesystemStore(
        root_path='abfs://my-container-name/some/path',
        credentials={'account_name': '<ACCOUNT_NAME>', 'account_key': '<ACCOUNT_KEY>'}
    )

The next step is using the :class:`~nannyml.io.store.file_store.FilesystemStore` to store our fitted calculator.
To do this we can provide an optional `path` string parameter. It allows us to set a custom subdirectory and file name.
If no `path` is provided a file will be created using a standard name within the root directory of the store.

.. code-block:: python

        store.store(calc, path='example/calc.pkl')

This concludes the first part: storing the fitted calculator. When running NannyML in a new session to perform
calculations on analysis data (e.g. repeated on a daily basis) we can load the pre-fitted calculator from the store.
First we define the analysis data and declare the store:

.. code-block:: python

    _, analysis_df, _ = nml.load_synthetic_binary_classification_dataset()
    store = nml.io.store.FilesystemStore(root_path='/tmp/nml-cache')

Now we'll use the store to load the pre-fitted calculator from disk. By providing the optional `as_type` parameter
we can have the store check the type of the loaded object before returning it. If it is not an instance of `as_type` the
:meth:`~nannyml.io.store.file_store.FilesystemStore.load` method will raise a :class:`~nannyml.exceptions.StoreException`.

If nothing is found at the given `path` the :meth:`~nannyml.io.store.file_store.FilesystemStore.load` method will return
`None`.

.. code-block:: python

    loaded_calc = store.load(path='example/calc.pkl', as_type=nml.UnivariateDriftCalculator)
    result = loaded_calc.calculate(analysis_df)
    display(result.to_df())


What's Next
===========

The :class:`~nannyml.io.store.file_store.FilesystemStore` can also be used when running NannyML using the CLI or as
a container. You can learn how in the :ref:`configuration file documentation<cli_configuration_store>`.

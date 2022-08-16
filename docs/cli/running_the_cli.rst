.. _using_the_cli:

=======================================================
Running the CLI
=======================================================

Installation
---------------------------------------------

The ``nml`` CLI is included with the NannyML library. It can be installed using ``pip``:


.. code-block:: bash

    pip install nannyml


or conda:


.. code-block:: bash

    conda config --add channels conda-forge
    conda install nannyml

The ``nml`` CLI is also distributed as a Docker container, allowing you to run NannyML without having Python installed
on your machine or use NannyML in containerized workflows.

.. code-block:: bash

    docker -v /local/config/dir/:/config/ run nannyml/nannyml nml run


Configuration
--------------

You can use a configuration file and some command line arguments to provide NannyML with the information it needs.
All CLI arguments can also be specified within the configuration file, eliminating the need for any CLI arguments
at all. Because of its dynamic nature, we've opted to require a configuration file to specify things like input data
and column mappings.

You can read more about the configuration file format in the
:ref:`configuration file format documentation<cli_configuration_format>`.

NannyML will look for the configuration file with a default name in default locations. If need be the path to a
configuration file with a custom name or location can be provided.

You can read more about configuration file locations in the
:ref:`configuration file locations documentation<cli_configuration_location>`.

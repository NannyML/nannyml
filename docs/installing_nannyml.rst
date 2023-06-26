.. _installing_nannyml:


------------------
Installing NannyML
------------------

NannyML depends on `LightGBM`_. This might require you to set install additional
OS-specific binaries. You can follow the `official LightGBM installation guide`_.

From the shell of your python environment type:

.. code-block:: bash

    $ pip install nannyml

or

.. code-block:: bash

    $ conda install -c conda-forge nannyml


or

.. code-block:: bash

    $ docker -v /local/config/dir/:/config/ run nannyml/nannyml nml run


:ref:`See the quickstart<quick-start>` to find out what you can do with NannyML in practice.

Any issues with installation? `Let us know`_ so we can help you.


.. _`Let us know`: https://github.com/NannyML/nannyml/issues
.. _`LightGBM`: https://github.com/microsoft/LightGBM
.. _`official LightGBM installation guide`: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html

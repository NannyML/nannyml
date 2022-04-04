.. raw:: html

    <p align="center"><img width=60% src="https://assets.website-files.com/6099466e98d9381b3f745b9a/60994ab2b5bd890780db9c84_NannyML%20logo%20horizontal%20typfont.png"></p>
    <p align="center">
        <a href="https://pypi.org/project/nannyml/">
            <img src="https://img.shields.io/pypi/v/nannyml.svg">
        </a>
        <a href="https://pypi.org/project/nannyml/">
            <img src="https://img.shields.io/pypi/pyversions/nannyml.svg">
        </a>
        <a href="https://github.com/nannyml/nannyml/actions/workflows/dev.yml">
            <img src="https://github.com/NannyML/nannyml/actions/workflows/dev.yml/badge.svg">
        </a>
        <a href="https://codecov.io/gh/NannyML/nannyml">
            <img src="https://codecov.io/gh/NannyML/nannyml/branch/main/graph/badge.svg?token=OGpF5gVzfR">
        </a>
    </p>

.. raw:: html

    <p align="center">
        <strong>
            <a href="https://nannyml.com/">Website</a>
            •
            <a href="https://docs.nannyml.com">Docs</a>
            •
            <a href="https://nannymlbeta.slack.com">Community Slack</a>
        </strong>
    </p>

Basic overview
==============

NannyML helps you monitor your ML models in production by:

* estimating performance in absence of ground truth
* calculating realized performance metrics
* detecting data drift on model inputs, model outputs and targets

Installing the last stable release
==================================

.. code-block:: bash
    pip install nannyml


Installing the last development changes
=======================================

.. code-block:: bash
    python -m pip install git+https://github.com/NannyML/nannyml


Getting started
===============

.. code-block:: python
    import nannyml as nml

    metadata = nml.extract_metadata(reference_data)
    estimator = nml.CBPE(metadata).fit(reference_data)
    estimates = estimator.estimate(data)

    estimates.plot(kind='performance').show()

Examples
========

* `Performance estimation <link URL>`_
* `Realized performance calculation <link URL>`_
* `Univariate model input drift detection <link URL>`_
* `Multivariate model input drift detection <link URL>`_
* `Model output drift detection <link URL>`_
* `Model target distribution <link URL>`_

Development setup
=================

* Read the `contributing docs <CONTRIBUTING.md>`_



===========================
Importing data into NannyML
===========================

TLDR
=====

1.  NannyML needs to understand what the inputs and outputs of your ML model look like.
2.  The actual *data* you provide to NannyML are the feature values given to your model
    and the predictions it makes for them.
3.  The ``ModelMetadata`` class stores all this information and uses it in calculations.
4.  You can construct this metadata or extract it based on a sample of your model inputs/outputs.
5.  For metadata extraction to work optimally there are some conventions to follow.

On data, metadata and observations
==================================

NannyML offers tools to help you monitor your **models** in production. To do this it needs to know the predictions your
model has made over time. This is the actual **data** you provide to NannyML: the **model inputs** (*feature values*)
and **outputs** (*prediction values*). Think of it as some kind of log on the usage of your model.

In order to apply the correct analysis on each of your model features NannyML needs some additional information,
such as the kind of data that a feature might hold (continuous, categorical, ordinal, ...),
the time the prediction was made etc. The set of this describing information is what we call **model metadata**.

.. image:: https://via.placeholder.com/900x300.png?text=model+invocation+process

..
    TODO: insert illustration showing model invocation and assigning names to everything

Combining model inputs/outputs with metadata results in a set of rows called **observations**.
NannyML will consume a ``pandas.DataFrame`` of these **observations** and turn these into drift and performance metrics.

.. image:: https://via.placeholder.com/900x300.png?text=annotated+tabular+data


..
    TODO: insert illustration that shows all data in tabular form with annotations


Data requirements
=================




Providing metadata
==================


Constructing metadata manually
------------------------------

Exporting/importing metadata
----------------------------

Extracting metadata from model inputs/outputs
---------------------------------------------

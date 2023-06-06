.. _dataset-titanic:

===============
Titanic Dataset
===============

NannyML provides the titanic dataset in order to help show case it's data quality features.
The titanic dataset provided here is compiled from two sources, kaggle_ and `data.world`_.

One difference of the titanic dataset compared to other dataset provided by NannyML is that
there is not in-built model making predictions. Hence the titanic dataset cannot be used for NannyML
performance estimation and realized performance modules. To find out what requirements NannyML
has for datasets, check out :ref:`Data Requirements<data_requirements>`.

Problem Description
===================

The titanic dataset covers the passengers of the RMS Titanic and tells us whether they
survived its sinking.

Dataset Description
===================

A sample of the dataset can be seen below.


.. code-block:: python

    >>> import nannyml as nml
    >>> reference, analysis, analysis_targets = nml.load_titanic_dataset()
    >>> reference.head()

+----+---------------+----------+-----------------------------------------------------+--------+-------+---------+---------+------------------+---------+---------+------------+--------+--------+---------------------------------------------------+------------+
|    | PassengerId   | Pclass   | Name                                                | Sex    | Age   | SibSp   | Parch   | Ticket           | Fare    | Cabin   | Embarked   | boat   | body   | home.dest                                         | Survived   |
+====+===============+==========+=====================================================+========+=======+=========+=========+==================+=========+=========+============+========+========+===================================================+============+
| 0  | 1             | 3        | Braund, Mr. Owen Harris                             | male   | 22    | 1       | 0       | A/5 21171        | 7.25    | nan     | S          | nan    | nan    | Bridgerule, Devon                                 | 0          |
+----+---------------+----------+-----------------------------------------------------+--------+-------+---------+---------+------------------+---------+---------+------------+--------+--------+---------------------------------------------------+------------+
| 1  | 2             | 1        | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38    | 1       | 0       | PC 17599         | 71.2833 | C85     | C          | 4      | nan    | New York, NY                                      | 1          |
+----+---------------+----------+-----------------------------------------------------+--------+-------+---------+---------+------------------+---------+---------+------------+--------+--------+---------------------------------------------------+------------+
| 2  | 3             | 3        | Heikkinen, Miss. Laina                              | female | 26    | 0       | 0       | STON/O2. 3101282 | 7.925   | nan     | S          | nan    | nan    | nan                                               | 1          |
+----+---------------+----------+-----------------------------------------------------+--------+-------+---------+---------+------------------+---------+---------+------------+--------+--------+---------------------------------------------------+------------+
| 3  | 4             | 1        | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female | 35    | 1       | 0       | 113803           | 53.1    | C123    | S          | D      | nan    | Scituate, MA                                      | 1          |
+----+---------------+----------+-----------------------------------------------------+--------+-------+---------+---------+------------------+---------+---------+------------+--------+--------+---------------------------------------------------+------------+
| 4  | 5             | 3        | Allen, Mr. William Henry                            | male   | 35    | 0       | 0       | 373450           | 8.05    | nan     | S          | nan    | nan    | Lower Clapton, Middlesex or Erdington, Birmingham | 0          |
+----+---------------+----------+-----------------------------------------------------+--------+-------+---------+---------+------------------+---------+---------+------------+--------+--------+---------------------------------------------------+------------+

The dataset has 13 features:

- **Pclass** - a proxy for socio-economic status, 1 is Upper, 2 is Middle and 3 is Lower class.
- **Age** - passenger's age. Fractional if less than one. If it is estimated it is in the form of xx.5.
- **SibSp** - number of Siblings (brother, sister, stepbrother or stepsister) or spouses (husband or wife - mistresses and fiances were ignored) aboard.
- **Parch** - number of parent (mother, father) or children (daughter, son, stepdaughter, stepson) aboard. Children who travelled only with a nanny have Parch=0.
  the employee's residence to the workplace.
- **Ticket** - passenger's Ticket Number.
- **Fare** - passenger's Fare.
- **Cabin** - passenger's cabin  number.
- **Embarked** - passenger's port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- **boat** - lifeboar information if the passenger survived.
- **body** - body number if the passenger did not survive and a body was recovered.
- **home.dest** - passenger's domicile and destination information, if available.

The **Survived** column tells us whether the passenger survived and is what we call the :term:`Target` column.


There is also an auxiliary column that kaggle_ uses and we have kept for compatibility:

- **PassengerId** - a unique number referencing each passenger. This is very useful for joining the target
  results on the analysis dataset.

The titanic dataset is used by NannyML at the :ref:`Data Quality Tutorials<data-quality>` showcasing our Missing values
and Unseen Values detection functionality.

.. _kaggle: https://www.kaggle.com/competitions/titanic/data
.. _`data.world`: https://data.world/nrippner/titanic-disaster-dataset

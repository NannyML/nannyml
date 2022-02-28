.. _california-housing:

==========================
California Housing Dataset
==========================

We are usng the `California Housing Dataset`_ to create a real data example dataset for
NannyML. There are three steps needed for this process:

- Enrich the data so they have the proper context for NannyML's usecase.
- Train a model from those data to play the role of the monitored model.
- Prepare the data


Before this however the data need to be acquired first:

.. code-block:: python

    # Import required libraries
    import pandas as pd
    import numpy as np
    import datetime as dt

    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    cali = fetch_california_housing(as_frame=True)
    df = pd.concat([cali.data, cali.target], axis=1)
    df.head(2)
        MedInc 	HouseAge 	AveRooms 	AveBedrms 	Population 	AveOccup 	Latitude 	Longitude 	MedHouseVal
    0   8.3252  41.0            6.984127        1.02381         322.0           2.555556        37.88           -122.23         4.526
    1   8.3014  21.0            6.238137        0.97188         2401.0          2.109842        37.86           -122.22         3.585


Enriching the data
==================

The things needed to be added to the dataset are:

- A time aspect
- Partitioning the data
- Specifying a target to make the problem a classification problem

.. code-block:: python

    # add artificiacl timestamp
    timestamps = [dt.datetime(2020,1,1) + dt.timedelta(hours=x/2) for x in df.index]
    df['timestamp'] = timestamps

    # add partitions
    train_beg = dt.datetime(2020,1,1)
    train_end = dt.datetime(2020,5,1)
    test_beg = dt.datetime(2020,5,1)
    test_end = dt.datetime(2020,9,1)
    df.loc[df['timestamp'].between(train_beg, train_end, inclusive='left'), 'partition'] = 'train'
    df.loc[df['timestamp'].between(test_beg, test_end, inclusive='left'), 'partition'] = 'test'
    df['partition'] = df['partition'].fillna('production')

    # create new classification target - house value higher than mean
    df_train = df[df['partition']=='train']
    df['clf_target'] = np.where(df['MedHouseVal'] > df_train['MedHouseVal'].median(), 1, 0)
    df = df.drop('MedHouseVal', axis=1)
    del df_train

Adding a Machine Learning Model
===============================

.. code-block:: python

    # fit classifier
    target = 'clf_target'
    meta = 'partition'
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']


    df_train = df[df['partition']=='train']

    clf = RandomForestClassifier(random_state=42)
    clf.fit(df_train[features], df_train[target])
    df['y_pred_proba'] = clf.predict_proba(df[features])[:,1]

    # Check roc auc scores
    for partition_name, partition_data in df.groupby('partition', sort=False):
        print(partition_name, roc_auc_score(partition_data[target], partition_data['y_pred_proba']))
    train 1.0
    test 0.8737681614409617
    production 0.8224322932364313

Preparing Data for NannyML
==========================

The data are now being splitted so they can be in a form required by NannyML.

.. code-block:: python

    df_for_nanny = df[df['partition']!='train'].reset_index(drop=True)
    df_for_nanny['partition'] = df_for_nanny['partition'].map({'test':'reference', 'production':'analysis'})
    df_for_nanny['identifier'] = df_for_nanny.index

    df_ref = df_for_nanny[df_for_nanny['partition']=='reference'].copy()
    df_ana = df_for_nanny[df_for_nanny['partition']=='analysis'].copy()
    df_gt = df_ana[['identifier', 'clf_target']].copy()
    df_ana = df_ana.drop('clf_target', axis=1)

The ``df_ref`` dataframe represents the reference :term:`Partition` and the ``df_ana``
dataframe represents the analysis partition. The ``df_gt`` dataframe contains the targets
for the analysis partition that is provided separately.


.. _California Housing Dataset: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

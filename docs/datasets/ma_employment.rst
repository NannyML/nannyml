.. _dataset-real-world-ma-employment:

============================
US Census Employment dataset
============================

This page shows how US Census MA dataset was obtained and prepared to serve as an example in
:ref:`quickstart<quick-start>`. Full notebook can be found in our `github repository`_.

To find out what requirements NannyML has for datasets, check out :ref:`Data Requirements<data_requirements>`.

Data Source
===========

The dataset comes from US Census and it was obtained using `folktables`_. Feature descriptions are given in `PUMS
documentation`_.

Dataset Description
===================

The task is to predict whether an individual is employed based on features like age, education etc. The data analyzed
comes from the state of Massachusetts and it covers the range from 2014 to 2018.

Preparing Data for NannyML
==========================

Fetching the Data
-----------------

First we import required libraries and fetch the data with `folktables`_:

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 1, 5

.. nbtable::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cell: 6

Data is fetched for each year separately and column `year` is created.

Descriptions of all the variables can be found in the :ref:`appendix<dataset-real-world-ma-employment-feature-description>`.


Defining Partitions and Preprocessing
-------------------------------------

We split the data into three partitions simulating model lifecycle: train, test and production (after deployment)
data. We will use 2014 data for training, 2015 for evaluation and 2016-2018 will simulate production data.

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 10

We now define categorical and numeric features:

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 14

Since categorical features are already encoded correctly for LGBM model (non-negative, integers-like), we don't need
any preprocessing. We will just turn them into proper ``integers``. We will also rename the target column and convert
target to ``int``:

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 17


Developing ML Model and Making Predictions
------------------------------------------

We will now fit a model that will be subject to monitoring (e.g. in :ref:`quickstart<quick-start>`):

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 19, 20

Let's turn categorical features into ``category`` ``dtype`` so that NannyML correctly recognizes them:

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 23

Splitting and Storing the Data
------------------------------

Now we will just split the data based on partitions, drop selected columns and store it in the relevant location in
NannyML repository so the data can be accessed from within the library:

.. nbimport::
    :path: ./example_notebooks/Datasets - Census Employment MA.ipynb
    :cells: 25, 26


.. _dataset-real-world-ma-employment-feature-description:

Appendix: Feature description
-----------------------------
This description comes from `PUMS documentation`_:

**AGEP** - age person, numeric

.. _dataset-real-world-ma-employment-feature-description-SCHL:

**SCHL** - Educational attainment:

- ``N/A`` - less than 3 years old
- ``1`` - No schooling completed
- ``2`` - Nursery school, preschool
- ``3`` - Kindergarten
- ``4`` - Grade 1
- ``5`` - Grade 2
- ``6`` - Grade 3
- ``7`` - Grade 4
- ``8`` - Grade 5
- ``9`` - Grade 6
- ``10`` - Grade 7
- ``11`` - Grade 8
- ``12`` - Grade 9
- ``13`` - Grade 10
- ``14`` - Grade 11
- ``15`` - 12th grade - no diploma
- ``16`` - Regular high school diploma
- ``17`` - GED or alternative credential
- ``18`` - Some college, but less than 1 year
- ``19`` - 1 or more years of college credit, no degree
- ``20`` - Associate's degree
- ``21`` - Bachelor's degree
- ``22`` - Master's degree
- ``23`` - Professional degree beyond a bachelor's degree
- ``24`` - Doctorate degree

**MAR** Character 1 - Marital status:

- ``1`` - Married
- ``2`` - Widowed
- ``3`` - Divorced
- ``4`` - Separated
- ``5`` - Never married or under 15 years old

.. _dataset-real-world-ma-employment-feature-description-RELP:

**RELP** Character 2 - Relationship:

- ``0`` - Reference person
- ``1`` - Husband/wife
- ``2`` - Biological son or daughter
- ``3`` - Adopted son or daughter
- ``4`` - Stepson or stepdaughter
- ``5`` - Brother or sister
- ``6`` - Father or mother
- ``7`` - Grandchild
- ``8`` - Parent-in-law
- ``9`` - Son-in-law or daughter-in-law
- ``10`` - Other relative
- ``11`` - Roomer or boarder
- ``12`` - Housemate or roommate
- ``13`` - Unmarried partner
- ``14`` - Foster child
- ``15`` - Other nonrelative
- ``16`` - Institutionalized group quarters population
- ``17``- Noninstitutionalized group quarters population

**DIS** - Disability recode:

- ``1`` - With a disability
- ``2`` - Without a disability

**ESP** - Employment status of parents:

- ``b`` - N/A (not own child of householder, and not child in subfamily)
- ``1`` - Living with two parents: both parents in labor force
- ``2`` - Living with two parents: Father only in labor force
- ``3`` - Living with two parents: Mother only in labor force
- ``4`` - Living with two parents: Neither parent in labor force
- ``5`` - Living with father: Father in the labor force
- ``6`` - Living with father: Father not in labor force
- ``7`` - Living with mother: Mother in the labor force
- ``8`` - Living with mother: Mother not in labor force

**CIT** - Citizenship status:

- ``1`` - Born in the U.S.
- ``2`` - Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas
- ``3`` - Born abroad of American parent(s)
- ``4`` - U.S. citizen by naturalization
- ``5`` - Not a citizen of the U.S.

**MIG** - Mobility status (lived here 1 year ago):

- ``N/A`` - less than 1 year old
- ``1`` - Yes, same house (nonmovers)
- ``2`` - No, outside US and Puerto Rico
- ``3`` - No, different house in US or Puerto Rico

**MIL** - Military service:

- ``N/A`` - less than 17 years old
- ``1`` - Now on active duty
- ``2`` - On active duty in the past, but not now
- ``3`` - Only on active duty for training in Reserves/National Guard
- ``4`` - Never served in the military

**ANC** - Ancestry recode:

- ``1`` - Single
- ``2`` - Multiple
- ``3`` - Unclassified
- ``4`` - Not reported
- ``8`` - Suppressed for data year 2018 for select PUMAs


**NATIVITY** - Nativity:

- ``1`` - Native
- ``2`` - Foreign born

**DEAR** - Hearing difficulty:

- ``1`` - Yes
- ``2`` - No

**DEYE** - Vision difficulty:

- ``1`` - Yes
- ``2`` - No

**DREM** - Cognitive difficulty:

- ``N/A`` - Less than 5 years old
- ``1`` - Yes
- ``2`` - No

**SEX** - Sex:

- ``1`` - Male
- ``2`` - Female

**RAC1P** - Recoded detailed race code:

- ``1`` - White alone
- ``2`` - Black or African American alone
- ``3`` - American Indian alone
- ``4`` - Alaska Native alone
- ``5`` - American Indian and Alaska Native tribes specified or American Indian or Alaska Native, not specified and no other races
- ``6`` - Asian alone
- ``7`` - Native Hawaiian and Other Pacific Islander alone
- ``8`` - Some Other Race alone
- ``9`` - Two or More Races

**ESR** - target:

- ``True`` - employed
- ``False`` - unemployed


References
----------

.. [1] Ding, F. et al. (2021). Retiring Adult: New Datasets for Fair Machine Learning. Advances in
       Neural Information Processing Systems, 34.

.. _`github repository`: https://github.com/NannyML/nannyml/tree/main/docs/example_notebooks
.. _`folktables`: https://github.com/socialfoundations/folktables
.. _`PUMS documentation`: https://www.census.gov/programs-surveys/acs/microdata/documentation.html

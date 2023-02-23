NannyML's univariate approach for data drift looks at each variable individually and compares the
:ref:`chunks<chunking>` created from the analysis :ref:`data period<data-drift-periods>` with the reference period.
You can read more about periods and other data requirements in our section on :ref:`data periods<data-drift-periods>`.

The comparison results in a single number, a drift metric, representing the amount of drift between the reference and
analysis chunks. NannyML calculates them for every chunk, allowing you to track them over time.

NannyML offers both statistical tests as well as distance measures to detect drift. They are being referred to as
`methods`. Some methods are only applicable to continuous data, others to categorical data and some might be used on both.
NannyML lets you choose which methods are to be used on these two types of data.

We begin by loading some synthetic data provided in the NannyML package. This is data for a binary classification model, but other model types operate in the same way.

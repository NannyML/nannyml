.. _data-reconstruction-pca:

============================
Data Reconstruction with PCA
============================

Data Drift in Multidimensional data
-----------------------------------

Machine Learning models have multidimensional input spaces. In binary
classification problems we train our models in order to find the optimal classification
boundary. This boundary is dependent on the structure of the data within the model input
space. However our world is not static, and the structure of our data can change. This
change can then cause our existing decision boundary to be suboptimal.

From :ref:`Univariate Drift Detection<data-drift-univariate>` section,
we saw how we can project our data to each feature individually and observe
whether there are changes in the resulting distributions over time. However
this is not enough to capture all the changes that may affect our model.
We can see an example where this approach would fail to capture a significant
change in the data below:

ADD PICTURE AND DESCRIBE TWO DISTRIBUTIONS...
LINK TO CREATION NOTEBOOK.

We see that the univariate distribution results are the same. (...)
However we can clearly see that the datasets are different. We want a metric that will
be able to capture this change.

Reconstruction Error with PCA
-----------------------------

We will use the PCA Reconstruction Error to capture comples changes in our feature space
such as the one demonstrated above. Let's describe what we mean first.
In general reconstruction error is the error we have when we
re-create a dataset after a dimensionality reduction transformation followed by its
inverse transformation. The error is computed to be the mean of the Euclidean distance
of all the points in our dataset.

Now let's go into more details on how we have implemented this process in NannyML.
The process goes through three steps. The first step is data preparation and includes
frequency encoding and scaling the data. We use frequency encoding
to convert all categorical features into numbers. Compared to one-hot encoding this
approach doesn't increas as much the dataset dimensionality. The next thing we do
is scale all the features to 0 mean and unit variance. This makes sure that all features
contribute to PCA on equal footing.

The second step is the dimensionality reduction part. We use PCA to perform this.
By default we are aiming to capture 65% of the dataset's variance but the user can
change that. The PCA algorithm is fitted on the reference dataset.
It learns a transofrmation from the pre-processed, from the first step,
model input space to a latent space. We then apply this transformtion to the data
we are analyzing. This step is very crucial for our process. It is key here
that our representation learning method captures the internal structure of the dataset
and ignores the random noise that is usually present.

The third step is to transform our data from the latent space back to the preprocessed
model input space that we got at the end of the first step. In our case all we need for that
is to apply the inverse PCA transformation.

Since the second step in our process is about compressing information we cannot expect
to end up precisely with the data we started at the end of step three. Some information will
have been lost and this will mean that our reconstructed data will be slightly different compared
to the original. Reconstruction error is a measure of how different the reconstructed data
are from the original.

Understanding PCA Reconstruction Error
--------------------------------------

At :ref:`Multivariate Drift Detection<data-drift-multivariate>` we saw how we can compute PCA
Reconstruction Error. Let's go a bit deeper in what it means.

The key thing we need to know is that reconstruction error on it's own doesn't convey
information. It is the change in reconstruction error values over time that does so.
It tells us whether there is data drift or not. This is because, when there is significant
data drift, the principal compoments of our data, that the PCA method has learnt, are now
slightly different. This will result in worse reconstruction of the new data and therefore
increased reconstruction error.

Because of the noise present in real world datasets, there will always be some
variability in reconstruction error results. We use this variability to determine
what a significant change in reconstruction error is. We compute the mean
and standard deviation of the reconstruction error with PCA on the reference
dataset. And we define as a threshold for a significant change any values that
are more than two standard deviations from the mean.

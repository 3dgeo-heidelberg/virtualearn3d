.. _Imputers page:

Imputers
**********

Imputers are the components that provide data imputation capabilities. It might
be that some computation on a point cloud yields a not valid output, for
instance a Not a Number (NaN) value. In those cases, we could benefit from
adding some imputer to our pipeline.

For example, one typical step in many point cloud
processing pipelines consists of computing geometric features (see
:class:`.GeometricFeaturesMiner`). In doing so, many 3D neighborhoods are
analyzed by the Singular Value Decomposition (SVD) of a
:math:`\pmb{X} \in \mathbb{R}^{m \times 3}` matrix representing the :math:`m`
points of the neighborhood. The SVD yields the following factorization
:math:`\pmb{X} = \pmb{U}\pmb{\Sigma}\pmb{V}^\intercal` from where the
:math:`\pmb{V}^\intercal` singular vectors and the :math:`\pmb{\Sigma}`
singular values can be used to derive linearity, planarity, sphericity, and
other geometric features.

Note that this factorization on a 3D space will be
problematic if there are not at least three linearly independent equations.
Fortunately, most points in a dense point cloud have populated enough
neighborhoods. However, some points at the boundaries of the point cloud,
points from scanning artifacts, or outlier points by whatever reason, might
have poorly populated neighborhoods. Consequently, NaN values might appear for
some features. In that case, a data imputation strategy will allow us to define
what do with problematic features in our pipeline.




Removal imputer
=================

A :class:`.RemovalImputer` defines a target value and removes from the point
cloud all the points that contain such value in their features.

The JSON below shows an example of :class:`.RemovalImputer`.

.. code-block:: json

    {
        "imputer": "RemovalImputer",
        "fnames": ["AUTO"]
        "target_val": "NaN",
    }

In the JSON above the target value is NaN and all the features considered
at the current state of the pipeline will be considered when searching for the
target value (``"AUTO"``).


**Arguments**

-- ``fnames``
    It can be an arbitrary list of feature names specifying what features
    consider. Alternatively, it can be a list containing the string "AUTO".
    In this case, the feature names will be automatically derived. Typically,
    the features that were considered by the most recent
    component that interacted with the features will be selected.

-- ``target_val``
    It can be the "NaN" string (it will be understood as NaN), or any integer
    or decimal number.









.. _Univariate imputer:

Univariate imputer
====================

A :class:`.UnivariateImputer` defines a target value and replaces it
considering the values for that feature that do not match the target value.
It is called univariate because it operates on each feature independently.
For example, if we have a feature for five points with values
:math:`(0.3, 0.1, 0.1, \mathrm{NaN}, 0.5)` we could use a mean-based univariate
imputer to achieve :math:`(0.3, 0.1, 0.1, 0.25, 0.5)`.

The JSON below shows an example of :class:`.UnivariateImputer`.

.. code-block:: json

    {
        "imputer": "UnivariateImputer",
        "fnames": ["AUTO"],
        "target_val": "NaN",
        "strategy": "mean",
        "constant_val": 0
    }

In the JSON above the target value is NaN and all the features considered at
the current state of the pipeline will be considered when searching for the
target value (``"AUTO"``). The NaN values will be replaced by the mean of the
numerical values. The ``constant_val`` is not used because the value to replace
the NaN is automatically derived as the mean.

**Arguments**

-- ``fnames``
    It can be an arbitrary list of feature names specifying what features
    consider. Alternatively, it can be a list containing the string "AUTO".
    In this case, the feature names will be automatically derived. Typically,
    the features that were considered by the most recent
    component that interacted with the features will be selected.

-- ``target_val``
    It can be the "NaN" string (it will be understood as NaN), or any integer
    or decimal number.

-- ``strategy``
    It can be any strategy supported by :class:`sklearn.impute.SimpleImputer`
    as a ``strategy`` parameter. See `sklearn SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

-- ``constant_val``
    Defines the new value when the strategy is to replace by a given constant
    value.

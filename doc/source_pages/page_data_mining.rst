.. _Data mining page:

Data mining
**************

Data miners are components that receive an input point cloud and extract
features characterizing it, typically in a point-wise fashion.
Data miners (:class:`.Miner`) can be included inside pipelines to generate
features that can later be used to train a machine learning model to perform
a classification or regression task on the points.




.. _Geometric features miner:

Geometric features miner
==========================

The :class:`.GeomFeatsMiner` uses
`Jakteristics <https://jakteristics.readthedocs.io/en/latest/installation.html>`_
as backend to compute point-wise geometric features. The point-wise features
are computed considering spherical neighborhoods of a given radius. The JSON
below shows how to define a :class:`.GeomFeatsMiner`.

.. code-block:: json

    {
        "miner": "GeometricFeatures",
        "in_pcloud": null,
        "out_pcloud": null,
        "radius": 0.3,
        "fnames": ["linearity", "planarity", "surface_variation", "verticality", "anisotropy"],
        "frenames": ["linearity_r0_3", "planarity_r0_3", "surface_variation_r0_3", "verticality_r0_3", "anisotropy_r0_3"],
        "nthreads": -1
    }


The JSON above defines a :class:`.GeomFeatsMiner` that computes the linearity,
planarity, surface variation, verticality, and anisotropy geometric features
considering a :math:`30\,\mathrm{cm}` radius for the spherical neighborhood.
The computed features will be named from the feature names and the neighborhood
radius. Parallel regions will be computed using all available threads.


**Arguments**

-- ``in_pcloud``
    When the data miner is used outside a pipeline, this argument can be used
    to specify which point cloud must be loaded to compute the geometric
    features on it. In pipelines, the input point cloud is considered to be
    the point cloud at the current pipeline's state.

-- ``out_pcloud``
    When the data miner is used outside a pipeline, this argument can be used
    to specify where to write the output point cloud with the computed
    geometric features. Otherwise, it is better to use a Writer to export the
    point cloud after the data mining.

-- ``radius``
    The radius for the spherical neighborhood.

-- ``fnames``
    The list with the names of the features that must be computed. Supported
    features are:
    ``["eigenvalue_sum", "omnivariance", "eigenentropy", "anisotropy",
    "planarity", "linearity", "PCA1", "PCA2", "surface_variation",
    "sphericity", "verticality"]``

-- ``frenames``
    The list of names for the generated features. If it is not given, then
    the generated features will be automatically named.

-- ``nthreads``
    How many threads use to compute parallel regions. The value -1 means as
    many threads as supported in parallel (typically including virtual cores).


**Output**

The figure below represents the planarity and verticality features mined for
a spherical neighborhood with :math:`30\,\mathrm{cm}` radius. The point cloud
for this example corresponds to the `Paris` point cloud from the
`Paris-Lille-3D dataset <https://npm3d.fr/paris-lille-3d>`_.

.. figure:: ../img/geomfeats.png
    :scale: 40
    :alt: Figure representing some point-wise geometric features.

    Visualization of the planarity (left) and verticality (right) computed in
    the `Paris` point cloud from the Paris-Lille-3D dataset using
    spherical negibhorhoods with :math:`30\,\mathrm{cm}` radius.




Covariance features miner
============================

The :class:`.CovarFeatsMiner` uses
`PDAL <https://pdal.io/en/2.6.0/stages/filters.covariancefeatures.html#filters-covariancefeatures>`_
as backend to compute point-wise geometric features. It can be used to compute
features on either spherical neighborhoods or k-nearest neighbors (knn)
neighborhoods. The JSON below shows how
to define a :class:`.CovarFeatsMiner`.

.. code-block:: json

    {
        "miner": "CovarianceFeatures",
        "neighborhood": "spherical",
        "radius": 0.3,
        "min_neighs": 3,
        "mode": "Raw",
        "optimize": false,
        "fnames": ["Linearity", "Planarity", "SurfaceVariation", "Verticality", "Anisotropy"],
        "frenames": ["linearity_r0_3", "planarity_r0_3", "surface_variation_r0_3", "verticality_r0_3", "anisotropy_r0_3"],
        "nthreads": 12
    }

The JSON above defines a :class:`.CovarFeatsMiner` that computes the linearity,
planarity, surface variation, verticality, and anisotropy features
considering a spherical neighborhood with :math:`30\,\mathrm{cm}` radius. The
computed features will be named from the feature names and the neighborhood
radius. Exactly 12 threads will be used for the computations.


**Arguments**

-- ``neighborhood``
    Either ``"spherical"`` to use a spherical neighborhood or ``"knn"`` to use
    a k-nearest neighbors neighborhood.

-- ``radius``
    The radius for the spherical neighborhood.

-- ``min_neighs``
    The minimum number of neighbors that is acceptable. When using ``"knn"``
    neighborhood, ``min_neighs`` defines :math:`k`.

-- ``mode``
    A string specifying how to compute the features:

    ``"SQRT"`` will consider the square root of the eigenvalues.
    ``"Normalized"`` will normalize the eigenvalues so they sum to one.
    ``"Raw"`` will directly use the raw eigenvalues.

-- ``optimize``
    When set to true the neighborhood configuration will be automatically
    determined at the expense of increasing the execution time. When set to
    false, nothing will happen.

    See `PDAL documentation on optimal neighborhood filter <https://pdal.io/en/2.6.0/stages/filters.optimalneighborhood.html#filters-optimalneighborhood>`_
    for further details.

-- ``fnames``
    The list with the names of the features that must be computed. Supported
    features are:
    ``["Anisotropy", "DemantkeVerticality", "Density", "Eigenentropy",
    "Linearity", "Omnivariance", "Planarity", "Scattering", "EigenvalueSum",
    "SurfaceVariation", "Verticality"]``

    See `PDAL documentation on covariance features <https://pdal.io/en/2.6.0/stages/filters.covariancefeatures.html#filters-covariancefeatures>`_
    for further details.

-- ``frenames``
    The list of names for the generated features. If it is not given, then
    the generated features will be automatically named.

-- ``nthreads``
    How many threads use to compute parallel regions. The value -1 means as
    many threads as supported in parallel (typically including virtual cores).


**Output**

The figure below represents the anisotropy and linearity features mined for
a spherical neighborhood with :math:`30\,\mathrm{cm}` radius. The point cloud
fro this example correponds to the Paris point cloud from the
`Paris-Lille-3D dataset <https://npm3d.fr/paris-lille-3d>`_.

.. figure:: ../img/covarfeats.png
    :scale: 40
    :alt: Figure representing some point-wise covariance features.

    Visualization of the anisotropy (left) and linearity (right) computed
    in the Paris point cloud from the `Paris-Lille-3D` dataset using spherical
    neighborhoods with :math:`30\,\mathrm{cm}` radius.




Height features miner
========================

**To be implemented yet**.
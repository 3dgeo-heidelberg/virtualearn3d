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
for this example correponds to the Paris point cloud from the
`Paris-Lille-3D dataset <https://npm3d.fr/paris-lille-3d>`_.

.. figure:: ../img/covarfeats.png
    :scale: 40
    :alt: Figure representing some point-wise covariance features.

    Visualization of the anisotropy (left) and linearity (right) computed
    in the Paris point cloud from the `Paris-Lille-3D` dataset using spherical
    neighborhoods with :math:`30\,\mathrm{cm}` radius.




Height features miner
========================

The :class:`.HeightFeatsMiner` supports the computation of height-based
features. These features assume that the :math:`z` axis corresponds to the
vertical axis and derive features depending on the :math:`z` values of
many neighborhoods. The neighborhoods are centered on support points. Finally,
each point in the point cloud will take the features from the closest support
point. The JSON below shows how to define a :class:`.HeightFeatsMiner`:

.. code-block:: json

    {
        "miner": "HeightFeatures",
        "support_chunk_size": 50,
        "support_subchunk_size": 10,
        "pwise_chunk_size": 1000,
        "nthreads": 12,
        "neighborhood": {
            "type": "Rectangular2D",
            "radius": 50.0,
            "separation_factor": 0.35
        },
        "outlier_filter": null,
        "fnames": ["floor_distance", "ceil_distance"]
    }

The JSON above defines a :class:`.HeightFeatsMiner` that computes the distance
to the floor (lowest point) and to the ceil (highest point). It considers
a rectangular neighborhood for the support points with side length
:math:`50 \times 2 = 100` meters. Not outlier filter is applied.

**Arguments**

-- ``support_chunk_size``
    The number of support points per chunk for parallel computations.

-- ``support_subchunk_size``
    The number of simultaneous neighborhoods considered when computing a chunk.
    It can be used to prevent memory exhaustion scenarios.

-- ``pwise_chunk_size``
    The number of points per chunk when computing the height features for the
    points in the point cloud (not the support points).

-- ``nthreads``
    How many threads must be used for parallel computations (-1 means as many
    threads as available cores).

-- ``neighborhood``
    The neighborhood definition. The type can be either ``"Rectangular2D"``
    (the radius describes half of the side) or ``"Cylinder"`` (the radius
    describes the disk of the cylinder). The separation factor governs the
    separation of the support points considering the radius.
    See :class:`.GridSubsamplingPreProcessor` for more details.

-- ``outlier_filter``
    The strategy to filter outlier points (it can be None). Supported
    strategies are ``"IQR"`` and ``"stdev"``. The ``"IQR"`` strategy considers
    the interquartile range and discards any height value outside
    :math:`[Q_1-3\mathrm{IQR}/2, Q_3+3\mathrm{IQR}/2]`. The ``"stdev"``
    strategy discards any height value outside
    :math:`[\mu - 3\sigma, \mu + 3\sigma]` where :math:`\mu` is the mean and
    :math:`\sigma` is the standard deviation.

-- ``fnames``
    The name of the height features that must be computed. Supported height
    features are:
    ``["floor_coordinate", "floor_distance", "ceil_coordinate",
    "ceil_distance", "height_range", "mean_height", "median_height",
    "height_quartiles", "height_deciles", "height_variance",
    "height_stdev", "height_skewness", "height_kurtosis"]``

**Output**

The figure below represents the floor distance mined for a spherical
Rectangular2D neighborhood with :math:`50` meters radius. The point cloud from
this example corresponds to the March2018 validation point cloud from the
`Hessigheim dataset <https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx>`_.

.. figure:: ../img/height_feats.png
    :scale: 40
    :alt: Figure representing the floor distance height feature.

    Visualization of the floor distance height feature computed for the
    Hessigheim March2018 validation point cloud using using a Rectangular2D
    neighborhood with :math:`50\,\mathrm{m}` radius.




HSV from RGB miner
=====================

The :class:`.HSVFromRGBMiner` can be used when red, green, and blue color channels
are available for the points in the point cloud. It will generate the
corresponding hue (H), saturation (S), and value (V) components derived from
the available RGB information. The JSON below shows how to define a
:class:`.HSVFromRGBMiner`:


.. code-block:: json

    {
        "miner": "HSVFromRGB",
        "hue_unit": "radians",
        "frenames": ["HSV_Hrad", "HSV_S", "HSV_V"]
    }

The JSON above defines a :class:`.HSVFromRGBMiner` that computes the HSV
representation of the original RGB color components.


**Arguments**

-- ``hue_unit``
    The unit for the hue (H) component. It can be either ``"radians"`` or
    ``"degrees"``.

-- ``frenames``
    The name for the output features. If not given, they will be
    ``["HSV_H", "HSV_S", "HSV_V"]`` by default.


**Output**

The figure below represents the saturation (S) computed for the March2018
validation point cloud from the
`Hessigheim dataset <https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx>`_.

.. figure:: ../img/hsv_from_rgb_feats.png
    :scale: 40
    :alt: Figure representing the saturation (S).

    Figure representing the saturation (S) in the March2018 validation point
    cloud of the Hessigheim dataset.




Smooth features miner
========================

The :class:`.SmoothFeatsMiner` can be used to derive smooth features from
already available features. The mean, weighted mean, and Guassian
Radial Basis Function (RBF) strategies can be used for this purpose. The JSON
below shows how to define a :class:`.SmoothFeatsMiner`:

.. code-block:: json

    {
        "miner": "SmoothFeatures",
        "nan_policy": "propagate",
        "chunk_size": 1000000,
        "subchunk_size": 1000,
        "neighborhood": {
            "type": "sphere",
            "radius": 0.25
        },
        "input_fnames": ["Reflectance", "HSV_Hrad", "HSV_S", "HSV_V"],
        "fnames": ["mean"],
        "nthreads": 12
    }

The JSON above defines a :class:`.SmoothFeatsMiner` that computes the smooth
reflectance, and HSV components considering a spherical neighborhood with
:math:`25\,\mathrm{cm}` radius. The strategy consists of computing the mean
value for each neighborhood. The computations are run in parallel using 12
threads.


**Arguments**

-- ``nan_policy``
    It can be ``"propagate"`` (default) so NaN features will be included
    in computations (potentially leading to NaN smooth features).
    Alternatively, it can be ``"replace"`` so NaN values are replaced with the
    feature-wise mean for each neighborhood. However, using ``"replace"`` leads
    to longer executions times. Therefore, ``"propagate"`` should be used
    always that NaN handling is not necessary.

-- ``chunk_size``
    How many points per chunk must be considered for parallel computations.

-- ``subchunk_size``
    How many neighborhoods per iteration must be considered when computing a
    chunk. It can be useful to prevent memory exhaustion scenarios.

-- ``neighborhood``
    The definition of the neighborhood to be used. Supported neighborhoods are
    ``"knn"`` (for which a ``"k"`` value must be given), ``"sphere"``
    (for which a ``"radius"`` value must be given), and ``"cylinder"`` (the
    ``"radius"`` refers to the disk of the cylinder).

-- ``weighted_mean_omega``
    The :math:`\omega` parameter for the weighted mean strategy (see
    :class:`.SmoothFeatsMiner` for a description of the maths).

-- ``gaussian_rbf_omega``
    The :math:`\omega` parameter for the Gaussian RBF strategy (see
    :class:`.SmoothFeatsMiner` for a description of the maths).

-- ``input_fnames``
    The names of the features that must be smoothed.

-- ``fnames``
    The names of the smoothing strategies to be used. Supported strategies are
    ``"mean"``, ``"weighted_mean"``, and ``"gaussian_rbf"``.

-- ``frenames``
    The desired names for the generated output features. If not given, the
    names will be automatically derived.

-- ``nthreads``
    The number of threads to be used for parallel computations (-1 means as
    many threads as available cores).


**Output**

The figure below represents the smoothed saturation computed for two
spherical neighborhoods with :math:`25\,\mathrm{cm}` and :math:`3\,\mathrm{m}`
radius, respectively. The point cloud is the March2018 validation one from the
`Hessigheim dataset <https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx>`_.

.. figure:: ../img/smooth_features.png
    :scale: 40
    :alt: Figure representing the smoothed saturation for two different
        spherical neighborhoods.

    Figure representing the smoothed saturation for two different spherical
    neighborhoods with :math:`25\,\mathrm{cm}` and :math:`3\,\mathrm{m}`
    radius, respectively.



Recount miner
================

The :class:`.RecountMiner` can be used to derive features based on counting
the number of points. In doing so, many condition-based filters can be applied
to filter the points. Furthermore, the recount of points can be used as a
feature directly but also to derive the relative frequency, the surface density
(points per area), the volume density (points per volume), and the number of
vertical segments along a cylinder that contain at least one point passing the
filters. The JSON below shows how to define a :class:`.RecountMiner`:

.. code-block:: json

    {
        "miner": "Recount",
        "chunk_size": 100000,
        "subchunk_size": 1000,
        "nthreads": 16,
        "neighborhood": {
            "type": "cylinder",
            "radius": 3.0
        },
        "input_fnames": ["vegetation", "tower", "PointWiseEntropy", "Prediction"],
        "filters": [
            {
                "filter_name": "pdensity",
                "ignore_nan": false,
                "absolute_frequency": true,
                "relative_frequency": false,
                "surface_density": true,
                "volume_density": true,
                "vertical_segments": 0,
                "conditions": null
            },
            {
                "filter_name": "maybe_tower",
                "ignore_nan": true,
                "absolute_frequency": true,
                "relative_frequency": true,
                "surface_density": true,
                "volume_density": true,
                "vertical_segments": 0,
                "conditions": [
                    {
                        "value_name": "tower",
                        "condition_type": "greater_than_or_equal_to",
                        "value_target": 0.333333

                    }
                ]
            },
            {
                "filter_name": "as_tower",
                "ignore_nan": true,
                "absolute_frequency": true,
                "relative_frequency": true,
                "surface_density": true,
                "volume_density": true,
                "vertical_segments": 8,
                "conditions": [
                    {
                        "value_name": "Prediction",
                        "condition_type": "equals",
                        "value_target": 4
                    }
                ]
            },
            {
                "filter_name": "unsure_veg",
                "ignore_nan": true,
                "absolute_frequency": true,
                "relative_frequency": true,
                "surface_density": false,
                "volume_density": false,
                "vertical_segments": 0,
                "conditions": [
                    {
                        "value_name": "Prediction",
                        "condition_type": "equals",
                        "value_target": 2
                    },
                    {
                        "value_name": "PointWiseEntropy",
                        "condition_type": "greater_than_or_equal_to",
                        "value_target": 0.1
                    }
                ]
            },
            {
                "filter_name": "unsure_veg2",
                "ignore_nan": true,
                "absolute_frequency": true,
                "relative_frequency": true,
                "surface_density": false,
                "volume_density": false,
                "vertical_segments": 0,
                "conditions": [
                    {
                        "value_name": "Prediction",
                        "condition_type": "equals",
                        "value_target": 2
                    },
                    {
                        "value_name": "vegetation",
                        "condition_type": "less_than",
                        "value_target": 0.666667
                    }
                ]
            }
        ]
    }


The JSON above defines a :class:`.RecountMiner` that computes features from
a previously classified point cloud. First, it computes the absolute frequency,
and the densities considering all points.
Then, it computes the frequencies and densities for
points whose likelihood to be a tower is equal to or above
:math:`0.\overline{3}`.
Afterwards, the frequencies, densities, and counts how many of eight vertical
segments contain at least one point, considering points predicted as tower.
Later, the frequencies for points that have been predicted as vegetation and
have a point-wise entropy greater than or equal to :math:`0.1`. Finally, the
frequencies for points predicted as vegetation with a likelihood less than
:math:`0.\overline{6}`.

**Arguments**

-- ``chunk_size``
    How many points per chunk must be considered for parallel computations.

-- ``subchunk_size``
    How many neighborhoods per iteration must be considered when computing a
    chunk. It can be useful to prevent memory exhaustion scenarios.

-- ``nthreads``
    The number of threads to be used for parallel computations (-1 means as
    many threads as available cores).

-- ``neighborhood``
    The definition of the neighborhood to be used. Supported neighborhoods are
    ``"knn"`` (for which a ``"k"`` value must be given), ``"sphere"``
    (for which a ``"radius"`` value must be given), and ``"cylinder"`` (the
    ``"radius"`` refers to the disk of the cylinder).

-- ``input_fnames``
    The names of the features to be considered when filtering the points.

-- ``filters``
    A list with all the filters that must be computed. One set of output
    features will be generated for each filter. Any filter can consist of none
    or many conditions. The filters can be defined such that:

    -- ``filter_name``
        The name for the filter. It will be used to name the generated
        features.

    -- ``ignore_nan``
        A flag governing how to handle nans. When set to ``true``, the filters
        will ignore points with nan values, i.e., they will not be counted.

    -- ``absolute_frequency``
        Whether to generate a feature with the absolute frequency or raw count
        (``true``) or not (``false``). The generated feature will be named by
        appending ``"_abs"`` to the filter name.

    -- ``relative_frequency``
        Whether to generate a feature with the relative frequency (``true``)
        or not (``false``). The generated feature will be named by appending
        ``"_rel"`` to the filter name.

    -- ``surface_density``
        Whether to generate a feature by dividing the number of points by the
        surface area. The surface density is computed assuming the area of
        a circle. The radius of the circle will be the given one when using
        spherical or cylindrical neighborhoods but it will be derived as the
        distance between the center point and the furthest neighbor for
        knn neighborhoods. The generated feature will be named by appending
        ``"_sd"`` to the filter name.

    -- ``volume_density``
        Whether to generate a feature by dividing the number of points by the
        volume. The volume is computed assuming a sphere. The radius of the
        sphere will be the given one when using spherical neighborhoods but
        it will be derived as the distance between the center point and the
        furthest neighbor for knn neighborhoods. For cylindrical neighborhoods,
        a circle will be considered instead of the sphere, and the volume
        will be computed as the area of the circle along the boundaries of the
        vertical axis. The generated feature will be named by appending
        ``"_vd"`` to the filter name.

    -- ``vertical_segments``
        Whether to generate a feature by dividing the neighborhood into
        linearly spaced segments along the vertical axis and counting how many
        partitions contain at least one point satisfying the conditions. The
        generated feature will be named by appending ``"_vs"`` to the filter
        name.

    -- ``conditions``
        The list of conditions is a list of elements defined in the same way
        as the conditions of the
        :ref:`Advanced input <Advanced input>`
        but without the ``action`` parameter, that is always assumed to be
        ``"preserve"``.

**Output**

The generated output is a point cloud that includes the recount-based features.




Take closest miner
=====================

The :class:`.TakeClosestMiner` can be used to derive features from another
point cloud. It works by defining a pool of point clouds such that the closest
neighbor between the input point cloud and any point cloud in the pool will be
considered. Then, the features for each point will be taken from its closest
neighbor.

.. code-block:: json

    {
        "miner": "TakeClosestMiner",
        "fnames": [
            "HSV_Hrad", "HSV_S", "HSV_V",
            "floor_distance_r50.0_sep0.35",
            "eigenvalue_sum_r0.3", "omnivariance_r0.3", "eigenentropy_r0.3",
            "anisotropy_r0.3", "planarity_r0.3", "linearity_r0.3",
            "PCA1_r0.3", "PCA2_r0.3",
            "surface_variation_r0.3", "sphericity_r0.3", "verticality_r0.3",
        ],
        "pcloud_pool": [
            "/home/point_clouds/point_cloud_A.laz",
            "/home/point_clouds/point_cloud_B.laz",
            "/home/point_clouds/point_cloud_C.laz"
        ],
        "distance_upper_bound": 0.1,
        "nthreads": 12
    }

The JSON above defines a :class:`.TakeClosestMiner` that finds the features of
the closest point in a pool of three point clouds. Neighbors further than
:math:`0.1\,\mathrm{m}` will not be considered, even if they are the closest
neighbor.


**Arguments**

-- ``fnames``
    The names of the features that must be taken from the closest neighbor in
    the pool.

-- ``frenames``
    An optional list with the name of the output features. When not given, the
    output features will be named as specified by ``fnames``.

-- ``y_default``
    An optional value to be considered as the default label/class. If not
    given, it will be the max integer supported by the system.

-- ``pcloud_pool``
    A list with the paths to the point clouds composing the pool.

-- ``distance_upper_bound``
    The max distance threshold. Neighbors further than this distance will be
    ignored.

-- ``nthreads``
    The number of threads for parallel queries.


**Output**

The generated output is a point cloud where the features correspond to the
closest neighbor in the pool, assuming there is at least one neighbor that
is closer than the ``distance upper bound``.








Decorators
================

.. _FPS decorated miner:

Furthest point sampling decorator
--------------------------------------------

The :class:`.FPSDecoratedMiner` can be used to decorate a data miner such that
the computations can take place in a transformed space of reduced
dimensionality. Typically, the domain of a data miner is the entire point
cloud, let us say :math:`m` points. When using a :class:`.FPSDecoratedMiner`
this domain will be transformed to a subset of the original point cloud with
:math:`R` points, such that :math:`m \geq R`. Decorating a data miner with this
decorator can be useful to reduce its execution time.


.. code-block:: json

    {
        "miner": "FPSDecorated",
        "fps_decorator": {
            "num_points": "m/3",
            "fast": true,
            "num_encoding_neighbors": 1,
            "num_decoding_neighbors": 1,
            "release_encoding_neighborhoods": false,
            "threads": 16,
            "representation_report_path": "*/fps_repr/geom_r3_representation.laz"
        },
        "decorated_miner": {
            "miner": "GeometricFeatures",
            "in_pcloud": null,
            "out_pcloud": null,
            "radius": 3.0,
            "fnames": ["linearity", "planarity", "surface_variation", "verticality", "anisotropy", "PCA1", "PCA2"],
            "frenames": ["linearity_r3", "planarity_r3", "surface_variation_r3", "verticality_r3", "anisotropy_r3", "PCA1_r3", "PCA2_r3"],
            "nthreads": 16
        }
    }

**Arguments**

-- ``fps_decorator``
    The specification of the furthest point sampling (FPS) decoration carried
    out through the :class:`.FPSDecoratorTransformer`.

    -- ``num_points``
        The target number of points :math:`R` for the transformed point cloud.
        It can be an integer or an expression that will be evaluated with
        :math:`m` representing the number of points of the original point
        cloud, e.g., ``"m/2"`` will downscale the point cloud to half the
        number of points.

    -- ``fast``
        Whether to use exact furthest point sampling (``false``) or a faster
        stochastic approximation (``true``).

    -- ``num_encoding_neighbors``
        How many closest neighbors in the original point cloud are considered
        for each point in the transformed point cloud to reduce from the
        original space to the transformed one.

    -- ``num_decoding_neighbors``
        How many closest neighbors in the transformed point cloud are
        considered for each point in the original point cloud to propagate back
        from the transformed space to the original one.

    -- ``release_encoding_neighborhoods``
        Whether the encoding neighborhoods can be released after computing the
        transformation (``true``) or not (``false``). Releasing these
        neighborhoods means the :meth:`.FPSDecoratorTransformer.reduce` method
        must not be called, otherwise errors will arise. Setting this flag to
        true can help saving memory when needed.

    -- ``threads``
        The number of parallel threads to consider for the parallel
        computations. Note that ``-1`` means using as many threads as available
        cores.

    -- ``representation_report_path``
        Where to export the transformed point cloud. In general, it should be
        ``null`` to prevent unnecessary operations. However, it can be enabled
        (by given any valid path to write a point cloud file) to visualize the
        points that are seen by the data miner.

-- ``decorated_miner``
    A typical data mining specification. See
    :ref:`the Geometric features miner <Geometric features miner>`
    for an example.
